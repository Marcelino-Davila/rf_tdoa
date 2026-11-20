#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# GGBLUE - TDOA with Software Sync
#
# HOW IT WORKS:
#   TX sends: [Zadoff-Chu preamble][BPSK data][Zadoff-Chu preamble]...
#   Each HackRF runs a matched filter looking for the ZC preamble.
#   When detected, the sample index of detection = arrival time.
#   TDOA = (arrival_index_H0 - arrival_index_Hx) / samp_rate
#   This eliminates the need for hardware sample-level sync.
#
# UPDATE ANTENNA_POSITIONS WITH YOUR REAL MEASUREMENTS (meters):
#   H0 = origin (0,0), measure H1 and H2 relative to H0
#

from PyQt5 import Qt
from gnuradio import qtgui
from gnuradio import blocks
from gnuradio import analog
from gnuradio import gr
from gnuradio.fft import window
from gnuradio import uhd
import sys
import signal
import threading
import time
import numpy as np
from scipy.optimize import minimize
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import osmosdr
import sip


# =============================================================================
# ANTENNA POSITIONS — UPDATE WITH TAPE MEASURE (meters)
# H0 = origin. Measure H1 and H2 relative to H0.
# =============================================================================
ANTENNA_POSITIONS = {
    0: np.array([0.0, 0.0]),    # HackRF 0 — origin
    1: np.array([5.0, 0.0]),    # HackRF 1 — UPDATE ME
    2: np.array([2.5, 4.0]),    # HackRF 2 — UPDATE ME
}

# =============================================================================
# RF CONFIG
# =============================================================================
CENTER_FREQ_HZ  = 2.420e9
RX_SAMP_RATE    = 20e6
RX_BW_HZ        = 20e6
TX_FREQ_HZ      = CENTER_FREQ_HZ
TX_GAIN_DB      = 10
SPEED_OF_LIGHT  = 3e8

# =============================================================================
# ZADOFF-CHU PREAMBLE CONFIG
# ZC sequences have perfect autocorrelation — ideal for timing detection
# =============================================================================
ZC_LENGTH       = 127       # ZC sequence length (prime number)
ZC_ROOT         = 25        # ZC root index
DETECTION_THRESHOLD = 0.6   # normalized correlation threshold (0-1)
PREAMBLE_REPEAT_MS  = 50    # TX repeats preamble every N ms

# =============================================================================
# TDOA CONFIG
# =============================================================================
TDOA_UPDATE_HZ  = 4.0
SEARCH_AREA_M   = 20.0
GRID_POINTS     = 150


# =============================================================================
# Zadoff-Chu sequence generator
# =============================================================================

def zadoff_chu(length, root):
    """Generate a Zadoff-Chu sequence of given length and root."""
    n = np.arange(length)
    if length % 2 == 0:
        return np.exp(-1j * np.pi * root * n * (n + 1) / length)
    else:
        return np.exp(-1j * np.pi * root * n * (n + 1) / length)


def make_tx_signal(samp_rate=RX_SAMP_RATE, repeat_ms=PREAMBLE_REPEAT_MS):
    """
    Build one TX burst: ZC preamble repeated to fill repeat_ms worth of samples.
    Returns complex64 numpy array.
    """
    zc = zadoff_chu(ZC_LENGTH, ZC_ROOT).astype(np.complex64)
    # Normalize
    zc = zc / np.max(np.abs(zc))
    # Repeat to fill the burst duration
    n_samples = int(samp_rate * repeat_ms / 1000)
    repeats = int(np.ceil(n_samples / ZC_LENGTH))
    burst = np.tile(zc, repeats)[:n_samples]
    return burst.astype(np.complex64)


# Precompute TX signal and matched filter kernel
TX_BURST         = make_tx_signal()
ZC_SEQ           = zadoff_chu(ZC_LENGTH, ZC_ROOT).astype(np.complex64)
ZC_SEQ_CONJ      = np.conj(ZC_SEQ[::-1])   # matched filter kernel


# =============================================================================
# Preamble Detector
# Runs matched filter on incoming samples, returns detection timestamps
# =============================================================================

class PreambleDetector:
    """
    Sliding matched filter for ZC preamble detection.
    Maintains a sample counter so detections have absolute sample indices.
    """

    def __init__(self, rx_id, samp_rate=RX_SAMP_RATE):
        self.rx_id          = rx_id
        self.samp_rate      = float(samp_rate)
        self._buf           = np.zeros(ZC_LENGTH * 4, dtype=np.complex64)
        self._sample_count  = 0
        self._detections    = []    # list of sample indices
        self._lock          = threading.Lock()
        self._last_detect   = -ZC_LENGTH * 10   # prevent double-detect

    def push(self, samples):
        with self._lock:
            # Append to buffer
            n = len(samples)
            buf = np.concatenate([self._buf, samples])

            # Matched filter (cross-correlation with ZC conjugate)
            corr = np.abs(np.convolve(buf, ZC_SEQ_CONJ, mode='valid'))

            # Normalize by ZC energy
            zc_energy = np.sum(np.abs(ZC_SEQ) ** 2)
            corr_norm = corr / zc_energy

            # Find peaks above threshold
            for i in range(len(corr_norm)):
                abs_idx = self._sample_count - len(buf) + i + ZC_LENGTH
                if (corr_norm[i] > DETECTION_THRESHOLD and
                        abs_idx - self._last_detect > ZC_LENGTH):
                    self._detections.append(abs_idx)
                    self._last_detect = abs_idx

            # Keep buffer tail for continuity
            self._buf = buf[-ZC_LENGTH * 4:]
            self._sample_count += n

    def pop_detections(self):
        """Return and clear all pending detections."""
        with self._lock:
            d = list(self._detections)
            self._detections.clear()
            return d

    def get_sample_count(self):
        with self._lock:
            return self._sample_count


# =============================================================================
# IQ Probe — passthrough GNU Radio block that feeds the detector
# =============================================================================

class IQProbe(gr.sync_block):
    def __init__(self, rx_id, detector: PreambleDetector):
        gr.sync_block.__init__(self, name=f'IQProbe_{rx_id}',
                               in_sig=[np.complex64], out_sig=[np.complex64])
        self.rx_id    = rx_id
        self.detector = detector

    def work(self, input_items, output_items):
        inp = input_items[0]
        output_items[0][:] = inp
        self.detector.push(inp)
        return len(inp)


# =============================================================================
# TDOA Engine — matches detections across receivers and computes position
# =============================================================================

class TDOAEngine:

    def __init__(self, detectors, samp_rate=RX_SAMP_RATE):
        self.detectors  = detectors   # dict {rx_id: PreambleDetector}
        self.samp_rate  = float(samp_rate)
        self._callbacks = []
        self.last_tdoa  = {}
        self.last_pos   = None

    def add_callback(self, fn):
        self._callbacks.append(fn)

    def compute(self):
        """
        Match preamble detections across receivers.
        Uses H0 as reference — finds nearest detection in H1/H2
        within a ±search window.
        """
        max_lag_samples = int(SEARCH_AREA_M * 2 / SPEED_OF_LIGHT * self.samp_rate)

        d0 = self.detectors[0].pop_detections()
        d1 = self.detectors[1].pop_detections()
        d2 = self.detectors[2].pop_detections()

        if not d0:
            return None, None

        tdoa_01_list = []
        tdoa_02_list = []

        for t0 in d0:
            # Find closest detection in H1 within search window
            for t1 in d1:
                if abs(t1 - t0) < max_lag_samples:
                    tdoa_01_list.append((t0 - t1) / self.samp_rate)
                    break

            # Find closest detection in H2 within search window
            for t2 in d2:
                if abs(t2 - t0) < max_lag_samples:
                    tdoa_02_list.append((t0 - t2) / self.samp_rate)
                    break

        if not tdoa_01_list or not tdoa_02_list:
            return None, None

        # Average over all matched detections this window
        tdoa_01 = float(np.median(tdoa_01_list))
        tdoa_02 = float(np.median(tdoa_02_list))

        self.last_tdoa = {(0,1): tdoa_01, (0,2): tdoa_02}

        # Sanity check — reject physically impossible values
        max_tdoa = SEARCH_AREA_M * 2 / SPEED_OF_LIGHT
        if abs(tdoa_01) > max_tdoa or abs(tdoa_02) > max_tdoa:
            print(f"[TDOA] Rejected — values too large: "
                  f"{tdoa_01*1e6:.1f}µs, {tdoa_02*1e6:.1f}µs")
            return self.last_tdoa, None

        pos = self._estimate_position(tdoa_01, tdoa_02)
        self.last_pos = pos

        print(f"[TDOA] H0-H1: {tdoa_01*1e6:+.2f} µs  ({tdoa_01*SPEED_OF_LIGHT:+.2f} m)")
        print(f"[TDOA] H0-H2: {tdoa_02*1e6:+.2f} µs  ({tdoa_02*SPEED_OF_LIGHT:+.2f} m)")
        if pos is not None:
            print(f"[TDOA] TX position estimate: ({pos[0]:.2f}, {pos[1]:.2f}) m")

        for cb in self._callbacks:
            try:
                cb(self.last_tdoa, pos)
            except Exception as e:
                print(f"[TDOA] callback error: {e}")

        return self.last_tdoa, pos

    def _estimate_position(self, tdoa_01, tdoa_02):
        p0 = ANTENNA_POSITIONS[0]
        p1 = ANTENNA_POSITIONS[1]
        p2 = ANTENNA_POSITIONS[2]

        d01 = tdoa_01 * SPEED_OF_LIGHT
        d02 = tdoa_02 * SPEED_OF_LIGHT

        def cost(xy):
            pos = np.array(xy)
            r0 = np.linalg.norm(pos - p0)
            r1 = np.linalg.norm(pos - p1)
            r2 = np.linalg.norm(pos - p2)
            e1 = (r0 - r1) - d01
            e2 = (r0 - r2) - d02
            return e1**2 + e2**2

        # Try multiple starting points to avoid local minima
        best = None
        best_cost = 1e9
        starts = [
            [0.0, 0.0],
            [p1[0]/2, p1[1]/2],
            [p2[0]/2, p2[1]/2],
            [(p1[0]+p2[0])/3, (p1[1]+p2[1])/3],
            [0.0, 5.0], [5.0, 5.0], [-5.0, 0.0], [0.0, -5.0],
        ]
        for s in starts:
            res = minimize(cost, s, method='Nelder-Mead',
                           options={'xatol': 0.05, 'fatol': 1e-8, 'maxiter': 10000})
            if res.fun < best_cost:
                best_cost = res.fun
                best = res.x

        # Reject if residual is too high (bad geometry or bad TDOA)
        if best_cost > 2.0:
            print(f"[TDOA] Position rejected — residual too high: {best_cost:.3f}")
            return None

        return np.array(best)


# =============================================================================
# TDOA Compute Thread
# =============================================================================

class TDOAComputeThread(threading.Thread):
    def __init__(self, engine: TDOAEngine):
        super().__init__(daemon=True)
        self.engine   = engine
        self._running = False

    def run(self):
        self._running = True
        period = 1.0 / TDOA_UPDATE_HZ
        # Wait for receivers to buffer some samples first
        time.sleep(1.0)
        print(f"[TDOA] Compute thread running at {TDOA_UPDATE_HZ} Hz")
        while self._running:
            try:
                self.engine.compute()
            except Exception as e:
                print(f"[TDOA] compute error: {e}")
            time.sleep(period)

    def stop(self):
        self._running = False


# =============================================================================
# Live 2D Plot Widget
# =============================================================================

class TDOAPlotWidget(Qt.QWidget):

    _update_signal = Qt.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.fig, self.ax = plt.subplots(figsize=(6, 6), facecolor='#1a1a2e')
        self.canvas = FigureCanvas(self.fig)
        layout = Qt.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)
        self._tdoa = {}
        self._pos  = None
        self._pos_history = []   # trail of last N positions
        self._lock = threading.Lock()
        self._update_signal.connect(self._redraw)
        self._setup_axes()

    def _setup_axes(self):
        ax = self.ax
        ax.set_facecolor('#0d0d1a')
        ax.set_xlim(-SEARCH_AREA_M, SEARCH_AREA_M)
        ax.set_ylim(-SEARCH_AREA_M, SEARCH_AREA_M)
        ax.set_aspect('equal')
        ax.set_xlabel('X (meters)', color='white')
        ax.set_ylabel('Y (meters)', color='white')
        ax.set_title('TDOA Position Estimate', color='white', fontsize=12)
        ax.tick_params(colors='white')
        ax.grid(True, alpha=0.2, color='white')
        for spine in ax.spines.values():
            spine.set_edgecolor('white')
        self.fig.tight_layout()

    def push_update(self, tdoa_dict, pos):
        with self._lock:
            self._tdoa = tdoa_dict.copy() if tdoa_dict else {}
            if pos is not None:
                self._pos = pos.copy()
                self._pos_history.append(pos.copy())
                if len(self._pos_history) > 20:   # keep last 20 positions
                    self._pos_history.pop(0)
            else:
                self._pos = None
        self._update_signal.emit()

    def _redraw(self):
        with self._lock:
            tdoa    = dict(self._tdoa)
            pos     = self._pos.copy() if self._pos is not None else None
            history = [p.copy() for p in self._pos_history]

        self.ax.cla()
        self._setup_axes()

        p0 = ANTENNA_POSITIONS[0]
        p1 = ANTENNA_POSITIONS[1]
        p2 = ANTENNA_POSITIONS[2]

        xs = np.linspace(-SEARCH_AREA_M, SEARCH_AREA_M, GRID_POINTS)
        ys = np.linspace(-SEARCH_AREA_M, SEARCH_AREA_M, GRID_POINTS)
        XX, YY = np.meshgrid(xs, ys)

        # Hyperbola H0-H1
        if tdoa.get((0,1)) is not None:
            d01 = tdoa[(0,1)] * SPEED_OF_LIGHT
            R0 = np.sqrt((XX-p0[0])**2 + (YY-p0[1])**2)
            R1 = np.sqrt((XX-p1[0])**2 + (YY-p1[1])**2)
            Z01 = R0 - R1 - d01
            try:
                self.ax.contour(XX, YY, Z01, levels=[0],
                                colors=['#00ffff'], linewidths=2, alpha=0.9)
            except Exception:
                pass

        # Hyperbola H0-H2
        if tdoa.get((0,2)) is not None:
            d02 = tdoa[(0,2)] * SPEED_OF_LIGHT
            R0 = np.sqrt((XX-p0[0])**2 + (YY-p0[1])**2)
            R2 = np.sqrt((XX-p2[0])**2 + (YY-p2[1])**2)
            Z02 = R0 - R2 - d02
            try:
                self.ax.contour(XX, YY, Z02, levels=[0],
                                colors=['#ff00ff'], linewidths=2, alpha=0.9)
            except Exception:
                pass

        # Position trail (fading dots)
        if len(history) > 1:
            for i, hp in enumerate(history[:-1]):
                alpha = (i + 1) / len(history) * 0.5
                self.ax.plot(hp[0], hp[1], 'o', color='#ff4444',
                             markersize=4, alpha=alpha)

        # Antenna positions
        ant_colors = ['#00ff00', '#00ffff', '#ff00ff']
        ant_labels = ['HackRF 0', 'HackRF 1', 'HackRF 2']
        for rx_id, ant_pos in ANTENNA_POSITIONS.items():
            self.ax.plot(ant_pos[0], ant_pos[1], '^',
                         color=ant_colors[rx_id], markersize=14,
                         label=ant_labels[rx_id], zorder=5)
            self.ax.annotate(f'  H{rx_id}', ant_pos,
                             color=ant_colors[rx_id], fontsize=10,
                             fontweight='bold')

        # TX position estimate
        if pos is not None:
            self.ax.plot(pos[0], pos[1], '*',
                         color='#ff3333', markersize=22,
                         label=f'TX  ({pos[0]:.1f}, {pos[1]:.1f}) m',
                         zorder=6)
            self.ax.annotate(f'  TX\n  ({pos[0]:.1f}, {pos[1]:.1f}) m',
                             pos, color='#ff3333', fontsize=9,
                             fontweight='bold')
        else:
            self.ax.text(0, 0, 'Waiting for\npreamble detections...',
                         ha='center', va='center', color='yellow',
                         fontsize=11, alpha=0.7)

        # TDOA info box
        lines = []
        for pair, td in tdoa.items():
            if td is not None:
                lines.append(
                    f"TDOA H{pair[0]}-H{pair[1]}: "
                    f"{td*1e6:+.2f} µs  |  {td*SPEED_OF_LIGHT:+.1f} m"
                )
        if lines:
            self.ax.text(0.02, 0.02, '\n'.join(lines),
                         transform=self.ax.transAxes,
                         fontsize=8, color='white',
                         bbox=dict(boxstyle='round', facecolor='#0d0d1a',
                                   edgecolor='white', alpha=0.85),
                         verticalalignment='bottom')

        self.ax.legend(loc='upper right', fontsize=8,
                       facecolor='#1a1a2e', edgecolor='white',
                       labelcolor='white')
        self.canvas.draw()


# =============================================================================
# TX Vector Source feeding USRP with repeating ZC preamble
# =============================================================================

class ZCVectorSource(gr.sync_block):
    """Outputs the ZC preamble burst repeatedly."""
    def __init__(self):
        gr.sync_block.__init__(self, name='ZCVectorSource',
                               in_sig=None, out_sig=[np.complex64])
        self._burst = TX_BURST.astype(np.complex64)
        self._idx   = 0

    def work(self, input_items, output_items):
        out = output_items[0]
        n   = len(out)
        buf = self._burst
        L   = len(buf)
        written = 0
        while written < n:
            chunk = min(n - written, L - self._idx)
            out[written:written+chunk] = buf[self._idx:self._idx+chunk]
            self._idx = (self._idx + chunk) % L
            written  += chunk
        return n


# =============================================================================
# Top Block
# =============================================================================

class GGBLUE(gr.top_block, Qt.QWidget):

    def __init__(self):
        gr.top_block.__init__(self, "GGBLUE - TDOA Software Sync", catch_exceptions=True)
        Qt.QWidget.__init__(self)
        self.setWindowTitle("GGBLUE — TDOA Software Sync")
        qtgui.util.check_set_qss()

        # Layout: waterfalls left, TDOA plot right
        self.top_h_layout = Qt.QHBoxLayout()
        self.setLayout(self.top_h_layout)

        self.wf_widget = Qt.QWidget()
        self.wf_layout = Qt.QVBoxLayout(self.wf_widget)
        self.wf_layout.setContentsMargins(2, 2, 2, 2)
        self.top_h_layout.addWidget(self.wf_widget, stretch=2)

        self.tdoa_plot = TDOAPlotWidget()
        self.top_h_layout.addWidget(self.tdoa_plot, stretch=1)

        self.settings = Qt.QSettings("gnuradio/flowgraphs", "GGBLUE_TDOA_SW")
        try:
            geometry = self.settings.value("geometry")
            if geometry:
                self.restoreGeometry(geometry)
        except Exception:
            pass

        self.flowgraph_started = threading.Event()
        self.samp_rate = RX_SAMP_RATE

        ##################################################
        # TX — USRP N210
        ##################################################
        self.usrp_sink = uhd.usrp_sink(
            ",".join(('addr=192.168.10.3', '')),
            uhd.stream_args(cpu_format="fc32", args='', channels=[0]),
            "",
        )
        self.usrp_sink.set_clock_source('internal', 0)
        self.usrp_sink.set_samp_rate(RX_SAMP_RATE)
        self.usrp_sink.set_center_freq(TX_FREQ_HZ, 0)
        self.usrp_sink.set_antenna("TX/RX", 0)
        self.usrp_sink.set_bandwidth(RX_BW_HZ, 0)
        self.usrp_sink.set_gain(TX_GAIN_DB, 0)

        # ZC preamble source
        self.zc_source = ZCVectorSource()

        ##################################################
        # RX — 3x HackRF, all on same center freq
        ##################################################
        print(f"[CFG] All RX tuned to {CENTER_FREQ_HZ/1e9:.3f} GHz")
        print(f"[CFG] ZC preamble length: {ZC_LENGTH} samples")
        print(f"[CFG] Detection threshold: {DETECTION_THRESHOLD}")
        print(f"[!!!] UPDATE ANTENNA_POSITIONS with real tape-measure values!")

        self.osmosdr_source_0 = self._make_hackrf(0, CENTER_FREQ_HZ)
        self.osmosdr_source_1 = self._make_hackrf(1, CENTER_FREQ_HZ)
        self.osmosdr_source_2 = self._make_hackrf(2, CENTER_FREQ_HZ)

        ##################################################
        # Preamble detectors + TDOA engine
        ##################################################
        self.detectors = {
            i: PreambleDetector(i, RX_SAMP_RATE) for i in range(3)
        }
        self.tdoa_engine = TDOAEngine(self.detectors, RX_SAMP_RATE)
        self.tdoa_engine.add_callback(self.tdoa_plot.push_update)

        self.iq_probe_0 = IQProbe(0, self.detectors[0])
        self.iq_probe_1 = IQProbe(1, self.detectors[1])
        self.iq_probe_2 = IQProbe(2, self.detectors[2])

        self.tdoa_thread = TDOAComputeThread(self.tdoa_engine)

        ##################################################
        # Waterfalls
        ##################################################
        for i in range(3):
            wf = qtgui.waterfall_sink_c(
                4096,
                window.WIN_BLACKMAN_hARRIS,
                CENTER_FREQ_HZ, RX_BW_HZ,
                f"HackRF {i}  |  {CENTER_FREQ_HZ/1e9:.3f} GHz",
                1, None
            )
            wf.set_update_time(0.05)
            wf.enable_grid(True)
            wf.enable_axis_labels(True)
            wf.set_intensity_range(-140, 10)
            win = sip.wrapinstance(wf.qwidget(), Qt.QWidget)
            self.wf_layout.addWidget(win)
            setattr(self, f'waterfall_{i}', wf)

        ##################################################
        # Connections
        ##################################################
        # TX: ZC source -> USRP
        self.connect((self.zc_source,  0), (self.usrp_sink, 0))

        # RX: HackRF -> IQ probe (detection) -> waterfall
        self.connect((self.osmosdr_source_0, 0), (self.iq_probe_0, 0))
        self.connect((self.osmosdr_source_1, 0), (self.iq_probe_1, 0))
        self.connect((self.osmosdr_source_2, 0), (self.iq_probe_2, 0))

        self.connect((self.iq_probe_0, 0), (self.waterfall_0, 0))
        self.connect((self.iq_probe_1, 0), (self.waterfall_1, 0))
        self.connect((self.iq_probe_2, 0), (self.waterfall_2, 0))

    def _make_hackrf(self, idx, center_freq):
        src = osmosdr.source(args=f"numchan=1 hackrf={idx}")
        src.set_time_unknown_pps(osmosdr.time_spec_t())
        src.set_sample_rate(RX_SAMP_RATE)
        src.set_center_freq(center_freq, 0)
        src.set_freq_corr(0, 0)
        src.set_dc_offset_mode(1, 0)
        src.set_iq_balance_mode(1, 0)
        src.set_gain_mode(False, 0)
        src.set_gain(14, 0)
        src.set_if_gain(24, 0)
        src.set_bb_gain(28, 0)
        src.set_antenna('', 0)
        src.set_bandwidth(RX_BW_HZ, 0)
        return src

    def start(self):
        super().start()
        self.tdoa_thread.start()

    def stop(self):
        self.tdoa_thread.stop()
        super().stop()

    def closeEvent(self, event):
        self.settings = Qt.QSettings("gnuradio/flowgraphs", "GGBLUE_TDOA_SW")
        self.settings.setValue("geometry", self.saveGeometry())
        self.stop()
        self.wait()
        event.accept()


# =============================================================================
# Main
# =============================================================================

def main(top_block_cls=GGBLUE, options=None):
    qapp = Qt.QApplication(sys.argv)
    tb = top_block_cls()
    tb.start()
    tb.flowgraph_started.set()
    tb.show()

    def sig_handler(sig=None, frame=None):
        tb.stop()
        tb.wait()
        Qt.QApplication.quit()

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    timer = Qt.QTimer()
    timer.start(500)
    timer.timeout.connect(lambda: None)

    qapp.exec_()


if __name__ == '__main__':
    main()