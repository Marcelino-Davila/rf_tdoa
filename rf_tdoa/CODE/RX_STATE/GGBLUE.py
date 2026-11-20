#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: GGBLUE - 2.4GHz Channel Activity Monitor + 3-Freq TX
# GNU Radio version: 3.10.12.0

from PyQt5 import Qt
from gnuradio import qtgui
from gnuradio import blocks
from gnuradio import filter
from gnuradio.filter import firdes
from gnuradio import gr
from gnuradio.fft import window
from gnuradio import uhd
import sys
import signal
import threading
import time
import numpy as np
import osmosdr
import sip
import channel_monitor as ch_monitor


# ---------------------------------------------------------------------------
# TX hop frequencies — visible on all 3 HackRF waterfalls
# ---------------------------------------------------------------------------
TX_FREQUENCIES = [2.41e9, 2.43e9, 2.45e9, 2.43e9]
TX_DWELL_SEC   = 0.5


class TXHopController(threading.Thread):
    """
    Cycles USRP TX through 3 fixed frequencies:
      2.410 GHz -> 2.430 GHz -> 2.450 GHz -> repeat
    """
    def __init__(self, usrp_sink):
        super().__init__(daemon=True)
        self.usrp_sink = usrp_sink
        self._running  = False
        self.hop_index = 0
        self.current_freq = TX_FREQUENCIES[0]

    def run(self):
        self._running = True
        print(f"[TXHop] Started — cycling through: "
              f"{[f/1e9 for f in TX_FREQUENCIES]} GHz, "
              f"dwell={TX_DWELL_SEC*1000:.0f}ms each")
        while self._running:
            freq = TX_FREQUENCIES[self.hop_index % len(TX_FREQUENCIES)]
            self.current_freq = freq
            self.usrp_sink.set_center_freq(freq, 0)
            print(f"[TXHop] -> {freq/1e9:.3f} GHz")
            self.hop_index += 1
            time.sleep(TX_DWELL_SEC)

    def stop(self):
        self._running = False
        print("[TXHop] Stopped")


class TrafficAggregator:
    """
    Collects IQ samples from all 3 HackRFs and determines:
      1. Which receiver is seeing the most traffic (highest mean power)
      2. Which 5 MHz sub-band within that receiver is hottest

    Uses a 5-second rolling window of power spectra so results reflect
    sustained activity rather than instantaneous bursts.
    """

    WINDOW_SEC = 5.0

    def __init__(self, samp_rate=20e6, fft_size=32768):
        self.samp_rate    = samp_rate
        self.fft_size     = fft_size
        self.center_freqs = {}          # rx_id -> center_freq
        self._sample_bufs = {}          # rx_id -> rolling IQ accumulator (numpy)
        self._power_history = {}        # rx_id -> deque of (timestamp, power_db array)
        self._ready       = {}          # rx_id -> bool
        self._lock        = threading.Lock()
        self._report_counter = 0
        self._report_every   = 50       # check every N probe-0 calls

    def register_receiver(self, rx_id, center_freq):
        self.center_freqs[rx_id]    = center_freq
        self._sample_bufs[rx_id]    = np.zeros(self.fft_size, dtype=np.complex64)
        self._power_history[rx_id]  = []   # list of (timestamp, power_db ndarray)
        self._ready[rx_id]          = False

    def update(self, rx_id, samples):
        with self._lock:
            # Roll the sample buffer and fill with new samples
            buf = self._sample_bufs[rx_id]
            n   = len(samples)
            if n >= self.fft_size:
                self._sample_bufs[rx_id] = samples[-self.fft_size:].copy()
            else:
                self._sample_bufs[rx_id] = np.roll(buf, -n)
                self._sample_bufs[rx_id][-n:] = samples

            self._ready[rx_id] = True

            # Compute power spectrum for this chunk and store with timestamp
            power_db = self._compute_spectrum(self._sample_bufs[rx_id])
            now      = time.time()
            self._power_history[rx_id].append((now, power_db))

            # Evict entries older than WINDOW_SEC
            cutoff = now - self.WINDOW_SEC
            self._power_history[rx_id] = [
                (t, p) for t, p in self._power_history[rx_id] if t >= cutoff
            ]

            # Only probe 0 triggers printing
            if rx_id == 0:
                self._report_counter += 1
                if self._report_counter % self._report_every == 0:
                    if all(self._ready.values()):
                        self._print_report()

    def _compute_spectrum(self, buf):
        win      = np.blackman(self.fft_size)
        spectrum = np.fft.fftshift(np.fft.fft(buf * win, self.fft_size))
        power    = np.maximum(np.abs(spectrum) ** 2, 1e-20)
        return 10.0 * np.log10(power / self.fft_size)

    def _windowed_power_db(self, rx_id):
        """Mean power spectrum averaged across all frames in the 5s window."""
        history = self._power_history[rx_id]
        if not history:
            return np.full(self.fft_size, -120.0)
        # Stack all power arrays and average across time axis
        stacked = np.stack([p for _, p in history], axis=0)
        return np.mean(stacked, axis=0)

    # How many dB above the estimated noise floor a bin must be to count as signal.
    # Increase this if you're still seeing noise dominate; decrease if real signals are being missed.
    SIGNAL_THRESHOLD_DB = 6.0

    def _noise_floor_db(self, spectrum):
        """
        Estimate noise floor from the quietest 20% of bins.
        This adapts per-receiver per-report so thermal noise differences
        between HackRFs don't skew the comparison.
        """
        sorted_bins = np.sort(spectrum)
        quiet_bins  = sorted_bins[:max(1, len(sorted_bins) // 5)]
        return float(np.median(quiet_bins))

    def _excess_power_db(self, spectrum):
        """
        Mean power of bins that exceed noise floor + SIGNAL_THRESHOLD_DB.
        Returns 0.0 if no bins clear the threshold (pure noise).
        """
        floor     = self._noise_floor_db(spectrum)
        threshold = floor + self.SIGNAL_THRESHOLD_DB
        above     = spectrum[spectrum > threshold]
        if len(above) == 0:
            return 0.0
        return float(np.mean(above - floor))

    def _print_report(self):
        """Called under lock — builds report from 5s windowed averages."""
        # Build per-receiver averaged spectra
        spectra = {
            rx_id: self._windowed_power_db(rx_id)
            for rx_id in self.center_freqs
        }

        # Compare excess power above each receiver's own noise floor
        excess = {rx_id: self._excess_power_db(spectra[rx_id]) for rx_id in spectra}
        floors = {rx_id: self._noise_floor_db(spectra[rx_id]) for rx_id in spectra}

        busiest_rx = max(excess, key=excess.get)
        center     = self.center_freqs[busiest_rx]

        # Find hottest 2 MHz slice (10 slices across the 20 MHz band)
        avg_spectrum  = spectra[busiest_rx]
        floor         = floors[busiest_rx]
        threshold     = floor + self.SIGNAL_THRESHOLD_DB
        bins_per_2mhz = self.fft_size // 10
        slice_excess  = []
        for i in range(10):
            chunk = avg_spectrum[i * bins_per_2mhz:(i + 1) * bins_per_2mhz]
            above = chunk[chunk > threshold]
            slice_excess.append(float(np.mean(above - floor)) if len(above) > 0 else 0.0)

        best_slice = int(np.argmax(slice_excess))
        low_edge   = (center - 10e6) + best_slice * 2e6
        high_edge  = low_edge + 2e6

        # Actual window duration
        oldest = min(
            self._power_history[rx_id][0][0]
            for rx_id in self.center_freqs
            if self._power_history[rx_id]
        )
        window_actual = min(self.WINDOW_SEC, time.time() - oldest)

        lines = [
            f"\n╔══════════════════════════════════════════╗",
            f"║   TRAFFIC SUMMARY  ({window_actual:.1f}s window)          ║",
            f"╠══════════════════════════════════════════╣",
        ]
        for rx_id in sorted(excess):
            marker = " <<<" if rx_id == busiest_rx else "    "
            if excess[rx_id] == 0.0:
                signal_str = "  no signal above threshold   "
            else:
                signal_str = f"+{excess[rx_id]:5.1f} dB above floor ({floors[rx_id]:.0f} dBFS)"
            lines.append(f"║  HackRF {rx_id}  ({self.center_freqs[rx_id]/1e9:.3f} GHz): {signal_str}{marker}║")
        lines.append("╠══════════════════════════════════════════╣")
        if slice_excess[best_slice] == 0.0:
            lines.append("║  Hottest 2MHz: no slice clears threshold  ║")
        else:
            lines.append(
                f"║  Hottest 2MHz: {low_edge/1e6:.0f}-{high_edge/1e6:.0f} MHz"
                f"  +{slice_excess[best_slice]:.1f} dB above floor"
            )
        lines.append("╚══════════════════════════════════════════╝")
        print("\n".join(lines))


class IQProbe(gr.sync_block):
    """
    Thin passthrough that feeds samples into the TrafficAggregator.
    All probes feed samples; only probe 0 triggers printing.
    """
    def __init__(self, rx_id, aggregator):
        gr.sync_block.__init__(self, name='IQProbe',
                               in_sig=[np.complex64],
                               out_sig=[np.complex64])
        self.rx_id      = rx_id
        self.aggregator = aggregator

    def work(self, input_items, output_items):
        inp = input_items[0]
        output_items[0][:] = inp
        self.aggregator.update(self.rx_id, inp)
        return len(inp)


class GGBLUE(gr.top_block, Qt.QWidget):

    def __init__(self):
        gr.top_block.__init__(self, "GGBLUE - 2.4GHz Monitor + 3-Freq TX", catch_exceptions=True)
        Qt.QWidget.__init__(self)
        self.setWindowTitle("GGBLUE - 2.4GHz Monitor + 3-Freq TX")
        qtgui.util.check_set_qss()
        try:
            self.setWindowIcon(Qt.QIcon.fromTheme('gnuradio-grc'))
        except BaseException as exc:
            print(f"Qt GUI: Could not set Icon: {str(exc)}", file=sys.stderr)

        self.top_scroll_layout = Qt.QVBoxLayout()
        self.setLayout(self.top_scroll_layout)
        self.top_scroll = Qt.QScrollArea()
        self.top_scroll.setFrameStyle(Qt.QFrame.NoFrame)
        self.top_scroll_layout.addWidget(self.top_scroll)
        self.top_scroll.setWidgetResizable(True)
        self.top_widget = Qt.QWidget()
        self.top_scroll.setWidget(self.top_widget)
        self.top_layout = Qt.QVBoxLayout(self.top_widget)
        self.top_grid_layout = Qt.QGridLayout()
        self.top_layout.addLayout(self.top_grid_layout)

        self.settings = Qt.QSettings("gnuradio/flowgraphs", "GGBLUE")
        try:
            geometry = self.settings.value("geometry")
            if geometry:
                self.restoreGeometry(geometry)
        except BaseException as exc:
            print(f"Qt GUI: Could not restore geometry: {str(exc)}", file=sys.stderr)

        self.flowgraph_started = threading.Event()

        ##################################################
        # Variables
        ##################################################
        self.samp_rate   = samp_rate   = 20e6
        self.frequency   = frequency   = 2.42e9
        self.sps         = sps         = 4
        self.symbol_rate = symbol_rate = 2e6 / sps   # TX samp_rate is 2e6 (USRP runs at 1e6 but signal chain uses 2e6)
        self.rolloff     = rolloff     = 0.35
        self.ntaps       = ntaps       = 11 * sps
        self.tx_amp      = tx_amp      = 0.25
        self.tone_hz     = tone_hz     = 100e3
        self.rot_inc     = 2 * 3.141596 * tone_hz / 2e6   # based on TX signal chain samp_rate

        self.rrc_taps = firdes.root_raised_cosine(
            1.0, 2e6, symbol_rate, rolloff, ntaps
        )

        # RX HackRF center frequencies (unchanged)
        self.freq_0 = frequency - 10e6   # 2.410 GHz
        self.freq_1 = frequency + 10e6   # 2.430 GHz
        self.freq_2 = frequency + 30e6   # 2.450 GHz

        ##################################################
        # TX Chain — USRP N210 (UNCHANGED)
        ##################################################
        self.usrp_sink = uhd.usrp_sink(
            ",".join(('addr=192.168.10.3', '')),
            uhd.stream_args(cpu_format="fc32", args='', channels=list(range(0, 1))),
            "",
        )
        self.usrp_sink.set_clock_source('internal', 0)
        self.usrp_sink.set_samp_rate(1e6)
        self.usrp_sink.set_center_freq(TX_FREQUENCIES[0], 0)
        self.usrp_sink.set_antenna("TX/RX", 0)
        self.usrp_sink.set_bandwidth(1e6, 0)
        self.usrp_sink.set_gain(30, 0)

        # Signal chain: BPSK vector -> RRC filter -> amplitude -> frequency rotator -> USRP
        self.vector_source = blocks.vector_source_f(
            [1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1], True, 1, []
        )
        self.rrc_filter = filter.interp_fir_filter_fcc(sps, self.rrc_taps)
        self.rrc_filter.declare_sample_delay(0)
        self.amp     = blocks.multiply_const_cc(tx_amp, 1)
        self.rotator = blocks.rotator_cc(self.rot_inc, False)

        ##################################################
        # RX Sources — 3x HackRF
        # Gain changed to 14 (RF) to match working config
        ##################################################
        self.osmosdr_source_0 = osmosdr.source(args="numchan=1 hackrf=0")
        self.osmosdr_source_0.set_time_unknown_pps(osmosdr.time_spec_t())
        self.osmosdr_source_0.set_sample_rate(20e6)
        self.osmosdr_source_0.set_center_freq(self.freq_0, 0)
        self.osmosdr_source_0.set_freq_corr(0, 0)
        self.osmosdr_source_0.set_dc_offset_mode(0, 0)
        self.osmosdr_source_0.set_iq_balance_mode(0, 0)
        self.osmosdr_source_0.set_gain_mode(False, 0)
        self.osmosdr_source_0.set_gain(14, 0)
        self.osmosdr_source_0.set_if_gain(24, 0)
        self.osmosdr_source_0.set_bb_gain(28, 0)
        self.osmosdr_source_0.set_antenna('', 0)
        self.osmosdr_source_0.set_bandwidth(20e6, 0)

        self.osmosdr_source_1 = osmosdr.source(args="numchan=1 hackrf=1")
        self.osmosdr_source_1.set_time_unknown_pps(osmosdr.time_spec_t())
        self.osmosdr_source_1.set_sample_rate(20e6)
        self.osmosdr_source_1.set_center_freq(self.freq_1, 0)
        self.osmosdr_source_1.set_freq_corr(0, 0)
        self.osmosdr_source_1.set_dc_offset_mode(0, 0)
        self.osmosdr_source_1.set_iq_balance_mode(0, 0)
        self.osmosdr_source_1.set_gain_mode(False, 0)
        self.osmosdr_source_1.set_gain(14, 0)
        self.osmosdr_source_1.set_if_gain(24, 0)
        self.osmosdr_source_1.set_bb_gain(28, 0)
        self.osmosdr_source_1.set_antenna('', 0)
        self.osmosdr_source_1.set_bandwidth(20e6, 0)

        self.osmosdr_source_2 = osmosdr.source(args="numchan=1 hackrf=2")
        self.osmosdr_source_2.set_time_unknown_pps(osmosdr.time_spec_t())
        self.osmosdr_source_2.set_sample_rate(20e6)
        self.osmosdr_source_2.set_center_freq(self.freq_2, 0)
        self.osmosdr_source_2.set_freq_corr(0, 0)
        self.osmosdr_source_2.set_dc_offset_mode(0, 0)
        self.osmosdr_source_2.set_iq_balance_mode(0, 0)
        self.osmosdr_source_2.set_gain_mode(False, 0)
        self.osmosdr_source_2.set_gain(14, 0)
        self.osmosdr_source_2.set_if_gain(24, 0)
        self.osmosdr_source_2.set_bb_gain(28, 0)
        self.osmosdr_source_2.set_antenna('', 0)
        self.osmosdr_source_2.set_bandwidth(20e6, 0)

        ##################################################
        # Traffic Aggregator + IQ Probes
        ##################################################
        self.aggregator = TrafficAggregator(samp_rate=20e6, fft_size=32768)
        self.aggregator.register_receiver(0, self.freq_0)
        self.aggregator.register_receiver(1, self.freq_1)
        self.aggregator.register_receiver(2, self.freq_2)

        self.iq_probe_0 = IQProbe(rx_id=0, aggregator=self.aggregator)
        self.iq_probe_1 = IQProbe(rx_id=1, aggregator=self.aggregator)
        self.iq_probe_2 = IQProbe(rx_id=2, aggregator=self.aggregator)

        ##################################################
        # Channel Activity Monitors (kept, but tapped in parallel — see connections)
        ##################################################
        self.ch_mon_0 = ch_monitor.blk(
            center_freq=self.freq_0,
            samp_rate=20e6,
            channel_bw=20e6,
            threshold_db=-60.0,
            report_every_n_blocks=10
        )
        self.ch_mon_1 = ch_monitor.blk(
            center_freq=self.freq_1,
            samp_rate=20e6,
            channel_bw=20e6,
            threshold_db=-60.0,
            report_every_n_blocks=10
        )
        self.ch_mon_2 = ch_monitor.blk(
            center_freq=self.freq_2,
            samp_rate=20e6,
            channel_bw=20e6,
            threshold_db=-60.0,
            report_every_n_blocks=10
        )

        ##################################################
        # Waterfall Sinks
        ##################################################
        self.waterfall_0 = qtgui.waterfall_sink_c(
            32768, window.WIN_BLACKMAN_hARRIS,
            self.freq_0, 20e6,
            "HackRF 0  |  2400-2420 MHz  |  TX @ 2.410 GHz",
            1, None
        )
        self.waterfall_0.set_update_time(0.10)
        self.waterfall_0.enable_grid(True)
        self.waterfall_0.enable_axis_labels(True)
        self.waterfall_0.set_intensity_range(-140, 10)
        self._waterfall_0_win = sip.wrapinstance(self.waterfall_0.qwidget(), Qt.QWidget)
        self.top_layout.addWidget(self._waterfall_0_win)

        self.waterfall_1 = qtgui.waterfall_sink_c(
            32768, window.WIN_BLACKMAN_hARRIS,
            self.freq_1, 20e6,
            "HackRF 1  |  2420-2440 MHz  |  TX @ 2.430 GHz",
            1, None
        )
        self.waterfall_1.set_update_time(0.10)
        self.waterfall_1.enable_grid(True)
        self.waterfall_1.enable_axis_labels(True)
        self.waterfall_1.set_intensity_range(-140, 10)
        self._waterfall_1_win = sip.wrapinstance(self.waterfall_1.qwidget(), Qt.QWidget)
        self.top_layout.addWidget(self._waterfall_1_win)

        self.waterfall_2 = qtgui.waterfall_sink_c(
            32768, window.WIN_BLACKMAN_hARRIS,
            self.freq_2, 20e6,
            "HackRF 2  |  2440-2460 MHz  |  TX @ 2.450 GHz",
            1, None
        )
        self.waterfall_2.set_update_time(0.10)
        self.waterfall_2.enable_grid(True)
        self.waterfall_2.enable_axis_labels(True)
        self.waterfall_2.set_intensity_range(-140, 10)
        self._waterfall_2_win = sip.wrapinstance(self.waterfall_2.qwidget(), Qt.QWidget)
        self.top_layout.addWidget(self._waterfall_2_win)

        ##################################################
        # Connections
        ##################################################
        # TX: vector -> RRC -> amp -> rotator -> USRP  (UNCHANGED order from working file)
        self.connect((self.vector_source, 0), (self.rrc_filter, 0))
        self.connect((self.rrc_filter,   0), (self.amp,         0))
        self.connect((self.amp,          0), (self.rotator,     0))
        self.connect((self.rotator,      0), (self.usrp_sink,   0))

        # RX: HackRF -> IQ probe -> waterfall
        # IQ probe is a passthrough that feeds the TrafficAggregator
        self.connect((self.osmosdr_source_0, 0), (self.iq_probe_0,   0))
        self.connect((self.osmosdr_source_1, 0), (self.iq_probe_1,   0))
        self.connect((self.osmosdr_source_2, 0), (self.iq_probe_2,   0))
        self.connect((self.iq_probe_0,       0), (self.waterfall_0,  0))
        self.connect((self.iq_probe_1,       0), (self.waterfall_1,  0))
        self.connect((self.iq_probe_2,       0), (self.waterfall_2,  0))

        # Channel monitors in parallel (tap directly from source)
        self.connect((self.osmosdr_source_0, 0), (self.ch_mon_0, 0))
        self.connect((self.osmosdr_source_1, 0), (self.ch_mon_1, 0))
        self.connect((self.osmosdr_source_2, 0), (self.ch_mon_2, 0))

        # Null sinks for channel monitor outputs (output port must be connected)
        self.null_sink_0 = blocks.null_sink(gr.sizeof_gr_complex)
        self.null_sink_1 = blocks.null_sink(gr.sizeof_gr_complex)
        self.null_sink_2 = blocks.null_sink(gr.sizeof_gr_complex)
        self.connect((self.ch_mon_0, 0), (self.null_sink_0, 0))
        self.connect((self.ch_mon_1, 0), (self.null_sink_1, 0))
        self.connect((self.ch_mon_2, 0), (self.null_sink_2, 0))

        # TX hop controller (started in start())
        self.tx_hopper = TXHopController(self.usrp_sink)

    def start(self):
        super().start()
        self.tx_hopper.start()

    def stop(self):
        self.tx_hopper.stop()
        super().stop()

    def closeEvent(self, event):
        self.settings = Qt.QSettings("gnuradio/flowgraphs", "GGBLUE")
        self.settings.setValue("geometry", self.saveGeometry())
        self.stop()
        self.wait()
        event.accept()

    def get_frequency(self):
        return self.frequency

    def set_frequency(self, frequency):
        self.frequency = frequency
        self.freq_0 = frequency - 10e6
        self.freq_1 = frequency + 10e6
        self.freq_2 = frequency + 30e6
        self.osmosdr_source_0.set_center_freq(self.freq_0, 0)
        self.osmosdr_source_1.set_center_freq(self.freq_1, 0)
        self.osmosdr_source_2.set_center_freq(self.freq_2, 0)
        self.waterfall_0.set_frequency_range(self.freq_0, 20e6)
        self.waterfall_1.set_frequency_range(self.freq_1, 20e6)
        self.waterfall_2.set_frequency_range(self.freq_2, 20e6)

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate


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