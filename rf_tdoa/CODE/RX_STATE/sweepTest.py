#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# GGBLUE - Purple Guy SSTV-style waterfall painter
#
# Transmits an image as a signal visible on the RX waterfall:
#   - Each row of the image = one time slice
#   - Each pixel column = a frequency bin
#   - Pixel brightness = amplitude at that frequency
#   - Loops forever
#
# TX: USRP N210 @ 192.168.10.3
# RX: 3x HackRF on external GPSDO 10MHz clock, centered at 2.420 GHz
#

from PyQt5 import Qt
from gnuradio import qtgui
from gnuradio import blocks
from gnuradio import gr
from gnuradio.fft import window as fft_window
import sys
import signal
import threading
import numpy as np
from PIL import Image

from gnuradio import uhd
import osmosdr
import sip

# =============================================================================
# CONFIG
# =============================================================================
RX_CENTER_HZ   = 2.420e9
RX_SAMP_RATE   = 2e6
RX_BW_HZ       = 2e6
TX_SAMP_RATE   = 2e6        # match RX so image fills the same bandwidth

USE_EXTERNAL_CLOCK = True
SINK_PATH = "G:/GITFINAL/rf_tdoa/CODE/RX_STATE/capture"

IMAGE_PATH     = "G:/GITFINAL/rf_tdoa/CODE/RX_STATE/freddy.jpg"
IMAGE_WIDTH    = 512        # frequency bins (cols) — must be power of 2
IMAGE_HEIGHT   = 512        # rows to resize image to
ROW_DURATION   = 0.10       # seconds per image row → full image takes ~12.8s
TX_GAIN        = 15


# =============================================================================
# Image → IQ sample generator
# =============================================================================

def load_image_as_iq(path, width, height, samp_rate, row_duration):
    """
    Convert image to a flat array of IQ samples.
    Each image row becomes one IFFT frame: pixel brightness → spectral amplitude.
    Returns: np.array of complex64, length = width * height * samples_per_row
    """
    img = Image.open(path).convert("L")           # grayscale
    img = img.resize((width, height), Image.LANCZOS)
    pixels = np.array(img, dtype=np.float32) / 255.0   # 0..1
    # Contrast stretch + invert + gamma to maximise waterfall visibility
    pixels = pixels - pixels.min()
    if pixels.max() > 0:
        pixels /= pixels.max()
    # Dark art: already dark bg, bright highlights = signal — no invert needed
    # Stretch contrast and boost highlights
    pixels = np.clip((pixels - 0.05) / 0.95, 0, 1)  # crush near-blacks
    pixels = np.power(pixels, 0.6)   # boost midtones/highlights

    # Flip vertically so top of image appears at top of waterfall
    pixels = np.flipud(pixels)

    samples_per_row = int(samp_rate * row_duration)
    # Each row repeats for samples_per_row samples via IFFT tiling
    frames_per_row  = samples_per_row // width

    all_samples = []
    for row in pixels:
        # Build spectrum: magnitude = pixel brightness, random phase
        # Centre the spectrum (DC in middle)
        spectrum = np.fft.ifftshift(row)
        phases   = np.exp(1j * np.random.uniform(0, 2*np.pi, width))
        freq_domain = spectrum * phases

        # IFFT to get time-domain IQ
        frame = np.fft.ifft(freq_domain).astype(np.complex64)

        # Normalize
        peak = np.max(np.abs(frame))
        if peak > 0:
            frame /= peak

        # Tile the frame to fill row duration
        tiled = np.tile(frame, frames_per_row)
        all_samples.append(tiled)

    return np.concatenate(all_samples).astype(np.complex64)


# =============================================================================
# IQ Vector Source with looping
# =============================================================================

class LoopingVectorSource(gr.sync_block):
    """Outputs a pre-computed IQ buffer, looping forever."""

    def __init__(self, samples):
        gr.sync_block.__init__(
            self,
            name="LoopingVectorSource",
            in_sig=None,
            out_sig=[np.complex64],
        )
        self.samples = samples
        self.pos     = 0

    def work(self, input_items, output_items):
        out = output_items[0]
        n   = len(out)
        written = 0

        while written < n:
            remaining_in_buf = len(self.samples) - self.pos
            chunk = min(n - written, remaining_in_buf)
            out[written:written+chunk] = self.samples[self.pos:self.pos+chunk]
            written   += chunk
            self.pos  += chunk
            if self.pos >= len(self.samples):
                self.pos = 0   # loop

        return n


# =============================================================================
# Top Block
# =============================================================================

class GGBLUE(gr.top_block, Qt.QWidget):

    def __init__(self):
        gr.top_block.__init__(self, "GGBLUE - Purple Guy Waterfall",
                              catch_exceptions=True)
        Qt.QWidget.__init__(self)
        self.setWindowTitle("GGBLUE - Purple Guy Waterfall")

        qtgui.util.check_set_qss()
        try:
            self.setWindowIcon(Qt.QIcon.fromTheme('gnuradio-grc'))
        except Exception as exc:
            print(f"[GUI] Could not set icon: {exc}", file=sys.stderr)

        self.top_scroll_layout = Qt.QVBoxLayout()
        self.setLayout(self.top_scroll_layout)
        self.top_scroll = Qt.QScrollArea()
        self.top_scroll.setFrameStyle(Qt.QFrame.NoFrame)
        self.top_scroll_layout.addWidget(self.top_scroll)
        self.top_scroll.setWidgetResizable(True)
        self.top_widget = Qt.QWidget()
        self.top_scroll.setWidget(self.top_widget)
        self.top_layout = Qt.QVBoxLayout(self.top_widget)

        self.settings = Qt.QSettings("gnuradio/flowgraphs", "GGBLUE")
        try:
            geom = self.settings.value("geometry")
            if geom:
                self.restoreGeometry(geom)
        except Exception:
            pass

        self.flowgraph_started = threading.Event()
        self.samp_rate = RX_SAMP_RATE
        self.frequency = RX_CENTER_HZ

        # ------------------------------------------------------------------
        # Load and encode image
        # ------------------------------------------------------------------
        print(f"[IMG] Loading image from {IMAGE_PATH} ...")
        try:
            iq_samples = load_image_as_iq(
                IMAGE_PATH, IMAGE_WIDTH, IMAGE_HEIGHT,
                TX_SAMP_RATE, ROW_DURATION
            )
        except FileNotFoundError:
            print(f"[IMG] File not found at {IMAGE_PATH}")
            print(f"[IMG] Copy purpleguy.jpg to that path, or update IMAGE_PATH in the script")
            raise

        print(f"[IMG] Encoded: {len(iq_samples)} samples "
              f"({len(iq_samples)/TX_SAMP_RATE:.1f}s per loop)")

        self.img_source = LoopingVectorSource(iq_samples)
        self.tx_scale   = blocks.multiply_const_cc(0.9)

        # ------------------------------------------------------------------
        # TX — USRP N210
        # ------------------------------------------------------------------
        print(f"[TX] Connecting to N210 @ 192.168.10.3 ...")
        self.uhd_usrp_sink = uhd.usrp_sink(
            ",".join(("addr=192.168.10.3", "")),
            uhd.stream_args(cpu_format="fc32", args="", channels=[0]),
            "",
        )
        self.uhd_usrp_sink.set_clock_source("internal", 0)
        self.uhd_usrp_sink.set_time_now(uhd.time_spec(0.0), uhd.ALL_MBOARDS)
        self.uhd_usrp_sink.set_samp_rate(TX_SAMP_RATE)
        self.uhd_usrp_sink.set_center_freq(RX_CENTER_HZ, 0)
        self.uhd_usrp_sink.set_antenna("TX/RX", 0)
        self.uhd_usrp_sink.set_bandwidth(TX_SAMP_RATE, 0)
        self.uhd_usrp_sink.set_gain(TX_GAIN, 0)
        print(f"[TX] N210 ready  gain={TX_GAIN}dB  samp_rate={TX_SAMP_RATE/1e6:.0f}Msps")

        # ------------------------------------------------------------------
        # RX — 3x HackRF
        # ------------------------------------------------------------------
        clock_mode = "external" if USE_EXTERNAL_CLOCK else "internal"
        print(f"[RX] HackRF clock={clock_mode}  "
              f"center={RX_CENTER_HZ/1e9:.3f}GHz  "
              f"BW={RX_BW_HZ/1e6:.0f}MHz  "
              f"rate={RX_SAMP_RATE/1e6:.0f}Msps")

        self.hackrf = [self._make_hackrf(i, self.frequency) for i in range(3)]

        # ------------------------------------------------------------------
        # File sinks
        # ------------------------------------------------------------------
        self.file_sinks = []
        for i in range(3):
            path = f"{SINK_PATH}_rx{i}.cfile"
            sink = blocks.file_sink(gr.sizeof_gr_complex, path, False)
            sink.set_unbuffered(False)
            self.file_sinks.append(sink)
            print(f"[SINK] rx{i} -> {path}")

        # ------------------------------------------------------------------
        # Waterfalls
        # ------------------------------------------------------------------
        self.waterfalls = []
        for i in range(3):
            wf = qtgui.waterfall_sink_c(
                IMAGE_WIDTH,                    # FFT size = image width
                fft_window.WIN_BLACKMAN_hARRIS,
                self.frequency,
                self.samp_rate,
                f"HackRF {i}",
                1, None
            )
            wf.set_update_time(ROW_DURATION)    # update rate matches row duration
            wf.enable_grid(True)
            wf.enable_axis_labels(True)
            wf.set_intensity_range(-80, 0)
            win = sip.wrapinstance(wf.qwidget(), Qt.QWidget)
            self.top_layout.addWidget(win)
            self.waterfalls.append(wf)

        # ------------------------------------------------------------------
        # Connections
        # ------------------------------------------------------------------
        self.connect((self.img_source, 0), (self.tx_scale,      0))
        self.connect((self.tx_scale,   0), (self.uhd_usrp_sink, 0))

        for i in range(3):
            self.connect((self.hackrf[i], 0), (self.file_sinks[i], 0))
            self.connect((self.hackrf[i], 0), (self.waterfalls[i],  0))

    # ------------------------------------------------------------------
    # HackRF factory
    # ------------------------------------------------------------------
    def _make_hackrf(self, idx, center_freq):
        candidates = []
        if USE_EXTERNAL_CLOCK:
            candidates += [
                f"numchan=1 hackrf={idx},clock_source=external",
                f"numchan=1 hackrf={idx},ext_clock=1",
            ]
        candidates.append(f"numchan=1 hackrf={idx}")

        src = None
        for args in candidates:
            try:
                src = osmosdr.source(args=args)
                print(f"[HackRF{idx}] Opened: {args}")
                break
            except Exception as exc:
                print(f"[HackRF{idx}] Failed ({args}): {exc}")

        if src is None:
            raise RuntimeError(f"Could not open HackRF {idx}")

        src.set_sample_rate(RX_SAMP_RATE)
        src.set_center_freq(center_freq, 0)
        src.set_freq_corr(0, 0)
        src.set_dc_offset_mode(0, 0)
        src.set_iq_balance_mode(0, 0)
        src.set_gain_mode(False, 0)
        src.set_gain(0, 0)
        src.set_if_gain(16, 0)
        src.set_bb_gain(16, 0)
        src.set_antenna("", 0)
        src.set_bandwidth(RX_BW_HZ, 0)
        return src

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self):
        super().start()

    def stop(self):
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
        for src in self.hackrf:
            src.set_center_freq(frequency, 0)
        for wf in self.waterfalls:
            wf.set_frequency_range(frequency, RX_BW_HZ)


# =============================================================================
# Entry point
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

    signal.signal(signal.SIGINT,  sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    timer = Qt.QTimer()
    timer.start(500)
    timer.timeout.connect(lambda: None)

    qapp.exec_()


if __name__ == "__main__":
    main()