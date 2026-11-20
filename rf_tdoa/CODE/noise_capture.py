#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Noise-only capture — transmitter OFF
# Records rx0.cfile, rx1.cfile, rx2.cfile for a set duration then stops.
#

from PyQt5 import Qt
from gnuradio import qtgui
from gnuradio import blocks
from gnuradio import gr
from gnuradio.fft import window
import sys
import signal
import osmosdr
import sip
import threading
import time


SAVE_DIR    = r"G:\GITFINAL\rf_tdoa\DATA\NOISE_ONLY"
DURATION_SEC = 30          # how many seconds to record
SAMP_RATE    = 1e6
FREQUENCY    = 2.426e9
RX_GAIN      = 14
IF_GAIN      = 24
BB_GAIN      = 0


class NoiseCap(gr.top_block, Qt.QWidget):

    def __init__(self):
        gr.top_block.__init__(self, "NoiseCap", catch_exceptions=True)
        Qt.QWidget.__init__(self)
        self.setWindowTitle(f"Noise Capture — {DURATION_SEC}s @ {SAMP_RATE/1e6:.1f} MHz")

        self.top_scroll_layout = Qt.QVBoxLayout()
        self.setLayout(self.top_scroll_layout)
        self.top_scroll = Qt.QScrollArea()
        self.top_scroll.setFrameStyle(Qt.QFrame.NoFrame)
        self.top_scroll_layout.addWidget(self.top_scroll)
        self.top_scroll.setWidgetResizable(True)
        self.top_widget = Qt.QWidget()
        self.top_scroll.setWidget(self.top_widget)
        self.top_layout = Qt.QVBoxLayout(self.top_widget)

        self.flowgraph_started = threading.Event()

        import os
        os.makedirs(SAVE_DIR, exist_ok=True)

        ##################################################
        # HackRF sources
        ##################################################
        self.osmosdr_source_0 = osmosdr.source(args="numchan=1 hackrf=0")
        self.osmosdr_source_0.set_time_unknown_pps(osmosdr.time_spec_t())
        self.osmosdr_source_0.set_sample_rate(SAMP_RATE)
        self.osmosdr_source_0.set_center_freq(FREQUENCY, 0)
        self.osmosdr_source_0.set_freq_corr(0, 0)
        self.osmosdr_source_0.set_dc_offset_mode(0, 0)
        self.osmosdr_source_0.set_iq_balance_mode(0, 0)
        self.osmosdr_source_0.set_gain_mode(False, 0)
        self.osmosdr_source_0.set_gain(RX_GAIN, 0)
        self.osmosdr_source_0.set_if_gain(IF_GAIN, 0)
        self.osmosdr_source_0.set_bb_gain(BB_GAIN, 0)
        self.osmosdr_source_0.set_antenna("", 0)
        self.osmosdr_source_0.set_bandwidth(SAMP_RATE, 0)

        self.osmosdr_source_1 = osmosdr.source(args="numchan=1 hackrf=1")
        self.osmosdr_source_1.set_time_unknown_pps(osmosdr.time_spec_t())
        self.osmosdr_source_1.set_sample_rate(SAMP_RATE)
        self.osmosdr_source_1.set_center_freq(FREQUENCY, 0)
        self.osmosdr_source_1.set_freq_corr(0, 0)
        self.osmosdr_source_1.set_dc_offset_mode(0, 0)
        self.osmosdr_source_1.set_iq_balance_mode(0, 0)
        self.osmosdr_source_1.set_gain_mode(False, 0)
        self.osmosdr_source_1.set_gain(RX_GAIN, 0)
        self.osmosdr_source_1.set_if_gain(IF_GAIN, 0)
        self.osmosdr_source_1.set_bb_gain(BB_GAIN, 0)
        self.osmosdr_source_1.set_antenna("", 0)
        self.osmosdr_source_1.set_bandwidth(SAMP_RATE, 0)

        self.osmosdr_source_2 = osmosdr.source(args="numchan=1 hackrf=2")
        self.osmosdr_source_2.set_time_unknown_pps(osmosdr.time_spec_t())
        self.osmosdr_source_2.set_sample_rate(SAMP_RATE)
        self.osmosdr_source_2.set_center_freq(FREQUENCY, 0)
        self.osmosdr_source_2.set_freq_corr(0, 0)
        self.osmosdr_source_2.set_dc_offset_mode(0, 0)
        self.osmosdr_source_2.set_iq_balance_mode(0, 0)
        self.osmosdr_source_2.set_gain_mode(False, 0)
        self.osmosdr_source_2.set_gain(RX_GAIN, 0)
        self.osmosdr_source_2.set_if_gain(IF_GAIN, 0)
        self.osmosdr_source_2.set_bb_gain(BB_GAIN, 0)
        self.osmosdr_source_2.set_antenna("", 0)
        self.osmosdr_source_2.set_bandwidth(SAMP_RATE, 0)

        ##################################################
        # Waterfalls (so you can see what's being captured)
        ##################################################
        self.waterfall_0 = qtgui.waterfall_sink_c(
            1024, window.WIN_BLACKMAN_hARRIS,
            FREQUENCY, SAMP_RATE, "HackRF 0 — NOISE", 1, None)
        self.waterfall_0.set_update_time(0.10)
        self.waterfall_0.set_intensity_range(-140, 10)
        self.top_layout.addWidget(sip.wrapinstance(self.waterfall_0.qwidget(), Qt.QWidget))

        self.waterfall_1 = qtgui.waterfall_sink_c(
            1024, window.WIN_BLACKMAN_hARRIS,
            FREQUENCY, SAMP_RATE, "HackRF 1 — NOISE", 1, None)
        self.waterfall_1.set_update_time(0.10)
        self.waterfall_1.set_intensity_range(-140, 10)
        self.top_layout.addWidget(sip.wrapinstance(self.waterfall_1.qwidget(), Qt.QWidget))

        self.waterfall_2 = qtgui.waterfall_sink_c(
            1024, window.WIN_BLACKMAN_hARRIS,
            FREQUENCY, SAMP_RATE, "HackRF 2 — NOISE", 1, None)
        self.waterfall_2.set_update_time(0.10)
        self.waterfall_2.set_intensity_range(-140, 10)
        self.top_layout.addWidget(sip.wrapinstance(self.waterfall_2.qwidget(), Qt.QWidget))

        ##################################################
        # File sinks
        ##################################################
        import os
        self.sink_rx0 = blocks.file_sink(gr.sizeof_gr_complex,
                                          os.path.join(SAVE_DIR, "rx0.cfile"), False)
        self.sink_rx0.set_unbuffered(False)

        self.sink_rx1 = blocks.file_sink(gr.sizeof_gr_complex,
                                          os.path.join(SAVE_DIR, "rx1.cfile"), False)
        self.sink_rx1.set_unbuffered(False)

        self.sink_rx2 = blocks.file_sink(gr.sizeof_gr_complex,
                                          os.path.join(SAVE_DIR, "rx2.cfile"), False)
        self.sink_rx2.set_unbuffered(False)

        ##################################################
        # Connections
        ##################################################
        self.connect((self.osmosdr_source_0, 0), (self.waterfall_0, 0))
        self.connect((self.osmosdr_source_0, 0), (self.sink_rx0,    0))

        self.connect((self.osmosdr_source_1, 0), (self.waterfall_1, 0))
        self.connect((self.osmosdr_source_1, 0), (self.sink_rx1,    0))

        self.connect((self.osmosdr_source_2, 0), (self.waterfall_2, 0))
        self.connect((self.osmosdr_source_2, 0), (self.sink_rx2,    0))

    def closeEvent(self, event):
        self.stop()
        self.wait()
        event.accept()


def main():
    qapp = Qt.QApplication(sys.argv)
    tb = NoiseCap()

    tb.start()
    tb.flowgraph_started.set()
    tb.show()

    print(f"\n{'='*50}")
    print(f"  NOISE CAPTURE STARTED")
    print(f"  Duration : {DURATION_SEC} seconds")
    print(f"  Saving to: {SAVE_DIR}")
    print(f"  Make sure the transmitter is OFF!")
    print(f"{'='*50}\n")

    def sig_handler(sig=None, frame=None):
        print("\n[!] Interrupted — stopping capture.")
        tb.stop()
        tb.wait()
        Qt.QApplication.quit()

    signal.signal(signal.SIGINT,  sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    # Auto-stop after DURATION_SEC
    def auto_stop():
        for remaining in range(DURATION_SEC, 0, -5):
            time.sleep(5)
            print(f"  [{remaining - 5}s remaining...]" if remaining > 5 else "  [Stopping...]")
        tb.stop()
        tb.wait()
        print(f"\n{'='*50}")
        print(f"  ✓ Capture complete!")
        print(f"  Files saved to: {SAVE_DIR}")
        print(f"  Now run: python build_filters.py --noise-dir \"{SAVE_DIR}\" --out-dir <your filter dir>")
        print(f"{'='*50}\n")
        Qt.QApplication.quit()

    t = threading.Thread(target=auto_stop, daemon=True)
    t.start()

    timer = Qt.QTimer()
    timer.start(500)
    timer.timeout.connect(lambda: None)

    qapp.exec_()


if __name__ == "__main__":
    main()