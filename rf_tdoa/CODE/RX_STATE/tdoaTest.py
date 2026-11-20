#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from gnuradio import gr, blocks, filter
from gnuradio.filter import firdes
from gnuradio.fft import window
import numpy as np
import osmosdr
import time
import signal
import sys

# =============================
# User settings
# =============================
CENTER_FREQ_HZ = 2.420e9
SAMP_RATE      = 20e6
BW_HZ          = 20e6

TONE_OFFSET_HZ = 200e3

DECIM          = 10                 # 20 Msps -> 2 Msps
FS_DECIM       = SAMP_RATE / DECIM
LP_CUTOFF_HZ   = 120e3
LP_TRANS_HZ    = 40e3

FRAME_N        = 16384              # ~8.2 ms at 2 Msps
PRINT_EVERY    = 0.5

USE_GPSDO_EXTERNAL_10MHZ = True
ENABLE_DC_OFFSET_MODE    = True
ENABLE_IQ_BALANCE_MODE   = True

MAX_LAG_SAMPLES = 5000


def make_hackrf(idx: int) -> osmosdr.source:
    base = f"numchan=1 hackrf={idx}"
    if USE_GPSDO_EXTERNAL_10MHZ:
        candidates = [
            f"{base},clock_source=external",
            f"{base},ext_clock=1",
            base,
        ]
    else:
        candidates = [base]

    last = None
    for args in candidates:
        try:
            src = osmosdr.source(args=args)
            print(f"[HackRF{idx}] args OK: {args}")
            src.set_time_unknown_pps(osmosdr.time_spec_t())
            src.set_sample_rate(SAMP_RATE)
            src.set_center_freq(CENTER_FREQ_HZ, 0)
            src.set_freq_corr(0, 0)
            src.set_dc_offset_mode(1 if ENABLE_DC_OFFSET_MODE else 0, 0)
            src.set_iq_balance_mode(1 if ENABLE_IQ_BALANCE_MODE else 0, 0)
            src.set_gain_mode(False, 0)
            src.set_gain(14, 0)
            src.set_if_gain(24, 0)
            src.set_bb_gain(28, 0)
            src.set_bandwidth(BW_HZ, 0)
            return src
        except Exception as e:
            print(f"[HackRF{idx}] args failed: {args} -> {e}")
            last = e

    raise RuntimeError(f"Could not create HackRF{idx}. Last error: {last}")


def gcc_phat(x: np.ndarray, y: np.ndarray, max_lag: int) -> tuple[int, float]:
    """
    GCC-PHAT delay estimate.
    Returns (lag_samples, peak_value).
    lag_samples > 0 means y lags behind x (y occurs later).
    """
    n = len(x) + len(y)
    nfft = 1 << (n - 1).bit_length()

    X = np.fft.rfft(x, nfft)
    Y = np.fft.rfft(y, nfft)
    R = X * np.conj(Y)
    denom = np.abs(R)
    denom[denom < 1e-12] = 1e-12
    R /= denom

    cc = np.fft.irfft(R, nfft)
    cc = np.concatenate((cc[-(len(y)-1):], cc[:len(x)]))

    mid = len(cc) // 2
    lo = max(0, mid - max_lag)
    hi = min(len(cc), mid + max_lag + 1)
    window_cc = cc[lo:hi]

    k = int(np.argmax(np.abs(window_cc)))
    peak = float(window_cc[k])
    lag = (lo + k) - mid
    return lag, peak


class TDOAEstimator3(gr.sync_block):
    def __init__(self):
        gr.sync_block.__init__(
            self,
            name="TDOAEstimator3",
            in_sig=[np.complex64, np.complex64, np.complex64],
            out_sig=None
        )
        self.buf0 = np.empty(0, dtype=np.complex64)
        self.buf1 = np.empty(0, dtype=np.complex64)
        self.buf2 = np.empty(0, dtype=np.complex64)
        self.last_print = time.time()

    def work(self, input_items, output_items):
        x0 = input_items[0]
        x1 = input_items[1]
        x2 = input_items[2]

        self.buf0 = np.concatenate((self.buf0, x0))
        self.buf1 = np.concatenate((self.buf1, x1))
        self.buf2 = np.concatenate((self.buf2, x2))

        while len(self.buf0) >= FRAME_N and len(self.buf1) >= FRAME_N and len(self.buf2) >= FRAME_N:
            f0 = self.buf0[:FRAME_N]; self.buf0 = self.buf0[FRAME_N:]
            f1 = self.buf1[:FRAME_N]; self.buf1 = self.buf1[FRAME_N:]
            f2 = self.buf2[:FRAME_N]; self.buf2 = self.buf2[FRAME_N:]

            f0 = f0 - np.mean(f0)
            f1 = f1 - np.mean(f1)
            f2 = f2 - np.mean(f2)

            lag01, pk01 = gcc_phat(f0, f1, MAX_LAG_SAMPLES)
            lag02, pk02 = gcc_phat(f0, f2, MAX_LAG_SAMPLES)

            now = time.time()
            if now - self.last_print >= PRINT_EVERY:
                t01_us = (lag01 / FS_DECIM) * 1e6
                t02_us = (lag02 / FS_DECIM) * 1e6
                print(
                    f"[TDOA] Fs={FS_DECIM/1e6:.3f} Msps | "
                    f"lag01={lag01:+6d} ({t01_us:+9.2f} us) pk={pk01:+.3f} | "
                    f"lag02={lag02:+6d} ({t02_us:+9.2f} us) pk={pk02:+.3f}"
                )
                self.last_print = now

        return len(x0)


class Top(gr.top_block):
    def __init__(self):
        super().__init__("TDOA Test 3x HackRF", catch_exceptions=True)

        s0 = make_hackrf(0)
        s1 = make_hackrf(1)
        s2 = make_hackrf(2)

        taps = firdes.low_pass(
            1.0,
            SAMP_RATE,
            LP_CUTOFF_HZ,
            LP_TRANS_HZ,
            window.WIN_HAMMING
        )

        x0 = filter.freq_xlating_fir_filter_ccc(DECIM, taps, -TONE_OFFSET_HZ, SAMP_RATE)
        x1 = filter.freq_xlating_fir_filter_ccc(DECIM, taps, -TONE_OFFSET_HZ, SAMP_RATE)
        x2 = filter.freq_xlating_fir_filter_ccc(DECIM, taps, -TONE_OFFSET_HZ, SAMP_RATE)

        tdoa = TDOAEstimator3()

        self.connect((s0, 0), (x0, 0))
        self.connect((s1, 0), (x1, 0))
        self.connect((s2, 0), (x2, 0))

        self.connect((x0, 0), (tdoa, 0))
        self.connect((x1, 0), (tdoa, 1))
        self.connect((x2, 0), (tdoa, 2))


def main():
    tb = Top()

    def sig_handler(sig=None, frame=None):
        try:
            tb.stop()
            tb.wait()
        except Exception:
            pass
        sys.exit(0)

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    tb.start()
    print("[INFO] Running. Ctrl+C to stop.")
    tb.wait()


if __name__ == "__main__":
    main()