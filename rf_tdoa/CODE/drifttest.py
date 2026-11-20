#!/usr/bin/env python3
# drifttest.py
#
# Robust frequency drift estimator for complex64 (.cfile) IQ recordings.
# Adds:
#  - tighter tracking around last estimate (prevents jumping to random peaks)
#  - outlier rejection via --max-jump-hz
#  - optional hold-last behavior
#  - optional NaN dropping in summary
#
# Typical use (tone at +1 MHz baseband, fs=2 MHz):
#   python drifttest.py --fs 2e6 --target-hz 1e6 --search-hz 20e3 --nfft 65536 --block-len 65536 --hop-len 32768 \
#       --max-jump-hz 2000 --hold-last G:/gpsdo_test_rx0.cfile
#
# Multi-RX:
#   python drifttest.py --fs 2e6 --target-hz 1e6 --search-hz 20e3 --nfft 65536 --block-len 65536 --hop-len 32768 \
#       --max-jump-hz 2000 --hold-last G:/gpsdo_test_rx0.cfile G:/gpsdo_test_rx1.cfile G:/gpsdo_test_rx2.cfile

import argparse
import os
import sys
import numpy as np


def parsenum(x: str) -> float:
    """Allow things like 2e6, 1e6, 200e3."""
    try:
        return float(x)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Could not parse number: {x}")


def parabolic_interpolation(y_minus, y0, y_plus):
    """
    Quadratic interpolation around a peak, returns delta in bins.
    Works on magnitudes (or log magnitudes). Avoid if peak at edge.
    """
    denom = (y_minus - 2.0 * y0 + y_plus)
    if denom == 0:
        return 0.0
    return 0.5 * (y_minus - y_plus) / denom


def estimate_tone_freqs(
    path: str,
    fs: float,
    target_hz: float,
    search_hz: float,
    nfft: int,
    block_len: int,
    hop_len: int,
    skip_seconds: float,
    max_seconds: float,
    window: str,
    max_jump_hz: float,
    hold_last: bool,
    track_around_last: bool,
):
    """
    Return times (s) and estimated tone frequency (Hz) per block.

    Tracking logic:
      - Peak search is done within a window around target_hz.
      - If we have a previous estimate and track_around_last is enabled,
        the window is further constrained to be around last estimate (±search_hz).
      - If the new estimate jumps more than max_jump_hz, it's treated as an outlier:
          * hold_last=True  -> repeat last value
          * hold_last=False -> emit NaN
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    if hop_len > block_len:
        raise ValueError("hop_len must be <= block_len")

    dtype = np.complex64
    itemsize = np.dtype(dtype).itemsize

    skip_samps = int(skip_seconds * fs)
    max_samps = int(max_seconds * fs) if max_seconds and max_seconds > 0 else None

    if window.lower() == "hann":
        win = np.hanning(block_len).astype(np.float32)
    elif window.lower() == "rect":
        win = np.ones(block_len, dtype=np.float32)
    else:
        raise ValueError("window must be 'hann' or 'rect'")

    # fftshifted bin frequencies
    freqs = np.fft.fftshift(np.fft.fftfreq(nfft, d=1.0 / fs))

    # Base search window around expected tone
    base_lo = target_hz - search_hz
    base_hi = target_hz + search_hz

    times = []
    est_hz = []

    block_bytes = block_len * itemsize
    hop_bytes = hop_len * itemsize
    keep = block_len - hop_len

    with open(path, "rb") as f:
        f.seek(skip_samps * itemsize, os.SEEK_SET)

        buf = f.read(block_bytes)
        if len(buf) < block_bytes:
            return np.array([]), np.array([])

        block = np.frombuffer(buf, dtype=dtype)
        total_read = block_len
        i = 0

        while True:
            if max_samps is not None and total_read > max_samps:
                break

            x = block * win
            X = np.fft.fftshift(np.fft.fft(x, n=nfft))
            mag = np.abs(X)

            # Dynamic window: around target, optionally around last estimate
            if track_around_last and est_hz and np.isfinite(est_hz[-1]):
                lo = max(base_lo, est_hz[-1] - search_hz)
                hi = min(base_hi, est_hz[-1] + search_hz)
            else:
                lo, hi = base_lo, base_hi

            mask = (freqs >= lo) & (freqs <= hi)
            if not np.any(mask):
                # If window became empty, fallback to base window
                mask = (freqs >= base_lo) & (freqs <= base_hi)
                if not np.any(mask):
                    # Can't do anything
                    t = (i * hop_len + (block_len / 2)) / fs
                    times.append(t)
                    est_hz.append(np.nan)
                    # move to next block
                    pass
                # continue normally with fallback mask

            mask_idx = np.where(mask)[0]
            sub = mag[mask_idx]
            k_sub = int(np.argmax(sub))
            k = int(mask_idx[k_sub])

            if 1 <= k < (nfft - 1):
                delta = parabolic_interpolation(mag[k - 1], mag[k], mag[k + 1])
            else:
                delta = 0.0

            f_est = freqs[k] + delta * (fs / nfft)

            # Outlier rejection
            if est_hz and np.isfinite(est_hz[-1]):
                prev = est_hz[-1]
                if abs(f_est - prev) > max_jump_hz:
                    if hold_last:
                        f_est = prev
                    else:
                        f_est = np.nan

            t = (i * hop_len + (block_len / 2)) / fs
            times.append(t)
            est_hz.append(f_est)

            # Next block
            overlap = block[-keep:] if keep > 0 else np.empty((0,), dtype=dtype)
            new = f.read(hop_bytes)
            if len(new) < hop_bytes:
                break

            new_samps = np.frombuffer(new, dtype=dtype)
            block = np.concatenate([overlap, new_samps])
            if block.shape[0] != block_len:
                break

            total_read += hop_len
            i += 1

    return np.array(times), np.array(est_hz)


def summarize(freqs_hz: np.ndarray, label: str, fs: float, target_hz: float, drop_nans: bool):
    if freqs_hz.size == 0:
        print(f"[{label}] No estimates produced.")
        return None

    f = freqs_hz
    if drop_nans:
        f = f[np.isfinite(f)]

    if f.size == 0:
        print(f"[{label}] All estimates were NaN (tone not tracked).")
        return None

    fmin = float(np.min(f))
    fmax = float(np.max(f))
    fmean = float(np.mean(f))
    fpp = fmax - fmin

    ref = abs(target_hz) if abs(target_hz) > 1.0 else fs
    ppm = (fpp / ref) * 1e6

    print(f"\n[{label}]")
    print(f"  mean   : {fmean:,.3f} Hz")
    print(f"  min/max: {fmin:,.3f} / {fmax:,.3f} Hz")
    print(f"  p-p drift: {fpp:,.3f} Hz")
    print(f"  approx drift ppm (ref={ref:,.3f} Hz): {ppm:.3f} ppm")

    return (fmean, fmin, fmax, fpp, ppm)


def main():
    ap = argparse.ArgumentParser(description="Robust tone drift estimator for GNU Radio complex .cfile IQ.")
    ap.add_argument("files", nargs="+", help="Path(s) to .cfile complex64 IQ recordings.")
    ap.add_argument("--fs", type=parsenum, required=True, help="Sample rate in Hz (e.g. 2e6).")
    ap.add_argument(
        "--target-hz",
        type=parsenum,
        required=True,
        help="Expected baseband tone frequency in Hz (e.g. 1e6 if you tuned 1 MHz low).",
    )
    ap.add_argument(
        "--search-hz",
        type=parsenum,
        default=20e3,
        help="Search window +/- around target (default 20e3). Keep tight.",
    )
    ap.add_argument("--nfft", type=int, default=65536, help="FFT size (default 65536).")
    ap.add_argument("--block-len", type=int, default=65536, help="Block length in samples (default 65536).")
    ap.add_argument("--hop-len", type=int, default=32768, help="Hop length in samples (default 32768).")
    ap.add_argument("--skip-sec", type=float, default=0.0, help="Skip initial seconds (default 0).")
    ap.add_argument("--max-sec", type=float, default=0.0, help="Only analyze first N seconds (0 = full file).")
    ap.add_argument("--window", choices=["hann", "rect"], default="hann", help="FFT window (default hann).")

    ap.add_argument(
        "--max-jump-hz",
        type=parsenum,
        default=2000.0,
        help="Reject estimates that jump more than this from previous (default 2000 Hz).",
    )
    ap.add_argument(
        "--hold-last",
        action="store_true",
        help="If a block is rejected, repeat last valid estimate instead of NaN.",
    )
    ap.add_argument(
        "--no-track-around-last",
        action="store_true",
        help="Disable additional windowing around last estimate (not recommended).",
    )
    ap.add_argument(
        "--drop-nans",
        action="store_true",
        help="Drop NaN estimates from summary stats (recommended if not using --hold-last).",
    )
    ap.add_argument("--no-plot", action="store_true", help="Do not plot.")

    args = ap.parse_args()

    fs = float(args.fs)
    target_hz = float(args.target_hz)

    if args.hop_len <= 0 or args.block_len <= 0:
        print("block-len and hop-len must be > 0", file=sys.stderr)
        return 2
    if args.hop_len > args.block_len:
        print("hop-len must be <= block-len", file=sys.stderr)
        return 2

    results = []

    for path in args.files:
        label = os.path.basename(path)
        t, f = estimate_tone_freqs(
            path=path,
            fs=fs,
            target_hz=target_hz,
            search_hz=float(args.search_hz),
            nfft=int(args.nfft),
            block_len=int(args.block_len),
            hop_len=int(args.hop_len),
            skip_seconds=float(args.skip_sec),
            max_seconds=float(args.max_sec),
            window=args.window,
            max_jump_hz=float(args.max_jump_hz),
            hold_last=bool(args.hold_last),
            track_around_last=not bool(args.no_track_around_last),
        )
        summarize(f, label, fs, target_hz, drop_nans=bool(args.drop_nans))
        results.append((label, t, f))

    if len(results) >= 2:
        print("\n[Inter-file mean offsets (Hz)]")
        base_label, base_t, base_f = results[0]
        for label, t, f in results[1:]:
            n = min(base_f.size, f.size)
            if n == 0:
                print(f"  {label} vs {base_label}: (no data)")
                continue
            a = base_f[:n]
            b = f[:n]
            mask = np.isfinite(a) & np.isfinite(b)
            if not np.any(mask):
                print(f"  {label} - {base_label}: (no finite overlap)")
                continue
            diff = b[mask] - a[mask]
            print(f"  {label} - {base_label}: mean {np.mean(diff):+.3f} Hz, p-p {np.ptp(diff):.3f} Hz")

    if not args.no_plot:
        try:
            import matplotlib.pyplot as plt

            for label, t, f in results:
                if f.size == 0:
                    continue
                mask = np.isfinite(f)
                if not np.any(mask):
                    continue
                ff = f[mask]
                tt = t[mask]
                plt.figure()
                plt.plot(tt, ff - np.mean(ff))
                plt.xlabel("Time (s)")
                plt.ylabel("Frequency (Hz) relative to mean")
                plt.title(f"Drift: {label}")
                plt.grid(True)
            plt.show()
        except Exception as e:
            print(f"[WARN] Plot failed: {e}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())