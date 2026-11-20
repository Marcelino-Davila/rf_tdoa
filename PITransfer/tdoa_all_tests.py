#!/usr/bin/env python3
# tdoa_all_tests.py
# ------------------------------------------------------------
# Plug-and-play TDOA-ish correlator for:
#
#   G:\GITFINAL\capture\node0_00000.iq
#   G:\GITFINAL\capture\node1_00000.iq
#   G:\GITFINAL\capture\node2_00000.iq
#
#   G:\GITFINAL\capture\node0_00001.iq
#   G:\GITFINAL\capture\node1_00001.iq
#   G:\GITFINAL\capture\node2_00001.iq
#
# etc.
#
# Each suffix group is treated as a separate test.
#
# Default assumptions:
#   HackRF raw IQ: signed int8 interleaved I,Q,I,Q...
#   Sample rate: 20 MS/s
#   Triangle: equilateral, 14 ft side length
# ------------------------------------------------------------

import os
import re
import math
import argparse
import numpy as np

# =========================
# EDIT HERE IF NEEDED
# =========================
CAPTURE_DIR = r"G:\GITFINAL\capture"

FS = 20_000_000.0        # sample rate
FMT = "ci8"              # ci8 = HackRF raw int8 IQ
SIDE_FT = 14.0           # equilateral triangle side length

NSAMPS = 2_000_000       # samples per test chunk
START = 0                # start sample inside each file
SEARCH_SAMP = 64         # +/- lag search window

# Use more than 1 chunk per test if wanted.
# Example: CHUNKS_PER_TEST = 5
CHUNKS_PER_TEST = 1
CHUNK_STEP = 20_000_000  # sample jump between chunks

C = 299_792_458.0
FT_TO_M = 0.3048


def next_pow2(n):
    return 1 << (n - 1).bit_length()


def file_samples(path, fmt):
    size = os.path.getsize(path)

    if fmt == "ci8":
        return size // 2       # int8 I + int8 Q
    if fmt == "cs16":
        return size // 4       # int16 I + int16 Q
    if fmt == "c64":
        return size // 8       # complex64

    raise ValueError(f"bad fmt: {fmt}")


def load_iq(path, fmt, start, nsamps):
    total = file_samples(path, fmt)

    if start >= total:
        raise ValueError(f"start {start:,} past EOF for {path}")

    nsamps = min(nsamps, total - start)

    if fmt == "ci8":
        raw = np.memmap(
            path,
            dtype=np.int8,
            mode="r",
            offset=start * 2,
            shape=(nsamps * 2,),
        )
        raw = np.asarray(raw, dtype=np.float32).reshape(-1, 2)
        x = (raw[:, 0] + 1j * raw[:, 1]).astype(np.complex64) / 127.0

    elif fmt == "cs16":
        raw = np.memmap(
            path,
            dtype=np.int16,
            mode="r",
            offset=start * 4,
            shape=(nsamps * 2,),
        )
        raw = np.asarray(raw, dtype=np.float32).reshape(-1, 2)
        x = (raw[:, 0] + 1j * raw[:, 1]).astype(np.complex64) / 32767.0

    elif fmt == "c64":
        x = np.memmap(
            path,
            dtype=np.complex64,
            mode="r",
            offset=start * 8,
            shape=(nsamps,),
        )
        x = np.asarray(x, dtype=np.complex64)

    else:
        raise ValueError(f"bad fmt: {fmt}")

    # DC removal + RMS normalize
    x = x - np.mean(x)
    rms = np.sqrt(np.mean(np.abs(x) ** 2))
    if rms > 1e-12:
        x = x / rms

    return x


def gcc_phat_delay(a, b, fs, max_lag):
    """
    Delay of b relative to a.
    Positive lag means b appears later than a.
    Negative lag means b appears earlier than a.
    """

    nfft = next_pow2(len(a) + len(b) - 1)

    A = np.fft.fft(a, nfft)
    B = np.fft.fft(b, nfft)

    R = B * np.conj(A)
    R = R / (np.abs(R) + 1e-12)

    cc = np.fft.ifft(R)
    mag = np.abs(cc)

    lags = np.arange(-max_lag, max_lag + 1)
    vals = mag[lags % nfft]

    k = int(np.argmax(vals))
    lag = float(lags[k])

    # fractional peak interpolation
    frac = 0.0
    if 0 < k < len(vals) - 1:
        y0 = vals[k - 1]
        y1 = vals[k]
        y2 = vals[k + 1]
        denom = y0 - 2.0 * y1 + y2

        if abs(denom) > 1e-20:
            frac = 0.5 * (y0 - y2) / denom
            frac = float(np.clip(frac, -0.5, 0.5))

    lag_frac = lag + frac
    delay_s = lag_frac / fs

    peak = vals[k]
    floor = np.median(vals) + 1e-12
    quality = float(peak / floor)

    return lag_frac, delay_s, quality


def farfield_bearing(t10, t20, side_ft):
    """
    t10 = node1 - node0 delay
    t20 = node2 - node0 delay

    Receiver layout:
      node0 = (0, 0)
      node1 = (side, 0)
      node2 = (side/2, side*sqrt(3)/2)

    Bearing reference:
      0 degrees points from node0 toward node1.
    """

    side_m = side_ft * FT_TO_M

    r0 = np.array([0.0, 0.0])
    r1 = np.array([side_m, 0.0])
    r2 = np.array([side_m / 2.0, side_m * np.sqrt(3.0) / 2.0])

    A = np.vstack([r1 - r0, r2 - r0])
    b = -C * np.array([t10, t20])

    u, *_ = np.linalg.lstsq(A, b, rcond=None)
    norm = float(np.linalg.norm(u))

    if norm > 1e-12:
        u = u / norm

    bearing = math.degrees(math.atan2(u[1], u[0]))
    return bearing, norm


def find_tests(capture_dir):
    pat = re.compile(r"node([012])_(\d+)\.iq$", re.IGNORECASE)

    groups = {}

    for name in os.listdir(capture_dir):
        m = pat.match(name)
        if not m:
            continue

        node = int(m.group(1))
        idx = m.group(2)

        if idx not in groups:
            groups[idx] = {}

        groups[idx][node] = os.path.join(capture_dir, name)

    complete = []

    for idx in sorted(groups.keys()):
        g = groups[idx]
        if 0 in g and 1 in g and 2 in g:
            complete.append((idx, g[0], g[1], g[2]))

    return complete


def analyze_one(idx, rx0, rx1, rx2, start, nsamps, fs, fmt, side_ft, search_samp):
    side_m = side_ft * FT_TO_M
    max_phys_s = side_m / C
    max_phys_samples = max_phys_s * fs

    x0 = load_iq(rx0, fmt, start, nsamps)
    x1 = load_iq(rx1, fmt, start, nsamps)
    x2 = load_iq(rx2, fmt, start, nsamps)

    n = min(len(x0), len(x1), len(x2))
    x0 = x0[:n]
    x1 = x1[:n]
    x2 = x2[:n]

    lag10, t10, q10 = gcc_phat_delay(x0, x1, fs, search_samp)
    lag20, t20, q20 = gcc_phat_delay(x0, x2, fs, search_samp)
    lag21, t21, q21 = gcc_phat_delay(x1, x2, fs, search_samp)

    bearing, norm = farfield_bearing(t10, t20, side_ft)

    print()
    print("======================================================================")
    print(f"TEST {idx}     start={start:,}     nsamps={n:,}")
    print("======================================================================")
    print(f"node0: {os.path.basename(rx0)}")
    print(f"node1: {os.path.basename(rx1)}")
    print(f"node2: {os.path.basename(rx2)}")
    print("----------------------------------------------------------------------")
    print(f"Physical max delay for {side_ft:.2f} ft spacing:")
    print(f"  +/- {max_phys_s * 1e9:.3f} ns")
    print(f"  +/- {max_phys_samples:.4f} samples at {fs:,.0f} S/s")
    print("----------------------------------------------------------------------")

    rows = [
        ("node1 - node0", lag10, t10, q10),
        ("node2 - node0", lag20, t20, q20),
        ("node2 - node1", lag21, t21, q21),
    ]

    any_bad = False

    for label, lag, t, q in rows:
        ns = t * 1e9
        feet = t * C / FT_TO_M
        physical = abs(t) <= max_phys_s * 1.25

        if not physical:
            any_bad = True

        print(
            f"{label:14s}: "
            f"{lag:+9.4f} samples   "
            f"{ns:+10.3f} ns   "
            f"{feet:+10.3f} ft equiv   "
            f"peak/median={q:8.1f}   "
            f"{'OK-ish' if physical else 'TOO BIG'}"
        )

    print("----------------------------------------------------------------------")
    print(f"Far-field bearing: {bearing:+.2f} deg")
    print("Bearing reference: 0 deg = from node0 toward node1")
    print(f"Consistency norm:  {norm:.3f}")

    if norm > 1.25:
        print("WARNING: norm > 1 means the TDOA pair is physically inconsistent/noisy.")

    if any_bad:
        print()
        print("WARNING:")
        print("  One or more delays are larger than physically possible.")
        print("  For a 14 ft triangle at 20 MS/s, real propagation TDOA is sub-sample.")
        print("  Big lags usually mean stream start offset, buffering offset,")
        print("  repeated waveform ambiguity, or not the same RF event.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=CAPTURE_DIR)
    ap.add_argument("--fs", type=float, default=FS)
    ap.add_argument("--fmt", choices=["ci8", "cs16", "c64"], default=FMT)
    ap.add_argument("--side-ft", type=float, default=SIDE_FT)
    ap.add_argument("--start", type=int, default=START)
    ap.add_argument("--nsamps", type=int, default=NSAMPS)
    ap.add_argument("--search-samp", type=int, default=SEARCH_SAMP)
    ap.add_argument("--chunks", type=int, default=CHUNKS_PER_TEST)
    ap.add_argument("--chunk-step", type=int, default=CHUNK_STEP)
    args = ap.parse_args()

    tests = find_tests(args.dir)

    print()
    print("======================================================================")
    print("3-NODE IQ TDOA ALL TESTS")
    print("======================================================================")
    print(f"Capture dir:    {args.dir}")
    print(f"Format:         {args.fmt}")
    print(f"Sample rate:    {args.fs:,.0f} S/s")
    print(f"Side length:    {args.side_ft:.3f} ft")
    print(f"Chunk samples:  {args.nsamps:,}")
    print(f"Search window:  +/- {args.search_samp} samples")
    print("======================================================================")

    if not tests:
        print("No complete node0/node1/node2 test groups found.")
        return

    print()
    print("Found complete tests:")
    for idx, _, _, _ in tests:
        print(f"  test {idx}")

    for idx, rx0, rx1, rx2 in tests:
        for k in range(args.chunks):
            start = args.start + k * args.chunk_step

            try:
                analyze_one(
                    idx=idx,
                    rx0=rx0,
                    rx1=rx1,
                    rx2=rx2,
                    start=start,
                    nsamps=args.nsamps,
                    fs=args.fs,
                    fmt=args.fmt,
                    side_ft=args.side_ft,
                    search_samp=args.search_samp,
                )

            except Exception as e:
                print()
                print("======================================================================")
                print(f"TEST {idx} failed at start={start:,}")
                print(f"Reason: {e}")
                print("======================================================================")
                break

    print()
    print("Done.")


if __name__ == "__main__":
    main()