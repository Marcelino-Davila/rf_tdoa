#!/usr/bin/env python3
"""
TDOA solver for the 3-node HackRF array  (v2 - counter-aligned).

KEY CHANGE FROM v1: markers are matched across nodes by their PPS COUNTER
(the 4 big-endian bytes after each MAGIC), not by index. Each counter ticks
+1 per PPS pulse, so if a dropped chunk eats a marker in one node, the counter
reveals the gap and we still line up the SAME physical pulse across nodes.
v1 matched by index, which silently misaligned everything after the first
dropped marker -> that produced the ~500-sample lags and huge TDOA variance.

Pipeline:
  1. Find markers (position + counter) in all three files.
  2. Build correspondence by counter: keep only pulses present in all three.
  3. Measure each node's sample rate from marker spacing.
  4. Barker matched filter at the RX rate.
  5. At each common pulse: window right after the marker (same instant in all
     nodes), matched-filter, envelope, cross-correlate node pairs -> per-pulse
     TDOA. Prints per-anchor lags so consistency is visible.
  6. Robust median TDOA + spread.
  7. Rough position / bearing for the 12 ft equilateral array.

Run:  & "C:\\Users\\Madness\\radioconda\\python.exe" tdoa_solve.py <seg> 2>&1
"""

import numpy as np
import os
import sys

# ============================ EDIT FOR YOUR RUN =============================
FILES = {
    0: r"G:\GITFINAL\capture\node0_00000.iq",
    1: r"G:\GITFINAL\capture\node1_00000.iq",
    2: r"G:\GITFINAL\capture\node2_00000.iq",
}
SAMPLE_RATE = 20_000_000
CHIP_RATE   = 1_000_000          # Barker chip rate transmitted (--chip-rate)
PRI_US      = 50.0               # beacon PRI (n210_barker_tx.py default)
WINDOW_SAMPLES = 8000            # window after each marker (8 PRIs)
MAX_ANCHORS = 120                # how many common pulses to use

SIDE_FT = 12.0
NODE_POS = {
    0: (0.0, 0.0),
    1: (SIDE_FT, 0.0),
    2: (SIDE_FT / 2.0, SIDE_FT * np.sqrt(3) / 2.0),
}
C_FT_PER_S = 299_792_458.0 * 3.280839895
# ===========================================================================

MAGIC = bytes([0x7F, 0x80, 0x7F, 0x80,
               0x00, 0xFF, 0x00, 0xFF,
               0x55, 0xAA, 0x55, 0xAA])
MARKER_BYTES = 16
BYTES_PER_SAMPLE = 2


def read_iq(path, start_sample, num_samples):
    raw = np.fromfile(path, dtype=np.int8,
                      count=num_samples * 2, offset=start_sample * 2)
    if len(raw) < 2:
        return np.zeros(0, dtype=np.complex64)
    if len(raw) % 2:
        raw = raw[:-1]
    return raw[0::2].astype(np.float32) + 1j * raw[1::2].astype(np.float32)


def find_markers(path):
    """Return list of (sample_pos, counter) using big-endian counter bytes."""
    out = []
    chunk = 16 << 20
    tail = b""
    base = 0
    with open(path, "rb") as f:
        while True:
            buf = f.read(chunk)
            if not buf:
                break
            hay = tail + buf
            start = 0
            while True:
                idx = hay.find(MAGIC, start)
                if idx == -1:
                    break
                bytepos = base - len(tail) + idx
                if bytepos >= 0:
                    cb = hay[idx + len(MAGIC): idx + len(MAGIC) + 4]
                    ctr = int.from_bytes(cb, "big") if len(cb) == 4 else -1
                    out.append((bytepos // BYTES_PER_SAMPLE, ctr))
                start = idx + 1
            tail = hay[-(len(MAGIC) - 1):]
            base += len(buf)
    return out


def barker_matched_filter(chip_rate, sample_rate):
    b = np.array([1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1], dtype=np.float32)
    spc = max(1, int(round(sample_rate / chip_rate)))
    return np.repeat(b, spc).astype(np.complex64), spc


def fft_correlate(a, b):
    n = len(a) + len(b) - 1
    nfft = 1 << (n - 1).bit_length()
    xc = np.fft.ifft(np.fft.fft(a, nfft) * np.conj(np.fft.fft(b, nfft)))
    return np.concatenate((xc[-(len(b) - 1):], xc[:len(a)]))


def matched_envelope(sig, mf):
    return np.abs(fft_correlate(sig.astype(np.complex64), mf))


def parabolic_offset(y, i):
    if i <= 0 or i >= len(y) - 1:
        return 0.0
    a, b, c = y[i - 1], y[i], y[i + 1]
    d = a - 2 * b + c
    return 0.0 if d == 0 else 0.5 * (a - c) / d


def xcorr_lag(env_a, env_b, max_lag):
    xc = np.abs(fft_correlate(env_a.astype(np.complex64), env_b.astype(np.complex64)))
    zero = len(env_b) - 1
    lo, hi = max(0, zero - max_lag), min(len(xc), zero + max_lag + 1)
    seg = xc[lo:hi]
    pk = int(np.argmax(seg)) + lo
    lag = (pk - zero) + parabolic_offset(xc, pk)
    mask = np.ones(len(xc), bool)
    mask[max(0, pk - 4): pk + 5] = False
    sl = xc[mask].max() if mask.any() else 0.0
    return lag, xc[pk], (xc[pk] / sl if sl > 0 else np.inf)


def build_correspondence(markers):
    """Match markers across nodes by relative pulse number (counter - counter[0])."""
    rel, idx = {}, {}
    for n, m in markers.items():
        if not m:
            return []
        c0 = m[0][1]
        rel[n] = [c - c0 for (_, c) in m]
        idx[n] = {r: k for k, r in enumerate(rel[n])}
    common = set(rel[0]) & set(rel[1]) & set(rel[2])
    anchors = []
    for r in sorted(common):
        if r <= 0:
            continue
        trip = {n: markers[n][idx[n][r]][0] for n in markers}
        anchors.append((r, trip))
    return anchors


def solve_position(tdoa01, tdoa02, P, c, bound=500.0):
    rd01, rd02 = c * tdoa01, c * tdoa02

    def resid(xy):
        r0 = np.hypot(xy[0] - P[0][0], xy[1] - P[0][1])
        r1 = np.hypot(xy[0] - P[1][0], xy[1] - P[1][1])
        r2 = np.hypot(xy[0] - P[2][0], xy[1] - P[2][1])
        return np.array([(r0 - r1) - rd01, (r0 - r2) - rd02])

    g = np.linspace(-bound, bound, 201)
    best = None
    for x in g:
        for y in g:
            e = resid((x, y)); c2 = float(e @ e)
            if best is None or c2 < best[0]:
                best = (c2, x, y)
    xy = np.array([best[1], best[2]], float); bc = best[0]; lam = 1e-2
    for _ in range(100):
        e = resid(xy); eps = 1e-3; J = np.zeros((2, 2))
        for k in range(2):
            d = np.zeros(2); d[k] = eps
            J[:, k] = (resid(xy + d) - e) / eps
        try:
            step = np.linalg.solve(J.T @ J + lam * np.eye(2), -(J.T @ e))
        except np.linalg.LinAlgError:
            break
        nxy = np.clip(xy + step, -2 * bound, 2 * bound)
        nc = float(resid(nxy) @ resid(nxy))
        if nc < bc:
            xy, bc = nxy, nc; lam = max(lam * 0.5, 1e-7)
        else:
            lam = min(lam * 2, 1e4)
        if np.linalg.norm(step) < 1e-4:
            break
    return xy, float(np.sqrt(bc)), best


def bearing_from_centroid(xy, P):
    cx = np.mean([P[i][0] for i in P]); cy = np.mean([P[i][1] for i in P])
    return (np.degrees(np.arctan2(xy[1] - cy, xy[0] - cx)) + 360) % 360, \
        np.hypot(xy[0] - cx, xy[1] - cy)


def robust(v):
    v = np.array(v, float)
    if len(v) == 0:
        return 0.0, 0.0, 0
    med = np.median(v); mad = np.median(np.abs(v - med)) + 1e-12
    keep = v[np.abs(v - med) < 4 * mad]
    return (keep.mean() if len(keep) else med), (keep.std() if len(keep) else 0.0), len(keep)


def resolve_files():
    args = sys.argv[1:]
    if len(args) == 3:
        return {i: args[i] for i in range(3)}
    if len(args) == 1 and args[0].isdigit():
        seg = f"_{int(args[0]):05d}"
        return {n: p.replace("_00000", seg) for n, p in FILES.items()}
    return dict(FILES)


def main():
    print("=" * 70)
    print("TDOA SOLVE v2  (counter-aligned)")
    print("=" * 70)
    files = resolve_files()
    for n, p in files.items():
        print(f"    node {n}: {os.path.basename(p)}")
        if not os.path.exists(p):
            print(f"ERROR: not found: {p}"); sys.exit(1)

    print("\n[1] finding markers (position + PPS counter)...")
    markers = {n: find_markers(p) for n, p in files.items()}
    for n in files:
        m = markers[n]
        cok = "counters +1 (clean)" if m and all(
            m[k + 1][1] - m[k][1] == 1 for k in range(min(len(m) - 1, 20))) else \
            "counter gaps (dropped markers)"
        print(f"    node {n}: {len(m)} markers; {cok}")

    print("\n[2] correspondence by counter (same physical pulse in all 3)...")
    anchors = build_correspondence(markers)
    print(f"    matched pulses present in all three nodes: {len(anchors)}")
    if len(anchors) < 5:
        print("    Too few common pulses -> files may not overlap, or bad counters.")
        sys.exit(1)
    dropped = {n: (markers[n][-1][1] - markers[n][0][1] + 1) - len(markers[n]) for n in files}
    print(f"    inferred dropped markers per node: {dropped}")

    print("\n[3] per-node sample rate...")
    rate = {}
    for n in files:
        pos = np.array([p for p, _ in markers[n]])
        gaps = np.diff(pos)
        gaps = gaps[(gaps > 0.8 * SAMPLE_RATE) & (gaps < 1.2 * SAMPLE_RATE)]
        rate[n] = float(np.median(gaps)) if len(gaps) else SAMPLE_RATE
        print(f"    node {n}: {rate[n]:,.0f} samp/s ({(rate[n]-SAMPLE_RATE)/SAMPLE_RATE*1e6:+.0f} ppm)")

    mf, spc = barker_matched_filter(CHIP_RATE, SAMPLE_RATE)
    pri = int(round(PRI_US * 1e-6 * SAMPLE_RATE))
    print(f"\n[4] Barker MF: {len(mf)} samples ({spc}/chip); PRI ~{pri} samples")

    use = anchors[2:2 + MAX_ANCHORS]
    print(f"\n[5] correlating {len(use)} counter-matched windows "
          f"(lag search +/-{pri//2} samples)...")
    snr = {0: [], 1: [], 2: []}
    lags01, lags02 = [], []
    shown = 0
    for r, trip in use:
        envs, ok = {}, True
        for n in files:
            sig = read_iq(files[n], int(trip[n]) + MARKER_BYTES // 2, WINDOW_SAMPLES)
            if len(sig) < WINDOW_SAMPLES // 2:
                ok = False; break
            envs[n] = matched_envelope(sig, mf)
            snr[n].append(envs[n].max() / (np.median(envs[n]) + 1e-9))
        if not ok:
            continue
        l01, _, _ = xcorr_lag(envs[0], envs[1], pri // 2)
        l02, _, _ = xcorr_lag(envs[0], envs[2], pri // 2)
        lags01.append(l01); lags02.append(l02)
        if shown < 12:
            print(f"    pulse {r:4d}: SNR n0={snr[0][-1]:5.1f} n1={snr[1][-1]:5.1f} "
                  f"n2={snr[2][-1]:5.1f} | lag01={l01:+8.2f} lag02={l02:+8.2f} samp")
            shown += 1

    if not lags01:
        print("    No usable windows."); sys.exit(1)

    med_snr = {n: np.median(snr[n]) for n in files}
    print(f"\n    median beacon SNR: n0={med_snr[0]:.1f} n1={med_snr[1]:.1f} n2={med_snr[2]:.1f}")
    if min(med_snr.values()) < 4:
        print("    ** WEAK beacon. Check capture freq == beacon freq and CHIP_RATE.")

    print("\n[6] TDOA (robust median over matched pulses)...")
    m01, s01, k01 = robust(np.array(lags01) / SAMPLE_RATE)
    m02, s02, k02 = robust(np.array(lags02) / SAMPLE_RATE)
    print(f"    lag01 median {np.median(lags01):+.2f} samp, "
          f"lag02 median {np.median(lags02):+.2f} samp")
    print(f"    TDOA(0-1) = {m01*1e9:+8.1f} ns  (std {s01*1e9:6.1f} ns / {k01} pulses, "
          f"{m01*C_FT_PER_S:+.1f} ft)")
    print(f"    TDOA(0-2) = {m02*1e9:+8.1f} ns  (std {s02*1e9:6.1f} ns / {k02} pulses, "
          f"{m02*C_FT_PER_S:+.1f} ft)")
    print(f"    (1 sample = {1e9/SAMPLE_RATE:.0f} ns = {C_FT_PER_S/SAMPLE_RATE:.1f} ft)")
    if s01 * 1e9 < 200 and s02 * 1e9 < 200:
        print("    ** consistent TDOA (low std) -- real measurement.")
    else:
        print("    ** still high variance -> TDOA may be below resolution (12 ft array,")
        print("       50 ns/sample) or residual misalignment. See per-pulse lags above.")

    print("\n[7] position (12 ft equilateral)...")
    xy, res, _ = solve_position(m01, m02, NODE_POS, C_FT_PER_S)
    brg, rng = bearing_from_centroid(xy, NODE_POS)
    print(f"    emitter (assumed labeling): x={xy[0]:.1f} ft, y={xy[1]:.1f} ft")
    print(f"    bearing {brg:.0f} deg, range {rng:.0f} ft, residual {res:.2f} ft")
    print("    (bearing is the trustworthy output; range weak for this geometry;")
    print("     absolute bearing depends on which file = which physical corner)")
    print("\ndone.")


if __name__ == "__main__":
    main()