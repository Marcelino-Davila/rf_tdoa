#!/usr/bin/env python3
"""
build_filters.py
================
All-in-one: takes noise-only .cfile recordings for rx0, rx1, rx2 and produces:
  1. Noise covariance matrices  -> Rw_rx0.npy, Rw_rx1.npy, Rw_rx2.npy
  2. Pulse template             -> pulse_template_s.npy
  3. Optimal filters            -> optimal_filter_g_rx0.npy, etc.

Usage:
    python build_filters.py --noise-dir G:/GITFINAL/rf_tdoa/DATA/NOISE_ONLY --out-dir G:/GITFINAL/rf_tdoa/CODE/MATCHED_FILTER

    # Custom TX params if they differ from defaults:
    python build_filters.py \
        --noise-dir G:/GITFINAL/rf_tdoa/DATA/NOISE_ONLY \
        --out-dir   G:/GITFINAL/rf_tdoa/CODE/MATCHED_FILTER \
        --N         2000 \
        --amp       0.25 \
        --omega     0.31416 \
        --symbols   "1,1,1,1,1,-1,-1,1,1,-1,1,-1,1"

NOTE: --noise-dir should contain rx0.cfile, rx1.cfile, rx2.cfile recorded
      with the transmitter OFF. This captures your true noise floor with
      the GPSDO and attenuators in place.
"""

import argparse
import os
import sys
import numpy as np
import scipy.linalg
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import upfirdn


# ── Defaults matching your GRC flowgraph ──────────────────────────────────────
DEFAULT_SYMBOLS  = [1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1]
DEFAULT_AMP      = 0.25
DEFAULT_OMEGA    = 0.31416   # rad/sample (50kHz tone at 1MHz fs)
DEFAULT_INTERP   = 4
DEFAULT_RRC_BETA = 0.35
DEFAULT_RRC_SPAN = 11
DEFAULT_RRC_GAIN = 1.0
DEFAULT_N        = 2000


# ═════════════════════════════════════════════════════════════════════════════
# Step 1: RRC taps
# ═════════════════════════════════════════════════════════════════════════════

def make_rrc_taps(beta: float, sps: int, span: int, gain: float = 1.0) -> np.ndarray:
    N = span * sps
    t = np.arange(-N / 2, N / 2 + 1, dtype=np.float64) / float(sps)
    taps = np.zeros_like(t)
    for i, ti in enumerate(t):
        if abs(ti) < 1e-12:
            taps[i] = 1.0 + beta * (4.0 / np.pi - 1.0)
        elif abs(abs(4.0 * beta * ti) - 1.0) < 1e-12:
            a = (1.0 + 2.0 / np.pi) * np.sin(np.pi / (4.0 * beta))
            b = (1.0 - 2.0 / np.pi) * np.cos(np.pi / (4.0 * beta))
            taps[i] = (beta / np.sqrt(2.0)) * (a + b)
        else:
            num = (np.sin(np.pi * ti * (1.0 - beta)) +
                   4.0 * beta * ti * np.cos(np.pi * ti * (1.0 + beta)))
            den = np.pi * ti * (1.0 - (4.0 * beta * ti) ** 2)
            taps[i] = num / den
    taps /= np.sqrt(np.sum(taps ** 2))
    return taps * gain


# ═════════════════════════════════════════════════════════════════════════════
# Step 2: Pulse template
# ═════════════════════════════════════════════════════════════════════════════

def build_pulse_template(
    N: int,
    symbols: list,
    interp: int,
    amp: float,
    omega: float,
    rrc_beta: float,
    rrc_span: int,
    rrc_gain: float,
) -> np.ndarray:
    taps = make_rrc_taps(rrc_beta, interp, rrc_span, rrc_gain)
    a = np.asarray(symbols, dtype=np.float64)

    # tile symbols until we have enough samples after filtering
    reps = 1
    while interp * reps * len(a) < N + taps.size:
        reps *= 2
    a_tiled = np.tile(a, reps)

    x = upfirdn(taps, a_tiled, up=interp).astype(np.float64)
    x *= amp

    # apply rotator
    n = np.arange(x.size, dtype=np.float64)
    s = (x * np.exp(1j * omega * n)).astype(np.complex64)

    # trim to N
    if s.size >= N:
        s = s[:N]
    else:
        s = np.pad(s, (0, N - s.size))

    return s.astype(np.complex64)


# ═════════════════════════════════════════════════════════════════════════════
# Step 3: Noise covariance
# ═════════════════════════════════════════════════════════════════════════════

def compute_noise_covariance(
    cfile_path: str,
    N: int,
    downsample_factor: int = 50,
    chunk_size: int = 1_000_000,
    verbose: bool = True,
) -> np.ndarray:
    if not os.path.isfile(cfile_path):
        raise FileNotFoundError(f"cfile not found: {cfile_path}")

    Rw_sum = np.zeros((N, N), dtype=np.complex64)
    count = 0
    snap_index = 0
    carry = np.zeros(0, dtype=np.complex64)
    samples_read = 0

    with open(cfile_path, "rb") as fh:
        while True:
            raw = fh.read(chunk_size * 8)
            if not raw:
                break
            x = np.frombuffer(raw, dtype=np.complex64).copy()
            samples_read += x.size

            # append carry
            buf = np.concatenate((carry, x)) if carry.size else x

            if buf.size < N:
                carry = buf
                continue

            snaps = sliding_window_view(buf, window_shape=N)  # (M, N)
            M = snaps.shape[0]
            global_idx = snap_index + np.arange(M)
            keep = (global_idx % downsample_factor) == 0
            snaps_ds = snaps[keep]

            if snaps_ds.shape[0] > 0:
                Rw_sum += snaps_ds.conj().T @ snaps_ds
                count += snaps_ds.shape[0]

            snap_index += M
            carry = buf[-(N - 1):] if N > 1 else np.zeros(0, dtype=np.complex64)

    if count == 0:
        raise RuntimeError(f"No snapshots accumulated from {cfile_path}. File too short?")

    if verbose:
        print(f"    samples read : {samples_read:,}")
        print(f"    snapshots    : {count:,}")
        avg_power = float(np.real(np.trace(Rw_sum / count)) / N)
        print(f"    avg noise pwr: {avg_power:.6e}")

    return (Rw_sum / count).astype(np.complex64)


# ═════════════════════════════════════════════════════════════════════════════
# Step 4: Optimal filter  g = Rw^{-1} s
# ═════════════════════════════════════════════════════════════════════════════

def design_optimal_filter(
    Rw: np.ndarray,
    s: np.ndarray,
    verbose: bool = True,
) -> np.ndarray:
    g = scipy.linalg.solve(Rw, s, assume_a="her")
    snr_lin = float(np.real(np.vdot(s, g)))
    snr_db  = 10.0 * np.log10(max(snr_lin, 1e-30))
    if verbose:
        print(f"    SNR_max      : {snr_db:.2f} dB  (linear={snr_lin:.4e})")
    return g.astype(np.complex64)


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Build noise covariance matrices and optimal filters from noise-only cfiles."
    )
    ap.add_argument("--noise-dir", required=True,
                    help="Folder containing rx0.cfile, rx1.cfile, rx2.cfile (transmitter OFF)")
    ap.add_argument("--out-dir",   required=True,
                    help="Output folder — subfolders NOISE_COVARIANCE and OPTIMAL_FILTERS will be created")
    ap.add_argument("--N",         type=int,   default=DEFAULT_N,
                    help=f"Filter / template length in samples (default {DEFAULT_N})")
    ap.add_argument("--symbols",   default=",".join(str(x) for x in DEFAULT_SYMBOLS),
                    help="Comma-separated TX symbol sequence")
    ap.add_argument("--amp",       type=float, default=DEFAULT_AMP,
                    help=f"TX amplitude (default {DEFAULT_AMP})")
    ap.add_argument("--omega",     type=float, default=DEFAULT_OMEGA,
                    help=f"Rotator freq rad/sample (default {DEFAULT_OMEGA})")
    ap.add_argument("--interp",    type=int,   default=DEFAULT_INTERP,
                    help=f"Interpolation factor / sps (default {DEFAULT_INTERP})")
    ap.add_argument("--rrc-beta",  type=float, default=DEFAULT_RRC_BETA,
                    help=f"RRC roll-off beta (default {DEFAULT_RRC_BETA})")
    ap.add_argument("--rrc-span",  type=int,   default=DEFAULT_RRC_SPAN,
                    help=f"RRC span in symbols (default {DEFAULT_RRC_SPAN})")
    ap.add_argument("--downsample",type=int,   default=50,
                    help="Covariance snapshot downsample factor (default 50)")
    ap.add_argument("--receivers", default="rx0,rx1,rx2",
                    help="Comma-separated receiver names (default rx0,rx1,rx2)")
    args = ap.parse_args()

    receivers = [r.strip() for r in args.receivers.split(",")]
    symbols   = [float(x) for x in args.symbols.split(",")]

    cov_dir    = os.path.join(args.out_dir, "NOISE_COVARIANCE")
    filter_dir = os.path.join(args.out_dir, "OPTIMAL_FILTERS")
    os.makedirs(cov_dir,    exist_ok=True)
    os.makedirs(filter_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  build_filters.py")
    print(f"{'='*60}")
    print(f"  noise-dir : {args.noise_dir}")
    print(f"  out-dir   : {args.out_dir}")
    print(f"  N         : {args.N}")
    print(f"  symbols   : {symbols}")
    print(f"  amp       : {args.amp}  omega: {args.omega}  interp: {args.interp}")
    print(f"  rrc beta  : {args.rrc_beta}  span: {args.rrc_span}")
    print(f"  receivers : {receivers}")

    # ── Pulse template ────────────────────────────────────────────────────────
    print(f"\n[1/3] Building pulse template (N={args.N})...")
    s = build_pulse_template(
        N=args.N,
        symbols=symbols,
        interp=args.interp,
        amp=args.amp,
        omega=args.omega,
        rrc_beta=args.rrc_beta,
        rrc_span=args.rrc_span,
        rrc_gain=DEFAULT_RRC_GAIN,
    )
    template_path = os.path.join(args.out_dir, "pulse_template_s.npy")
    np.save(template_path, s)
    print(f"  ✓ Template saved: {template_path}")
    print(f"    mean(|s|^2) = {float(np.mean(np.abs(s)**2)):.6e}")

    # ── Noise covariance + optimal filter per receiver ────────────────────────
    results = []
    for rx in receivers:
        print(f"\n{'='*60}")
        print(f"[2+3/3] Processing {rx}")
        print(f"{'='*60}")

        cfile = os.path.join(args.noise_dir, f"{rx}.cfile")
        if not os.path.isfile(cfile):
            print(f"  ✗ SKIP — file not found: {cfile}")
            continue

        fsize_mb = os.path.getsize(cfile) / 1e6
        print(f"  cfile : {cfile}  ({fsize_mb:.1f} MB)")

        # covariance
        print(f"  Computing noise covariance...")
        try:
            Rw = compute_noise_covariance(
                cfile, args.N,
                downsample_factor=args.downsample,
                verbose=True,
            )
        except Exception as e:
            print(f"  ✗ Covariance failed: {e}")
            continue

        rw_path = os.path.join(cov_dir, f"Rw_{rx}.npy")
        np.save(rw_path, Rw)
        print(f"  ✓ Covariance saved: {rw_path}")

        # optimal filter
        print(f"  Designing optimal filter...")
        try:
            g = design_optimal_filter(Rw, s, verbose=True)
        except Exception as e:
            print(f"  ✗ Filter design failed: {e}")
            continue

        g_path = os.path.join(filter_dir, f"optimal_filter_g_{rx}.npy")
        np.save(g_path, g)
        print(f"  ✓ Filter saved: {g_path}")

        snr_lin = float(np.real(np.vdot(s, g)))
        results.append((rx, snr_lin))

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  DONE")
    print(f"{'='*60}")
    print(f"  Template  : {template_path}")
    print(f"  Cov dir   : {cov_dir}")
    print(f"  Filter dir: {filter_dir}")
    print()
    for rx, snr_lin in results:
        snr_db = 10.0 * np.log10(max(snr_lin, 1e-30))
        print(f"  {rx}  SNR_max = {snr_db:.2f} dB")
    print()
    if len(results) == len(receivers):
        print("  ✓ All filters built successfully. Run diagnostic plots to validate.")
    else:
        missing = len(receivers) - len(results)
        print(f"  ⚠  {missing} receiver(s) failed — check paths above.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()