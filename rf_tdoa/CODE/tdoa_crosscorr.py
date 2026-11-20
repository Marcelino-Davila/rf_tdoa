#!/usr/bin/env python3
"""
tdoa_pipeline.py
================
Two-path TDOA pipeline:

  PATH 1 — Detection:
    raw IQ -> Matched Filter -> peak detection -> "signal present?"

  PATH 2 — TDOA estimation:
    raw IQ -> Bandpass Filter -> GCC-PHAT (complex) -> subsample interpolation -> TDOA

The MF tells you WHEN and WHETHER the signal is present.
The bandpass+crosscorr tells you the precise TIME DIFFERENCE between receivers.

Subsample parabolic interpolation fits a parabola around the GCC-PHAT peak to
find the true maximum between integer samples — giving 10-20x better resolution
than the sample rate alone.

Resolution guide:
   10 MHz sample rate + integer peak    =  ~30 m
   10 MHz sample rate + subsample interp =  ~3-5 m (theoretical)
   20 MHz sample rate + subsample interp =  ~1-3 m (theoretical)

Usage:
    python tdoa_pipeline.py \
        --data-dir   G:/GITFINAL/rf_tdoa/DATA/RAW_BIN \
        --filter-dir G:/GITFINAL/rf_tdoa/CODE/MATCHED_FILTER/OPTIMAL_FILTERS \
        --out-dir    G:/GITFINAL/rf_tdoa/DATA/TDOA_OUT \
        --fs         20e6 \
        --tone-hz    50000 \
        --bp-bw      400000

    # For 10MHz:
        --fs 10e6 --bp-bw 200000
"""

import argparse
import glob
import os
import sys
import numpy as np
from scipy.signal import firwin, lfilter, sosfilt, butter

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    os.system(f"{sys.executable} -m pip install matplotlib --quiet")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

COLORS   = {"rx0": "#00CFFF", "rx1": "#FF6B35", "rx2": "#A8FF3E"}
DARK_BG  = "#0d0d0d"
PANEL_BG = "#141414"


# ── File helpers ───────────────────────────────────────────────────────────────

def find_cfile(base: str, rx: str) -> str:
    if os.path.isfile(base):
        return base
    direct = os.path.join(base, f"{rx}.cfile")
    if os.path.isfile(direct):
        return direct
    pattern = os.path.join(base, "*", f"{rx}.cfile")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No .cfile found for {rx} under {base}")
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return files[0]


def load_cfile(path: str, max_samps: int) -> np.ndarray:
    mm = np.memmap(path, dtype=np.complex64, mode="r")
    data = mm[:min(max_samps, len(mm))].copy()
    del mm
    return data


# ── Bandpass filter ────────────────────────────────────────────────────────────

def bandpass_filter(
    x: np.ndarray,
    fs: float,
    center_hz: float,
    bw_hz: float,
    numtaps: int = 256,
) -> np.ndarray:
    """
    Complex bandpass filter centered at center_hz with bandwidth bw_hz.
    Implemented as a real bandpass FIR then frequency-shifted to center.
    For a complex baseband signal with a tone at center_hz, this isolates
    just that tone and its immediate neighborhood.
    """
    # design a lowpass prototype
    nyq = fs / 2.0
    cutoff = (bw_hz / 2.0) / nyq
    cutoff = np.clip(cutoff, 1e-4, 0.9999)

    taps = firwin(numtaps + 1, cutoff, window="hamming")

    # frequency shift to center_hz (complex mix)
    n = np.arange(len(taps))
    shift = np.exp(1j * 2.0 * np.pi * center_hz / fs * n)
    taps_bp = taps * shift

    # apply filter
    filtered = np.convolve(x.astype(np.complex128), taps_bp, mode="same")
    return filtered.astype(np.complex64)


# ── Matched filter ─────────────────────────────────────────────────────────────

class MatchedFilterOS:
    def __init__(self, h: np.ndarray, fft_size: int = None):
        h = np.asarray(h).astype(np.complex64, copy=False)
        self.h = h
        self.L = int(h.size)
        if fft_size is None:
            n = 1
            target = max(2048, 8 * self.L)
            while n < target:
                n *= 2
            fft_size = n
        self.Nfft = int(fft_size)
        self.B = self.Nfft - self.L + 1
        h_pad = np.zeros(self.Nfft, dtype=np.complex64)
        h_pad[:self.L] = self.h
        self.H = np.fft.fft(h_pad)
        self._carry = np.zeros(self.L - 1, dtype=np.complex64) if self.L > 1 else np.zeros(0, dtype=np.complex64)
        self._inbuf = np.zeros(0, dtype=np.complex64)

    def run(self, x: np.ndarray) -> np.ndarray:
        """Run full array through MF, return magnitude float32."""
        x = np.asarray(x).astype(np.complex64, copy=False)
        self._inbuf = np.concatenate((self._inbuf, x)) if self._inbuf.size else x.copy()
        out_chunks = []
        while self._inbuf.size >= self.B:
            take = self._inbuf[:self.B]
            self._inbuf = self._inbuf[self.B:]
            x_block = np.concatenate((self._carry, take))
            if x_block.size < self.Nfft:
                x_block = np.pad(x_block, (0, self.Nfft - x_block.size))
            X = np.fft.fft(x_block[:self.Nfft])
            y = np.fft.ifft(X * self.H).astype(np.complex64)
            y_valid = y[self.L - 1: self.L - 1 + self.B]
            if self.L > 1:
                self._carry = x_block[-(self.L - 1):]
            out_chunks.append(np.abs(y_valid).astype(np.float32))
        if out_chunks:
            return np.concatenate(out_chunks)
        return np.zeros(0, dtype=np.float32)


# ── Cross-correlation TDOA (complex) ──────────────────────────────────────────

def subsample_parabolic(xcorr: np.ndarray, peak_idx: int) -> float:
    """
    Fit a parabola through the peak and its two neighbours to find the
    true sub-sample maximum.

    Returns fractional offset from peak_idx (between -0.5 and +0.5).
    Falls back to 0.0 if peak is at the edge or parabola is degenerate.
    """
    if peak_idx <= 0 or peak_idx >= len(xcorr) - 1:
        return 0.0
    y0 = float(xcorr[peak_idx - 1])
    y1 = float(xcorr[peak_idx])
    y2 = float(xcorr[peak_idx + 1])
    denom = 2.0 * (2.0 * y1 - y0 - y2)
    if abs(denom) < 1e-30:
        return 0.0
    offset = (y0 - y2) / denom
    # clamp to ±0.5 (should always be within this range for a true peak)
    return float(np.clip(offset, -0.5, 0.5))


def xcorr_tdoa_complex(
    ref: np.ndarray,
    tgt: np.ndarray,
    fs: float,
    max_lag_ms: float = 5.0,
) -> dict:
    """
    TDOA via GCC-PHAT on complex bandpass-filtered IQ with subsample
    parabolic interpolation for sub-sample resolution.

    Integer resolution  : 1/fs  seconds  (e.g. 50ns at 20MHz = 15m)
    Subsample resolution: ~1/fs/10 to /20 (e.g. 3-5m at 20MHz theoretical)
    """
    N = min(len(ref), len(tgt))
    ref = ref[:N].astype(np.complex64)
    tgt = tgt[:N].astype(np.complex64)

    max_lag_samps = int(max_lag_ms * 1e-3 * fs)
    max_lag_samps = min(max_lag_samps, N // 2)

    Nfft = int(2 ** np.ceil(np.log2(2 * N - 1)))

    # GCC-PHAT
    R_ref  = np.fft.fft(ref, n=Nfft)
    R_tgt  = np.fft.fft(tgt, n=Nfft)
    G      = R_ref * np.conj(R_tgt)
    G_phat = G / (np.abs(G) + 1e-10)

    xcorr_full     = np.fft.ifft(G_phat).real
    xcorr_centered = np.fft.fftshift(xcorr_full)

    center = len(xcorr_centered) // 2
    lo = max(0, center - max_lag_samps)
    hi = min(len(xcorr_centered), center + max_lag_samps + 1)

    xcorr_window = xcorr_centered[lo:hi]
    lags         = np.arange(lo - center, hi - center)

    # Integer peak
    peak_idx   = int(np.argmax(xcorr_window))
    peak_val   = float(xcorr_window[peak_idx])
    tdoa_int   = int(lags[peak_idx])

    # Subsample refinement — parabolic interpolation
    subsamp_offset = subsample_parabolic(xcorr_window, peak_idx)
    tdoa_subsamp   = tdoa_int + subsamp_offset   # fractional samples

    # PSLR
    guard = max(5, len(xcorr_window) // 200)
    lo_g  = max(0, peak_idx - guard)
    hi_g  = min(len(xcorr_window), peak_idx + guard)
    sidelobes = np.concatenate([xcorr_window[:lo_g], xcorr_window[hi_g:]])
    pslr_db = 20 * np.log10(
        peak_val / (np.max(np.abs(sidelobes)) + 1e-12)
    ) if sidelobes.size else 0.0

    tdoa_us_int    = tdoa_int    / fs * 1e6
    tdoa_us_subsamp = tdoa_subsamp / fs * 1e6

    return {
        # integer results
        "tdoa_samples":        tdoa_int,
        "tdoa_us":             tdoa_us_int,
        "tdoa_m":              tdoa_int / fs * 3e8,
        # subsample results
        "tdoa_subsamp_samples": tdoa_subsamp,
        "tdoa_us_subsamp":     tdoa_us_subsamp,
        "tdoa_m_subsamp":      tdoa_subsamp / fs * 3e8,
        "subsamp_offset":      subsamp_offset,
        # quality
        "peak_val":            peak_val,
        "pslr_db":             pslr_db,
        "lags":                lags,
        "xcorr":               xcorr_window,
    }


# ── Plotting ───────────────────────────────────────────────────────────────────

def style_ax(ax, title, xlabel, ylabel):
    ax.set_facecolor(PANEL_BG)
    ax.set_title(title, color="white", fontsize=10, pad=6)
    ax.set_xlabel(xlabel, color="#888", fontsize=8)
    ax.set_ylabel(ylabel, color="#888", fontsize=8)
    ax.tick_params(colors="#666", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#333")
    ax.grid(True, color="#2a2a2a", linewidth=0.5, linestyle="--", alpha=0.7)


def make_plots(
    raw_mags:     dict,
    mf_mags:      dict,
    bp_signals:   dict,
    tdoa_results: dict,
    fs:           float,
    out_dir:      str,
    receivers:    list,
    tone_hz:      float,
    bp_bw:        float,
):
    os.makedirs(out_dir, exist_ok=True)
    ref    = receivers[0]
    pairs  = [(ref, rx) for rx in receivers[1:] if rx in bp_signals]

    # ── Plot 1: Raw IQ vs Bandpass overlay ────────────────────────────────────
    print("  [plot 1] Raw IQ vs Bandpass...")
    fig, axes = plt.subplots(len(receivers), 2, figsize=(16, 3 * len(receivers)),
                             facecolor=DARK_BG)
    for i, rx in enumerate(receivers):
        ax_raw = axes[i][0]
        ax_bp  = axes[i][1]
        c = COLORS.get(rx, "w")

        if rx in raw_mags:
            mag = raw_mags[rx]
            step = max(1, len(mag) // 20_000)
            t = np.arange(0, len(mag), step) / fs * 1e3
            ax_raw.set_facecolor(PANEL_BG)
            ax_raw.plot(t, mag[::step], color=c, linewidth=0.5, alpha=0.8)
            style_ax(ax_raw, f"{rx} Raw IQ Magnitude", "Time (ms)", "Amplitude")

        if rx in bp_signals:
            bp = np.abs(bp_signals[rx])
            step = max(1, len(bp) // 20_000)
            t = np.arange(0, len(bp), step) / fs * 1e3
            ax_bp.set_facecolor(PANEL_BG)
            ax_bp.plot(t, bp[::step], color=c, linewidth=0.5, alpha=0.8)
            style_ax(ax_bp,
                     f"{rx} Bandpass Magnitude ({tone_hz/1e3:.0f} kHz ±{bp_bw/2e3:.0f} kHz)",
                     "Time (ms)", "Amplitude")

    fig.tight_layout(pad=1.5)
    p = os.path.join(out_dir, "plot1_raw_vs_bandpass.png")
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"    ✓ {p}")

    # ── Plot 2: MF detection output ───────────────────────────────────────────
    print("  [plot 2] MF detection outputs...")
    fig, axes = plt.subplots(len(receivers), 1, figsize=(14, 3 * len(receivers)),
                             facecolor=DARK_BG, squeeze=False)
    for i, rx in enumerate(receivers):
        ax = axes[i][0]
        ax.set_facecolor(PANEL_BG)
        if rx not in mf_mags:
            continue
        mag = mf_mags[rx]
        step = max(1, len(mag) // 30_000)
        t = np.arange(0, len(mag), step) / fs * 1e3
        c = COLORS.get(rx, "w")
        ax.fill_between(t, mag[::step], alpha=0.15, color=c)
        ax.plot(t, mag[::step], color=c, linewidth=0.6, label=rx)

        # detection threshold at mean + 3*std
        threshold = np.mean(mag) + 3 * np.std(mag)
        ax.axhline(threshold, color="yellow", linewidth=0.8,
                   linestyle="--", alpha=0.7, label=f"threshold")
        detections = np.where(mag > threshold)[0]
        n_det = len(detections)
        style_ax(ax, f"{rx} — MF Detection Output  |  {n_det} samples above threshold",
                 "Time (ms)", "Correlation Magnitude")
        ax.legend(facecolor="#1a1a1a", edgecolor="#333", labelcolor="white", fontsize=8)

    fig.tight_layout(pad=1.5)
    p = os.path.join(out_dir, "plot2_mf_detection.png")
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"    ✓ {p}")

    # ── Plot 3: GCC-PHAT cross-correlations ───────────────────────────────────
    print("  [plot 3] GCC-PHAT cross-correlations...")
    n_pairs = len(pairs)
    fig, axes = plt.subplots(n_pairs, 1,
                             figsize=(14, 4 * n_pairs),
                             facecolor=DARK_BG, squeeze=False)
    for i, (r, t_rx) in enumerate(pairs):
        ax = axes[i][0]
        ax.set_facecolor(PANEL_BG)
        if (r, t_rx) not in tdoa_results:
            continue
        res      = tdoa_results[(r, t_rx)]
        lags_us  = res["lags"] / fs * 1e6
        xcorr    = res["xcorr"]
        peak_i   = np.argmax(xcorr)
        c        = COLORS.get(t_rx, "w")

        ax.fill_between(lags_us, xcorr, alpha=0.2, color=c)
        ax.plot(lags_us, xcorr, color=c, linewidth=0.8)
        ax.axvline(lags_us[peak_i], color="yellow", linewidth=1.2, linestyle="--")
        ax.annotate(
            f"TDOA (integer)   = {res['tdoa_us']:+.3f} µs  ({res['tdoa_m']:+.2f} m)\n"
            f"TDOA (subsample) = {res['tdoa_us_subsamp']:+.4f} µs  ({res['tdoa_m_subsamp']:+.2f} m)\n"
            f"PSLR = {res['pslr_db']:.1f} dB",
            xy=(lags_us[peak_i], xcorr[peak_i]),
            xytext=(lags_us[peak_i] + (lags_us[-1] - lags_us[0]) * 0.03,
                    xcorr[peak_i] * 0.7),
            color="yellow", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#1a1a1a",
                      edgecolor="#555", alpha=0.9),
            arrowprops=dict(arrowstyle="->", color="yellow", lw=0.8),
        )
        style_ax(ax,
                 f"GCC-PHAT: {r} × {t_rx}  |  "
                 f"integer={res['tdoa_us']:+.3f} µs  "
                 f"subsample={res['tdoa_us_subsamp']:+.4f} µs  ({res['tdoa_m_subsamp']:+.2f} m)",
                 "Lag (µs)", "GCC-PHAT")

    fig.tight_layout(pad=1.5)
    p = os.path.join(out_dir, "plot3_gcc_phat.png")
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"    ✓ {p}")

    # ── Plot 4: PSLR summary ──────────────────────────────────────────────────
    print("  [plot 4] PSLR summary...")
    fig, ax = plt.subplots(figsize=(8, 4), facecolor=DARK_BG)
    ax.set_facecolor(PANEL_BG)
    pair_labels = [f"{r}×{t}" for r, t in pairs if (r, t) in tdoa_results]
    pslr_vals   = [tdoa_results[(r, t)]["pslr_db"] for r, t in pairs if (r, t) in tdoa_results]
    bar_colors  = [COLORS.get(t, "w") for _, t in pairs if (_, t) in tdoa_results]
    bars = ax.bar(pair_labels, pslr_vals, color=bar_colors, width=0.4, zorder=3)
    ax.axhline(13.3, color="yellow", linewidth=1.0, linestyle="--",
               alpha=0.7, label="Ideal (13.3 dB)")
    ax.axhline(6.0,  color="orange", linewidth=1.0, linestyle=":",
               alpha=0.7, label="Minimum usable (6 dB)")
    for bar, val in zip(bars, pslr_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.2,
                f"{val:.1f} dB", ha="center", color="white", fontsize=9)
    style_ax(ax, "GCC-PHAT PSLR per Pair", "Pair", "PSLR (dB)")
    ax.legend(facecolor="#1a1a1a", edgecolor="#333", labelcolor="white", fontsize=8)
    ax.set_ylim(0, max(pslr_vals + [14]) * 1.25 if pslr_vals else 20)
    fig.tight_layout()
    p = os.path.join(out_dir, "plot4_pslr.png")
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"    ✓ {p}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  TDOA RESULTS (GCC-PHAT on bandpass IQ)")
    print(f"{'='*60}")
    for (r, t_rx), res in tdoa_results.items():
        print(f"  {r} → {t_rx}:")
        print(f"    TDOA (integer)   = {res['tdoa_us']:+.3f} µs  "
              f"({res['tdoa_samples']:+d} samples)  "
              f"→ {res['tdoa_m']:+.2f} m")
        print(f"    TDOA (subsample) = {res['tdoa_us_subsamp']:+.4f} µs  "
              f"(offset={res['subsamp_offset']:+.4f} samples)  "
              f"→ {res['tdoa_m_subsamp']:+.2f} m")
        print(f"    PSLR             = {res['pslr_db']:.1f} dB")
    print(f"{'='*60}")
    if pslr_vals:
        avg = np.mean(pslr_vals)
        if avg >= 6:
            print("  ✓ GOOD — peaks are clear, TDOA estimates are trustworthy")
        elif avg >= 3:
            print("  ⚠  WEAK — increase signal strength or narrow bandpass BW")
        else:
            print("  ✗ POOR — check signal is present and bandpass is centred correctly")
    print()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Two-path TDOA: MF detection + bandpass GCC-PHAT estimation."
    )
    ap.add_argument("--data-dir",   required=True,
                    help="Folder with rx0.cfile, rx1.cfile, rx2.cfile")
    ap.add_argument("--filter-dir", default=None,
                    help="Folder with optimal_filter_g_rx*.npy (required for MF path)")
    ap.add_argument("--out-dir",    default=None,
                    help="Output folder for plots")
    ap.add_argument("--fs",         type=float, default=20e6,
                    help="Sample rate Hz (default 20e6)")
    ap.add_argument("--tone-hz",    type=float, default=50e3,
                    help="TX tone offset in Hz (default 50000 — matches your GRC rotator)")
    ap.add_argument("--bp-bw",      type=float, default=400e3,
                    help="Bandpass bandwidth in Hz (default 400000 for 20MHz fs)")
    ap.add_argument("--bp-taps",    type=int,   default=256,
                    help="Bandpass FIR tap count (default 256)")
    ap.add_argument("--max-lag-ms", type=float, default=5.0,
                    help="Max TDOA search window ±ms (default 5)")
    ap.add_argument("--max-samps",  type=int,   default=4_000_000,
                    help="Max samples to load per rx (default 4M)")
    ap.add_argument("--receivers",  default="rx0,rx1,rx2",
                    help="Comma-separated receiver names")
    ap.add_argument("--no-mf",      action="store_true",
                    help="Skip MF detection path (bandpass+TDOA only)")
    args = ap.parse_args()

    receivers = [r.strip() for r in args.receivers.split(",")]
    out_dir   = args.out_dir or os.path.join(args.data_dir, "tdoa_out")
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print("  tdoa_pipeline.py")
    print(f"{'='*60}")
    print(f"  fs        : {args.fs/1e6:.3f} MHz")
    print(f"  tone      : {args.tone_hz/1e3:.1f} kHz")
    print(f"  bp bw     : {args.bp_bw/1e3:.1f} kHz")
    print(f"  max lag   : ±{args.max_lag_ms} ms")
    print(f"  max samps : {args.max_samps:,}")

    # load filters
    g_by_rx = {}
    if not args.no_mf:
        if not args.filter_dir:
            print("\nWARNING: --filter-dir not set, skipping MF detection path.")
            args.no_mf = True
        else:
            for rx in receivers:
                fpath = os.path.join(args.filter_dir, f"optimal_filter_g_{rx}.npy")
                if os.path.isfile(fpath):
                    g_by_rx[rx] = np.load(fpath).astype(np.complex64)
                    print(f"  Loaded filter {rx}: length={g_by_rx[rx].size}")
                else:
                    print(f"  WARNING: filter not found for {rx}: {fpath}")

    raw_iq   = {}
    raw_mags = {}
    mf_mags  = {}
    bp_sigs  = {}

    for rx in receivers:
        print(f"\n{'='*50}  {rx}")
        try:
            cfile = find_cfile(args.data_dir, rx)
        except FileNotFoundError as e:
            print(f"  SKIP: {e}")
            continue

        fsize_mb = os.path.getsize(cfile) / 1e6
        print(f"  Loading: {cfile}  ({fsize_mb:.1f} MB)")

        iq = load_cfile(cfile, args.max_samps)
        raw_iq[rx]   = iq
        raw_mags[rx] = np.abs(iq).astype(np.float32)
        print(f"  Loaded {iq.size:,} samples")

        # PATH 1: Matched filter detection
        if not args.no_mf and rx in g_by_rx:
            print(f"  Running matched filter...")
            mf = MatchedFilterOS(g_by_rx[rx])
            mf_out = mf.run(iq)
            mf_mags[rx] = mf_out
            threshold = np.mean(mf_out) + 3 * np.std(mf_out)
            n_det = int(np.sum(mf_out > threshold))
            print(f"  MF: {n_det:,} samples above threshold ({threshold:.1f})")

        # PATH 2: Bandpass filter
        print(f"  Applying bandpass ({args.tone_hz/1e3:.1f} kHz ±{args.bp_bw/2e3:.1f} kHz)...")
        bp = bandpass_filter(iq, args.fs, args.tone_hz, args.bp_bw, args.bp_taps)
        bp_sigs[rx] = bp
        print(f"  Bandpass done  mean(|bp|)={float(np.mean(np.abs(bp))):.4f}")

    if len(bp_sigs) < 2:
        print("\nERROR: Need at least 2 receivers. Check --data-dir paths.")
        sys.exit(1)

    # GCC-PHAT cross-correlations
    ref_rx = receivers[0]
    if ref_rx not in bp_sigs:
        ref_rx = list(bp_sigs.keys())[0]
        print(f"\nWARNING: {receivers[0]} unavailable, using {ref_rx} as reference")

    tdoa_results = {}
    print(f"\n{'='*50}")
    print(f"  GCC-PHAT cross-correlations (ref={ref_rx})")
    print(f"{'='*50}")
    for rx in receivers[1:]:
        if rx not in bp_sigs:
            print(f"  SKIP {rx}")
            continue
        print(f"  {ref_rx} × {rx}...")
        res = xcorr_tdoa_complex(
            bp_sigs[ref_rx],
            bp_sigs[rx],
            fs=args.fs,
            max_lag_ms=args.max_lag_ms,
        )
        tdoa_results[(ref_rx, rx)] = res
        print(f"  ✓  TDOA={res['tdoa_us']:+.3f} µs  PSLR={res['pslr_db']:.1f} dB")

    print(f"\n  Generating plots...")
    make_plots(
        raw_mags, mf_mags, bp_sigs,
        tdoa_results, args.fs, out_dir,
        receivers, args.tone_hz, args.bp_bw,
    )
    print(f"  All outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()