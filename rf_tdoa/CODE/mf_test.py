#!/usr/bin/env python3
"""
mf_diagnostic_plots.py
=======================
Generates 4 diagnostic plots to validate matched filter performance:

  1. Raw IQ magnitude (pre-MF) — all 3 rx overlaid
  2. MF magnitude output (post-MF) — all 3 rx overlaid, shows peaks
  3. TDOA overlay — zoomed around peaks, shows time offsets between rx
  4. Peak-to-sidelobe ratio — per rx, shows filter quality

Usage:
    python mf_diagnostic_plots.py \
        --data-dir   G:/GITFINAL/rf_tdoa/DATA/RAW_BIN \
        --filter-dir G:/GITFINAL/rf_tdoa/CODE/MATCHED_FILTER/OPTIMAL_FILTERS \
        --out-dir    G:/GITFINAL/rf_tdoa/DATA/DIAGNOSTIC \
        --fs         1e6

    # If MF .bin files already exist from the pipeline script, skip re-running MF:
    python mf_diagnostic_plots.py \
        --data-dir   G:/GITFINAL/rf_tdoa/DATA/RAW_BIN \
        --mf-dir     G:/GITFINAL/rf_tdoa/DATA/PIPELINE_OUT \
        --out-dir    G:/GITFINAL/rf_tdoa/DATA/DIAGNOSTIC \
        --fs         1e6
"""

import argparse
import glob
import os
import sys
import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
except ImportError:
    os.system(f"{sys.executable} -m pip install matplotlib --quiet")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec


COLORS = {"rx0": "#00CFFF", "rx1": "#FF6B35", "rx2": "#A8FF3E"}
DARK_BG = "#0d0d0d"
PANEL_BG = "#141414"
GRID_COLOR = "#2a2a2a"


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


def load_raw_mag(cfile: str, max_samps: int = 2_000_000) -> np.ndarray:
    """Load raw IQ and return magnitude, downsampled for plotting."""
    mm = np.memmap(cfile, dtype=np.complex64, mode="r")
    data = mm[:min(max_samps, len(mm))]
    mag = np.abs(data).astype(np.float32)
    del mm
    return mag


def load_mf_mag(bin_path: str, max_samps: int = 2_000_000) -> np.ndarray:
    """Load pre-computed MF magnitude .bin (float32)."""
    mm = np.memmap(bin_path, dtype=np.float32, mode="r")
    data = mm[:min(max_samps, len(mm))].copy()
    del mm
    return data


# ── Matched filter (overlap-save) ─────────────────────────────────────────────

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
        if fft_size <= self.L:
            raise ValueError("fft_size must be > filter length")
        self.Nfft = int(fft_size)
        self.B = self.Nfft - self.L + 1
        h_pad = np.zeros(self.Nfft, dtype=np.complex64)
        h_pad[:self.L] = self.h
        self.H = np.fft.fft(h_pad)
        self._carry = np.zeros(self.L - 1, dtype=np.complex64) if self.L > 1 else np.zeros(0, dtype=np.complex64)
        self._inbuf = np.zeros(0, dtype=np.complex64)

    def process_file(self, path: str, block_samps: int = 1_000_000):
        with open(path, "rb") as fh:
            while True:
                raw = fh.read(block_samps * 8)
                if not raw:
                    break
                x = np.frombuffer(raw, dtype=np.complex64)
                self._inbuf = np.concatenate((self._inbuf, x)) if self._inbuf.size else x
                while self._inbuf.size >= self.B:
                    take = self._inbuf[:self.B]
                    self._inbuf = self._inbuf[self.B:]
                    x_block = np.concatenate((self._carry, take))
                    if x_block.size < self.Nfft:
                        x_block = np.pad(x_block, (0, self.Nfft - x_block.size))
                    X = np.fft.fft(x_block[:self.Nfft])
                    y = np.fft.ifft(X * self.H).astype(np.complex64, copy=False)
                    y_valid = y[self.L - 1: self.L - 1 + self.B]
                    if self.L > 1:
                        self._carry = x_block[-(self.L - 1):]
                    yield np.abs(y_valid).astype(np.float32)
        if self._inbuf.size:
            take = np.pad(self._inbuf, (0, self.B - self._inbuf.size)).astype(np.complex64)
            self._inbuf = np.zeros(0, dtype=np.complex64)
            x_block = np.concatenate((self._carry, take))
            if x_block.size < self.Nfft:
                x_block = np.pad(x_block, (0, self.Nfft - x_block.size))
            X = np.fft.fft(x_block[:self.Nfft])
            y = np.fft.ifft(X * self.H).astype(np.complex64, copy=False)
            y_valid = y[self.L - 1: self.L - 1 + self.B]
            yield np.abs(y_valid).astype(np.float32)


def run_mf(cfile: str, g: np.ndarray, max_samps: int = 2_000_000) -> np.ndarray:
    mf = MatchedFilterOS(g)
    chunks = []
    total = 0
    for chunk in mf.process_file(cfile):
        chunks.append(chunk)
        total += chunk.size
        if total >= max_samps:
            break
    return np.concatenate(chunks)[:max_samps].astype(np.float32)


# ── Stats helpers ──────────────────────────────────────────────────────────────

def find_peaks_simple(mag: np.ndarray, threshold_factor: float = 0.5):
    """Find indices of peaks above threshold_factor * max."""
    threshold = threshold_factor * np.max(mag)
    above = mag > threshold
    # find rising edges
    edges = np.diff(above.astype(np.int8))
    starts = np.where(edges == 1)[0]
    ends   = np.where(edges == -1)[0]
    if len(starts) == 0:
        return np.array([np.argmax(mag)])
    # clamp
    if len(ends) == 0 or (len(starts) > 0 and starts[0] > ends[0] if len(ends) > 0 else True):
        ends = np.append(ends, len(mag) - 1)
    peaks = []
    for s, e in zip(starts, ends[:len(starts)]):
        peaks.append(s + np.argmax(mag[s:e+1]))
    return np.array(peaks)


def compute_pslr(mag: np.ndarray, peak_idx: int, guard_samps: int = None) -> float:
    """Peak-to-sidelobe ratio in dB."""
    if guard_samps is None:
        guard_samps = max(10, len(mag) // 500)
    lo = max(0, peak_idx - guard_samps)
    hi = min(len(mag), peak_idx + guard_samps)
    sidelobe_mag = np.concatenate([mag[:lo], mag[hi:]])
    if sidelobe_mag.size == 0:
        return 0.0
    pslr = 20 * np.log10(mag[peak_idx] / (np.max(sidelobe_mag) + 1e-12))
    return float(pslr)


def downsample_for_plot(arr: np.ndarray, target: int = 50_000) -> np.ndarray:
    if len(arr) <= target:
        return arr
    step = len(arr) // target
    return arr[::step]


# ── Plot style ─────────────────────────────────────────────────────────────────

def style_ax(ax, title: str, xlabel: str, ylabel: str):
    ax.set_facecolor(PANEL_BG)
    ax.set_title(title, color="white", fontsize=11, pad=8)
    ax.set_xlabel(xlabel, color="#888888", fontsize=9)
    ax.set_ylabel(ylabel, color="#888888", fontsize=9)
    ax.tick_params(colors="#666666", labelsize=8)
    ax.spines["bottom"].set_color("#333333")
    ax.spines["top"].set_color("#333333")
    ax.spines["left"].set_color("#333333")
    ax.spines["right"].set_color("#333333")
    ax.grid(True, color=GRID_COLOR, linewidth=0.5, linestyle="--", alpha=0.7)


# ── Main plot function ─────────────────────────────────────────────────────────

def make_diagnostic_plots(
    raw_mags: dict,
    mf_mags: dict,
    fs: float,
    out_dir: str,
    receivers: list,
):
    os.makedirs(out_dir, exist_ok=True)

    # ── Figure 1: Raw IQ magnitude (pre-MF) ───────────────────────────────────
    print("  [plot 1/4] Raw IQ magnitude...")
    fig, ax = plt.subplots(figsize=(14, 4), facecolor=DARK_BG)
    for rx in receivers:
        if rx not in raw_mags:
            continue
        mag = downsample_for_plot(raw_mags[rx])
        t = np.arange(len(mag)) * (len(raw_mags[rx]) / len(mag)) / fs * 1e3  # ms
        ax.plot(t, mag, color=COLORS.get(rx, "white"), linewidth=0.6, alpha=0.85, label=rx)
    style_ax(ax, "RAW IQ Magnitude (Pre Matched Filter)", "Time (ms)", "Amplitude")
    ax.legend(facecolor="#1a1a1a", edgecolor="#333", labelcolor="white", fontsize=9)
    fig.tight_layout()
    p = os.path.join(out_dir, "plot1_raw_iq_magnitude.png")
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"    ✓ {p}")

    # ── Figure 2: MF magnitude output (post-MF) ───────────────────────────────
    print("  [plot 2/4] MF magnitude output...")
    fig, axes = plt.subplots(len(receivers), 1, figsize=(14, 3 * len(receivers)),
                             facecolor=DARK_BG, sharex=False)
    if len(receivers) == 1:
        axes = [axes]

    peak_indices = {}
    for i, rx in enumerate(receivers):
        ax = axes[i]
        if rx not in mf_mags:
            ax.set_visible(False)
            continue
        mag = mf_mags[rx]
        mag_ds = downsample_for_plot(mag)
        t = np.arange(len(mag_ds)) * (len(mag) / len(mag_ds)) / fs * 1e3

        ax.set_facecolor(PANEL_BG)
        ax.fill_between(t, mag_ds, alpha=0.15, color=COLORS.get(rx, "white"))
        ax.plot(t, mag_ds, color=COLORS.get(rx, "white"), linewidth=0.7, label=rx)

        # mark peaks
        peak_idx = np.argmax(mag)
        peak_indices[rx] = peak_idx
        peak_t = peak_idx / fs * 1e3
        peak_val_ds = mag_ds[int(peak_idx * len(mag_ds) / len(mag))]
        ax.axvline(peak_t, color="yellow", linewidth=1.0, linestyle="--", alpha=0.8)
        ax.annotate(f"peak @ {peak_t:.3f} ms",
                    xy=(peak_t, peak_val_ds),
                    xytext=(peak_t + (t[-1] * 0.02), peak_val_ds * 0.85),
                    color="yellow", fontsize=8,
                    arrowprops=dict(arrowstyle="->", color="yellow", lw=0.8))

        pslr = compute_pslr(mag, peak_idx)
        style_ax(ax, f"{rx} — MF Output  |  Peak PSLR: {pslr:.1f} dB", "Time (ms)", "Correlation Magnitude")
        ax.legend(facecolor="#1a1a1a", edgecolor="#333", labelcolor="white", fontsize=9)

    fig.tight_layout(pad=1.5)
    p = os.path.join(out_dir, "plot2_mf_magnitude_per_rx.png")
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"    ✓ {p}")

    # ── Figure 3: TDOA overlay — all rx zoomed around peaks ───────────────────
    print("  [plot 3/4] TDOA overlay...")
    fig, ax = plt.subplots(figsize=(14, 5), facecolor=DARK_BG)
    ax.set_facecolor(PANEL_BG)

    # find global peak region
    all_peak_times = []
    for rx in receivers:
        if rx in peak_indices:
            all_peak_times.append(peak_indices[rx] / fs)

    if all_peak_times:
        center_t = np.mean(all_peak_times)
        # zoom window: ±5ms around center or ±500 samples, whichever is larger
        zoom_sec = max(0.005, (max(all_peak_times) - min(all_peak_times)) * 3 + 0.002)
        t_lo = max(0.0, center_t - zoom_sec)
        t_hi = center_t + zoom_sec
    else:
        t_lo, t_hi = 0.0, 0.01

    ref_peak = None
    tdoa_results = {}

    for rx in receivers:
        if rx not in mf_mags:
            continue
        mag = mf_mags[rx]
        i_lo = int(t_lo * fs)
        i_hi = min(int(t_hi * fs), len(mag))
        seg = mag[i_lo:i_hi]
        t_seg = (np.arange(len(seg)) + i_lo) / fs * 1e3  # ms

        ax.plot(t_seg, seg, color=COLORS.get(rx, "white"), linewidth=1.2,
                alpha=0.9, label=rx)

        # peak in segment
        local_peak = i_lo + np.argmax(seg)
        peak_t_ms = local_peak / fs * 1e3
        ax.axvline(peak_t_ms, color=COLORS.get(rx, "white"),
                   linewidth=0.8, linestyle=":", alpha=0.6)

        if ref_peak is None:
            ref_peak = local_peak
            tdoa_results[rx] = 0.0
        else:
            tdoa_samps = local_peak - ref_peak
            tdoa_us = tdoa_samps / fs * 1e6
            tdoa_results[rx] = tdoa_us

    # annotate TDOAs
    rx_list = list(tdoa_results.keys())
    for j, rx in enumerate(rx_list[1:], 1):
        dt = tdoa_results[rx]
        ax.annotate(
            f"TDOA {rx_list[0]}→{rx}: {dt:+.2f} µs  ({dt * 0.3e3:.1f} m at c)",
            xy=(0.02, 0.92 - j * 0.1),
            xycoords="axes fraction",
            color=COLORS.get(rx, "white"),
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#1a1a1a", edgecolor="#333", alpha=0.8)
        )

    style_ax(ax, "TDOA Overlay — MF Peaks (Zoomed)", "Time (ms)", "Correlation Magnitude")
    ax.legend(facecolor="#1a1a1a", edgecolor="#333", labelcolor="white", fontsize=9)
    fig.tight_layout()
    p = os.path.join(out_dir, "plot3_tdoa_overlay.png")
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"    ✓ {p}")
    print(f"    TDOA results: { {k: f'{v:.2f} µs' for k, v in tdoa_results.items()} }")

    # ── Figure 4: Peak-to-sidelobe ratio bar chart + noise floor ──────────────
    print("  [plot 4/4] PSLR and noise floor...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), facecolor=DARK_BG)

    pslr_vals = []
    snr_vals = []
    rx_labels = []

    for rx in receivers:
        if rx not in mf_mags:
            continue
        mag = mf_mags[rx]
        peak_idx = np.argmax(mag)
        pslr = compute_pslr(mag, peak_idx)
        noise_floor = np.median(mag)
        snr = 20 * np.log10(mag[peak_idx] / (noise_floor + 1e-12))
        pslr_vals.append(pslr)
        snr_vals.append(snr)
        rx_labels.append(rx)

    colors_list = [COLORS.get(rx, "white") for rx in rx_labels]

    # PSLR bars
    ax1.set_facecolor(PANEL_BG)
    bars = ax1.bar(rx_labels, pslr_vals, color=colors_list, width=0.4, zorder=3)
    ax1.axhline(13.3, color="yellow", linewidth=1.0, linestyle="--", alpha=0.7,
                label="Ideal sinc PSLR (13.3 dB)")
    for bar, val in zip(bars, pslr_vals):
        ax1.text(bar.get_x() + bar.get_width() / 2, val + 0.3,
                 f"{val:.1f} dB", ha="center", va="bottom", color="white", fontsize=9)
    style_ax(ax1, "Peak-to-Sidelobe Ratio (PSLR)", "Receiver", "PSLR (dB)")
    ax1.legend(facecolor="#1a1a1a", edgecolor="#333", labelcolor="white", fontsize=8)
    ax1.set_ylim(0, max(pslr_vals + [14]) * 1.2 if pslr_vals else 20)

    # SNR bars
    ax2.set_facecolor(PANEL_BG)
    bars2 = ax2.bar(rx_labels, snr_vals, color=colors_list, width=0.4, zorder=3)
    for bar, val in zip(bars2, snr_vals):
        ax2.text(bar.get_x() + bar.get_width() / 2, val + 0.3,
                 f"{val:.1f} dB", ha="center", va="bottom", color="white", fontsize=9)
    style_ax(ax2, "Peak SNR (peak / median noise floor)", "Receiver", "SNR (dB)")
    ax2.set_ylim(0, max(snr_vals + [1]) * 1.2 if snr_vals else 20)

    fig.tight_layout(pad=2.0)
    p = os.path.join(out_dir, "plot4_pslr_snr.png")
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"    ✓ {p}")

    # ── Summary print ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    for rx in rx_labels:
        idx = rx_labels.index(rx)
        print(f"  {rx}:  PSLR={pslr_vals[idx]:.1f} dB  |  SNR={snr_vals[idx]:.1f} dB", end="")
        if rx in tdoa_results and tdoa_results[rx] != 0.0:
            print(f"  |  TDOA={tdoa_results[rx]:+.2f} µs", end="")
        print()
    print(f"\n  Filter verdict:")
    avg_pslr = np.mean(pslr_vals) if pslr_vals else 0
    avg_snr  = np.mean(snr_vals)  if snr_vals  else 0
    if avg_snr > 10 and avg_pslr > 6:
        print("  ✓ MF appears to be WORKING — clear peaks detected above noise floor")
    elif avg_snr > 3:
        print("  ⚠  Weak peaks detected — check signal strength or filter alignment")
    else:
        print("  ✗ No clear peaks — wrong signal, bad filter, or no signal present")
    print(f"{'='*60}\n")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="MF diagnostic plots — validate matched filter performance.")
    ap.add_argument("--data-dir",   required=True,  help="Folder with rx0.cfile, rx1.cfile, rx2.cfile")
    ap.add_argument("--filter-dir", default=None,   help="Folder with optimal_filter_g_rx*.npy (not needed if --mf-dir given)")
    ap.add_argument("--mf-dir",     default=None,   help="Folder with pre-computed mf_mag_rx*.bin files (skips MF step)")
    ap.add_argument("--out-dir",    default=None,   help="Output folder for PNGs (default: <data-dir>/diagnostic)")
    ap.add_argument("--fs",         type=float, default=1e6,  help="Sample rate Hz (default 1e6)")
    ap.add_argument("--max-samps",  type=int,   default=4_000_000, help="Max samples to load per rx (default 4M)")
    ap.add_argument("--receivers",  default="rx0,rx1,rx2", help="Comma-separated receiver names")
    args = ap.parse_args()

    receivers = [r.strip() for r in args.receivers.split(",")]
    out_dir = args.out_dir or os.path.join(args.data_dir, "diagnostic")
    os.makedirs(out_dir, exist_ok=True)

    raw_mags = {}
    mf_mags  = {}

    # Load filters if we need to run MF
    g_by_rx = {}
    if args.mf_dir is None:
        if args.filter_dir is None:
            print("ERROR: provide --filter-dir or --mf-dir")
            sys.exit(1)
        for rx in receivers:
            fpath = os.path.join(args.filter_dir, f"optimal_filter_g_{rx}.npy")
            if not os.path.isfile(fpath):
                print(f"ERROR: filter not found: {fpath}")
                sys.exit(1)
            g_by_rx[rx] = np.load(fpath).astype(np.complex64, copy=False)
            print(f"  Loaded filter {rx}: length={g_by_rx[rx].size}")

    for rx in receivers:
        print(f"\n{'='*50}  {rx}")

        # raw IQ
        try:
            cfile = find_cfile(args.data_dir, rx)
            print(f"  Loading raw IQ: {cfile}")
            raw_mags[rx] = load_raw_mag(cfile, max_samps=args.max_samps)
        except FileNotFoundError as e:
            print(f"  SKIP raw: {e}")

        # MF magnitude
        if args.mf_dir:
            bin_path = os.path.join(args.mf_dir, f"mf_mag_{rx}.bin")
            if os.path.isfile(bin_path):
                print(f"  Loading pre-computed MF: {bin_path}")
                mf_mags[rx] = load_mf_mag(bin_path, max_samps=args.max_samps)
            else:
                print(f"  WARNING: MF bin not found: {bin_path}")
        else:
            if rx in raw_mags and rx in g_by_rx:
                print(f"  Running matched filter...")
                try:
                    cfile = find_cfile(args.data_dir, rx)
                    mf_mags[rx] = run_mf(cfile, g_by_rx[rx], max_samps=args.max_samps)
                    print(f"  MF output samples: {mf_mags[rx].size:,}")
                except Exception as e:
                    print(f"  ERROR running MF: {e}")

    if not mf_mags:
        print("ERROR: No MF data to plot. Check paths and filter files.")
        sys.exit(1)

    print(f"\n{'='*50}")
    print("  Generating diagnostic plots...")
    print(f"{'='*50}")
    make_diagnostic_plots(raw_mags, mf_mags, args.fs, out_dir, receivers)
    print(f"  All plots saved to: {out_dir}")


if __name__ == "__main__":
    main()