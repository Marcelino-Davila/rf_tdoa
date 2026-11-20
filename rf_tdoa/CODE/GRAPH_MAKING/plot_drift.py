import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def infer_time_col(df: pd.DataFrame) -> str:
    for c in ["t_s", "t_sec", "t_seconds", "time_s", "time_sec", "t"]:
        if c in df.columns:
            return c
    if "t_unix" in df.columns:
        return "t_unix"
    raise ValueError(f"Could not find a time column. Columns={list(df.columns)}")

def infer_freq_col(df: pd.DataFrame) -> str:
    # common names we might have produced
    for c in ["peak_freq_hz", "fft_peak_hz", "peak_hz", "peak_freq"]:
        if c in df.columns:
            return c
    # fallback: any column containing "freq" and "hz"
    for c in df.columns:
        lc = c.lower()
        if ("freq" in lc) and ("hz" in lc):
            return c
    raise ValueError(f"Could not find frequency column. Columns={list(df.columns)}")

def reject_jumps(y: np.ndarray, max_jump_hz: float = 5000.0) -> np.ndarray:
    """
    Mask instantaneous FFT peak mis-picks by breaking the line when
    sample-to-sample jumps exceed max_jump_hz.
    """
    y = np.asarray(y, dtype=float)
    y2 = y.copy()
    if y2.size < 2:
        return y2
    dy = np.abs(np.diff(y2))
    bad = np.where(dy > max_jump_hz)[0] + 1
    y2[bad] = np.nan
    return y2

def robust_center(y: np.ndarray) -> np.ndarray:
    med = np.nanmedian(y)
    return y - med

def robust_ylim(y_by_rx: dict, clip_hz: float) -> tuple[float, float]:
    """
    If clip_hz is provided, use +/-clip_hz.
    Otherwise compute a robust span from percentiles.
    """
    if clip_hz is not None:
        return (-clip_hz, clip_hz)

    all_vals = []
    for y in y_by_rx.values():
        yy = np.asarray(y, dtype=float)
        yy = yy[np.isfinite(yy)]
        if yy.size:
            all_vals.append(yy)
    if not all_vals:
        return (-200, 200)

    all_vals = np.concatenate(all_vals)
    lo, hi = np.percentile(all_vals, [2, 98])
    span = max(abs(lo), abs(hi))
    span = max(span, 50.0)
    return (-span, span)

# ------------------------------------------------------------
# Main plotting
# ------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Plot drift benchmark from bench_drift CSV")
    ap.add_argument("csv", help="Path to bench_drift_*.csv")
    ap.add_argument("--outdir", default=".", help="Output directory for PNGs")
    ap.add_argument("--receivers", nargs="*", default=["rx0", "rx1", "rx2"], help="Receiver names to plot")
    ap.add_argument("--max-jump-hz", type=float, default=5000.0, help="Jump threshold to mask FFT mis-picks")
    ap.add_argument("--clip-hz", type=float, default=200.0,
                    help="Y-limit for relative drift plot (Hz). Use 0 to auto-scale robustly.")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    if "rx" not in df.columns:
        raise ValueError(f"CSV missing required column 'rx'. Columns={list(df.columns)}")

    tcol = infer_time_col(df)
    fcol = infer_freq_col(df)

    # Build per-receiver series
    t0 = None
    per = {}

    for rx in args.receivers:
        sub = df[df["rx"] == rx].copy()
        if sub.empty:
            continue

        # time vector
        t = sub[tcol].astype(float).to_numpy()
        if t0 is None:
            # if unix time, normalize to start at 0
            t0 = float(np.nanmin(t))
        if tcol == "t_unix":
            t = t - t0  # seconds since start

        y = sub[fcol].astype(float).to_numpy()

        per[rx] = {"t": t, "y": y}

    if not per:
        raise ValueError(f"No receiver data found for {args.receivers} in {args.csv}")

    ensure_dir(args.outdir)

    base = os.path.splitext(os.path.basename(args.csv))[0]

    # ------------------------------------------------------------
    # 1) Absolute peak frequency vs time (baseband)
    # ------------------------------------------------------------
    plt.figure()
    for rx, d in per.items():
        plt.plot(d["t"], d["y"], label=rx)
    plt.title("Drift baseline: FFT peak frequency vs time (baseband)")
    plt.xlabel("time since start (s)")
    plt.ylabel("FFT peak frequency (Hz, baseband)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    out1 = os.path.join(args.outdir, f"{base}_drift_peak_freq.png")
    plt.savefig(out1, dpi=150, bbox_inches="tight")

    # ------------------------------------------------------------
    # 2) Relative drift (each receiver centered) + outlier masking
    # ------------------------------------------------------------
    plt.figure()
    y_centered_by_rx = {}

    for rx, d in per.items():
        t = d["t"]
        y = d["y"]

        y_centered = robust_center(y)
        y_centered = reject_jumps(y_centered, max_jump_hz=args.max_jump_hz)

        y_centered_by_rx[rx] = y_centered
        plt.plot(t, y_centered, label=rx)

    plt.title("Relative drift (each receiver centered)")
    plt.xlabel("time since start (s)")
    plt.ylabel("peak freq minus median (Hz)")
    # y-limits
    clip = None if (args.clip_hz is not None and args.clip_hz <= 0) else args.clip_hz
    lo, hi = robust_ylim(y_centered_by_rx, clip)
    plt.ylim(lo, hi)
    plt.grid(True, alpha=0.3)
    plt.legend()
    out2 = os.path.join(args.outdir, f"{base}_drift_relative.png")
    plt.savefig(out2, dpi=150, bbox_inches="tight")

    print(f"[i] Wrote:\n  {out1}\n  {out2}")

if __name__ == "__main__":
    main()
