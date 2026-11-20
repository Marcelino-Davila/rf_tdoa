# plot_linearity_csvs.py
# ------------------------------------------------------------
# Usage:
#   python plot_linearity_csvs.py run1.csv run2.csv run3.csv --labels "1.5ft" "4.5ft" "20ft" --outdir GRAPHS
#
# What it generates (PNG files in --outdir):
#   1) noise_floor_db vs time (one plot per run, with 3 receiver traces)
#   2) peak_abs vs time (one plot per run, with 3 receiver traces)
#   3) summary_median_noise_floor.png  (grouped bars: run x receiver)
#   4) summary_median_peak_abs.png     (grouped bars: run x receiver)
#
# Notes:
# - Assumes your CSV has columns: t_unix, rx, chunk_idx, peak_abs, noise_floor_db
# - Time axis is converted to "seconds since start of that run"
# ------------------------------------------------------------

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


RECEIVERS = ["rx0", "rx1", "rx2"]


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # defensive: enforce expected columns
    needed = {"t_unix", "rx", "peak_abs", "noise_floor_db"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")

    df["t_unix"] = pd.to_numeric(df["t_unix"], errors="coerce")
    df["peak_abs"] = pd.to_numeric(df["peak_abs"], errors="coerce")
    df["noise_floor_db"] = pd.to_numeric(df["noise_floor_db"], errors="coerce")
    df = df.dropna(subset=["t_unix", "rx", "peak_abs", "noise_floor_db"]).copy()

    # seconds since start (per run)
    t0 = df["t_unix"].min()
    df["t_rel_s"] = df["t_unix"] - t0

    return df


def plot_timeseries(df: pd.DataFrame, label: str, outdir: str, ycol: str, ylabel: str, fname_prefix: str):
    plt.figure()
    for rx in RECEIVERS:
        d = df[df["rx"] == rx].sort_values("t_rel_s")
        if len(d) == 0:
            continue
        plt.plot(d["t_rel_s"].to_numpy(), d[ycol].to_numpy(), label=rx)

    plt.xlabel("time since run start (s)")
    plt.ylabel(ylabel)
    plt.title(f"{fname_prefix} vs time — {label}")
    plt.legend()
    plt.tight_layout()

    outpath = os.path.join(outdir, f"{fname_prefix}_{safe_name(label)}.png")
    plt.savefig(outpath, dpi=160)
    plt.close()
    return outpath


def safe_name(s: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in s)


def grouped_bar(summary: pd.DataFrame, value_col: str, title: str, ylabel: str, outpath: str):
    """
    summary columns: run_label, rx, value_col
    """
    run_labels = list(summary["run_label"].unique())
    rx_labels = RECEIVERS

    # grid: runs x receivers
    values = np.full((len(run_labels), len(rx_labels)), np.nan)
    for i, rl in enumerate(run_labels):
        for j, rx in enumerate(rx_labels):
            v = summary[(summary["run_label"] == rl) & (summary["rx"] == rx)][value_col]
            if len(v) > 0:
                values[i, j] = float(v.iloc[0])

    x = np.arange(len(run_labels))
    width = 0.25  # one bar per rx within each run group

    plt.figure()
    for j, rx in enumerate(rx_labels):
        plt.bar(x + (j - 1) * width, values[:, j], width, label=rx)

    plt.xticks(x, run_labels)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csvs", nargs="+", help="Paths to bench_linearity CSVs (3 recommended)")
    ap.add_argument("--labels", nargs="*", default=None, help='Optional labels matching CSVs, e.g. "1.5ft" "4.5ft" "20ft"')
    ap.add_argument("--outdir", default="GRAPHS", help="Output directory for PNGs")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    csvs = args.csvs
    if args.labels is None or len(args.labels) == 0:
        labels = [os.path.splitext(os.path.basename(p))[0] for p in csvs]
    else:
        if len(args.labels) != len(csvs):
            raise SystemExit("If you pass --labels, you must provide exactly one label per CSV.")
        labels = args.labels

    all_summ_rows = []
    written = []

    for path, label in zip(csvs, labels):
        df = load_csv(path)

        # timeseries plots
        written.append(plot_timeseries(df, label, args.outdir, "noise_floor_db", "noise floor (dB)", "noise_floor_db"))
        written.append(plot_timeseries(df, label, args.outdir, "peak_abs", "peak |IQ|", "peak_abs"))

        # per-run median summary
        for rx in RECEIVERS:
            d = df[df["rx"] == rx]
            if len(d) == 0:
                continue
            all_summ_rows.append({
                "run_label": label,
                "rx": rx,
                "median_noise_floor_db": float(d["noise_floor_db"].median()),
                "median_peak_abs": float(d["peak_abs"].median()),
            })

    summary = pd.DataFrame(all_summ_rows)
    if len(summary) == 0:
        raise SystemExit("No data found in provided CSVs.")

    # grouped bar summaries
    out1 = os.path.join(args.outdir, "summary_median_noise_floor.png")
    out2 = os.path.join(args.outdir, "summary_median_peak_abs.png")

    grouped_bar(
        summary[["run_label", "rx", "median_noise_floor_db"]].drop_duplicates(),
        "median_noise_floor_db",
        "Median noise floor by run and receiver",
        "median noise floor (dB)",
        out1,
    )
    grouped_bar(
        summary[["run_label", "rx", "median_peak_abs"]].drop_duplicates(),
        "median_peak_abs",
        "Median peak amplitude by run and receiver",
        "median peak |IQ|",
        out2,
    )

    print("[i] Wrote plots:")
    for p in written:
        print("  -", p)
    print("  -", out1)
    print("  -", out2)


if __name__ == "__main__":
    main()
