# analyze_meeting_plots.py
# ------------------------------------------------------------
# Creates meeting-grade comparison plots for:
#   Test 1 (Linearity/Overload): noise_floor_db_mean (+std), peak_abs_mean (+std)
#   Test 2 (Peak quality): peak_idx_std (timing stability), psr_db_mean, width_mean
#
# Input: summary CSVs you already generated:
#   G:\GITFINAL\rf_tdoa\DATA\linearity_summary.csv
#   G:\GITFINAL\rf_tdoa\DATA\peak_quality_summary.csv
#
# Output: PNG + PDF plots in the same DATA folder
# ------------------------------------------------------------

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = r"G:\GITFINAL\rf_tdoa\DATA"
LINEARITY_SUMMARY = os.path.join(DATA_DIR, "linearity_summary.csv")
PEAKQ_SUMMARY = os.path.join(DATA_DIR, "peak_quality_summary.csv")

# Optional: if your notes don't always include DIST/ATTN fields, we still label nicely
def norm_dist(s: str) -> str:
    if not isinstance(s, str) or not s.strip():
        return ""
    t = s.strip().lower()

    # map common variants
    if "20" in t:
        return "20ft"
    if "4.5" in t or "4ft" in t:
        return "4.5ft"
    if "1.5" in t or "1ft" in t:
        return "1.5ft"
    # fallback
    return s.strip()

def norm_attn(s: str) -> str:
    if not isinstance(s, str) or not s.strip():
        return ""
    t = s.strip().lower()
    if t in ("on", "yes", "true"):
        return "on"
    if t in ("off", "no", "false"):
        return "off"
    # accept dB strings too (e.g. "10dB")
    return s.strip()

def label_condition(dist: str, attn: str, note: str = "") -> str:
    d = norm_dist(dist)
    a = norm_attn(attn)
    if d and a:
        return f"{d} | attn {a}"
    if d:
        return d
    if note and isinstance(note, str) and note.strip():
        # squeeze
        nt = re.sub(r"\s+", " ", note.strip())
        return nt[:40] + ("..." if len(nt) > 40 else "")
    return "unknown"

def pick_conditions(df: pd.DataFrame):
    """
    Try to pick the 3 meeting conditions you described:
      - 4.5ft attn off
      - 4.5ft attn on
      - 20ft attn off (CONTROL)
    If not found, fall back to "all conditions available".
    """
    # normalize dist/attn columns if present
    df = df.copy()
    df["dist_n"] = df.get("dist", "").astype(str).apply(norm_dist)
    df["attn_n"] = df.get("attn", "").astype(str).apply(norm_attn)

    want = [
        ("4.5ft", "off"),
        ("4.5ft", "on"),
        ("20ft", "off"),
    ]

    found = []
    for d, a in want:
        m = df[(df["dist_n"] == d) & (df["attn_n"] == a)]
        if len(m) > 0:
            found.append((d, a))

    if len(found) == 3:
        return want  # perfect

    # If you didn't label with dist/attn consistently, attempt by note keywords
    # (still deterministic; no prompts)
    if len(found) == 0 and "note" in df.columns:
        note = df["note"].astype(str).str.lower()
        # rough matches
        conds = []
        def add_if(mask, d, a):
            if df[mask].shape[0] > 0:
                conds.append((d, a))
        add_if(note.str.contains("dist=4.5ft") & note.str.contains("attn=off"), "4.5ft", "off")
        add_if(note.str.contains("dist=4.5ft") & note.str.contains("attn=on"),  "4.5ft", "on")
        add_if(note.str.contains("dist=20ft")  & note.str.contains("attn=off"), "20ft",  "off")
        if len(conds) == 3:
            return want

    # Fallback: show everything, sorted
    conds = sorted(df[["dist_n", "attn_n"]].drop_duplicates().values.tolist())
    return [(d, a) for d, a in conds if d or a]

def filter_to_conditions(df: pd.DataFrame, conditions):
    df = df.copy()
    df["dist_n"] = df.get("dist", "").astype(str).apply(norm_dist)
    df["attn_n"] = df.get("attn", "").astype(str).apply(norm_attn)
    keep = None
    for d, a in conditions:
        mask = (df["dist_n"] == d) & (df["attn_n"] == a)
        keep = mask if keep is None else (keep | mask)
    if keep is None:
        return df
    return df[keep].copy()

def grouped_bar(ax, x_labels, series_by_rx, y_label, title, control_label=None):
    """
    series_by_rx: dict rx -> (means array, stds array)
    """
    rxs = sorted(series_by_rx.keys())
    x = np.arange(len(x_labels))
    width = 0.24 if len(rxs) == 3 else (0.75 / max(1, len(rxs)))

    for i, rx in enumerate(rxs):
        means, stds = series_by_rx[rx]
        ax.bar(x + (i - (len(rxs)-1)/2)*width, means, width, yerr=stds, capsize=3, label=rx)

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=0)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="best")

    # highlight control with a subtle vertical band (optional)
    if control_label and control_label in x_labels:
        idx = x_labels.index(control_label)
        ax.axvspan(idx - 0.5, idx + 0.5, alpha=0.08)

def make_plots():
    if not os.path.exists(LINEARITY_SUMMARY):
        raise FileNotFoundError(f"Missing: {LINEARITY_SUMMARY}")
    if not os.path.exists(PEAKQ_SUMMARY):
        raise FileNotFoundError(f"Missing: {PEAKQ_SUMMARY}")

    lin = pd.read_csv(LINEARITY_SUMMARY)
    pq  = pd.read_csv(PEAKQ_SUMMARY)

    # Choose conditions (4.5 off, 4.5 on, 20 off) if available
    conditions = pick_conditions(lin)
    lin = filter_to_conditions(lin, conditions)
    pq  = filter_to_conditions(pq,  conditions)

    # Build ordered condition labels
    cond_labels = []
    for d, a in conditions:
        # find any note for this condition (for readability)
        sub = lin[(lin["dist"].astype(str).apply(norm_dist) == d) &
                  (lin["attn"].astype(str).apply(norm_attn) == a)]
        note = sub["note"].iloc[0] if (len(sub) and "note" in sub.columns) else ""
        cond_labels.append(label_condition(d, a, note))

    # Determine control label (20ft | attn off if present)
    control_label = None
    for lab, (d, a) in zip(cond_labels, conditions):
        if norm_dist(d) == "20ft" and norm_attn(a) == "off":
            control_label = lab
            break

    # Helper to extract per-condition per-rx
    def extract(df, mean_col, std_col):
        out = {}
        for rx in sorted(df["rx"].unique()):
            means = []
            stds = []
            for (d, a) in conditions:
                sub = df[
                    (df["rx"] == rx) &
                    (df["dist"].astype(str).apply(norm_dist) == norm_dist(d)) &
                    (df["attn"].astype(str).apply(norm_attn) == norm_attn(a))
                ]
                if len(sub) == 0:
                    means.append(np.nan)
                    stds.append(np.nan)
                else:
                    means.append(float(sub[mean_col].iloc[0]))
                    stds.append(float(sub[std_col].iloc[0]))
            out[rx] = (np.array(means), np.array(stds))
        return out

    # -------------------------
    # TEST 1: Linearity/Overload
    # -------------------------
    # Noise floor (mean +/- std)
    fig, ax = plt.subplots(figsize=(10, 5))
    series = extract(lin, "noise_floor_db_mean", "noise_floor_db_std")
    grouped_bar(
        ax, cond_labels, series,
        y_label="Noise floor (dB, median(|IQ|^2) proxy)",
        title="Test 1: Linearity / Overload check — noise floor vs condition",
        control_label=control_label
    )
    fig.tight_layout()
    fig.savefig(os.path.join(DATA_DIR, "MEETING_test1_noise_floor.png"), dpi=180)
    fig.savefig(os.path.join(DATA_DIR, "MEETING_test1_noise_floor.pdf"))
    plt.close(fig)

    # Peak abs (mean +/- std) — flags clipping/compression trends
    fig, ax = plt.subplots(figsize=(10, 5))
    series = extract(lin, "peak_abs_mean", "peak_abs_std")
    grouped_bar(
        ax, cond_labels, series,
        y_label="peak_abs (max |IQ| in chunk)",
        title="Test 1: Linearity / Overload check — peak amplitude vs condition",
        control_label=control_label
    )
    fig.tight_layout()
    fig.savefig(os.path.join(DATA_DIR, "MEETING_test1_peak_abs.png"), dpi=180)
    fig.savefig(os.path.join(DATA_DIR, "MEETING_test1_peak_abs.pdf"))
    plt.close(fig)

    # -------------------------
    # TEST 2: Correlation peak quality
    # -------------------------
    # Timing stability: std(peak_idx) — THIS is the headline metric
    fig, ax = plt.subplots(figsize=(10, 5))
    series = extract(pq, "peak_idx_std", "peak_idx_std")  # show std with itself as "error" (no errorbars)
    # Replace errorbars with zeros to avoid doubling
    series2 = {rx: (m, np.zeros_like(m)) for rx, (m, _) in series.items()}
    grouped_bar(
        ax, cond_labels, series2,
        y_label="std(peak_idx) [samples of MF magnitude stream]",
        title="Test 2: Timing stability — lower is better",
        control_label=control_label
    )
    fig.tight_layout()
    fig.savefig(os.path.join(DATA_DIR, "MEETING_test2_peak_idx_std.png"), dpi=180)
    fig.savefig(os.path.join(DATA_DIR, "MEETING_test2_peak_idx_std.pdf"))
    plt.close(fig)

    # PSR mean +/- std
    fig, ax = plt.subplots(figsize=(10, 5))
    series = extract(pq, "psr_db_mean", "psr_db_std")
    grouped_bar(
        ax, cond_labels, series,
        y_label="PSR (dB) [peak / median sidelobes]",
        title="Test 2: Peak-to-sidelobe ratio — higher is better",
        control_label=control_label
    )
    fig.tight_layout()
    fig.savefig(os.path.join(DATA_DIR, "MEETING_test2_psr.png"), dpi=180)
    fig.savefig(os.path.join(DATA_DIR, "MEETING_test2_psr.pdf"))
    plt.close(fig)

    # Width mean +/- std
    fig, ax = plt.subplots(figsize=(10, 5))
    series = extract(pq, "width_mean", "width_std")
    grouped_bar(
        ax, cond_labels, series,
        y_label="Peak width at -6 dB (samples)",
        title="Test 2: Peak sharpness — lower/narrower is better",
        control_label=control_label
    )
    fig.tight_layout()
    fig.savefig(os.path.join(DATA_DIR, "MEETING_test2_width.png"), dpi=180)
    fig.savefig(os.path.join(DATA_DIR, "MEETING_test2_width.pdf"))
    plt.close(fig)

    # A one-file scoreboard CSV for the meeting
    # (Average across receivers per condition)
    lin_s = lin.copy()
    lin_s["cond"] = [
        label_condition(norm_dist(d), norm_attn(a), n)
        for d, a, n in zip(lin_s.get("dist",""), lin_s.get("attn",""), lin_s.get("note",""))
    ]
    pq_s = pq.copy()
    pq_s["cond"] = [
        label_condition(norm_dist(d), norm_attn(a), n)
        for d, a, n in zip(pq_s.get("dist",""), pq_s.get("attn",""), pq_s.get("note",""))
    ]

    scoreboard = pd.DataFrame({"cond": cond_labels})
    # mean across rx
    scoreboard["noise_floor_db_mean_avg"] = [
        float(lin_s[lin_s["cond"] == c]["noise_floor_db_mean"].mean()) for c in cond_labels
    ]
    scoreboard["peak_abs_mean_avg"] = [
        float(lin_s[lin_s["cond"] == c]["peak_abs_mean"].mean()) for c in cond_labels
    ]
    scoreboard["peak_idx_std_avg"] = [
        float(pq_s[pq_s["cond"] == c]["peak_idx_std"].mean()) for c in cond_labels
    ]
    scoreboard["psr_db_mean_avg"] = [
        float(pq_s[pq_s["cond"] == c]["psr_db_mean"].mean()) for c in cond_labels
    ]
    scoreboard["width_mean_avg"] = [
        float(pq_s[pq_s["cond"] == c]["width_mean"].mean()) for c in cond_labels
    ]

    scoreboard.to_csv(os.path.join(DATA_DIR, "MEETING_scoreboard.csv"), index=False)

    print("[OK] Wrote meeting plots to:", DATA_DIR)
    print("  - MEETING_test1_noise_floor.(png/pdf)")
    print("  - MEETING_test1_peak_abs.(png/pdf)")
    print("  - MEETING_test2_peak_idx_std.(png/pdf)")
    print("  - MEETING_test2_psr.(png/pdf)")
    print("  - MEETING_test2_width.(png/pdf)")
    print("  - MEETING_scoreboard.csv")

if __name__ == "__main__":
    make_plots()
