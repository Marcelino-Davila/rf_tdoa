"""
TDOA GUI (3 receivers) — quick, practical starter

What it does:
- Shows RX positions and the current estimated target position on a 2D map
- Displays per-receiver metrics: SNR (dB), correlation peak, normalized correlation, peak-to-sidelobe ratio (PSR),
  and (optionally) an arrival-time / TDOA estimate if you provide it
- Designed to be easy to hook into your existing pipeline:
    - implement `get_latest_snapshot()` to pull your real-time results
    - or point it at a JSON file that your backend updates continuously

Dependencies:
    pip install matplotlib numpy

If you want it prettier / faster at scale, we can move to PyQtGraph, but this is the fastest "works everywhere" GUI.
"""

import json
import os
import time
import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


# ----------------------------
# Metric helpers (backend-side)
# ----------------------------

def normalized_peak(corr: np.ndarray) -> float:
    """Peak normalized by RMS of correlation magnitude (robust-ish scale)."""
    mag = np.abs(corr)
    peak = float(np.max(mag))
    rms = float(np.sqrt(np.mean(mag**2)) + 1e-12)
    return peak / rms


def peak_to_sidelobe_ratio_db(corr: np.ndarray, guard_bins: int = 8) -> float:
    """
    PSR in dB: 20*log10(peak / sidelobe_rms)
    guard_bins excludes bins around the main peak.
    """
    mag = np.abs(corr)
    k = int(np.argmax(mag))
    peak = float(mag[k])

    # mask out a guard region around the peak
    mask = np.ones_like(mag, dtype=bool)
    lo = max(0, k - guard_bins)
    hi = min(len(mag), k + guard_bins + 1)
    mask[lo:hi] = False

    sidelobes = mag[mask]
    if sidelobes.size == 0:
        return float("inf")

    sidelobe_rms = float(np.sqrt(np.mean(sidelobes**2)) + 1e-12)
    return 20.0 * np.log10((peak + 1e-12) / sidelobe_rms)


def matched_filter_snr_db(y: np.ndarray, peak_idx: int | None = None, noise_guard: int = 16) -> float:
    """
    Simple SNR estimate from matched-filter output y:
      SNR ≈ peak_power / noise_power
    where noise_power is estimated from y excluding a guard region around the peak.
    """
    mag2 = np.abs(y) ** 2
    if peak_idx is None:
        peak_idx = int(np.argmax(mag2))
    peak_power = float(mag2[peak_idx])

    mask = np.ones_like(mag2, dtype=bool)
    lo = max(0, peak_idx - noise_guard)
    hi = min(len(mag2), peak_idx + noise_guard + 1)
    mask[lo:hi] = False
    noise = mag2[mask]
    if noise.size == 0:
        return float("inf")
    noise_power = float(np.mean(noise) + 1e-12)

    return 10.0 * np.log10((peak_power + 1e-12) / noise_power)


# ------------------------------------
# Data interface (plug your backend in)
# ------------------------------------

def demo_snapshot(t: float) -> dict:
    """
    Demo generator so the GUI runs immediately.
    Replace this with your real pipeline output.
    """
    # receivers (x,y) in meters
    rxs = np.array([[0.0, 0.0],
                    [100.0, 0.0],
                    [40.0, 80.0]], dtype=float)

    # moving target
    target = np.array([50 + 20 * np.cos(t / 2.0), 40 + 15 * np.sin(t / 1.5)], dtype=float)

    per_rx = []
    rng = np.random.default_rng(1234)
    for i in range(3):
        # fake correlation vector with a peak
        n = 256
        corr = (rng.normal(0, 1, n) + 1j * rng.normal(0, 1, n)) * 0.2
        peak_bin = int(60 + 10 * i + 5 * np.sin(t + i))
        corr[peak_bin] += 5.0 + 2.0 * np.cos(t + i)

        snr_db = matched_filter_snr_db(corr, peak_idx=peak_bin)
        peak = float(np.max(np.abs(corr)))
        npeak = normalized_peak(corr)
        psr = peak_to_sidelobe_ratio_db(corr, guard_bins=8)

        per_rx.append({
            "rx": f"rx{i}",
            "snr_db": snr_db,
            "corr_peak": peak,
            "corr_norm": npeak,
            "psr_db": psr,
            # Optional fields you might add from your backend:
            # "toa_s": ...,        # time-of-arrival estimate
            # "tdoa_s": ...,       # relative time diff vs reference
        })

    return {
        "timestamp": time.time(),
        "rx_positions": rxs.tolist(),
        "target_position": target.tolist(),
        "metrics": per_rx,
    }


def read_json_snapshot(path: str) -> dict | None:
    """
    Reads a JSON snapshot from disk.
    Your backend can overwrite this file continuously.

    Expected JSON schema (example):
    {
      "timestamp": 1730000000.0,
      "rx_positions": [[0,0],[100,0],[40,80]],
      "target_position": [52.1, 41.3],
      "metrics": [
        {"rx":"rx0","snr_db":12.3,"corr_peak":5.1,"corr_norm":8.2,"psr_db":10.7},
        {"rx":"rx1","snr_db":...},
        {"rx":"rx2","snr_db":...}
      ]
    }
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        # file might be mid-write; ignore this update
        return None


# ----------------------------
# GUI
# ----------------------------

class TDOAGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("TDOA Monitor (3 RX)")

        # --- controls
        controls = ttk.Frame(self, padding=8)
        controls.pack(side=tk.TOP, fill=tk.X)

        self.mode = tk.StringVar(value="DEMO")
        ttk.Label(controls, text="Data source:").pack(side=tk.LEFT)
        ttk.Combobox(
            controls,
            textvariable=self.mode,
            values=["DEMO", "JSON_FILE"],
            state="readonly",
            width=10
        ).pack(side=tk.LEFT, padx=(6, 12))

        self.json_path = tk.StringVar(value="tdoa_snapshot.json")
        ttk.Label(controls, text="JSON path:").pack(side=tk.LEFT)
        ttk.Entry(controls, textvariable=self.json_path, width=40).pack(side=tk.LEFT, padx=(6, 12))

        self.refresh_ms = tk.IntVar(value=200)  # 5 Hz default
        ttk.Label(controls, text="Refresh (ms):").pack(side=tk.LEFT)
        ttk.Spinbox(controls, from_=50, to=2000, increment=50, textvariable=self.refresh_ms, width=6).pack(
            side=tk.LEFT, padx=(6, 12)
        )

        ttk.Button(controls, text="Recenter", command=self.recenter).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(controls, text="Quit", command=self.destroy).pack(side=tk.LEFT)

        # --- main layout
        main = ttk.Frame(self, padding=8)
        main.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # left: plot
        plot_frame = ttk.Frame(main)
        plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.fig = Figure(figsize=(6, 5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Receiver / Target Map")
        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Y (m)")
        self.ax.grid(True)

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # right: metrics table
        table_frame = ttk.Frame(main)
        table_frame.pack(side=tk.RIGHT, fill=tk.Y)

        ttk.Label(table_frame, text="Per-Receiver Metrics", font=("Segoe UI", 11, "bold")).pack(
            side=tk.TOP, pady=(0, 6)
        )

        cols = ("rx", "snr_db", "corr_peak", "corr_norm", "psr_db")
        self.tree = ttk.Treeview(table_frame, columns=cols, show="headings", height=12)

        headings = {
            "rx": "RX",
            "snr_db": "SNR (dB)",
            "corr_peak": "Corr Peak",
            "corr_norm": "Corr Norm",
            "psr_db": "PSR (dB)",
        }
        widths = {"rx": 55, "snr_db": 75, "corr_peak": 85, "corr_norm": 85, "psr_db": 75}

        for c in cols:
            self.tree.heading(c, text=headings[c])
            self.tree.column(c, width=widths[c], anchor=tk.CENTER)

        self.tree.pack(side=tk.TOP, fill=tk.Y, expand=False)

        self.status = tk.StringVar(value="Ready.")
        ttk.Label(table_frame, textvariable=self.status, wraplength=320).pack(side=tk.TOP, pady=(8, 0), fill=tk.X)

        # state for plotting
        self._xlim = None
        self._ylim = None

        # kick off updates
        self.after(50, self.update_loop)

    def recenter(self):
        self._xlim = None
        self._ylim = None

    def get_latest_snapshot(self) -> dict | None:
        if self.mode.get() == "DEMO":
            return demo_snapshot(time.time())
        else:
            path = self.json_path.get().strip()
            if not path:
                return None
            return read_json_snapshot(path)

    def update_loop(self):
        snap = self.get_latest_snapshot()

        if snap is None:
            self.status.set("No data yet. (If using JSON_FILE, ensure backend writes a valid JSON snapshot.)")
        else:
            try:
                self.render_snapshot(snap)
                ts = snap.get("timestamp", None)
                if ts is not None:
                    self.status.set(f"Last update: {time.strftime('%H:%M:%S', time.localtime(ts))}")
                else:
                    self.status.set("Updated.")
            except Exception as e:
                self.status.set(f"Render error: {e}")

        self.after(int(self.refresh_ms.get()), self.update_loop)

    def render_snapshot(self, snap: dict):
        rxs = np.array(snap["rx_positions"], dtype=float)
        tgt = np.array(snap["target_position"], dtype=float)
        metrics = snap.get("metrics", [])

        # --- plot
        self.ax.cla()
        self.ax.set_title("Receiver / Target Map")
        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Y (m)")
        self.ax.grid(True)

        # RX points
        self.ax.scatter(rxs[:, 0], rxs[:, 1], marker="^", s=90, label="Receivers")
        for i, (x, y) in enumerate(rxs):
            self.ax.annotate(f"rx{i}", (x, y), textcoords="offset points", xytext=(6, 6))

        # Target point
        self.ax.scatter([tgt[0]], [tgt[1]], marker="*", s=150, label="Target")
        self.ax.annotate("target", (tgt[0], tgt[1]), textcoords="offset points", xytext=(6, 6))

        # auto limits (sticky until recenter)
        allx = np.r_[rxs[:, 0], tgt[0]]
        ally = np.r_[rxs[:, 1], tgt[1]]
        pad = 0.10
        xmin, xmax = float(allx.min()), float(allx.max())
        ymin, ymax = float(ally.min()), float(ally.max())
        dx = max(1.0, xmax - xmin)
        dy = max(1.0, ymax - ymin)

        if self._xlim is None or self._ylim is None:
            self._xlim = (xmin - pad * dx, xmax + pad * dx)
            self._ylim = (ymin - pad * dy, ymax + pad * dy)

        self.ax.set_xlim(*self._xlim)
        self.ax.set_ylim(*self._ylim)
        self.ax.legend(loc="upper right")

        self.canvas.draw_idle()

        # --- table
        for row in self.tree.get_children():
            self.tree.delete(row)

        # sort by RX name to keep stable order
        def rx_key(m): return m.get("rx", "")
        metrics_sorted = sorted(metrics, key=rx_key)

        for m in metrics_sorted:
            self.tree.insert(
                "",
                tk.END,
                values=(
                    m.get("rx", ""),
                    f"{m.get('snr_db', float('nan')):.2f}",
                    f"{m.get('corr_peak', float('nan')):.3f}",
                    f"{m.get('corr_norm', float('nan')):.3f}",
                    f"{m.get('psr_db', float('nan')):.2f}",
                ),
            )


if __name__ == "__main__":
    try:
        app = TDOAGUI()
        app.mainloop()
    except Exception as e:
        messagebox.showerror("Fatal error", str(e))
        raise
