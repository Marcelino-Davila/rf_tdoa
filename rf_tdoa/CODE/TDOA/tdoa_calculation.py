# TDOA/tdoa_estimator.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
import numpy as np


def _parabolic_subsample_peak(y: np.ndarray, i: int) -> float:
    """
    Sub-sample peak refinement via a parabola through (i-1, i, i+1).
    Returns refined peak index as float. Falls back to integer index at edges.
    """
    if i <= 0 or i >= (len(y) - 1):
        return float(i)

    y0 = float(y[i - 1])
    y1 = float(y[i])
    y2 = float(y[i + 1])
    denom = (y0 - 2.0 * y1 + y2)
    if abs(denom) < 1e-24:
        return float(i)

    # Vertex offset relative to i
    delta = 0.5 * (y0 - y2) / denom
    # Clamp to avoid wild results from noisy local curvature
    delta = float(np.clip(delta, -1.0, 1.0))
    return float(i) + delta


def peak_index_from_mag(mag: np.ndarray) -> Optional[float]:
    """
    Returns refined peak index (float) from a magnitude array.
    """
    if mag is None or len(mag) < 3:
        return None
    i = int(np.argmax(mag))
    return _parabolic_subsample_peak(mag, i)


@dataclass
class TDOAEstimator:
    """
    Streaming-friendly TDOA estimator.

    Inputs:
      - mags_by_rx: dict { "rx0": mag_array, "rx1": mag_array, ... }
        where mag_array is the matched-filter magnitude for that chunk.

    Output:
      - dict { "rx1": (delta_samples, delta_seconds), ... } relative to ref_rx

    Notes:
      - Positive delta means that receiver peak occurs LATER than the reference.
      - EMA smoothing is optional, and only affects returned delta_samples/time.
    """
    fs_hz: float
    ref_rx: str = "rx0"
    ema_alpha: float = 1.0  # 1.0 = no smoothing, 0.2 = smooth, etc.

    _ema_state: Dict[str, Optional[float]] = field(default_factory=dict)

    def update(self, mags_by_rx: Dict[str, np.ndarray]) -> Dict[str, Tuple[float, float]]:
        # 1) Extract peaks
        peaks: Dict[str, float] = {}
        for rx, mag in mags_by_rx.items():
            p = peak_index_from_mag(mag)
            if p is not None:
                peaks[rx] = p

        if self.ref_rx not in peaks:
            return {}

        ref = peaks[self.ref_rx]

        # 2) Compute relative offsets (and optional EMA smoothing)
        out: Dict[str, Tuple[float, float]] = {}
        for rx, p in peaks.items():
            if rx == self.ref_rx:
                continue

            dsamp = float(p - ref)

            # init state if missing
            if rx not in self._ema_state:
                self._ema_state[rx] = None

            # EMA smoothing
            prev = self._ema_state[rx]
            if prev is None or self.ema_alpha >= 1.0:
                smoothed = dsamp
            else:
                smoothed = (self.ema_alpha * dsamp) + ((1.0 - self.ema_alpha) * prev)

            self._ema_state[rx] = smoothed

            dsec = smoothed / float(self.fs_hz)
            out[rx] = (smoothed, dsec)

        return out
