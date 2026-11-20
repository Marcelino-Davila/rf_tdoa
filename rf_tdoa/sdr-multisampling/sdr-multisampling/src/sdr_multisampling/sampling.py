"""Explicit sample clocks, bounded file access, and fractional-grid alignment."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np

MARKER_SYNC = bytes.fromhex("7f807f8000ff00ff55aa55aa")


@dataclass(frozen=True)
class Clock:
    """t(n) = epoch_s + n / rate_hz, relative to a shared run epoch.

    This describes ADC sample times, NOT network receive timestamps.
    The epoch/rate must come from independently justified calibration.
    """
    rate_hz: float
    epoch_s: float = 0.0

    def __post_init__(self):
        if not np.isfinite(self.rate_hz) or self.rate_hz <= 0:
            raise ValueError("clock rate must be positive and finite")
        if not np.isfinite(self.epoch_s):
            raise ValueError("clock epoch must be finite")


def fit_clock(indices, times_s, nominal_rate_hz, max_residual_s):
    """Fit sample-latched anchors only; reject jitter rather than hide it.

    Do not pass USB-block marker offsets as precise ADC anchors. Counter resets
    and wraps must first be resolved by the capture adapter.
    """
    n, t = np.asarray(indices, float), np.asarray(times_s, float)
    if n.ndim != 1 or n.shape != t.shape or len(n) < 3:
        raise ValueError("need at least three paired anchors")
    if not np.all(np.isfinite(n)) or not np.all(np.isfinite(t)):
        raise ValueError("nonfinite clock anchors")
    if np.any(np.diff(n) <= 0) or np.any(np.diff(t) <= 0):
        raise ValueError("anchors must increase without resets")
    if nominal_rate_hz <= 0 or max_residual_s <= 0:
        raise ValueError("positive nominal rate and residual limit required")
    slope, intercept = np.polyfit(n - n[0], t - t[0], 1)
    predicted = t[0] + intercept + slope * (n - n[0])
    residual = t - predicted
    if slope <= 0 or np.max(np.abs(residual)) > max_residual_s:
        raise ValueError("clock anchors exceed timing residual budget")
    clock = Clock(1 / slope, t[0] + intercept - slope * n[0])
    return clock, {"max_residual_s": float(np.max(np.abs(residual))),
                   "rate_error_ppm": float((clock.rate_hz / nominal_rate_hz - 1) * 1e6)}


class IQFile:
    """Memory-mapped cf32_le or native interleaved signed int8 IQ.

    Invalid ranges are half-open ORIGINAL sample indices; gaps must not be
    deleted. Embedded firmware markers are rejected when encountered, because
    whether they replace or insert samples cannot be inferred safely here.
    """
    def __init__(self, path, clock, fmt="cf32_le", invalid_ranges=()):
        self.path, self.clock, self.fmt = Path(path), clock, fmt
        if fmt not in ("cf32_le", "ci8"):
            raise ValueError("format must be cf32_le or ci8")
        stride = 8 if fmt == "cf32_le" else 2
        size = self.path.stat().st_size
        if not size or size % stride:
            raise ValueError("empty file or incomplete IQ sample")
        self.count = size // stride
        self.raw = np.memmap(self.path, mode="r", dtype="<c8" if stride == 8 else "i1")
        self.invalid_ranges = []
        for lo, hi in invalid_ranges:
            if not (isinstance(lo, int) and isinstance(hi, int) and 0 <= lo < hi <= self.count):
                raise ValueError("invalid sample range")
            self.invalid_ranges.append((lo, hi))

    def read(self, start, count):
        if start < 0 or count < 1 or start + count > self.count:
            raise ValueError("window outside capture")
        if any(start < hi and start + count > lo for lo, hi in self.invalid_ranges):
            raise ValueError("window intersects a documented gap/invalid range")
        if self.fmt == "cf32_le":
            x = np.asarray(self.raw[start:start + count], dtype=np.complex128)
            # Detect the legacy marker after a float conversion as well.
            signature = np.frombuffer(MARKER_SYNC, dtype=np.int8).astype(float)
            for scale in (1.0, 1 / 128, 1 / 127):
                iq = signature[::2] + 1j * signature[1::2]
                hits = np.flatnonzero(np.isclose(x, iq[0] * scale, rtol=0, atol=1e-7))
                if any(k + 6 <= len(x) and np.allclose(x[k:k+6], iq * scale,
                                                       rtol=0, atol=1e-7) for k in hits):
                    raise ValueError("converted PPS marker in IQ: normalize firmware format first")
        else:
            # Include 15 neighboring bytes so marker overlaps at slice edges are caught.
            a, b = max(0, 2 * start - 15), min(len(self.raw), 2 * (start + count) + 15)
            if MARKER_SYNC in self.raw[a:b].tobytes():
                raise ValueError("embedded PPS marker: normalize firmware format first")
            v = self.raw[2 * start:2 * (start + count)].astype(float).reshape(-1, 2)
            x = (v[:, 0] + 1j * v[:, 1]) / 128
        if not np.all(np.isfinite(x)):
            raise ValueError("nonfinite IQ")
        return x


def aligned_window(source, start_s, count, output_rate_hz, half_width=24):
    """Windowed-sinc interpolation onto a common grid, including clock drift.

    This clock correction only permits ±1% rate difference. Substantial rate
    reduction belongs in the anti-aliased channelizer, not this interpolator.
    Reads only the required support. No extrapolation or silent zero fill.
    """
    if count < 1 or half_width < 4 or output_rate_hz <= 0:
        raise ValueError("invalid alignment parameters")
    ratio = source.clock.rate_hz / output_rate_hz
    if not .99 <= ratio <= 1.01:
        raise ValueError("clock correction exceeds 1%; resample acquisition separately")
    origin = (start_s - source.clock.epoch_s) * source.clock.rate_hz
    positions = origin + np.arange(count) * ratio
    if ratio == 1 and abs(origin - round(origin)) < 1e-7:
        return source.read(int(round(origin)), count)
    low = int(np.floor(positions[0])) - half_width + 1
    high = int(np.floor(positions[-1])) + half_width + 1
    x = source.read(low, high - low)
    out = np.empty(count, complex)
    offsets = np.arange(-half_width + 1, half_width + 1)
    # Bounded matrix size regardless of file duration.
    for a in range(0, count, 2048):
        p = positions[a:a + 2048]
        ix = np.floor(p).astype(np.int64)[:, None] + offsets
        d = p[:, None] - ix
        w = np.sinc(d) * np.i0(8.6 * np.sqrt(np.maximum(0, 1 - (d / half_width)**2))) / np.i0(8.6)
        w /= w.sum(axis=1, keepdims=True)
        out[a:a + len(p)] = np.sum(x[ix - low] * w, axis=1)
    return out


class ContiguousBuffer:
    """Adapter contract for live blocks; rejects drops/reordering before DSP.

    One buffer per receiver. Consumers remove processed samples via pop().
    Reset after a discontinuity only when a new timing epoch is established.
    """
    def __init__(self, max_samples=1_000_000):
        self.max_samples = max_samples
        self.next_index = None
        self.data = np.empty(0, complex)
        self.first_index = None

    def feed(self, first_index, samples):
        x = np.asarray(samples, complex)
        if not isinstance(first_index, (int, np.integer)) or first_index < 0:
            raise ValueError("absolute nonnegative sample index required")
        if x.ndim != 1 or not x.size or not np.all(np.isfinite(x)):
            raise ValueError("invalid IQ block")
        if self.next_index is not None and first_index != self.next_index:
            raise ValueError("sample discontinuity: drop, duplicate, reorder, or reset")
        if len(self.data) + len(x) > self.max_samples:
            raise BufferError("consumer fell behind; timing epoch must be re-established")
        if not len(self.data):
            self.first_index = first_index
        self.data = np.concatenate((self.data, x))
        self.next_index = first_index + len(x)

    def pop(self, count):
        if not 0 < count <= len(self.data):
            raise ValueError("insufficient buffered samples")
        index, x = self.first_index, self.data[:count].copy()
        self.data = self.data[count:]
        self.first_index += count
        return index, x
