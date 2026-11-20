"""Complex generalized cross-correlation and conservative measurement fusion."""
from dataclasses import asdict, dataclass
import numpy as np
from scipy import fft, signal


@dataclass
class DelayEstimate:
    delay_s: float
    delay_samples: float
    grid_step_s: float
    normalized_correlation: float
    peak_to_sidelobe_db: float
    rms_bandwidth_hz: float
    boundary_peak: bool
    accepted: bool
    reasons: list

    def to_dict(self):
        return asdict(self)


def estimate_delay(reference, target, fs, max_delay_s, interpolation=32,
                   phat_exponent=.5, min_correlation=.2, min_pslr_db=3,
                   min_rms_bandwidth_hz=1e4):
    """Positive delay means target arrives LATER than reference.

    Linear correlation uses FFT(target)*conj(FFT(reference)). Complex magnitude
    handles unknown constant receiver phase. A local inverse DFT evaluates the
    continuous bandlimited correlation near the best coarse peak. This avoids
    upsampling an entire capture. The grid is not an accuracy claim.
    """
    x, y = np.asarray(reference, complex), np.asarray(target, complex)
    if x.ndim != 1 or x.shape != y.shape or len(x) < 32:
        raise ValueError("equal 1-D windows of >=32 samples required")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("nonfinite IQ")
    if not np.isfinite([fs, max_delay_s, phat_exponent, min_correlation,
                        min_pslr_db, min_rms_bandwidth_hz]).all():
        raise ValueError("nonfinite delay configuration")
    if fs <= 0 or max_delay_s <= 0 or not 0 <= phat_exponent <= 1:
        raise ValueError("invalid delay configuration")
    if not isinstance(interpolation, int) or not 1 <= interpolation <= 256:
        raise ValueError("interpolation must be integer 1..256")
    maxlag = max_delay_s * fs
    if maxlag > len(x) / 4:
        raise ValueError("search lag exceeds quarter window; align capture starts first")
    x, y = x - x.mean(), y - y.mean()
    # Same taper on all receivers; search lag should remain small vs window.
    w = signal.windows.tukey(len(x), .1)
    x, y = x * w, y * w
    norm = float(np.linalg.norm(x) * np.linalg.norm(y))
    if norm <= 1e-25:
        raise ValueError("zero-energy window")
    nfft = fft.next_fast_len(2 * len(x) - 1)
    X, Y = fft.fft(x, nfft), fft.fft(y, nfft)
    cross = Y * X.conj()
    amplitude = abs(cross)
    # Relative regularization avoids PHAT amplification of empty stopbands.
    floor = max(float(amplitude.max()) * 1e-3, 1e-30)
    G = cross / np.maximum(amplitude, floor)**phat_exponent
    G[amplitude < floor] = 0
    freq = fft.fftfreq(nfft)
    cc = fft.ifft(G)
    # Include neighboring integer peaks even for a sub-sample physical search.
    span = int(np.ceil(maxlag)) + 1
    lags = np.arange(-span, span + 1)
    coarse = abs(cc[lags % nfft])
    k = int(lags[np.argmax(coarse)])
    # Search the requested interval; always cover [-maxlag,maxlag] if <2 samples.
    left, right = max(-maxlag, k - 1.5), min(maxlag, k + 1.5)
    if left >= right:
        left, right = -maxlag, maxlag
    grid = np.linspace(left, right, max(3, int(np.ceil((right - left) * interpolation)) + 1))
    values = np.empty(len(grid), complex)
    for a in range(0, len(grid), 32):
        values[a:a+32] = np.exp(2j * np.pi * grid[a:a+32, None] * freq) @ G / nfft
    mag = abs(values)
    peak = int(np.argmax(mag))
    lag = float(grid[peak])
    # Correct three-point vertex sign; refinement stays inside the local grid.
    if 0 < peak < len(grid) - 1:
        u, v, z = mag[peak-1:peak+2]
        den = u - 2 * v + z
        if den < -1e-30:
            lag += float(np.clip(.5 * (u-z) / den, -.5, .5)) * (grid[1]-grid[0])
    raw_peak = abs(np.sum(cross * np.exp(2j * np.pi * freq * lag)) / nfft)
    rho = float(np.clip(raw_peak / norm, 0, 1))
    power = amplitude.copy()
    power[power < floor] = 0
    f_hz = freq * fs
    center = float(np.sum(power * f_hz) / power.sum())
    rms = float(np.sqrt(np.sum(power * (f_hz-center)**2) / power.sum()))
    # Diagnose ambiguity beyond the physical search window as well: constraining
    # a periodic waveform to a tiny interval must not manufacture confidence.
    diagnostic_span = len(x)//4
    diagnostic_lags = np.arange(-diagnostic_span, diagnostic_span+1)
    diagnostic = abs(cc[diagnostic_lags % nfft])
    guard = max(2., fs / max(2 * rms, 1e-30))
    side = diagnostic[abs(diagnostic_lags - lag) > guard]
    pslr = float(20 * np.log10((mag[peak] + 1e-30) / (side.max() + 1e-30))) if len(side) else 0.
    boundary = bool(abs(lag) >= maxlag - .51 / interpolation)
    reasons = []
    if boundary:
        reasons.append("peak_at_search_boundary")
    if rho < min_correlation:
        reasons.append("low_normalized_correlation")
    if pslr < min_pslr_db:
        reasons.append("ambiguous_correlation_peak")
    if rms < min_rms_bandwidth_hz:
        reasons.append("insufficient_signal_bandwidth")
    return DelayEstimate(lag / fs, lag, float((grid[1]-grid[0])/fs), rho,
                         pslr, rms, boundary, not reasons, reasons)


def robust_fuse(values_s, minimum=3, floor_s=1e-9):
    """Median/MAD aggregation; scatter is repeatability, NOT absolute accuracy.

    Inputs must be one estimate per nonoverlapping time window. Frequency
    branches and overlapping windows are not independent trials.
    """
    a = np.asarray(values_s, float)
    if a.ndim != 1 or not np.isfinite(a).all() or minimum < 1 or floor_s <= 0:
        raise ValueError("invalid fusion inputs")
    if not len(a):
        return {"accepted": False, "reason": "no_valid_windows", "count": 0}
    center = float(np.median(a))
    mad = float(1.4826 * np.median(abs(a-center)))
    keep = abs(a-center) <= 3 * max(mad, floor_s)
    inliers = a[keep]
    return {"accepted": bool(len(inliers) >= minimum),
            "reason": "ok" if len(inliers) >= minimum else "too_few_valid_windows",
            "delay_s": float(np.median(inliers)),
            "repeatability_mad_s": float(1.4826*np.median(abs(inliers-np.median(inliers)))),
            "count": int(len(inliers)), "rejected_outliers": int((~keep).sum()),
            "accuracy_claim": None}
