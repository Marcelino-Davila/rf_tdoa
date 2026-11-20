"""Identical timing filters and optional noise-aware detection filters."""
from dataclasses import dataclass
from functools import lru_cache
import numpy as np
from scipy import signal, linalg, fft


@dataclass(frozen=True)
class Band:
    name: str
    center_hz: float
    bandwidth_hz: float
    decimation: int = 1
    taps: int = 129

    def validate(self, fs):
        if not self.name or not np.isfinite([self.center_hz, self.bandwidth_hz]).all():
            raise ValueError("invalid band")
        if not isinstance(self.decimation, int) or self.decimation < 1:
            raise ValueError("decimation must be a positive integer")
        if not isinstance(self.taps, int) or self.taps < 3 or self.taps % 2 != 1:
            raise ValueError("odd FIR length >= 3 required")
        if not 0 < self.bandwidth_hz < .8 * fs / self.decimation:
            raise ValueError("bandwidth needs transition space below output Nyquist")
        if abs(self.center_hz) + self.bandwidth_hz / 2 >= fs / 2:
            raise ValueError("band crosses input Nyquist")

    @lru_cache(maxsize=64)
    def coefficients(self, fs):
        self.validate(fs)
        h = signal.firwin(self.taps, self.bandwidth_hz / 2, fs=fs, window=("kaiser", 8.6))
        if self.decimation > 1:
            f, response = signal.freqz(h, worN=32768, fs=fs)
            stop = abs(response[f >= fs / (2 * self.decimation)])
            if not len(stop) or np.max(stop) > 1e-3:
                raise ValueError("FIR gives <60 dB anti-alias rejection; increase taps or reduce bandwidth")
        return h

    def apply(self, x, fs, start_sample=0):
        """Valid FIR removes edges; all receivers use exactly the same filter.

        Output sample k corresponds to input time (start_sample + delay + k*D)/fs.
        Mixing uses the shared sample index. Keep center/rate constant per epoch.
        """
        h = self.coefficients(fs)
        if len(x) < len(h):
            raise ValueError("window shorter than FIR")
        phase = 2 * np.pi * self.center_hz / fs * (start_sample + np.arange(len(x)))
        y = signal.fftconvolve(x * np.exp(-1j * phase), h, mode="valid")
        return y[::self.decimation], fs / self.decimation, (self.taps - 1) / (2 * fs)


class StreamingFIR:
    """FFT overlap-save with one output per input, independent of chunk boundaries.

    Startup L-1 outputs contain zero history and must be excluded from timing.
    No partial block is silently dropped. flush_tail closes the finite stream.
    """
    def __init__(self, h, block_size=8192):
        self.h = np.asarray(h, complex)
        if self.h.ndim != 1 or not len(self.h) or not np.isfinite(self.h).all() or block_size < 1:
            raise ValueError("invalid FIR")
        self.nfft = fft.next_fast_len(block_size + len(self.h) - 1)
        self.block_size = self.nfft - len(self.h) + 1
        self.H = fft.fft(self.h, self.nfft)
        self.history = np.zeros(len(self.h) - 1, complex)
        self.closed = False

    def feed(self, x):
        if self.closed:
            raise ValueError("filter already closed")
        x = np.asarray(x, complex)
        if x.ndim != 1 or not np.isfinite(x).all():
            raise ValueError("invalid FIR input")
        out = []
        for start in range(0, len(x), self.block_size):
            take = x[start:start + self.block_size]
            block = np.concatenate((self.history, take))
            y = fft.ifft(fft.fft(block, self.nfft) * self.H)
            out.append(y[len(self.h) - 1:len(self.h) - 1 + len(take)])
            if len(self.history):
                self.history = block[-len(self.history):].copy()
        return np.concatenate(out) if out else np.empty(0, complex)

    def flush_tail(self):
        y = self.feed(np.zeros(len(self.h) - 1, complex))
        self.closed = True
        return y


def noise_covariance(noise, length, stride=None, batch_size=256):
    """E[w w^H] for chronological complex snapshots, with global mean removed.

    Batches bound temporary storage. Overlapping snapshots are correlated and
    should not be counted as independent training examples.
    """
    x = np.asarray(noise, complex)
    if x.ndim != 1 or not np.isfinite(x).all() or not 1 <= length <= len(x):
        raise ValueError("invalid noise/template length")
    stride = length if stride is None else stride
    if stride < 1 or batch_size < 1:
        raise ValueError("positive stride and batch size required")
    x = x - x.mean()
    windows = np.lib.stride_tricks.sliding_window_view(x, length)[::stride]
    R = np.zeros((length, length), complex)
    for a in range(0, len(windows), batch_size):
        W = windows[a:a + batch_size]
        R += W.T @ W.conj()  # NOT W.conj().T @ W
    return R / len(windows)


def matched_filter(template, covariance=None, loading=1e-3):
    """Return convolution taps h=conj(reverse(g)), g=(R+lambda I)^-1 s.

    Unit template response. Covariance is chronological E[w w^H]. Loading is
    relative to mean diagonal power. Detection use only; do not independently
    whiten receiver timing paths without calibrating their phase responses.
    """
    s = np.asarray(template, complex)
    if s.ndim != 1 or not len(s) or not np.isfinite(s).all() or np.vdot(s, s).real <= 0:
        raise ValueError("invalid template")
    if not np.isfinite(loading) or loading < 0:
        raise ValueError("invalid diagonal loading")
    if covariance is None:
        g = s.copy()
    else:
        R = np.asarray(covariance, complex)
        if R.shape != (len(s), len(s)) or not np.isfinite(R).all():
            raise ValueError("covariance shape/data mismatch")
        if not np.allclose(R, R.conj().T):
            raise ValueError("covariance must be Hermitian")
        power = np.trace(R).real / len(s)
        if power <= 0:
            raise ValueError("noise power must be positive")
        R = R + loading * power * np.eye(len(s))
        linalg.cholesky(R)  # fail explicitly if model is not positive definite
        g = linalg.solve(R, s, assume_a="pos")
    g /= np.vdot(s, g).real
    return g[::-1].conj()


def detection_score(x, taps):
    """Peak/median MF power ratio, explicitly NOT calibrated SNR or CFAR."""
    if len(x) < len(taps):
        raise ValueError("detection window shorter than template")
    y = signal.fftconvolve(x, taps, mode="valid")
    power = abs(y)**2
    return float(10 * np.log10((power.max() + 1e-30) / (np.median(power) + 1e-30)))
