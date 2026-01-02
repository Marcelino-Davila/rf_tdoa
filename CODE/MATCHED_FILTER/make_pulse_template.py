# make_pulse_template.py
# ------------------------------------------------------------
# Callable utilities for building a pulse template "s" used by:
#   SNR_max = s^H Rw^{-1} s
#
# Supports TWO modes:
#   (A) Build from your GNU Radio TX chain definition:
#       Vector symbols -> (interp FIR w/ RRC taps) -> *AMP -> Rotator(omega)
#
#   (B) Build from a captured .cfile (complex64 IQ):
#       - load samples
#       - optionally slice [start:start+N] OR auto-find best window
#       - optionally normalize
#
# Designed to be imported and called from main. No forced saving unless you call save/build helpers.
# ------------------------------------------------------------

from __future__ import annotations

import os
import numpy as np
from typing import Optional, Sequence, Tuple, Dict, Any

from scipy.signal import upfirdn  # OK to use; available in SciPy


# -------------------------
# Defaults (match your GRC screenshots)
# -------------------------

DEFAULT_SYMBOLS = [1, 1, 1, 1, 1, -1, -1, 1, 1, 1, -1, 1]
DEFAULT_OMEGA = 0.31416   # rad/sample (314.16 mrad)
DEFAULT_AMP = 0.25
DEFAULT_INTERP = 4

# These MUST match your GNU Radio RRC settings (edit if needed)
DEFAULT_RRC_BETA = 0.35
DEFAULT_RRC_SPAN = 11
DEFAULT_RRC_GAIN = 1.0


# =========================
# Helpers
# =========================

def rrc_taps(
    *,
    beta: float,
    sps: int,
    span: int,
    gain: float = 1.0,
) -> np.ndarray:
    """
    Root Raised Cosine (RRC) FIR taps (float64).

    Args:
      beta: roll-off (0 < beta <= 1)
      sps: samples per symbol (interp factor)
      span: filter span in symbols (total length = span*sps + 1)
      gain: overall gain scaling

    Returns:
      taps: float64 1D array length (span*sps + 1)

    Notes:
      Standard RRC impulse response sampled at Fs=sps (symbol period T=1).
      To match GNU Radio, ensure beta/span/gain match your flowgraph.
    """
    if not (0.0 < beta <= 1.0):
        raise ValueError("beta must be in (0, 1].")
    if sps < 1:
        raise ValueError("sps must be >= 1.")
    if span < 1:
        raise ValueError("span must be >= 1.")

    N = span * sps  # taps-1
    t = np.arange(-N / 2, N / 2 + 1, dtype=np.float64) / float(sps)  # in symbols (T=1)
    taps = np.zeros_like(t)

    for i, ti in enumerate(t):
        if abs(ti) < 1e-12:
            # t = 0
            taps[i] = (1.0 + beta * (4.0 / np.pi - 1.0))
        elif abs(abs(4.0 * beta * ti) - 1.0) < 1e-12:
            # t = ±T/(4β) special case
            a = (1.0 + 2.0 / np.pi) * np.sin(np.pi / (4.0 * beta))
            b = (1.0 - 2.0 / np.pi) * np.cos(np.pi / (4.0 * beta))
            taps[i] = (beta / np.sqrt(2.0)) * (a + b)
        else:
            num = (
                np.sin(np.pi * ti * (1.0 - beta)) +
                4.0 * beta * ti * np.cos(np.pi * ti * (1.0 + beta))
            )
            den = np.pi * ti * (1.0 - (4.0 * beta * ti) ** 2)
            taps[i] = num / den

    # normalize energy then apply gain
    taps = taps / np.sqrt(np.sum(taps ** 2))
    taps = taps * float(gain)
    return taps


def _load_rrc_taps(
    *,
    rrc_taps_arr: Optional[np.ndarray] = None,
    rrc_taps_npy: Optional[str] = None,
    beta: float = DEFAULT_RRC_BETA,
    sps: int = DEFAULT_INTERP,
    span: int = DEFAULT_RRC_SPAN,
    gain: float = DEFAULT_RRC_GAIN,
) -> np.ndarray:
    """
    Load taps from:
      - rrc_taps_arr (explicit array), OR
      - rrc_taps_npy (saved .npy), OR
      - generate using beta/sps/span/gain (default)
    """
    if rrc_taps_arr is not None:
        taps = np.asarray(rrc_taps_arr, dtype=np.float64).ravel()
        if taps.ndim != 1 or taps.size < 2:
            raise ValueError(f"RRC taps must be 1D with len>=2. Got shape={taps.shape}")
        return taps

    if rrc_taps_npy is not None:
        if not os.path.exists(rrc_taps_npy):
            raise FileNotFoundError(f"RRC taps file not found: {rrc_taps_npy}")
        taps = np.asarray(np.load(rrc_taps_npy), dtype=np.float64).ravel()
        if taps.ndim != 1 or taps.size < 2:
            raise ValueError(f"RRC taps must be 1D with len>=2. Got shape={taps.shape}")
        return taps

    return rrc_taps(beta=beta, sps=sps, span=span, gain=gain)


def _rotator(x_real: np.ndarray, omega: float) -> np.ndarray:
    n = np.arange(x_real.size, dtype=np.float64)
    return x_real * np.exp(1j * float(omega) * n)


def _ensure_len(x: np.ndarray, N: int) -> np.ndarray:
    if x.size == N:
        return x
    if x.size > N:
        return x[:N]
    out = np.zeros(N, dtype=x.dtype)
    out[: x.size] = x
    return out


def _normalize_power(x: np.ndarray, target_power: float = 1.0, eps: float = 1e-30) -> np.ndarray:
    p = float(np.mean(np.abs(x) ** 2))
    if p < eps:
        return x
    return x * np.sqrt(target_power / p)


def _load_cfile(path: str, *, max_samps: Optional[int] = None) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(f"cfile not found: {path}")
    x = np.fromfile(path, dtype=np.complex64)
    if max_samps is not None and x.size > max_samps:
        x = x[:max_samps]
    return x


def _auto_window_by_energy(x: np.ndarray, N: int, *, hop: int = 256) -> Tuple[int, np.ndarray]:
    """
    Finds the N-sample window with maximum energy using hop strides (fast).
    Returns (start_index, window).
    """
    if x.size < N:
        raise ValueError(f"Signal too short: len={x.size} < N={N}")

    best_i = 0
    best_e = -1.0

    for i in range(0, x.size - N + 1, hop):
        w = x[i:i+N]
        e = float(np.vdot(w, w).real)  # sum |w|^2
        if e > best_e:
            best_e = e
            best_i = i

    return best_i, x[best_i:best_i+N]


# =========================
# (A) Build template from GNU Radio TX definition
# =========================

def make_template_from_gnuradio(
    *,
    N: int,
    rrc_taps: Optional[np.ndarray] = None,
    rrc_taps_npy: Optional[str] = None,
    symbols: Sequence[float] = DEFAULT_SYMBOLS,
    interp: int = DEFAULT_INTERP,
    amp: float = DEFAULT_AMP,
    omega: float = DEFAULT_OMEGA,
    repeat_symbols: bool = True,
    normalize: bool = False,
    normalize_target_power: float = 1.0,
    # RRC params (edit these to match your GRC)
    rrc_beta: float = DEFAULT_RRC_BETA,
    rrc_span: int = DEFAULT_RRC_SPAN,
    rrc_gain: float = DEFAULT_RRC_GAIN,
) -> np.ndarray:
    """
    Reconstructs the complex baseband waveform feeding the USRP:

        symbols (float) -> interpolating FIR (interp, RRC taps)
                         -> multiply const (amp)
                         -> rotator (omega rad/sample)

    Returns:
        s (complex64) length N
    """
    taps = _load_rrc_taps(
        rrc_taps_arr=rrc_taps,
        rrc_taps_npy=rrc_taps_npy,
        beta=float(rrc_beta),
        sps=int(interp),
        span=int(rrc_span),
        gain=float(rrc_gain),
    )

    a = np.asarray(symbols, dtype=np.float64).ravel()
    if a.size < 1:
        raise ValueError("symbols must be non-empty")

    # ensure enough samples to cover N after filtering
    if repeat_symbols:
        reps = 1
        while (taps.size + interp * (reps * a.size) - 1) < (N + taps.size + interp * a.size):
            reps *= 2
        a_use = np.tile(a, reps)
    else:
        a_use = a

    # pulse shaping
    x = upfirdn(taps, a_use, up=int(interp)).astype(np.float64, copy=False)

    # multiply const
    x *= float(amp)

    # rotator -> complex
    s = _rotator(x, omega).astype(np.complex64, copy=False)

    # trim/pad
    s = _ensure_len(s, int(N)).astype(np.complex64, copy=False)

    if normalize:
        s = _normalize_power(s, target_power=float(normalize_target_power)).astype(np.complex64, copy=False)

    return s


# =========================
# (B) Build template from a .cfile capture
# =========================

def make_template_from_cfile(
    *,
    cfile_path: str,
    N: int,
    start: Optional[int] = None,
    auto_find: bool = True,
    auto_hop: int = 256,
    max_samps: Optional[int] = None,
    normalize: bool = False,
    normalize_target_power: float = 1.0,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Build template s from a captured complex64 .cfile.

    If start is provided:
        s = x[start:start+N]
    Else if auto_find=True:
        choose the N-sample window with max energy (strided by auto_hop)

    Returns:
        (s, info)
    """
    x = _load_cfile(cfile_path, max_samps=max_samps)

    if x.size < N:
        raise ValueError(f"cfile too short: {x.size} < N={N}")

    if start is not None:
        start_i = int(start)
        if start_i < 0 or (start_i + N) > x.size:
            raise ValueError(f"Invalid start={start_i} for len={x.size}, N={N}")
        s = x[start_i:start_i+N]
        info = {"mode": "slice", "start": start_i, "N": int(N), "len_file": int(x.size)}
    else:
        if not auto_find:
            start_i = 0
            s = x[:N]
            info = {"mode": "first", "start": start_i, "N": int(N), "len_file": int(x.size)}
        else:
            start_i, s = _auto_window_by_energy(x, N=int(N), hop=int(auto_hop))
            info = {"mode": "auto_energy", "start": int(start_i), "N": int(N), "len_file": int(x.size), "hop": int(auto_hop)}

    s = s.astype(np.complex64, copy=False)
    if normalize:
        s = _normalize_power(s, target_power=float(normalize_target_power)).astype(np.complex64, copy=False)

    return s, info


# =========================
# Convenience
# =========================

def save_template(path: str, s: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, np.asarray(s, dtype=np.complex64))


def build_and_save_template(
    *,
    out_path: str,
    N: int,
    mode: str = "gnuradio",  # "gnuradio" or "cfile"
    cfile_path: Optional[str] = None,
    cfile_start: Optional[int] = None,
    cfile_auto_find: bool = True,
    cfile_auto_hop: int = 256,
    normalize: bool = False,
    normalize_target_power: float = 1.0,
    # gnuradio params
    rrc_beta: float = DEFAULT_RRC_BETA,
    rrc_span: int = DEFAULT_RRC_SPAN,
    rrc_gain: float = DEFAULT_RRC_GAIN,
    interp: int = DEFAULT_INTERP,
    amp: float = DEFAULT_AMP,
    omega: float = DEFAULT_OMEGA,
    symbols: Sequence[float] = DEFAULT_SYMBOLS,
) -> np.ndarray:
    """
    One-call helper for main:
      - builds s
      - saves to out_path
      - returns s
    """
    mode = mode.lower().strip()
    if mode == "gnuradio":
        s = make_template_from_gnuradio(
            N=N,
            symbols=symbols,
            interp=interp,
            amp=amp,
            omega=omega,
            repeat_symbols=True,
            normalize=normalize,
            normalize_target_power=normalize_target_power,
            rrc_beta=rrc_beta,
            rrc_span=rrc_span,
            rrc_gain=rrc_gain,
        )
    elif mode == "cfile":
        if not cfile_path:
            raise ValueError("mode='cfile' requires cfile_path")
        s, info = make_template_from_cfile(
            cfile_path=cfile_path,
            N=N,
            start=cfile_start,
            auto_find=cfile_auto_find,
            auto_hop=cfile_auto_hop,
            normalize=normalize,
            normalize_target_power=normalize_target_power,
        )
    else:
        raise ValueError("mode must be 'gnuradio' or 'cfile'")

    save_template(out_path, s)
    return s


# =========================
# Optional CLI test
# =========================

if __name__ == "__main__":
    OUT = r"G:\GITFINAL\rf_tdoa\DATA\pulse_template_s.npy"
    N = 2000

    s = build_and_save_template(
        out_path=OUT,
        N=N,
        mode="gnuradio",
        normalize=False,
        # if your GRC RRC params differ, edit these:
        rrc_beta=DEFAULT_RRC_BETA,
        rrc_span=DEFAULT_RRC_SPAN,
        rrc_gain=DEFAULT_RRC_GAIN,
        interp=DEFAULT_INTERP,
        amp=DEFAULT_AMP,
        omega=DEFAULT_OMEGA,
        symbols=DEFAULT_SYMBOLS,
    )

    print(f"[i] saved: {OUT}  shape={s.shape} dtype={s.dtype}  mean(|s|^2)={np.mean(np.abs(s)**2):.6e}")
