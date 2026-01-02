import os
import glob
import numpy as np
import scipy.linalg
from typing import Dict, Optional, Tuple, Iterable

RX_NAMES = ("rx0", "rx1", "rx2")


# =========================
# Helpers
# =========================

def _safe_makedirs(p: str) -> None:
    if p and not os.path.exists(p):
        os.makedirs(p, exist_ok=True)


def _load_matrix(path: str) -> np.ndarray:
    R = np.load(path)
    if R.ndim != 2 or R.shape[0] != R.shape[1]:
        raise ValueError(f"Matrix must be square: {path} has shape {R.shape}")
    return R.astype(np.complex64, copy=False)


def _load_vector(path: str) -> np.ndarray:
    v = np.load(path)
    v = np.asarray(v)
    if v.ndim != 1:
        raise ValueError(f"Vector must be 1D: {path} has shape {v.shape}")
    return v


def _infer_rx_from_name(path: str) -> Optional[str]:
    base = os.path.basename(path)
    for rx in RX_NAMES:
        if rx in base:
            return rx
    return None


# =========================
# 1) THEORETICAL MAX SNR (UPPER BOUND)
#    SNR_max = s^H Rw^{-1} s   (circled equation)
# =========================

def compute_max_snr_linear(
    s: np.ndarray,
    Rw: np.ndarray,
    *,
    assume_hermitian: bool = True,
    diag_load: float = 0.0,
) -> float:
    """
    Returns SNR_max (linear) using the circled upper-bound equation:

        SNR_max = s^H * Rw^{-1} * s

    Implementation detail:
      - We do NOT form Rw^{-1} explicitly.
      - We solve: Rw * x = s
      - Then SNR_max = s^H x

    Inputs:
      - s: complex template length N
      - Rw: NxN noise covariance
      - diag_load: optional diagonal loading epsilon (adds epsilon*I)
    """
    s = np.asarray(s)
    if s.ndim != 1:
        raise ValueError("s must be 1D")
    N = s.size

    if Rw.shape != (N, N):
        raise ValueError(f"Rw shape {Rw.shape} does not match N={N}")

    R = Rw.astype(np.complex128, copy=False)
    if diag_load > 0:
        R = R + (diag_load * np.eye(N, dtype=np.complex128))

    # Solve Rw * x = s  -> x = Rw^{-1} s
    x = scipy.linalg.solve(
        R,
        s.astype(np.complex128, copy=False),
        assume_a="her" if assume_hermitian else "gen",
    )

    # SNR_max = s^H * x
    snr_lin = np.vdot(s.astype(np.complex128, copy=False), x)

    return float(np.real(snr_lin))


def compute_max_snr_db(
    s: np.ndarray,
    Rw: np.ndarray,
    **kwargs
) -> float:
    lin = compute_max_snr_linear(s, Rw, **kwargs)
    if lin <= 0:
        return float("-inf")
    return float(10.0 * np.log10(lin))


def compute_max_snr_all_receivers(
    *,
    s: np.ndarray,
    rw_paths: Dict[str, str],
    diag_load: float = 0.0,
    assume_hermitian: bool = True,
    save_dir: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Dict[str, float]]:
    """
    Compute theoretical SNR_max for rx0/rx1/rx2 given:
      - s: complex template (length >= N)
      - rw_paths: dict like {"rx0": "...Rw_rx0.npy", ...}

    Optionally saves:
      max_theoretical_snr_linear_rx0.npy
      max_theoretical_snr_db_rx0.npy
    """
    if save_dir:
        _safe_makedirs(save_dir)

    results: Dict[str, Dict[str, float]] = {}
    for rx, path in rw_paths.items():
        Rw = _load_matrix(path)

        # Match dimensions automatically: truncate s if needed
        N = Rw.shape[0]
        if s.size < N:
            raise ValueError(f"s length {s.size} < required N={N} for {rx}")
        s_use = s[:N].astype(np.complex64, copy=False)

        lin = compute_max_snr_linear(
            s_use,
            Rw,
            assume_hermitian=assume_hermitian,
            diag_load=diag_load,
        )
        db = float(10.0 * np.log10(lin)) if lin > 0 else float("-inf")
        results[rx] = {"snr_linear": float(lin), "snr_db": float(db), "N": float(N)}

        if verbose:
            print(f"[i] {rx}: SNR_max = {db:.2f} dB (linear={lin:.4e})  N={N}")

        if save_dir:
            np.save(os.path.join(save_dir, f"max_theoretical_snr_linear_{rx}.npy"), np.array(lin, dtype=np.float64))
            np.save(os.path.join(save_dir, f"max_theoretical_snr_db_{rx}.npy"), np.array(db, dtype=np.float64))

    return results


# =========================
# 2) EMPIRICAL SNR from MF output (stream or file)
#    SNR_est = 10log10( peak_power / noise_power )
# =========================

def estimate_snr_from_mf_mag(
    mag: np.ndarray,
    *,
    guard: int = 100,
    noise_floor_mode: str = "median",
) -> Dict[str, float]:
    """
    Estimate SNR from matched filter magnitude output.

    We find peak at index k*, then estimate noise power from samples
    outside a +/- guard window around peak.

    Returns dict:
      peak_idx, peak_mag, noise_power, snr_db
    """
    mag = np.asarray(mag)
    if mag.ndim != 1 or mag.size < (2 * guard + 10):
        raise ValueError("mag must be 1D and reasonably long")

    k = int(np.argmax(mag))
    peak = float(mag[k])

    # noise region excludes peak neighborhood
    mask = np.ones(mag.size, dtype=bool)
    lo = max(0, k - guard)
    hi = min(mag.size, k + guard + 1)
    mask[lo:hi] = False
    noise = mag[mask]

    # power estimate from magnitude -> power ~ mag^2
    p = noise.astype(np.float64) ** 2

    if noise_floor_mode == "median":
        noise_power = float(np.median(p))
    elif noise_floor_mode == "mean":
        noise_power = float(np.mean(p))
    else:
        raise ValueError("noise_floor_mode must be 'median' or 'mean'")

    peak_power = float(peak ** 2)
    snr_lin = peak_power / max(noise_power, 1e-30)
    snr_db = float(10.0 * np.log10(snr_lin))

    return {
        "peak_idx": float(k),
        "peak_mag": float(peak),
        "noise_power": float(noise_power),
        "snr_db": snr_db,
    }


def estimate_snr_streaming(
    *,
    mf_mag_chunk_iter: Iterable[Dict[str, np.ndarray]],
    receivers: Tuple[str, ...] = RX_NAMES,
    window_samples: int = 200_000,
    guard: int = 100,
    noise_floor_mode: str = "median",
    verbose: bool = True,
) -> Dict[str, Dict[str, float]]:
    """
    Consumes matched-filter magnitude chunks in a stream until it has
    ~window_samples per rx, then estimates SNR per rx.

    mf_mag_chunk_iter yields dict: {"rx0": mag_chunk, "rx1": mag_chunk, ...}

    Returns results dict per rx.
    """
    buf = {rx: [] for rx in receivers}
    got = {rx: 0 for rx in receivers}

    for pkt in mf_mag_chunk_iter:
        for rx in receivers:
            if rx in pkt and pkt[rx] is not None:
                x = np.asarray(pkt[rx], dtype=np.float32)
                buf[rx].append(x)
                got[rx] += x.size

        if all(got[rx] >= window_samples for rx in receivers):
            break

    results = {}
    for rx in receivers:
        mag = np.concatenate(buf[rx])[:window_samples]
        r = estimate_snr_from_mf_mag(mag, guard=guard, noise_floor_mode=noise_floor_mode)
        results[rx] = r
        if verbose:
            print(f"[i] {rx}: SNR_est={r['snr_db']:.2f} dB  peak_idx={int(r['peak_idx'])}  peak_mag={r['peak_mag']:.3g}")

    return results


# =========================
# Convenience: build rw_paths from a folder/pattern
# =========================

def find_rw_paths(
    *,
    rw_dir_or_pattern: str,
    receivers: Tuple[str, ...] = RX_NAMES,
) -> Dict[str, str]:
    """
    Accepts:
      - a folder containing Rw_rx0.npy etc, OR
      - a glob pattern like ".../Rw_*.npy"

    Returns dict rx -> path
    """
    paths: Dict[str, str] = {}

    if os.path.isdir(rw_dir_or_pattern):
        files = glob.glob(os.path.join(rw_dir_or_pattern, "*.npy"))
    else:
        files = glob.glob(rw_dir_or_pattern)

    for f in files:
        rx = _infer_rx_from_name(f)
        if rx and rx in receivers:
            paths[rx] = f

    missing = [rx for rx in receivers if rx not in paths]
    if missing:
        raise FileNotFoundError(f"Missing Rw for {missing} in {rw_dir_or_pattern}")

    return paths
