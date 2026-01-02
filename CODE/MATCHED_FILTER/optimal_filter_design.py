import os
import glob
import numpy as np
import scipy.linalg
from typing import Dict, Any, Optional, List, Tuple


def _infer_receiver_name_from_filename(path: str) -> str:
    """
    Tries to map filenames like:
      noise_covariance_Rw_rx0.npy  -> rx0
      Rw_rx1.npy                   -> rx1
      noise_covariance_Rw0.npy     -> rx0 (legacy)
      noise_covariance_Rw1.npy     -> rx1
    Falls back to basename without extension.
    """
    base = os.path.basename(path)
    stem = os.path.splitext(base)[0]

    # common cases: "...rx0" / "..._rx0"
    for rx in ("rx0", "rx1", "rx2"):
        if stem.endswith(rx) or f"_{rx}" in stem:
            return rx

    # legacy: "...Rw0" / "...Rw1"
    if "Rw0" in stem or stem.endswith("Rw0"):
        return "rx0"
    if "Rw1" in stem or stem.endswith("Rw1"):
        return "rx1"
    if "Rw2" in stem or stem.endswith("Rw2"):
        return "rx2"

    return stem


def _resolve_matrix_files(
    matrix_path_or_pattern: str,
    *,
    prefer_rx: Optional[str] = None,
) -> List[str]:
    """
    Accepts:
      - a direct file path to .npy
      - a glob pattern
      - a directory (loads all *.npy within)

    Returns a sorted list of files.
    """
    p = matrix_path_or_pattern

    if os.path.isdir(p):
        files = sorted(glob.glob(os.path.join(p, "*.npy")))
    elif os.path.isfile(p):
        files = [p]
    else:
        files = sorted(glob.glob(p))

    # optional filter to only one receiver
    if prefer_rx is not None:
        files = [f for f in files if prefer_rx in os.path.basename(f)]

    return files


def design_optimal_filters_from_covariances(
    *,
    s: np.ndarray,
    matrices: str,
    output_dir: Optional[str] = None,
    receiver_filter: Optional[str] = None,
    require_shape_match: bool = True,
    verbose: bool = True,
) -> Tuple[Dict[str, np.ndarray], List[Dict[str, Any]]]:
    """
    Designs optimal filter(s) g = Rw^{-1} s for one or more receivers.

    Args:
      s: complex pulse vector, shape (N,)
      matrices: file path OR glob pattern OR directory containing covariance .npy files
      output_dir: if provided, saves g vectors as .npy
      receiver_filter: if provided (e.g. "rx0"), only process matching files
      require_shape_match: if True, skip matrices not (N,N)
      verbose: prints status

    Returns:
      (g_by_receiver, results_list)

      g_by_receiver: {"rx0": g0, "rx1": g1, ...}
      results_list: list of dicts with SNR and file info

    Raises:
      FileNotFoundError if no matrix files found.
      ValueError if s is invalid.
    """
    if s is None:
        raise ValueError("s must not be None")
    s = np.asarray(s)
    if s.ndim != 1:
        raise ValueError("s must be a 1D vector")
    if not np.iscomplexobj(s):
        s = s.astype(np.complex64, copy=False)

    N = int(s.size)
    if N <= 0:
        raise ValueError("s length must be > 0")

    files = _resolve_matrix_files(matrices, prefer_rx=receiver_filter)
    if not files:
        raise FileNotFoundError(f"No covariance matrices found for: {matrices}")

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    if verbose:
        print(f"[i] Designing optimal filters from {len(files)} matrix file(s)")
        print(f"[i] N={N} (pulse length)")

    g_by_rx: Dict[str, np.ndarray] = {}
    results: List[Dict[str, Any]] = []

    for fpath in files:
        rx = _infer_receiver_name_from_filename(fpath)
        if verbose:
            print(f"\n[i] Loading Rw for {rx}: {fpath}")

        try:
            Rw = np.load(fpath)
        except Exception as e:
            if verbose:
                print(f"  [error] load failed: {e}")
            continue

        if require_shape_match and Rw.shape != (N, N):
            if verbose:
                print(f"  [skip] shape {Rw.shape} != ({N},{N})")
            continue

        try:
            # Solve Rw g = s (Hermitian assumption for speed/stability)
            g = scipy.linalg.solve(Rw, s, assume_a="her")

            # Max SNR = s^H g
            snr_lin = np.vdot(s, g)  # complex scalar
            snr_lin_real = float(np.real(snr_lin))
            snr_db = 10.0 * np.log10(max(snr_lin_real, 1e-30))

            g_by_rx[rx] = g

            outpath = None
            if output_dir is not None:
                outpath = os.path.join(output_dir, f"optimal_filter_g_{rx}.npy")
                np.save(outpath, g)

            results.append({
                "receiver": rx,
                "matrix_file": fpath,
                "N": N,
                "SNR_max_linear": snr_lin_real,
                "SNR_max_dB": snr_db,
                "filter_file": outpath,
            })

            if verbose:
                if outpath:
                    print(f"  [ok] SNR_max={snr_db:.2f} dB | saved: {outpath}")
                else:
                    print(f"  [ok] SNR_max={snr_db:.2f} dB | (not saved)")

        except scipy.linalg.LinAlgError:
            if verbose:
                print("  [error] solve failed: Rw singular/ill-conditioned")
        except Exception as e:
            if verbose:
                print(f"  [error] calculation error: {e}")

    if verbose:
        print("\n" + "=" * 50)
        print("      FILTER DESIGN COMPLETE")
        print("=" * 50)
        if results:
            print(f"[ok] Designed {len(results)} filter(s).")
        else:
            print("[!] No filters generated (no valid matrices).")

    return g_by_rx, results


# -------------------------
# Optional CLI test usage
# -------------------------
if __name__ == "__main__":
    # Example pulse (replace with your real one)
    N = 100
    t = np.linspace(0, 1, N, endpoint=False)
    s = np.exp(1j * 2 * np.pi * 10 * t).astype(np.complex64)

    g_by_rx, results = design_optimal_filters_from_covariances(
        s=s,
        matrices=r"G:\GITFINAL\rf_tdoa\CODE\MATCHED_FILTER\NOISE_COVARIANCE\Rw_rx*.npy",  # pattern OR folder OR file
        output_dir=r"G:\GITFINAL\rf_tdoa\CODE\MATCHED_FILTER\OPTIMAL_FILTERS",
        verbose=True,
    )
    print(results)
