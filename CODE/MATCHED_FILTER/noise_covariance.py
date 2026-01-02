import os
import glob
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from typing import Dict, Optional, Iterable, Tuple, Any, List


RX_NAMES = ("rx0", "rx1", "rx2")


def _resolve_noise_files(base_path: str, rx: str) -> List[str]:
    """
    Supports both layouts:
      A) base_path/rx0.cfile
      B) base_path/*/rx0.cfile   (runs)
    Returns list of filepaths.
    """
    direct = os.path.join(base_path, f"{rx}.cfile")
    if os.path.exists(direct):
        return [direct]

    pattern = os.path.join(base_path, "*", f"{rx}.cfile")
    files = glob.glob(pattern)
    return sorted(files)


class NoiseCovEstimator:
    """
    Streaming / incremental covariance estimator for one receiver.

    Maintains a rolling buffer so sliding windows work across chunk boundaries.
    Accumulates Rw = E[x x^H] using snapshots of length N.

    Downsampling:
      - takes every downsample_factor-th snapshot globally across the stream.
    """

    def __init__(self, N: int, downsample_factor: int = 50, dtype=np.complex64):
        if N <= 0:
            raise ValueError("N must be > 0")
        if downsample_factor <= 0:
            raise ValueError("downsample_factor must be > 0")

        self.N = int(N)
        self.k = int(downsample_factor)
        self.dtype = dtype

        self._buf = np.zeros(0, dtype=self.dtype)          # carry-over for boundary windows
        self._sum = np.zeros((self.N, self.N), dtype=self.dtype)
        self._count = 0

        # global snapshot index for downsampling (across feeds)
        self._snap_index = 0

    def feed(self, x: np.ndarray) -> None:
        """
        Feed new complex64 samples. Can be any length.
        """
        if x is None:
            return
        x = np.asarray(x)
        if x.size == 0:
            return
        if x.dtype != self.dtype:
            x = x.astype(self.dtype, copy=False)

        # Append to buffer
        if self._buf.size == 0:
            self._buf = x
        else:
            self._buf = np.concatenate((self._buf, x))

        # Need at least N samples to form one snapshot
        if self._buf.size < self.N:
            return

        # Build sliding windows on the available buffer
        snaps = sliding_window_view(self._buf, window_shape=self.N)  # shape (M, N)

        # Apply global downsampling (every k-th snapshot)
        # We want to keep snapshots where global index % k == 0.
        # Compute indices of snaps relative to global snap index.
        M = snaps.shape[0]
        global_indices = self._snap_index + np.arange(M)
        keep = (global_indices % self.k) == 0
        snaps_ds = snaps[keep]

        if snaps_ds.shape[0] > 0:
            self._sum += snaps_ds.conj().T @ snaps_ds
            self._count += snaps_ds.shape[0]

        # Advance global snapshot index
        self._snap_index += M

        # Keep only the last N-1 samples for boundary continuity
        self._buf = self._buf[-(self.N - 1):] if (self.N > 1) else np.zeros(0, dtype=self.dtype)

    def finalize(self) -> np.ndarray:
        """
        Returns Rw (NxN). Raises if no snapshots were accumulated.
        """
        if self._count <= 0:
            raise RuntimeError("No snapshots accumulated. Feed more data or lower downsample_factor.")
        return self._sum / self._count

    @property
    def snapshots_used(self) -> int:
        return int(self._count)


def compute_noise_covariances_from_folder(
    *,
    N: int,
    base_path: str,
    receivers: Tuple[str, ...] = RX_NAMES,
    chunk_size_samples: int = 1_000_000,
    downsample_factor: int = 50,
    save_dir: Optional[str] = None,
    save_prefix: str = "Rw_",
    verbose: bool = True,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """
    Folder mode: reads .cfile(s) for each receiver and computes Rw per receiver.

    - Supports base_path being either a run folder containing rx0.cfile,rx1.cfile,rx2.cfile
      OR a parent folder containing many runs (subfolders).
    - For each receiver, averages across all files found for that receiver.
    """
    results: Dict[str, np.ndarray] = {}
    info: Dict[str, Any] = {"N": N, "base_path": base_path, "downsample_factor": downsample_factor}

    for rx in receivers:
        files = _resolve_noise_files(base_path, rx)
        if not files:
            raise FileNotFoundError(f"No files found for {rx} under {base_path}")

        if verbose:
            print(f"[i] {rx}: {len(files)} file(s)")

        est_sum = np.zeros((N, N), dtype=np.complex64)
        valid = 0
        snaps_total = 0
        samples_total = 0

        # Process each file; average within file via estimator (same math),
        # then average across files.
        for fpath in files:
            if verbose:
                run_name = os.path.basename(os.path.dirname(fpath))
                print(f"  [i] {rx} processing: {fpath} (run={run_name})")

            w = np.fromfile(fpath, dtype=np.complex64)
            samples_total += int(w.size)

            if w.size < N:
                if verbose:
                    print(f"    [skip] too short ({w.size} < {N})")
                continue

            est = NoiseCovEstimator(N=N, downsample_factor=downsample_factor)

            # Feed in memory-safe chunks (with overlap handled inside estimator)
            for i in range(0, w.size, chunk_size_samples):
                est.feed(w[i:i + chunk_size_samples])

            try:
                Rw_file = est.finalize()
            except RuntimeError:
                if verbose:
                    print("    [skip] no snapshots accumulated in this file")
                continue

            est_sum += Rw_file
            valid += 1
            snaps_total += est.snapshots_used

        if valid == 0:
            raise RuntimeError(f"{rx}: No valid files processed.")

        Rw = est_sum / valid
        results[rx] = Rw

        avg_power = float(np.real(np.trace(Rw)) / N)
        info[rx] = {
            "files_found": len(files),
            "files_used": valid,
            "snapshots_used_total": int(snaps_total),
            "samples_read_total": int(samples_total),
            "avg_noise_power": avg_power,
        }

        if verbose:
            print(f"[ok] {rx}: avg noise power(trace/N) = {avg_power:.6f}")

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            outpath = os.path.join(save_dir, f"{save_prefix}{rx}.npy")
            np.save(outpath, Rw)
            if verbose:
                print(f"[i] saved {rx} -> {outpath}")

    return results, info


def compute_noise_covariances_from_stream(
    *,
    N: int,
    chunk_iter: Iterable[Dict[str, np.ndarray]],
    receivers: Tuple[str, ...] = RX_NAMES,
    downsample_factor: int = 50,
    save_dir: Optional[str] = None,
    save_prefix: str = "Rw_",
    verbose: bool = True,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """
    Stream mode: you provide an iterator that yields dicts like:
        {"rx0": chunk0, "rx1": chunk1, "rx2": chunk2}
    Chunks can arrive unevenly; missing keys are fine.

    Runs until the iterator ends (or you Ctrl+C in your caller).
    """
    ests = {rx: NoiseCovEstimator(N=N, downsample_factor=downsample_factor) for rx in receivers}

    total_feeds = {rx: 0 for rx in receivers}

    for pkt in chunk_iter:
        if pkt is None:
            continue
        for rx in receivers:
            if rx in pkt and pkt[rx] is not None:
                ests[rx].feed(pkt[rx])
                total_feeds[rx] += 1

    results: Dict[str, np.ndarray] = {}
    info: Dict[str, Any] = {"N": N, "downsample_factor": downsample_factor}

    for rx in receivers:
        Rw = ests[rx].finalize()
        results[rx] = Rw
        avg_power = float(np.real(np.trace(Rw)) / N)
        info[rx] = {
            "feed_calls": int(total_feeds[rx]),
            "snapshots_used": int(ests[rx].snapshots_used),
            "avg_noise_power": avg_power,
        }

        if verbose:
            print(f"[ok] {rx}: snapshots={ests[rx].snapshots_used:,}  avgP(trace/N)={avg_power:.6f}")

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            outpath = os.path.join(save_dir, f"{save_prefix}{rx}.npy")
            np.save(outpath, Rw)
            if verbose:
                print(f"[i] saved {rx} -> {outpath}")

    return results, info
