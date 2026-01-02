import os
import glob
import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, Iterable, Tuple, Any, List


RX_NAMES = ("rx0", "rx1", "rx2")


# =========================
# Utilities
# =========================

def _safe_makedirs(p: str) -> None:
    if p and not os.path.exists(p):
        os.makedirs(p, exist_ok=True)


def _find_cfile(base: str, rx: str) -> str:
    """
    Supports:
      - base is a folder containing rx0.cfile directly
      - base is a folder containing runs: base/*/rx0.cfile (chooses newest by mtime)
      - base is a direct file path (then rx ignored by caller)
    """
    if os.path.isfile(base):
        return base

    direct = os.path.join(base, f"{rx}.cfile")
    if os.path.isfile(direct):
        return direct

    pattern = os.path.join(base, "*", f"{rx}.cfile")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No .cfile found for {rx} under {base}")

    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return files[0]


def load_filters(
    *,
    filters_dir_or_pattern: str,
    receivers: Tuple[str, ...] = RX_NAMES,
    strict: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Loads optimal filters g for each receiver.

    Accepts:
      - directory containing optimal_filter_g_rx0.npy, etc.
      - glob pattern like ".../optimal_filter_g_*.npy"
      - direct file paths not recommended for multi-rx

    Returns dict rx->g (complex64)
    """
    g_by_rx: Dict[str, np.ndarray] = {}

    # directory case
    if os.path.isdir(filters_dir_or_pattern):
        for rx in receivers:
            f = os.path.join(filters_dir_or_pattern, f"optimal_filter_g_{rx}.npy")
            if os.path.isfile(f):
                g_by_rx[rx] = np.load(f).astype(np.complex64, copy=False)

    else:
        # glob/pattern case
        files = sorted(glob.glob(filters_dir_or_pattern))
        for f in files:
            base = os.path.splitext(os.path.basename(f))[0]
            for rx in receivers:
                if rx in base:
                    g_by_rx[rx] = np.load(f).astype(np.complex64, copy=False)

    if strict:
        missing = [rx for rx in receivers if rx not in g_by_rx]
        if missing:
            raise FileNotFoundError(f"Missing filter files for: {missing} (looked in {filters_dir_or_pattern})")

    return g_by_rx


# =========================
# Streaming FFT matched filter (overlap-save)
# =========================

@dataclass
class MFChunk:
    """One output chunk from the matched filter."""
    y: np.ndarray          # complex output
    mag: np.ndarray        # float32 magnitude
    n_in: int              # input samples consumed for this output
    rx: str


class MatchedFilterOS:
    """
    Overlap-save FFT convolution for streaming matched filtering.

    Matched filter output:
      y[n] = sum_k x[n-k] * h[k]
    where h is the matched filter impulse response.

    We assume user gives g (optimal filter) as a vector length L.
    For classic matched filter, you'd often use h = conj(s[::-1]).
    For your optimal filter, using g as h is fine (it’s already designed for max SNR).
    """
    def __init__(self, h: np.ndarray, *, fft_size: Optional[int] = None):
        h = np.asarray(h).astype(np.complex64, copy=False)
        if h.ndim != 1 or h.size == 0:
            raise ValueError("h must be a 1D non-empty vector")
        self.h = h
        self.L = int(h.size)

        # pick a sane FFT size if not given
        if fft_size is None:
            # choose power of two >= 8*L (good speed) but not tiny
            n = 1
            target = max(2048, 8 * self.L)
            while n < target:
                n *= 2
            fft_size = n

        if fft_size <= self.L:
            raise ValueError("fft_size must be > filter length")
        self.Nfft = int(fft_size)

        # overlap-save block size
        self.B = self.Nfft - self.L + 1  # number of *valid* output samples per block

        # precompute FFT of filter (zero-padded)
        h_pad = np.zeros(self.Nfft, dtype=np.complex64)
        h_pad[:self.L] = self.h
        self.H = np.fft.fft(h_pad)

        # carry buffer holds last L-1 samples
        self._carry = np.zeros(self.L - 1, dtype=np.complex64) if self.L > 1 else np.zeros(0, dtype=np.complex64)

        # input staging buffer (so we can produce whole blocks)
        self._inbuf = np.zeros(0, dtype=np.complex64)

    def feed(self, x: np.ndarray) -> List[MFChunk]:
        """
        Feed complex64 samples. Returns a list of MFChunk outputs.
        Each MFChunk contains:
          - y: complex matched filter output (length <= B)
          - mag: abs(y) float32
        """
        x = np.asarray(x)
        if x.size == 0:
            return []
        if x.dtype != np.complex64:
            x = x.astype(np.complex64, copy=False)

        # append to input buffer
        if self._inbuf.size == 0:
            self._inbuf = x
        else:
            self._inbuf = np.concatenate((self._inbuf, x))

        outs: List[MFChunk] = []

        # while we can form at least one block worth of new samples (B)
        while self._inbuf.size >= self.B:
            take = self._inbuf[:self.B]
            self._inbuf = self._inbuf[self.B:]

            # form overlap-save FFT input: carry + take => length (L-1 + B) = Nfft
            x_block = np.concatenate((self._carry, take))
            # safety: pad to Nfft if tiny numerical edge (should match exactly)
            if x_block.size < self.Nfft:
                x_block = np.pad(x_block, (0, self.Nfft - x_block.size))
            elif x_block.size > self.Nfft:
                x_block = x_block[:self.Nfft]

            X = np.fft.fft(x_block)
            Y = X * self.H
            y_time = np.fft.ifft(Y).astype(np.complex64, copy=False)

            # discard first L-1 (corrupted) samples
            y_valid = y_time[self.L - 1 : self.L - 1 + self.B]
            mag = np.abs(y_valid).astype(np.float32)

            outs.append(MFChunk(y=y_valid, mag=mag, n_in=self.B, rx=""))

            # update carry: last L-1 samples of x_block (which equals last L-1 of (carry+take))
            if self.L > 1:
                self._carry = x_block[-(self.L - 1):]

        return outs

    def flush(self) -> List[MFChunk]:
        """
        Flush remaining buffered input (produces one final partial block).
        This is optional; in infinite streaming you don’t call it.
        """
        if self._inbuf.size == 0:
            return []

        # pad remaining to a full B block
        pad_len = self.B - self._inbuf.size
        take = np.pad(self._inbuf, (0, pad_len)).astype(np.complex64, copy=False)
        self._inbuf = np.zeros(0, dtype=np.complex64)

        x_block = np.concatenate((self._carry, take))
        if x_block.size < self.Nfft:
            x_block = np.pad(x_block, (0, self.Nfft - x_block.size))

        X = np.fft.fft(x_block)
        Y = X * self.H
        y_time = np.fft.ifft(Y).astype(np.complex64, copy=False)

        y_valid = y_time[self.L - 1 : self.L - 1 + self.B]
        mag = np.abs(y_valid).astype(np.float32)

        return [MFChunk(y=y_valid, mag=mag, n_in=self.B, rx="")]


# =========================
# Stream mode: consume chunks from main and return/write MF output
# =========================

def run_matched_filter_stream(
    *,
    chunk_iter: Iterable[Dict[str, np.ndarray]],
    g_by_rx: Dict[str, np.ndarray],
    receivers: Tuple[str, ...] = RX_NAMES,
    fft_size: Optional[int] = None,
    return_outputs: bool = True,
    save_dir: Optional[str] = None,
    save_complex: bool = False,
    verbose: bool = True,
) -> Tuple[Optional[Dict[str, List[np.ndarray]]], Dict[str, Any]]:
    """
    chunk_iter yields dicts like {"rx0": x0, "rx1": x1, "rx2": x2}.
    Applies matched filter per receiver, streaming forever until iterator stops.

    If save_dir is set:
      - saves magnitude outputs as float32 .bin files per rx: mf_mag_rx0.bin
      - optionally saves complex outputs as complex64 .cfile-ish: mf_cplx_rx0.cfile

    If return_outputs True:
      - returns dict rx -> list of np.ndarray blocks (mag blocks)
      - WARNING: this grows in RAM if used for long runs
    """
    _safe_makedirs(save_dir) if save_dir else None

    # init MF engines
    engines: Dict[str, MatchedFilterOS] = {}
    for rx in receivers:
        if rx not in g_by_rx:
            raise KeyError(f"Missing g for {rx}")
        engines[rx] = MatchedFilterOS(g_by_rx[rx], fft_size=fft_size)

    # output sinks
    fh_mag = {}
    fh_cplx = {}
    if save_dir:
        for rx in receivers:
            fh_mag[rx] = open(os.path.join(save_dir, f"mf_mag_{rx}.bin"), "ab")
            if save_complex:
                fh_cplx[rx] = open(os.path.join(save_dir, f"mf_cplx_{rx}.cfile"), "ab")

    collected = {rx: [] for rx in receivers} if return_outputs else None
    stats = {rx: {"blocks": 0, "samples_in": 0, "samples_out": 0} for rx in receivers}

    try:
        for pkt in chunk_iter:
            if pkt is None:
                continue

            for rx in receivers:
                x = pkt.get(rx, None)
                if x is None:
                    continue

                outs = engines[rx].feed(x)
                for oc in outs:
                    oc.rx = rx
                    stats[rx]["blocks"] += 1
                    stats[rx]["samples_in"] += int(oc.n_in)
                    stats[rx]["samples_out"] += int(oc.mag.size)

                    # save
                    if save_dir:
                        oc.mag.tofile(fh_mag[rx])
                        if save_complex:
                            oc.y.astype(np.complex64, copy=False).tofile(fh_cplx[rx])

                    # return
                    if return_outputs:
                        collected[rx].append(oc.mag)

            if verbose:
                # light periodic prints (every ~200 blocks rx0)
                if stats[receivers[0]]["blocks"] % 200 == 0 and stats[receivers[0]]["blocks"] > 0:
                    s0 = stats[receivers[0]]["blocks"]
                    print(f"[i] mf blocks so far: {s0}")

    finally:
        for rx in receivers:
            if rx in fh_mag:
                try:
                    fh_mag[rx].flush(); fh_mag[rx].close()
                except Exception:
                    pass
            if rx in fh_cplx:
                try:
                    fh_cplx[rx].flush(); fh_cplx[rx].close()
                except Exception:
                    pass

    return collected, stats


# =========================
# File mode: read rx*.cfile(s) and produce MF outputs to folder or return
# =========================

def run_matched_filter_files(
    *,
    base_path: str,
    g_by_rx: Dict[str, np.ndarray],
    receivers: Tuple[str, ...] = RX_NAMES,
    fft_size: Optional[int] = None,
    max_samples: Optional[int] = None,
    offset_samples: int = 0,
    read_block_samps: int = 1_000_000,
    return_outputs: bool = False,
    save_dir: Optional[str] = None,
    save_complex: bool = False,
    verbose: bool = True,
) -> Tuple[Optional[Dict[str, np.ndarray]], Dict[str, Any]]:
    """
    Reads rx*.cfile from base_path (folder/run layout supported) and matched-filters.

    Saves:
      mf_mag_rx0.bin (float32)
      optional mf_cplx_rx0.cfile (complex64)

    If return_outputs True:
      returns a dict rx -> concatenated mag array (float32)
      WARNING: can be huge
    """
    _safe_makedirs(save_dir) if save_dir else None

    # init engines
    engines = {rx: MatchedFilterOS(g_by_rx[rx], fft_size=fft_size) for rx in receivers}

    # file paths
    files = {rx: _find_cfile(base_path, rx) for rx in receivers}

    fh_mag = {}
    fh_cplx = {}
    if save_dir:
        for rx in receivers:
            fh_mag[rx] = open(os.path.join(save_dir, f"mf_mag_{rx}.bin"), "ab")
            if save_complex:
                fh_cplx[rx] = open(os.path.join(save_dir, f"mf_cplx_{rx}.cfile"), "ab")

    collected_blocks = {rx: [] for rx in receivers} if return_outputs else None
    stats = {rx: {"blocks": 0, "samples_read": 0, "samples_out": 0, "file": files[rx]} for rx in receivers}

    try:
        for rx in receivers:
            fpath = files[rx]
            if verbose:
                print(f"[i] {rx} reading: {fpath}")

            with open(fpath, "rb") as fh:
                # seek
                if offset_samples > 0:
                    fh.seek(offset_samples * np.dtype(np.complex64).itemsize)

                total_read = 0
                while True:
                    if max_samples is not None and total_read >= max_samples:
                        break

                    # compute how many samples to read this round
                    to_read = read_block_samps
                    if max_samples is not None:
                        to_read = min(to_read, max_samples - total_read)
                    raw = fh.read(to_read * np.dtype(np.complex64).itemsize)
                    if not raw:
                        break

                    x = np.frombuffer(raw, dtype=np.complex64)
                    if x.size == 0:
                        break

                    total_read += int(x.size)
                    stats[rx]["samples_read"] += int(x.size)

                    outs = engines[rx].feed(x)
                    for oc in outs:
                        oc.rx = rx
                        stats[rx]["blocks"] += 1
                        stats[rx]["samples_out"] += int(oc.mag.size)

                        if save_dir:
                            oc.mag.tofile(fh_mag[rx])
                            if save_complex:
                                oc.y.tofile(fh_cplx[rx])

                        if return_outputs:
                            collected_blocks[rx].append(oc.mag)

            # optional flush tail
            tail = engines[rx].flush()
            for oc in tail:
                oc.rx = rx
                stats[rx]["blocks"] += 1
                stats[rx]["samples_out"] += int(oc.mag.size)
                if save_dir:
                    oc.mag.tofile(fh_mag[rx])
                    if save_complex:
                        oc.y.tofile(fh_cplx[rx])
                if return_outputs:
                    collected_blocks[rx].append(oc.mag)

            if verbose:
                print(f"[ok] {rx} blocks={stats[rx]['blocks']}  read={stats[rx]['samples_read']:,}  out={stats[rx]['samples_out']:,}")

    finally:
        for rx in receivers:
            if rx in fh_mag:
                try:
                    fh_mag[rx].flush(); fh_mag[rx].close()
                except Exception:
                    pass
            if rx in fh_cplx:
                try:
                    fh_cplx[rx].flush(); fh_cplx[rx].close()
                except Exception:
                    pass

    if return_outputs:
        out = {rx: np.concatenate(collected_blocks[rx]).astype(np.float32, copy=False) if collected_blocks[rx] else np.zeros(0, np.float32)
               for rx in receivers}
        return out, stats

    return None, stats


# =========================
# Example usage (optional)
# =========================

if __name__ == "__main__":
    # Example: file mode
    g = load_filters(filters_dir_or_pattern=r"G:\GITFINAL\rf_tdoa\CODE\MATCHED_FILTER\OPTIMAL_FILTERS", strict=False)
    out, stats = run_matched_filter_files(
        base_path=r"G:\GITFINAL\rf_tdoa\DATA\RAW_BIN",
        g_by_rx=g,
        save_dir=r"G:\GITFINAL\rf_tdoa\DATA\MF_OUT",
        save_complex=False,
        verbose=True,
    )
    print(stats)
