import os
import csv
import time
import zmq
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ==========================================
# DEFAULT CONFIG
# ==========================================
DEFAULT_RADIOS = [
    {"name": "rx0", "endpoint": "tcp://127.0.0.1:5555"},
    {"name": "rx1", "endpoint": "tcp://127.0.0.1:5556"},
    {"name": "rx2", "endpoint": "tcp://127.0.0.1:5557"},
]


# ==========================================
# DATA CONTAINER
# ==========================================
@dataclass
class radio_chunk:
    samples: np.ndarray
    ts: float
    idx: int


@dataclass
class radio_data:
    """
    Holds received chunks per radio.
    Each entry is a list of np.ndarray (dtype=complex64) of length chunk_samps.
    """
    rx0: List[np.ndarray] = field(default_factory=list)
    rx1: List[np.ndarray] = field(default_factory=list)
    rx2: List[np.ndarray] = field(default_factory=list)

    def append(self, radio_name: str, chunk: np.ndarray) -> None:
        if not hasattr(self, radio_name):
            raise KeyError(f"radio_data has no field '{radio_name}'")
        getattr(self, radio_name).append(chunk)

    def get(self, radio_name: str) -> List[np.ndarray]:
        if not hasattr(self, radio_name):
            raise KeyError(f"radio_data has no field '{radio_name}'")
        return getattr(self, radio_name)

    def radios_present(self) -> List[str]:
        return [k for k in ("rx0", "rx1", "rx2") if hasattr(self, k)]

    def total_chunks(self) -> Dict[str, int]:
        return {k: len(getattr(self, k)) for k in ("rx0", "rx1", "rx2") if hasattr(self, k)}

    def clear(self) -> None:
        for k in ("rx0", "rx1", "rx2"):
            if hasattr(self, k):
                getattr(self, k).clear()

    def concat(self, radio_name: str) -> np.ndarray:
        chunks = self.get(radio_name)
        if not chunks:
            return np.zeros(0, dtype=np.complex64)
        return np.concatenate(chunks).astype(np.complex64, copy=False)

    def stack(self, radio_name: str) -> np.ndarray:
        chunks = self.get(radio_name)
        if not chunks:
            return np.zeros((0, 0), dtype=np.complex64)
        return np.stack(chunks).astype(np.complex64, copy=False)


# ==========================================
# INTERNAL HELPERS
# ==========================================
def _ensure_outdir(outdir: str) -> None:
    os.makedirs(outdir, exist_ok=True)


def _open_index_csv(index_csv: str) -> tuple:
    """
    Opens index CSV in append mode and writes header if needed.
    Returns (fh, writer).
    """
    index_exists = os.path.exists(index_csv)
    fh = open(index_csv, "a", newline="")
    writer = csv.writer(fh)
    if not index_exists:
        writer.writerow(["radio", "ts_unix", "ts_iso", "chunk_idx", "samples_per_chunk"])
        fh.flush()
    return fh, writer


# ==========================================
# MAIN RECEIVER API (IMPORT THIS)
# ==========================================
def receive_one_chunk_each(
    radios,
    chunk_samps,
    *,
    timeout_ms: int = 1000,
    overall_timeout_s: float | None = None,
    rcv_hwm: int = 200,
    verbose: bool = False,
):
    """
    Blocks until it has ONE full chunk from EACH radio in `radios`.
    Returns: (chunks_dict, meta_dict)

    chunks_dict = {"rx0": np.ndarray, "rx1": np.ndarray, "rx2": np.ndarray}
    meta_dict   = {"rx0": {...}, "rx1": {...}, "rx2": {...}}
    """
    ctx = zmq.Context.instance()
    poller = zmq.Poller()

    state = {}
    for r in radios:
        name = r["name"]
        endpoint = r["endpoint"]

        sock = ctx.socket(zmq.PULL)
        sock.setsockopt(zmq.RCVHWM, int(rcv_hwm))
        sock.connect(endpoint)
        poller.register(sock, zmq.POLLIN)

        state[sock] = {
            "name": name,
            "endpoint": endpoint,
            "buf": np.zeros(0, dtype=np.complex64),
            "chunk_idx": 0,
        }

        if verbose:
            print(f"[i] {name} connected to {endpoint}")

    needed = {r["name"] for r in radios}
    got_chunks = {}
    got_meta = {}
    start = time.time()

    try:
        while True:
            if overall_timeout_s is not None and (time.time() - start) > overall_timeout_s:
                missing = sorted(needed - set(got_chunks.keys()))
                raise TimeoutError(f"Timed out waiting for all radios: missing {missing}")

            events = dict(poller.poll(timeout=timeout_ms))
            if not events:
                continue

            for sock, ev in events.items():
                if not (ev & zmq.POLLIN):
                    continue

                s = state[sock]
                raw = sock.recv()
                x = np.frombuffer(raw, dtype=np.complex64)
                if x.size == 0:
                    continue

                s["buf"] = np.concatenate((s["buf"], x))

                # Already got this radio's chunk for this round
                if s["name"] in got_chunks:
                    continue

                if s["buf"].size >= chunk_samps:
                    seg = s["buf"][:chunk_samps].astype(np.complex64, copy=False)
                    s["buf"] = s["buf"][chunk_samps:]

                    ts = time.time()
                    idx = s["chunk_idx"]
                    s["chunk_idx"] += 1

                    got_chunks[s["name"]] = seg
                    got_meta[s["name"]] = {"ts": ts, "idx": idx}

                    if verbose:
                        print(f"[{s['name']}] got chunk idx={idx}")

                    if needed.issubset(got_chunks.keys()):
                        return got_chunks, got_meta

    finally:
        for sock in list(state.keys()):
            try:
                poller.unregister(sock)
            except Exception:
                pass
            try:
                sock.close(0)
            except Exception:
                pass


def receive_radio_chunks(
    radios: Optional[List[Dict[str, str]]] = None,
    chunk_samps: int = 4096 * 8,
    *,
    timeout_ms: int = 1000,
    max_chunks_total: Optional[int] = None,
    max_chunks_per_radio: Optional[Dict[str, int]] = None,
    duration_s: Optional[float] = None,
    save_raw: bool = False,
    outdir: str = "raw_bin",
    index_csv_name: str = "chunks_index.csv",
    rcv_hwm: int = 200,
    verbose: bool = True,
) -> radio_data:
    """
    Infinite by default: runs until Ctrl+C unless you set a stop condition.

    Receives complex64 IQ from multiple ZMQ PULL sockets, chunks it into fixed-sized blocks,
    and returns a radio_data object containing the chunks.

    Optional stopping conditions (any one):
      - max_chunks_total reached (across all radios)
      - max_chunks_per_radio reached for each radio listed (if provided)
      - duration_s elapsed (if provided)

    If save_raw=True:
      - writes raw IQ chunks to outdir/<radio>.cfile
      - appends to outdir/chunks_index.csv

    Returns:
      radio_data with lists of chunks per radio.
    """
    radios = radios or DEFAULT_RADIOS
    index_csv = os.path.join(outdir, index_csv_name)

    ctx = zmq.Context.instance()
    poller = zmq.Poller()

    state = {}
    for r in radios:
        name = r["name"]
        endpoint = r["endpoint"]

        sock = ctx.socket(zmq.PULL)
        sock.setsockopt(zmq.RCVHWM, int(rcv_hwm))
        sock.connect(endpoint)
        poller.register(sock, zmq.POLLIN)

        state[sock] = {
            "name": name,
            "endpoint": endpoint,
            "buf": np.zeros(0, dtype=np.complex64),
            "chunk_idx": 0,
            "fh": None,
            "bin_path": None,
        }

        if verbose:
            print(f"[i] {name} connected to {endpoint}")

    # Optional file outputs
    index_fh = None
    index_writer = None
    if save_raw:
        _ensure_outdir(outdir)
        index_fh, index_writer = _open_index_csv(index_csv)

        for sock, s in state.items():
            bin_path = os.path.join(outdir, f"{s['name']}.cfile")
            s["bin_path"] = bin_path
            s["fh"] = open(bin_path, "ab")
            if verbose:
                print(f"[i] {s['name']} -> {bin_path}")

    rows_since_flush = 0
    CSV_FLUSH_EVERY = 10
    start_t = time.time()

    out = radio_data()
    total_chunks = 0

    def _should_stop() -> bool:
        nonlocal total_chunks
        if max_chunks_total is not None and total_chunks >= max_chunks_total:
            return True
        if duration_s is not None and (time.time() - start_t) >= duration_s:
            return True
        if max_chunks_per_radio is not None:
            # stop only if every specified radio reached its limit
            for rname, lim in max_chunks_per_radio.items():
                if lim is None:
                    continue
                if not hasattr(out, rname):
                    continue
                if len(out.get(rname)) < lim:
                    return False
            return True
        return False

    try:
        while True:
            # infinite unless you set a stop condition
            if _should_stop():
                break

            events = dict(poller.poll(timeout=timeout_ms))

            # IMPORTANT: empty poll is normal -> keep looping forever
            if not events:
                continue

            for sock, ev in events.items():
                if not (ev & zmq.POLLIN):
                    continue

                s = state[sock]

                # Drain socket so we don't fall behind
                while True:
                    try:
                        raw = sock.recv(zmq.NOBLOCK)
                    except zmq.Again:
                        break

                    x = np.frombuffer(raw, dtype=np.complex64)
                    if x.size == 0:
                        continue

                    s["buf"] = np.concatenate((s["buf"], x))

                    while s["buf"].size >= chunk_samps:
                        seg = s["buf"][:chunk_samps].astype(np.complex64, copy=False)
                        s["buf"] = s["buf"][chunk_samps:]

                        ts = time.time()
                        ts_iso = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(ts))
                        idx = s["chunk_idx"]
                        s["chunk_idx"] += 1

                        out.append(s["name"], seg)
                        total_chunks += 1

                        if save_raw and s["fh"] is not None:
                            seg.tofile(s["fh"])
                            index_writer.writerow([s["name"], f"{ts:.6f}", ts_iso, idx, chunk_samps])

                            rows_since_flush += 1
                            if rows_since_flush >= CSV_FLUSH_EVERY:
                                if index_fh is not None:
                                    index_fh.flush()
                                for _sock, _s in state.items():
                                    if _s["fh"] is not None:
                                        _s["fh"].flush()
                                rows_since_flush = 0

                        if verbose and (idx % 100 == 0):
                            print(f"[{s['name']}] got chunk {idx}")

                        if _should_stop():
                            break

                if _should_stop():
                    break

    except KeyboardInterrupt:
        if verbose:
            print("\n[!] Stopped by user (Ctrl+C).")
    finally:
        if index_fh is not None:
            try:
                index_fh.flush()
                index_fh.close()
            except Exception:
                pass

        for sock, s in state.items():
            try:
                if s["fh"] is not None:
                    s["fh"].flush()
                    s["fh"].close()
            except Exception:
                pass

            try:
                poller.unregister(sock)
            except Exception:
                pass

            try:
                sock.close(0)
            except Exception:
                pass

    return out


# ==========================================
# OPTIONAL: Example CLI usage
# ==========================================
if __name__ == "__main__":
    # Infinite capture, no saving: Ctrl+C to stop
    data = receive_radio_chunks(
        chunk_samps=4096 * 8,
        save_raw=False,
        verbose=True,
    )
    print("Done.")
    print("Chunks:", data.total_chunks())
