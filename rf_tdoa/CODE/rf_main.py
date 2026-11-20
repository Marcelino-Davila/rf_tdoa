import os
import csv
import time

import sys
import numpy as np
from typing import Dict, Tuple, Optional

from DATA_COLLECTION.rx_cfile_capture_multi import DEFAULT_RADIOS, receive_one_chunk_each
from MATCHED_FILTER.matched_filter import load_filters, MatchedFilterOS
from MATCHED_FILTER.snr_compute import find_rw_paths

from TDOA.tdoa_calculation import TDOAEstimator
from TDOA.position_estimator import equilateral_triangle_receivers, TDOALocator2D


# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
RECEIVERS = ("rx0", "rx1", "rx2")

CHUNK_SAMPS = 4096 * 2
TIMEOUT_MS = 500

FILTERS_DIR = r"G:\GITFINAL\rf_tdoa\CODE\MATCHED_FILTER\OPTIMAL_FILTERS"
RW_DIR_OR_PATTERN = r"G:\GITFINAL\rf_tdoa\CODE\MATCHED_FILTER\NOISE_COVARIANCE"

# >>> SET THIS CORRECTLY <<<
FS_HZ = 2_000_000  # sample rate (Hz)

# TDOA configuration
TDOA_REF_RX = "rx0"
TDOA_EMA_ALPHA = 0.2  # smoothing for displayed TDOA (1.0 = none)

# Geometry: equilateral triangle side length (meters)
TRI_SIDE_M = 1.0

# Simple gating: only attempt TDOA/position if receivers have decent post-SNR
MIN_POST_SNR_DB_FOR_LOC = 8.0


# ------------------------------------------------------------
# SNR helpers
# ------------------------------------------------------------

def _parabolic_interpolation(y_m1, y_0, y_p1) -> float:
    """
    Parabolic interpolation around a peak.
    Returns delta in bins relative to center bin (range ~[-0.5, +0.5]).
    """
    denom = (y_m1 - 2.0 * y_0 + y_p1)
    if abs(denom) < 1e-12:
        return 0.0
    return 0.5 * (y_m1 - y_p1) / denom


def run_bench_drift(
    receivers=("rx0", "rx1", "rx2"),
    chunk_samps=8192,
    timeout_ms=500,
    out_dir=r"G:\GITFINAL\rf_tdoa\DATA",
    note: str = "",
    iters: int = 1200,          # 1200 chunks ~ depends on your chunk rate; adjust
    fs_hz: float = 2_000_000.0, # MUST match your capture sample rate
    ignore_dc_hz: float = 5_000.0,  # ignore ±5 kHz around DC so DC spike doesn't win
) -> str:
    """
    Drift benchmark:
      For each chunk per receiver, compute FFT peak frequency (baseband) and log it.

    Output CSV columns:
      t_unix, rx, chunk_idx, peak_freq_hz, peak_mag_db, noise_floor_db

    Notes:
      - peak_freq_hz is relative to your tuned center frequency (baseband).
      - Use a steady carrier so the FFT peak is meaningful.
      - Run in your linear regime (ex: 20 ft or with attenuation).
    """
    os.makedirs(out_dir, exist_ok=True)
    ts_tag = time.strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(out_dir, f"bench_drift_{ts_tag}.csv")
    meta_path = os.path.join(out_dir, f"bench_drift_{ts_tag}_meta.txt")

    with open(meta_path, "w", encoding="utf-8") as f:
        f.write("BENCH: drift\n")
        f.write(f"time: {ts_tag}\n")
        f.write(f"receivers: {receivers}\n")
        f.write(f"chunk_samps: {chunk_samps}\n")
        f.write(f"timeout_ms: {timeout_ms}\n")
        f.write(f"fs_hz: {fs_hz}\n")
        f.write(f"ignore_dc_hz: {ignore_dc_hz}\n")
        f.write(f"note: {note}\n")

    header = ["t_unix", "rx", "chunk_idx", "peak_freq_hz", "peak_mag_db", "noise_floor_db"]

    # Precompute frequency axis
    freqs = np.fft.fftshift(np.fft.fftfreq(chunk_samps, d=1.0 / fs_hz))
    dc_mask = np.abs(freqs) < ignore_dc_hz

    window = np.hanning(chunk_samps).astype(np.float32)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)

        print(f"[BENCH] Writing: {csv_path}")
        print("[BENCH] Drift run... Ctrl+C to stop early.")
        start = time.time()

        try:
            for i in range(iters):
                chunks, meta = receive_one_chunk_each(
                    DEFAULT_RADIOS,
                    chunk_samps=chunk_samps,
                    timeout_ms=timeout_ms,
                    verbose=False,
                )

                for rx in receivers:
                    x = chunks.get(rx, None)
                    if x is None or x.size != chunk_samps:
                        continue

                    # Remove DC offset (time-domain)
                    x = x - np.mean(x)

                    # Window -> FFT
                    X = np.fft.fftshift(np.fft.fft(x * window, n=chunk_samps))
                    mag = np.abs(X)

                    # Avoid DC peak dominating
                    mag2 = mag.copy()
                    mag2[dc_mask] = 0.0

                    k = int(np.argmax(mag2))
                    # Parabolic interpolation in log-mag domain for sub-bin peak estimate
                    if 1 <= k < (chunk_samps - 1):
                        y_m1 = np.log(mag2[k - 1] + 1e-12)
                        y_0  = np.log(mag2[k] + 1e-12)
                        y_p1 = np.log(mag2[k + 1] + 1e-12)
                        dk = _parabolic_interpolation(y_m1, y_0, y_p1)
                    else:
                        dk = 0.0

                    peak_freq_hz = float(freqs[k] + dk * (fs_hz / chunk_samps))

                    peak_mag_db = float(20.0 * np.log10(mag2[k] + 1e-12))
                    noise_floor_db = float(20.0 * np.log10(np.median(mag) + 1e-12))

                    rx_meta = meta.get(rx, {})
                    t_unix = float(rx_meta.get("ts", time.time()))
                    chunk_idx = int(rx_meta.get("idx", -1))

                    w.writerow([f"{t_unix:.6f}", rx, chunk_idx, f"{peak_freq_hz:.2f}", f"{peak_mag_db:.2f}", f"{noise_floor_db:.2f}"])

                if (i + 1) % 100 == 0:
                    elapsed = time.time() - start
                    print(f"[BENCH] {i+1}/{iters} iterations ({elapsed:.1f}s)")

        except KeyboardInterrupt:
            print("\n[BENCH] Stopped early by user.")

    print(f"[BENCH] Done. Meta: {meta_path}")
    return csv_path

def run_bench_linearity(
    receivers=("rx0", "rx1", "rx2"),
    chunk_samps=8192,
    timeout_ms=500,
    out_dir=r"G:\GITFINAL\rf_tdoa\DATA",
    note: str = "",
    iters: int = 400,
) -> str:
    """
    Benchmark 1: Linearity / overload boundary (distance-based).

    You run this multiple times at different TX distances (ex: 1.0m, 0.5m, 1ft).
    This logs per receiver:
      - peak_abs: max(|IQ|) in the chunk
      - noise_floor_db: 10*log10(median(|IQ|^2))
      - (optional) ts/idx from meta

    Output:
      DATA/bench_linearity_<timestamp>.csv
      DATA/bench_linearity_<timestamp>_meta.txt
    """
    os.makedirs(out_dir, exist_ok=True)
    ts_tag = time.strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(out_dir, f"bench_linearity_{ts_tag}.csv")
    meta_path = os.path.join(out_dir, f"bench_linearity_{ts_tag}_meta.txt")

    # Write meta so future-you knows what this run was
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write("BENCH: linearity\n")
        f.write(f"time: {ts_tag}\n")
        f.write(f"receivers: {receivers}\n")
        f.write(f"chunk_samps: {chunk_samps}\n")
        f.write(f"timeout_ms: {timeout_ms}\n")
        f.write(f"note: {note}\n")

    header = ["t_unix", "rx", "chunk_idx", "peak_abs", "noise_floor_db"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)

        print(f"[BENCH] Writing: {csv_path}")
        print("[BENCH] Running... Ctrl+C to stop early.")
        start = time.time()

        try:
            for i in range(iters):
                chunks, meta = receive_one_chunk_each(
                    DEFAULT_RADIOS,
                    chunk_samps=chunk_samps,
                    timeout_ms=timeout_ms,
                    verbose=False,
                )

                for rx in receivers:
                    x = chunks.get(rx, None)
                    if x is None:
                        continue

                    # peak amplitude (flags saturation/compression regime)
                    peak_abs = float(np.max(np.abs(x)))

                    # noise floor proxy: median power in time-domain
                    p = np.abs(x) ** 2
                    noise_floor_db = float(10.0 * np.log10(np.median(p) + 1e-12))

                    rx_meta = meta.get(rx, {})
                    t_unix = float(rx_meta.get("ts", time.time()))
                    chunk_idx = int(rx_meta.get("idx", -1))

                    w.writerow([f"{t_unix:.6f}", rx, chunk_idx, f"{peak_abs:.6f}", f"{noise_floor_db:.3f}"])

                # light progress so you know it's alive
                if (i + 1) % 50 == 0:
                    elapsed = time.time() - start
                    print(f"[BENCH] {i+1}/{iters} iterations ({elapsed:.1f}s)")

        except KeyboardInterrupt:
            print("\n[BENCH] Stopped early by user.")

    print(f"[BENCH] Done. Meta: {meta_path}")
    return csv_path

def get_snr_stats(raw_data: np.ndarray,
                  mag_output: np.ndarray,
                  g: np.ndarray,
                  Rw: np.ndarray) -> Tuple[float, float]:
    """
    Pre-SNR: mean power / median power
    Post-SNR: (peak^2) / (g^H Rw g)   [theoretical output noise power]
    """
    raw_pwr = float(np.mean(np.abs(raw_data) ** 2))
    raw_noise = float(np.median(np.abs(raw_data) ** 2))
    pre_db = 10.0 * np.log10(raw_pwr / max(raw_noise, 1e-12))

    peak_pwr = float(np.max(mag_output) ** 2)
    theo_noise = float(np.real(np.vdot(g, Rw @ g)))
    post_db = 10.0 * np.log10(peak_pwr / max(theo_noise, 1e-12))

    return pre_db, post_db


def format_tdoa(tdoa: Dict[str, Tuple[float, float]], ref_rx: str) -> str:
    """
    tdoa: dict rx -> (delta_samples, delta_seconds) relative to ref_rx
    """
    if not tdoa:
        return ""

    parts = []
    for rx in sorted(tdoa.keys()):
        dsamp, dsec = tdoa[rx]
        dus = dsec * 1e6
        parts.append(f"{rx}-{ref_rx}: {dsamp:+7.2f} samp ({dus:+8.3f} us)")

    return "  ||  " + "  |  ".join(parts)


def format_pos(xy: Optional[Tuple[float, float]]) -> str:
    if xy is None:
        return ""
    x, y = xy
    return f"  ||  POS: x={x:+.3f} m, y={y:+.3f} m"

def _peak_metrics_from_mag(
    mag: np.ndarray,
    guard: int = 64,
    width_db: float = 6.0,
) -> Tuple[int, float, float, int]:
    """
    Given matched-filter magnitude array `mag`, compute:
      - k_peak: peak index
      - peak: peak magnitude
      - psr_db: peak-to-sidelobe ratio in dB
      - width_samps: -width_db dB peak width in samples (approx)
    """
    if mag is None or mag.size < 8:
        return -1, 0.0, -np.inf, 0

    k = int(np.argmax(mag))
    peak = float(mag[k])

    # --- PSR (peak-to-sidelobe ratio) ---
    # Exclude a guard band around the peak to avoid counting mainlobe as sidelobe.
    m = mag
    mask = np.ones(m.size, dtype=bool)
    lo = max(0, k - guard)
    hi = min(m.size, k + guard + 1)
    mask[lo:hi] = False

    sidelobes = m[mask]
    # robust "sidelobe floor"
    side = float(np.median(sidelobes)) if sidelobes.size else 1e-12
    psr_db = 20.0 * np.log10((peak + 1e-12) / (side + 1e-12))

    # --- Peak width at -width_db dB ---
    # threshold = peak * 10^(-width_db/20)
    thr = peak * (10.0 ** (-width_db / 20.0))
    # Walk left/right from peak while above threshold
    L = k
    while L > 0 and m[L] >= thr:
        L -= 1
    R = k
    while R < (m.size - 1) and m[R] >= thr:
        R += 1
    width_samps = int(max(0, R - L - 1))

    return k, peak, psr_db, width_samps


def run_bench_peak_quality(
    receivers=("rx0", "rx1", "rx2"),
    chunk_samps=8192,
    timeout_ms=500,
    out_dir=r"G:\GITFINAL\rf_tdoa\DATA",
    note: str = "",
    iters: int = 800,
    guard: int = 64,
    width_db: float = 6.0,
) -> str:
    """
    Benchmark: Correlation (matched-filter) peak quality vs attenuation.

    Run this bench multiple times with different *static analog attenuators* inline.
    Keep ALL RX/TX settings & geometry the same; only change attenuation.

    Logs per receiver per iteration:
      - peak_idx: argmax of MF magnitude
      - peak_mag: peak magnitude
      - psr_db: peak-to-sidelobe ratio (dB)
      - width_samps: peak width at -width_db (samples)

    Output CSV columns:
      t_unix, rx, chunk_idx, peak_idx, peak_mag, psr_db, width_samps
    """
    os.makedirs(out_dir, exist_ok=True)
    ts_tag = time.strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(out_dir, f"bench_peak_quality_{ts_tag}.csv")
    meta_path = os.path.join(out_dir, f"bench_peak_quality_{ts_tag}_meta.txt")

    with open(meta_path, "w", encoding="utf-8") as f:
        f.write("BENCH: peak_quality\n")
        f.write(f"time: {ts_tag}\n")
        f.write(f"receivers: {receivers}\n")
        f.write(f"chunk_samps: {chunk_samps}\n")
        f.write(f"timeout_ms: {timeout_ms}\n")
        f.write(f"iters: {iters}\n")
        f.write(f"guard: {guard}\n")
        f.write(f"width_db: {width_db}\n")
        f.write(f"note: {note}\n")

    # Load MF + covariance exactly like main()
    rw_paths = find_rw_paths(rw_dir_or_pattern=RW_DIR_OR_PATTERN, receivers=receivers)
    g_by_rx = load_filters(filters_dir_or_pattern=FILTERS_DIR, receivers=receivers, strict=True)

    # Keep Rw loaded even if not used here; it validates paths & consistency
    _Rw_matrices = {rx: np.load(rw_paths[rx]) for rx in receivers}
    engines = {rx: MatchedFilterOS(g_by_rx[rx]) for rx in receivers}

    header = ["t_unix", "rx", "chunk_idx", "peak_idx", "peak_mag", "psr_db", "width_samps"]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)

        print(f"[BENCH] Writing: {csv_path}")
        print("[BENCH] Peak-quality run... Ctrl+C to stop early.")
        start = time.time()

        try:
            for i in range(iters):
                chunks, meta = receive_one_chunk_each(
                    DEFAULT_RADIOS,
                    chunk_samps=chunk_samps,
                    timeout_ms=timeout_ms,
                    verbose=False,
                )

                for rx in receivers:
                    x = chunks.get(rx, None)
                    if x is None or x.size != chunk_samps:
                        continue

                    outs = engines[rx].feed(x)
                    if not outs:
                        continue

                    mags = np.concatenate([oc.mag for oc in outs])

                    peak_idx, peak_mag, psr_db, width_samps = _peak_metrics_from_mag(
                        mags, guard=guard, width_db=width_db
                    )

                    rx_meta = meta.get(rx, {})
                    t_unix = float(rx_meta.get("ts", time.time()))
                    chunk_idx = int(rx_meta.get("idx", -1))

                    w.writerow([
                        f"{t_unix:.6f}",
                        rx,
                        chunk_idx,
                        peak_idx,
                        f"{peak_mag:.6f}",
                        f"{psr_db:.2f}",
                        width_samps,
                    ])

                if (i + 1) % 50 == 0:
                    elapsed = time.time() - start
                    print(f"[BENCH] {i+1}/{iters} iterations ({elapsed:.1f}s)")

        except KeyboardInterrupt:
            print("\n[BENCH] Stopped early by user.")

    print(f"[BENCH] Done. Meta: {meta_path}")
    return csv_path





# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def main() -> None:
    print("[i] rf_main.py starting...")
    print(f"[i] Receivers: {RECEIVERS}")
    print(f"[i] CHUNK_SAMPS={CHUNK_SAMPS}, TIMEOUT_MS={TIMEOUT_MS}")
    print(f"[i] Fs={FS_HZ} Hz")
    print(f"[i] TDOA ref={TDOA_REF_RX}, ema_alpha={TDOA_EMA_ALPHA}")
    print(f"[i] Geometry: equilateral triangle, side={TRI_SIDE_M} m")
    print(f"[i] Loc gate: MIN_POST_SNR_DB_FOR_LOC={MIN_POST_SNR_DB_FOR_LOC} dB")

    # 1) Load filter + noise covariance
    rw_paths = find_rw_paths(rw_dir_or_pattern=RW_DIR_OR_PATTERN, receivers=RECEIVERS)
    g_by_rx = load_filters(filters_dir_or_pattern=FILTERS_DIR, receivers=RECEIVERS, strict=True)

    Rw_matrices = {rx: np.load(rw_paths[rx]) for rx in RECEIVERS}
    engines = {rx: MatchedFilterOS(g_by_rx[rx]) for rx in RECEIVERS}

    # 2) TDOA estimator (separate file)
    tdoa_est = TDOAEstimator(fs_hz=FS_HZ, ref_rx=TDOA_REF_RX, ema_alpha=TDOA_EMA_ALPHA)

    # 3) Position solver (separate file)
    rx_pos = equilateral_triangle_receivers(side_m=TRI_SIDE_M)
    locator = TDOALocator2D(rx_positions=rx_pos, ref_rx=TDOA_REF_RX)

    print("[i] Streaming... Ctrl+C to stop.")

    try:
        while True:
            # Read one chunk per radio
            chunks, _meta = receive_one_chunk_each(
                DEFAULT_RADIOS,
                chunk_samps=CHUNK_SAMPS,
                timeout_ms=TIMEOUT_MS,
                verbose=False,
            )

            mags_by_rx: Dict[str, np.ndarray] = {}
            post_snr_by_rx: Dict[str, float] = {}

            line_parts = []

            for rx in RECEIVERS:
                data = chunks.get(rx, None)
                if data is None:
                    line_parts.append(f"{rx}: (no data)")
                    continue

                outs = engines[rx].feed(data)
                if not outs:
                    line_parts.append(f"{rx}: (no MF out)")
                    continue

                mags = np.concatenate([oc.mag for oc in outs])
                mags_by_rx[rx] = mags

                pre_db, post_db = get_snr_stats(data, mags, g_by_rx[rx], Rw_matrices[rx])
                post_snr_by_rx[rx] = post_db

                line_parts.append(f"{rx}: Pre {pre_db:>5.1f}dB | Post {post_db:>5.1f}dB")

            # --- TDOA ---
            tdoa = tdoa_est.update(mags_by_rx)
            tdoa_str = format_tdoa(tdoa, TDOA_REF_RX)

            # --- Position estimate (2D) ---
            xy = None
            # Require both rx1 and rx2 relative to rx0, and require post-SNR gate
            gate_ok = all(post_snr_by_rx.get(rx, -1e9) >= MIN_POST_SNR_DB_FOR_LOC for rx in RECEIVERS)
            have_pairs = ("rx1" in tdoa) and ("rx2" in tdoa)

            if gate_ok and have_pairs:
                # locator wants seconds offsets relative to ref_rx: {"rx1": dt10, "rx2": dt20}
                tdoa_seconds = {rx: dsec for rx, (_, dsec) in tdoa.items()}
                xy = locator.solve(tdoa_seconds)

            pos_str = format_pos(xy)

            # Print line
            full_line = "  ||  ".join(line_parts) + tdoa_str + pos_str
            sys.stdout.write("\r" + full_line[:420])  # prevent runaway terminal width
            sys.stdout.flush()

    except KeyboardInterrupt:
        print("\n[!] Stopped.")


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "bench_linearity":
        note = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else ""
        run_bench_linearity(
            receivers=RECEIVERS,
            chunk_samps=CHUNK_SAMPS,
            timeout_ms=TIMEOUT_MS,
            out_dir=r"G:\GITFINAL\rf_tdoa\DATA",
            note=note,
            iters=400,
        )

    elif len(sys.argv) >= 2 and sys.argv[1] == "bench_drift":
        note = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else ""
        run_bench_drift(
            receivers=RECEIVERS,
            chunk_samps=CHUNK_SAMPS,
            timeout_ms=TIMEOUT_MS,
            out_dir=r"G:\GITFINAL\rf_tdoa\DATA",
            note=note,
            iters=1200,      # bump to ~3000+ for longer
            fs_hz=float(FS_HZ),
            ignore_dc_hz=5000.0,
        )
        
    elif len(sys.argv) >= 2 and sys.argv[1] == "bench_peak_quality":
        note = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else ""
        run_bench_peak_quality(
            receivers=RECEIVERS,
            chunk_samps=CHUNK_SAMPS,
            timeout_ms=TIMEOUT_MS,
            out_dir=r"G:\GITFINAL\rf_tdoa\DATA",
            note=note,
            iters=800,
            guard=64,
            width_db=6.0,
        )


    else:
        main()

