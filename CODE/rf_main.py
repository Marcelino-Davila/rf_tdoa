# rf_main_stream_snr.py
# ------------------------------------------------------------
# Stream pipeline:
#   ZMQ radios (rx0/rx1/rx2) -> matched filter (streaming)
#   + Every PRINT_EVERY_CHUNKS rounds: print theoretical SNR_max = s^H Rw^{-1} s
#
# NOTE:
#   SNR_max is *theoretical* (depends only on s and Rw) so it will NOT change
#   unless you change s or Rw. This just re-prints it periodically while streaming.
# ------------------------------------------------------------

import os
import numpy as np

from DATA_COLLECTION.rx_cfile_capture_multi import DEFAULT_RADIOS, receive_one_chunk_each
from MATCHED_FILTER.matched_filter import load_filters, MatchedFilterOS
from MATCHED_FILTER.snr_compute import find_rw_paths, compute_max_snr_all_receivers
from MATCHED_FILTER.make_pulse_template import build_and_save_template


# -----------------------------
# CONFIG (edit these)
# -----------------------------
CHUNK_SAMPS = 4096 * 8
TIMEOUT_MS = 1000
FILTERS_DIR = r"G:\GITFINAL\rf_tdoa\CODE\MATCHED_FILTER\OPTIMAL_FILTERS"

S_TEMPLATE_PATH = r"G:\GITFINAL\rf_tdoa\DATA\pulse_template_s.npy"
RW_DIR_OR_PATTERN = r"G:\GITFINAL\rf_tdoa\CODE\MATCHED_FILTER\NOISE_COVARIANCE"

# GNU Radio TX params (must match your flowgraph)
GNURADIO_INTERP = 4
GNURADIO_AMP = 0.25
GNURADIO_OMEGA = 0.31416  # rad/sample

# RRC params (must match your flowgraph)
RRC_BETA = 0.35
RRC_SPAN = 11
RRC_GAIN = 1.0

# Print every N "rounds" (a round = one chunk from each radio)
PRINT_EVERY_CHUNKS = 10


def radio_chunk_generator():
    """Infinite generator yielding one chunk per radio."""
    while True:
        chunks, _meta = receive_one_chunk_each(
            DEFAULT_RADIOS,
            chunk_samps=CHUNK_SAMPS,
            timeout_ms=TIMEOUT_MS,
            overall_timeout_s=None,
            verbose=False,
        )
        yield chunks


def mf_mag_generator(*, receivers=("rx0", "rx1", "rx2"), fft_size=None):
    """
    Infinite generator:
      - pulls radio chunks
      - streams them through per-rx matched filter engines
      - yields MF magnitude chunks: {"rx0": mag, "rx1": mag, "rx2": mag}
    """
    g_by_rx = load_filters(filters_dir_or_pattern=FILTERS_DIR, receivers=receivers, strict=True)
    engines = {rx: MatchedFilterOS(g_by_rx[rx], fft_size=fft_size) for rx in receivers}

    for pkt in radio_chunk_generator():
        out_pkt = {}
        for rx in receivers:
            x = pkt.get(rx, None)
            if x is None:
                continue

            outs = engines[rx].feed(x)
            if not outs:
                continue

            mags = [oc.mag for oc in outs]
            out_pkt[rx] = np.concatenate(mags).astype(np.float32, copy=False)

        if out_pkt:
            yield out_pkt


def _print_snr_max(snr_max: dict):
    print("\n===== THEORETICAL SNR_max (s^H Rw^{-1} s) =====")
    for rx in ("rx0", "rx1", "rx2"):
        r = snr_max[rx]
        print(f"{rx}: SNR_max = {r['snr_db']:.2f} dB  (linear={r['snr_linear']:.4e})  N={int(r['N'])}")
    print("================================================\n")


def main():
    print("[i] Streaming: radios -> matched filter (Ctrl+C to stop)")
    print(f"[i] CHUNK_SAMPS={CHUNK_SAMPS}  PRINT_EVERY_CHUNKS={PRINT_EVERY_CHUNKS}")
    print(f"[i] FILTERS_DIR={FILTERS_DIR}")
    print(f"[i] S_TEMPLATE_PATH={S_TEMPLATE_PATH}")
    print(f"[i] RW_DIR_OR_PATTERN={RW_DIR_OR_PATTERN}")

    # 1) Find Rw files (infer N)
    rw_paths = find_rw_paths(
        rw_dir_or_pattern=RW_DIR_OR_PATTERN,
        receivers=("rx0", "rx1", "rx2"),
    )
    N = int(np.load(rw_paths["rx0"]).shape[0])

    # 2) Load or build pulse template s
    if os.path.exists(S_TEMPLATE_PATH):
        s = np.load(S_TEMPLATE_PATH).astype(np.complex64, copy=False)
        if s.ndim != 1 or s.size < N:
            raise ValueError(f"Template at {S_TEMPLATE_PATH} has shape {s.shape}, need 1D length >= {N}.")
        s = s[:N].astype(np.complex64, copy=False)
        print(f"[i] Loaded pulse template: len={s.size}")
    else:
        print("[!] pulse_template_s.npy missing -> generating from GNU Radio definition...")
        s = build_and_save_template(
            out_path=S_TEMPLATE_PATH,
            N=N,
            mode="gnuradio",
            interp=GNURADIO_INTERP,
            amp=GNURADIO_AMP,
            omega=GNURADIO_OMEGA,
            rrc_beta=RRC_BETA,
            rrc_span=RRC_SPAN,
            rrc_gain=RRC_GAIN,
            normalize=False,
        )
        print(f"[i] Generated + saved pulse template: {S_TEMPLATE_PATH} len={s.size}")

    # 3) Streaming loop: run MF continuously; re-print SNR_max every 10 rounds
    try:
        rounds = 0
        for _mf_pkt in mf_mag_generator(receivers=("rx0", "rx1", "rx2")):
            rounds += 1
            if rounds % PRINT_EVERY_CHUNKS == 0:
                snr_max = compute_max_snr_all_receivers(
                    s=s,
                    rw_paths=rw_paths,
                    save_dir=None,
                    diag_load=0.0,
                    assume_hermitian=True,
                    verbose=False,
                )
                print(f"[i] after {rounds} chunk-rounds:")
                _print_snr_max(snr_max)

    except KeyboardInterrupt:
        print("\n[!] Stopped by user.")


if __name__ == "__main__":
    main()
