import os
import numpy as np
from typing import Iterable

RX_NAMES = ("rx0", "rx1", "rx2")


def _safe_makedirs(p: str) -> None:
    if p and not os.path.exists(p):
        os.makedirs(p, exist_ok=True)


def convert_mag_bin_to_cfile(
    *,
    in_dir: str,
    out_dir: str,
    rx_names: Iterable[str] = RX_NAMES,
    scale: float = 1.0,
    verbose: bool = True,
):
    """
    Converts float32 magnitude .bin files into complex64 .cfile
    by placing magnitude on the real axis (imag=0).

    Input expected:
      mf_mag_rx0.bin, mf_mag_rx1.bin, ...

    Output:
      rx0.cfile, rx1.cfile, ...

    NOTE:
      Phase information is LOST (this is magnitude-only).
      This is usually fine for peak detection / visualization.
    """
    _safe_makedirs(out_dir)

    for rx in rx_names:
        in_path = os.path.join(in_dir, f"mf_mag_{rx}.bin")
        out_path = os.path.join(out_dir, f"{rx}.cfile")

        if not os.path.isfile(in_path):
            raise FileNotFoundError(f"Missing input file: {in_path}")

        if verbose:
            print(f"[i] Converting {in_path} -> {out_path}")

        mag = np.fromfile(in_path, dtype=np.float32)
        if mag.size == 0:
            raise RuntimeError(f"{in_path} is empty")

        # Put magnitude on real axis
        cplx = (mag * scale).astype(np.complex64)

        cplx.tofile(out_path)

        if verbose:
            print(f"[ok] {rx}: {mag.size:,} samples written")

    if verbose:
        print("[✓] Magnitude bin → cfile conversion complete.")


def copy_complex_cfiles(
    *,
    in_dir: str,
    out_dir: str,
    rx_names: Iterable[str] = RX_NAMES,
    verbose: bool = True,
):
    """
    Copies/renames complex matched-filter outputs directly.

    Input expected:
      mf_cplx_rx0.cfile, ...

    Output:
      rx0.cfile, ...
    """
    _safe_makedirs(out_dir)

    for rx in rx_names:
        in_path = os.path.join(in_dir, f"mf_cplx_{rx}.cfile")
        out_path = os.path.join(out_dir, f"{rx}.cfile")

        if not os.path.isfile(in_path):
            raise FileNotFoundError(f"Missing input file: {in_path}")

        if verbose:
            print(f"[i] Copying {in_path} -> {out_path}")

        with open(in_path, "rb") as fin, open(out_path, "wb") as fout:
            fout.write(fin.read())

        if verbose:
            print(f"[ok] {rx}: copied")

    if verbose:
        print("[✓] Complex cfile copy complete.")


# -------------------------
# Example CLI usage
# -------------------------
if __name__ == "__main__":
    # CHANGE THESE PATHS
    IN_DIR = r"G:\GITFINAL\rf_tdoa\DATA\MF_OUT"
    OUT_DIR = r"G:\GITFINAL\rf_tdoa\DATA\MF_CFILE"

    # Pick ONE of these:

    # 1) If you only have mf_mag_rx*.bin (float32)
    convert_mag_bin_to_cfile(
        in_dir=IN_DIR,
        out_dir=OUT_DIR,
        verbose=True,
    )

    # 2) If you already saved complex MF output (mf_cplx_rx*.cfile)
    # copy_complex_cfiles(
    #     in_dir=IN_DIR,
    #     out_dir=OUT_DIR,
    #     verbose=True,
    # )
