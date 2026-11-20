import os
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt


def _safe_makedirs(p: str) -> None:
    if p and not os.path.exists(p):
        os.makedirs(p, exist_ok=True)


def _db10(x: np.ndarray, floor_db: float = -160.0) -> np.ndarray:
    x = np.asarray(x)
    mag = np.maximum(x, 10 ** (floor_db / 10.0))
    return 10.0 * np.log10(mag)


def _infer_rx_name(path: str) -> str:
    base = os.path.splitext(os.path.basename(path))[0]
    for rx in ("rx0", "rx1", "rx2"):
        if rx in base:
            return rx
    return base


def _plot_heatmap(mat: np.ndarray, title: str, outpath: str, mode: str = "mag_db") -> None:
    plt.figure()
    if mode == "mag_db":
        img = _db10(np.abs(mat) ** 2)  # power dB
        plt.imshow(img, aspect="auto", origin="lower")
        plt.colorbar(label="Power (dB)")
    elif mode == "real":
        plt.imshow(np.real(mat), aspect="auto", origin="lower")
        plt.colorbar(label="Real")
    elif mode == "imag":
        plt.imshow(np.imag(mat), aspect="auto", origin="lower")
        plt.colorbar(label="Imag")
    else:
        raise ValueError(f"Unknown mode: {mode}")

    plt.title(title)
    plt.xlabel("Column")
    plt.ylabel("Row")
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


def _plot_diag(diag: np.ndarray, title: str, outpath: str) -> None:
    plt.figure()
    plt.plot(np.real(diag))
    plt.title(title)
    plt.xlabel("Index")
    plt.ylabel("Diag value (real)")
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


def _plot_eigs(eigs: np.ndarray, title: str, outpath: str) -> None:
    # eigs are >= 0 for Hermitian PSD-ish matrices; take real part safely
    ev = np.real(eigs)
    ev = np.maximum(ev, 1e-30)
    ev_db = 10.0 * np.log10(ev)

    plt.figure()
    plt.plot(np.sort(ev_db)[::-1])
    plt.title(title)
    plt.xlabel("Eigenvalue index (sorted desc)")
    plt.ylabel("Eigenvalue (dB)")
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


def analyze_and_graph_one(Rw_path: str, outdir: str, floor_db: float = -160.0, verbose: bool = True) -> dict:
    Rw = np.load(Rw_path)
    rx = _infer_rx_name(Rw_path)

    if Rw.ndim != 2 or Rw.shape[0] != Rw.shape[1]:
        raise ValueError(f"{Rw_path}: expected square 2D matrix, got shape {Rw.shape}")

    N = Rw.shape[0]
    _safe_makedirs(outdir)

    base = os.path.splitext(os.path.basename(Rw_path))[0]
    prefix = os.path.join(outdir, base)

    # Heatmaps
    # Magnitude as power in dB
    def db10_local(x):
        mag = np.maximum(x, 10 ** (floor_db / 10.0))
        return 10 * np.log10(mag)

    # We’ll inline floor_db into the heatmap for consistency
    plt.figure()
    img = db10_local(np.abs(Rw) ** 2)
    plt.imshow(img, aspect="auto", origin="lower")
    plt.colorbar(label="Power (dB)")
    plt.title(f"{rx} |Rw|^2 (dB)  N={N}")
    plt.xlabel("Column")
    plt.ylabel("Row")
    plt.tight_layout()
    plt.savefig(prefix + "_heat_mag_db.png", dpi=160)
    plt.close()

    _plot_heatmap(Rw, f"{rx} Re(Rw)  N={N}", prefix + "_heat_real.png", mode="real")
    _plot_heatmap(Rw, f"{rx} Im(Rw)  N={N}", prefix + "_heat_imag.png", mode="imag")

    # Diagonal (noise power proxy)
    diag = np.diag(Rw)
    _plot_diag(diag, f"{rx} diag(Rw) (real)  N={N}", prefix + "_diag.png")

    # Eigen spectrum + condition estimate
    # Use eigvalsh (Hermitian) for stability; if not Hermitian, still works-ish if close.
    try:
        eigs = np.linalg.eigvalsh((Rw + Rw.conj().T) / 2.0)
    except Exception:
        eigs = np.linalg.eigvals(Rw)

    eigs_r = np.real(eigs)
    eigs_r = np.maximum(eigs_r, 1e-30)

    _plot_eigs(eigs_r, f"{rx} eigenvalue spectrum (dB)  N={N}", prefix + "_eigs.png")

    cond_est = float(np.max(eigs_r) / np.min(eigs_r))

    avg_noise_power = float(np.real(np.trace(Rw)) / N)

    info = {
        "rx": rx,
        "file": Rw_path,
        "N": N,
        "avg_noise_power_trace_over_N": avg_noise_power,
        "cond_est_eigs": cond_est,
        "min_eig": float(np.min(eigs_r)),
        "max_eig": float(np.max(eigs_r)),
        "out_prefix": prefix,
    }

    if verbose:
        print(f"[ok] {rx}  N={N}  avgP(trace/N)={avg_noise_power:.6g}  cond~{cond_est:.3g}")
        print(f"     saved: {prefix}_heat_mag_db.png, _heat_real.png, _heat_imag.png, _diag.png, _eigs.png")

    return info


def main():
    ap = argparse.ArgumentParser(description="Generate graphs for noise covariance matrices Rw_*.npy")
    ap.add_argument("--in", dest="in_path", required=True,
                    help="Input: a folder containing .npy matrices OR a glob pattern OR a single .npy file")
    ap.add_argument("--out", dest="outdir", default="GRAPHS_NOISE_COV",
                    help="Output folder for PNGs")
    ap.add_argument("--floor_db", type=float, default=-160.0,
                    help="Floor in dB for heatmap power")
    ap.add_argument("--quiet", action="store_true", help="Less printing")
    args = ap.parse_args()

    in_path = args.in_path
    outdir = args.outdir
    verbose = not args.quiet

    # Resolve input files
    if os.path.isdir(in_path):
        files = sorted(glob.glob(os.path.join(in_path, "*.npy")))
    elif os.path.isfile(in_path):
        files = [in_path]
    else:
        files = sorted(glob.glob(in_path))

    if not files:
        raise FileNotFoundError(f"No .npy files found from: {in_path}")

    _safe_makedirs(outdir)

    all_info = []
    for f in files:
        try:
            info = analyze_and_graph_one(f, outdir, floor_db=args.floor_db, verbose=verbose)
            all_info.append(info)
        except Exception as e:
            print(f"[skip] {f}: {e}")

    # Save a small summary txt
    summary_path = os.path.join(outdir, "summary.txt")
    with open(summary_path, "w") as fh:
        for info in all_info:
            fh.write(
                f"{info['rx']}  N={info['N']}  avgP={info['avg_noise_power_trace_over_N']:.6g}  "
                f"cond~{info['cond_est_eigs']:.3g}  file={info['file']}\n"
            )

    if verbose:
        print(f"[i] Summary written: {summary_path}")


if __name__ == "__main__":
    main()
