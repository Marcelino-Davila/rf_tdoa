import os
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt


def _safe_makedirs(p: str) -> None:
    if p and not os.path.exists(p):
        os.makedirs(p, exist_ok=True)


def _find_cfile(base: str, rx: str) -> str:
    """
    Supports:
      - base is a folder containing rx0.cfile directly
      - base is a folder containing runs: base/*/rx0.cfile (chooses newest by mtime)
      - base is a direct file path
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

    # choose newest capture by modified time
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return files[0]


def _iter_complex64_frames(
    fpath: str,
    nfft: int,
    hop: int,
    *,
    frame_stride: int = 1,
    max_frames: int | None = None,
    offset_samples: int = 0,
    dtype=np.complex64,
):
    """
    Yields windowed frames of length nfft, hop-separated.
    Reads file incrementally; no full file load.
    frame_stride: keep every k-th frame (time downsample)
    """
    itemsize = np.dtype(dtype).itemsize
    frame_count = 0
    kept = 0

    with open(fpath, "rb") as fh:
        # seek to offset in samples
        if offset_samples > 0:
            fh.seek(offset_samples * itemsize)

        buf = np.zeros(0, dtype=dtype)
        read_samps = 0

        # choose a block read size (in samples)
        block = max(4 * 1024 * 1024 // itemsize, nfft * 8)  # ~4MB minimum, but at least several frames

        while True:
            raw = fh.read(block * itemsize)
            if not raw:
                break
            x = np.frombuffer(raw, dtype=dtype)
            if x.size == 0:
                break

            if buf.size == 0:
                buf = x
            else:
                buf = np.concatenate((buf, x))

            # produce frames
            while buf.size >= nfft:
                # frame at start of buf
                if (frame_count % frame_stride) == 0:
                    yield buf[:nfft]
                    kept += 1
                    if max_frames is not None and kept >= max_frames:
                        return

                frame_count += 1
                buf = buf[hop:] if hop > 0 else buf[nfft:]
                if hop <= 0:
                    break


def _compute_waterfall_db(
    fpath: str,
    nfft: int,
    hop: int,
    *,
    window: str = "hann",
    frame_stride: int = 1,
    max_frames: int | None = 2000,
    offset_samples: int = 0,
    eps: float = 1e-12,
):
    """
    Returns:
      S_db: (T, nfft) float32, fftshifted
    """
    if window == "hann":
        w = np.hanning(nfft).astype(np.float32)
    elif window == "rect":
        w = np.ones(nfft, dtype=np.float32)
    else:
        raise ValueError("window must be 'hann' or 'rect'")

    rows = []
    for frame in _iter_complex64_frames(
        fpath, nfft, hop, frame_stride=frame_stride, max_frames=max_frames, offset_samples=offset_samples
    ):
        # window + FFT
        xf = frame.astype(np.complex64, copy=False) * w
        X = np.fft.fftshift(np.fft.fft(xf, nfft))
        p = (np.abs(X) ** 2).astype(np.float32)
        rows.append(10.0 * np.log10(p + eps))

    if not rows:
        raise RuntimeError(f"No frames produced from {fpath}. Check nfft/hop/offset/max_frames.")
    return np.stack(rows, axis=0)


def _plot_waterfall(
    S_db: np.ndarray,
    outpath: str,
    *,
    title: str,
    fs: float | None = None,
    center_freq: float | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
):
    """
    S_db: shape (T, nfft), already fftshifted.
    """
    T, nfft = S_db.shape

    # x-axis freq labels if fs given
    if fs is not None:
        freqs = np.linspace(-fs / 2.0, fs / 2.0, nfft, endpoint=False)
        if center_freq is not None:
            freqs = freqs + center_freq
        x0, x1 = freqs[0], freqs[-1]
        extent = [x0, x1, 0, T]
        xlabel = "Frequency (Hz)" if center_freq is None else "Frequency (Hz, absolute)"
    else:
        extent = None
        xlabel = "FFT bin"

    plt.figure()
    plt.imshow(S_db, aspect="auto", origin="lower", extent=extent, vmin=vmin, vmax=vmax)
    plt.colorbar(label="Power (dB)")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Time frame")
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


def make_waterfall_from_bin(
    *,
    base_path: str,
    outdir: str,
    receivers: tuple[str, ...] = ("rx0", "rx1", "rx2"),
    fs: float | None = None,
    center_freq: float | None = None,
    nfft: int = 2048,
    hop: int = 512,
    window: str = "hann",
    max_frames: int = 2000,
    frame_stride: int = 1,
    offset_samples: int = 0,
    vmin: float | None = None,
    vmax: float | None = None,
    verbose: bool = True,
):
    """
    Reads raw .cfile(s) and writes waterfall PNGs for each receiver.
    base_path can be:
      - folder containing rx0.cfile...
      - folder containing runs (subfolders)
      - or direct file path (then receivers should be one-item)
    """
    _safe_makedirs(outdir)

    infos = []
    for rx in receivers:
        fpath = _find_cfile(base_path, rx)
        if verbose:
            print(f"[i] {rx}: using {fpath}")

        S_db = _compute_waterfall_db(
            fpath,
            nfft=nfft,
            hop=hop,
            window=window,
            max_frames=max_frames,
            frame_stride=frame_stride,
            offset_samples=offset_samples,
        )

        outpath = os.path.join(outdir, f"waterfall_{rx}_nfft{nfft}_hop{hop}.png")
        title = f"{rx} Waterfall | nfft={nfft} hop={hop} frames={S_db.shape[0]}"

        _plot_waterfall(
            S_db,
            outpath,
            title=title,
            fs=fs,
            center_freq=center_freq,
            vmin=vmin,
            vmax=vmax,
        )

        if verbose:
            print(f"[ok] saved: {outpath}")

        infos.append({"rx": rx, "file": fpath, "png": outpath, "frames": int(S_db.shape[0])})

    return infos


def main():
    ap = argparse.ArgumentParser(description="Generate waterfall spectrogram PNGs from raw .cfile captures")
    ap.add_argument("--base", required=True,
                    help="Folder containing rx*.cfile OR folder of runs (base/*/rx*.cfile) OR direct .cfile path")
    ap.add_argument("--out", default="GRAPHS/WATERFALLS", help="Output folder for PNGs")
    ap.add_argument("--rx", default="rx0,rx1,rx2", help="Comma list: rx0,rx1,rx2")
    ap.add_argument("--fs", type=float, default=None, help="Sample rate (Hz) for frequency axis labeling")
    ap.add_argument("--fc", type=float, default=None, help="Center frequency (Hz) for absolute frequency axis")
    ap.add_argument("--nfft", type=int, default=2048, help="FFT size")
    ap.add_argument("--hop", type=int, default=512, help="Hop size (samples) between frames")
    ap.add_argument("--window", default="hann", choices=["hann", "rect"], help="Window type")
    ap.add_argument("--frames", type=int, default=2000, help="Max frames to plot (time truncation)")
    ap.add_argument("--stride", type=int, default=1, help="Keep every k-th frame (time downsample)")
    ap.add_argument("--offset", type=int, default=0, help="Offset into file in samples")
    ap.add_argument("--vmin", type=float, default=None, help="Fixed color scale min (dB)")
    ap.add_argument("--vmax", type=float, default=None, help="Fixed color scale max (dB)")
    ap.add_argument("--quiet", action="store_true", help="Less printing")
    args = ap.parse_args()

    rxs = tuple([s.strip() for s in args.rx.split(",") if s.strip()])

    infos = make_waterfall_from_bin(
        base_path=args.base,
        outdir=args.out,
        receivers=rxs,
        fs=args.fs,
        center_freq=args.fc,
        nfft=args.nfft,
        hop=args.hop,
        window=args.window,
        max_frames=args.frames,
        frame_stride=args.stride,
        offset_samples=args.offset,
        vmin=args.vmin,
        vmax=args.vmax,
        verbose=(not args.quiet),
    )

    if not args.quiet:
        print("[i] done")
        for inf in infos:
            print(inf)


if __name__ == "__main__":
    main()
