#!/usr/bin/env python3
"""
waterfall.py — fast waterfall PNG generator for large complex64 .cfile recordings.
Streams the file in chunks, never loads it all into RAM.

Usage:
    python waterfall.py --fs 2e6 G:/gpsdo_test_rx1.cfile
    python waterfall.py --fs 2e6 --nfft 4096 --rows 2000 --db-range 60 --cmap inferno G:/gpsdo_test_rx1.cfile

Output: <filename>_waterfall.png  (same directory as input file)
"""

import argparse
import os
import sys
import numpy as np

try:
    from PIL import Image
except ImportError:
    print("Installing Pillow...")
    os.system(f"{sys.executable} -m pip install pillow --quiet")
    from PIL import Image


# ── Colormaps ──────────────────────────────────────────────────────────────
CMAPS = {
    "inferno": np.array([
        [0,0,4],[20,11,52],[58,9,99],[101,21,110],[140,36,99],
        [177,52,85],[211,73,66],[238,107,44],[253,150,16],[252,200,37],[252,255,164]
    ], dtype=np.float32),
    "viridis": np.array([
        [68,1,84],[72,40,120],[62,83,160],[49,124,183],[38,162,173],
        [53,183,121],[109,206,89],[180,222,44],[253,231,37]
    ], dtype=np.float32),
    "plasma": np.array([
        [13,8,135],[84,2,163],[139,10,165],[185,50,137],[219,92,104],
        [244,136,73],[254,188,43],[240,249,33]
    ], dtype=np.float32),
    "turbo": np.array([
        [48,18,59],[70,100,170],[55,185,132],[140,225,40],[243,225,27],[209,55,78]
    ], dtype=np.float32),
    "greys": np.array([[10,10,15],[128,135,140],[255,255,255]], dtype=np.float32),
    "hot": np.array([[0,0,0],[180,0,0],[255,120,0],[255,255,0],[255,255,255]], dtype=np.float32),
}


def make_lut(cmap_name: str) -> np.ndarray:
    """Build a 256x3 uint8 LUT from colormap control points."""
    pts = CMAPS[cmap_name]
    n = len(pts)
    lut = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        t = i / 255.0 * (n - 1)
        lo, hi = int(t), min(int(t) + 1, n - 1)
        f = t - lo
        lut[i] = np.clip(pts[lo] + f * (pts[hi] - pts[lo]), 0, 255).astype(np.uint8)
    return lut


def hann_window(n: int) -> np.ndarray:
    return np.hanning(n).astype(np.float32)


def generate_waterfall(
    path: str,
    fs: float,
    nfft: int,
    rows: int,
    db_range: float,
    cmap_name: str,
    skip_sec: float,
    max_sec: float,
    hop: int,
):
    itemsize = 8  # complex64 = 2x float32
    file_bytes = os.path.getsize(path)
    total_samps = file_bytes // itemsize

    skip_samps = int(skip_sec * fs)
    if max_sec > 0:
        usable_samps = min(total_samps - skip_samps, int(max_sec * fs))
    else:
        usable_samps = total_samps - skip_samps

    total_blocks = usable_samps // hop
    if total_blocks == 0:
        print("ERROR: File too short for given parameters.", file=sys.stderr)
        sys.exit(1)

    # Stride to get exactly `rows` rows spread across the file
    stride = max(1, total_blocks // rows)
    actual_rows = min(rows, total_blocks // stride)

    print(f"  File       : {path}")
    print(f"  Size       : {file_bytes / 1e9:.2f} GB")
    print(f"  Total samps: {total_samps:,}")
    print(f"  Duration   : {total_samps / fs:.1f} s")
    print(f"  FFT size   : {nfft}")
    print(f"  Hop        : {hop}")
    print(f"  Rows       : {actual_rows} (stride={stride} blocks)")
    print(f"  Colormap   : {cmap_name}")
    print(f"  dB range   : {db_range} dB")
    print()

    lut = make_lut(cmap_name)
    win = hann_window(nfft)

    img_rows = []

    with open(path, "rb") as f:
        f.seek(skip_samps * itemsize)

        for row in range(actual_rows):
            pct = int(row / actual_rows * 100)
            if row % max(1, actual_rows // 50) == 0:
                bar = "█" * (pct // 2) + "░" * (50 - pct // 2)
                print(f"\r  [{bar}] {pct:3d}%", end="", flush=True)

            block_offset = row * stride * hop
            byte_offset = skip_samps * itemsize + block_offset * itemsize
            f.seek(byte_offset)
            buf = f.read(nfft * itemsize)
            if len(buf) < nfft * itemsize:
                break

            samples = np.frombuffer(buf, dtype=np.complex64).copy()
            samples *= win

            X = np.fft.fftshift(np.fft.fft(samples, n=nfft))
            mag_db = 20 * np.log10(np.abs(X) + 1e-12)

            # Normalize to [0, 255]
            top = np.max(mag_db)
            bot = top - db_range
            normed = np.clip((mag_db - bot) / db_range, 0, 1)
            indices = (normed * 255).astype(np.uint8)

            img_rows.append(lut[indices])

    print(f"\r  [{'█'*50}] 100%")
    print()

    if not img_rows:
        print("ERROR: No rows rendered.", file=sys.stderr)
        sys.exit(1)

    img_array = np.stack(img_rows, axis=0)  # (rows, nfft, 3)
    img = Image.fromarray(img_array, mode="RGB")

    # Add frequency axis labels by drawing on the image
    try:
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(img)
        tick_count = 9
        for i in range(tick_count + 1):
            x = int(i * (nfft - 1) / tick_count)
            freq_khz = ((i / tick_count) - 0.5) * fs / 1e3
            label = f"{freq_khz:.0f}k"
            draw.line([(x, 0), (x, min(20, img.height))], fill=(0, 255, 200, 128), width=1)
            draw.text((max(0, x - 10), 2), label, fill=(0, 255, 200))
    except Exception:
        pass  # Labels are nice but not essential

    out_path = os.path.splitext(path)[0] + "_waterfall.png"
    img.save(out_path)
    print(f"  ✓ Saved: {out_path}")
    print(f"  ✓ Image size: {img.width} x {img.height} px")
    return out_path


def main():
    ap = argparse.ArgumentParser(description="Waterfall PNG generator for large complex64 .cfile IQ recordings.")
    ap.add_argument("files", nargs="+", help="Path(s) to .cfile recordings.")
    ap.add_argument("--fs",       type=float, default=2e6,    help="Sample rate Hz (default 2e6)")
    ap.add_argument("--nfft",     type=int,   default=4096,   help="FFT size / image width (default 4096)")
    ap.add_argument("--rows",     type=int,   default=2000,   help="Waterfall rows / image height (default 2000)")
    ap.add_argument("--db-range", type=float, default=60.0,   help="dB dynamic range (default 60)")
    ap.add_argument("--cmap",     choices=list(CMAPS.keys()), default="inferno", help="Colormap (default inferno)")
    ap.add_argument("--skip-sec", type=float, default=0.0,    help="Skip first N seconds")
    ap.add_argument("--max-sec",  type=float, default=0.0,    help="Only process first N seconds (0=all)")
    ap.add_argument("--hop",      type=int,   default=0,      help="Hop size in samples (default = nfft)")
    args = ap.parse_args()

    hop = args.hop if args.hop > 0 else args.nfft

    for path in args.files:
        if not os.path.exists(path):
            print(f"ERROR: File not found: {path}", file=sys.stderr)
            continue
        print(f"\n{'='*60}")
        generate_waterfall(
            path=path,
            fs=args.fs,
            nfft=args.nfft,
            rows=args.rows,
            db_range=args.db_range,
            cmap_name=args.cmap,
            skip_sec=args.skip_sec,
            max_sec=args.max_sec,
            hop=hop,
        )


if __name__ == "__main__":
    main()