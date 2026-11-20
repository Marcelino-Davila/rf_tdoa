#!/usr/bin/env python3
"""
Read and inspect a raw HackRF .iq capture (int8 interleaved I/Q).

These files are NOT text/CSV -- they're raw bytes: [I0][Q0][I1][Q1]... one
signed 8-bit value each, 2 bytes per complex sample, 20e6 samples/sec.

This tool lets you:
  * peek at the first N samples as numbers (the closest thing to "view as a table")
  * dump a slice to an actual CSV if you want to open it in Excel
  * search for the MAGIC marker and report where/how many
  * print basic stats (power, DC offset, clipping)

Usage (through your radioconda python):
  run_gr_python.bat inspect_iq.py G:\GITFINAL\capture\node0_00000.iq
  run_gr_python.bat inspect_iq.py node0_00000.iq --samples 20
  run_gr_python.bat inspect_iq.py node0_00000.iq --csv first1000.csv --n-csv 1000
  run_gr_python.bat inspect_iq.py node0_00000.iq --find-marker
"""

import argparse
import os
import sys
import numpy as np

# The marker your firmware injects on each PPS edge (12 bytes).
MAGIC = bytes([0x7F, 0x80, 0x7F, 0x80,
               0x00, 0xFF, 0x00, 0xFF,
               0x55, 0xAA, 0x55, 0xAA])
SAMPLE_RATE = 20_000_000


def load_samples(path, start_sample=0, num_samples=None):
    """Load int8 IQ as a complex64 array. 2 bytes = 1 sample."""
    byte_offset = start_sample * 2
    count = -1 if num_samples is None else num_samples * 2
    raw = np.fromfile(path, dtype=np.int8, count=count, offset=byte_offset)
    if len(raw) % 2:                      # drop a trailing half-sample if present
        raw = raw[:-1]
    iq = raw.astype(np.float32).view()    # keep as int8-valued floats
    i = raw[0::2].astype(np.float32)
    q = raw[1::2].astype(np.float32)
    return i + 1j * q


def show_table(path, n):
    iq = load_samples(path, 0, n)
    print(f"\nFirst {len(iq)} samples of {os.path.basename(path)}:")
    print(f"{'idx':>8} {'I':>6} {'Q':>6} {'|amp|':>8}")
    for k, s in enumerate(iq):
        print(f"{k:>8} {int(s.real):>6} {int(s.imag):>6} {abs(s):>8.1f}")


def dump_csv(path, out, n):
    iq = load_samples(path, 0, n)
    arr = np.column_stack([iq.real.astype(int), iq.imag.astype(int)])
    np.savetxt(out, arr, fmt="%d", delimiter=",", header="I,Q", comments="")
    print(f"wrote {len(iq)} samples to {out} (open in Excel)")


def find_marker(path, max_bytes=None):
    """Scan the file for the MAGIC pattern; report positions and spacing."""
    size = os.path.getsize(path)
    scan = size if max_bytes is None else min(size, max_bytes)
    print(f"\nScanning {scan/1e6:.0f} MB of {os.path.basename(path)} for the marker...")
    positions = []
    chunk = 8 << 20
    tail = b""
    read = 0
    with open(path, "rb") as f:
        base = 0
        while read < scan:
            buf = f.read(min(chunk, scan - read))
            if not buf:
                break
            hay = tail + buf
            start = 0
            while True:
                idx = hay.find(MAGIC, start)
                if idx == -1:
                    break
                positions.append(base - len(tail) + idx)
                start = idx + 1
            keep = len(MAGIC) - 1
            tail = hay[-keep:]
            base += len(buf)
            read += len(buf)

    print(f"markers found: {len(positions)}")
    if len(positions) >= 2:
        gaps = np.diff(positions)                    # in BYTES
        samp_gaps = gaps // 2                         # 2 bytes/sample
        print(f"first marker at byte {positions[0]} (sample {positions[0]//2})")
        print(f"marker spacing: mean {samp_gaps.mean():.0f} samples "
              f"({samp_gaps.mean()/SAMPLE_RATE:.6f} s), "
              f"min {samp_gaps.min()}, max {samp_gaps.max()}")
        print("  (nominal 1 PPS at 20 Msps ~= 20,000,000 samples; ~19,996,672 = free-run drift)")

        # Read the 4-byte value after EVERY marker (candidate PPS counter).
        counters = []
        with open(path, "rb") as f:
            for p in positions:
                f.seek(p + len(MAGIC))
                b = f.read(4)
                counters.append(int.from_bytes(b, "little") if len(b) == 4 else None)

        # Per-marker table: is the spacing ragged only at the start, or throughout?
        print("\n  idx   sample_pos     gap_samples   counter(LE)   counter_delta")
        for k, p in enumerate(positions):
            gap = "" if k == 0 else f"{(p - positions[k-1])//2:>12d}"
            ctr = counters[k]
            cdelta = ""
            if k > 0 and counters[k] is not None and counters[k-1] is not None:
                cdelta = f"{counters[k]-counters[k-1]:>6d}"
            cstr = "" if ctr is None else f"{ctr:>11d}"
            flag = ""
            if k > 0:
                g = (p - positions[k-1])//2
                if abs(g - 19_996_672) > 100_000:     # off nominal by >100k samples
                    flag = "  <-- off nominal"
            print(f"  {k:>3d} {p//2:>12d} {gap:>14} {cstr:>13} {cdelta:>13}{flag}")

        # Summarize the settled region (skip the first 2, which are startup/boundary).
        if len(samp_gaps) > 3:
            settled = samp_gaps[2:]
            print(f"\n  settled gaps (skipping first 2): mean {settled.mean():.0f}, "
                  f"std {settled.std():.0f}, min {settled.min()}, max {settled.max()}")
            print(f"  spread {settled.max()-settled.min()} samples "
                  f"({(settled.max()-settled.min())/SAMPLE_RATE*1e6:.1f} us across the capture)")
            if counters[2] is not None:
                cd = np.diff([c for c in counters[2:] if c is not None])
                if len(cd):
                    print(f"  counter deltas (settled): all == 1? {np.all(cd==1)}  "
                          f"(unique: {sorted(set(cd.tolist()))[:6]})")
    elif len(positions) == 1:
        print(f"only one marker, at byte {positions[0]} (sample {positions[0]//2})")
    else:
        print("NO markers -- this capture has no PPS/time reference in it.")


def stats(path):
    size = os.path.getsize(path)
    n = min(2_000_000, size // 2)                    # sample up to 2M samples
    iq = load_samples(path, 0, n)
    p = np.mean(np.abs(iq) ** 2)
    dbfs = 10 * np.log10(p / (127.0 ** 2) + 1e-12)
    clip = np.mean((np.abs(iq.real) >= 127) | (np.abs(iq.imag) >= 127)) * 100
    print(f"\n{os.path.basename(path)}: {size/1e9:.2f} GB, "
          f"~{size//2/1e6:.1f} M samples ({size//2/SAMPLE_RATE:.2f} s)")
    print(f"  mean power  : {dbfs:.1f} dBfs")
    print(f"  DC offset   : I={iq.real.mean():.2f}  Q={iq.imag.mean():.2f}")
    print(f"  clipping    : {clip:.2f}% of samples at full scale")


def main():
    ap = argparse.ArgumentParser(description="Inspect a raw int8 IQ capture")
    ap.add_argument("file")
    ap.add_argument("--samples", type=int, default=16, help="print first N samples as a table")
    ap.add_argument("--csv", help="dump samples to this CSV (open in Excel)")
    ap.add_argument("--n-csv", type=int, default=1000, help="how many samples to put in the CSV")
    ap.add_argument("--find-marker", action="store_true", help="scan for the MAGIC marker")
    ap.add_argument("--scan-mb", type=int, default=None, help="limit marker scan to first N MB")
    args = ap.parse_args()

    if not os.path.exists(args.file):
        print(f"file not found: {args.file}", file=sys.stderr)
        sys.exit(1)

    stats(args.file)
    if args.find_marker:
        find_marker(args.file, None if args.scan_mb is None else args.scan_mb * (1 << 20))
    if args.csv:
        dump_csv(args.file, args.csv, args.n_csv)
    else:
        show_table(args.file, args.samples)


if __name__ == "__main__":
    main()