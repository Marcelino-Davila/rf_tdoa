#!/usr/bin/env python3
"""
hackrf_tdoa_capture.py  --  plug-and-play: stream RX from a HackRF and check PPS markers.

What it does, end to end, in one command:
  1. records native 8-bit IQ to disk via `hackrf_transfer -r`  (markers survive byte-exact,
     because nothing converts them to float on the way out)
  2. runs the PPS-marker interval analysis on that capture and prints the verdict

USAGE
  python hackrf_tdoa_capture.py                      # capture with defaults, then analyze
  python hackrf_tdoa_capture.py --seconds 60 --plot rx.png
  python hackrf_tdoa_capture.py --serial 0000...ABCD # pick one of your three units
  python hackrf_tdoa_capture.py --list               # list connected HackRFs and exit
  python hackrf_tdoa_capture.py --selftest           # validate the analyzer (no hardware)

Only dependency beyond Python+numpy is the `hackrf_transfer` CLI (the script tells you how
to install it if it is missing). matplotlib is optional (used only for --plot).
"""

import argparse
import mmap
import os
import shutil
import subprocess
import time
from datetime import datetime

import numpy as np

# ===================== DEFAULTS — taken from your flowgraph =====================
# NOTE: your osmocom Source showed 2.426 GHz, but you said 4.246 GHz in chat earlier.
# One of those is a transposed digit. Set DEF_FREQ to your ACTUAL transmit frequency.
DEF_FREQ = 2_426_000_000      # Hz   (osmocom Ch0 Frequency)
DEF_RATE = 20_000_000         # Sps  (flowgraph Sample Rate 20M)
DEF_BW   = 20_000_000         # Hz   baseband filter (flowgraph Bandwidth 20M)
DEF_AMP  = 1                  # front-end amp ON  (flowgraph RF Gain 14 => amp on)
DEF_LNA  = 24                 # IF/LNA gain, 0-40 step 8   (flowgraph IF Gain 24)
DEF_VGA  = 0                  # BB/VGA gain, 0-62 step 2   (flowgraph BB Gain 0)
DEF_SECS = 30                 # capture length, seconds (need several PPS edges)
# ================================================================================

# PPS marker, from your firmware (usb_api_transceiver.c)
SYNC = bytes([0x7F, 0x80, 0x7F, 0x80, 0x00, 0xFF, 0x00, 0xFF, 0x55, 0xAA, 0x55, 0xAA])
MARKER_LEN = 16
BLOCK_BYTES = 0x4000          # USB_TRANSFER_SIZE
BLOCK_SAMPLES = BLOCK_BYTES // 2   # 8192 samples/block


# ----------------------------- capture (hackrf_transfer) -----------------------------
def find_tool(name):
    return shutil.which(name) or shutil.which(name + ".exe")


def preflight():
    xfer = find_tool("hackrf_transfer")
    if not xfer:
        print("ERROR: 'hackrf_transfer' is not on your PATH.")
        print("Install the HackRF host tools, then re-run:")
        print("   Windows : install radioconda (bundles hackrf_transfer), or PothosSDR")
        print("   Pi/Linux: sudo apt install hackrf")
        print("   macOS   : brew install hackrf")
        return None
    return xfer


def list_devices():
    info = find_tool("hackrf_info")
    if not info:
        print("hackrf_info not found (install the HackRF host tools).")
        return
    subprocess.run([info])


def capture(xfer, args, outpath):
    n_samples = int(args.rate * args.seconds)
    cmd = [xfer]
    if args.serial:
        cmd += ["-d", args.serial]
    cmd += ["-r", outpath,
            "-f", str(int(args.freq)),
            "-s", str(int(args.rate)),
            "-a", str(args.amp),
            "-l", str(args.lna),
            "-g", str(args.vga),
            "-b", str(int(args.bw)),
            "-n", str(n_samples)]
    print("Running:", " ".join(cmd))
    print(f"Capturing ~{args.seconds}s = {n_samples:,} samples (~{n_samples*2/1e6:.0f} MB) "
          f"at {args.freq/1e9:.3f} GHz, {args.rate/1e6:.0f} Msps")
    print("Ctrl+C stops early; the partial capture is still analyzed.\n")
    t0 = time.time()
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nStopped early.")
    dt = time.time() - t0
    if not os.path.exists(outpath) or os.path.getsize(outpath) == 0:
        print("No data captured. Is the HackRF connected, selected, and receiving?")
        return False
    print(f"\nCaptured {os.path.getsize(outpath)/1e6:.1f} MB in {dt:.1f}s -> {outpath}\n")
    return True


# ----------------------------- marker analysis -----------------------------
def find_markers(path):
    markers = []
    with open(path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            n = len(mm)
            pos = mm.find(SYNC, 0)
            while pos != -1:
                if pos + MARKER_LEN <= n:
                    cnt = int.from_bytes(mm[pos + 12:pos + 16], "big")
                    markers.append((pos, cnt))
                pos = mm.find(SYNC, pos + MARKER_LEN)
        finally:
            mm.close()
    return markers


def analyze(markers, rate):
    offs = np.array([m[0] for m in markers], dtype=np.int64)
    cnts = np.array([m[1] for m in markers], dtype=np.int64)
    odd_off = int(np.sum(offs % 2 != 0))
    blk_resid = offs % BLOCK_BYTES
    blk_aligned_const = (np.unique(blk_resid).size == 1)

    d_off = np.diff(offs)
    d_cnt = np.diff(cnts)
    d_samp = d_off // 2
    n_missed = int(np.sum(np.clip(d_cnt - 1, 0, None)))
    bad_cnt = int(np.sum(d_cnt <= 0))

    adj = d_samp[d_cnt == 1]
    if adj.size == 0:
        raise SystemExit("No consecutive (count+1) markers - too many gaps; "
                         "check for dropped data in the capture path.")
    per_sec_all = d_samp[d_cnt >= 1] / d_cnt[d_cnt >= 1]
    implied_rate = float(per_sec_all.mean())

    base = float(np.rint(adj.mean()))
    resid = adj - base
    pp = float(adj.max() - adj.min())
    blk_mult_frac = float(np.mean(adj % BLOCK_SAMPLES == 0))
    span_blocks = pp / BLOCK_SAMPLES
    values, counts = np.unique(adj, return_counts=True)
    sample_ns = 1e9 / rate
    return dict(
        n_markers=len(markers), n_intervals=int(adj.size), n_missed=n_missed,
        bad_cnt=bad_cnt, odd_off=odd_off, blk_aligned_const=blk_aligned_const,
        blk_resid0=int(blk_resid[0]), implied_rate=implied_rate, nominal_rate=rate,
        base=base, resid=resid, adj=adj, pp_samp=pp, pp_ns=pp * sample_ns,
        pp_blocks=span_blocks, blk_mult_frac=blk_mult_frac, values=values, counts=counts,
        sample_ns=sample_ns, block_us=BLOCK_SAMPLES * sample_ns / 1e3,
        first_cnt=int(cnts[0]), last_cnt=int(cnts[-1]),
    )


def report(r):
    print("=" * 70)
    print("PPS MARKER INTERVAL  (samples between consecutive PPS markers)")
    print("=" * 70)
    print(f"Markers found          : {r['n_markers']}  (pps_count {r['first_cnt']} -> {r['last_cnt']})")
    print(f"1-second intervals      : {r['n_intervals']}")
    print(f"Missed markers (count)  : {r['n_missed']}"
          + ("" if r['n_missed'] == 0 else "   <-- gaps in the stream"))
    if r['bad_cnt']:
        print(f"Non-increasing counts   : {r['bad_cnt']}   <-- ordering/drop problem")
    print("-" * 70)
    drift = r['implied_rate'] - r['nominal_rate']
    print(f"Nominal sample rate     : {r['nominal_rate']:,.0f} Sps ({r['sample_ns']:.4f} ns/sample)")
    print(f"Implied rate (mean gap) : {r['implied_rate']:,.1f} Sps ({drift:+,.1f}, "
          f"{1e6*drift/r['nominal_rate']:+.2f} ppm)")
    print(f"Mean interval           : {r['base']:,.0f} samples")
    print(f"Peak-to-peak spread     : {r['pp_samp']:,.0f} samples = {r['pp_ns']/1e3:.1f} us "
          f"= {r['pp_blocks']:.2f} USB blocks")
    print("Distinct interval values (samples : count):")
    for v, c in zip(r['values'], r['counts']):
        d = v - r['base']
        tag = 'block-aligned' if v % BLOCK_SAMPLES == 0 else 'NOT block-aligned'
        print(f"    {int(v):>12,}  : {int(c):<5} ({d:+,.0f} from mean, {tag})")
    print("-" * 70)
    if r['blk_mult_frac'] > 0.99 and 0.5 <= r['pp_blocks'] <= 1.5:
        print(f"BLOCK-QUANTIZED, ~1 block ({r['block_us']:.0f} us) peak-to-peak.")
        print("The marker is pinned to the USB block start, not the PPS sample - EXPECTED")
        print("with the current firmware. Fine timing must come from signal cross-correlation,")
        print("not marker positions. To shrink it, latch m0_state.m0_count in pin_int0_isr.")
    elif r['blk_mult_frac'] > 0.99:
        print(f"BLOCK-ALIGNED but spread is {r['pp_blocks']:.1f} blocks - larger than one block")
        print("suggests gaps or a clocking offset; check the ppm line and the plot.")
    else:
        print(f"NOT cleanly block-aligned ({100*r['blk_mult_frac']:.0f}% block multiples) -")
        print("suggests byte-level loss in the capture path. Inspect alignment below + plot.")
    print("-" * 70)
    print("Alignment checks:")
    print("  offset mod 16384 : "
          + (f"constant ({r['blk_resid0']}) - blocks preserved" if r['blk_aligned_const']
             else "VARIES - blocks NOT preserved (drops?)"))
    print(f"  odd I/Q offsets  : {r['odd_off']}"
          + ("" if r['odd_off'] == 0 else "  <-- drop 1 byte at file start to fix I/Q"))
    print("=" * 70)
    print("This checks marker spacing only. Final precision + inter-unit alignment come")
    print("from the zero-baseline splitter test.")
    print("=" * 70)


def make_plot(r, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"(plot skipped: matplotlib unavailable: {e})")
        return
    us = r['resid'] * r['sample_ns'] / 1e3
    fig, ax = plt.subplots(2, 1, figsize=(11, 7))
    ax[0].plot(us, ".-", lw=0.8, ms=4)
    grid = r['block_us']
    for k in np.arange(np.floor(us.min()/grid) - 1, np.ceil(us.max()/grid) + 2):
        ax[0].axhline(k * grid, color="r", lw=0.4, ls="--")
    ax[0].set(xlabel="interval #", ylabel="interval - mean (us)",
              title=f"PPS interval residual (red grid = {grid:.0f} us block; steps on grid = quantization)")
    ax[1].plot(r['adj'], ".", ms=4)
    ax[1].axhline(r['base'], color="k", lw=0.6)
    ax[1].set(xlabel="interval #", ylabel="samples", title="Samples between consecutive PPS markers")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"Saved plot -> {path}")


# ----------------------------- self-test (no hardware) -----------------------------
def _build_synthetic(path, rate, n_pps, start_count=50):
    rng = np.random.default_rng(1)
    true_pps = np.arange(n_pps) * rate
    marker_samp = (true_pps // BLOCK_SAMPLES) * BLOCK_SAMPLES
    marker_byte = marker_samp * 2
    total = int(marker_byte[-1] + BLOCK_BYTES)
    buf = bytearray(rng.integers(-128, 128, size=total, dtype=np.int8).tobytes())
    for i, off in enumerate(marker_byte):
        off = int(off)
        buf[off:off + 12] = SYNC
        buf[off + 12:off + 16] = int(start_count + i).to_bytes(4, "big")
    with open(path, "wb") as f:
        f.write(buf)


def selftest():
    import tempfile
    rate, n = 1_000_000, 30
    path = os.path.join(tempfile.gettempdir(), "_tdoa_selftest.iq")
    _build_synthetic(path, rate, n)
    r = analyze(find_markers(path), rate)
    report(r)
    ok = (r['n_markers'] == n and r['n_missed'] == 0 and r['blk_mult_frac'] > 0.99
          and 0.99 <= r['pp_blocks'] <= 1.01 and r['blk_aligned_const'] and r['odd_off'] == 0)
    os.remove(path)
    print("\nSELF-TEST:", "PASS" if ok else "FAIL")


# ----------------------------- main -----------------------------
def main():
    p = argparse.ArgumentParser(description="Stream RX from a HackRF and check PPS markers.")
    p.add_argument("--freq", type=float, default=DEF_FREQ, help=f"Hz (default {DEF_FREQ})")
    p.add_argument("--rate", type=float, default=DEF_RATE, help=f"Sps (default {DEF_RATE})")
    p.add_argument("--bw", type=float, default=DEF_BW, help=f"baseband filter Hz (default {DEF_BW})")
    p.add_argument("--amp", type=int, default=DEF_AMP, choices=[0, 1])
    p.add_argument("--lna", type=int, default=DEF_LNA, help="IF gain 0-40 step 8")
    p.add_argument("--vga", type=int, default=DEF_VGA, help="BB gain 0-62 step 2")
    p.add_argument("--seconds", type=float, default=DEF_SECS)
    p.add_argument("--serial", default=None, help="HackRF serial to select one of several")
    p.add_argument("--out", default=None, help="output .iq path (default: timestamped)")
    p.add_argument("--plot", default=None, help="save residual plot to this PNG")
    p.add_argument("--no-analyze", action="store_true", help="capture only, skip analysis")
    p.add_argument("--keep", action="store_true", help="(kept by default; flag is a no-op reminder)")
    p.add_argument("--list", action="store_true", help="list connected HackRFs and exit")
    p.add_argument("--analyze-file", default=None, help="skip capture; analyze an existing file")
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args()

    if a.selftest:
        selftest(); return
    if a.list:
        list_devices(); return

    if a.analyze_file:
        path = a.analyze_file
    else:
        xfer = preflight()
        if not xfer:
            return
        path = a.out or f"rx_{datetime.now():%Y%m%d_%H%M%S}.iq"
        if not capture(xfer, a, path):
            return
        if a.no_analyze:
            print("Capture done (analysis skipped).")
            return

    markers = find_markers(path)
    if len(markers) < 2:
        print("=" * 70)
        print(f"Found {len(markers)} PPS marker(s) - not enough to measure intervals.")
        print("Likely causes:")
        print("  - PPS not wired to the pin (P7_6 / GPIO3[14]) -> ISR never fires")
        print("  - this HackRF isn't running your custom marker firmware")
        print("  - capture too short (need several 1 Hz edges; try --seconds 30+)")
        print("The stream itself may be fine - this only means no markers were present.")
        print("=" * 70)
        return
    r = analyze(markers, a.rate)
    report(r)
    if a.plot:
        make_plot(r, a.plot)


if __name__ == "__main__":
    main()