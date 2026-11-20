#!/usr/bin/env python3
"""
pps_marker_interval.py - measure samples between PPS markers in a HackRF TDOA capture.

Tailored to YOUR firmware. Each PPS marker is 16 bytes written at the start of a
USB transfer block:

    bytes  0-11 : sync pattern  7F 80 7F 80 00 FF 00 FF 55 AA 55 AA
    bytes 12-15 : pps_count, 32-bit big-endian

The capture is raw interleaved signed 8-bit IQ (I0,Q0,I1,Q1,...), 2 bytes/sample.

WHAT IT REPORTS
---------------
  - samples between consecutive markers (the thing you asked for), with the
    embedded pps_count used to detect/normalize any MISSED markers
  - whether the intervals are quantized to the 8192-sample USB block boundary
    (USB_TRANSFER_SIZE = 0x4000 bytes = 8192 samples). With the current firmware
    they will be: the marker sits at the block start, not at the PPS sample, so
    intervals hop between two values exactly one block (512 us @ 16 Msps) apart.
  - alignment sanity: marker byte-offset mod 16384 (blocks preserved?) and mod 2
    (I/Q not byte-swapped?)

USAGE
-----
  python pps_marker_interval.py --iq rx0.iq --rate 16e6 --plot rx0.png
  python pps_marker_interval.py --selftest        # validate the analyzer itself
"""

import argparse
import mmap
import numpy as np

# --- firmware-defined constants (from usb_api_transceiver.c) ---
SYNC = bytes([0x7F, 0x80, 0x7F, 0x80, 0x00, 0xFF, 0x00, 0xFF, 0x55, 0xAA, 0x55, 0xAA])
MARKER_LEN = 16          # 12-byte sync + 4-byte count
BLOCK_BYTES = 0x4000     # USB_TRANSFER_SIZE
BLOCK_SAMPLES = BLOCK_BYTES // 2   # 8192 samples per block


def find_markers(path):
    """Return list of (byte_offset, pps_count) for every marker, in file order.
    Searches the whole stream (the 12-byte sync is unique enough that false hits
    are ~1e-29), so it is robust to whatever the Pi/Ethernet layer did to byte
    alignment."""
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
    if len(markers) < 2:
        raise SystemExit(f"Need >=2 markers; found {len(markers)}. Capture longer "
                         "or check that markers are present.")

    offs = np.array([m[0] for m in markers], dtype=np.int64)
    cnts = np.array([m[1] for m in markers], dtype=np.int64)

    # I/Q + block alignment diagnostics
    odd_off = int(np.sum(offs % 2 != 0))
    blk_resid = offs % BLOCK_BYTES          # constant => blocks preserved
    blk_aligned_const = (np.unique(blk_resid).size == 1)

    d_off = np.diff(offs)
    d_cnt = np.diff(cnts)
    if np.any(d_off % 2 != 0):
        # odd byte gaps mean a sample-level misalignment somewhere
        pass
    d_samp = d_off // 2                      # samples between consecutive markers

    # Use the embedded count to find gaps (missed markers / dropped blocks).
    n_missed = int(np.sum(np.clip(d_cnt - 1, 0, None)))
    bad_cnt = int(np.sum(d_cnt <= 0))        # counter went backwards/stuck => trouble

    adj = d_samp[d_cnt == 1]                 # clean 1-second gaps: the real signal
    if adj.size == 0:
        raise SystemExit("No consecutive (count+1) markers found - too many gaps. "
                         "Check for dropped data in the Pi/Ethernet path.")

    # Per-second rate from gapped pairs too (normalize by count delta), for the mean.
    per_sec_all = d_samp[d_cnt >= 1] / d_cnt[d_cnt >= 1]
    implied_rate = float(per_sec_all.mean())

    base = float(np.rint(adj.mean()))
    resid = adj - base                       # samples
    pp = float(adj.max() - adj.min())

    # Block-quantization test: are all 1-second intervals exact multiples of a block?
    blk_mult_frac = float(np.mean(adj % BLOCK_SAMPLES == 0))
    span_blocks = pp / BLOCK_SAMPLES
    values, counts = np.unique(adj, return_counts=True)

    sample_ns = 1e9 / rate
    return dict(
        n_markers=len(markers), n_intervals=int(adj.size), n_missed=n_missed,
        bad_cnt=bad_cnt, odd_off=odd_off, blk_aligned_const=blk_aligned_const,
        blk_resid0=int(blk_resid[0]), implied_rate=implied_rate, nominal_rate=rate,
        base=base, resid=resid, adj=adj, pp_samp=pp, pp_ns=pp * sample_ns,
        pp_blocks=span_blocks, blk_mult_frac=blk_mult_frac,
        values=values, counts=counts, sample_ns=sample_ns,
        block_us=BLOCK_SAMPLES * sample_ns / 1e3,
        first_cnt=int(cnts[0]), last_cnt=int(cnts[-1]),
    )


def report(r):
    print("=" * 70)
    print("PPS MARKER INTERVAL  (samples between consecutive PPS markers)")
    print("=" * 70)
    print(f"Markers found          : {r['n_markers']}  "
          f"(pps_count {r['first_cnt']} -> {r['last_cnt']})")
    print(f"1-second intervals      : {r['n_intervals']}")
    print(f"Missed markers (count)  : {r['n_missed']}"
          + ("" if r['n_missed'] == 0 else "   <-- gaps in the stream"))
    if r['bad_cnt']:
        print(f"Non-increasing counts   : {r['bad_cnt']}   <-- ordering/drop problem")
    print("-" * 70)
    print(f"Nominal sample rate     : {r['nominal_rate']:,.0f} Sps "
          f"({r['sample_ns']:.4f} ns/sample)")
    drift = r['implied_rate'] - r['nominal_rate']
    print(f"Implied rate (mean gap) : {r['implied_rate']:,.1f} Sps "
          f"({drift:+,.1f}, {1e6*drift/r['nominal_rate']:+.2f} ppm)")
    print("-" * 70)
    print(f"Mean interval           : {r['base']:,.0f} samples")
    print(f"Peak-to-peak spread     : {r['pp_samp']:,.0f} samples = "
          f"{r['pp_ns']/1e3:.1f} us = {r['pp_blocks']:.2f} USB blocks")
    print("Distinct interval values (samples : count):")
    for v, c in zip(r['values'], r['counts']):
        d = v - r['base']
        print(f"    {int(v):>12,}  : {int(c):<5}  ({d:+,.0f} from mean, "
              f"{(v % BLOCK_SAMPLES == 0) and 'block-aligned' or 'NOT block-aligned'})")
    print("-" * 70)
    # block-quantization verdict
    if r['blk_mult_frac'] > 0.99 and 0.5 <= r['pp_blocks'] <= 1.5:
        print(f"BLOCK-QUANTIZED, ~1 block ({r['block_us']:.0f} us) peak-to-peak.")
        print("This is the buffer-boundary effect: the marker is pinned to the USB")
        print("block start, not the PPS sample. EXPECTED with the current firmware.")
        print("Fine timing must come from signal cross-correlation, not marker")
        print("positions. To shrink this, latch m0_state.m0_count in pin_int0_isr.")
    elif r['blk_mult_frac'] > 0.99:
        print(f"BLOCK-ALIGNED but spread is {r['pp_blocks']:.1f} blocks - larger than")
        print("one block suggests gaps or a clocking offset; check the ppm line and plot.")
    else:
        print(f"NOT cleanly block-aligned ({100*r['blk_mult_frac']:.0f}% of intervals are")
        print("block multiples). Suggests byte-level loss in the capture/Ethernet path,")
        print("or markers landing off the block grid - inspect the plot and alignment below.")
    print("-" * 70)
    print("Alignment checks:")
    print(f"  marker offset mod 16384 : {'constant ('+str(r['blk_resid0'])+') - blocks preserved' if r['blk_aligned_const'] else 'VARIES - blocks NOT preserved (drops?)'}")
    print(f"  odd (I/Q-swapped) offsets: {r['odd_off']}"
          + ("" if r['odd_off'] == 0 else "  <-- drop 1 byte at file start to fix I/Q"))
    print("=" * 70)
    print("Reminder: this checks marker spacing only. Final timing precision and")
    print("inter-unit alignment are certified by the zero-baseline splitter test.")
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
    lo = np.floor(us.min() / grid) - 1
    hi = np.ceil(us.max() / grid) + 1
    for k in np.arange(lo, hi + 1):
        ax[0].axhline(k * grid, color="r", lw=0.4, ls="--")
    ax[0].set(xlabel="interval #", ylabel="interval - mean (us)",
              title=f"PPS interval residual  (red grid = {grid:.0f} us USB block; "
                    "steps on the grid = buffer-boundary quantization)")
    ax[1].plot(r['adj'], ".", ms=4)
    ax[1].axhline(r['base'], color="k", lw=0.6)
    ax[1].set(xlabel="interval #", ylabel="interval (samples)",
              title="Samples between consecutive PPS markers")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"Saved plot -> {path}")


# --------------------------------------------------------------------------
# Self-test: build a file with markers placed EXACTLY as the firmware does
# (at block boundaries), then confirm the analyzer recovers the truth.
# --------------------------------------------------------------------------
def _build_synthetic(path, rate, n_pps, start_count=50, start_byte=0):
    rng = np.random.default_rng(1)
    true_pps = np.arange(n_pps) * rate                      # exact 1 Hz, locked clock
    # firmware places marker at the START of the block containing the PPS sample
    marker_samp = (true_pps // BLOCK_SAMPLES) * BLOCK_SAMPLES
    marker_byte = marker_samp * 2 + start_byte
    total = int(marker_byte[-1] + BLOCK_BYTES)
    buf = bytearray(rng.integers(-128, 128, size=total, dtype=np.int8).tobytes())
    for i, off in enumerate(marker_byte):
        off = int(off)
        buf[off:off + 12] = SYNC
        buf[off + 12:off + 16] = int(start_count + i).to_bytes(4, "big")
    with open(path, "wb") as f:
        f.write(buf)
    return marker_samp


def selftest():
    import tempfile, os
    rate, n = 1_000_000, 30
    path = os.path.join(tempfile.gettempdir(), "_pps_selftest.iq")
    truth = _build_synthetic(path, rate, n, start_count=50)
    r = analyze(find_markers(path), rate)
    report(r)
    print("\n--- self-test assertions ---")
    ok = True
    def check(name, cond):
        nonlocal ok
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        ok = ok and cond
    check("found all markers", r['n_markers'] == n)
    check("counts contiguous (no missed)", r['n_missed'] == 0 and r['bad_cnt'] == 0)
    check("implied rate within 1 block of true", abs(r['implied_rate'] - rate) < BLOCK_SAMPLES)
    check("all intervals are block multiples", r['blk_mult_frac'] > 0.99)
    check("spread is ~1 block (quantization visible)", 0.99 <= r['pp_blocks'] <= 1.01)
    check("offsets block-aligned + I/Q aligned", r['blk_aligned_const'] and r['odd_off'] == 0)
    os.remove(path)
    print("\nSELF-TEST:", "PASS" if ok else "FAIL")


def main():
    p = argparse.ArgumentParser(description="Samples between PPS markers (your firmware format).")
    p.add_argument("--iq", help="raw interleaved int8 IQ capture")
    p.add_argument("--rate", type=float, default=16e6, help="sample rate in Sps (default 16e6)")
    p.add_argument("--plot", metavar="PNG", help="save a residual plot here")
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args()
    if a.selftest:
        selftest()
        return
    if not a.iq:
        p.error("provide --iq <file> (or --selftest)")
    r = analyze(find_markers(a.iq), a.rate)
    report(r)
    if a.plot:
        make_plot(r, a.plot)


if __name__ == "__main__":
    main()