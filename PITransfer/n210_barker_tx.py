#!/usr/bin/env python3
"""
Wideband Barker-13 ranging beacon for the USRP N210 -- TDOA transmit source.

Transmits a repeating Barker-13 pulse at 2.426 GHz. Barker-13 has a 13:1
autocorrelation peak-to-sidelobe ratio (22.3 dB) -> one clean, unambiguous
cross-correlation peak, which is exactly what TDOA wants. Because you generate
it, the saved reference (--ref) is also your matched-filter template for the
receive side.

Wideband on purpose: TDOA timing resolution scales with signal bandwidth.
Defaults put ~5 Mchip/s of Barker into your 20 MHz HackRF capture, giving tens
of nanoseconds of timing resolution instead of the ~microsecond you'd get from a
narrowband version.

Pulsed, not continuous: one Barker-13 pulse per PRI (pulse repetition interval),
the rest of the PRI is silence. That keeps the autocorrelation unambiguous out
to the PRI (default 50 us -> 15 km, far beyond your 100 ft baseline) and gives
~20,000 independent TDOA measurements per second to average and check jitter.

The N210's clock does NOT need to match the receivers: in TDOA the unknown
transmit time cancels in the difference of arrivals. What must be synchronized is
the three HackRFs to each other (your shared marker), not the transmitter.

Requires: UHD Python API (the same UHD your GNU Radio USRP sink already uses).

Example:
    python3 n210_barker_tx.py --addr 192.168.10.3 --gain 25
    python3 n210_barker_tx.py --chip-rate 5e6 --sps 2 --pri-us 50 --amp 0.5
"""

import argparse
import sys
import time

import numpy as np

try:
    import uhd
except ImportError:
    print("ERROR: UHD Python API not found. Run this where 'import uhd' works "
          "(the same environment your USRP sink uses).", file=sys.stderr)
    sys.exit(1)

# Barker-13 (matches the sequence in your GGBLUE flowgraph's vector source).
BARKER13 = np.array([1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1], dtype=np.float32)


def build_pri(sps, pri_samples, amp):
    """One PRI: a rectangular-chip Barker-13 pulse followed by silence."""
    chips = np.repeat(BARKER13, sps).astype(np.complex64)   # rectangular chips
    pulse = (amp * chips).astype(np.complex64)
    if pri_samples < len(pulse):
        raise ValueError(f"PRI ({pri_samples} samples) shorter than the pulse "
                         f"({len(pulse)} samples) -- increase --pri-us or lower --chip-rate")
    pri = np.zeros(pri_samples, dtype=np.complex64)
    pri[:len(pulse)] = pulse
    return pri, pulse


def main():
    ap = argparse.ArgumentParser(description="Wideband Barker-13 TDOA beacon (N210)")
    ap.add_argument("--addr", default="192.168.10.3", help="N210 IP address")
    ap.add_argument("--freq", type=float, default=2.426e9, help="center freq (Hz)")
    ap.add_argument("--chip-rate", type=float, default=5e6, help="Barker chip rate (chips/s)")
    ap.add_argument("--sps", type=int, default=2, help="samples per chip (TX rate = chip_rate*sps)")
    ap.add_argument("--pri-us", type=float, default=50.0, help="pulse repetition interval (us)")
    ap.add_argument("--amp", type=float, default=0.5, help="pulse amplitude 0..1 (DAC headroom)")
    ap.add_argument("--gain", type=float, default=25.0, help="TX gain (dB) -- start low")
    ap.add_argument("--antenna", default="TX/RX", help="TX antenna port")
    ap.add_argument("--ref", default="barker13_ref.bin",
                    help="save the pulse as a complex64 matched-filter template")
    args = ap.parse_args()

    rate = args.chip_rate * args.sps
    pri_samples = int(round(rate * args.pri_us * 1e-6))
    pri, pulse = build_pri(args.sps, pri_samples, args.amp)

    # Save the matched-filter reference (one pulse, complex64 interleaved I/Q).
    pulse.tofile(args.ref)

    occupied = 2.0 * args.chip_rate      # rough main-lobe width of rectangular chips
    duty = 100.0 * len(pulse) / pri_samples
    print(f"N210 @ {args.addr}")
    print(f"  center      : {args.freq/1e9:.3f} GHz")
    print(f"  TX rate     : {rate/1e6:.3f} Msps  (chip {args.chip_rate/1e6:.3f} Mcps x sps {args.sps})")
    print(f"  ~main lobe  : {occupied/1e6:.1f} MHz  (fits a 20 MHz capture)")
    print(f"  PRI         : {args.pri_us:.1f} us  (PRF {1e6/args.pri_us:.0f} Hz, "
          f"unambig range {3e8*args.pri_us*1e-6/1e3:.1f} km, duty {duty:.1f}%)")
    print(f"  pulse len   : {len(pulse)} samples")
    print(f"  reference   : {args.ref}  ({len(pulse)} complex64 samples)")

    # ---- bring up the USRP ----
    usrp = uhd.usrp.MultiUSRP(f"addr={args.addr},send_frame_size=8000,recv_frame_size=8000")
    usrp.set_tx_rate(rate)
    usrp.set_tx_freq(uhd.types.TuneRequest(args.freq), 0)
    usrp.set_tx_gain(args.gain, 0)
    try:
        usrp.set_tx_antenna(args.antenna, 0)
    except Exception:
        pass
    try:
        usrp.set_tx_bandwidth(max(occupied, 1e6), 0)
    except Exception:
        pass

    st_args = uhd.usrp.StreamArgs("fc32", "sc16")
    st_args.channels = [0]
    tx = usrp.get_tx_stream(st_args)

    # Tile many PRIs into one send buffer so we're not sending tiny packets.
    reps = max(1, 65536 // len(pri))
    sendbuf = np.tile(pri, reps).astype(np.complex64).reshape(1, -1)

    md = uhd.types.TXMetadata()
    md.start_of_burst = True
    md.end_of_burst = False
    md.has_time_spec = False

    print("\ntransmitting -- Ctrl-C to stop\n", flush=True)
    try:
        while True:
            tx.send(sendbuf, md, 1.0)
            md.start_of_burst = False
    except KeyboardInterrupt:
        print("\nstopping...", flush=True)
        md.end_of_burst = True
        try:
            tx.send(np.zeros((1, 1), dtype=np.complex64), md, 1.0)
        except Exception:
            pass


if __name__ == "__main__":
    main()