#!/usr/bin/env python3
"""
Scan a captured HackRF .bin file for PPS markers.
Simplest possible firmware interrupt test - no streaming, no drivers.

Step 1: capture a few seconds with hackrf_transfer:
  hackrf_transfer.exe -r test.bin -f 2420000000 -s 20000000 -l 16 -g 20 -a 0 -n 100000000

Step 2: scan it:
  python scan_pps_file.py test.bin
"""

import sys
import struct

SYNC_WORD  = bytes([0x7F, 0x80, 0x7F, 0x80, 0x00, 0xFF, 0x00, 0xFF, 0x55, 0xAA, 0x55, 0xAA])
SYNC_LEN   = len(SYNC_WORD)
MARKER_LEN = 16
SAMPLE_RATE = 20000000


def main():
    if len(sys.argv) < 2:
        print("Usage: python scan_pps_file.py <capture.bin>")
        return

    filename = sys.argv[1]
    print(f"Scanning {filename} for PPS markers...")
    print(f"Sync word: {SYNC_WORD.hex()}")
    print()

    with open(filename, 'rb') as f:
        data = f.read()

    print(f"File size: {len(data)} bytes = {len(data)//2} IQ samples "
          f"= {len(data)/2/SAMPLE_RATE:.2f} seconds\n")

    found = []
    idx = 0
    while idx <= len(data) - MARKER_LEN:
        if data[idx:idx + SYNC_LEN] == SYNC_WORD:
            pps_count = struct.unpack('>I', data[idx+SYNC_LEN:idx+SYNC_LEN+4])[0]
            sample_index = idx // 2
            found.append((pps_count, sample_index))
            idx += MARKER_LEN
        else:
            idx += 1

    if not found:
        print("NO MARKERS FOUND.")
        print("This means the firmware interrupt is not injecting markers.")
        print("Either the interrupt isn't firing, or the marker injection is broken.")
        return

    print(f"FOUND {len(found)} markers!\n")
    print(f"{'PPS #':<10} {'Sample Index':<20} {'Sample Delta':<15} {'Time Delta'}")
    print("-" * 60)

    last_sample = None
    for pps_count, sample_index in found:
        if last_sample is not None:
            sample_delta = sample_index - last_sample
            time_delta_ms = sample_delta / SAMPLE_RATE * 1000
            print(f"{pps_count:<10} {sample_index:<20} {sample_delta:<15} "
                  f"{time_delta_ms:.1f}ms")
        else:
            print(f"{pps_count:<10} {sample_index:<20} {'--':<15} first")
        last_sample = sample_index

    print()
    print("If sample deltas are ~20,000,000 the interrupt + clock are working!")


if __name__ == '__main__':
    main()