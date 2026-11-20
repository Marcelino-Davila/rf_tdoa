#!/usr/bin/env python3
"""
PPS Interrupt Test — Windows native version.
Uses hackrf_transfer to capture IQ and scans for PPS markers.

Usage: python pps_test_windows.py
Requires: hackrf_transfer.exe in PATH or same directory
"""

import subprocess
import struct
import time
import threading
import sys
from datetime import datetime

SYNC_WORD    = bytes([0x7F, 0x80, 0x7F, 0x80, 0x00, 0xFF, 0x00, 0xFF, 0x55, 0xAA, 0x55, 0xAA])
SYNC_LEN     = len(SYNC_WORD)
MARKER_LEN   = 16
SAMPLE_RATE  = 20000000
CENTER_FREQ  = 2420000000
MIN_DELTA_MS = 800

sample_counter  = 0
leftover        = bytearray()
last_pps_time   = None
last_pps_sample = None


def process_chunk(chunk):
    global sample_counter, leftover, last_pps_time, last_pps_sample

    buf = leftover + bytearray(chunk)
    idx = 0

    while idx <= len(buf) - MARKER_LEN:
        if buf[idx:idx + SYNC_LEN] == SYNC_WORD:
            pps_count     = struct.unpack('>I', buf[idx+SYNC_LEN:idx+SYNC_LEN+4])[0]
            global_sample = sample_counter + (idx // 2)
            wall_time     = time.time()
            now_str       = datetime.now().strftime('%H:%M:%S.%f')

            if last_pps_time is not None:
                delta_ms     = (wall_time - last_pps_time) * 1000
                sample_delta = global_sample - last_pps_sample

                if delta_ms > MIN_DELTA_MS:
                    print(f"[{now_str}] PPS #{pps_count:<6} "
                          f"sample={global_sample:<15} "
                          f"delta={delta_ms:.1f}ms  "
                          f"sample_delta={sample_delta}")
                    last_pps_time   = wall_time
                    last_pps_sample = global_sample
            else:
                print(f"[{now_str}] PPS #{pps_count:<6} "
                      f"sample={global_sample:<15} "
                      f"FIRST PULSE!")
                last_pps_time   = wall_time
                last_pps_sample = global_sample

            idx += MARKER_LEN
        else:
            idx += 1

    leftover        = bytearray(buf[max(0, len(buf) - MARKER_LEN):])
    sample_counter += len(chunk) // 2


def main():
    print("PPS Interrupt Test (Windows native)")
    print(f"Freq: {CENTER_FREQ/1e9:.3f} GHz  Rate: {SAMPLE_RATE/1e6:.0f} Msps")
    print("Waiting for PPS pulses — one per second expected...\n")

    # Use hackrf_transfer to stream IQ to stdout
    cmd = [
        r"C:\Users\Madness\radioconda\Library\bin\hackrf_transfer.exe",
        "-r", "-",           # receive to stdout
        "-f", str(CENTER_FREQ),
        "-s", str(SAMPLE_RATE),
        "-l", "16",          # LNA gain
        "-g", "20",          # VGA gain
        "-a", "0",           # amp off
    ]

    print(f"Running: {' '.join(cmd)}\n")

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0
        )

        chunk_size = 131072  # 128KB chunks

        while True:
            chunk = proc.stdout.read(chunk_size)
            if not chunk:
                print("Stream ended")
                break
            process_chunk(chunk)

    except FileNotFoundError:
        print("ERROR: hackrf_transfer not found!")
        print("Make sure hackrf_transfer.exe is in your PATH")
        print("Download from: https://github.com/greatscottgadgets/hackrf/releases")
    except KeyboardInterrupt:
        print("\nStopping...")
        proc.terminate()

    print("Done")


if __name__ == '__main__':
    main()