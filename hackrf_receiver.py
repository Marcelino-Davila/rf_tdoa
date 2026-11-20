#!/usr/bin/env python3
"""
hackrf_receiver.py  —  PC-side IQ stream receiver (run in WSL)
Listens for a TCP connection from the Pi, receives raw IQ data,
writes to a .iq file, and optionally checks for PPS sync markers.

Usage:
    python3 hackrf_receiver.py                        # write to timestamped file
    python3 hackrf_receiver.py --duration 10          # capture for 10 seconds then stop
    python3 hackrf_receiver.py --output myfile.iq     # custom output filename
    python3 hackrf_receiver.py --check-markers        # print PPS marker info live

Edit LISTEN_HOST and LISTEN_PORT to match the daemon script.
"""

import socket
import sys
import time
import argparse
import datetime
import os

# ─── CONSTANTS ────────────────────────────────────────────────────────────────
LISTEN_HOST   = "0.0.0.0"   # listen on all interfaces
LISTEN_PORT   = 7373         # must match Pi daemon PC_PORT
SAMPLE_RATE   = 20_000_000  # for display purposes only
RECV_BUF_SIZE = 1_048_576   # 1 MB socket receive buffer per recv() call

# PPS sync marker from custom HackRF firmware (first 12 bytes)
PPS_MARKER = bytes.fromhex("7F807F8000FF00FF55AA55AA")
# ──────────────────────────────────────────────────────────────────────────────


def find_markers(buf, leftover=b""):
    """Find PPS markers in buffer. Returns list of (offset, pps_count) tuples."""
    data = leftover + buf
    markers = []
    pos = 0
    while True:
        idx = data.find(PPS_MARKER, pos)
        if idx < 0:
            break
        if idx + 16 <= len(data):
            pps_count = int.from_bytes(data[idx+12:idx+16], 'big')
            markers.append((idx, pps_count))
            pos = idx + 16
        else:
            break
    # Return unprocessed tail (might contain partial marker)
    new_leftover = data[pos:]
    return markers, new_leftover


def main():
    parser = argparse.ArgumentParser(description="HackRF IQ stream receiver")
    parser.add_argument("--output",         type=str,   default=None,  help="Output filename (default: timestamped)")
    parser.add_argument("--duration",       type=float, default=None,  help="Stop after N seconds (default: run until Ctrl+C)")
    parser.add_argument("--check-markers",  action="store_true",       help="Print PPS marker info as data arrives")
    parser.add_argument("--no-write",       action="store_true",       help="Don't write to file (useful for throughput testing)")
    args = parser.parse_args()

    # Output filename
    if args.output:
        outfile = args.output
    else:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        outfile = f"capture_{ts}.iq"

    print(f"[INFO] hackrf_receiver starting")
    print(f"[INFO] Listening on {LISTEN_HOST}:{LISTEN_PORT}")
    if not args.no_write:
        print(f"[INFO] Writing to: {outfile}")
    if args.duration:
        print(f"[INFO] Will stop after {args.duration:.1f} seconds")
    print(f"[INFO] Ctrl+C to stop early")
    print()

    # TCP server
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8 * 1024 * 1024)  # 8MB recv buffer
    server.bind((LISTEN_HOST, LISTEN_PORT))
    server.listen(1)
    print(f"[INFO] Waiting for Pi to connect...")

    try:
        conn, addr = server.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        print(f"[INFO] Pi connected from {addr[0]}:{addr[1]}")
        print()
    except KeyboardInterrupt:
        print("\n[INFO] Cancelled.")
        server.close()
        return

    # Receive loop
    total_bytes  = 0
    total_markers = 0
    leftover     = b""
    start_time   = time.time()
    last_print   = start_time
    last_marker_offset = None

    f = None
    if not args.no_write:
        f = open(outfile, 'wb')

    try:
        while True:
            now = time.time()
            elapsed = now - start_time

            # Duration limit
            if args.duration and elapsed >= args.duration:
                print(f"\n[INFO] Duration reached ({args.duration:.1f}s). Stopping.")
                break

            try:
                data = conn.recv(RECV_BUF_SIZE)
            except (ConnectionResetError, OSError):
                print("\n[WARN] Pi disconnected.")
                break

            if not data:
                print("\n[INFO] Stream ended.")
                break

            # Write to file
            if f:
                f.write(data)

            total_bytes += len(data)

            # PPS marker search
            if args.check_markers:
                markers, leftover = find_markers(data, leftover)
                for (offset, pps_count) in markers:
                    total_markers += 1
                    abs_offset = total_bytes - len(data) + offset
                    if last_marker_offset is not None:
                        delta_bytes   = abs_offset - last_marker_offset
                        delta_samples = delta_bytes // 2
                        delta_ms      = delta_samples / SAMPLE_RATE * 1000
                        print(f"[PPS] marker #{total_markers}  pps_count={pps_count}  "
                              f"delta={delta_samples:,} samples  ({delta_ms:.2f} ms)")
                    else:
                        print(f"[PPS] marker #{total_markers}  pps_count={pps_count}  (first marker)")
                    last_marker_offset = abs_offset

            # Throughput print every second
            if now - last_print >= 1.0:
                mb_total  = total_bytes / 1e6
                mb_per_s  = mb_total / elapsed if elapsed > 0 else 0
                print(f"[RX]  {elapsed:6.1f}s  |  {mb_total:8.1f} MB total  |  "
                      f"{mb_per_s:5.1f} MB/s avg  |  {total_markers} PPS markers")
                last_print = now

    except KeyboardInterrupt:
        print("\n[INFO] Ctrl+C — stopping.")
    finally:
        if f:
            f.close()
        conn.close()
        server.close()

    elapsed = time.time() - start_time
    mb = total_bytes / 1e6
    print()
    print(f"[DONE] Received {mb:.1f} MB in {elapsed:.1f}s ({mb/elapsed:.1f} MB/s avg)")
    print(f"[DONE] PPS markers found: {total_markers}")
    if not args.no_write:
        size_on_disk = os.path.getsize(outfile) / 1e6
        print(f"[DONE] File: {outfile}  ({size_on_disk:.1f} MB)")


if __name__ == "__main__":
    main()
