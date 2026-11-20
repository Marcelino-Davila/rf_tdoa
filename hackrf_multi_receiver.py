#!/usr/bin/env python3
"""
hackrf_multi_receiver.py  —  PC-side receiver for multiple Pi streams
Accepts simultaneous IQ streams from Pi 2 (port 7372) and Pi 3 (port 7373).
Writes each stream to a separate timestamped .iq file.
Checks for PPS sync markers in each stream.

Usage:
    python3 hackrf_multi_receiver.py                  # run until Ctrl+C
    python3 hackrf_multi_receiver.py --duration 10    # capture 10 seconds
    python3 hackrf_multi_receiver.py --no-write       # throughput test only
"""

import socket
import sys
import time
import argparse
import datetime
import threading
import os

# ─── CONSTANTS ────────────────────────────────────────────────────────────────
LISTEN_HOST   = "0.0.0.0"
SAMPLE_RATE   = 20_000_000
RECV_BUF_SIZE = 1_048_576   # 1 MB per recv() call

PIS = [
    {"name": "pi1", "port": 7371},
    {"name": "pi2", "port": 7372},
    {"name": "pi3", "port": 7373},
]

PPS_MARKER = bytes.fromhex("7F807F8000FF00FF55AA55AA")
# ──────────────────────────────────────────────────────────────────────────────

stop_event = threading.Event()
MARKER_LEN = len(PPS_MARKER) + 4

def find_markers(data, leftover=b""):
    combined = leftover + data
    markers = []
    pos = 0
    while True:
        idx = combined.find(PPS_MARKER, pos)
        if idx < 0 or idx + MARKER_LEN > len(combined):
            break
        pps_count = int.from_bytes(combined[idx+12:idx+16], 'big')
        markers.append((idx, pps_count))
        pos = idx + MARKER_LEN
    keep_from = max(pos, len(combined) - (MARKER_LEN - 1))
    return markers, combined[keep_from:]


def stream_receiver(pi_cfg, duration, no_write, ts):
    """Runs in its own thread — one per Pi."""
    name   = pi_cfg["name"]
    port   = pi_cfg["port"]
    outfile = f"capture_{name}_{ts}.iq"

    print(f"[{name}] Listening on port {port}...")

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8 * 1024 * 1024)
    server.bind((LISTEN_HOST, port))
    server.listen(1)
    server.settimeout(2.0)  # allows checking stop_event periodically

    conn = None
    while not stop_event.is_set():
        try:
            conn, addr = server.accept()
            print(f"[{name}] Connected from {addr[0]}:{addr[1]}")
            break
        except socket.timeout:
            continue

    if conn is None:
        print(f"[{name}] No connection received. Exiting.")
        server.close()
        return

    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    total_bytes    = 0
    total_markers  = 0
    leftover       = b""
    last_marker_offset = None
    start_time     = time.time()
    last_print     = start_time

    f = None
    if not no_write:
        f = open(outfile, 'wb')
        print(f"[{name}] Writing to {outfile}")

    try:
        while not stop_event.is_set():
            now = time.time()
            elapsed = now - start_time

            if duration and elapsed >= duration:
                break

            try:
                conn.settimeout(1.0)
                data = conn.recv(RECV_BUF_SIZE)
            except socket.timeout:
                continue
            except (ConnectionResetError, OSError):
                print(f"[{name}] Pi disconnected.")
                break

            if not data:
                print(f"[{name}] Stream ended.")
                break

            if f:
                f.write(data)

            total_bytes += len(data)

            # PPS marker check
            markers, leftover = find_markers(data, leftover)
            for (offset, pps_count) in markers:
                total_markers += 1
                abs_offset = total_bytes - len(data) + offset
                if last_marker_offset is not None:
                    delta_bytes   = abs_offset - last_marker_offset
                    delta_samples = delta_bytes // 2
                    delta_ms      = delta_samples / SAMPLE_RATE * 1000
                    print(f"[{name}] PPS #{total_markers}  count={pps_count}  "
                          f"delta={delta_samples:,} samples  ({delta_ms:.2f} ms)")
                else:
                    print(f"[{name}] PPS #{total_markers}  count={pps_count}  (first marker)")
                last_marker_offset = abs_offset

            # Throughput every second
            if now - last_print >= 1.0:
                mb_total = total_bytes / 1e6
                mb_per_s = mb_total / elapsed if elapsed > 0 else 0
                print(f"[{name}] {elapsed:5.1f}s | {mb_total:8.1f} MB | "
                      f"{mb_per_s:5.1f} MB/s | {total_markers} PPS markers")
                last_print = now

    finally:
        if f:
            f.close()
        conn.close()
        server.close()

    elapsed = time.time() - start_time
    mb = total_bytes / 1e6
    print(f"[{name}] DONE — {mb:.1f} MB in {elapsed:.1f}s ({mb/elapsed:.1f} MB/s) | "
          f"{total_markers} PPS markers")
    if not no_write:
        print(f"[{name}] File: {outfile}")


def main():
    parser = argparse.ArgumentParser(description="HackRF multi-stream receiver")
    parser.add_argument("--duration", type=float, default=None,
                        help="Stop after N seconds (default: run until Ctrl+C)")
    parser.add_argument("--no-write", action="store_true",
                        help="Don't write to file (throughput test only)")
    args = parser.parse_args()

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 60)
    print("  HackRF Multi-Stream Receiver")
    print(f"  Streams: {', '.join(p['name'] for p in PIS)}")
    print(f"  Ports:   {', '.join(str(p['port']) for p in PIS)}")
    if args.duration:
        print(f"  Duration: {args.duration:.1f}s")
    if args.no_write:
        print(f"  Mode: throughput test (no files written)")
    print("=" * 60)
    print()

    threads = []
    for pi_cfg in PIS:
        t = threading.Thread(
            target=stream_receiver,
            args=(pi_cfg, args.duration, args.no_write, ts),
            daemon=True
        )
        t.start()
        threads.append(t)

    try:
        while any(t.is_alive() for t in threads):
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[INFO] Ctrl+C — stopping all streams...")
        stop_event.set()

    for t in threads:
        t.join(timeout=5)

    print("\n[INFO] All streams closed.")


if __name__ == "__main__":
    main()