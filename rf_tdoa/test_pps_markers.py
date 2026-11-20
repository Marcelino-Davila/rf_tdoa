#!/usr/bin/env python3
import os
import sys
import time
import shutil
import signal
import subprocess
import threading

# =========================
# CONFIG
# =========================

HACKRF_TRANSFER_CANDIDATES = [
    r"C:\Program Files\PothosSDR\bin\hackrf_transfer.exe",
    "hackrf_transfer",
]

FREQ = "2400000000"
SAMPLE_RATE = 20_000_000
GAIN_LNA = "16"
GAIN_VGA = "20"

MAGIC = bytes([
    0x7F, 0x80, 0x7F, 0x80,
    0x00, 0xFF, 0x00, 0xFF,
    0x55, 0xAA, 0x55, 0xAA,
])

MARKER_LEN = 16
BYTES_PER_COMPLEX_SAMPLE = 2
CHUNK_SIZE = 1024 * 1024


def find_hackrf_transfer():
    for candidate in HACKRF_TRANSFER_CANDIDATES:
        if os.path.exists(candidate):
            return candidate

        found = shutil.which(candidate)
        if found:
            return found

    return None


def stderr_reader(proc):
    while True:
        line = proc.stderr.readline()
        if not line:
            break

        try:
            text = line.decode(errors="replace").rstrip()
        except Exception:
            text = str(line)

        if text:
            print(f"[hackrf] {text}")


def main():
    exe = find_hackrf_transfer()

    if exe is None:
        print("ERROR: Could not find hackrf_transfer.exe")
        print("Expected:")
        print(r"  C:\Program Files\PothosSDR\bin\hackrf_transfer.exe")
        return

    cmd = [
        exe,
        "-r", "-",                 # stdout live stream
        "-f", FREQ,
        "-s", str(SAMPLE_RATE),
        "-l", GAIN_LNA,
        "-g", GAIN_VGA,
    ]

    print()
    print("=" * 72)
    print("LIVE HackRF PPS Marker Test")
    print("=" * 72)
    print("Running:")
    print(" ".join(f'"{x}"' if " " in x else x for x in cmd))
    print()
    print("Watching live stream for PPS markers...")
    print("Press Ctrl+C to stop.")
    print()

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
    )

    threading.Thread(target=stderr_reader, args=(proc,), daemon=True).start()

    overlap = b""
    total_bytes = 0

    marker_count = 0
    last_sample = None
    last_pps_count = None
    start_time = time.time()

    try:
        while True:
            chunk = proc.stdout.read(CHUNK_SIZE)

            if not chunk:
                if proc.poll() is not None:
                    print()
                    print(f"hackrf_transfer exited with code {proc.returncode}")
                    break
                continue

            data = overlap + chunk
            base_offset = total_bytes - len(overlap)

            pos = 0
            while True:
                idx = data.find(MAGIC, pos)
                if idx < 0:
                    break

                if idx + MARKER_LEN <= len(data):
                    abs_byte = base_offset + idx
                    sample = abs_byte // BYTES_PER_COMPLEX_SAMPLE

                    count_bytes = data[idx + 12:idx + 16]
                    pps_count = int.from_bytes(count_bytes, "big", signed=False)

                    marker_count += 1
                    now = time.time()
                    runtime = now - start_time

                    if last_sample is None:
                        print(
                            f"[{runtime:8.3f}s] "
                            f"MARKER {marker_count:04d}  "
                            f"sample={sample:12d}  "
                            f"pps_count={pps_count}"
                        )
                    else:
                        sample_gap = sample - last_sample
                        seconds_gap = sample_gap / SAMPLE_RATE
                        count_gap = pps_count - last_pps_count

                        status = "OK"
                        if count_gap != 1:
                            status = "COUNT_GAP"
                        if sample_gap < SAMPLE_RATE // 2:
                            status = "SPAM"

                        print(
                            f"[{runtime:8.3f}s] "
                            f"MARKER {marker_count:04d}  "
                            f"sample={sample:12d}  "
                            f"gap={sample_gap:12d} samples  "
                            f"dt={seconds_gap:9.6f}s  "
                            f"pps={pps_count}  "
                            f"count_gap={count_gap}  "
                            f"{status}"
                        )

                    last_sample = sample
                    last_pps_count = pps_count

                pos = idx + 1

            overlap = data[-(MARKER_LEN - 1):]
            total_bytes += len(chunk)

    except KeyboardInterrupt:
        print()
        print("Stopping...")

    finally:
        try:
            proc.terminate()
            proc.wait(timeout=2)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    print()
    print("=" * 72)
    print("Summary")
    print("=" * 72)
    print(f"Markers seen: {marker_count}")
    print(f"Bytes scanned: {total_bytes:,}")
    print(f"Approx samples scanned: {total_bytes // 2:,}")
    print()


if __name__ == "__main__":
    main()