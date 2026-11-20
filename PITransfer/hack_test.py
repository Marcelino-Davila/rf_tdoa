#!/usr/bin/env python3
"""
hack_test.py  -  plug-and-play HackRF health check (PC / USB).

Plug ONE HackRF into the PC over USB, then run:
    & "C:\\Users\\Madness\\radioconda\\python.exe" hack_test.py

It exercises the board from "is it there" to "can it stream 40 MB/s
sustained", and tells you exactly where (if anywhere) it fails.
No Pi, no daemon, no network - just the board and USB.
"""

import os
import subprocess
import sys
import time
import glob

# ---- locate the PothosSDR hackrf tools on the PC -------------------------
CANDIN = [
    r"C:\Program Files\PothosSDR\bin",
    r"C:\Program Files (x86)\PothosSDR\bin",
]
def find_tool(name):
    for d in CANDIN:
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    # fall back to PATH
    return name

HACKRF_INFO = find_tool("hackrf_info.exe")
HACKRF_XFER = find_tool("hackrf_transfer.exe")

FREQ = 2426000000
RATE = 20000000
OUTDIR = os.environ.get("TEMP", ".")

def run(cmd, timeout):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except subprocess.TimeoutExpired as e:
        out = ""
        if e.stdout: out += e.stdout if isinstance(e.stdout, str) else e.stdout.decode(errors="ignore")
        if e.stderr: out += e.stderr if isinstance(e.stderr, str) else e.stderr.decode(errors="ignore")
        return 0, out            # timeout is expected for capture (we cap the run)

def sz(path):
    try: return os.path.getsize(path)
    except OSError: return 0

def line(): print("-" * 66)

print("=" * 66)
print(" HackRF plug-and-play health check (PC/USB)")
print("=" * 66)
print(f" info tool: {HACKRF_INFO}")
print(f" xfer tool: {HACKRF_XFER}")

# ---- [1] presence --------------------------------------------------------
print("\n[1] is the board visible?")
rc, out = run([HACKRF_INFO], 15)
if "Found HackRF" in out:
    ser = next((l.strip() for l in out.splitlines() if "Serial number" in l), "serial?")
    fw  = next((l.strip() for l in out.splitlines() if "Firmware Version" in l), "fw?")
    print(f"    PASS - {ser}")
    print(f"           {fw}")
else:
    print("    FAIL - no HackRF found on USB.")
    print("    -> bad cable/port/power, or board not in normal mode.")
    print(out)
    sys.exit(1)

# ---- [2] short capture ---------------------------------------------------
print("\n[2] SHORT capture (1 s) - can it start streaming at all?")
short = os.path.join(OUTDIR, "hacktest_short.bin")
if os.path.exists(short): os.remove(short)
rc, out = run([HACKRF_XFER, "-r", short, "-f", str(FREQ), "-s", str(RATE),
               "-l", "16", "-g", "20", "-a", "0", "-n", str(RATE)], 20)  # -n = stop after RATE samples (~1s)
s = sz(short)
print(f"    bytes={s:,}")
if s > 1_000_000:
    print("    PASS - it streams.")
else:
    print("    FAIL - board is visible but won't stream.")
    print("    ---- hackrf_transfer output ----")
    print(out)
    print("    -> device-busy, USB, or firmware issue. If another program holds")
    print("       the HackRF (SDR#, GQRX, a daemon), close it and retry.")
    sys.exit(1)

# ---- [3] sustained capture ----------------------------------------------
print("\n[3] SUSTAINED capture (10 s) - can it hold 40 MB/s?")
lng = os.path.join(OUTDIR, "hacktest_long.bin")
if os.path.exists(lng): os.remove(lng)
t0 = time.time()
rc, out = run([HACKRF_XFER, "-r", lng, "-f", str(FREQ), "-s", str(RATE),
               "-l", "16", "-g", "20", "-a", "0", "-n", str(RATE * 10)], 25)  # ~10s
dt = time.time() - t0
s = sz(lng)
expect = RATE * 2 * 9   # ~9s of good bytes
rate_mbs = s / dt / 1e6 if dt > 0 else 0
print(f"    bytes={s:,}  elapsed={dt:.1f}s  avg={rate_mbs:.1f} MB/s  (expect >{expect:,} bytes)")
if s > expect:
    print("    PASS - sustained streaming works.")
else:
    print("    FAIL/PARTIAL - streamed some, then fell short.")
    print("    ---- hackrf_transfer output ----")
    print(out)
    print("    -> sustained-throughput/power problem: USB can't hold the rate,")
    print("       or power sags under load. Try a different port / powered hub /")
    print("       shorter USB cable.")
    sys.exit(1)

# ---- [4] error scan ------------------------------------------------------
print("\n[4] scanning for overflow / USB errors in the stream log...")
bad = [l for l in out.splitlines()
       if any(k in l.lower() for k in ("overrun", "overflow", "usb error", "couldn't transfer", "error"))]
if bad:
    print("    WARNING - errors reported under load:")
    for l in bad[:10]: print("      " + l.strip())
    print("    -> streams but marginal (throughput/power).")
else:
    print("    clean - no overflow/USB errors.")

# ---- verdict -------------------------------------------------------------
print("\n" + "=" * 66)
print(" VERDICT: this board passed presence + short + sustained streaming")
print(" on the PC's USB. The board itself is healthy.")
print(" If it fails only on the Pi, the problem is that Pi's USB power/port")
print(" or the daemon's start/stop handoff - not the HackRF.")
print("=" * 66)

# cleanup
for f in (short, lng):
    try: os.remove(f)
    except OSError: pass