#!/usr/bin/env python3
"""
PC-side controller + collector for the 3-node HackRF array -- STDLIB ONLY.

Runs on the MAIN PC. Uses only Python's standard library (no ZeroMQ), so the
only install anywhere is Python itself.

What it does:
  * Sends control commands to each Pi over TCP (newline-delimited JSON):
    start / stop / status.
  * Listens on one TCP port per node; each Pi connects and streams raw int8 IQ.
  * Writes ONE file per node to the SSD, rotating on the inline MAGIC marker so
    file K on every node begins on the same reference edge (free alignment).
  * Auto-calibrates each node's real samples-per-marker (they drift ~166 ppm,
    e.g. 19,996,672 instead of 20,000,000) and flags any marker gap that
    deviates -> loss located to a 1-second window.
  * 'monitor' polls each Pi's status once a second for a live dashboard.

Usage:
    python pc_control.py collect     # listen, start all Pis, write to SSD (Ctrl-C stops)
    python pc_control.py monitor     # live per-node dashboard (run in a 2nd terminal)
    python pc_control.py start        # just tell all Pis to start
    python pc_control.py stop         # just tell all Pis to stop
    python pc_control.py status       # one-shot status query
"""

import argparse
import json
import os
import socket
import threading
import time

MAGIC = bytes([0x7F, 0x80, 0x7F, 0x80,
               0x00, 0xFF, 0x00, 0xFF,
               0x55, 0xAA, 0x55, 0xAA])
BYTES_PER_SAMPLE = 2

# =========================== EDIT FOR YOUR NETWORK ===========================
PC_IP_FOR_PIS = "10.42.0.10"        # this PC's address on the array LAN (Pis connect here)

NODES = [
    {"id": 0, "host": "10.42.0.11", "ctrl": 5555, "data_port": 5560},
    {"id": 1, "host": "10.42.0.12", "ctrl": 5555, "data_port": 5561},
    {"id": 2, "host": "10.42.0.13", "ctrl": 5555, "data_port": 5562},
]

OUT_DIR = r"G:\GITFINAL\capture"        # SSD path (NVMe recommended; ~120 MB/s aggregate)

# capture parameters pushed to every Pi on 'start'
FREQ = 2426000000
SAMPLE_RATE = 20000000
LNA, VGA, AMP = 16, 20, 0

# file rotation + loss detection
MARKERS_PER_FILE = 60               # ~60 s per file at 1 pps
SAMPLES_PER_MARKER = 20000000       # nominal only; real rate is auto-calibrated per node
MARKER_TOLERANCE = 20000            # samples of deviation from calibrated gap before flagging
# ============================================================================


# ----------------------------- control helpers ------------------------------
def send_cmd(node, obj, timeout=3.0):
    """One TCP round-trip: connect, send JSON+newline, read one JSON reply line."""
    try:
        with socket.create_connection((node["host"], node["ctrl"]), timeout=timeout) as s:
            s.settimeout(timeout)
            s.sendall((json.dumps(obj) + "\n").encode())
            buf = b""
            while b"\n" not in buf:
                chunk = s.recv(4096)
                if not chunk:
                    break
                buf += chunk
            return json.loads(buf.decode()) if buf else {"ok": False, "error": "no reply"}
    except Exception as e:
        return {"ok": False, "error": f"{e} (Pi unreachable / daemon down)"}


def start_all():
    for n in NODES:
        dest = f"{PC_IP_FOR_PIS}:{n['data_port']}"
        r = send_cmd(n, {"cmd": "start", "dest": dest,
                         "freq": FREQ, "sample_rate": SAMPLE_RATE,
                         "lna": LNA, "vga": VGA, "amp": AMP})
        print(f"node {n['id']} start: {r}")


def stop_all():
    for n in NODES:
        print(f"node {n['id']} stop: {send_cmd(n, {'cmd': 'stop'})}")


def status_all():
    for n in NODES:
        print(f"node {n['id']} status: {send_cmd(n, {'cmd': 'status'})}")


# ------------------------------- collector ----------------------------------
class NodeCollector(threading.Thread):
    """Listen for one node, read its raw stream, write to SSD, rotate at markers."""

    def __init__(self, node, stop_evt):
        super().__init__(daemon=True, name=f"collector-{node['id']}")
        self.node = node
        self.stop = stop_evt

        self.srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.srv.bind(("", node["data_port"]))
        self.srv.listen(1)
        self.srv.settimeout(0.5)
        self.conn = None

        self.f = None
        self.file_idx = 0
        self.global_off = 0            # total bytes received (stream offset)
        self.markers = 0
        self.prev_marker_off = None
        self.nominal_gap = None        # this node's calibrated samples/marker
        self.tail = b""                # carry for markers spanning recv boundaries
        self.bytes_written = 0
        self.loss_flags = 0

    def _open_new(self):
        if self.f:
            self.f.close()
        name = os.path.join(OUT_DIR, f"node{self.node['id']}_{self.file_idx:05d}.iq")
        self.f = open(name, "wb", buffering=1024 * 1024)
        self.file_idx += 1

    def _marker_check(self, marker_off):
        if self.prev_marker_off is not None:
            gap = (marker_off - self.prev_marker_off) // BYTES_PER_SAMPLE
            if self.nominal_gap is None and gap > SAMPLES_PER_MARKER * 0.8:
                self.nominal_gap = gap                      # calibrate to real rate
            elif self.nominal_gap:
                dev = abs(gap - self.nominal_gap)
                if dev > MARKER_TOLERANCE:
                    self.loss_flags += 1
                    print(f"[node {self.node['id']}] LOSS @ marker {self.markers}: "
                          f"gap={gap} samples (calibrated {self.nominal_gap}, "
                          f"dev {dev}) -> discard this window", flush=True)
        self.prev_marker_off = marker_off

    def _process(self, data):
        buf = self.tail + data
        base = self.global_off - len(self.tail)   # global offset of buf[0]

        idx = buf.find(MAGIC)
        write_from = 0
        while idx != -1:
            self.markers += 1
            self._marker_check(base + idx)
            if self.markers % MARKERS_PER_FILE == 0:
                self.f.write(buf[write_from:idx])
                self.bytes_written += idx - write_from
                self._open_new()               # new file STARTS at this marker
                write_from = idx
            idx = buf.find(MAGIC, idx + 1)

        hold = len(MAGIC) - 1                   # hold back a sliver for boundary markers
        end = len(buf) - hold if len(buf) > hold else 0
        if end > write_from:
            self.f.write(buf[write_from:end])
            self.bytes_written += end - write_from
            self.tail = buf[end:]
        else:
            self.tail = buf[write_from:]

        self.global_off += len(data)

    def run(self):
        self._open_new()
        # wait for the Pi to connect
        addr = None
        while not self.stop.is_set() and self.conn is None:
            try:
                self.conn, addr = self.srv.accept()
            except socket.timeout:
                continue
            except OSError:
                break

        if self.conn is not None:
            self.conn.settimeout(0.5)
            try:
                self.conn.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8 << 20)
            except OSError:
                pass
            print(f"[node {self.node['id']}] data connected from {addr}", flush=True)
            while not self.stop.is_set():
                try:
                    data = self.conn.recv(65536)
                except socket.timeout:
                    continue
                except OSError:
                    break
                if not data:                    # Pi closed the stream
                    print(f"[node {self.node['id']}] stream ended (Pi closed)", flush=True)
                    break
                self._process(data)

        if self.conn is not None:
            try:
                self.conn.close()
            except OSError:
                pass
        try:
            self.srv.close()
        except OSError:
            pass
        if self.f:
            self.f.close()
        print(f"[node {self.node['id']}] closed: files={self.file_idx} "
              f"bytes={self.bytes_written} markers={self.markers} "
              f"loss_flags={self.loss_flags}", flush=True)


def cmd_collect():
    os.makedirs(OUT_DIR, exist_ok=True)
    stop_evt = threading.Event()
    collectors = [NodeCollector(n, stop_evt) for n in NODES]
    for c in collectors:
        c.start()
    time.sleep(0.3)                 # ensure listeners are up before Pis connect
    start_all()
    print(f"\ncollecting to {OUT_DIR} -- Ctrl-C to stop\n", flush=True)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nstopping...", flush=True)
    stop_all()
    stop_evt.set()
    for c in collectors:
        c.join(timeout=3)


# ------------------------------- dashboard ----------------------------------
def cmd_monitor():
    print("polling status... Ctrl-C to stop")
    try:
        while True:
            rows = []
            for n in NODES:
                d = send_cmd(n, {"cmd": "status"}, timeout=1.5)
                rows.append((n["id"], d))
            os.system("clear")
            print(f"{'node':>4} {'state':>8} {'up_s':>6} {'MB_sent':>9} "
                  f"{'qdrop':>6} {'serr':>5} {'ovf':>4} {'marks':>6} "
                  f"{'since_mark':>11} {'pow_dBfs':>9} {'MiB/s':>6}")
            for nid, d in rows:
                if not d.get("ok"):
                    print(f"{nid:>4} {'DOWN':>8}   {d.get('error','')[:60]}")
                    continue
                pw = d.get("avg_power_dbfs")
                rt = d.get("rate_mibs")
                print(f"{nid:>4} {d['state']:>8} {d['uptime_s']:>6.0f} "
                      f"{d['bytes_sent'] / 1e6:>9.1f} {d['queue_drops']:>6} "
                      f"{d['send_errors']:>5} {d['hackrf_overflows']:>4} "
                      f"{d['marker_count']:>6} {d['samples_since_marker']:>11} "
                      f"{(pw if pw is not None else 0):>9.1f} "
                      f"{(rt if rt is not None else 0):>6.1f}")
            time.sleep(1)
    except KeyboardInterrupt:
        pass


def main():
    ap = argparse.ArgumentParser(description="PC-side control + collect (stdlib TCP)")
    ap.add_argument("command",
                    choices=["collect", "monitor", "start", "stop", "status"])
    args = ap.parse_args()

    if args.command == "collect":
        cmd_collect()
    elif args.command == "monitor":
        cmd_monitor()
    elif args.command == "start":
        start_all()
    elif args.command == "stop":
        stop_all()
    elif args.command == "status":
        status_all()


if __name__ == "__main__":
    main()
