#!/usr/bin/env python3
"""
HackRF capture daemon for Raspberry Pi 5 -- STANDARD LIBRARY ONLY (no ZeroMQ).

Runs on EACH Pi. Controlled by the main PC over plain TCP. Deployment is just:
copy this one file and run it. Nothing to install.

Data path:
    hackrf_transfer -r -            (raw int8 IQ; MAGIC marker embedded by firmware)
      -> reader thread              (drains the USB pipe so the HackRF never
                                     overflows because of a Pi- or network-side stall)
      -> bounded queue              (ride-through buffer for transient stalls)
      -> sender thread -> TCP       -> the PC (PC listens, Pi connects)

    The data connection is a raw, continuous int8 stream (no framing). Message
    boundaries are unnecessary: the inline MAGIC marker provides all structure,
    and the PC detects it directly in the byte stream. A dropped chunk simply
    becomes one flagged marker-gap window on the PC, then normal flow resumes.

Control (PC -> Pi): newline-delimited JSON over TCP, one command per connection:
    {"cmd":"ping"}  {"cmd":"start", ...}  {"cmd":"stop"}  {"cmd":"status"}  {"cmd":"shutdown"}

Loss accounting (nothing is silent):
    hackrf_overflows -> USB/capture behind (rare on Pi 5 USB 3)
    queue_drops      -> network/PC behind  (whole chunk dropped -> flagged on PC)
    send_errors      -> data socket died   (capture ends, reported via status)
    TCP              -> guarantees no additional in-transit gaps
    inline MAGIC     -> independent PC-side cross-check (samples between markers)
"""

import argparse
import json
import os
import queue
import shutil
import signal
import socket
import subprocess
import threading
import time

# --- inline marker: MUST match the firmware that injects it ------------------
MAGIC = bytes([0x7F, 0x80, 0x7F, 0x80,
               0x00, 0xFF, 0x00, 0xFF,
               0x55, 0xAA, 0x55, 0xAA])
BYTES_PER_SAMPLE = 2   # int8 I + int8 Q

DEFAULTS = {
    "node_id": 0,
    "control_port": 5555,
    "chunk_bytes": 262144,     # 256 KiB read/send granularity (~6.4 ms @ 40 MB/s)
    "queue_max": 256,          # ~64 MB / ~1.6 s ride-through buffer
    "so_sndbuf": 8 << 20,      # 8 MB kernel send buffer hint (kernel may clamp)
    # capture defaults (the PC 'start' command overrides these):
    "freq": 2400000000,
    "sample_rate": 20000000,
    "lna": 16,
    "vga": 20,
    "amp": 0,
}


def find_hackrf_transfer():
    for c in ("hackrf_transfer",
              "/usr/bin/hackrf_transfer",
              "/usr/local/bin/hackrf_transfer"):
        p = shutil.which(c) if "/" not in c else (c if os.path.exists(c) else None)
        if p:
            return p
    return None


def parse_dest(dest):
    """Accept 'tcp://10.42.0.10:5560' or '10.42.0.10:5560'."""
    d = dest.strip()
    if d.startswith("tcp://"):
        d = d[6:]
    host, _, port = d.rpartition(":")
    return host, int(port)


class Stats:
    """Each field has a single writer thread; torn reads are fine for telemetry."""
    def __init__(self):
        self.state = "idle"           # idle | running | error | stopped
        self.error = ""
        self.start_ts = 0.0
        self.bytes_sent = 0
        self.chunks_sent = 0
        self.queue_drops = 0          # reader couldn't enqueue -> net/PC slow
        self.send_errors = 0          # data socket send failed  -> connection dead
        self.hackrf_overflows = 0     # parsed from stderr        -> USB slow
        self.marker_count = 0
        self.samples_since_marker = 0
        self.last_marker_ts = 0.0
        self.last_power_dbfs = None
        self.last_rate_mibs = None

    def snapshot(self, node_id):
        up = (time.time() - self.start_ts) if self.start_ts else 0.0
        return {
            "node": node_id, "state": self.state, "error": self.error,
            "uptime_s": round(up, 1),
            "bytes_sent": self.bytes_sent, "chunks_sent": self.chunks_sent,
            "queue_drops": self.queue_drops, "send_errors": self.send_errors,
            "hackrf_overflows": self.hackrf_overflows,
            "marker_count": self.marker_count,
            "samples_since_marker": self.samples_since_marker,
            "avg_power_dbfs": self.last_power_dbfs,
            "rate_mibs": self.last_rate_mibs,
            "ts": time.time(),
        }


class HackRFCapture:
    def __init__(self, cfg):
        self.cfg = cfg
        self.stats = Stats()
        self.proc = None
        self.sock = None
        self.q = None
        self._threads = []
        self._running = threading.Event()
        self._lock = threading.Lock()
        self.bin = find_hackrf_transfer()

    def is_running(self):
        return self._running.is_set()

    def _reap(self):
        for t in self._threads:
            t.join(timeout=1.0)
        self._threads = []
        if self.sock is not None:
            try:
                self.sock.close()
            except OSError:
                pass
            self.sock = None
        self.proc = None

    def _connect_data(self, dest):
        """Connect the raw data stream to the PC, with a few retries for bring-up."""
        host, port = parse_dest(dest)
        last_err = None
        for _ in range(10):                       # ~5 s window for the PC listener
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF,
                             int(self.cfg["so_sndbuf"]))
                s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                s.settimeout(5.0)
                s.connect((host, port))
                s.settimeout(None)                # blocking sendall => backpressure
                return s, None
            except OSError as e:
                last_err = e
                try:
                    s.close()
                except OSError:
                    pass
                time.sleep(0.5)
        return None, f"connect {host}:{port} failed: {last_err}"

    def start(self, params):
        with self._lock:
            if self._running.is_set():
                return False, "already running"
            if self.proc is not None:
                self._reap()
            if not self.bin:
                return False, "hackrf_transfer not found on PATH"
            dest = params.get("dest")
            if not dest:
                return False, "missing 'dest' (e.g. 10.42.0.10:5560)"

            # Open the data socket FIRST so we fail cleanly if the PC isn't ready,
            # without ever starting the HackRF.
            sock, err = self._connect_data(dest)
            if sock is None:
                return False, err

            freq = int(params.get("freq", self.cfg["freq"]))
            rate = int(params.get("sample_rate", self.cfg["sample_rate"]))
            lna = int(params.get("lna", self.cfg["lna"]))
            vga = int(params.get("vga", self.cfg["vga"]))
            amp = int(params.get("amp", self.cfg["amp"]))

            cmd = [self.bin, "-r", "-",
                   "-f", str(freq), "-s", str(rate),
                   "-l", str(lna), "-g", str(vga),
                   "-a", str(1 if amp else 0)]

            self.sock = sock
            self.q = queue.Queue(maxsize=int(self.cfg["queue_max"]))
            self.stats = Stats()
            self.stats.state = "running"
            self.stats.start_ts = time.time()

            try:
                self.proc = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            except Exception as e:
                try:
                    self.sock.close()
                except OSError:
                    pass
                self.sock = None
                self.stats.state = "error"
                self.stats.error = f"spawn: {e}"
                return False, str(e)

            self._running.set()
            self._threads = [
                threading.Thread(target=self._reader, daemon=True, name="reader"),
                threading.Thread(target=self._sender, daemon=True, name="sender"),
                threading.Thread(target=self._stderr, daemon=True, name="stderr"),
            ]
            for t in self._threads:
                t.start()
            return True, {"cmd": " ".join(cmd), "dest": dest}

    def _reader(self):
        """Continuously drain hackrf_transfer stdout so the HackRF never blocks."""
        chunk = int(self.cfg["chunk_bytes"])
        out = self.proc.stdout
        try:
            while self._running.is_set():
                data = out.read(chunk)          # blocks to fill chunk, short at EOF
                if not data:                    # EOF -> hackrf_transfer exited
                    break
                try:
                    self.q.put_nowait(data)
                except queue.Full:
                    # net/PC too slow: drop OLDEST whole chunk, keep flowing.
                    try:
                        self.q.get_nowait()
                    except queue.Empty:
                        pass
                    self.stats.queue_drops += 1
                    try:
                        self.q.put_nowait(data)
                    except queue.Full:
                        pass
        except Exception as e:
            self.stats.error = f"reader: {e}"
        finally:
            if self._running.is_set():          # we did NOT ask it to stop
                rc = self.proc.poll()
                self.stats.state = "error"
                self.stats.error = f"hackrf_transfer exited rc={rc}"
            self._running.clear()

    def _sender(self):
        """Stream chunks over TCP (blocking sendall = backpressure). Marker telemetry."""
        off = 0
        last_marker_global = None
        sock = self.sock
        while self._running.is_set() or (self.q is not None and not self.q.empty()):
            try:
                data = self.q.get(timeout=0.3)
            except queue.Empty:
                continue
            if MAGIC in data:                    # cheap: PC does the exact per-sec check
                self.stats.marker_count += data.count(MAGIC)
                last_marker_global = off + data.rfind(MAGIC)
                self.stats.last_marker_ts = time.time()
            try:
                sock.sendall(data)
                self.stats.bytes_sent += len(data)
                self.stats.chunks_sent += 1
                off += len(data)
                if last_marker_global is not None:
                    self.stats.samples_since_marker = \
                        (off - last_marker_global) // BYTES_PER_SAMPLE
            except OSError as e:
                self.stats.send_errors += 1
                self.stats.error = f"data socket: {e}"
                self._running.clear()            # connection dead -> end capture
                break

    def _stderr(self):
        """Drain stderr (prevents stall) and parse health lines."""
        err = self.proc.stderr
        try:
            for raw in iter(err.readline, b""):
                line = raw.decode(errors="replace").strip()
                low = line.lower()
                if "overrun" in low or "dropped" in low or "usb error" in low:
                    self.stats.hackrf_overflows += 1
                if "mib/second" in low:
                    try:
                        self.stats.last_rate_mibs = \
                            float(line.split("=")[1].split("MiB")[0].strip())
                    except Exception:
                        pass
                    if "power" in low:
                        try:
                            self.stats.last_power_dbfs = \
                                float(low.split("power")[1].split("dbfs")[0].strip())
                        except Exception:
                            pass
        except Exception:
            pass

    def stop(self):
        with self._lock:
            if self.proc is None:
                return True, "not running"
            self._running.clear()
            # Close the data socket first to unblock a blocked sendall.
            if self.sock is not None:
                try:
                    self.sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
            p = self.proc
            try:
                p.send_signal(signal.SIGINT)     # clean Ctrl-C; hackrf handles it
            except Exception:
                pass
            try:
                p.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                try:
                    p.terminate()
                    p.wait(timeout=2.0)
                except Exception:
                    try:
                        p.kill()
                    except Exception:
                        pass
            for t in self._threads:
                t.join(timeout=2.0)
            self._threads = []
            if self.sock is not None:
                try:
                    self.sock.close()
                except OSError:
                    pass
                self.sock = None
            snap = self.stats.snapshot(self.cfg["node_id"])
            self.proc = None
            if self.stats.state == "running":
                self.stats.state = "stopped"
            return True, snap


def recv_line(conn):
    buf = b""
    while b"\n" not in buf:
        chunk = conn.recv(4096)
        if not chunk:
            break
        buf += chunk
    return buf


class ControlServer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.cap = HackRFCapture(cfg)
        self._stop = threading.Event()

    def _handle(self, msg):
        cmd = msg.get("cmd")
        nid = self.cfg["node_id"]
        if cmd == "ping":
            return {"ok": True, "node": nid, "state": self.cap.stats.state,
                    "ts": time.time()}
        if cmd == "status":
            return {"ok": True, **self.cap.stats.snapshot(nid)}
        if cmd == "start":
            ok, info = self.cap.start(msg)
            return {"ok": ok, "node": nid, "info": info}
        if cmd == "stop":
            ok, info = self.cap.stop()
            return {"ok": ok, "node": nid, "info": info}
        if cmd == "shutdown":
            self._stop.set()
            return {"ok": True, "node": nid, "info": "daemon shutting down"}
        return {"ok": False, "error": f"unknown cmd {cmd!r}"}

    def run(self):
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("", int(self.cfg["control_port"])))
        srv.listen(8)
        srv.settimeout(0.5)

        print(f"[node {self.cfg['node_id']}] control on :{self.cfg['control_port']}  "
              f"(hackrf_transfer: {self.cap.bin})  stdlib TCP, no ZMQ", flush=True)

        try:
            signal.signal(signal.SIGTERM, lambda *_: self._stop.set())
            signal.signal(signal.SIGINT, lambda *_: self._stop.set())
        except ValueError:
            pass  # not in the main thread (e.g. embedded/tested) -> skip handlers

        while not self._stop.is_set():
            try:
                conn, _addr = srv.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            try:
                conn.settimeout(5.0)
                line = recv_line(conn)
                if line:
                    try:
                        msg = json.loads(line.decode())
                    except Exception as e:
                        reply = {"ok": False, "error": f"bad json: {e}"}
                    else:
                        try:
                            reply = self._handle(msg)
                        except Exception as e:
                            reply = {"ok": False, "error": str(e)}
                    conn.sendall((json.dumps(reply) + "\n").encode())
            except OSError:
                pass
            finally:
                try:
                    conn.close()
                except OSError:
                    pass

        try:
            self.cap.stop()
        except Exception:
            pass
        srv.close()
        print(f"[node {self.cfg['node_id']}] stopped.", flush=True)


def load_config(path):
    cfg = dict(DEFAULTS)
    if path and os.path.exists(path):
        with open(path) as f:
            cfg.update(json.load(f))
    return cfg


def main():
    ap = argparse.ArgumentParser(description="HackRF capture daemon (Pi side, stdlib)")
    ap.add_argument("--config", default="/opt/hackrf-capture/capture_config.json")
    ap.add_argument("--node-id", type=int)
    ap.add_argument("--control-port", type=int)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.node_id is not None:
        cfg["node_id"] = args.node_id
    if args.control_port is not None:
        cfg["control_port"] = args.control_port

    ControlServer(cfg).run()


if __name__ == "__main__":
    main()
