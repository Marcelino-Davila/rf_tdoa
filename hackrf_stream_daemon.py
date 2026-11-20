#!/usr/bin/env python3
"""
hackrf_stream_daemon.py  —  Pi-side HackRF capture daemon
Captures raw IQ from HackRF at 20 Msps and streams over TCP to a PC.
PPS sync markers from custom firmware pass through untouched.

USB RX callback and TCP send run in separate threads via a queue,
so neither can stall the other. Achieves full 40 MB/s throughput.

Usage: python3 hackrf_stream_daemon.py
Runs forever. Reconnects automatically if the PC disconnects.
"""

import ctypes
import ctypes.util
import socket
import time
import sys
import signal
import threading
import queue

# ─── CONSTANTS ────────────────────────────────────────────────────────────────
PC_HOST         = "10.42.0.1"      # IP of your PC
PC_PORT         = 7373             # TCP port — 7371=pi1, 7372=pi2, 7373=pi3
FREQUENCY_HZ    = 2_400_000_000   # 2.4 GHz
SAMPLE_RATE_HZ  = 20_000_000      # 20 Msps
LNA_GAIN_DB     = 14              # LNA gain
VGA_GAIN_DB     = 20              # VGA (baseband) gain
AMP_ENABLE      = 0               # RF amp: 0=off, 1=on
QUEUE_MAXSIZE   = 128             # max chunks buffered (128 x 256KB = 32MB max RAM)
# ──────────────────────────────────────────────────────────────────────────────

# ─── libhackrf ctypes setup ───────────────────────────────────────────────────
lib_name = ctypes.util.find_library("hackrf")
if lib_name is None:
    print("[FATAL] libhackrf not found. Run: sudo apt install libhackrf0")
    sys.exit(1)

_lib = ctypes.CDLL(lib_name)

HACKRF_SUCCESS = 0

class hackrf_transfer(ctypes.Structure):
    pass

CALLBACK = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(hackrf_transfer))

hackrf_transfer._fields_ = [
    ("device",          ctypes.c_void_p),
    ("buffer",          ctypes.POINTER(ctypes.c_uint8)),
    ("buffer_length",   ctypes.c_int),
    ("valid_length",    ctypes.c_int),
    ("rx_ctx",          ctypes.c_void_p),
    ("tx_ctx",          ctypes.c_void_p),
]

_lib.hackrf_init.restype                  = ctypes.c_int
_lib.hackrf_open.restype                  = ctypes.c_int
_lib.hackrf_open.argtypes                 = [ctypes.POINTER(ctypes.c_void_p)]
_lib.hackrf_close.restype                 = ctypes.c_int
_lib.hackrf_close.argtypes                = [ctypes.c_void_p]
_lib.hackrf_exit.restype                  = ctypes.c_int
_lib.hackrf_set_sample_rate.restype       = ctypes.c_int
_lib.hackrf_set_sample_rate.argtypes      = [ctypes.c_void_p, ctypes.c_double]
_lib.hackrf_set_freq.restype              = ctypes.c_int
_lib.hackrf_set_freq.argtypes             = [ctypes.c_void_p, ctypes.c_uint64]
_lib.hackrf_set_lna_gain.restype          = ctypes.c_int
_lib.hackrf_set_lna_gain.argtypes         = [ctypes.c_void_p, ctypes.c_uint32]
_lib.hackrf_set_vga_gain.restype          = ctypes.c_int
_lib.hackrf_set_vga_gain.argtypes         = [ctypes.c_void_p, ctypes.c_uint32]
_lib.hackrf_set_amp_enable.restype        = ctypes.c_int
_lib.hackrf_set_amp_enable.argtypes       = [ctypes.c_void_p, ctypes.c_uint8]
_lib.hackrf_set_hw_sync_mode.restype      = ctypes.c_int
_lib.hackrf_set_hw_sync_mode.argtypes     = [ctypes.c_void_p, ctypes.c_uint8]
_lib.hackrf_start_rx.restype              = ctypes.c_int
_lib.hackrf_start_rx.argtypes             = [ctypes.c_void_p, CALLBACK, ctypes.c_void_p]
_lib.hackrf_stop_rx.restype               = ctypes.c_int
_lib.hackrf_stop_rx.argtypes              = [ctypes.c_void_p]
_lib.hackrf_is_streaming.restype          = ctypes.c_int
_lib.hackrf_is_streaming.argtypes         = [ctypes.c_void_p]
_lib.hackrf_error_name.restype            = ctypes.c_char_p
_lib.hackrf_error_name.argtypes           = [ctypes.c_int]
# ──────────────────────────────────────────────────────────────────────────────

# ─── Globals ──────────────────────────────────────────────────────────────────
_running     = True
_data_queue  = queue.Queue(maxsize=QUEUE_MAXSIZE)
_dropped     = 0
_bytes_usb   = 0
_bytes_sent  = 0
_last_print  = time.time()
# ──────────────────────────────────────────────────────────────────────────────

def check(ret, label):
    if ret != HACKRF_SUCCESS:
        name = _lib.hackrf_error_name(ret).decode()
        print(f"[ERROR] {label} failed: {name} ({ret})")
        sys.exit(1)


def rx_callback(transfer_ptr):
    global _bytes_usb, _dropped
    t = transfer_ptr.contents
    buf = ctypes.string_at(t.buffer, t.valid_length)   # C memcpy → bytes, no per-int boxing
    _bytes_usb += len(buf)
    try:
        _data_queue.put_nowait(buf)
    except queue.Full:
        _dropped += len(buf)
    return 0


def connect_to_pc():
    """Blocking connect loop — retries until PC receiver is up."""
    while _running:
        try:
            print(f"[INFO] Connecting to {PC_HOST}:{PC_PORT} ...")
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 8 * 1024 * 1024)
            s.connect((PC_HOST, PC_PORT))
            print(f"[INFO] Connected to PC.")
            return s
        except (ConnectionRefusedError, OSError) as e:
            print(f"[WARN] Cannot connect ({e}). Retrying in 3s...")
            time.sleep(3)


def sender_thread():
    """Dedicated thread: drains queue and sends to PC. Reconnects on disconnect."""
    global _bytes_sent, _running

    sock = None

    while _running:
        # Connect if needed
        if sock is None:
            sock = connect_to_pc()

        # Drain queue
        try:
            chunk = _data_queue.get(timeout=1.0)
        except queue.Empty:
            continue

        try:
            sock.sendall(chunk)
            _bytes_sent += len(chunk)
        except (BrokenPipeError, ConnectionResetError, OSError):
            print("[WARN] PC disconnected. Reconnecting...")
            try:
                sock.close()
            except:
                pass
            sock = None
            # Put chunk back so we don't lose it
            try:
                _data_queue.put_nowait(chunk)
            except queue.Full:
                pass

    if sock:
        try:
            sock.close()
        except:
            pass


def stats_thread():
    """Prints throughput and queue stats every second."""
    global _last_print

    while _running:
        time.sleep(1.0)
        now = time.time()
        usb_mb   = _bytes_usb  / 1e6
        sent_mb  = _bytes_sent / 1e6
        drop_mb  = _dropped    / 1e6
        q_depth  = _data_queue.qsize()
        print(f"[INFO] USB:{usb_mb:.1f}MB  Sent:{sent_mb:.1f}MB  "
              f"Dropped:{drop_mb:.1f}MB  Queue:{q_depth}/{QUEUE_MAXSIZE}")


def signal_handler(sig, frame):
    global _running
    print("\n[INFO] Shutting down...")
    _running = False


def main():
    global _running

    signal.signal(signal.SIGINT,  signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print("[INFO] hackrf_stream_daemon starting")
    print(f"[INFO] Frequency:   {FREQUENCY_HZ/1e6:.1f} MHz")
    print(f"[INFO] Sample rate: {SAMPLE_RATE_HZ/1e6:.0f} Msps")
    print(f"[INFO] LNA gain:    {LNA_GAIN_DB} dB")
    print(f"[INFO] VGA gain:    {VGA_GAIN_DB} dB")
    print(f"[INFO] Target:      {PC_HOST}:{PC_PORT}")

    check(_lib.hackrf_init(), "hackrf_init")

    device = ctypes.c_void_p()
    check(_lib.hackrf_open(ctypes.byref(device)), "hackrf_open")
    check(_lib.hackrf_set_sample_rate(device, float(SAMPLE_RATE_HZ)), "set_sample_rate")
    check(_lib.hackrf_set_freq(device, FREQUENCY_HZ),                  "set_freq")
    check(_lib.hackrf_set_lna_gain(device, LNA_GAIN_DB),               "set_lna_gain")
    check(_lib.hackrf_set_vga_gain(device, VGA_GAIN_DB),               "set_vga_gain")
    check(_lib.hackrf_set_amp_enable(device, AMP_ENABLE),              "set_amp_enable")
    check(_lib.hackrf_set_hw_sync_mode(device, 0),                     "set_hw_sync_mode")

    # Start sender and stats threads
    threading.Thread(target=sender_thread, daemon=True).start()
    threading.Thread(target=stats_thread,  daemon=True).start()

    # Start RX
    cb = CALLBACK(rx_callback)
    check(_lib.hackrf_start_rx(device, cb, None), "hackrf_start_rx")
    print("[INFO] RX started.")

    try:
        while _running:
            if _lib.hackrf_is_streaming(device) != 1:
                print("[ERROR] HackRF stopped streaming unexpectedly.")
                break
            time.sleep(0.5)
    finally:
        print("[INFO] Stopping...")
        _lib.hackrf_stop_rx(device)
        _lib.hackrf_close(device)
        _lib.hackrf_exit()
        print("[INFO] Done.")


if __name__ == "__main__":
    main()