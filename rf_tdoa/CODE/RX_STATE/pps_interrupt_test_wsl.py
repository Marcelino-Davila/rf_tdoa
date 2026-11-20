#!/usr/bin/env python3
"""
PPS Interrupt Test — uses libhackrf directly via ctypes.
No GNU Radio, no osmosdr, works in WSL2.
Filters false positives by only accepting detections ~1 second apart.

Usage: sudo python3 pps_interrupt_test_wsl.py
"""

import ctypes
import struct
import time
from datetime import datetime

SYNC_WORD   = bytes([0x7F, 0x80, 0x7F, 0x80, 0x00, 0xFF, 0x00, 0xFF, 0x55, 0xAA, 0x55, 0xAA])
SYNC_LEN    = len(SYNC_WORD)
MARKER_LEN  = 16
SAMPLE_RATE = 20000000
CENTER_FREQ = 2420000000
LNA_GAIN    = 16
VGA_GAIN    = 20
MIN_DELTA_MS = 800  # minimum ms between valid PPS detections

lib = ctypes.CDLL("libhackrf.so.0")

sample_counter  = 0
leftover        = bytearray()
last_pps_time   = None
last_pps_sample = None


class hackrf_transfer(ctypes.Structure):
    _fields_ = [
        ("device",        ctypes.c_void_p),
        ("buffer",        ctypes.POINTER(ctypes.c_uint8)),
        ("buffer_length", ctypes.c_int),
        ("valid_length",  ctypes.c_int),
        ("rx_ctx",        ctypes.c_void_p),
        ("tx_ctx",        ctypes.c_void_p),
    ]


CALLBACK_TYPE = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(hackrf_transfer))


def rx_callback(transfer):
    global sample_counter, leftover, last_pps_time, last_pps_sample
    print(f"callback: {transfer.contents.valid_length} bytes, sample={sample_counter}")


    length = transfer.contents.valid_length
    raw    = bytes(transfer.contents.buffer[:length])
    buf    = leftover + bytearray(raw)

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
    sample_counter += length // 2
    return 0


def main():
    print("PPS Interrupt Test (WSL2 / libhackrf direct)")
    print(f"Freq: {CENTER_FREQ/1e9:.3f} GHz  Rate: {SAMPLE_RATE/1e6:.0f} Msps")
    print(f"False positive filter: only printing if delta > {MIN_DELTA_MS}ms")
    print("Waiting for PPS pulses — one per second expected...\n")

    ret = lib.hackrf_init()
    if ret != 0:
        print(f"hackrf_init failed: {ret}")
        return

    device = ctypes.c_void_p()
    ret = lib.hackrf_open(ctypes.byref(device))
    if ret != 0:
        print(f"hackrf_open failed: {ret}")
        lib.hackrf_exit()
        return
    print("HackRF opened successfully")

    lib.hackrf_set_sample_rate(device, ctypes.c_double(SAMPLE_RATE))
    lib.hackrf_set_freq(device, ctypes.c_uint64(CENTER_FREQ))
    lib.hackrf_set_lna_gain(device, ctypes.c_uint32(LNA_GAIN))
    lib.hackrf_set_vga_gain(device, ctypes.c_uint32(VGA_GAIN))
    lib.hackrf_set_amp_enable(device, ctypes.c_uint8(0))

    print("HackRF configured — starting RX stream")
    print("Press Ctrl+C to stop\n")

    cb  = CALLBACK_TYPE(rx_callback)
    ret = lib.hackrf_start_rx(device, cb, None)
    lib.hackrf_set_transceiver_mode(device, ctypes.c_uint32(1))  # 1 = RX mode
    
    if ret != 0:
        print(f"hackrf_start_rx failed: {ret}")
        lib.hackrf_close(device)
        lib.hackrf_exit()
        return

    try:
        while True:
            time.sleep(0.1)
            if lib.hackrf_is_streaming(device) != 1:
                print("HackRF stopped streaming unexpectedly")
                break
    except KeyboardInterrupt:
        print("\nStopping...")

    lib.hackrf_stop_rx(device)
    lib.hackrf_close(device)
    lib.hackrf_exit()
    print("Done")


if __name__ == '__main__':
    main()