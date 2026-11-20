#!/usr/bin/env python3
"""
Minimal PPS interrupt test.
1 HackRF, 1 GPIO wire from GPSDO PPS to HackRF P2_8.
Just proves the firmware interrupt is firing.

Usage:
  Terminal 1: python pps_receiver.py
  Terminal 2: python pps_interrupt_test.py
"""

import socket
import struct
import time
import numpy as np
import osmosdr
from gnuradio import gr, blocks

SYNC_WORD   = bytes([0x7F, 0x7F, 0x7F, 0x7F, 0x80, 0x80, 0x80, 0x80])
SYNC_LEN    = len(SYNC_WORD)
MARKER_LEN  = 12
PC_IP       = '127.0.0.1'
PC_PORT     = 5005
SAMPLE_RATE = 20e6
CENTER_FREQ = 2.42e9


class PPSProbe(gr.sync_block):
    """Scans IQ bytes for PPS sync marker and prints when found."""

    def __init__(self):
        gr.sync_block.__init__(self, name='PPSProbe',
                               in_sig=[np.complex64],
                               out_sig=None)
        self.sock           = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sample_counter = 0
        self._leftover      = bytearray()
        self.last_pps       = None

    def work(self, input_items, output_items):
        inp = input_items[0]

        raw = inp.view(np.int8).tobytes()
        buf = self._leftover + bytearray(raw)

        idx = 0
        while idx <= len(buf) - MARKER_LEN:
            if buf[idx:idx + SYNC_LEN] == SYNC_WORD:
                pps_count = struct.unpack('>I',
                    buf[idx + SYNC_LEN:idx + SYNC_LEN + 4])[0]
                global_sample = self.sample_counter + (idx // 2)
                wall_time = time.time()

                # Print to terminal
                if self.last_pps is not None:
                    delta = wall_time - self.last_pps
                    print(f"PPS #{pps_count:>6} | sample {global_sample:<15} | "
                          f"wall {wall_time:.3f} | delta {delta*1000:.1f}ms")
                else:
                    print(f"PPS #{pps_count:>6} | sample {global_sample:<15} | "
                          f"wall {wall_time:.3f} | first pulse!")
                self.last_pps = wall_time

                # Also send to pps_receiver.py
                packet = struct.pack('>BIQd', 0, pps_count, global_sample, wall_time)
                self.sock.sendto(packet, (PC_IP, PC_PORT))

                idx += MARKER_LEN
            else:
                idx += 1

        self._leftover = bytearray(buf[max(0, len(buf) - MARKER_LEN):])
        self.sample_counter += len(inp)
        return len(inp)


class PPSTest(gr.top_block):

    def __init__(self):
        gr.top_block.__init__(self, "PPS Interrupt Test")

        self.source = osmosdr.source(args="numchan=1 hackrf=0")
        self.source.set_sample_rate(SAMPLE_RATE)
        self.source.set_center_freq(CENTER_FREQ, 0)
        self.source.set_gain_mode(False, 0)
        self.source.set_gain(14, 0)
        self.source.set_if_gain(24, 0)
        self.source.set_bb_gain(28, 0)
        self.source.set_bandwidth(SAMPLE_RATE, 0)

        self.probe = PPSProbe()
        self.sink  = blocks.null_sink(gr.sizeof_gr_complex)

        self.connect((self.source, 0), (self.probe, 0))

    def start(self):
        super().start()


def main():
    print("PPS Interrupt Test")
    print(f"HackRF: 1 unit at {CENTER_FREQ/1e9:.3f} GHz, {SAMPLE_RATE/1e6:.0f} Msps")
    print(f"Sending to {PC_IP}:{PC_PORT}")
    print("Waiting for PPS pulses — should fire once per second...\n")
    print(f"{'PPS #':<10} {'Sample Index':<20} {'Wall Time':<20} {'Delta'}")
    print("-" * 65)

    tb = PPSTest()
    tb.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        tb.stop()
        tb.wait()
        print("\nStopped")


if __name__ == '__main__':
    main()