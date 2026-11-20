#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Channel Activity Monitor - Embedded Python Block
Computes per-channel energy across 2.4GHz WiFi band (2.400 - 2.460 GHz)
Drop this in as an embedded Python block in GNU Radio.

Each instance monitors one HackRF's stream.
Constructor args:
    center_freq   : center frequency of this HackRF (Hz)
    samp_rate     : sample rate (Hz), default 20e6
    channel_bw    : per-channel bandwidth to measure (Hz), default 10e6
    threshold_db  : energy threshold to flag a channel as active (dBFS)
"""

import numpy as np
from gnuradio import gr
import pmt

# WiFi 2.4GHz channel center frequencies (Hz)
WIFI_CHANNELS = {
    1:  2.412e9,
    2:  2.417e9,
    3:  2.422e9,
    4:  2.427e9,
    5:  2.432e9,
    6:  2.437e9,
    7:  2.442e9,
    8:  2.447e9,
    9:  2.452e9,
    10: 2.457e9,
    11: 2.462e9,
    12: 2.467e9,
    13: 2.472e9,
}

class blk(gr.sync_block):
    """
    Channel Activity Monitor
    Input:  complex IQ stream from one HackRF
    Output: passthrough (same complex stream)
    Messages: publishes channel activity report on 'activity' port
    """

    def __init__(self,
                 center_freq=2.42e9,
                 samp_rate=20e6,
                 channel_bw=10e6,
                 threshold_db=-60.0,
                 report_every_n_blocks=10):

        gr.sync_block.__init__(
            self,
            name='Channel Activity Monitor',
            in_sig=[np.complex64],
            out_sig=[np.complex64]
        )

        self.center_freq   = center_freq
        self.samp_rate     = samp_rate
        self.channel_bw    = channel_bw
        self.threshold_db  = threshold_db
        self.report_every  = report_every_n_blocks
        self.block_count   = 0

        # FFT size — power of 2, gives ~610 Hz/bin at 20 Msps
        self.fft_size = 32768

        # Precompute which FFT bins correspond to each WiFi channel
        self._build_channel_bins()

        # Message output port for activity reports
        self.message_port_register_out(pmt.intern('activity'))

        # State: rolling average energy per channel (dBFS)
        self.channel_energy_db = {ch: -120.0 for ch in WIFI_CHANNELS}

        print(f"[ChannelMonitor] Monitoring {len(self._channel_bins)} channels "
              f"visible from center={center_freq/1e9:.3f} GHz, "
              f"samp_rate={samp_rate/1e6:.1f} MHz")

    def _build_channel_bins(self):
        """Map each WiFi channel to FFT bin indices within our capture band."""
        hz_per_bin = self.samp_rate / self.fft_size
        half_ch_bw = self.channel_bw / 2.0
        half_sr    = self.samp_rate / 2.0

        self._channel_bins = {}  # ch_num -> (bin_start, bin_stop)

        for ch, ch_freq in WIFI_CHANNELS.items():
            offset = ch_freq - self.center_freq
            # Skip channels outside our capture bandwidth
            if abs(offset) > half_sr:
                continue

            # Convert offset to FFT bin (FFT is zero-centered, then fftshifted)
            bin_center = int(round(offset / hz_per_bin)) + self.fft_size // 2
            bin_half   = int(round(half_ch_bw / hz_per_bin))

            bin_start = max(0, bin_center - bin_half)
            bin_stop  = min(self.fft_size - 1, bin_center + bin_half)

            if bin_stop > bin_start:
                self._channel_bins[ch] = (bin_start, bin_stop)
                print(f"  Ch{ch:2d} @ {ch_freq/1e6:.1f} MHz  bins [{bin_start}:{bin_stop}]")

    def work(self, input_items, output_items):
        inp = input_items[0]
        output_items[0][:] = inp  # passthrough

        self.block_count += 1
        if self.block_count % self.report_every != 0:
            return len(inp)

        # Only process if we have enough samples for one FFT
        if len(inp) < self.fft_size:
            return len(inp)

        # Take the most recent fft_size samples
        chunk = inp[-self.fft_size:]

        # Apply Blackman-Harris window to reduce spectral leakage
        window  = np.blackman(self.fft_size)
        windowed = chunk * window

        # FFT + shift so DC is in the center
        spectrum = np.fft.fftshift(np.fft.fft(windowed, self.fft_size))
        power    = np.abs(spectrum) ** 2

        # Avoid log(0)
        power = np.maximum(power, 1e-20)
        power_db = 10.0 * np.log10(power / self.fft_size)

        # Measure mean energy in each channel's bins
        active_channels = []
        report = {}

        for ch, (b0, b1) in self._channel_bins.items():
            mean_db = float(np.mean(power_db[b0:b1]))
            # Exponential moving average (alpha=0.3)
            self.channel_energy_db[ch] = (
                0.3 * mean_db + 0.7 * self.channel_energy_db[ch]
            )
            report[ch] = round(self.channel_energy_db[ch], 1)
            if self.channel_energy_db[ch] > self.threshold_db:
                active_channels.append(ch)

        # Sort by energy descending
        ranked = sorted(report.items(), key=lambda x: x[1], reverse=True)

        # Print to console
        print("\n=== Channel Activity Report ===")
        for ch, db in ranked:
            if ch in self._channel_bins:
                flag = " <<< ACTIVE" if ch in active_channels else ""
                print(f"  Ch{ch:2d} ({WIFI_CHANNELS[ch]/1e6:.0f} MHz): {db:6.1f} dBFS{flag}")
        if active_channels:
            print(f"  Most active: Ch{ranked[0][0]}")
        print("================================")

        # Publish as PMT message
        msg_dict = pmt.make_dict()
        for ch, db in report.items():
            msg_dict = pmt.dict_add(
                msg_dict,
                pmt.intern(f"ch{ch}"),
                pmt.from_float(db)
            )
        # Add ranked most-active channel
        if ranked:
            msg_dict = pmt.dict_add(
                msg_dict,
                pmt.intern("most_active_ch"),
                pmt.from_long(ranked[0][0])
            )
            msg_dict = pmt.dict_add(
                msg_dict,
                pmt.intern("most_active_db"),
                pmt.from_float(ranked[0][1])
            )

        self.message_port_pub(pmt.intern('activity'), msg_dict)

        return len(inp)
