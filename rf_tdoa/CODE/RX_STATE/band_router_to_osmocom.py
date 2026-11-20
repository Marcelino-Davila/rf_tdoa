# band_router_to_osmocom.py
import zmq
from gnuradio import pmt

# Map selector -> center frequency (Hz). Change these to your 3 bands.
BANDS_HZ = {
    0: 2.402e9,   # example: BT low
    1: 2.426e9,   # example: mid
    2: 2.480e9,   # example: high
}

PULL_ADDR = "tcp://127.0.0.1:6000"  # script A sends to this
PUB_ADDR  = "tcp://127.0.0.1:7000"  # GRC ZMQ SUB Message Source listens to this

def make_freq_cmd(freq_hz: float):
    """
    osmocom Source 'command' port expects a PMT dict.
    Key is typically "freq" for center frequency.
    """
    d = pmt.make_dict()
    d = pmt.dict_add(d, pmt.intern("freq"), pmt.from_double(freq_hz))
    return d

def main():
    ctx = zmq.Context.instance()

    pull = ctx.socket(zmq.PULL)
    pull.bind(PULL_ADDR)

    pub = ctx.socket(zmq.PUB)
    pub.bind(PUB_ADDR)

    print(f"[router] listening selector on {PULL_ADDR}")
    print(f"[router] publishing PMT commands on {PUB_ADDR}")
    print(f"[router] bands: {BANDS_HZ}")

    while True:
        msg = pull.recv()  # blocks
        if not msg:
            continue

        sel = msg[0]
        if sel not in BANDS_HZ:
            print(f"[router] unknown selector={sel}, ignoring")
            continue

        freq = float(BANDS_HZ[sel])
        cmd = make_freq_cmd(freq)
        payload = pmt.serialize_str(cmd)  # bytes

        pub.send(payload)
        print(f"[router] selector={sel} -> freq={freq/1e9:.6f} GHz (published)")

if __name__ == "__main__":
    main()
