# send_band_push.py
import sys
import zmq

def main():
    if len(sys.argv) != 3:
        print("Usage: python send_band_push.py <addr> <0|1|2>")
        print("Example: python send_band_push.py tcp://127.0.0.1:7000 1")
        sys.exit(1)

    addr = sys.argv[1]
    sel = sys.argv[2].strip()
    if sel not in ("0", "1", "2"):
        raise ValueError("selector must be 0, 1, or 2")

    ctx = zmq.Context.instance()
    s = ctx.socket(zmq.PUSH)
    s.connect(addr)          # GRC binds, Python connects
    s.send_string(sel)
    print(f"sent {sel} -> {addr}")

if __name__ == "__main__":
    main()
