# send_band_select.py
import sys
import zmq

def main():
    if len(sys.argv) != 3:
        print("Usage: python send_band_select.py <pull_addr> <0|1|2>")
        print("Example: python send_band_select.py tcp://127.0.0.1:6000 1")
        sys.exit(1)

    pull_addr = sys.argv[1]
    sel = int(sys.argv[2])
    if sel not in (0, 1, 2):
        raise ValueError("selector must be 0, 1, or 2")

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.PUSH)
    sock.connect(pull_addr)

    # send a single byte is fine
    sock.send(bytes([sel]))
    print(f"sent selector={sel} -> {pull_addr}")

if __name__ == "__main__":
    main()
