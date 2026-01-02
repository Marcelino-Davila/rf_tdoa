from DATA_COLLECTION import rx_cfile_capture_multi as capture
import os

if __name__ == "__main__":
    print("[i] starting infinite capture (Ctrl+C to stop)")
    print("CWD =", os.getcwd())
    print("OUTDIR(abs) =", os.path.abspath(r"DATA\RAW_BIN"))
    data = capture.receive_radio_chunks(
        chunk_samps=4096 * 8,   # must match transmitter
        save_raw=True,          # write .cfile to disk
        outdir="DATA\RAW_BIN",       # where files go
        verbose=True,
    )

    # This only runs AFTER Ctrl+C
    print("Stopped.")
    print("Final chunk counts:", data.total_chunks())
