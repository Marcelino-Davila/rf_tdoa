import os, csv, time, zmq, numpy as np
import scipy.signal # Used for fast convolution (applying the filter)

# You only need the scipy library installed (which you already do).
# The filter design function is NOT used here, only the pre-calculated coefficients.

# ==========================================
# CONFIGURATION
# ==========================================
RADIOS = [
    {"name": "rx0", "endpoint": "tcp://127.0.0.1:5555"},
    {"name": "rx1", "endpoint": "tcp://127.0.0.1:5556"},
    {"name": "rx2", "endpoint": "tcp://127.0.0.1:5557"},
]

# --- Filter-Specific Configuration ---
# 1. Base directory where your optimal filter 'h' files are saved
FILTER_DIR        = "RF_TESTS/OPTIMAL_FILTERS" 
# 2. Threshold for printing a detection event in the console (magnitude value)
DETECTION_THRESHOLD = 0.5 

# --- Streaming/File Configuration ---
CHUNK_SAMPS       = 4096 * 8      # How many complex samples per ZMQ chunk
OUTDIR            = "raw_bin"     # Output directory for raw data (preserved)
INDEX_CSV         = os.path.join(OUTDIR, "chunks_index.csv")
CSV_FLUSH_EVERY   = 10            # Flush every N rows

# ==========================================
# FILTER SETUP: LOAD PRE-CALCULATED COEFFICIENTS
# ==========================================
print("--- Loading Optimal Filter Coefficients ---")
for r in RADIOS:
    filter_filename = f"optimal_filter_g_{r['name']}.npy"
    filter_path = os.path.join(FILTER_DIR, filter_filename)
    
    try:
        # Load the saved impulse response (h) for convolution
        # NOTE: The optimal filter g is used in inner product. 
        # For convolution, you need the complex conjugate time-reversal (h).
        # Assuming the filter design script saved the impulse response h:
        r["h_filter"] = np.load(filter_path)
        print(f"[i] {r['name']} filter loaded (Length: {len(r['h_filter'])} taps)")
        
        # Determine the filter delay length (N) for convolution mode
        r["N_filter"] = len(r["h_filter"])
        
    except FileNotFoundError:
        print(f"[!] WARNING: Filter file not found for {r['name']} at {filter_path}")
        print("    This receiver will only save RAW data, no matched filtering will be applied.")
        r["h_filter"] = None

# ==========================================
# OUTPUT DIR AND ZMQ SETUP (Unchanged)
# ==========================================

os.makedirs(OUTDIR, exist_ok=True)
ctx = zmq.Context.instance()
poller = zmq.Poller()

for r in RADIOS:
    sock = ctx.socket(zmq.PULL)
    sock.setsockopt(zmq.RCVHWM, 200)
    sock.connect(r["endpoint"])
    r["sock"] = sock
    r["buf"] = np.zeros(0, dtype=np.complex64)

    bin_path = os.path.join(OUTDIR, f"{r['name']}.cfile")
    r["bin_path"] = bin_path
    r["fh"] = open(bin_path, "ab")

    poller.register(sock, zmq.POLLIN)
    print(f"[i] {r['name']} connected to {r['endpoint']} -> {bin_path}")

sock_to_radio = {r["sock"]: r for r in RADIOS}

# --------- INDEX CSV ----------
index_exists = os.path.exists(INDEX_CSV)
index_fh = open(INDEX_CSV, "a", newline="")
index_writer = csv.writer(index_fh)
if not index_exists:
    index_writer.writerow([
        "radio",
        "ts_unix",
        "ts_iso",
        "chunk_idx",
        "samples_per_chunk"
    ])

rows_since_flush = 0
chunk_counter = {r["name"]: 0 for r in RADIOS}

# ==========================================
# MAIN LOOP: STREAMING & FILTERING
# ==========================================
try:
    while True:
        events = dict(poller.poll(timeout=1000))
        for sock, ev in events.items():
            if not (ev & zmq.POLLIN):
                continue

            r = sock_to_radio[sock]
            raw = sock.recv()
            x = np.frombuffer(raw, dtype=np.complex64)
            if x.size == 0:
                continue

            r["buf"] = np.concatenate((r["buf"], x))

            while r["buf"].size >= CHUNK_SAMPS:
                seg = r["buf"][:CHUNK_SAMPS]
                r["buf"] = r["buf"][CHUNK_SAMPS:]

                ts = time.time()
                ts_iso = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(ts))
                idx = chunk_counter[r["name"]]
                chunk_counter[r["name"]] += 1

                # ---------------------------------------------
                # STEP 1: APPLY MATCHED FILTER (REAL-TIME)
                # ---------------------------------------------
                if r["h_filter"] is not None:
                    # Apply the optimal matched filter using convolution
                    # 'same' mode centers the result and keeps the length the same as input seg
                    y_filtered = scipy.signal.convolve(seg, r["h_filter"], mode='same')
                    
                    # Detection Statistic is the magnitude of the filtered output
                    peak_magnitude = np.max(np.abs(y_filtered))
                    
                    if peak_magnitude > DETECTION_THRESHOLD:
                         print(f"[{r['name']}] !!! DETECTED SIGNAL !!! Chunk {idx}, Peak Mag: {peak_magnitude:.4f}")
                    # else:
                        # print(f"[{r['name']}] chunk {idx} written. Max filtered output: {peak_magnitude:.4f}")

                # ---------------------------------------------
                # STEP 2: SAVE RAW IQ DATA (Original Logic)
                # ---------------------------------------------
                seg.astype(np.complex64).tofile(r["fh"])

                # log index entry
                index_writer.writerow([
                    r["name"],
                    f"{ts:.6f}",
                    ts_iso,
                    idx,
                    CHUNK_SAMPS
                ])

                rows_since_flush += 1
                if rows_since_flush >= CSV_FLUSH_EVERY:
                    index_fh.flush()
                    r["fh"].flush()
                    rows_since_flush = 0

                if r["h_filter"] is None or peak_magnitude <= DETECTION_THRESHOLD:
                    # Print standard log only if no detection or filter is missing
                    print(f"[{r['name']}] wrote chunk {idx}")


except KeyboardInterrupt:
    print("\n[!] Stopped by user.")
finally:
    try:
        index_fh.flush()
        index_fh.close()
    except Exception:
        pass
    for r in RADIOS:
        try:
            r["fh"].flush()
            r["fh"].close()
        except Exception:
            pass