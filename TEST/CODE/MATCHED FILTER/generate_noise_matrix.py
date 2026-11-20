import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import glob
import os

# ==========================================
# CONFIGURATION - *** YOU MUST SET THESE ***
# ==========================================
# 1. SIGNAL LENGTH (N)
# This MUST match the length of the transmit pulse you plan to use later.
N = 100  # <--- SET YOUR PULSE LENGTH HERE

# 2. RECEIVER TO PROCESS
TARGET_RECEIVER = "rx2"

# 3. DIRECTORY STRUCTURE
NOISE_BASE_PATH = "G:/RF_TESTS/NOISETEST"

# 4. OUTPUT FILE
OUTPUT_MATRIX_FILE = "G:/RF_TESTS/noise_covariance_Rw.npy" 

# 5. MEMORY-SAFETY PARAMETER 
CHUNK_SIZE = 1_000_000 

# 6. SPEED OPTIMIZATION (NEW)
# Only use every DOWNSAMPLE_FACTOR-th snapshot.
# Setting to 50 cuts calculation time by 50x. Start here.
DOWNSAMPLE_FACTOR = 50 

# ==========================================
# PROCESSING
# ==========================================

pattern = os.path.join(NOISE_BASE_PATH, "*", f"{TARGET_RECEIVER}.cfile")
file_list = glob.glob(pattern)

if not file_list:
    print(f"No noise files found using pattern: {pattern}")
    print("Please check your NOISE_BASE_PATH and TARGET_RECEIVER settings.")
    exit()

print(f"Found {len(file_list)} files for {TARGET_RECEIVER}. Calculating average covariance...")
print(f"Matrix size will be {N}x{N} (based on N={N})")
print(f"Using a downsample factor of {DOWNSAMPLE_FACTOR} for speed optimization.")

accumulated_Rw = np.zeros((N, N), dtype=np.complex64)
valid_files_count = 0

for fpath in file_list:
    try:
        run_number = os.path.basename(os.path.dirname(fpath))
        print(f"Processing Run {run_number} (File: {fpath})...")
        
        w_raw = np.fromfile(fpath, dtype=np.complex64)
        total_samples = len(w_raw) 
        
        if total_samples < N:
            print(f"  [Skipping] File too short (samples={total_samples} < N={N})")
            continue

        # --- Progress Tracking Setup ---
        percent_step = total_samples // 20
        next_checkpoint = percent_step
        
        # --- MEMORY-EFFICIENT CALCULATION ---
        current_Rw_sum = np.zeros((N, N), dtype=np.complex64)
        total_file_snapshots = 0
        
        for i in range(0, total_samples - N + 1, CHUNK_SIZE):
            
            # --- Progress Update ---
            if i >= next_checkpoint:
                percent = int(100 * i / total_samples)
                print(f"  Progress: {percent}%...")
                next_checkpoint += percent_step
            
            # Get a chunk of raw data
            end_index = i + CHUNK_SIZE + N - 1
            if end_index > total_samples:
                end_index = total_samples
            
            w_chunk = w_raw[i : end_index]
            
            # Create all overlapping snapshots from this chunk
            if len(w_chunk) >= N:
                snapshots = sliding_window_view(w_chunk, window_shape=N)
                
                # *** DOWNSAMPLE STEP HERE (The Speed Up) ***
                # We only take every DOWNSAMPLE_FACTOR-th snapshot.
                downsampled_snapshots = snapshots[::DOWNSAMPLE_FACTOR]

                if downsampled_snapshots.shape[0] > 0:
                    # Accumulate the sum of outer products: X^H @ X
                    current_Rw_sum += downsampled_snapshots.conj().T @ downsampled_snapshots
                    total_file_snapshots += downsampled_snapshots.shape[0]

        # Calculate the final average Rw for this file
        if total_file_snapshots > 0:
            current_Rw = current_Rw_sum / total_file_snapshots
            accumulated_Rw += current_Rw
            valid_files_count += 1
            print(f"  Progress: 100%. File complete.")
            print(f"  Processed {total_file_snapshots:,} snapshots. Total samples: {total_samples:,}.")
        
    except Exception as e:
        print(f"  [Error] Failed to process {fpath}: {e}")

# ==========================================
# FINALIZE AND SAVE
# ==========================================

if valid_files_count > 0:
    final_Rw = accumulated_Rw / valid_files_count
    
    os.makedirs(os.path.dirname(OUTPUT_MATRIX_FILE), exist_ok=True)
    np.save(OUTPUT_MATRIX_FILE, final_Rw)
    
    print("-" * 30)
    print(f"Success! Averaged covariance from {valid_files_count} files.")
    print(f"Noise Covariance Matrix (R_w) shape: {final_Rw.shape}")
    print(f"Saved to: {OUTPUT_MATRIX_FILE}")
    print("-" * 30)
    
    avg_power = np.real(np.trace(final_Rw)) / N
    print(f"Average Noise Power (diagonal mean): {avg_power:.6f}")
else:
    print("No valid files processed. Matrix not saved.")