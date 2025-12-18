import numpy as np
import scipy.linalg
import glob
import os

# ==========================================
# CONFIGURATION - *** YOU MUST SET THESE ***
# ==========================================

# 1. SIGNAL PULSE (s) - MUST MATCH THE LENGTH N 
N = 2000 # Filter/Chunk size increased for benchmark
t = np.linspace(0, 1, N)

# === REPLACE THIS WITH YOUR ACTUAL PULSE SHAPE ===
# Example: A Simple Tone 
s = np.exp(1j * 2 * np.pi * 10 * t) 
# =================================================

# 2. FILE PATHS (Based on your file structure image)
BASE_DIR = "G:/rf_tests/NOISETEST/1" 
# Target only the Rw0 matrix for comparison
MATRIX_PATTERN = os.path.join(BASE_DIR, "noise_covariance_Rw0.npy")
OUTPUT_DIR = "G:/rf_tests/NOISETEST/OPTIMAL_FILTERS" 
# ==========================================

# ==========================================
# CALCULATION LOOP (Will only run once for Rw0)
# ==========================================

file_list = glob.glob(MATRIX_PATTERN)
results = []
os.makedirs(OUTPUT_DIR, exist_ok=True)

if not file_list:
    print(f"ERROR: No covariance matrices found matching pattern: {MATRIX_PATTERN}")
    exit()

print(f"--- Found {len(file_list)} Noise Covariance Matrices. Designing filters... ---")

# The loop runs exactly once for Rw0
for fpath in file_list:
    filename = os.path.basename(fpath)
    receiver_name = filename.split('_')[-1].split('.')[0].replace('Rw', 'rx')
    
    print(f"\nProcessing {filename} (Receiver {receiver_name}, N={N})...")
    
    try:
        # NOTE: This file must be 2000x2000 for the script to proceed!
        Rw = np.load(fpath)
        if Rw.shape != (N, N):
             print(f"  [SKIPPING] Matrix shape {Rw.shape} does not match required N={N}. Please provide a 2000x2000 matrix.")
             continue
    except Exception as e:
        print(f"  [ERROR] Could not load {filename}: {e}")
        continue

    # --- 1. Calculate the Optimal Filter (g = Rw^-1 * s) ---
    try:
        # This is the expensive O(N^3) operation.
        g = scipy.linalg.solve(Rw, s, assume_a='her') 
        
        # --- 2. Calculate the Maximum SNR (SNR_max = s^H * g) ---
        SNR_max_linear = np.vdot(s, g) 
        SNR_max_dB = 10 * np.log10(np.real(SNR_max_linear))
        
        # --- Save Filter (g vector) ---
        output_g_path = os.path.join(OUTPUT_DIR, f"optimal_filter_g_{receiver_name}.npy")
        np.save(output_g_path, g) 

        results.append({
            'Receiver': receiver_name,
            'SNR_max_dB': SNR_max_dB,
            'SNR_max_linear': np.real(SNR_max_linear),
        })
        print(f"  Max Theoretical SNR: {SNR_max_dB:.2f} dB. Filter saved to:\n  {output_g_path}")
        
    except scipy.linalg.LinAlgError:
        print("\n[ERROR] Matrix inversion failed (Rw is singular or nearly singular).")
    except Exception as e:
        print(f"  [ERROR] Calculation error: {e}")

# ==========================================
# NEXT STEP INSTRUCTIONS
# ==========================================

print("\n" + "=" * 50)
print("      FILTER DESIGN COMPLETE")
print("=" * 50)
if results:
    print(f"Filter file for {results[0]['Receiver']} saved.")
else:
    print("Could not generate filter file.")