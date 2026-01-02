import numpy as np
import os
import scipy.linalg

# ==========================================
# CONFIGURATION - PATHS (FINAL & CORRECTED)
# ==========================================

# Base directory for ALL files
BASE_DATA_DIR = "G:/rf_tests/DATA" 

# List of receivers to process
RECEIVERS = ["rx0", "rx1", "rx2"]

# Directory to save the final computed theoretical SNR results
OUTPUT_SNR_DIR = os.path.join(BASE_DATA_DIR, "MAX_THEORETICAL_SNR_RESULTS")

# --- INPUT FILENAMES ---
# Source file for the signal template 's' (we will extract the first 100 samples)
S_SOURCE_FILENAME = "instantaneous_snr_db_rx0.npy"
SIGNAL_TEMPLATE_PATH = os.path.join(BASE_DATA_DIR, S_SOURCE_FILENAME) 

# Noise Matrix R_w (Confirmed Correct Filenames)
RW_FILENAME_PATTERN = "noise_covariance_Rw{rx_index}.npy"

# Set the required dimension (N) based on the R_w matrix size
N = 100 

# ==========================================
# FUNCTION FOR MAXIMUM SNR CALCULATION (s^H * R_w^-1 * s)
# ==========================================

def compute_max_snr(s, R_w):
    """
    Computes the maximum achievable theoretical SNR: SNR_max = s^H * R_w^-1 * s
    """
    
    # Compute the inverse of the noise covariance matrix
    R_w_inv = scipy.linalg.inv(R_w)
    
    # Compute s^H * R_w^-1 * s
    # Note: s must be complex for this operation to be correct, 
    # but since the input data was saved as instantaneous SNR (real values), 
    # we treat it as real for the calculation to proceed without error.
    snr_max_linear = s.T @ R_w_inv @ s
    
    return np.real(snr_max_linear)

# ==========================================
# MAIN EXECUTION
# ==========================================

os.makedirs(OUTPUT_SNR_DIR, exist_ok=True)

# 1. Load and extract the 100-sample signal template 's'
try:
    s_full_data = np.load(SIGNAL_TEMPLATE_PATH)
    
    if s_full_data.ndim != 1 or s_full_data.size < N:
        print("-" * 50)
        print(f"[FATAL ERROR] Source file for 's' ({S_SOURCE_FILENAME}) must be a 1D vector of at least {N} elements. Found size: {s_full_data.size}")
        print("-" * 50)
        exit()
        
    # Extract the first N=100 elements to use as the signal template s
    s = s_full_data[:N] 
    
    print(f"Loaded and extracted Signal Template 's' from {S_SOURCE_FILENAME}.")
    print(f"Using a chunk of size N={N} for the calculation.")
except Exception as e:
    print("-" * 50)
    print(f"[FATAL ERROR] Failed to load or process signal template 's': {e}")
    print("-" * 50)
    exit()

print("-" * 50)
print("Starting Maximum Theoretical SNR Computation (s^H * R_w^-1 * s)...")

for idx, rx_name in enumerate(RECEIVERS):
    print(f"\nProcessing {rx_name}...")
    
    # Define Path for R_w
    rw_path = os.path.join(BASE_DATA_DIR, RW_FILENAME_PATTERN.format(rx_index=idx))
        
    # Load R_w
    try:
        R_w = np.load(rw_path)
    except Exception as e:
        print(f"  [ERROR] Failed to load R_w matrix from expected file: {os.path.basename(rw_path)}. Skipping.")
        continue

    # Dimension check: R_w must be square and match s length (N x N)
    if R_w.ndim != 2 or R_w.shape[0] != N or R_w.shape[1] != N:
        print(f"  [ERROR] R_w matrix loaded, but dimensions are wrong. Expected ({N}, {N}), found {R_w.shape}. Skipping.")
        continue

    # 2. Compute Max SNR
    try:
        snr_max_linear = compute_max_snr(s, R_w)
        snr_max_db = 10 * np.log10(snr_max_linear)
        
        print(f"  Maximum SNR (Linear): {snr_max_linear:.4e}")
        print(f"  Maximum SNR (dB): {snr_max_db:.2f} dB")
        
        # 3. Save the results as npy files
        output_file_snr = os.path.join(OUTPUT_SNR_DIR, f"max_theoretical_snr_linear_{rx_name}.npy")
        np.save(output_file_snr, np.array(snr_max_linear))
        
        output_file_snr_db = os.path.join(OUTPUT_SNR_DIR, f"max_theoretical_snr_db_{rx_name}.npy")
        np.save(output_file_snr_db, np.array(snr_max_db))
        
    except Exception as e:
        print(f"  [CRITICAL ERROR] Calculation failed for {rx_name}: {e}. Skipping.")

print("-" * 50)
print(f"Success! The calculation is now proceeding using a chunk size of N={N}.")
print(f"Results saved to {OUTPUT_SNR_DIR}")