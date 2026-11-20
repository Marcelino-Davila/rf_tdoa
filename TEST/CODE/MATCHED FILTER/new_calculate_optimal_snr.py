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

# 2. NOISE PARAMETER (PSD)
NOISE_PSD = 4.443472e-04 

# 3. FILE PATHS 
OUTPUT_DIR = "G:/rf_tests/NOISETEST/OPTIMAL_FILTERS" 
# ==========================================

# ==========================================
# CALCULATION (Simplified for White Noise)
# ==========================================

# 1. Calculate the Optimal Filter (g = s / NOISE_PSD) - O(N) operation
g = s / NOISE_PSD 

# 2. Calculate the Maximum SNR (SNR_max = s^H * g = E_s / NOISE_PSD)
E_s = np.vdot(s, s) 
SNR_max_linear = np.real(E_s / NOISE_PSD)
SNR_max_dB = 10 * np.log10(SNR_max_linear)

# 3. Setup output saving (ONLY FOR RX0)
RECEIVER_NAME = "rx0"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("--- Calculating Optimal Filter for White Noise Model ---")
print(f"Noise Power Density (sigma^2): {NOISE_PSD:.6e}")
print(f"Signal Energy (E_s): {E_s:.2f}")

# --- Save Filter (g vector) ---
output_g_path = os.path.join(OUTPUT_DIR, f"optimal_filter_g_{RECEIVER_NAME}.npy")
np.save(output_g_path, g)
    
print(f"\nProcessing Receiver {RECEIVER_NAME} (N={N})...")
print(f"  Max Theoretical SNR: {SNR_max_dB:.2f} dB.")
print(f"  Filter saved to:\n  {output_g_path}")

# ==========================================
# NEXT STEP INSTRUCTIONS
# ==========================================

print("\n" + "=" * 50)
print("             FILTER DESIGN COMPLETE")
print("=" * 50)
print(f"Result for {RECEIVER_NAME} saved.")