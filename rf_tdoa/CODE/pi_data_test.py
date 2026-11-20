import numpy as np
import matplotlib
matplotlib.use('Agg')  # prevents Qt/display issues
import matplotlib.pyplot as plt

# Fix: use raw strings for Windows paths
direct  = np.fromfile(r'G:\pi_test\pcdata.iq',  dtype=np.complex64)
pi_path = np.fromfile(r'G:\pi_test\pidata.iq', dtype=np.complex64)

# Trim to same length
n = min(len(direct), len(pi_path))
direct, pi_path = direct[:n], pi_path[:n]

# Fix: downsample for plotting only — keeps full data for SNR math
PLOT_MAX = 500_000
step = max(1, n // PLOT_MAX)
d_plot = direct[::step]
p_plot = pi_path[::step]

sample_rate = 2e6

# --- SNR ---
def snr(signal):
    power = np.mean(np.abs(signal)**2)
    noise = np.var(signal - np.mean(signal))
    return 10 * np.log10(power / (noise + 1e-12))

print(f"SNR direct:  {snr(direct):.2f} dB")
print(f"SNR pi path: {snr(pi_path):.2f} dB")
print(f"SNR delta:   {snr(direct) - snr(pi_path):.2f} dB")

diff = direct - pi_path
print(f"Mean diff magnitude: {np.mean(np.abs(diff)):.6f}")
print(f"Max  diff magnitude: {np.max(np.abs(diff)):.6f}")

# --- PSD on downsampled data ---
def psd(signal, fs):
    f = np.fft.fftfreq(len(signal), 1/fs)
    p = 20 * np.log10(np.abs(np.fft.fft(signal)) + 1e-12)
    return np.fft.fftshift(f), np.fft.fftshift(p)

f, p_direct = psd(d_plot, sample_rate)
_, p_pi     = psd(p_plot, sample_rate)

# --- Plot ---
fig, axes = plt.subplots(3, 1, figsize=(12, 10))

axes[0].plot(f/1e6, p_direct, label='Direct', alpha=0.7)
axes[0].plot(f/1e6, p_pi,     label='Pi path', alpha=0.7)
axes[0].set(xlabel='MHz', ylabel='dB', title='PSD Comparison')
axes[0].legend(loc='upper right')  # explicit loc avoids slow "best" search

axes[1].plot(f/1e6, p_direct - p_pi)
axes[1].axhline(0, color='red', linestyle='--')
axes[1].set(xlabel='MHz', ylabel='dB', title='PSD Difference (Direct - Pi)')

axes[2].plot(np.abs(diff[:5000]))
axes[2].set(xlabel='Sample', ylabel='Magnitude', title='Sample-by-Sample Difference')

plt.tight_layout()
plt.savefig(r'G:\pi_test\degradation_report.png', dpi=150)
print("Saved to G:\\pi_test\\degradation_report.png")