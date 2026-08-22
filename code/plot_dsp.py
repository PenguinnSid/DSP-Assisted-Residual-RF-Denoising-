import numpy as np
import matplotlib.pyplot as plt


# ==========================================
# CHANGE THESE IF YOU WANT QPSK OR TEST DATA
# ==========================================

MODULATION = "bpsk"
SPLIT = "train"

SAMPLE_INDEX = 0
NUM_SAMPLES_TO_SHOW = 1000


# ==========================================
# LOAD DATA
# ==========================================

clean = np.load(
    f"data/{MODULATION}/{SPLIT}/{SPLIT}_clean.npy"
)

noisy = np.load(
    f"data/{MODULATION}/{SPLIT}/{SPLIT}_noisy.npy"
)

dsp = np.load(
    f"data/dsp/{MODULATION}_dsp/{SPLIT}_dsp.npy"
)

clean_signal = clean[SAMPLE_INDEX]
noisy_signal = noisy[SAMPLE_INDEX]
dsp_signal = dsp[SAMPLE_INDEX]


# ==========================================
# LIMIT DISPLAY SIZE
# ==========================================

clean_plot = np.real(
    clean_signal[:NUM_SAMPLES_TO_SHOW]
)

noisy_plot = np.real(
    noisy_signal[:NUM_SAMPLES_TO_SHOW]
)

dsp_plot = np.real(
    dsp_signal[:NUM_SAMPLES_TO_SHOW]
)


# ==========================================
# METRICS
# ==========================================

mse_noisy = np.mean(
    np.abs(clean_signal - noisy_signal) ** 2
)

mse_dsp = np.mean(
    np.abs(clean_signal - dsp_signal) ** 2
)

improvement = (
    (mse_noisy - mse_dsp)
    / mse_noisy
) * 100


error_noisy = np.abs(
    clean_plot - noisy_plot
)

error_dsp = np.abs(
    clean_plot - dsp_plot
)

removed_noise = (
    noisy_plot - dsp_plot
)


# ==========================================
# PLOTS
# ==========================================

fig, axs = plt.subplots(
    5,
    1,
    figsize=(16, 14)
)

fig.suptitle(
    f"{MODULATION.upper()} DSP Analysis",
    fontsize=16
)


# Clean
axs[0].plot(clean_plot)
axs[0].set_title("Clean Signal")
axs[0].grid(True)


# Noisy
axs[1].plot(noisy_plot)
axs[1].set_title(
    f"Noisy Signal | MSE = {mse_noisy:.6f}"
)
axs[1].grid(True)


# DSP
axs[2].plot(dsp_plot)
axs[2].set_title(
    f"DSP Output | MSE = {mse_dsp:.6f}"
)
axs[2].grid(True)


# Error comparison
axs[3].plot(
    error_noisy,
    label="Noisy Error"
)

axs[3].plot(
    error_dsp,
    label="DSP Error"
)

axs[3].legend()

axs[3].set_title(
    f"Error Comparison | Improvement = {improvement:.2f}%"
)

axs[3].grid(True)


# Removed noise
axs[4].plot(
    removed_noise
)

axs[4].set_title(
    "Noise Removed By DSP"
)

axs[4].grid(True)


plt.tight_layout()
plt.show()


# ==========================================
# TERMINAL OUTPUT
# ==========================================

print("\n==========================")
print("DSP PERFORMANCE")
print("==========================")
print(f"Noisy MSE : {mse_noisy:.6f}")
print(f"DSP MSE   : {mse_dsp:.6f}")
print(f"Improvement : {improvement:.2f}%")
print("==========================")