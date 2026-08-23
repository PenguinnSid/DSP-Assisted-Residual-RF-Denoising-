import numpy as np
from pathlib import Path


# ============================================================
# PATHS AND CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"

MODULATIONS = ["bpsk", "qpsk"]
SPLITS = ["train", "test", "validation"]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def rms(x):
    """
    Calculates the RMS amplitude of a signal.
    """

    return np.sqrt(np.mean(np.abs(x) ** 2))


def calculate_snr(clean, received):
    """
    Calculates the SNR of a received signal relative
    to a clean reference signal.

    SNR = 10 * log10(signal_power / noise_power)

    The noise is defined as:

        noise = received - clean
    """

    signal_power = np.mean(np.abs(clean) ** 2)

    noise = received - clean
    noise_power = np.mean(np.abs(noise) ** 2)

    if noise_power < 1e-12:
        return np.inf

    return 10 * np.log10(
        signal_power / noise_power
    )


# ============================================================
# DATASET VERIFICATION
# ============================================================

def verify_dataset(modulation, split):

    noisy_path = (
        DATA_DIR
        / modulation
        / split
        / f"{split}_noisy.npy"
    )

    clean_path = (
        DATA_DIR
        / modulation
        / split
        / f"{split}_clean.npy"
    )

    dsp_path = (
        DATA_DIR
        / modulation
        / split
        / f"{split}_dsp.npy"
    )

    print("\n")
    print("=" * 70)
    print(f"{modulation.upper()} - {split.upper()}")
    print("=" * 70)

    # ========================================================
    # CHECK FILES
    # ========================================================

    print("\nFILES")

    for path in [clean_path, noisy_path, dsp_path]:

        if path.exists():
            print(f"✓ Found: {path}")
        else:
            print(f"✗ Missing: {path}")

    if (
        not clean_path.exists()
        or not noisy_path.exists()
        or not dsp_path.exists()
    ):
        print("Skipping dataset verification.")
        return

    # ========================================================
    # LOAD DATA
    # ========================================================

    clean = np.load(clean_path)
    noisy = np.load(noisy_path)
    dsp = np.load(dsp_path)

    # ========================================================
    # SHAPE
    # ========================================================

    print("\nSHAPE")

    print("Clean :", clean.shape)
    print("Noisy :", noisy.shape)
    print("DSP   :", dsp.shape)

    if (
        clean.shape == noisy.shape
        and noisy.shape == dsp.shape
    ):
        print("✓ All signal shapes match")
    else:
        print("✗ Signal shapes do not match")

    # ========================================================
    # DTYPE
    # ========================================================

    print("\nDTYPE")

    print("Clean :", clean.dtype)
    print("Noisy :", noisy.dtype)
    print("DSP   :", dsp.dtype)

    # ========================================================
    # NAN / INF CHECK
    # ========================================================

    print("\nVALIDITY")

    noisy_nan = np.isnan(noisy).any()
    noisy_inf = np.isinf(noisy).any()

    dsp_nan = np.isnan(dsp).any()
    dsp_inf = np.isinf(dsp).any()

    print(
        "Noisy NaN:",
        noisy_nan,
        "| Inf:",
        noisy_inf
    )

    print(
        "DSP NaN:",
        dsp_nan,
        "| Inf:",
        dsp_inf
    )

    if not noisy_nan and not noisy_inf:
        print("✓ Noisy signals contain no NaN/Inf")

    if not dsp_nan and not dsp_inf:
        print("✓ DSP signals contain no NaN/Inf")
    else:
        print("✗ DSP contains NaN/Inf")

    # ========================================================
    # DC OFFSET
    # ========================================================

    print("\nDC OFFSET")

    noisy_mean = np.mean(
        noisy,
        axis=1
    )

    dsp_mean = np.mean(
        dsp,
        axis=1
    )

    noisy_dc = np.abs(noisy_mean)
    dsp_dc = np.abs(dsp_mean)

    average_noisy_dc = np.mean(noisy_dc)
    average_dsp_dc = np.mean(dsp_dc)

    print(
        "Average |mean| before:",
        average_noisy_dc
    )

    print(
        "Average |mean| after :",
        average_dsp_dc
    )

    if average_dsp_dc < average_noisy_dc:
        print("✓ DC component was reduced")
    else:
        print("⚠ DC component was not reduced")

    # ========================================================
    # RMS / AMPLITUDE NORMALIZATION
    # ========================================================

    print("\nAMPLITUDE NORMALIZATION")

    noisy_rms = np.sqrt(
        np.mean(
            np.abs(noisy) ** 2,
            axis=1
        )
    )

    dsp_rms = np.sqrt(
        np.mean(
            np.abs(dsp) ** 2,
            axis=1
        )
    )

    print(
        "Average RMS before:",
        np.mean(noisy_rms)
    )

    print(
        "Average RMS after :",
        np.mean(dsp_rms)
    )

    print(
        "Minimum DSP RMS:",
        np.min(dsp_rms)
    )

    print(
        "Maximum DSP RMS:",
        np.max(dsp_rms)
    )

    if np.allclose(
        dsp_rms,
        1.0,
        atol=0.05
    ):
        print(
            "✓ All DSP signals are approximately unit RMS"
        )
    else:
        print(
            "⚠ Some DSP signals are not approximately unit RMS"
        )

    # ========================================================
    # DATA CHANGE
    # ========================================================

    print("\nDATA CHANGE")

    difference = dsp - noisy

    mean_difference = np.mean(
        np.abs(difference)
    )

    print(
        "Mean |DSP - noisy|:",
        mean_difference
    )

    if mean_difference > 1e-6:
        print(
            "✓ DSP significantly changed the input data"
        )
    else:
        print(
            "✗ DSP output is almost identical to input"
        )

    # ========================================================
    # MSE AGAINST CLEAN SIGNAL
    # ========================================================

    print("\nCOMPARISON WITH CLEAN TARGET")

    noisy_mse = np.mean(
        np.abs(clean - noisy) ** 2
    )

    dsp_mse = np.mean(
        np.abs(clean - dsp) ** 2
    )

    print(
        "Noisy → Clean MSE:",
        noisy_mse
    )

    print(
        "DSP   → Clean MSE:",
        dsp_mse
    )

    if dsp_mse < noisy_mse:
        print(
            "✓ DSP reduced MSE relative to noisy signal"
        )
    else:
        print(
            "⚠ DSP increased MSE relative to noisy signal"
        )

    # ========================================================
    # SNR COMPARISON
    # ========================================================

    print("\nSNR COMPARISON")

    noisy_snr_values = []
    dsp_snr_values = []

    for i in range(len(clean)):

        noisy_snr = calculate_snr(
            clean[i],
            noisy[i]
        )

        dsp_snr = calculate_snr(
            clean[i],
            dsp[i]
        )

        noisy_snr_values.append(
            noisy_snr
        )

        dsp_snr_values.append(
            dsp_snr
        )

    noisy_snr_values = np.asarray(
        noisy_snr_values
    )

    dsp_snr_values = np.asarray(
        dsp_snr_values
    )

    snr_improvement = (
        dsp_snr_values
        - noisy_snr_values
    )

    print(
        "Average noisy SNR:",
        np.mean(noisy_snr_values),
        "dB"
    )

    print(
        "Average DSP SNR:",
        np.mean(dsp_snr_values),
        "dB"
    )

    print(
        "Average SNR improvement:",
        np.mean(snr_improvement),
        "dB"
    )

    if np.mean(snr_improvement) > 0:
        print(
            "✓ DSP improved average SNR"
        )
    else:
        print(
            "⚠ DSP did not improve average SNR"
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\nSUMMARY")

    print(
        f"Average SNR improvement: "
        f"{np.mean(snr_improvement):.4f} dB"
    )

    print(
        f"MSE before DSP: "
        f"{noisy_mse:.6f}"
    )

    print(
        f"MSE after DSP:  "
        f"{dsp_mse:.6f}"
    )

    print(
        f"Average RMS after DSP: "
        f"{np.mean(dsp_rms):.6f}"
    )


# ============================================================
# RUN VERIFICATION
# ============================================================

if __name__ == "__main__":

    for modulation in MODULATIONS:

        for split in SPLITS:

            verify_dataset(
                modulation,
                split
            )

    print("\n")
    print("=" * 70)
    print("DSP DATASET VERIFICATION COMPLETE")
    print("=" * 70)