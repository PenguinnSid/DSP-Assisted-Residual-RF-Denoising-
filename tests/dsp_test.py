import numpy as np
from pathlib import Path
from scipy.signal import firwin, filtfilt


# ============================================================
# PATHS AND CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"

MODULATIONS = ["bpsk", "qpsk"]
SPLITS = ["train", "test", "validation"]

SPS = 8  # must match generation config

# same DSP params as dsp.py
CUTOFF = 0.18
NUM_TAPS = 101


# ============================================================
# DSP CHAIN (reimplemented locally so this file has no
# dependency on dsp.py / clean_generator.py imports)
# ============================================================

def rrc_filter(beta=0.35, span=8, sps=8):
    """
    Generates a root raised cosine filter for pulse shaping.
    Mirrors clean_generator.rrc_filter exactly.
    """

    N = span * sps

    time = np.arange(-N / 2, N / 2 + 1) / sps

    h = np.zeros_like(time)

    h[time == 0] = 1 - beta + (4 * beta / np.pi)

    t_special = np.abs(time) == (1 / (4 * beta))
    h[t_special] = (beta / np.sqrt(2)) * (
        ((1 + 2 / np.pi) * (np.sin(np.pi / (4 * beta)))) +
        ((1 - 2 / np.pi) * (np.cos(np.pi / (4 * beta))))
    )

    general_case = ~t_special & (time != 0)
    h[general_case] = (
        (np.sin(np.pi * time[general_case] * (1 - beta)) +
         4 * beta * time[general_case] *
         np.cos(np.pi * time[general_case] * (1 + beta))) /
        (np.pi * time[general_case] *
         (1 - (4 * beta * time[general_case]) ** 2))
    )

    h /= np.sqrt(np.sum(h ** 2))

    return h


def remove_dc_offset(signal):
    signal = np.asarray(signal, dtype=np.complex64)

    dc_offset = np.mean(signal)

    return (signal - dc_offset).astype(np.complex64)


def matched_filter(signal, rrc):
    filtered_signal = np.convolve(signal, rrc, mode="same")

    return filtered_signal.astype(np.complex64)


def low_pass_filter(signal, cutoff=CUTOFF, num_taps=NUM_TAPS):
    signal = np.asarray(signal, dtype=np.complex64)

    coefficients = firwin(numtaps=num_taps, cutoff=cutoff, window=("kaiser", 8.0))

    real_filtered = filtfilt(coefficients, [1.0], signal.real)
    imag_filtered = filtfilt(coefficients, [1.0], signal.imag)

    filtered_signal = (real_filtered + 1j * imag_filtered)

    return filtered_signal.astype(np.complex64)


def normalize_amplitude(signal):
    signal = np.asarray(signal, dtype=np.complex64)

    rms = np.sqrt(np.mean(np.abs(signal) ** 2))

    if rms < 1e-12:
        return signal

    return (signal / rms).astype(np.complex64)


def preprocess_signal(signal, rrc):
    signal = remove_dc_offset(signal)
    signal = matched_filter(signal, rrc)
    signal = low_pass_filter(signal)
    signal = normalize_amplitude(signal)

    return signal.astype(np.complex64)


def preprocess_dataset(noisy_signals, sps=8):
    """
    Applies DSP preprocessing to an entire dataset.
    """

    noisy_signals = np.asarray(noisy_signals)

    if noisy_signals.ndim != 2:
        raise ValueError(f"Expected 2D input, got {noisy_signals.shape}")

    rrc = rrc_filter(beta=0.35, span=8, sps=sps)

    dsp_signals = np.empty(noisy_signals.shape, dtype=np.complex64)

    for i in range(len(noisy_signals)):
        dsp_signals[i] = preprocess_signal(noisy_signals[i], rrc)

    return dsp_signals


# ============================================================
# METRIC HELPERS
# ============================================================

def calculate_snr(reference, received):
    """
    Calculates the SNR of a received signal relative
    to a reference signal.

    SNR = 10 * log10(signal_power / noise_power)

    The noise is defined as:

        noise = received - reference
    """

    signal_power = np.mean(np.abs(reference) ** 2)

    noise = received - reference
    noise_power = np.mean(np.abs(noise) ** 2)

    if noise_power < 1e-12:
        return np.inf

    return 10 * np.log10(
        signal_power / noise_power
    )


def report_metrics(label, reference, candidates):
    """
    Prints MSE and per-sample-averaged SNR for each named candidate
    signal against a shared reference signal.

    candidates: dict of {name: array}, each same shape as reference.
    """

    print(f"\n{label}")

    n = len(reference)

    for name, signal in candidates.items():

        mse = np.mean(np.abs(reference - signal) ** 2)

        snr_values = np.empty(n, dtype=np.float64)

        for i in range(n):
            snr_values[i] = calculate_snr(reference[i], signal[i])

        print(
            f"{name:>12} MSE: {mse:.6f}   |   "
            f"Avg SNR: {np.mean(snr_values):.4f} dB"
        )


# ============================================================
# DATASET VERIFICATION
# ============================================================

def verify_dataset(modulation, split):

    paths = {
        "target": DATA_DIR / modulation / split / f"{split}_target.npy",
        "faded": DATA_DIR / modulation / split / f"{split}_faded.npy",
        "faded_target": DATA_DIR / modulation / split / f"{split}_faded_target.npy",
        "noisy": DATA_DIR / modulation / split / f"{split}_noisy.npy",
        "dsp": DATA_DIR / modulation / split / f"{split}_dsp.npy",
    }

    if any(not p.exists() for p in paths.values()):
        missing = [k for k, p in paths.items() if not p.exists()]
        print(f"{modulation.upper()} - {split.upper()}: missing {missing}, skipping.")
        return

    target = np.load(paths["target"])
    faded = np.load(paths["faded"])
    faded_target = np.load(paths["faded_target"])
    noisy = np.load(paths["noisy"])
    dsp = np.load(paths["dsp"])

    print("\n")
    print("=" * 70)
    print(f"{modulation.upper()} - {split.upper()}")
    print("=" * 70)

    # ========================================================
    # TEST 1: FADING ONLY (no AWGN)
    #
    # Uses the saved `faded` signal directly (no noise was ever
    # added to it). Reference is `target` (clean, no fading).
    # Since there's no AWGN here, any error is entirely attributable
    # to the unequalized fading distortion. Since the current DSP
    # chain has no equalizer, expect little to no improvement here.
    # ========================================================

    fading_only_dsp = preprocess_dataset(faded, sps=SPS)

    report_metrics(
        "TEST 1 - FADING ONLY (reference: target)",
        target,
        {
            "Faded": faded,
            "DSP": fading_only_dsp,
        }
    )

    faded_dsp_mse = np.mean(np.abs(target - fading_only_dsp) ** 2)
    faded_raw_mse = np.mean(np.abs(target - faded) ** 2)

    if faded_dsp_mse < faded_raw_mse:
        print("✓ DSP reduced fading-attributable error")
    else:
        print("⚠ DSP did not reduce fading-attributable error (expected — no equalizer yet)")

    # ========================================================
    # TEST 2: AWGN REDUCTION (fading held constant on both sides)
    #
    # faded_target = fading applied, no AWGN, matched-filtered/
    # LPF'd/normalized. Both `noisy` and `dsp` came from the same
    # faded signal, so comparing against faded_target holds the
    # fading distortion constant on both sides of the residual,
    # isolating whether AWGN specifically is being reduced.
    # ========================================================

    report_metrics(
        "TEST 2 - AWGN REDUCTION (reference: faded_target, fading held constant)",
        faded_target,
        {
            "Noisy": noisy,
            "DSP": dsp,
        }
    )

    dsp_awgn_mse = np.mean(np.abs(faded_target - dsp) ** 2)
    noisy_awgn_mse = np.mean(np.abs(faded_target - noisy) ** 2)

    if dsp_awgn_mse < noisy_awgn_mse:
        print("✓ DSP reduced AWGN-attributable error")
    else:
        print("⚠ DSP did not reduce AWGN-attributable error")

    # ========================================================
    # SANITY: AMPLITUDE NORMALIZATION
    # ========================================================

    dsp_rms = np.sqrt(np.mean(np.abs(dsp) ** 2, axis=1))

    print("\nAMPLITUDE NORMALIZATION")
    print(f"Average DSP RMS: {np.mean(dsp_rms):.6f}")

    if np.allclose(dsp_rms, 1.0, atol=0.05):
        print("✓ All DSP signals are approximately unit RMS")
    else:
        print("⚠ Some DSP signals are not approximately unit RMS")


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