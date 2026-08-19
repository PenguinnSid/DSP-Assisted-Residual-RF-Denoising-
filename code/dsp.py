import numpy as np
from pathlib import Path
from scipy.signal import firwin, filtfilt


# DSP PARAMETERS

CUTOFF = 0.45
NUM_TAPS = 31

# DC OFFSET REMOVAL
def remove_dc_offset(signal):
    signal = np.asarray(signal, dtype=np.complex64)

    dc_offset = np.mean(signal)

    return (signal - dc_offset).astype(np.complex64)


# LOW-PASS FIR FILTER

def low_pass_filter(signal, cutoff=CUTOFF, num_taps=NUM_TAPS):
    signal = np.asarray(signal, dtype=np.complex64)

    coefficients = firwin(
        numtaps=num_taps,
        cutoff=cutoff
    )

    filtered_signal = filtfilt(
        coefficients,
        [1.0],
        signal
    )

    return filtered_signal.astype(np.complex64)


# AMPLITUDE NORMALIZATION
def normalize_amplitude(signal):
    signal = np.asarray(signal, dtype=np.complex64)

    rms = np.sqrt(
        np.mean(np.abs(signal) ** 2)
    )

    if rms < 1e-12:
        return signal

    return (signal / rms).astype(np.complex64)


# DSP PREPROCESSING
def preprocess_signal(signal):
    signal = remove_dc_offset(signal)
    signal = low_pass_filter(signal)
    signal = normalize_amplitude(signal)

    return signal.astype(np.complex64)

# DATASET PREPROCESSING

def preprocess_dataset(noisy_signals):
    noisy_signals = np.asarray(noisy_signals)

    if noisy_signals.ndim != 2:
        raise ValueError(
            f"Expected 2D input, got {noisy_signals.shape}"
        )

    dsp_signals = np.empty(
        noisy_signals.shape,
        dtype=np.complex64
    )

    for i in range(len(noisy_signals)):
        dsp_signals[i] = preprocess_signal(
            noisy_signals[i]
        )

    return dsp_signals


# PROCESS BPSK AND QPSK

DATA_DIR = Path("data")
OUTPUT_DIR = DATA_DIR / "dsp"

for modulation in ["bpsk", "qpsk"]:

    modulation_output = (
        OUTPUT_DIR / f"{modulation}_dsp"
    )

    modulation_output.mkdir(
        parents=True,
        exist_ok=True
    )

    for split in ["train", "test", "validation"]:

        input_path = (
            DATA_DIR
            / modulation
            / split
            / f"{split}_noisy.npy"
        )

        output_path = (
            modulation_output
            / f"{split}_dsp.npy"
        )

        if not input_path.exists():
            print(f"Skipping: {input_path}")
            continue

        noisy_signals = np.load(input_path)

        dsp_signals = preprocess_dataset(
            noisy_signals
        )

        np.save(
            output_path,
            dsp_signals
        )

        print(
            f"{modulation.upper()} {split}: "
            f"{noisy_signals.shape} → "
            f"{dsp_signals.shape}"
        )

print("DSP preprocessing complete.")