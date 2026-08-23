import numpy as np
from scipy.signal import firwin, filtfilt
import os
from clean_generator import rrc_filter

# DSP PARAMETERS


# cutoff chosen slightly above rrc bandwidth for beta = 0.35 and sps = 8
CUTOFF = 0.18
NUM_TAPS = 101


def remove_dc_offset(signal):
    signal = np.asarray(signal, dtype=np.complex64)

    dc_offset = np.mean(signal)

    return (signal - dc_offset).astype(np.complex64)


def matched_filter(signal, rrc):
    """
    applies a matched filter following the same params as the rrc filter.
    """

    filtered_signal = np.convolve(signal,rrc,mode="same")

    return filtered_signal.astype(np.complex64)


def low_pass_filter(signal,cutoff=CUTOFF,num_taps=NUM_TAPS):

    signal = np.asarray(signal,dtype=np.complex64)

    coefficients = firwin(numtaps=num_taps,cutoff=cutoff,window=("kaiser", 8.0))

    real_filtered = filtfilt(coefficients,[1.0],signal.real)

    imag_filtered = filtfilt(coefficients,[1.0],signal.imag)

    filtered_signal = (real_filtered +1j * imag_filtered)

    return filtered_signal.astype(np.complex64)


def normalize_amplitude(signal):
    """
    Normalizes the amplitude of the signal and normalizes the average power to 1.
    ensures that all the signals are consistent in terms of power.
    """

    signal = np.asarray(signal, dtype=np.complex64)

    rms = np.sqrt(np.mean(np.abs(signal) ** 2))

    # check to avoid division by zero
    if rms < 1e-12:
        return signal

    return (signal / rms).astype(np.complex64)


# DSP PREPROCESSING
def preprocess_signal(signal,rrc):
    """
    processes the signals and returns them
    """

    signal = remove_dc_offset(signal)

    signal = matched_filter(signal,rrc)

    signal = low_pass_filter(signal)    

    signal = normalize_amplitude(signal)

    return signal.astype(np.complex64)


def preprocess_dataset(noisy_signals,sps=8):
    """
    Applies DSP preprocessing to an entire dataset.
    """

    noisy_signals = np.asarray(noisy_signals)

    if noisy_signals.ndim != 2:
        raise ValueError(f"Expected 2D input, got {noisy_signals.shape}")

    # Generate the RRC filter once
    rrc = rrc_filter(beta=0.35,span=8,sps=sps)

    dsp_signals = np.empty(noisy_signals.shape,dtype=np.complex64)

    for i in range(len(noisy_signals)):
        dsp_signals[i] = preprocess_signal(noisy_signals[i],rrc)

    return dsp_signals

def generate_reference(signal,rrc):
    """
    Ground truth target reference generation for training and evaluation.
    To ensure that its on the same scale as the processed signals
    """

    signal = matched_filter(signal,rrc)

    signal = low_pass_filter(signal)

    signal = normalize_amplitude(signal)

    return signal.astype(np.complex64)

def preprocess_dataset_reference(clean_signals,sps = 8):
    """
    Applies the reference-generation chain to an entire clean dataset.
    Mirrors preprocess_dataset but for building eval/training targets.
    """

    clean_signals = np.asarray(clean_signals)

    if clean_signals.ndim != 2:
        raise ValueError(f"Expected 2D input, got {clean_signals.shape}")

    # Generate the RRC filter once
    rrc = rrc_filter(beta=0.35,span=8,sps=sps)

    target_signals = np.empty(clean_signals.shape,dtype=np.complex64)

    for i in range(len(clean_signals)):
        target_signals[i] = generate_reference(clean_signals[i],rrc)

    return target_signals

def generate_dsp(split_config = {"train": 10000,"test": 3000,"validation": 2000}, sps = 8):
    """
    processes the noisy signals and saves them
    """

    modulations = ["BPSK", "QPSK"]

    for modulation in modulations:

        for split, samples in split_config.items():

            save_dir = os.path.join("data",modulation.lower(),split)

            noisy_path = os.path.join(save_dir,f"{split}_noisy.npy")
            clean_path = os.path.join(save_dir,f"{split}_clean.npy")
            faded_path = os.path.join(save_dir,f"{split}_faded.npy")


            if not os.path.exists(noisy_path):
                print(f"Skipping: {noisy_path}")
                continue

            noisy_signals = np.load(noisy_path)

            dsp_signals = preprocess_dataset(noisy_signals, sps=sps)

            dsp_path = os.path.join(save_dir,f"{split}_dsp.npy")

            np.save(dsp_path,dsp_signals)

            print(f"Saved DSP signals: {dsp_path}")

            if not os.path.exists(clean_path):
                print(f"Skipping target generation: {clean_path}")
                continue

            clean_signals = np.load(clean_path)

            target_signals = preprocess_dataset_reference(clean_signals, sps=sps)

            target_path = os.path.join(save_dir,f"{split}_target.npy")

            np.save(target_path,target_signals)

            print(f"Saved target signals: {target_path}")

            if os.path.exists(faded_path):
                faded_signals = np.load(faded_path)

                faded_target_signals = preprocess_dataset_reference(faded_signals,sps=sps)

                faded_target_path = os.path.join(save_dir,f"{split}_faded_target.npy")

                np.save(faded_target_path,faded_target_signals)

                print(f"Saved faded target signals: {faded_target_path}")

