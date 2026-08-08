import numpy as np
import os

# Discrete SNR values in dB
disc_SNR_dB = [-5, 0, 5, 10, 15, 20]

def snr_transform(snr_db):
    """Converts SNR from dB to a linear form"""
    snr_linear = 10 ** (snr_db / 10)
    return snr_linear

def rayleigh(signal):
    """ Applies a rayleigh fading channel to the signal
    
    received signal = h * transmitted signal

    1/sqrt(2) is used to normalize such that the average symbol power = 1
    the underlying dist. of the real and imaginary components are gaussian
    """

    sigma = 1/np.sqrt(2)
    
    fade_coefficient = (
        np.random.normal(0,sigma,len(signal)) # I copmponent
        + 1j * np.random.normal(0,sigma,len(signal)) # Q copmponent
    )
    
    faded_signal = fade_coefficient * signal
    
    return faded_signal

def awgn(signal,snr_db,modulation):
    """Computes and Adds the required AWGN noise to the required bits
    
    received signal = transmitted signal + n

    diff sigmas for qpsk and bpsk since qpsk has twice the bits trasmitted per signal
    BPSK:  1 bit/symbol -> Eb = 1
    QPSK:  2 bits/symbol -> Eb = 1/2
    where Eb is the energy per bit
    """

    snr_linear = snr_transform(snr_db)
    
    # computes the standard deviation of the guassian dist.
    if modulation == "BPSK":
        sigma = np.sqrt(1 / (2 * snr_linear))
        
    elif modulation == "QPSK":
        sigma = np.sqrt(1 / (4 * snr_linear))

    awgn_noise = (
        np.random.normal(0,sigma,len(signal)) # I copmponent
        + 1j * np.random.normal(0, sigma, len(signal)) # Q copmponent
    )
    
    received_signal = signal + awgn_noise
    
    return received_signal,awgn_noise

def noisy_data(clean_signals, modulation):
    """Generates the noise for the given signal and modulation type"""
    
    samples, signal_length = clean_signals.shape

    noisy_signals = np.empty((samples, signal_length), dtype=np.complex64)

    snr_values = np.empty((samples,), dtype=np.float32)

    for i in range(samples):
        snr_db = np.random.choice(disc_SNR_dB)
        snr_values[i] = snr_db

        faded_signal = rayleigh(clean_signals[i])
        received_signal,awgn_noise = awgn(faded_signal, snr_db, modulation)
        noisy_signals[i] = received_signal

    return noisy_signals, snr_values

def generate_noise():
    """
    generates the noise for the clean signals and saves them

    """
    modulations = ["BPSK", "QPSK"]
    split_config = {
        "train": 10000,
        "test": 5000,
        "validation": 2000
        }
    
    for modulation in modulations:

        for split, samples in split_config.items():

            save_dir = os.path.join("data",modulation.lower(),split)

            clean_path = os.path.join(save_dir, f"{split}_clean.npy")
            clean_signals = np.load(clean_path)

            noisy_signals = np.empty(clean_signals.shape, dtype=np.complex64)
            snr_values = np.empty(samples,dtype=np.float32)

            noisy_signals, snr_values = noisy_data(clean_signals, modulation)

            noisy_path = os.path.join(save_dir,f"{split}_noisy.npy")

            snr_path = os.path.join(save_dir,f"{split}_snr.npy")

            np.save(noisy_path, noisy_signals)
            np.save(snr_path, snr_values)

            print(f"Saved noisy signals: {noisy_path}")
            print(f"Saved SNR values: {snr_path}")