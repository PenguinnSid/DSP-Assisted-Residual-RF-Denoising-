import numpy as np
import os

# Discrete SNR values in dB
disc_SNR_dB = [-5, 0, 5, 10, 15, 20]

def snr_transform(snr_db):
    """Converts SNR from dB to a linear form"""
    snr_linear = 10 ** (snr_db / 10)
    return snr_linear

def rayleigh(signal,sps = 8):
    """ Applies a rayleigh fading channel to the signal
    
    received signal = h * transmitted signal

    1/sqrt(2) is used to normalize such that the average symbol power = 1
    the underlying dist. of the real and imaginary components are gaussian

    Assumes len(signal) is a multiple of sps
    """

    sigma = 1/np.sqrt(2)

    num_symbols = len(signal) // sps

    symbol_fade = (
    np.random.normal(0,sigma,num_symbols) # I component
    + 1j * np.random.normal(0,sigma,num_symbols) # Q component
    )
    
    fade_coefficient = np.repeat(symbol_fade, sps)

    faded_signal = fade_coefficient * signal

    return faded_signal, fade_coefficient

def awgn(signal,snr_db):
    """Computes and Adds the required AWGN noise to the required signal
    
    received signal = transmitted signal + n

    noise power is calculated from the actual signal power so that the requested SNR is maintained for both BPSK and QPSK.
    """

    snr_linear = snr_transform(snr_db)
    signal_power = np.mean(np.abs(signal) ** 2)
    noise_power = signal_power / snr_linear
    sigma = np.sqrt(noise_power / 2)

    awgn_noise = (
        np.random.normal(0,sigma,len(signal)) # I component
        + 1j * np.random.normal(0,sigma,len(signal)) # Q component
    )
    
    received_signal = signal + awgn_noise
    
    return received_signal,awgn_noise

def noisy_data(clean_signals,sps = 8):
    """Generates the noise for the given signal and modulation type"""
    
    samples, signal_length = clean_signals.shape
    faded_signals = np.empty((samples, signal_length), dtype=np.complex64)
    fade_coefficients = np.empty((samples, signal_length), dtype=np.complex64)
    noisy_signals = np.empty((samples, signal_length), dtype=np.complex64)

    snr_values = np.empty((samples,), dtype=np.float32)

    # creates a balanced pool for sampling SNRs to avoid any SNR imbalances
    # tiles the values till the required number of samples
    snr_pool = np.tile(disc_SNR_dB,int(np.ceil(samples / len(disc_SNR_dB))))[:samples]
    np.random.shuffle(snr_pool)

    for i in range(samples):
        snr_db = snr_pool[i]
        snr_values[i] = snr_db


        faded_signal, fade_coefficient = rayleigh(clean_signals[i], sps)
        faded_signals[i] = faded_signal
        fade_coefficients[i] = fade_coefficient

        #received_signal,awgn_noise = awgn(clean_signals[i], snr_db)

        received_signal,awgn_noise = awgn(faded_signal, snr_db)
        noisy_signals[i] = received_signal


    return noisy_signals, faded_signals, fade_coefficients, snr_values

def generate_noise(split_config = {"train": 10000,"test": 3000,"validation": 2000 },sps = 8):
    """
    generates the noise for the clean signals and saves them

    """
    modulations = ["BPSK", "QPSK"]
    
    for modulation in modulations:

        for split, samples in split_config.items():

            save_dir = os.path.join("data",modulation.lower(),split)

            clean_path = os.path.join(save_dir, f"{split}_clean.npy")
            clean_signals = np.load(clean_path)

            noisy_signals, faded_signals, fade_coefficients, snr_values = noisy_data(clean_signals, sps)

            noisy_path = os.path.join(save_dir,f"{split}_noisy.npy")
            faded_path = os.path.join(save_dir,f"{split}_faded.npy")
            h_path = os.path.join(save_dir,f"{split}_h.npy")
            snr_path = os.path.join(save_dir,f"{split}_snr.npy")

            np.save(noisy_path, noisy_signals)
            np.save(faded_path, faded_signals)
            np.save(h_path, fade_coefficients)
            np.save(snr_path, snr_values)

            print(f"Saved noisy signals: {noisy_path}")
            print(f"Saved faded signals: {faded_path}")
            print(f"Saved fade coefficients: {h_path}")
            print(f"Saved SNR values: {snr_path}")