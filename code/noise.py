import numpy as np

# Discrete SNR values in dB
disc_SNR_dB = [-5, 0, 5, 10, 15, 20]

def snr_transform(snr_db):
    """Converts SNR from dB to a linear form"""
    snr_linear = 10 ** (snr_db / 10)
    return snr_linear

def awgn(bpsk_symbols,snr_db):
    """Computes and Adds the required AWGN noise to the required bits"""

    snr_linear = snr_transform(snr_db)

    sigma = np.sqrt(1 / (2 * snr_linear)) # computes the standard deviation of the guassian dist.

    awgn_noise = np.random.normal(0,sigma,len(bpsk_symbols))
    
    received_signal = bpsk_symbols + awgn_noise
    
    return received_signal,awgn_noise