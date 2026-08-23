import numpy as np
import os


def generate_bits(bit_num):
    """Generates random bits of length bit_num"""
    bits = np.random.randint(0, 2, bit_num)
    return bits


def bpsk_mod(bits):
    """Converts raw signals into BPSK modulated symbols
    0 -> -1
    1 -> +1    
    """
    bpsk_symbols = 2 * bits - 1
    return bpsk_symbols


def qpsk_mod(bits):
    """Converts raw signals into QPSK modulated symbols
    It considers a set of 2 bits
    sqrt(2) is used to normalize such that the average symbol power = 1
    00 -> 1 + j / sqrt(2)
    01 -> -1 + j/ sqrt(2)
    11 -> -1 -j/ sqrt(2)
    10 ->  1 - j/ sqrt(2)    
    """

    if len(bits)%2==0:
        pairs = bits.reshape(-1,2)

        qpsk_symbols = np.empty(len(pairs),dtype = np.complex64)

        for i, (bit1,bit2) in enumerate(pairs):
            # bit 1 -> complex part
            if bit1 == 0:
                imag = 1
            else:
                imag = -1

            # bit 2 -> real part    
            if bit2 == 0:
                real = 1
            else:
                real = -1
            qpsk_symbols[i] = (real + (imag*1j))/np.sqrt(2)
        return qpsk_symbols
    else :
        return None

    
def upsampling(symbols,sps = 8):
    """ 
    Upsamples the symbols by insterting 0s between the symbols

    sps = samples per symbol

    required to represent the symbols as waves, with time differing amplitudes
    useful for filtering and reducing noise further on
    """

    upsampled_symbols = np.zeros(len(symbols) * sps, dtype=symbols.dtype)
    upsampled_symbols[::sps] = symbols

    return upsampled_symbols


def rrc_filter(beta = 0.35,span = 8,sps = 8):
    """
    Generates a root raised cosine filter for pulse shaping
    beta: roll-off factor
    sps: samples per symbol (essentially the resolution)
    span: filter length (symbols)

    """
    
    N = span * sps 

    # creates a time vector centered around zero
    time = np.arange(-N / 2, N / 2 + 1) / sps

    h = np.zeros_like(time)

    # when t=0 
    h[time == 0] = 1 - beta + (4 * beta / np.pi)

    # when t=±T/(4 * beta)
    t_special = np.abs(time) == (1 / (4 * beta))
    h[t_special] = (beta / np.sqrt(2)) * (
        ((1 + 2 / np.pi) * (np.sin(np.pi / (4 * beta)))) +
        ((1 - 2 / np.pi) * (np.cos(np.pi / (4 * beta))))
    )

    # otherwise 
    general_case = ~t_special & (time != 0)
    h[general_case] = (
        (np.sin(np.pi * time[general_case] * (1 - beta)) +
         4 * beta * time[general_case] *
         np.cos(np.pi * time[general_case] * (1 + beta))) /
        (np.pi * time[general_case] *
         (1 - (4 * beta * time[general_case]) ** 2))
    )

    # normalize energy to be 1
    h /= np.sqrt(np.sum(h ** 2))

    return h


def generate_data(signal_length = 1024, sps = 8, rrc_beta = 0.35, rrc_span = 8, split_config = {"train": 10000, "test": 3000, "validation": 2000}):
    """
    Dataset Generation

    Twice the number of bits are required for QPSK as compared to BPSK since each symbol in QPSK represents 2 bits
    
    """
    modulations = ["BPSK", "QPSK"]

    waveform_length = signal_length * sps
    
    for modulation in modulations:

        save_dir = f"data/{modulation.lower()}"
        os.makedirs(save_dir, exist_ok=True)

        for split, samples in split_config.items():
            save_dir = os.path.join("data",modulation.lower(),split)

            os.makedirs(save_dir, exist_ok=True)

            if modulation == "BPSK":
                clean_signals = np.empty((samples,waveform_length), dtype=np.float32)
                
            elif modulation == "QPSK":
                clean_signals = np.empty((samples, waveform_length), dtype=np.complex64)

            rrc = rrc_filter(rrc_beta, rrc_span, sps)
            np.save("data/rrc_filter.npy", rrc)

            for i in range(samples):
                
                if modulation == "BPSK":
                    bits = generate_bits(signal_length)
                    symbols = bpsk_mod(bits).astype(np.float32)
                    
                elif modulation == "QPSK":
                    bits = generate_bits(signal_length * 2) 
                    symbols = qpsk_mod(bits).astype(np.complex64)

                upsampled = upsampling(symbols, sps)

                waveform = np.convolve(upsampled,rrc,mode="same")

                clean_signals[i] = waveform
            
            save_path = os.path.join(save_dir, f"{split}_clean.npy")
            np.save(save_path, clean_signals)
            print(f"Saved {split} clean signals for {modulation} at {save_path}")
