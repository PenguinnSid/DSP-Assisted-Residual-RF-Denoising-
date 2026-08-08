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

def generate_data(samples = 10000, signal_length = 1024):
    """
    Dataset Generation

    Twice the number of samples are required for QPSK as compared to BPSK since each symbol in QPSK represents 2 bits
    
    """
    modulations = ["BPSK", "QPSK"]
    
    for modulation in modulations:

        save_dir = f"data/{modulation.lower()}"
        os.makedirs(save_dir, exist_ok=True)

        split_config = {
            "train": 10000,
            "test": 5000,
            "validation": 2000 
        }

        for split, samples in split_config.items():
            save_dir = os.path.join("data",modulation.lower(),split)

            os.makedirs(save_dir, exist_ok=True)

            if modulation == "BPSK":
                clean_signals = np.empty((samples,signal_length), dtype=np.float32)
                
            elif modulation == "QPSK":
                clean_signals = np.empty((samples, signal_length), dtype=np.complex64)

            for i in range(samples):
                
                if modulation == "BPSK":
                    bits = generate_bits(signal_length)
                    symbols = bpsk_mod(bits).astype(np.float32)
                    
                elif modulation == "QPSK":
                    bits = generate_bits(signal_length * 2) 
                    symbols = qpsk_mod(bits).astype(np.complex64)

                clean_signals[i] = symbols
            
            save_path = os.path.join(save_dir, f"{split}_clean.npy")
            np.save(save_path, clean_signals)
            print(f"Saved {split} clean signals for {modulation} at {save_path}")
