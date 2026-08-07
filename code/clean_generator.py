import numpy as np
import os


def generate_bits(bit_num):
    """Generates random bits of length bit_num"""
    bits = np.random.randint(0, 2, bit_num)
    return bits


def bpsk_mod(bits):
    """Converts raw signals into BPSK modulated symbols"""
    bpsk_symbols = 2 * bits - 1
    return bpsk_symbols


def qpsk_mod(bits):
    print("Needs to be added.")
    #qpsk_symbols = 
    #return qpsk_symbols

def generate_data(samples = 10000, signal_length = 1024):
    """Dataset Generation Test"""


    modulations = ["BPSK", "QPSK"]
    
    # only generate bpsk for now, qpsk will be added later
    for modulation in modulations[:1]:

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
                clean_signals = np.empty((samples, signal_length // 2), dtype=np.complex64)

            for i in range(samples):
                
                bits = generate_bits(signal_length)
            
                if modulation == "BPSK":
                    symbols = bpsk_mod(bits).astype(np.float32)
                    
                elif modulation == "QPSK":
                    symbols = qpsk_mod(bits).astype(np.complex64)

                clean_signals[i] = symbols
            
            save_path = os.path.join(save_dir, f"{split}_clean.npy")
            np.save(save_path, clean_signals)
            print(f"Saved {split} clean signals for {modulation} at {save_path}")
