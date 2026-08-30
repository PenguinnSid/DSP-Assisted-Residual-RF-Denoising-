import numpy as np


def load_split(split, sps=8, seed=42, data_root="data"):
    """
    Loads the noisy signals and channel coefficients.

    -> input = noisy signals
    -> output/labels = channel coefficients

    Combines the BPSK and QPSK splits into one dataset to avoid creating 2 models

    Stacks the real and imaginary parts of the complex signals into 2 channels for both input and output

    Shuffles the dataset with a fixed  and consistent random seed to maintain the relative association between the different files

    """

    modulations = ["bpsk", "qpsk"]

    noisy_list = []
    h_list = []

    for modulation in modulations:
        base = f"{data_root}/{modulation}/{split}"

        noisy = np.load(f"{base}/{split}_noisy.npy")     
        h_full = np.load(f"{base}/{split}_h.npy")        

        noisy_list.append(noisy)
        h_list.append(h_full[:, 0])                       

    noisy_all = np.concatenate(noisy_list, axis=0)        
    h_all = np.concatenate(h_list, axis=0)                 

    """ Shuffling dataset with a consistent random seed to maintain relative association between diff files """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(noisy_all))

    noisy_all = noisy_all[perm]
    h_all = h_all[perm]

    """ Stacking inputs and outputs """
    X = np.stack([noisy_all.real, noisy_all.imag], axis=-1).astype(np.float32) 

    y = np.stack([h_all.real, h_all.imag], axis=-1).astype(np.float32)         

    return X, y