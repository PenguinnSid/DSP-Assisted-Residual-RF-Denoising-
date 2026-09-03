import numpy as np


def load_split(split, sps=8, seed=42, data_root="data", downsample_factor=8):
    """
    Loads the noisy signals and channel coefficients.

    -> input = noisy signals (downsampled)
    -> output/labels = channel coefficients

    Combines the BPSK and QPSK splits into one dataset to avoid creating 2 models

    Stacks the real and imaginary parts of the complex signals into 2 channels for both input and output

    Shuffles the dataset with a fixed and consistent random seed to maintain the relative association between the different files

    downsample_factor: reduces the 8192-length sequence by averaging over
    non-overlapping blocks of this size (default 8, matching sps — since h
    is constant across the whole sequence, this preserves the signal needed
    to estimate h while making the sequence far more tractable for an LSTM,
    which otherwise struggles with vanishing gradients over 8192 timesteps)
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

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(noisy_all))

    noisy_all = noisy_all[perm]
    h_all = h_all[perm]

    # --- downsampling: average-pool non-overlapping blocks ---
    if downsample_factor > 1:
        n_samples, seq_len = noisy_all.shape
        new_len = seq_len // downsample_factor

        noisy_all = noisy_all[:, :new_len * downsample_factor]           # trim to a multiple
        noisy_all = noisy_all.reshape(n_samples, new_len, downsample_factor)
        noisy_all = noisy_all.mean(axis=2)                                # average-pool each block

    X = np.stack([noisy_all.real, noisy_all.imag], axis=-1).astype(np.float32)
    y = np.stack([h_all.real, h_all.imag], axis=-1).astype(np.float32)

    return X, y