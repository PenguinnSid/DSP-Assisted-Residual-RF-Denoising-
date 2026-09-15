import numpy as np


def load_split(split, sps=8, seed=42, data_root="data"):
    """
    Loads the faded+noisy signals and their AWGN-only denoising targets.

    -> input = noisy signals (fading + AWGN)
    -> output/labels = faded_target signals (fading present, AWGN removed)

    Fading is held constant on both input and label, so the model only
    needs to learn to remove the additive noise component — fading
    correction is left to a separate equalizer stage.

    Combines the BPSK and QPSK splits into one dataset to avoid creating 2 models

    Stacks the real and imaginary parts of the complex signals into 2 channels for both input and output

    Shuffles the dataset with a fixed and consistent random seed to maintain the relative association between the different files

    Unlike the coefficient predictor, this is a sequence-to-sequence task:
    both X and y have shape (N, seq_len, 2), not (N, 2).
    """

    modulations = ["bpsk", "qpsk"]

    noisy_list = []
    target_list = []

    for modulation in modulations:
        base = f"{data_root}/{modulation}/{split}"

        noisy = np.load(f"{base}/{split}_noisy.npy")
        faded_target = np.load(f"{base}/{split}_faded_target.npy")

        noisy_list.append(noisy)
        target_list.append(faded_target)

    noisy_all = np.concatenate(noisy_list, axis=0)
    target_all = np.concatenate(target_list, axis=0)

    """ Shuffling dataset with a consistent random seed to maintain relative association between diff files """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(noisy_all))

    noisy_all = noisy_all[perm]
    target_all = target_all[perm]

    """ Stacking inputs and outputs """
    X = np.stack([noisy_all.real, noisy_all.imag], axis=-1).astype(np.float32)

    y = np.stack([target_all.real, target_all.imag], axis=-1).astype(np.float32)

    return X, y