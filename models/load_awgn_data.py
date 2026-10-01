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

    noisy_list, target_list, snr_list, mod_label_list = [], [], [], []

    for modulation in modulations:
        base = f"{data_root}/{modulation}/{split}"

        # Load the noisy (fading+AWGN) input
        noisy = np.load(f"{base}/{split}_noisy.npy")
        # Load the target (faded signal without AWGN). The original code expected
        # a file named `{split}_faded_target.npy` which does not exist in the
        # repository. The available dataset provides `{split}_faded.npy` (faded
        # signal) and `{split}_clean.npy` (clean signal). For the denoising task
        # we want the faded signal as the label, so we fall back to that filename.
        try:
            faded_target = np.load(f"{base}/{split}_faded_target.npy")
        except FileNotFoundError:
            # Use the faded version if the *_faded_target.npy file is missing
            faded_target = np.load(f"{base}/{split}_faded.npy")
        snr = np.load(f"{base}/{split}_snr.npy")

        noisy_list.append(noisy)
        target_list.append(faded_target)
        snr_list.append(snr)
        mod_label_list.append(np.full(len(noisy), modulation))

    noisy_all = np.concatenate(noisy_list, axis=0)
    target_all = np.concatenate(target_list, axis=0)
    snr_all = np.concatenate(snr_list, axis=0)
    mod_labels_all = np.concatenate(mod_label_list, axis=0)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(noisy_all))

    noisy_all = noisy_all[perm]
    target_all = target_all[perm]
    snr_all = snr_all[perm]             
    mod_labels_all = mod_labels_all[perm]  

    X = np.stack([noisy_all.real, noisy_all.imag], axis=-1).astype(np.float32)
    y = np.stack([target_all.real, target_all.imag], axis=-1).astype(np.float32)

    return X, y, snr_all, mod_labels_all