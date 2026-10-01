import numpy as np


def extract_features(noisy_signal):
    """
    Extracts hand-engineered statistical features from a complex noisy
    signal, intended to carry information about the fade coefficient h
    without requiring a full sequence model.

    |h| is recoverable from received signal power (RMS).
    Phase of h is partially recoverable from the correlation structure
    between the real and imaginary components (most directly for BPSK,
    where both parts are scaled copies of the same real waveform).
    """

    real = noisy_signal.real
    imag = noisy_signal.imag

    features = np.stack([
        np.mean(np.abs(noisy_signal) ** 2, axis=1),     # RMS power -> proxy for |h|^2
        np.mean(real, axis=1),
        np.mean(imag, axis=1),
        np.std(real, axis=1),
        np.std(imag, axis=1),
        np.mean(real * imag, axis=1),                    # cross-correlation -> phase info
        np.mean(real ** 2, axis=1) - np.mean(imag ** 2, axis=1),  # power difference -> phase info
        np.arctan2(np.mean(imag, axis=1), np.mean(real, axis=1)),
    ], axis=1)

    return features.astype(np.float32)


def load_split(split, seed=42, data_root="data"):
    """
    Loads noisy signals and channel coefficients, converts noisy signals
    into engineered feature vectors (rather than raw sequences), for use
    with a classical regression model (RandomForestRegressor etc).

    -> input  = engineered features from the noisy signal
    -> output = h.real, h.imag

    Combines BPSK and QPSK into one dataset. Shuffles with a fixed seed,
    keeping X/y aligned via a single permutation.
    """

    modulations = ["bpsk", "qpsk"]

    noisy_list = []
    h_list = []
    mod_label_list = []

    for modulation in modulations:
        base = f"{data_root}/{modulation}/{split}"

        noisy = np.load(f"{base}/{split}_noisy.npy")
        h_full = np.load(f"{base}/{split}_h.npy")

        noisy_list.append(noisy)
        h_list.append(h_full[:, 0])
        mod_label_list.append(np.full(len(noisy), modulation))

    noisy_all = np.concatenate(noisy_list, axis=0)
    h_all = np.concatenate(h_list, axis=0)
    mod_labels_all = np.concatenate(mod_label_list, axis=0)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(noisy_all))

    noisy_all = noisy_all[perm]
    h_all = h_all[perm]
    mod_labels_all = mod_labels_all[perm]

    X = extract_features(noisy_all)
    y = np.stack([h_all.real, h_all.imag], axis=-1).astype(np.float32)

    return X, y, mod_labels_all