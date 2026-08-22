import numpy as np


def matched_filter(signal):
    """
    Applies an RRC matched filter.

    Uses the same pulse shape used by the transmitter.
    """

    rrc = np.load(
        "data/rrc_filter.npy"
    )

    filtered_signal = np.convolve(
        signal,
        rrc,
        mode="same"
    )

    return filtered_signal.astype(
        np.complex64
    )