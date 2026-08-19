import numpy as np
from pathlib import Path

DATA_DIR = Path("data")
DSP_DIR = DATA_DIR / "dsp"

MODULATIONS = ["bpsk", "qpsk"]
SPLITS = ["train", "test", "validation"]


def rms(x):
    return np.sqrt(np.mean(np.abs(x) ** 2))


def verify_dataset(modulation, split):

    noisy_path = (
        DATA_DIR / modulation / split /
        f"{split}_noisy.npy"
    )

    clean_path = (
        DATA_DIR / modulation / split /
        f"{split}_clean.npy"
    )

    dsp_path = (
        DSP_DIR / f"{modulation}_dsp" /
        f"{split}_dsp.npy"
    )

    print("\n")
    #print("\n" + "=" * 70)
    print(f"{modulation.upper()} - {split.upper()}")
    print("=" * 70)

    # Check files

    for path in [noisy_path, clean_path, dsp_path]:

        if path.exists():
            print(f"✓ Found: {path}")
        else:
            print(f"✗ Missing: {path}")

    if not noisy_path.exists() or not dsp_path.exists():
        return

    # Load data

    noisy = np.load(noisy_path)
    dsp = np.load(dsp_path)

    clean = None

    if clean_path.exists():
        clean = np.load(clean_path)

    # Shape

    print("\nSHAPE")

    print("Noisy :", noisy.shape)
    print("DSP   :", dsp.shape)

    if clean is not None:
        print("Clean :", clean.shape)

    if noisy.shape == dsp.shape:
        print("✓ DSP shape matches noisy data")
    else:
        print("✗ DSP shape changed!")

    # Dtype

    print("\nDTYPE")

    print("Noisy :", noisy.dtype)
    print("DSP   :", dsp.dtype)

    # NaN / Inf

    print("\nVALIDITY")

    print(
        "Noisy NaN:",
        np.isnan(noisy).any(),
        "| Inf:",
        np.isinf(noisy).any()
    )

    print(
        "DSP NaN:",
        np.isnan(dsp).any(),
        "| Inf:",
        np.isinf(dsp).any()
    )

    if not np.isnan(dsp).any() and not np.isinf(dsp).any():
        print("✓ DSP contains no NaN/Inf")

    # Mean / DC offset

    print("\nDC OFFSET")

    noisy_mean = np.mean(noisy, axis=1)
    dsp_mean = np.mean(dsp, axis=1)

    print(
        "Average |mean| before:",
        np.mean(np.abs(noisy_mean))
    )

    print(
        "Average |mean| after :",
        np.mean(np.abs(dsp_mean))
    )

    if np.mean(np.abs(dsp_mean)) < np.mean(np.abs(noisy_mean)):
        print("✓ DC component was reduced")
    else:
        print("⚠ DC component was not reduced")

    # RMS / normalization

    print("\nAMPLITUDE NORMALIZATION")

    noisy_rms = np.mean(
        np.sqrt(
            np.mean(np.abs(noisy) ** 2, axis=1)
        )
    )

    dsp_rms = np.mean(
        np.sqrt(
            np.mean(np.abs(dsp) ** 2, axis=1)
        )
    )

    print(
        "Average RMS before:",
        noisy_rms
    )

    print(
        "Average RMS after :",
        dsp_rms
    )

    if np.isclose(dsp_rms, 1.0, atol=0.05):
        print("✓ DSP signals are approximately unit RMS")
    else:
        print("⚠ DSP RMS is not approximately 1")

    # Check whether DSP actually changed the data

    print("\nDATA CHANGE")

    difference = dsp - noisy

    mean_difference = np.mean(
        np.abs(difference)
    )

    print(
        "Mean |DSP - noisy|:",
        mean_difference
    )

    if mean_difference > 1e-6:
        print("✓ DSP changed the input data")
    else:
        print("✗ DSP output is almost identical to input")

    # Compare against clean signal

    if clean is not None:

        print("\nCOMPARISON WITH CLEAN TARGET")

        noisy_mse = np.mean(
            np.abs(clean - noisy) ** 2
        )

        dsp_mse = np.mean(
            np.abs(clean - dsp) ** 2
        )

        print(
            "Noisy → Clean MSE:",
            noisy_mse
        )

        print(
            "DSP   → Clean MSE:",
            dsp_mse
        )

        if dsp_mse < noisy_mse:
            print(
                "✓ DSP reduced MSE relative to noisy signal"
            )
        else:
            print(
                "⚠ DSP increased MSE relative to noisy signal"
            )


# ============================================================
# RUN VERIFICATION
# ============================================================

for modulation in MODULATIONS:

    for split in SPLITS:

        verify_dataset(
            modulation,
            split
        )