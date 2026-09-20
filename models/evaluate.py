from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# METRIC HELPERS
# ============================================================

def to_complex(x):
    """Converts a real/imag-stacked array (N, seq_len, 2) back to complex (N, seq_len)."""
    return x[..., 0] + 1j * x[..., 1]


def per_sample_mse(reference, received):
    """Per-sample MSE. reference, received: (N, seq_len) complex."""
    return np.mean(np.abs(reference - received) ** 2, axis=1)


def calculate_snr(reference, received):
    """
    SNR (dB) of `received` relative to `reference`, per-sample.
    reference, received: (N, seq_len) complex arrays.
    Returns: (N,) array of per-sample SNR values.
    """
    signal_power = np.mean(np.abs(reference) ** 2, axis=1)
    noise = received - reference
    noise_power = np.mean(np.abs(noise) ** 2, axis=1)

    noise_power = np.where(noise_power < 1e-12, np.nan, noise_power)
    snr = 10 * np.log10(signal_power / noise_power)
    return snr


# ============================================================
# MAIN EVALUATION — model-agnostic
# ============================================================

def evaluate_denoiser(model_name, y_pred_complex, X_test_complex, target_complex,
                       snr_values, modulation_labels, snr_pool=(-5, 0, 5, 10, 15, 20)):
    """
    Computes MSE/SNR comparison metrics and produces an SNR-stratified
    boxplot for any denoiser's predictions, regardless of framework
    or architecture (LSTM, CNN, autoencoder, classical DSP, etc.).

    model_name:         label for this model, e.g. "LSTM", "CNN", "Autoencoder", "DSP", "Raw"
    y_pred_complex:     (N, seq_len) complex — model's denoised output
    X_test_complex:     (N, seq_len) complex — the noisy input (for baseline comparison)
    target_complex:     (N, seq_len) complex — ground truth reference (faded_target)
    snr_values:         (N,) — per-sample SNR used at generation time
    modulation_labels:  (N,) array of "bpsk"/"qpsk" strings, one per sample
    snr_pool:           discrete SNR levels present in the dataset

    Returns a dict of per-sample metric arrays, and saves a boxplot PNG
    to models/evaluation_plots/{model_name}_snr_stratified.png
    """

    mse_pred = per_sample_mse(target_complex, y_pred_complex)
    mse_raw = per_sample_mse(target_complex, X_test_complex)

    snr_pred = calculate_snr(target_complex, y_pred_complex)
    snr_raw = calculate_snr(target_complex, X_test_complex)

    # ------------------------------------------------------
    # COMPARISON TABLE — per modulation + pooled
    # ------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"COMPARISON TABLE — {model_name}")
    print(f"{'='*70}")
    print(f"{'':10} {'Raw MSE':>10} {'Model MSE':>12} {'Raw SNR':>10} {'Model SNR':>12}")

    for mod in ["bpsk", "qpsk"]:
        mask = modulation_labels == mod
        print(f"{mod.upper():10} "
              f"{np.mean(mse_raw[mask]):10.4f} "
              f"{np.mean(mse_pred[mask]):12.4f} "
              f"{np.nanmean(snr_raw[mask]):10.2f} "
              f"{np.nanmean(snr_pred[mask]):12.2f}")

    print(f"{'Total':10} "
          f"{np.mean(mse_raw):10.4f} "
          f"{np.mean(mse_pred):12.4f} "
          f"{np.nanmean(snr_raw):10.2f} "
          f"{np.nanmean(snr_pred):12.2f}")

    # ------------------------------------------------------
    # SNR-STRATIFIED BOXPLOT
    # ------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    mse_by_snr = [mse_pred[snr_values == s] for s in snr_pool]
    snr_improvement_by_snr = [
        (snr_pred[snr_values == s] - snr_raw[snr_values == s]) for s in snr_pool
    ]

    axes[0].boxplot(mse_by_snr, tick_labels=[f"{s}dB" for s in snr_pool])
    axes[0].set_title(f"{model_name}: Per-sample MSE by input SNR")
    axes[0].set_xlabel("Input SNR")
    axes[0].set_ylabel("MSE (vs target)")

    axes[1].boxplot(snr_improvement_by_snr, tick_labels=[f"{s}dB" for s in snr_pool])
    axes[1].set_title(f"{model_name}: SNR improvement by input SNR")
    axes[1].set_xlabel("Input SNR")
    axes[1].set_ylabel("SNR improvement (dB)")

    plt.tight_layout()

    out_dir = Path(__file__).resolve().parent / "evaluation_plots"
    out_dir.mkdir(exist_ok=True)
    plt.savefig(out_dir / f"{model_name.lower()}_snr_stratified.png", dpi=150)
    print(f"\nSaved plot: {out_dir / f'{model_name.lower()}_snr_stratified.png'}")

    plt.close(fig)

    return {
        "mse_pred": mse_pred,
        "mse_raw": mse_raw,
        "snr_pred": snr_pred,
        "snr_raw": snr_raw,
    }