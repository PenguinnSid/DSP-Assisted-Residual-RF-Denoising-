"""
General evaluation of the ensemble vs. its individual members and the raw noisy input.

Run from anywhere:
    python models/ensemble/eval.py
    python models/ensemble/eval.py --modes mean median weighted --primary weighted
    python models/ensemble/eval.py --data-root path/to/data --split test

What you get:
  1. Overall table: MSE, output SNR and SNR gain for Raw / AutoEncoder / CNN / LSTM / each ensemble mode
  2. Table of mean SNR gain per input-SNR level
  3. The standard evaluate_denoiser() report + boxplot PNG for the primary ensemble mode
     (saved as models/evaluation_plots/ensemble_<mode>_snr_stratified.png)
  4. models/evaluation_plots/ensemble_comparison.png and ensemble_summary.csv

Individual members are NOT passed to evaluate_denoiser here, so your existing
autoencoder / cnn / lstm plots in evaluation_plots/ are left untouched.
"""
import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ENSEMBLE_DIR = Path(__file__).resolve().parent
MODELS_DIR = ENSEMBLE_DIR.parent
sys.path.insert(0, str(ENSEMBLE_DIR))
sys.path.insert(0, str(MODELS_DIR))  # inserted last -> first on the path, so `evaluate` is models/evaluate.py

from evaluate import calculate_snr, evaluate_denoiser, per_sample_mse, to_complex  # noqa: E402
from load_awgn_data import load_split  # noqa: E402
from ensemble import DenoiserEnsemble  # noqa: E402

DEFAULT_DATA_ROOT = MODELS_DIR.parent / "data"  # same location CNN/eval.py uses
SNR_POOL = (-5, 0, 5, 10, 15, 20)
PLOTS_DIR = MODELS_DIR / "evaluation_plots"


def evaluate_ensemble(ens, split="test", data_root=DEFAULT_DATA_ROOT,
                      modes=("mean", "median", "weighted"), primary_mode="mean",
                      batch_size=64, snr_pool=SNR_POOL, save_outputs=True):
    """
    ens:           DenoiserEnsemble (any set of members with (N, L, 2) -> (N, L, 2))
    modes:         combine modes to compare. "weighted" fits its weights on the validation split.
    primary_mode:  the mode that gets the full evaluate_denoiser() report and boxplot.
    Returns dict {name: {"mse": (N,), "snr": (N,)}} of per-sample metrics.
    """
    X, y, snr_values, mod_labels = load_split(split, data_root=str(data_root))
    X_c, y_c = to_complex(X), to_complex(y)

    if "weighted" in modes:
        Xv, yv, _, _ = load_split("validation", data_root=str(data_root))
        weights, val_mse = ens.fit_weights(Xv, yv, batch_size)
        print("Validation MSE per member :", {k: round(v, 5) for k, v in val_mse.items()})
        print("Ensemble weights (1/MSE)  :", {k: round(v, 3) for k, v in weights.items()})

    member_preds = ens.predict_members(X, batch_size)

    predictions = {"Raw (noisy)": X}
    predictions.update(member_preds)
    for m in modes:
        predictions[f"Ensemble ({m})"] = ens.combine(member_preds, m)

    # ---- per-sample metrics against the faded target (same definition as evaluate.py)
    results = {}
    for name, pred in predictions.items():
        pc = to_complex(pred)
        results[name] = {"mse": per_sample_mse(y_c, pc), "snr": calculate_snr(y_c, pc)}
    raw_snr = results["Raw (noisy)"]["snr"]

    # ---- 1. overall table
    print(f"\n{'=' * 72}\nOVERALL — split '{split}', {len(X)} samples\n{'=' * 72}")
    print(f"{'':<22}{'MSE':>12}{'Out SNR (dB)':>16}{'SNR gain (dB)':>16}")
    for name, r in results.items():
        gain = np.nanmean(r["snr"] - raw_snr)
        print(f"{name:<22}{np.mean(r['mse']):>12.5f}{np.nanmean(r['snr']):>16.2f}{gain:>16.2f}")

    # ---- 2. gain per input SNR
    print(f"\n{'=' * 72}\nMEAN SNR GAIN (dB) BY INPUT SNR\n{'=' * 72}")
    print(f"{'':<22}" + "".join(f"{str(s) + ' dB':>9}" for s in snr_pool))
    gain_table = {}
    for name, r in results.items():
        if name == "Raw (noisy)":
            continue
        gain_table[name] = [np.nanmean((r["snr"] - raw_snr)[snr_values == s]) for s in snr_pool]
        print(f"{name:<22}" + "".join(f"{g:>9.2f}" for g in gain_table[name]))

    # ---- 3. standard report + boxplot for the primary ensemble mode
    evaluate_denoiser(
        model_name=f"Ensemble_{primary_mode}",
        y_pred_complex=to_complex(predictions[f"Ensemble ({primary_mode})"]),
        X_test_complex=X_c,
        target_complex=y_c,
        snr_values=snr_values,
        modulation_labels=mod_labels,
        snr_pool=snr_pool,
    )

    # ---- 4. comparison plot + csv
    if save_outputs:
        PLOTS_DIR.mkdir(exist_ok=True)
        fig, ax = plt.subplots(figsize=(8, 5))
        for name, g in gain_table.items():
            ax.plot(snr_pool, g, marker="o", lw=2.2 if name.startswith("Ensemble") else 1.2, label=name)
        ax.set_xlabel("Input SNR (dB)")
        ax.set_ylabel("Mean SNR gain (dB)")
        ax.set_title("SNR gain vs input SNR: members vs ensemble")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / "ensemble_comparison.png", dpi=150)
        plt.close(fig)

        with open(PLOTS_DIR / "ensemble_summary.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["model", "mse", "out_snr_db", "snr_gain_db"] + [f"gain_at_{s}dB" for s in snr_pool])
            for name, r in results.items():
                if name == "Raw (noisy)":
                    w.writerow([name, np.mean(r["mse"]), np.nanmean(r["snr"]), 0.0] + [0.0] * len(snr_pool))
                else:
                    w.writerow([name, np.mean(r["mse"]), np.nanmean(r["snr"]),
                                np.nanmean(r["snr"] - raw_snr)] + gain_table[name])
        print(f"\nSaved: {PLOTS_DIR / 'ensemble_comparison.png'}\nSaved: {PLOTS_DIR / 'ensemble_summary.csv'}")

    return results


def main():
    p = argparse.ArgumentParser(description="Evaluate the AutoEncoder + CNN + LSTM ensemble")
    p.add_argument("--split", default="test", choices=["train", "validation", "test"])
    p.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    p.add_argument("--modes", nargs="+", default=["mean", "median", "weighted"],
                   choices=["mean", "median", "weighted"])
    p.add_argument("--primary", default=None, help="mode for the full report/boxplot (default: first of --modes)")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    primary = args.primary or args.modes[0]
    if primary not in args.modes:
        p.error("--primary must be one of --modes")

    ens = DenoiserEnsemble(device=args.device)
    print("Members:", ", ".join(ens.names), "| device:", ens.device)
    evaluate_ensemble(ens, split=args.split, data_root=args.data_root, modes=tuple(args.modes),
                      primary_mode=primary, batch_size=args.batch_size)


if __name__ == "__main__":
    main()