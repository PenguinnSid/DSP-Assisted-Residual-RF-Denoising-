from pathlib import Path
import sys

import numpy as np
import torch

MODELS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(MODELS_DIR))

from evaluate import evaluate_denoiser, to_complex
from load_awgn_data import load_split
from model import CNN


def main():
    # Keep evaluation aligned with the data-generation script's code/data/ output.
    data_root = MODELS_DIR.parent / "data"
    X_test, y_test, snr_values, modulation_labels = load_split(
        "test", data_root=str(data_root)
    )
    checkpoint_path = Path(__file__).resolve().parent / "checkpoints" / "cnn_v1_best.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}. Run train.py first.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    batch_size = 64
    predictions = []
    with torch.no_grad():
        for start in range(0, len(X_test), batch_size):
            batch = torch.from_numpy(X_test[start:start + batch_size]).to(device)
            predictions.append(model(batch).cpu().numpy())

    y_pred = np.concatenate(predictions, axis=0)
    return evaluate_denoiser(
        model_name="CNN",
        y_pred_complex=to_complex(y_pred),
        X_test_complex=to_complex(X_test),
        target_complex=to_complex(y_test),
        snr_values=snr_values,
        modulation_labels=modulation_labels,
    )


if __name__ == "__main__":
    main()
