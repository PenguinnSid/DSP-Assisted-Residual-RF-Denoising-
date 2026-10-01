from pathlib import Path
import sys

import numpy as np
import torch

MODELS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(MODELS_DIR))

from evaluate import evaluate_denoiser, to_complex
from load_awgn_data import load_split
from AutoEncoder.model import Autoencoder


def main():
    data_root = MODELS_DIR.parent / "code" / "data"
    X_test, y_test, snr_values, modulation_labels = load_split("test", data_root=str(data_root))
    checkpoint = Path(__file__).resolve().parent / "checkpoints" / "ae_v2_best.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}. Run train.py first.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Autoencoder().to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(X_test), 64):
            batch = torch.from_numpy(X_test[start:start + 64]).to(device)
            predictions.append(model(batch).cpu().numpy())
    return evaluate_denoiser(
        "Autoencoder", to_complex(np.concatenate(predictions, axis=0)),
        to_complex(X_test), to_complex(y_test), snr_values, modulation_labels,
    )


if __name__ == "__main__":
    main()
