from pathlib import Path
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

MODELS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(MODELS_DIR))

from load_awgn_data import load_split
from CNN.model import CNN
from training_utils import fit_model, make_loaders


def main():
    data_root = MODELS_DIR.parent / "data"
    train_loader, val_loader = make_loaders(load_split, data_root)
    model = CNN()
    fit_model(
        model, train_loader, val_loader,
        checkpoint_dir=Path(__file__).resolve().parent / "checkpoints",
        checkpoint_prefix="cnn_v2", epochs=40, learning_rate=7e-4,
    )


if __name__ == "__main__":
    main()
