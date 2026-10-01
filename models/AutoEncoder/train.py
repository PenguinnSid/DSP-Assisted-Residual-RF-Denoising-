from pathlib import Path
import sys

MODELS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(MODELS_DIR))

from load_awgn_data import load_split
from AutoEncoder.model import Autoencoder
from training_utils import fit_model, make_loaders


def main():
    data_root = MODELS_DIR.parent / "data"
    train_loader, val_loader = make_loaders(load_split, data_root)
    fit_model(
        Autoencoder(), train_loader, val_loader,
        checkpoint_dir=Path(__file__).resolve().parent / "checkpoints",
        checkpoint_prefix="ae_v2", epochs=40, learning_rate=7e-4,
    )


if __name__ == "__main__":
    main()
