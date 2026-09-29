from pathlib import Path
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

MODELS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(MODELS_DIR))

from load_awgn_data import load_split
from CNN.model import CNN


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    batch_size = 64
    learning_rate = 1e-3
    epochs = 30

    # The project stores generated splits under code/data/ (see code/main.py).
    data_root = MODELS_DIR.parent / "data"
    X_train, y_train, _, _ = load_split("train", data_root=str(data_root))
    X_val, y_val, _, _ = load_split("validation", data_root=str(data_root))

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=batch_size, shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)),
        batch_size=batch_size, shuffle=False,
    )

    model = CNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()
    checkpoint_dir = Path(__file__).resolve().parent / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(X_batch)
            batch_loss = loss_fn(prediction, y_batch)
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += batch_loss.item() * X_batch.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                val_loss += loss_fn(model(X_batch), y_batch).item() * X_batch.size(0)
        val_loss /= len(val_loader.dataset)
        log_path = Path(__file__).resolve().parent / "training_log.csv"
        if not log_path.is_file():
            with open(log_path, "w") as f:
                f.write("epoch,train_loss,val_loss\n")
        with open(log_path, "a") as f:
            f.write(f"{epoch + 1},{train_loss:.6f},{val_loss:.6f}\n")
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_dir / "cnn_v1_best.pt")

    torch.save(model.state_dict(), checkpoint_dir / "cnn_v1_full.pt")
    print(f"Training complete. Best Val Loss: {best_val_loss:.6f}")


if __name__ == "__main__":
    main()
