import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
print("Script started")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # -> models/
print("Imports done")

from model import Autoencoder
from load_awgn_data import load_split
print("Local imports done")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

""" Hyperparameters """
batch_size = 64
learning_rate = 1e-3
epochs = 30

""" Data Splits """
X_train, y_train, _, _ = load_split("train")
X_val, y_val, _, _ = load_split("validation")

train_loader = DataLoader(
    TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
    batch_size=batch_size,
    shuffle=True,
)

val_loader = DataLoader(
    TensorDataset(torch.tensor(X_val), torch.tensor(y_val)),
    batch_size=batch_size,
    shuffle=False,
)

""" Model Initialization """
model = Autoencoder().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.MSELoss()

checkpoint_dir = Path(__file__).resolve().parent / "checkpoints"
checkpoint_dir.mkdir(parents=True, exist_ok=True)

""" Training Loop """
best_val_loss = float("inf")

for epoch in range(epochs):
    model.train()
    train_loss = 0.0

    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        y_pred = model(X_batch)
        batch_loss = loss_fn(y_pred, y_batch)
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
            y_pred = model(X_batch)
            batch_loss = loss_fn(y_pred, y_batch)
            val_loss += batch_loss.item() * X_batch.size(0)

    val_loss /= len(val_loader.dataset)

    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), checkpoint_dir / "ae_v1_best.pt")

torch.save(model.state_dict(), checkpoint_dir / "ae_v1_full.pt")
print(f"Training complete. Best Val Loss: {best_val_loss:.6f}")