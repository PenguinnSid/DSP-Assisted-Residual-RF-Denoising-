from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def make_loaders(load_split, data_root, batch_size=64):
    """Load paired noisy/target I/Q arrays and construct CPU-backed loaders."""
    X_train, y_train, _, _ = load_split("train", data_root=str(data_root))
    X_val, y_val, _, _ = load_split("validation", data_root=str(data_root))
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=batch_size, shuffle=True,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)),
        batch_size=batch_size, shuffle=False,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader


def fit_model(model, train_loader, val_loader, checkpoint_dir, checkpoint_prefix,
              epochs=40, learning_rate=7e-4, patience=7):
    """Shared MSE training loop with scheduler, early stopping, and best save."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-5
    )
    loss_fn = nn.MSELoss()

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_path = checkpoint_dir.parent / "training_log.csv"
    best_path = checkpoint_dir / f"{checkpoint_prefix}_best.pt"
    full_path = checkpoint_dir / f"{checkpoint_prefix}_full.pt"
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    with log_path.open("w", encoding="utf-8") as log:
        log.write("epoch,train_loss,val_loss,learning_rate\n")
        for epoch in range(epochs):
            model.train()
            train_sum = 0.0
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(device, non_blocking=True)
                y_batch = y_batch.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                loss = loss_fn(model(X_batch), y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_sum += loss.item() * X_batch.size(0)
            train_loss = train_sum / len(train_loader.dataset)

            model.eval()
            val_sum = 0.0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(device, non_blocking=True)
                    y_batch = y_batch.to(device, non_blocking=True)
                    val_sum += loss_fn(model(X_batch), y_batch).item() * X_batch.size(0)
            val_loss = val_sum / len(val_loader.dataset)
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch + 1}/{epochs}, Train MSE: {train_loss:.6f}, "
                f"Val MSE: {val_loss:.6f}, LR: {current_lr:.2e}"
            )
            log.write(f"{epoch + 1},{train_loss:.8f},{val_loss:.8f},{current_lr:.8g}\n")
            log.flush()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                torch.save(model.state_dict(), best_path)
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    print(f"Early stopping after {epoch + 1} epochs.")
                    break

    torch.save(model.state_dict(), full_path)
    print(f"Training complete. Best validation MSE: {best_val_loss:.6f}")
    print(f"Best checkpoint: {best_path}")
