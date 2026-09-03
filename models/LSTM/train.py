import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model import LSTM
from load_data import load_split

# gpu 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

""" Hyperparameters"""

input_size = 2 # I/Q channels
hidden_size = 64 # number of neurons/memory cells
num_layers = 2 # number of LSTMS
output_size = 2 # final dimension of prediction
learning_rate = 1e-3
dropout_rate = 0.2
batch_size = 64
epochs = 30


""" data splits """

X_train, y_train = load_split("train")
X_val, y_val = load_split("validation")

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

""" Model Initialization"""

model = LSTM(input_size,hidden_size,num_layers,output_size,dropout_rate = dropout_rate).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss = nn.MSELoss()


# """Test the model on a tiny subset of the training data to ensure it can overfit"""

# X_tiny, y_tiny = X_train[:50], y_train[:50]

# X_tiny_t = torch.tensor(X_tiny).to(device)
# y_tiny_t = torch.tensor(y_tiny).to(device)

# model_test = LSTM(input_size, hidden_size, num_layers, output_size, dropout_rate=0).to(device)
# optimizer_test = optim.Adam(model_test.parameters(), lr=1e-3)
# loss_fn = nn.MSELoss()

# for step in range(500):
#     optimizer_test.zero_grad()
#     pred = model_test(X_tiny_t)
#     l = loss_fn(pred, y_tiny_t)
#     l.backward()

#     torch.nn.utils.clip_grad_norm_(model_test.parameters(), max_norm=1.0)

#     optimizer_test.step()
#     if step % 50 == 0:
#         print(f"step {step}: loss {l.item():.6f}")


        
""" Training Loop """

for epoch in range(epochs):

    # training

    model.train()
    train_loss = 0.0

    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad() # clears old gradients
        y_pred = model(X_batch) # pred
        batch_loss = loss(y_pred, y_batch) # loss
        batch_loss.backward() # backpropagation

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # gradient clipping to prevent exploding gradients

        optimizer.step() # update

        # total sum loss weighted by batch size
        train_loss += batch_loss.item() * X_batch.size(0)

    # average loss per sample over entire dataset
    train_loss /= len(train_loader.dataset)

    # evaluation

    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            batch_loss = loss(y_pred, y_batch)
            val_loss += batch_loss.item() * X_batch.size(0)

    val_loss /= len(val_loader.dataset)

    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

torch.save(model.state_dict(), "models/LSTM/checkpoints/lstm_v1.pt")