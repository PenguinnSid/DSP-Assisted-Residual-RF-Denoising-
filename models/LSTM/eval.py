import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # models/

import torch
import numpy as np

from evaluate import evaluate_denoiser, to_complex
from load_awgn_data import load_split
from LSTM.model import LSTM


X_test, y_test, snr_values, modulation_labels = load_split("test")

X_test_complex = to_complex(X_test)
target_complex = to_complex(y_test)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = LSTM(input_size=2, hidden_size=64, num_layers=2, output_size=2, dropout_rate=0.0)
model.load_state_dict(torch.load("models/LSTM/checkpoints/lstm_v1_best.pt", map_location=device))
model.to(device)
model.eval()

batch_size = 64
y_pred_list = []

with torch.no_grad():
    for i in range(0, len(X_test), batch_size):
        X_batch = torch.tensor(X_test[i:i+batch_size]).to(device)
        pred_batch = model(X_batch)
        y_pred_list.append(pred_batch.cpu().numpy())

y_pred = np.concatenate(y_pred_list, axis=0)
y_pred_complex = to_complex(y_pred)

results = evaluate_denoiser(
    model_name="LSTM",
    y_pred_complex=y_pred_complex,
    X_test_complex=X_test_complex,
    target_complex=target_complex,
    snr_values=snr_values,
    modulation_labels=modulation_labels,
)