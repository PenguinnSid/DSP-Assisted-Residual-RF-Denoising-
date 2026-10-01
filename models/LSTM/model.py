import torch
import torch.nn as nn


class LSTM(nn.Module):
    """Bidirectional sequence denoiser. Input/output: (B, T, 2)."""

    def __init__(self, input_size=2, hidden_size=64, num_layers=2,
                 output_size=2, dropout_rate=0.15):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_size)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_rate if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size * 2),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, output_size),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, x):
        recurrent, _ = self.lstm(self.input_norm(x))
        return x + self.head(recurrent)
