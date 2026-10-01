import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """Full-resolution temporal convolutions with a residual feature path."""

    def __init__(self, channels, kernel_size=7, dilation=1, dropout=0.05):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm1d(channels),
        )
        self.activation = nn.GELU()

    def forward(self, x):
        return self.activation(x + self.net(x))


class CNN(nn.Module):
    """Residual 1D CNN for I/Q denoising. Input/output: (B, T, 2)."""

    def __init__(self, input_channels=2, hidden_channels=48, num_blocks=5,
                 kernel_size=7, dropout_rate=0.05):
        super().__init__()
        self.input_projection = nn.Conv1d(input_channels, hidden_channels, kernel_size=1)
        dilations = [1, 2, 4, 2, 1][:num_blocks]
        self.blocks = nn.Sequential(*[
            ResidualBlock(hidden_channels, kernel_size, dilation, dropout_rate)
            for dilation in dilations
        ])
        self.output_projection = nn.Conv1d(hidden_channels, input_channels, kernel_size=1)
        # Start near the identity mapping, then learn the noise correction.
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, x):
        features = self.input_projection(x.transpose(1, 2))
        correction = self.output_projection(self.blocks(features)).transpose(1, 2)
        return x + correction
