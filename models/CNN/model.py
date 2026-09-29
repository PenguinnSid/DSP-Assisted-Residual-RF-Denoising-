import torch
import torch.nn as nn


class CNN(nn.Module):
    """
    Convolutional Autoencoder style model for sequence-to-sequence I/Q denoising.
    Mirrors the architecture of the AutoEncoder model but retains the class name CNN.
    Input/Output shape: (batch, seq_len, 2)
    """

    def __init__(self, input_channels=2, hidden_channels=64):
        super(CNN, self).__init__()
        # Encoder: down‑sample and compress
        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, hidden_channels, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(hidden_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(hidden_channels, hidden_channels * 2, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(hidden_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(hidden_channels * 2, hidden_channels * 4, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(hidden_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # Decoder: up‑sample back to original length
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(hidden_channels * 4, hidden_channels * 2, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm1d(hidden_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose1d(hidden_channels * 2, hidden_channels, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm1d(hidden_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(hidden_channels, input_channels, kernel_size=7, stride=1, padding=3),
        )

    def forward(self, x):
        """Expect input shape (batch, seq_len, 2). Returns same shape after denoising."""
        # Conv1d expects (batch, channels, seq_len)
        x = x.transpose(1, 2)
        latent = self.encoder(x)
        out = self.decoder(latent)
        out = out.transpose(1, 2)
        return out
