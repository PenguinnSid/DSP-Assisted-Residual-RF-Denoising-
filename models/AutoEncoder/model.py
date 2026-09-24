import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    """
    1D Convolutional Denoising Autoencoder for RF I/Q Signal Denoising.

    Input shape:  (N, seq_len, 2)  -- [Real, Imag] complex channels
    Output shape: (N, seq_len, 2)  -- Denoised [Real, Imag] complex channels
    """
    def __init__(self, input_channels=2, hidden_channels=64):
        super(Autoencoder, self).__init__()

        # Encoder: Downsamples along time dimension and compresses features
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

        # Decoder: Upsamples back to original sequence length
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(hidden_channels * 4, hidden_channels * 2, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm1d(hidden_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose1d(hidden_channels * 2, hidden_channels, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm1d(hidden_channels),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(hidden_channels, input_channels, kernel_size=7, stride=1, padding=3)
        )

    def forward(self, x):
        """
        x shape: (batch_size, seq_len, 2)
        returns: (batch_size, seq_len, 2)
        """
        # PyTorch Conv1d expects (batch_size, channels, seq_len)
        x = x.transpose(1, 2)

        latent = self.encoder(x)
        out = self.decoder(latent)

        # Transpose back to (batch_size, seq_len, 2)
        out = out.transpose(1, 2)
        return out

