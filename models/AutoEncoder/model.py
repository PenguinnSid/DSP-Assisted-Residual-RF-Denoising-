import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5):
        super().__init__()
        padding = kernel_size // 2
        self.layers = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.layers(x)


class Autoencoder(nn.Module):
    """Skip-connected 1D denoising autoencoder. Input/output: (B, T, 2)."""

    def __init__(self, input_channels=2, hidden_channels=32):
        super().__init__()
        c1, c2, c3 = hidden_channels, hidden_channels * 2, hidden_channels * 4
        self.enc1 = ConvBlock(input_channels, c1)
        self.down1 = nn.Conv1d(c1, c2, kernel_size=4, stride=2, padding=1)
        self.enc2 = ConvBlock(c2, c2)
        self.down2 = nn.Conv1d(c2, c3, kernel_size=4, stride=2, padding=1)
        self.bottleneck = ConvBlock(c3, c3)
        self.up2 = nn.ConvTranspose1d(c3, c2, kernel_size=4, stride=2, padding=1)
        self.dec2 = ConvBlock(c2 * 2, c2)
        self.up1 = nn.ConvTranspose1d(c2, c1, kernel_size=4, stride=2, padding=1)
        self.dec1 = ConvBlock(c1 * 2, c1)
        self.output = nn.Conv1d(c1, input_channels, kernel_size=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x):
        # Convert to channels-first for Conv1d and pad odd lengths for exact upsampling.
        signal = x.transpose(1, 2)
        original_length = signal.size(-1)
        if original_length % 4:
            signal = nn.functional.pad(signal, (0, 4 - original_length % 4), mode="replicate")

        skip1 = self.enc1(signal)
        skip2 = self.enc2(self.down1(skip1))
        latent = self.bottleneck(self.down2(skip2))
        decoded2 = self.dec2(torch.cat([self.up2(latent), skip2], dim=1))
        decoded1 = self.dec1(torch.cat([self.up1(decoded2), skip1], dim=1))
        correction = self.output(decoded1)[..., :original_length].transpose(1, 2)
        return x + correction
