#!/usr/bin/env python3
"""
Model definitions for VAE-MMD domain adaptation.

Includes:
    - ResidualBlock3D: 3D residual block with BN and dropout
    - HighQualityVAE: U-Net-style VAE with skip connections
    - Discriminator3D: 3D PatchGAN-style discriminator
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock3D(nn.Module):
    """
    3D residual block with batch normalization, dropout, and skip connection.

    Args:
        channels (int): Number of input and output channels.
        dropout_rate (float): Dropout probability applied between convolutions.
    """

    def __init__(self, channels: int, dropout_rate: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(channels)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(channels)
        self.dropout = nn.Dropout3d(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)


class HighQualityVAE(nn.Module):
    """
    VAE with U-Net-style skip connections for fine detail preservation.

    Encoder: 128^3 -> 8^3 across four strided conv stages.
    Decoder: 8^3 -> 128^3 with skip connections from each encoder stage.

    Args:
        latent_dim (int): Dimensionality of the latent space.
        output_activation (str): One of 'tanh', 'sigmoid', or 'linear'.
    """

    def __init__(self, latent_dim: int = 512, output_activation: str = 'tanh'):
        super().__init__()

        self.latent_dim = latent_dim
        self.output_activation = output_activation

        # Encoder: 128 -> 64 -> 32 -> 16 -> 8
        self.enc1 = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            ResidualBlock3D(32),
        )
        self.enc2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            ResidualBlock3D(64),
        )
        self.enc3 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            ResidualBlock3D(128),
        )
        self.enc4 = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            ResidualBlock3D(256),
        )

        # Latent space projections
        self.fc_mu = nn.Linear(256 * 8 * 8 * 8, latent_dim)
        self.fc_logvar = nn.Linear(256 * 8 * 8 * 8, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, 256 * 8 * 8 * 8)

        # Decoder: 8 -> 16 -> 32 -> 64 -> 128 (with skip concatenation)
        self.dec4 = nn.Sequential(
            nn.ConvTranspose3d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            ResidualBlock3D(128),
        )
        self.dec3 = nn.Sequential(
            nn.ConvTranspose3d(128 + 128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            ResidualBlock3D(64),
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose3d(64 + 64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            ResidualBlock3D(32),
        )
        self.dec1 = nn.Sequential(
            nn.ConvTranspose3d(32 + 32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )

        self.final = nn.Conv3d(16, 1, kernel_size=1)

    def encode(self, x: torch.Tensor):
        """
        Encode input volume to (mu, logvar) and return intermediate skip tensors.

        Returns:
            mu (Tensor): Latent mean, shape [B, latent_dim].
            logvar (Tensor): Latent log-variance, shape [B, latent_dim].
            skips (list[Tensor]): Skip tensors [skip1, skip2, skip3] from enc1/2/3.
        """
        skip1 = self.enc1(x)
        skip2 = self.enc2(skip1)
        skip3 = self.enc3(skip2)
        h = self.enc4(skip3)

        h = h.view(h.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        return mu, logvar, [skip1, skip2, skip3]

    def decode(self, z: torch.Tensor, skips: list) -> torch.Tensor:
        """
        Decode latent vector to reconstructed volume using skip connections.

        Args:
            z (Tensor): Latent vector, shape [B, latent_dim].
            skips (list[Tensor]): Skip tensors from encode().

        Returns:
            Reconstructed volume, shape [B, 1, 128, 128, 128].
        """
        h = self.fc_decode(z).view(z.size(0), 256, 8, 8, 8)

        h = self.dec4(h)
        h = torch.cat([h, skips[2]], dim=1)
        h = self.dec3(h)
        h = torch.cat([h, skips[1]], dim=1)
        h = self.dec2(h)
        h = torch.cat([h, skips[0]], dim=1)
        h = self.dec1(h)

        out = self.final(h)

        if self.output_activation == 'tanh':
            return torch.tanh(out)
        elif self.output_activation == 'sigmoid':
            return torch.sigmoid(out)
        else:
            return out

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick: z = mu + eps * std, eps ~ N(0, I).
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor):
        """
        Full forward pass.

        Returns:
            recon (Tensor): Reconstructed volume.
            mu (Tensor): Latent mean.
            logvar (Tensor): Latent log-variance.
            z (Tensor): Sampled latent vector.
        """
        mu, logvar, skips = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, skips)
        return recon, mu, logvar, z


class Discriminator3D(nn.Module):
    """
    3D convolutional discriminator for adversarial training.

    Outputs an unnormalized score map (LSGAN-style).
    """

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(256, 1, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
