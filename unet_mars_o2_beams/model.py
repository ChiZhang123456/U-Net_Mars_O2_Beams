"""Neural-network architecture used by U-Net v11.

The model treats an ion energy-time spectrogram as a two-dimensional image.
The input layout is ``[batch, channel, time, energy]`` and the output layout
is ``[batch, class, time, energy]``. Group normalization is used instead of
batch normalization so inference remains stable for a batch size of one.
"""

from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Module):
    """Two 3x3 convolutions, each followed by group normalization and ReLU."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        groups = 4 if out_channels % 4 == 0 else 1
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class StaticUNet(nn.Module):
    """Compact three-level U-Net for pixel-wise STATIC spectral classification.

    Parameters
    ----------
    in_channels
        Number of physical input channels. U-Net v11 uses three channels.
    classes
        Number of output classes. The released checkpoint uses three classes.
    base_channels
        Width of the first encoder block. The released checkpoint uses 12.
    """

    def __init__(
        self, in_channels: int = 3, classes: int = 3, base_channels: int = 12
    ) -> None:
        super().__init__()
        self.encoder_1 = ConvBlock(in_channels, base_channels)
        self.encoder_2 = ConvBlock(base_channels, base_channels * 2)
        self.encoder_3 = ConvBlock(base_channels * 2, base_channels * 4)
        self.bottleneck = ConvBlock(base_channels * 4, base_channels * 8)
        self.pool = nn.MaxPool2d(kernel_size=2)

        self.up_3 = nn.ConvTranspose2d(
            base_channels * 8, base_channels * 4, kernel_size=2, stride=2
        )
        self.decoder_3 = ConvBlock(base_channels * 8, base_channels * 4)
        self.up_2 = nn.ConvTranspose2d(
            base_channels * 4, base_channels * 2, kernel_size=2, stride=2
        )
        self.decoder_2 = ConvBlock(base_channels * 4, base_channels * 2)
        self.up_1 = nn.ConvTranspose2d(
            base_channels * 2, base_channels, kernel_size=2, stride=2
        )
        self.decoder_1 = ConvBlock(base_channels * 2, base_channels)
        self.output = nn.Conv2d(base_channels, classes, kernel_size=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        encoder_1 = self.encoder_1(inputs)
        encoder_2 = self.encoder_2(self.pool(encoder_1))
        encoder_3 = self.encoder_3(self.pool(encoder_2))
        bottleneck = self.bottleneck(self.pool(encoder_3))

        decoder_3 = self.decoder_3(
            torch.cat((self.up_3(bottleneck), encoder_3), dim=1)
        )
        decoder_2 = self.decoder_2(
            torch.cat((self.up_2(decoder_3), encoder_2), dim=1)
        )
        decoder_1 = self.decoder_1(
            torch.cat((self.up_1(decoder_2), encoder_1), dim=1)
        )
        return self.output(decoder_1)
