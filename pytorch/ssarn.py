"""SSARN architecture for the supervised hyperspectral baseline."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def spectral_width(bands: int) -> int:
    width = bands
    for _ in range(3):
        width = (width - 7) // 2 + 1
    if width < 1:
        raise ValueError("Too few bands for three spectral convolutions")
    return width


class SpectralStem(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv3d(in_channels, 32, (7, 1, 1), stride=(2, 1, 1))
        self.bn = nn.BatchNorm3d(32, eps=1e-5, momentum=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.leaky_relu(self.bn(self.conv(x)), negative_slope=0.01)


class SpectralResidualUnit(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv3d(32, 32, (7, 1, 1), padding=(3, 0, 0))
        self.bn = nn.BatchNorm3d(32, eps=1e-5, momentum=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(F.leaky_relu(self.conv(x), negative_slope=0.01))


class SpatialAttention(nn.Module):
    """Full-width Q/K/V; softmax on the second spatial axis."""

    def __init__(self) -> None:
        super().__init__()
        self.query = nn.Conv2d(32, 32, 1)
        self.key = nn.Conv2d(32, 32, 1)
        self.value = nn.Conv2d(32, 32, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, channels, height, width = x.shape
        q = self.query(x).reshape(n, channels, height * width)
        k = self.key(x).reshape(n, channels, height * width)
        v = self.value(x).reshape(n, channels, height * width)
        weights = torch.softmax(torch.bmm(q.transpose(1, 2), k), dim=2)
        return torch.bmm(v, weights).reshape(n, channels, height, width)


class SpatialResidualUnit(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(32, 32, 3, padding=1)
        self.bn = nn.BatchNorm2d(32, eps=1e-5, momentum=0.1)
        self.attention = SpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feature = self.bn(F.leaky_relu(self.conv(x), negative_slope=0.01))
        return x + feature + self.attention(feature)


class SSARN(nn.Module):
    """Input N×1×bands×9×9; output N×classes raw logits."""

    def __init__(self, bands: int, classes: int) -> None:
        super().__init__()
        if classes < 2:
            raise ValueError("SSARN needs at least two classes")
        width = spectral_width(bands)
        self.stem = nn.Sequential(
            SpectralStem(1), SpectralStem(32), SpectralStem(32)
        )
        self.spectral_first = SpectralResidualUnit()
        self.spectral_second = SpectralResidualUnit()
        self.collapse = nn.Conv3d(32, 32, (width, 3, 3))
        self.collapse_bn = nn.BatchNorm3d(32, eps=1e-5, momentum=0.1)
        self.spatial_residual = SpatialResidualUnit()
        self.classifier = nn.Linear(32, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        # Spectral residual: x + F2(x + F1(x)).
        residual = x + self.spectral_first(x)
        x = x + self.spectral_second(residual)
        x = F.leaky_relu(x, negative_slope=0.01)
        x = self.collapse(x)
        x = F.leaky_relu(x, negative_slope=0.01)
        x = self.collapse_bn(x).squeeze(2)
        x = self.spatial_residual(x)
        x = F.leaky_relu(x, negative_slope=0.01)
        return self.classifier(x.mean(dim=(-2, -1)))
