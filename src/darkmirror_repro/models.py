from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TCNBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        padding = ((kernel_size - 1) * dilation) // 2
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.net(x) + self.downsample(x))


class DirectionTCN(nn.Module):
    def __init__(self, channels=(32, 64, 128), kernel_size=5, dropout=0.1, fc_hidden=64):
        super().__init__()
        layers = []
        in_channels = 1
        for index, out_channels in enumerate(channels):
            layers.append(TCNBlock(in_channels, out_channels, kernel_size, dilation=2**index, dropout=dropout))
            in_channels = out_channels
        self.features = nn.Sequential(*layers, nn.AdaptiveAvgPool1d(1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels[-1], fc_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x)).squeeze(1)


class ResNetBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernels=(8, 5, 3), dropout: float = 0.0):
        super().__init__()
        k1, k2, k3 = kernels
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, k1, padding=k1 // 2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, k2, padding=k2 // 2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, k3, padding=k3 // 2),
            nn.BatchNorm1d(out_channels),
        )
        self.shortcut = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Sequential(nn.Conv1d(in_channels, out_channels, 1), nn.BatchNorm1d(out_channels))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        if y.shape[-1] != x.shape[-1]:
            y = y[..., : x.shape[-1]]
        return F.relu(y + self.shortcut(x))


class DirectionResNet1D(nn.Module):
    def __init__(self, channels=(64, 128, 128), kernels=(8, 5, 3), dropout=0.0):
        super().__init__()
        blocks = []
        in_channels = 1
        for out_channels in channels:
            blocks.append(ResNetBlock1D(in_channels, out_channels, kernels=kernels, dropout=dropout))
            in_channels = out_channels
        self.features = nn.Sequential(*blocks, nn.AdaptiveAvgPool1d(1))
        self.classifier = nn.Linear(channels[-1], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x).flatten(1)).squeeze(1)


def build_model(name: str) -> tuple[nn.Module, dict]:
    if name == "tcn":
        return DirectionTCN(), {"batch_size": 64, "lr": 0.001, "weight_decay": 0.0001}
    if name == "resnet1d":
        return DirectionResNet1D(), {"batch_size": 128, "lr": 0.001, "weight_decay": 0.0001}
    raise ValueError(f"Unsupported reference model: {name}")

