from __future__ import annotations

import math

import torch

import torch.nn as nn

import torch.nn.functional as F

SEQ_LEN = 506

class TCNBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.net = nn.Sequential(nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation), nn.BatchNorm1d(out_channels), nn.ReLU(), nn.Dropout(dropout), nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation), nn.BatchNorm1d(out_channels), nn.ReLU(), nn.Dropout(dropout))
        self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.net(x) + self.downsample(x))

class DirectionTCN(nn.Module):

    def __init__(self, channels=(32, 64, 128), kernel_size=5, dropout=0.1, fc_hidden=64):
        super().__init__()
        layers = []
        in_channels = 1
        for index, out_channels in enumerate(channels):
            layers.append(TCNBlock(in_channels, out_channels, kernel_size, dilation=2 ** index, dropout=dropout))
            in_channels = out_channels
        self.features = nn.Sequential(*layers, nn.AdaptiveAvgPool1d(1))
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(channels[-1], fc_hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(fc_hidden, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x)).squeeze(1)

class ResNetBlock1D(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernels=(8, 5, 3), dropout: float=0.0):
        super().__init__()
        k1, k2, k3 = kernels
        self.conv = nn.Sequential(nn.Conv1d(in_channels, out_channels, k1, padding=k1 // 2), nn.BatchNorm1d(out_channels), nn.ReLU(), nn.Dropout(dropout), nn.Conv1d(out_channels, out_channels, k2, padding=k2 // 2), nn.BatchNorm1d(out_channels), nn.ReLU(), nn.Dropout(dropout), nn.Conv1d(out_channels, out_channels, k3, padding=k3 // 2), nn.BatchNorm1d(out_channels))
        self.shortcut = nn.Identity() if in_channels == out_channels else nn.Sequential(nn.Conv1d(in_channels, out_channels, 1), nn.BatchNorm1d(out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        if y.shape[-1] != x.shape[-1]:
            y = y[..., :x.shape[-1]]
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

class DirectionOnlyCNN(nn.Module):

    def __init__(self, conv_channels=(32, 64, 128), kernel_size=5, fc_hidden=64, dropout=0.1):
        super().__init__()
        c1, c2, c3 = conv_channels
        pad = kernel_size // 2
        self.features = nn.Sequential(nn.Conv1d(1, c1, kernel_size=kernel_size, padding=pad), nn.BatchNorm1d(c1), nn.ReLU(), nn.MaxPool1d(2), nn.Conv1d(c1, c2, kernel_size=kernel_size, padding=pad), nn.BatchNorm1d(c2), nn.ReLU(), nn.MaxPool1d(2), nn.Conv1d(c2, c3, kernel_size=3, padding=1), nn.BatchNorm1d(c3), nn.ReLU(), nn.AdaptiveMaxPool1d(1))
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(c3, fc_hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(fc_hidden, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x)).squeeze(1)

class DirectionRNN(nn.Module):

    def __init__(self, cell_type: str, hidden_size: int=64, num_layers: int=2, dropout: float=0.1):
        super().__init__()
        rnn_cls = {'gru': nn.GRU, 'lstm': nn.LSTM}[cell_type]
        self.rnn = rnn_cls(input_size=1, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0, bidirectional=False)
        self.classifier = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_size, 1))
        self.cell_type = cell_type

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, hidden = self.rnn(x)
        if self.cell_type == 'lstm':
            hidden = hidden[0]
        last_hidden = hidden[-1]
        return self.classifier(last_hidden).squeeze(1)

class DirectionBiGRU(nn.Module):

    def __init__(self, hidden_size: int=64, num_layers: int=2, dropout: float=0.1):
        super().__init__()
        self.gru = nn.GRU(input_size=1, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True, dropout=dropout if num_layers > 1 else 0.0)
        self.classifier = nn.Sequential(nn.Linear(hidden_size * 2, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, h_n = self.gru(x)
        return self.classifier(torch.cat([h_n[-2], h_n[-1]], dim=1)).squeeze(1)

class FCN1D(nn.Module):

    def __init__(self, channels=(64, 128, 128), kernels=(8, 5, 3), dropout: float=0.1):
        super().__init__()
        layers = []
        in_channels = 1
        for out_channels, kernel_size in zip(channels, kernels):
            layers.extend([nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2), nn.BatchNorm1d(out_channels), nn.ReLU(), nn.Dropout(dropout)])
            in_channels = out_channels
        self.features = nn.Sequential(*layers, nn.AdaptiveAvgPool1d(1))
        self.classifier = nn.Linear(channels[-1], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x).flatten(1)).squeeze(1)

class InceptionBlock1D(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, bottleneck_channels: int=32):
        super().__init__()
        hidden = bottleneck_channels if in_channels > 1 else in_channels
        self.bottleneck = nn.Conv1d(in_channels, hidden, 1) if in_channels > 1 else nn.Identity()
        branch_channels = out_channels // 4
        self.branch9 = nn.Conv1d(hidden, branch_channels, 9, padding=4)
        self.branch19 = nn.Conv1d(hidden, branch_channels, 19, padding=9)
        self.branch39 = nn.Conv1d(hidden, branch_channels, 39, padding=19)
        self.branch_pool = nn.Sequential(nn.MaxPool1d(3, stride=1, padding=1), nn.Conv1d(in_channels, branch_channels, 1))
        self.bn = nn.BatchNorm1d(branch_channels * 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.bottleneck(x)
        y = torch.cat([self.branch9(z), self.branch19(z), self.branch39(z), self.branch_pool(x)], dim=1)
        return F.relu(self.bn(y))

class InceptionTimeLite(nn.Module):

    def __init__(self, channels: int=128, depth: int=3, dropout: float=0.1):
        super().__init__()
        blocks = []
        in_channels = 1
        for _ in range(depth):
            blocks.append(InceptionBlock1D(in_channels, channels))
            blocks.append(nn.Dropout(dropout))
            in_channels = channels
        self.features = nn.Sequential(*blocks, nn.AdaptiveAvgPool1d(1))
        self.classifier = nn.Linear(channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x).flatten(1)).squeeze(1)

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.shape[1], :]

class TransformerEncoder1D(nn.Module):

    def __init__(self, d_model: int=64, nhead: int=4, num_layers: int=2, dropout: float=0.1, dim_feedforward: int=128):
        super().__init__()
        self.proj = nn.Linear(1, d_model)
        self.pos = PositionalEncoding(d_model=d_model, max_len=SEQ_LEN)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True, activation='gelu')
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.classifier = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = x.transpose(1, 2)
        z = self.pos(self.proj(z))
        z = self.encoder(z)
        return self.classifier(z.mean(dim=1)).squeeze(1)
