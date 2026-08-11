"""A small CNN for MNIST — the quickstart's model."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    """Small CNN: 2 conv layers + 2 fully-connected layers (28x28 -> 10)."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 5, padding=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
