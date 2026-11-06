"""The cell classifier: a small CNN. Classes: 0 = blank, 1-9 = digits."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

NUM_CLASSES = 10


def _block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class DigitNet(nn.Module):
    """Input (N, 1, 32, 32) floats in [0, 1], ink bright on dark."""

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            _block(1, 32), _block(32, 32), nn.MaxPool2d(2),     # 16x16
            _block(32, 64), _block(64, 64), nn.MaxPool2d(2),    # 8x8
            _block(64, 128), nn.MaxPool2d(2),                   # 4x4
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.head(self.features(x))


def to_tensor(cells_u8: np.ndarray) -> torch.Tensor:
    """(N, 32, 32) uint8 -> (N, 1, 32, 32) float32 in [0, 1]."""
    x = torch.from_numpy(np.ascontiguousarray(cells_u8)).float().div_(255.0)
    return x.unsqueeze(1)
