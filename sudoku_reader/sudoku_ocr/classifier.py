"""Loads a trained DigitNet checkpoint and turns cells into class probabilities."""
from __future__ import annotations

import numpy as np


class TorchClassifier:
    """Callable: (N, 32, 32) uint8 cells -> (N, 10) probabilities."""

    def __init__(self, model_path, device: str | None = None):
        import torch

        from .model import DigitNet

        self.torch = torch
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        ckpt = torch.load(str(model_path), map_location=self.device)
        self.model = DigitNet().to(self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()

    def __call__(self, cells: np.ndarray) -> np.ndarray:
        from .model import to_tensor

        with self.torch.no_grad():
            logits = self.model(to_tensor(cells).to(self.device))
            return self.torch.softmax(logits, dim=1).cpu().numpy()
