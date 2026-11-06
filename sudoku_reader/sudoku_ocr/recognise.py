"""Full image -> grid pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .preprocess import find_grid_corners, grid_to_ink, split_cells, warp_grid
from .sudoku_utils import find_conflicts, repair_conflicts

DEFAULT_MODEL = Path(__file__).resolve().parent.parent / "models" / "digit_cnn.pt"


@dataclass
class ReadResult:
    grid: np.ndarray          # (9, 9) int, 0 = blank (after optional repair)
    confidence: np.ndarray    # (9, 9) float, probability of the chosen class
    probs: np.ndarray         # (81, 10) raw classifier output
    cells: np.ndarray         # (81, 32, 32) uint8 exactly as the classifier saw them
    ink: np.ndarray           # (450, 450) uint8 flattened grid (for debugging)
    corners: np.ndarray       # (4, 2) grid corners in the source image
    grid_found: bool          # False if outline detection failed and the whole image was used
    rotation: int = 0         # quarter-turns applied to the warped grid
    changed: list = field(default_factory=list)   # (row, col) cells altered by sudoku-rule repair


class SudokuReader:
    def __init__(self, classifier):
        """classifier: callable mapping (N, 32, 32) uint8 -> (N, 10) probabilities."""
        self.classifier = classifier

    @classmethod
    def from_model_file(cls, path=DEFAULT_MODEL, device=None) -> "SudokuReader":
        from .classifier import TorchClassifier

        return cls(TorchClassifier(path, device))

    def read(self, img: np.ndarray, corners=None, repair: bool = True, auto_rotate: bool = True) -> ReadResult:
        found = True
        if corners is None:
            corners, found = find_grid_corners(img)
        ink = grid_to_ink(warp_grid(img, corners))

        # Orientation: pick the quarter-turn whose reading is most confident AND has the fewest
        # sudoku-rule violations (a sideways grid reads as nonsense, which duplicates a lot).
        best = None
        for k in (range(4) if auto_rotate else (0,)):
            cells = split_cells(np.rot90(ink, k))
            probs = self.classifier(cells)
            n_bad = len(find_conflicts(probs.argmax(1)))
            logconf = float(np.log(probs.max(1) + 1e-6).sum())
            score = logconf - 2.0 * n_bad
            if best is None or score > best[0]:
                best = (score, k, cells, probs)
            if n_bad == 0 and logconf / 81 > -0.02:   # clean and confident: skip the other turns
                break
        _, k, cells, probs = best

        if repair:
            grid, changed = repair_conflicts(probs)
        else:
            grid, changed = probs.argmax(1).reshape(9, 9), []
        conf = probs[np.arange(81), grid.ravel()].reshape(9, 9)
        return ReadResult(
            grid=grid, confidence=conf, probs=probs, cells=cells,
            ink=np.ascontiguousarray(np.rot90(ink, k)), corners=corners,
            grid_found=found, rotation=k, changed=changed,
        )
