"""Validity checking, misread repair and CSV I/O for 9x9 grids (0 = blank)."""
from __future__ import annotations

import csv

import numpy as np


def _build_units():
    units = []
    for i in range(9):
        units.append([(i, c) for c in range(9)])
        units.append([(r, i) for r in range(9)])
    for br in range(0, 9, 3):
        for bc in range(0, 9, 3):
            units.append([(br + dr, bc + dc) for dr in range(3) for dc in range(3)])
    return units


_UNITS = _build_units()


def find_conflicts(grid) -> set:
    """Return the set of (row, col) cells involved in a duplicate in any row, column or box."""
    g = np.asarray(grid).reshape(9, 9)
    bad = set()
    for unit in _UNITS:
        seen = {}
        for r, c in unit:
            v = int(g[r, c])
            if v:
                seen.setdefault(v, []).append((r, c))
        for cells in seen.values():
            if len(cells) > 1:
                bad.update(cells)
    return bad


def repair_conflicts(probs, min_prob: float = 0.02, max_moves: int = 10):
    """Use sudoku rules to fix likely misreads.

    probs: (81, 10) class probabilities (class 0 = blank). Starting from the arg-max
    reading, repeatedly make the single change (to a runner-up class with at least
    min_prob) that removes the most duplicates. Stops when the grid is conflict-free
    or no change helps.

    Returns (grid (9, 9) int array, list of (row, col) cells that were changed).
    """
    probs = np.asarray(probs, dtype=np.float64).reshape(81, 10)
    pred = probs.argmax(1)
    changed = []
    for _ in range(max_moves):
        bad = find_conflicts(pred.reshape(9, 9))
        if not bad:
            break
        base = len(bad)
        best = None
        for r, c in bad:
            i = r * 9 + c
            for k in range(10):
                if k == pred[i] or probs[i, k] < min_prob:
                    continue
                trial = pred.copy()
                trial[i] = k
                n = len(find_conflicts(trial.reshape(9, 9)))
                if n < base:
                    key = (n, -probs[i, k])
                    if best is None or key < best[0]:
                        best = (key, i, k)
        if best is None:
            break
        _, i, k = best
        pred[i] = k
        changed.append((i // 9, i % 9))
    return pred.reshape(9, 9), changed


def save_csv(grid, path, blank: str = "0") -> None:
    """Write a 9x9 CSV. Blanks are written as `blank` ("0" by default; use "" for empty cells)."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        for row in np.asarray(grid).reshape(9, 9):
            w.writerow([int(v) if v else blank for v in row])


def load_csv(path) -> np.ndarray:
    rows = []
    with open(path, newline="") as f:
        for row in csv.reader(f):
            if not row:
                continue
            rows.append([int(x) if x.strip().isdigit() else 0 for x in row])
    if len(rows) != 9 or any(len(r) != 9 for r in rows):
        raise ValueError("CSV must contain 9 rows of 9 values")
    return np.array(rows, dtype=int)
