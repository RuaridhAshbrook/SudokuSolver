"""End-to-end check with your trained model: render puzzles with known answers in three
styles (screenshot, dark mode, tilted photo) and measure how many are read perfectly.

    python -m tests.test_pipeline                       # uses models/digit_cnn.pt
    python -m tests.test_pipeline --n 30 --model other.pt
"""
from __future__ import annotations

import argparse

import numpy as np

from sudoku_ocr.recognise import DEFAULT_MODEL, SudokuReader
from sudoku_ocr.synth import find_font_files
from tests.render_puzzle import make_photo, random_puzzle, render_flat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--n", type=int, default=20)
    args = ap.parse_args()

    reader = SudokuReader.from_model_file(args.model)
    fonts = find_font_files()
    rng = np.random.default_rng(123)
    for style in ("screenshot", "dark", "photo"):
        perfect, accs, found = 0, [], 0
        for _ in range(args.n):
            grid = random_puzzle(rng)
            flat = render_flat(grid, fonts[int(rng.integers(len(fonts)))], dark=(style == "dark"), rng=rng)
            img = make_photo(flat, rng) if style == "photo" else flat
            res = reader.read(img)
            found += res.grid_found
            accs.append(float((res.grid == grid).mean()))
            perfect += bool((res.grid == grid).all())
        print(f"{style:10s} outline found {found}/{args.n}  perfect grids {perfect}/{args.n}  "
              f"cell accuracy {np.mean(accs) * 100:.2f}%")


if __name__ == "__main__":
    main()
