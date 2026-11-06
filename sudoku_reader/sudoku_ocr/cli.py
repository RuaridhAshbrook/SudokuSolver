"""Command line: image(s) -> CSV.

    python -m sudoku_ocr.cli puzzle.jpg                # writes puzzle.csv next to the image
    python -m sudoku_ocr.cli a.png b.jpg --model models/digit_cnn.pt
    python -m sudoku_ocr.cli puzzle.jpg -o out.csv --debug-dir debug/
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from .preprocess import load_image
from .recognise import DEFAULT_MODEL, SudokuReader
from .sudoku_utils import find_conflicts, save_csv

LOW_CONF = 0.85


def print_grid(res):
    for r in range(9):
        if r and r % 3 == 0:
            print("------+-------+------")
        parts = []
        for c in range(9):
            v = res.grid[r, c]
            s = str(v) if v else "."
            if res.confidence[r, c] < LOW_CONF:
                s = s + "?" if v else ","
            parts.append(s)
            if c % 3 == 2 and c < 8:
                parts.append("|")
        print(" ".join(parts))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="+")
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("-o", "--out", help="CSV path (single image only); default is <image>.csv")
    ap.add_argument("--blank", default="0", help='text for empty cells, e.g. "" for truly empty (default 0)')
    ap.add_argument("--no-repair", action="store_true", help="skip sudoku-rule correction of misreads")
    ap.add_argument("--debug-dir", help="save the flattened grid for each image here")
    args = ap.parse_args()
    if args.out and len(args.images) > 1:
        ap.error("-o only works with a single image")

    reader = SudokuReader.from_model_file(args.model)
    for path in args.images:
        res = reader.read(load_image(path), repair=not args.no_repair)
        out = Path(args.out) if args.out else Path(path).with_suffix(".csv")
        save_csv(res.grid, out, blank=args.blank)
        print(f"\n{path} -> {out}")
        if not res.grid_found:
            print("  warning: grid outline not found, used the whole image")
        print_grid(res)
        low = int((res.confidence < LOW_CONF).sum())
        if low:
            print(f"  {low} low-confidence cells (marked ? for digits, , for blanks) - worth checking")
        if res.changed:
            print(f"  sudoku-rule repair changed {len(res.changed)} cell(s): {res.changed}")
        if find_conflicts(res.grid):
            print("  warning: grid still has duplicates, so some cell is misread")
        if args.debug_dir:
            Path(args.debug_dir).mkdir(parents=True, exist_ok=True)
            dbg = Path(args.debug_dir) / (Path(path).stem + "_grid.png")
            cv2.imencode(".png", res.ink)[1].tofile(str(dbg))


if __name__ == "__main__":
    main()
