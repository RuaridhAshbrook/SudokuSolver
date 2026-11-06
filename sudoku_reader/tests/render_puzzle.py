"""Render synthetic sudoku images with known ground truth, to test the pipeline.

    python -m tests.render_puzzle out_dir/        # writes a few images + .csv ground truth
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def random_puzzle(rng, blanks=None):
    """A valid sudoku solution with random cells blanked (0). Returns (9, 9) int array."""
    perm = rng.permutation(9) + 1
    base = np.array([[(r * 3 + r // 3 + c) % 9 for c in range(9)] for r in range(9)])
    rows = np.concatenate([rng.permutation(3) + 3 * b for b in rng.permutation(3)])
    cols = np.concatenate([rng.permutation(3) + 3 * b for b in rng.permutation(3)])
    grid = perm[base[rows][:, cols]]
    n = int(rng.integers(40, 56)) if blanks is None else blanks
    flat = grid.ravel().copy()
    flat[rng.choice(81, n, replace=False)] = 0
    return flat.reshape(9, 9)


def render_flat(grid, font_path, size=720, margin=60, dark=False, rng=None):
    """Clean 'screenshot'-style puzzle. Returns a BGR uint8 image."""
    rng = rng or np.random.default_rng(0)
    cell = size // 9
    total = cell * 9 + 2 * margin
    bg, fg = ((30, 33, 40), (225, 228, 235)) if dark else ((255, 255, 255), (20, 20, 20))
    img = Image.new("RGB", (total, total), bg)
    d = ImageDraw.Draw(img)
    font = ImageFont.truetype(font_path, int(cell * rng.uniform(0.55, 0.72)))
    for i in range(10):
        w = 5 if i % 3 == 0 else 2
        p = margin + i * cell
        d.line([(margin, p), (margin + 9 * cell, p)], fill=fg, width=w)
        d.line([(p, margin), (p, margin + 9 * cell)], fill=fg, width=w)
    for r in range(9):
        for c in range(9):
            v = int(grid[r, c])
            if v:
                d.text((margin + c * cell + cell / 2, margin + r * cell + cell / 2), str(v),
                       font=font, fill=fg, anchor="mm")
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def make_photo(flat, rng, out=(1000, 1400)):
    """Put a flat puzzle on a cluttered background with perspective, shading, blur, noise, JPEG."""
    H, W = out
    bg = np.full((H, W, 3), rng.integers(60, 140, 3), np.uint8)
    noise = rng.normal(0, 12, (H, W, 1))
    bg = np.clip(bg.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    h, w = flat.shape[:2]
    side = min(H, W) * rng.uniform(0.62, 0.8)
    cx, cy = W / 2 + rng.uniform(-60, 60), H / 2 + rng.uniform(-60, 60)
    ang = np.deg2rad(rng.uniform(-12, 12))
    base = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * side / 2
    rot = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    quad = base @ rot.T + [cx, cy] + rng.uniform(-side * 0.07, side * 0.07, (4, 2))
    src = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32)
    m = cv2.getPerspectiveTransform(src, quad.astype(np.float32))
    warped = cv2.warpPerspective(flat, m, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)
    mask = cv2.warpPerspective(np.full((h, w), 255, np.uint8), m, (W, H))
    comp = np.where(mask[..., None] > 0, warped, bg).astype(np.float32)
    # lighting gradient
    gx, gy = np.meshgrid(np.linspace(0, 1, W), np.linspace(0, 1, H))
    a = rng.uniform(0, 2 * np.pi)
    shade = 1.0 - rng.uniform(0.1, 0.35) * (np.cos(a) * gx + np.sin(a) * gy + 1) / 2
    comp = comp * shade[..., None]
    comp = cv2.GaussianBlur(comp, (0, 0), rng.uniform(0.6, 1.6))
    comp = comp + rng.normal(0, rng.uniform(2, 8), comp.shape)
    comp = np.clip(comp, 0, 255).astype(np.uint8)
    return cv2.imdecode(cv2.imencode(".jpg", comp, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(50, 90))])[1],
                        cv2.IMREAD_COLOR)


def main():
    from sudoku_ocr.synth import find_font_files

    out = Path(sys.argv[1] if len(sys.argv) > 1 else "test_images")
    out.mkdir(parents=True, exist_ok=True)
    fonts = find_font_files()
    rng = np.random.default_rng(1)
    from sudoku_ocr.sudoku_utils import save_csv

    for i, style in enumerate(["screenshot", "dark", "photo", "photo"]):
        grid = random_puzzle(rng)
        flat = render_flat(grid, fonts[int(rng.integers(len(fonts)))], dark=(style == "dark"), rng=rng)
        img = make_photo(flat, rng) if style == "photo" else flat
        cv2.imwrite(str(out / f"{i}_{style}.jpg"), img)
        save_csv(grid, out / f"{i}_{style}.csv")
    print(f"wrote test images to {out}/")


if __name__ == "__main__":
    main()
