"""Synthetic training cells.

Every cell is built the way a real one looks after grid detection: a glyph pasted into a
cell-sized canvas, grid-line fragments leaking in from the edges, slight rotation, blur,
noise and JPEG damage, then passed through preprocess.prepare_cell (the same function the
real pipeline uses).

Glyph sources are "banks": dict {digit 1..9: [uint8 glyph arrays, ink bright, tightly cropped]}.
Banks can come from system fonts (printed puzzles / screenshots) or MNIST (handwriting).
"""
from __future__ import annotations

import glob
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .preprocess import CELL_OUT, CELL_PX, prepare_cell

S = CELL_PX * 2   # cells are composed at 2x then downsampled

_BAD_FONT_NAMES = ("dingbat", "symbol", "wingding", "webding", "emoji", "icon", "awesome",
                   "ornament", "braille", "bats", "music", "fallback")


# --------------------------------------------------------------------------- fonts
def font_dirs(extra=()):
    dirs = list(extra)
    home = os.path.expanduser("~")
    if sys.platform.startswith("win"):
        dirs += [os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts"),
                 os.path.join(home, "AppData", "Local", "Microsoft", "Windows", "Fonts")]
    elif sys.platform == "darwin":
        dirs += ["/System/Library/Fonts", "/Library/Fonts", os.path.join(home, "Library", "Fonts")]
    else:
        dirs += ["/usr/share/fonts", "/usr/local/share/fonts", os.path.join(home, ".fonts"),
                 os.path.join(home, ".local", "share", "fonts")]
    return [d for d in dirs if os.path.isdir(d)]


def find_font_files(extra=()):
    files = []
    for d in font_dirs(extra):
        for ext in ("ttf", "otf", "ttc"):
            files += glob.glob(os.path.join(d, "**", f"*.{ext}"), recursive=True)
    files = [f for f in sorted(set(files)) if not any(b in os.path.basename(f).lower() for b in _BAD_FONT_NAMES)]
    return files


def _render_glyph(font, ch, canvas=128):
    img = Image.new("L", (canvas, canvas), 0)
    ImageDraw.Draw(img).text((canvas // 4, canvas // 8), ch, font=font, fill=255)
    a = np.asarray(img)
    ys, xs = np.nonzero(a > 32)
    if len(ys) == 0:
        return None
    return a[ys.min():ys.max() + 1, xs.min():xs.max() + 1].copy()


def _font_bank(path):
    try:
        font = ImageFont.truetype(path, 72)
        bank = {}
        for d in range(1, 10):
            g = _render_glyph(font, str(d))
            if g is None or g.shape[0] < 25:
                return None
            bank[d] = g
    except Exception:
        return None
    thumbs = {cv2.resize(g, (12, 12), interpolation=cv2.INTER_AREA).tobytes() for g in bank.values()}
    return bank if len(thumbs) == 9 else None   # all nine digits must look different


def _template(g):
    v = cv2.resize(g, (20, 20), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
    v -= v.mean()
    return v / (np.linalg.norm(v) + 1e-6)


def load_font_banks(extra_dirs=(), min_similarity=0.55, verbose=True):
    """Return (list of per-font banks, list of their paths). Weird fonts are filtered out by
    comparing each font's digits against the cross-font median shape."""
    banks, paths = [], []
    for p in find_font_files(extra_dirs):
        b = _font_bank(p)
        if b is not None:
            banks.append(b)
            paths.append(p)
    if len(banks) >= 5:
        vecs = np.stack([[_template(b[d]) for d in range(1, 10)] for b in banks])   # (F, 9, 400)
        med = np.median(vecs, axis=0)
        med /= np.linalg.norm(med, axis=1, keepdims=True) + 1e-6
        sim = np.einsum("fdk,dk->fd", vecs, med).mean(1)
        keep = [i for i, s in enumerate(sim) if s >= min_similarity]
        if verbose:
            print(f"fonts: {len(banks)} usable, {len(keep)} kept after outlier filter")
        banks, paths = [banks[i] for i in keep], [paths[i] for i in keep]
    return banks, paths


def merge_banks(banks):
    out = {d: [] for d in range(1, 10)}
    for b in banks:
        for d in range(1, 10):
            out[d].append(b[d])
    return out


def bank_from_images(images, labels, per_digit=None, thresh=60):
    """Build a bank from ink-bright digit images (e.g. MNIST 28x28, white on black)."""
    out = {d: [] for d in range(1, 10)}
    for img, lab in zip(images, labels):
        lab = int(lab)
        if lab == 0 or (per_digit and len(out[lab]) >= per_digit):
            continue
        ys, xs = np.nonzero(img > thresh)
        if len(ys) < 10:
            continue
        out[lab].append(np.ascontiguousarray(img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]))
    return out


# --------------------------------------------------------------------------- cell composition
def _paste(canvas, glyph, height, cx, cy, rng, width_jitter=0.15):
    h, w = glyph.shape
    scale = height / h
    nw = int(np.clip(w * scale * rng.uniform(1 - width_jitter, 1 + width_jitter), 2, S * 0.9))
    nh = max(2, int(height))
    g = cv2.resize(glyph, (nw, nh), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    x0, y0 = int(round(cx - nw / 2)), int(round(cy - nh / 2))
    xa, ya, xb, yb = max(x0, 0), max(y0, 0), min(x0 + nw, S), min(y0 + nh, S)
    if xb <= xa or yb <= ya:
        return
    patch = g[ya - y0:yb - y0, xa - x0:xb - x0]
    canvas[ya:yb, xa:xb] = np.maximum(canvas[ya:yb, xa:xb], patch)


def _grid_lines(canvas, rng):
    for side in range(4):
        if rng.random() < 0.35:
            continue
        thick = int(rng.choice((6, 8))) if rng.random() < 0.2 else int(rng.choice((2, 3, 4)))
        off = int(rng.integers(-6, 10))
        tilt = int(rng.integers(-3, 4))
        inten = float(rng.uniform(0.6, 1.0))
        if side == 0:
            p1, p2 = (0, off), (S, off + tilt)
        elif side == 1:
            p1, p2 = (0, S - 1 - off), (S, S - 1 - off + tilt)
        elif side == 2:
            p1, p2 = (off, 0), (off + tilt, S)
        else:
            p1, p2 = (S - 1 - off, 0), (S - 1 - off + tilt, S)
        cv2.line(canvas, p1, p2, inten, thick)


def compose_cell(label, bank, rng):
    """One augmented 32x32 uint8 training cell. label 0 = blank."""
    canvas = np.zeros((S, S), np.float32)
    ink = float(rng.uniform(0.55, 1.0))

    if label > 0:
        glyph = bank[label][int(rng.integers(len(bank[label])))]
        g = np.zeros((S, S), np.float32)
        _paste(g, glyph, rng.uniform(0.33, 0.78) * S,
               S / 2 + rng.uniform(-0.08, 0.08) * S, S / 2 + rng.uniform(-0.08, 0.08) * S, rng)
        if rng.random() < 0.3:
            g = cv2.dilate(g, np.ones((3, 3), np.uint8))
        canvas = g * ink
    else:
        if rng.random() < 0.2:      # pencil marks: small digits that must NOT be read as a given
            for _ in range(int(rng.integers(1, 5))):
                d = int(rng.integers(1, 10))
                gl = bank[d][int(rng.integers(len(bank[d])))]
                i = int(rng.integers(9))
                cx = S * (0.2 + 0.3 * (i % 3)) + rng.uniform(-4, 4)
                cy = S * (0.2 + 0.3 * (i // 3)) + rng.uniform(-4, 4)
                tmp = np.zeros((S, S), np.float32)
                _paste(tmp, gl, rng.uniform(0.14, 0.24) * S, cx, cy, rng)
                canvas = np.maximum(canvas, tmp * float(rng.uniform(0.5, 0.9)))
        if rng.random() < 0.08:     # stray specks
            for _ in range(int(rng.integers(1, 4))):
                cv2.circle(canvas, (int(rng.integers(10, S - 10)), int(rng.integers(10, S - 10))),
                           int(rng.integers(1, 4)), float(rng.uniform(0.4, 0.9)), -1)

    _grid_lines(canvas, rng)
    canvas = cv2.resize(canvas, (CELL_PX, CELL_PX), interpolation=cv2.INTER_AREA)

    if rng.random() < 0.8:          # small rotation / scale, as left over from imperfect warps
        angle = float(np.clip(rng.normal(0, 2.5), -6, 6))
        m = cv2.getRotationMatrix2D((CELL_PX / 2, CELL_PX / 2), angle, float(rng.uniform(0.96, 1.04)))
        canvas = cv2.warpAffine(canvas, m, (CELL_PX, CELL_PX), flags=cv2.INTER_LINEAR)
    if rng.random() < 0.7:
        sigma = float(rng.uniform(0.2, 1.3))
        canvas = cv2.GaussianBlur(canvas, (0, 0), sigma)
    canvas = canvas + float(rng.uniform(0, 0.03)) + rng.normal(0, float(rng.uniform(0, 0.04)), canvas.shape)
    img = (np.clip(canvas, 0, 1) * 255).astype(np.uint8)
    if rng.random() < 0.25:
        q = int(rng.integers(25, 90))
        img = cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])[1], cv2.IMREAD_GRAYSCALE)
    return prepare_cell(img)


# --------------------------------------------------------------------------- dataset
_BANK = None


def _init_worker(bank):
    global _BANK
    _BANK = bank


def _gen_chunk(args):
    labels, seed = args
    rng = np.random.default_rng(seed)
    out = np.empty((len(labels), CELL_OUT, CELL_OUT), np.uint8)
    for i, lab in enumerate(labels):
        out[i] = compose_cell(int(lab), _BANK, rng)
    return out


def generate_dataset(bank, n_per_digit, n_blank, seed=0, workers=1):
    """Return (X uint8 (N, 32, 32), y int64 (N,)). Classes: 0 blank, 1-9 digits."""
    labels = np.concatenate([np.repeat(np.arange(1, 10), n_per_digit), np.zeros(n_blank, dtype=np.int64)])
    rng = np.random.default_rng(seed)
    rng.shuffle(labels)
    chunks = np.array_split(labels, max(1, len(labels) // 2000))
    jobs = [(c, seed * 1_000_003 + i + 1) for i, c in enumerate(chunks)]
    if workers <= 1:
        _init_worker(bank)
        parts = [_gen_chunk(j) for j in jobs]
    else:
        with ProcessPoolExecutor(workers, initializer=_init_worker, initargs=(bank,)) as ex:
            parts = list(ex.map(_gen_chunk, jobs))
    return np.concatenate(parts), labels.astype(np.int64)
