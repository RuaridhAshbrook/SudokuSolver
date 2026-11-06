"""Grid detection, perspective correction and cell extraction (OpenCV only, no training)."""
from __future__ import annotations

import cv2
import numpy as np

GRID_PX = 450                 # side of the flattened grid image
CELL_PX = GRID_PX // 9        # 50 px per cell
CELL_OUT = 32                 # classifier input size (CELL_OUT x CELL_OUT)
CELL_MARGIN = 0.10            # fraction trimmed from every cell edge (removes grid lines)
_DETECT_MAX_SIDE = 1000       # images are shrunk to this for outline detection


def load_image(path) -> np.ndarray:
    """Read an image as BGR. Works with unicode paths and honours phone EXIF rotation."""
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    return img


def to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def order_corners(pts) -> np.ndarray:
    """Return 4 points ordered top-left, top-right, bottom-right, bottom-left."""
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = pts[:, 1] - pts[:, 0]
    return np.array(
        [pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]],
        dtype=np.float32,
    )


def _quad_from_contour(cnt):
    hull = cv2.convexHull(cnt)
    peri = cv2.arcLength(hull, True)
    for eps in (0.01, 0.02, 0.03, 0.04, 0.06):
        approx = cv2.approxPolyDP(hull, eps * peri, True)
        if len(approx) == 4:
            return approx.reshape(4, 2).astype(np.float32)
    return None


def _squareness(quad) -> float:
    q = order_corners(quad)
    sides = [np.linalg.norm(q[i] - q[(i + 1) % 4]) for i in range(4)]
    return min(sides) / max(max(sides), 1e-6)


def _grid_score(gray: np.ndarray, quad) -> float:
    """How much does this quad look like a sudoku grid? Warp it and measure how strong the
    ink lines are at the 10 expected row/column positions compared with everywhere else.
    A page edge, a table or a box around the puzzle scores low because its lines don't
    line up with multiples of 1/9."""
    b = grid_to_ink(warp_grid(gray, quad)) > 90
    pos = [min(int(round(i * (GRID_PX - 1) / 9)), GRID_PX - 1) for i in range(10)]
    total = 0.0
    for prof in (b.mean(axis=1), b.mean(axis=0)):
        lines = np.mean([prof[max(0, p - 4):p + 5].max() for p in pos])
        total += lines - float(np.median(prof))
    return total / 2


def find_grid_corners(img: np.ndarray):
    """Locate the sudoku outline.

    Returns (corners, found). corners is a (4, 2) float32 array in the original image's
    pixel coordinates ordered TL, TR, BR, BL. If nothing plausible is found the whole
    image is returned with found=False.
    """
    gray = to_gray(img)
    h, w = gray.shape
    scale = min(1.0, _DETECT_MAX_SIDE / max(h, w))
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else gray
    blur = cv2.GaussianBlur(small, (7, 7), 0)
    binary = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    binary = cv2.dilate(binary, np.ones((3, 3), np.uint8))
    # RETR_LIST (not EXTERNAL): a grid printed on a sheet of paper sits *inside* the page outline.
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    img_area = small.shape[0] * small.shape[1]
    best, best_key = None, None
    for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:12]:
        area = cv2.contourArea(cnt)
        if area < 0.05 * img_area:
            break
        quad = _quad_from_contour(cnt)
        if quad is None or _squareness(quad) < 0.5:
            continue
        score = _grid_score(small, quad)
        if score < 0.2:
            continue
        key = score + 0.1 * area / img_area      # tiny bonus: prefer the outer edge of a thick border
        if best_key is None or key > best_key:
            best, best_key = quad, key

    if best is None:
        full = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
        return full, False
    return order_corners(best / scale), True


def warp_grid(img: np.ndarray, corners) -> np.ndarray:
    """Perspective-correct the grid to a GRID_PX x GRID_PX greyscale image."""
    gray = to_gray(img)
    dst = np.array(
        [[0, 0], [GRID_PX - 1, 0], [GRID_PX - 1, GRID_PX - 1], [0, GRID_PX - 1]], dtype=np.float32
    )
    m = cv2.getPerspectiveTransform(order_corners(corners), dst)
    return cv2.warpPerspective(gray, m, (GRID_PX, GRID_PX), flags=cv2.INTER_LINEAR)


def grid_to_ink(gray: np.ndarray) -> np.ndarray:
    """Normalise a warped grid so ink is bright (~255) on a dark (~0) background.

    Handles dark-mode screenshots (polarity), uneven lighting (background division) and
    faint print (global contrast stretch).
    """
    g = gray
    _, otsu = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.count_nonzero(otsu) < otsu.size / 2:      # bright pixels are the minority -> bright ink
        g = 255 - g
    bg = cv2.dilate(g, np.ones((17, 17), np.uint8))   # wipes out strokes thinner than 17 px
    bg = cv2.GaussianBlur(bg, (0, 0), 12)
    flat = cv2.divide(g, np.maximum(bg, 1), scale=255)
    ink = 255 - flat
    p = float(np.percentile(ink, 99.5))
    ink = np.clip(ink.astype(np.float32) * (255.0 / max(p, 60.0)), 0, 255)
    return ink.astype(np.uint8)


def prepare_cell(cell: np.ndarray) -> np.ndarray:
    """Trim the edges of one ink-bright cell and resize to CELL_OUT x CELL_OUT.

    The synthetic data generator calls this exact function, so training and real
    inputs go through identical preprocessing.
    """
    cell = np.ascontiguousarray(cell)
    h, w = cell.shape[:2]
    my, mx = int(round(h * CELL_MARGIN)), int(round(w * CELL_MARGIN))
    crop = cell[my:h - my, mx:w - mx]
    return cv2.resize(crop, (CELL_OUT, CELL_OUT), interpolation=cv2.INTER_AREA)


def split_cells(ink: np.ndarray) -> np.ndarray:
    """Cut the GRID_PX x GRID_PX ink image into 81 prepared cells, row-major. Shape (81, 32, 32)."""
    cells = np.empty((81, CELL_OUT, CELL_OUT), dtype=np.uint8)
    for r in range(9):
        for c in range(9):
            cell = ink[r * CELL_PX:(r + 1) * CELL_PX, c * CELL_PX:(c + 1) * CELL_PX]
            cells[r * 9 + c] = prepare_cell(cell)
    return cells
