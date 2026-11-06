"""Train the cell classifier.

    python -m sudoku_ocr.train                      # printed (fonts) + handwritten (MNIST)
    python -m sudoku_ocr.train --no-mnist           # printed only, no download needed
    python -m sudoku_ocr.train --epochs 25 --samples-per-class 10000

Training data:
  * fonts on your system, rendered into augmented cells (validated on *held-out fonts*)
  * MNIST digits composed into cells for handwriting (validated on the MNIST test split)
  * optional real cells in data/extra/<0-9>/*.png (the GUI saves your corrections there)
"""
from __future__ import annotations

import argparse
import glob
import os
import random
import time

import cv2
import numpy as np

from .preprocess import CELL_OUT
from .synth import bank_from_images, generate_dataset, load_font_banks, merge_banks


def load_mnist(root):
    from torchvision import datasets

    tr = datasets.MNIST(root, train=True, download=True)
    te = datasets.MNIST(root, train=False, download=True)
    return (tr.data.numpy(), tr.targets.numpy()), (te.data.numpy(), te.targets.numpy())


def load_extra(extra_dir):
    xs, ys = [], []
    for d in range(10):
        for p in sorted(glob.glob(os.path.join(extra_dir, str(d), "*.png"))):
            im = cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_GRAYSCALE)
            if im is None:
                continue
            if im.shape != (CELL_OUT, CELL_OUT):
                im = cv2.resize(im, (CELL_OUT, CELL_OUT), interpolation=cv2.INTER_AREA)
            xs.append(im)
            ys.append(d)
    if not xs:
        return np.zeros((0, CELL_OUT, CELL_OUT), np.uint8), np.zeros(0, np.int64)
    return np.stack(xs), np.array(ys, dtype=np.int64)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="models/digit_cnn.pt")
    ap.add_argument("--samples-per-class", type=int, default=6000, help="printed-font cells per digit")
    ap.add_argument("--mnist-per-class", type=int, default=2500, help="handwritten cells per digit")
    ap.add_argument("--no-mnist", action="store_true")
    ap.add_argument("--mnist-dir", default="data/mnist")
    ap.add_argument("--extra-dir", default="data/extra")
    ap.add_argument("--extra-repeat", type=int, default=10, help="how many times to repeat your real cells")
    ap.add_argument("--fonts-dir", action="append", default=[], help="extra folder of .ttf/.otf fonts")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch
    from torch import nn

    from .model import DigitNet, to_tensor

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    t0 = time.time()
    xs_tr, ys_tr, val_sets = [], [], {}

    # ---- printed digits from fonts; validation uses fonts the model never saw
    banks, _ = load_font_banks(args.fonts_dir)
    if len(banks) < 3:
        raise SystemExit("Found fewer than 3 usable fonts. Pass --fonts-dir with a folder of .ttf files.")
    order = list(range(len(banks)))
    random.shuffle(order)
    n_val = max(1, int(len(banks) * 0.15))
    val_banks = [banks[i] for i in order[:n_val]]
    train_banks = [banks[i] for i in order[n_val:]]
    n = args.samples_per_class
    x, y = generate_dataset(merge_banks(train_banks), n, 2 * n, seed=args.seed + 1, workers=args.workers)
    xs_tr.append(x); ys_tr.append(y)
    vn = max(200, n // 8)
    val_sets["fonts"] = generate_dataset(merge_banks(val_banks), vn, 2 * vn, seed=args.seed + 2, workers=args.workers)
    print(f"printed: {len(x)} train cells from {len(train_banks)} fonts, validating on {len(val_banks)} unseen fonts")

    # ---- handwritten digits from MNIST
    if not args.no_mnist:
        try:
            (tri, trl), (tei, tel) = load_mnist(args.mnist_dir)
            m = args.mnist_per_class
            x, y = generate_dataset(bank_from_images(tri, trl), m, m, seed=args.seed + 3, workers=args.workers)
            xs_tr.append(x); ys_tr.append(y)
            vm = max(200, m // 8)
            val_sets["mnist"] = generate_dataset(bank_from_images(tei, tel), vm, vm, seed=args.seed + 4, workers=args.workers)
            print(f"handwritten: {len(x)} train cells from MNIST")
        except Exception as e:  # noqa: BLE001
            print(f"!! MNIST unavailable ({e}); continuing without handwriting. Use --no-mnist to silence.")

    # ---- real cells you collected
    ex, ey = load_extra(args.extra_dir)
    if len(ex):
        xs_tr.append(np.repeat(ex, args.extra_repeat, axis=0))
        ys_tr.append(np.repeat(ey, args.extra_repeat))
        print(f"extra: {len(ex)} real cells from {args.extra_dir} (x{args.extra_repeat})")

    X = np.concatenate(xs_tr)
    Y = np.concatenate(ys_tr)
    print(f"data ready in {time.time() - t0:.0f}s: {len(X)} training cells")

    model = DigitNet().to(device)
    steps = int(np.ceil(len(X) / args.batch_size))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=steps)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)

    def predict(xu8):
        model.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(xu8), 1024):
                out.append(model(to_tensor(xu8[i:i + 1024]).to(device)).argmax(1).cpu().numpy())
        return np.concatenate(out)

    best, best_state = -1.0, None
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = np.random.permutation(len(X))
        total = 0.0
        for i in range(0, len(X), args.batch_size):
            idx = perm[i:i + args.batch_size]
            xb = to_tensor(X[idx]).to(device)
            yb = torch.from_numpy(Y[idx]).to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * len(idx)
        accs = {k: float((predict(vx) == vy).mean()) for k, (vx, vy) in val_sets.items()}
        score = float(np.mean(list(accs.values())))
        print(f"epoch {epoch:2d}/{args.epochs}  loss {total / len(X):.4f}  " +
              "  ".join(f"val[{k}] {a * 100:.2f}%" for k, a in accs.items()))
        if score > best:
            best = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    for k, (vx, vy) in val_sets.items():
        p = predict(vx)
        per = [float((p[vy == c] == c).mean()) if (vy == c).any() else float("nan") for c in range(10)]
        print(f"per-class accuracy on {k}: " + " ".join(f"{c}:{a * 100:.1f}" for c, a in enumerate(per)))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save({"state_dict": best_state, "classes": list(range(10))}, args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
