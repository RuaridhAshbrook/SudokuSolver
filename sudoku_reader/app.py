"""Sudoku Reader: open a sudoku image, check/fix the recognised grid, then solve it with sudoku_solver.py.

    python app.py
"""
from __future__ import annotations

import queue
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from sudoku_ocr.preprocess import load_image, order_corners
from sudoku_ocr.recognise import DEFAULT_MODEL, SudokuReader
from sudoku_ocr.sudoku_utils import find_conflicts, load_csv
from sudoku_solver import SudokuError, solve_grid

HERE = Path(__file__).resolve().parent
LOW_CONF = 0.85           # cells below this probability get highlighted for review
CANVAS = 520

BG_NORMAL, BG_LOW, BG_REPAIRED, BG_CONFLICT = "#ffffff", "#ffd9a0", "#fff2a0", "#ffb3b3"
FG_GIVEN, FG_SOLVED = "#111111", "#1a4fd6"


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Sudoku Reader")
        self.resizable(False, False)

        self.image = None            # BGR source image
        self.image_path = None
        self.photo = None            # keeps the Tk image alive
        self.scale, self.offset = 1.0, (0, 0)
        self.corners = None          # grid corners in image coordinates
        self.manual = False          # True once the user picked corners by hand
        self.picking, self.pick_pts = False, []
        self.reader = None
        self.result = None           # last ReadResult
        self.low, self.repaired, self.solved = set(), set(), set()
        self._loading = False
        self._queue = queue.Queue()

        self._build_ui()
        self.status("Open an image (or paste a screenshot) to begin.")

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        pad = {"padx": 8, "pady": 6}
        left = ttk.Frame(self)
        left.grid(row=0, column=0, **pad)
        right = ttk.Frame(self)
        right.grid(row=0, column=1, **pad)

        self.canvas = tk.Canvas(left, width=CANVAS, height=CANVAS, bg="#2b2b2b", highlightthickness=0)
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self._on_canvas_click)
        bar = ttk.Frame(left)
        bar.pack(fill="x", pady=(6, 0))
        for text, cmd in (("Open image…", self.open_image), ("Paste image", self.paste_image),
                          ("Pick corners", self.start_picking), ("Auto corners", self.auto_corners)):
            ttk.Button(bar, text=text, command=cmd).pack(side="left", padx=2)

        grid_frame = tk.Frame(right, bg="#222222", padx=3, pady=3)
        grid_frame.pack()
        vcmd = (self.register(self._valid), "%P")
        self.vars, self.entries = [], []
        for r in range(9):
            vrow, erow = [], []
            for c in range(9):
                var = tk.StringVar()
                e = tk.Entry(grid_frame, textvariable=var, width=2, justify="center",
                             font=("Helvetica", 20, "bold"), relief="flat", bd=0,
                             validate="key", validatecommand=vcmd,
                             highlightthickness=1, highlightbackground="#999999")
                e.grid(row=r, column=c, ipady=5,
                       padx=(3 if c in (3, 6) else 0, 0), pady=(3 if r in (3, 6) else 0, 0))
                var.trace_add("write", lambda *_a, r=r, c=c: self._on_edit(r, c))
                vrow.append(var)
                erow.append(e)
            self.vars.append(vrow)
            self.entries.append(erow)

        style = ttk.Style(self)
        style.configure("Solve.TButton", font=("Helvetica", 14, "bold"), padding=8)
        btns = ttk.Frame(right)
        btns.pack(fill="x", pady=(10, 0))
        self.solve_btn = ttk.Button(btns, text="Solve", style="Solve.TButton", command=self.solve)
        self.solve_btn.grid(row=0, column=0, columnspan=3, padx=2, pady=(0, 6), sticky="ew")
        specs = (("Read grid", self.read_grid), ("Clear", self.clear), ("Open CSV…", self.open_csv),
                 ("Save corrections", self.save_corrections))
        for i, (text, cmd) in enumerate(specs):
            ttk.Button(btns, text=text, command=cmd).grid(
                row=1 + i // 3, column=i % 3, padx=2, pady=2, sticky="ew")
        for i in range(3):
            btns.columnconfigure(i, weight=1)

        legend = ttk.Label(right, justify="left", foreground="#555555",
                           text="orange = unsure, check it    yellow = auto-corrected by sudoku rules\n"
                                "red = duplicate in a row/column/box    blue = filled in by your solver")
        legend.pack(anchor="w", pady=(8, 0))

        self.status_var = tk.StringVar()
        ttk.Label(self, textvariable=self.status_var, anchor="w", relief="sunken").grid(
            row=1, column=0, columnspan=2, sticky="ew")

    def status(self, msg):
        self.status_var.set(msg)
        self.update_idletasks()

    # ------------------------------------------------------------------ grid widgets
    @staticmethod
    def _valid(proposed):
        return proposed == "" or (len(proposed) == 1 and proposed in "123456789")

    def _on_edit(self, r, c):
        if self._loading:
            return
        self.low.discard((r, c))
        self.repaired.discard((r, c))
        self.solved.discard((r, c))
        self._refresh_colours()

    def get_grid(self):
        return [[int(self.vars[r][c].get() or 0) for c in range(9)] for r in range(9)]

    def set_grid(self, grid):
        self._loading = True
        try:
            for r in range(9):
                for c in range(9):
                    v = int(grid[r][c])
                    self.vars[r][c].set(str(v) if v else "")
        finally:
            self._loading = False
        self._refresh_colours()

    def _refresh_colours(self):
        bad = find_conflicts(self.get_grid())
        for r in range(9):
            for c in range(9):
                if (r, c) in bad:
                    bg = BG_CONFLICT
                elif (r, c) in self.low:
                    bg = BG_LOW
                elif (r, c) in self.repaired:
                    bg = BG_REPAIRED
                else:
                    bg = BG_NORMAL
                self.entries[r][c].config(bg=bg, fg=FG_SOLVED if (r, c) in self.solved else FG_GIVEN)

    def clear(self):
        self.low, self.repaired, self.solved = set(), set(), set()
        self.set_grid([[0] * 9 for _ in range(9)])
        self.status("Cleared.")

    # ------------------------------------------------------------------ image handling
    def open_image(self):
        path = filedialog.askopenfilename(
            title="Choose a sudoku image",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.webp *.tif *.tiff"), ("All files", "*.*")])
        if not path:
            return
        try:
            img = load_image(path)
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("Can't open image", str(e))
            return
        self._set_image(img, path)

    def paste_image(self):
        try:
            from PIL import ImageGrab
            clip = ImageGrab.grabclipboard()
        except Exception as e:  # noqa: BLE001
            messagebox.showinfo("Paste image", f"Couldn't read the clipboard here ({e}).")
            return
        if isinstance(clip, Image.Image):
            self._set_image(np.array(clip.convert("RGB"))[:, :, ::-1].copy(), None)
        elif isinstance(clip, list) and clip:
            try:
                self._set_image(load_image(clip[0]), clip[0])
            except Exception as e:  # noqa: BLE001
                messagebox.showerror("Can't open image", str(e))
        else:
            self.status("There's no image on the clipboard.")

    def _set_image(self, img, path):
        self.image, self.image_path = img, path
        self.corners, self.manual, self.picking, self.pick_pts = None, False, False, []
        self.result = None
        self._draw()
        self.read_grid()

    def _draw(self):
        self.canvas.delete("all")
        if self.image is None:
            return
        h, w = self.image.shape[:2]
        self.scale = min(CANVAS / w, CANVAS / h)
        nw, nh = max(1, int(w * self.scale)), max(1, int(h * self.scale))
        self.offset = ((CANVAS - nw) // 2, (CANVAS - nh) // 2)
        rgb = cv2.cvtColor(cv2.resize(self.image, (nw, nh), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        self.photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.canvas.create_image(self.offset[0], self.offset[1], anchor="nw", image=self.photo)
        if self.corners is not None:
            pts = [self._to_canvas(p) for p in order_corners(self.corners)]
            self.canvas.create_polygon(*[v for p in pts for v in p], outline="#00e676", fill="", width=3)
        for p in self.pick_pts:
            x, y = self._to_canvas(p)
            self.canvas.create_oval(x - 5, y - 5, x + 5, y + 5, fill="#ff5252", outline="white")

    def _to_canvas(self, p):
        return p[0] * self.scale + self.offset[0], p[1] * self.scale + self.offset[1]

    def start_picking(self):
        if self.image is None:
            self.status("Open an image first.")
            return
        self.picking, self.pick_pts = True, []
        self._draw()
        self.status("Click the 4 outer corners of the grid (any order).")

    def auto_corners(self):
        if self.image is None:
            return
        self.manual, self.picking, self.pick_pts = False, False, []
        self.read_grid()

    def _on_canvas_click(self, event):
        if not (self.picking and self.image is not None):
            return
        x = (event.x - self.offset[0]) / self.scale
        y = (event.y - self.offset[1]) / self.scale
        h, w = self.image.shape[:2]
        if not (0 <= x < w and 0 <= y < h):
            return
        self.pick_pts.append((x, y))
        if len(self.pick_pts) < 4:
            self._draw()
            self.status(f"{len(self.pick_pts)}/4 corners picked.")
            return
        self.corners = order_corners(np.array(self.pick_pts, dtype=np.float32))
        self.manual, self.picking, self.pick_pts = True, False, []
        self.read_grid()

    # ------------------------------------------------------------------ recognition
    def _get_reader(self):
        if self.reader is not None:
            return self.reader
        if not Path(DEFAULT_MODEL).exists():
            messagebox.showerror(
                "No trained model yet",
                f"Expected a model at:\n{DEFAULT_MODEL}\n\nTrain one first:\n    python -m sudoku_ocr.train")
            return None
        self.status("Loading model…")
        try:
            self.reader = SudokuReader.from_model_file(DEFAULT_MODEL)
        except ImportError:
            messagebox.showerror("PyTorch missing", "Install it with:\n    pip install torch torchvision")
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("Couldn't load the model", str(e))
        return self.reader

    def read_grid(self):
        if self.image is None:
            self.status("Open an image first.")
            return
        reader = self._get_reader()
        if reader is None:
            return
        self.status("Reading…")
        try:
            res = reader.read(self.image, corners=self.corners if self.manual else None)
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("Couldn't read the image", str(e))
            return
        self.result = res
        self.corners = res.corners
        self._draw()
        self.solved = set()
        self.low = {(r, c) for r in range(9) for c in range(9) if res.confidence[r, c] < LOW_CONF}
        self.repaired = set(res.changed)
        self.set_grid(res.grid)

        msg = f"Read {int((res.grid > 0).sum())} digits."
        if not res.grid_found and not self.manual:
            msg = "Couldn't find the grid outline, so I used the whole image. Try 'Pick corners'. " + msg
        if self.low:
            msg += f" {len(self.low)} unsure cell(s) are orange: please check them."
        if find_conflicts(res.grid.tolist()):
            msg += " Duplicates remain (red)."
        self.status(msg)

    # ------------------------------------------------------------------ solving (background thread)
    def solve(self):
        givens = self._givens()          # the puzzle only, without digits a previous solve filled in
        if not any(any(row) for row in givens):
            self.status("Nothing to solve yet.")
            return
        self.solve_btn.state(["disabled"])
        self.status("Solving…")

        def work():
            t0 = time.perf_counter()
            try:
                solution, backtracks = solve_grid(givens)
                self._queue.put(("ok", givens, (solution, backtracks, time.perf_counter() - t0)))
            except SudokuError as e:
                self._queue.put(("invalid", givens, str(e)))
            except Exception as e:  # noqa: BLE001
                self._queue.put(("error", givens, e))

        threading.Thread(target=work, daemon=True).start()
        self.after(100, self._poll_solver)

    def _poll_solver(self):
        try:
            kind, givens, payload = self._queue.get_nowait()
        except queue.Empty:
            self.after(100, self._poll_solver)
            return
        self.solve_btn.state(["!disabled"])
        if kind == "ok":
            solution, backtracks, secs = payload
            self.solved = {(r, c) for r in range(9) for c in range(9) if givens[r][c] == 0}
            self.low, self.repaired = set(), set()
            self.set_grid(solution)
            self.status(f"Solved: filled in {len(self.solved)} cells (blue) in {secs:.2f}s, "
                        f"{backtracks} backtracks.")
        elif kind == "invalid":
            if find_conflicts(givens):
                hint = "Cells in red repeat a digit in their row, column or box."
            else:
                hint = "No valid solution exists for these digits. One may have been misread: check the orange cells."
            messagebox.showerror("Invalid sudoku", f"{payload}.\n\n{hint}")
            self.status("The sudoku is invalid, so nothing was solved.")
        else:
            messagebox.showerror("Solver error", str(payload))
            self.status("The solver raised an error.")

    # ------------------------------------------------------------------ files
    def _givens(self):
        """Current grid without the solver's digits."""
        g = self.get_grid()
        for r, c in self.solved:
            g[r][c] = 0
        return g

    def open_csv(self):
        path = filedialog.askopenfilename(title="Open CSV", filetypes=[("CSV", "*.csv"), ("All files", "*.*")])
        if not path:
            return
        try:
            grid = load_csv(path)
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("Can't open CSV", str(e))
            return
        self.low, self.repaired, self.solved = set(), set(), set()
        self.set_grid(grid)
        self.status(f"Loaded {path}")

    def save_corrections(self):
        """Store every cell you corrected as a labelled training example (data/extra/<digit>/)."""
        if self.result is None:
            self.status("Read an image first, then fix any mistakes, then save the corrections.")
            return
        given = self._givens()
        raw = self.result.probs.argmax(1)
        n = 0
        for r in range(9):
            for c in range(9):
                i = r * 9 + c
                if given[r][c] != raw[i]:
                    folder = HERE / "data" / "extra" / str(given[r][c])
                    folder.mkdir(parents=True, exist_ok=True)
                    ok, buf = cv2.imencode(".png", self.result.cells[i])
                    if ok:
                        buf.tofile(str(folder / f"{int(time.time())}_{i}.png"))
                        n += 1
        if n:
            self.status(f"Saved {n} corrected cell(s) to data/extra/. Retrain to learn from them.")
        else:
            self.status("Nothing to save: the grid matches what the model originally read.")


if __name__ == "__main__":
    App().mainloop()
