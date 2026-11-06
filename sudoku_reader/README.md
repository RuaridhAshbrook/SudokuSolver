# Sudoku Reader

Image of a sudoku (screenshot, phone photo of print, or handwritten) -> 9x9 grid -> CSV, with a tkinter app that lets you check the result and solve it.

## How it works

1. **Find the grid** (`sudoku_ocr/preprocess.py`, classical OpenCV, no training). Candidate outlines are scored by how much they look like a 9x9 grid, so a sheet of paper or table edge around the puzzle isn't mistaken for it. The grid is perspective-flattened, lighting is evened out, and dark-mode screenshots are inverted automatically.
2. **Cut 81 cells** and trim the edges so grid lines don't leak in.
3. **Classify each cell** with a small CNN (`model.py`): 10 classes, blank + digits 1-9. This is the only trained part.
4. **Sanity-check with sudoku rules** (`sudoku_utils.py`): duplicates in a row/column/box mean a misread, so the most likely wrong cell is swapped for its runner-up reading. Sideways or upside-down grids are detected the same way.
5. Write the CSV (9 rows, `0` = blank).

## Setup

```
pip install -r requirements.txt
```
tkinter ships with Python on Windows/macOS; on Debian/Ubuntu: `sudo apt install python3-tk`.

## Train

```
python -m sudoku_ocr.train
```
Training data is generated, so no labelling is needed:
- **Printed / screenshots:** digits 1-9 rendered in the fonts installed on your machine, with grid-line fragments, blur, noise, JPEG damage, rotation, and "pencil mark" cells labelled blank. Validation uses fonts the model never saw.
- **Handwritten:** MNIST digits composed into cells the same way (downloaded automatically; skip with `--no-mnist`).
- **Your own real cells:** anything in `data/extra/<0-9>/*.png` is added (see below).

Useful options: `--epochs 25`, `--samples-per-class 10000`, `--fonts-dir path/to/fonts`. The result is saved to `models/digit_cnn.pt`. Check it end to end with:
```
python -m tests.test_pipeline
```

## Use

- **App:** `python app.py`. Open or paste an image; the grid is read straight away. Orange cells are ones the model was unsure about, yellow were auto-corrected by sudoku rules, red are duplicates. Click a cell and type to fix it. If the outline is wrong, use *Pick corners* and click the four corners. *Solve* runs your solver (`sudoku_solver.py`) on the puzzle digits and shows the solved grid (solved digits in blue), or an error message if the sudoku is invalid or has no solution. Edit a given digit and press Solve again to re-solve.
- **Command line:** `python -m sudoku_ocr.cli puzzle.jpg` writes `puzzle.csv` next to the image.

## Your solver

`sudoku_solver.py` is your solver script. The algorithm is untouched; two things were added at the bottom: `main()` only runs when you start the file from the command line (otherwise importing it would run it), and `solve_grid(grid)` lets the app call it with a 9x9 list (0 = blank). It returns `(solved_grid, backtracks)` or raises `SudokuError` with the same "Sudoku is not solvable" message your script prints. The command-line use (`python sudoku_solver.py name` reading `Sudokus/name.csv`) still works.

CSV files can still be produced from the command line (`python -m sudoku_ocr.cli puzzle.jpg -o Sudokus/name.csv`); they use `0` for blanks, which is what your loader expects.

## Getting better on your own images

Handwriting and unusual print styles are the weak spot. In the app, fix any misread cells and press **Save corrections**: each corrected cell is saved to `data/extra/<digit>/`. Then run `python -m sudoku_ocr.train` again; those real examples are added to training (repeated 10x by default). A few dozen corrections from your own photos usually helps more than extra synthetic data.

## Limits

- Pencil marks inside cells are treated as blank, which is right for most apps but means candidate notes are ignored.
- Heavily curved pages, strong glare, or a grid cut off by the frame can defeat outline detection; use *Pick corners*.
- Auto-rotation and rule-based repair are heuristics. They help, but always glance at orange/yellow cells before trusting a result.

## Layout

```
app.py                  tkinter app
sudoku_solver.py        your solver (+ solve_grid() used by the app)
sudoku_ocr/
  preprocess.py         grid detection, warp, cell extraction
  synth.py              synthetic training-cell generator
  model.py train.py     CNN and training script
  classifier.py         loads a trained model
  recognise.py          image -> grid pipeline
  sudoku_utils.py       validity check, repair, CSV
  cli.py                command-line tool
tests/                  render_puzzle.py (test images with known answers), test_pipeline.py
```
