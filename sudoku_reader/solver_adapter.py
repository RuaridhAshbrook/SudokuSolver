"""Connects the app to your sudoku solver.

The GUI calls solve(grid) and nothing else, so this is the only file you need to touch.

  grid    : list of 9 lists of 9 ints, 0 = blank
  returns : the solved grid in the same shape, or None if there is no solution

To use YOUR solver, replace the body of solve() with a call to it, e.g.

    from my_solver import solve_sudoku
    def solve(grid):
        return solve_sudoku(grid)        # adapt the format here if yours differs

(for example if your solver wants strings, numpy arrays or '.' for blanks, convert here).
Until then, the built-in backtracking solver below is used so the app works out of the box.
"""
from __future__ import annotations


def solve(grid):
    return _default_solve(grid)


# --------------------------------------------------------------------------- built-in fallback
def _default_solve(grid):
    g = [[int(v) for v in row] for row in grid]
    rows, cols, boxes = [0] * 9, [0] * 9, [0] * 9
    empties = []
    for r in range(9):
        for c in range(9):
            v = g[r][c]
            if v:
                bit = 1 << v
                b = (r // 3) * 3 + c // 3
                if rows[r] & bit or cols[c] & bit or boxes[b] & bit:
                    return None
                rows[r] |= bit; cols[c] |= bit; boxes[b] |= bit
            else:
                empties.append((r, c))

    def candidates(r, c):
        used = rows[r] | cols[c] | boxes[(r // 3) * 3 + c // 3]
        return [v for v in range(1, 10) if not used & (1 << v)]

    def search():
        if not empties:
            return True
        # most-constrained cell first
        best_i, best_c = -1, None
        for i, (r, c) in enumerate(empties):
            cand = candidates(r, c)
            if best_c is None or len(cand) < len(best_c):
                best_i, best_c = i, cand
                if len(cand) <= 1:
                    break
        if not best_c:
            return False
        r, c = empties.pop(best_i)
        b = (r // 3) * 3 + c // 3
        for v in best_c:
            bit = 1 << v
            g[r][c] = v
            rows[r] |= bit; cols[c] |= bit; boxes[b] |= bit
            if search():
                return True
            rows[r] &= ~bit; cols[c] &= ~bit; boxes[b] &= ~bit
        g[r][c] = 0
        empties.insert(best_i, (r, c))
        return False

    return g if search() else None
