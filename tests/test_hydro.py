"""Hydrology must be exact, not approximately exact.

The bucket queue is the only place in the engine where a data structure is
chosen for speed over clarity, so it is the only place that needs a reference
implementation to check it against.  ``reference_fill`` is the textbook
priority flood: one heap, true water levels, no quantisation.
"""

from __future__ import annotations

import heapq

import numpy as np

from atlas import hydro


def _neighbours(i, H, W):
    y, x = divmod(i, W)
    for dy in (-1, 0, 1):
        ny = y + dy
        if ny < 0 or ny >= H:
            continue
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            nx = x + dx
            if 0 <= nx < W:
                yield ny * W + nx


def reference_fill(h, eps=hydro.EPS):
    H, W = h.shape
    flat = h.astype(np.float64).ravel()
    n = flat.size
    filled = flat.copy()
    closed = np.zeros(n, dtype=bool)
    heap = []

    seed = np.concatenate([
        np.where(flat <= 0.0)[0],
        np.arange(W), np.arange(n - W, n),
        np.arange(0, n, W), np.arange(W - 1, n, W),
    ])
    for i in np.unique(seed):
        closed[i] = True
        heapq.heappush(heap, (flat[i], int(i)))

    while heap:
        v, i = heapq.heappop(heap)
        base = v + eps
        for j in _neighbours(i, H, W):
            if closed[j]:
                continue
            closed[j] = True
            w = base if base > flat[j] else flat[j]
            filled[j] = w
            heapq.heappush(heap, (w, j))
    return filled.reshape(H, W)


def _bowl(n=64, seed=0):
    """A grid with plenty of closed basins: noise, scooped hollows, and a sea."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:n, 0:n]
    h = 0.6 * np.sin(x / 5.0) * np.cos(y / 7.0) + 0.2 * rng.standard_normal((n, n))
    for _ in range(6):
        cx, cy = rng.integers(6, n - 6, 2)
        r = rng.integers(3, 7)
        h -= 1.4 * np.exp(-(((x - cx) ** 2 + (y - cy) ** 2) / (2.0 * r * r)))
    h += 0.4
    h[:, :5] = -2.0                                  # the sea, on the west edge
    h[0, :] = h[-1, :] = h[:, -1] = 2.0              # a wall on the other three
    return h


def test_bucket_flood_matches_heap():
    for seed in range(5):
        h = _bowl(seed=seed)
        got = hydro.fill_depressions(h)
        want = reference_fill(h)
        err = float(np.abs(got - want).max())
        assert err == 0.0, f"seed {seed}: bucket flood differs from the heap by {err:.6f} m"


def test_fill_raises_only_closed_basins():
    h = _bowl(seed=7)
    filled = hydro.fill_depressions(h)
    assert (filled >= h - 1e-9).all(), "the fill lowered ground"
    raised = int((filled > h + 1e-9).sum())
    assert 0 < raised < h.size, f"raised {raised} of {h.size} cells"


def test_filled_surface_has_no_pit():
    h = _bowl(seed=11)
    filled = hydro.fill_depressions(h)
    land = filled > 0
    rec, _ = hydro.receivers(filled, land)
    rec = rec.reshape(filled.shape)
    interior = np.zeros_like(land)
    interior[1:-1, 1:-1] = True
    stuck = int(((rec < 0) & land & interior).sum())
    assert stuck == 0, f"{stuck} land cells still have no receiver after the fill"
