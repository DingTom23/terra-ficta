"""Hydrology and fluvial erosion.

A landscape without drainage looks like a model railway: bumps, but no reason
for the bumps to be where they are.  Water is the reason.  This module finds
where water goes and lets it carve.

Pipeline
--------
1. `fill_depressions` -- raise closed basins until every cell drains.  The
   fill surface is a minimax path problem: ``W[i]`` is the lowest level from
   which water can still reach an outlet, i.e. the minimum over all paths of
   the maximum height along the path.  Relaxing it with directional raster
   sweeps is the standard trick; each sweep is a max-plus prefix scan, which
   numpy does in one `maximum.accumulate`.  Cheap, and it converges in a
   handful of passes on terrain this open.
2. `receivers` -- steepest-descent (D8) flow directions, land only.
3. `accumulate` -- upstream area, by Kahn's algorithm in vectorised layers:
   every cell in a layer has all its donors resolved, so the whole layer is
   scattered in one `np.add.at`.
4. `erode` -- stream-power incision, ``E = K * A^m * S^n``, clamped so that no
   cell is cut below its own receiver (which would punch new pits).
"""

from __future__ import annotations

import numpy as np

from .filters import blur

OFFSETS = (
    (-1, -1, 1.41421356), (-1, 0, 1.0), (-1, 1, 1.41421356),
    (0, -1, 1.0), (0, 1, 1.0),
    (1, -1, 1.41421356), (1, 0, 1.0), (1, 1, 1.41421356),
)

EPS = 0.02  # metres of enforced descent per cell


def neighbour_min(a):
    """Minimum of the 8 neighbours; the outside world is an outlet."""
    h, w = a.shape
    p = np.pad(a, 1, mode="constant", constant_values=-np.inf)
    m = np.full_like(a, np.inf)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            np.minimum(m, p[1 + dy:1 + dy + h, 1 + dx:1 + dx + w], out=m)
    return m


def _envelope(a, eps, axis, forward):
    """max_{j before i} (a[j] + eps * steps), as an upper bound on the fill."""
    NEG = -1e18
    n = a.shape[axis]
    k = eps * np.arange(n, dtype=np.float64)
    shape = [1] * a.ndim
    shape[axis] = n
    k = k.reshape(shape)

    # Unknown cells must not poison the running maximum, so they enter the
    # scan as a very large negative number and are recognised again on exit.
    src = np.where(np.isfinite(a), a, NEG)
    if not forward:
        src = np.flip(src, axis)
    acc = np.maximum.accumulate(src - k, axis=axis)
    sh = np.full_like(acc, NEG)
    if axis == 0:
        sh[1:] = acc[:-1]
    else:
        sh[:, 1:] = acc[:, :-1]
    env = sh + k
    env = np.where(env < NEG * 0.5, np.inf, env)
    return env if forward else np.flip(env, axis)


def fill_depressions(h, eps=EPS, levels=8192):
    """Raise closed basins until every cell drains.  Priority flood.

    Cells are processed in order of water level, using a bucket queue over
    quantised heights instead of a heap: each cell is pushed once and popped
    once, and the level pointer only ever moves forward.

    The obvious relaxation, ``W[i] = max(h[i], min_j W[j] + eps)``, converges
    at one eps per pass.  A basin 300 m deep therefore needs fifteen thousand
    passes -- and in practice it diverges, because a flat floor has no lower
    neighbour and every cell in it rises forever.  Ordering is the whole
    trick: the spill level of a basin is known the moment its rim is popped.
    """
    H, W = h.shape
    flat = h.astype(np.float64).ravel()
    n = flat.size
    lo = float(flat.min())
    span = max(float(flat.max()) - lo, 1e-6)
    k = levels - 1

    filled = flat.copy()
    closed = np.zeros(n, dtype=bool)
    buckets = [[] for _ in range(levels)]

    def level(v):
        j = int((v - lo) / span * k)
        return 0 if j < 0 else (k if j >= k else j)

    seed = np.concatenate([
        np.where(flat <= 0.0)[0],
        np.arange(W), np.arange(n - W, n),
        np.arange(0, n, W), np.arange(W - 1, n, W),
    ])
    for i in np.unique(seed):
        closed[i] = True
        buckets[level(flat[i])].append(i)

    ptr = 0
    while ptr < levels:
        bucket = buckets[ptr]
        if not bucket:
            ptr += 1
            continue
        i = bucket.pop()
        y, x = divmod(i, W)
        base = filled[i] + eps
        for dy in (-1, 0, 1):
            ny = y + dy
            if ny < 0 or ny >= H:
                continue
            row = ny * W
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx = x + dx
                if nx < 0 or nx >= W:
                    continue
                j = row + nx
                if closed[j]:
                    continue
                closed[j] = True
                v = base if base > flat[j] else flat[j]
                filled[j] = v
                buckets[level(v)].append(j)
    return filled.reshape(H, W)


def receivers(h, land=None):
    """D8 steepest descent.  Returns (flat receiver index, slope)."""
    H, W = h.shape
    if land is None:
        land = h > 0
    ys, xs = np.mgrid[0:H, 0:W]
    p = np.pad(h, 1, mode="edge")
    best = np.zeros((H, W), dtype=np.float64)
    rec = np.full((H, W), -1, dtype=np.int64)
    for dy, dx, dist in OFFSETS:
        nb = p[1 + dy:1 + dy + H, 1 + dx:1 + dx + W]
        drop = (h - nb) / dist
        ny = ys + dy
        nx = xs + dx
        valid = (ny >= 0) & (ny < H) & (nx >= 0) & (nx < W) & (drop > best)
        best = np.where(valid, drop, best)
        rec = np.where(valid, ny * W + nx, rec)
    rec = np.where(land, rec, -1)
    best = np.where(land, best, 0.0)
    return rec.ravel(), best


def accumulate(rec, n):
    """Upstream cell count, including the cell itself."""
    valid = rec >= 0
    acc = np.ones(n, dtype=np.float64)
    indeg = np.bincount(rec[valid], minlength=n).astype(np.int64)
    frontier = np.where(indeg == 0)[0]
    while frontier.size:
        r = rec[frontier]
        keep = r >= 0
        if not keep.any():
            break
        r = r[keep]
        src = frontier[keep]
        np.add.at(acc, r, acc[src])
        np.add.at(indeg, r, -1)
        cand = np.unique(r)
        frontier = cand[indeg[cand] == 0]
    return acc


def channel_strength(acc, land, ref_percentile=99.5, exponent=0.45):
    """Normalise drainage area to 0..1, so the trunk stream is 1 and a gully
    is a few percent.  The exponent controls how fast importance falls off
    upstream: 0.45 gives a few dominant rivers with a long tail of creeks."""
    a = np.where(land, acc.reshape(land.shape), 0.0)
    ref = float(np.percentile(a[land], ref_percentile)) if land.any() else 1.0
    return np.clip((a / max(ref, 1.0)) ** exponent, 0.0, 1.0)


def carve(h, acc, depth=240.0, scales=((3.0, 0.42), (10.0, 0.33), (26.0, 0.25)),
          ref_percentile=99.5, floor=1.0):
    """Cut valleys along the drainage network at three scales.

    Blurring the channel-strength field turns a one-pixel river into a valley
    whose width and depth follow the size of the river.  Doing it at three
    radii gives the hierarchy real landscapes have: a broad trough for the
    trunk stream, a narrow notch for the creek that feeds it.

    This replaces explicit stream-power incision.  Stream power is physically
    honest but, discretised on a raster with a stability clamp, it terraces:
    it converges to a staircase.  Blurred carving cannot, because the valley
    walls are smooth by construction.
    """
    land = h > 0
    C = channel_strength(acc, land, ref_percentile)
    field = np.zeros_like(C)
    wsum = 0.0
    for sigma, weight in scales:
        field += weight * blur(C, sigma)
        wsum += weight
    field /= wsum
    field /= max(field.max(), 1e-9)

    out = h.astype(np.float64) - depth * field
    out = np.where(land, np.maximum(out, np.minimum(h, floor)), h)
    return out


def erode(h, iterations=20, rate=0.55, m=0.55, n=1.0, eps=EPS):
    """Stream-power incision.  Kept for comparison; see `carve` for why the
    default pipeline does not use it."""
    h = h.astype(np.float64).copy()
    for _ in range(iterations):
        land = h > 0
        filled = fill_depressions(h, eps=eps)
        rec, slope = receivers(filled, land)
        acc = accumulate(rec, h.size)
        an = acc / max(acc.max(), 1.0)
        E = rate * (an ** m) * (slope.ravel() ** n) * 100.0

        flat = h.ravel()
        ff = filled.ravel()
        keep = rec >= 0
        drop = np.zeros_like(flat)
        drop[keep] = ff[keep] - ff[rec[keep]]
        E = np.minimum(E, 0.45 * np.maximum(drop, 0.0))
        flat -= E
        h = flat.reshape(h.shape)
    return h


def drainage(h, eps=EPS):
    filled = fill_depressions(h, eps=eps)
    rec, slope = receivers(filled, h > 0)
    acc = accumulate(rec, h.size)
    return {
        "filled": filled,
        "receivers": rec,
        "slope": slope,
        "accumulation": acc.reshape(h.shape),
        "lake_depth": np.maximum(filled - h, 0.0) * (h > 0),
    }
