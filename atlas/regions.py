"""Provinces.

A province needs a boundary that means something.  Arbitrary clusters drawn on
a map give arbitrary borders; what a cartographer wants is a border that follows
a feature.  The cheapest such feature is already in the data: the drainage
divide.

So provinces are built from watersheds.  Every land cell belongs to the basin of
one river mouth, found by pointer-jumping down the D8 receiver graph.  River
mouths are then clustered geographically, weighted by how much land they drain,
and each basin joins the province of its mouth.  Borders therefore run along
ridges, which is where people have always put them.
"""

from __future__ import annotations

import numpy as np


def roots(rec, n):
    """Terminal outlet of every cell, by pointer jumping.

    ``rec`` is a flat array of receiver indices (-1 at outlets).  Repeatedly
    following the pointer halves the remaining path, so 24 rounds resolve a
    forest of any depth up to 16 million cells.
    """
    p = np.where(rec >= 0, rec, np.arange(n, dtype=np.int64))
    for _ in range(24):
        p = p[p]
    return p


def _weighted_kmeans(points, weights, k, rng, iters=25):
    k = min(k, len(points))
    idx = rng.choice(len(points), size=k, replace=False)
    cent = points[idx].astype(np.float64)
    lab = np.zeros(len(points), dtype=np.int64)
    for _ in range(iters):
        d = ((points[:, None, :] - cent[None, :, :]) ** 2).sum(-1)
        lab = d.argmin(1)
        for j in range(k):
            m = lab == j
            if m.any():
                w = weights[m]
                cent[j] = (points[m] * w[:, None]).sum(0) / max(w.sum(), 1e-9)
    return lab


def provinces(height, rec, k=5, seed=0):
    """Return (province index per cell, -1 for water, and the mouth list)."""
    h, w = height.shape
    n = h * w
    land = (height > 0).ravel()
    r = roots(rec, n)

    outlets, counts = np.unique(r[land], return_counts=True)
    keep = counts > 40
    outlets, counts = outlets[keep], counts[keep]
    if len(outlets) == 0:
        return np.full((h, w), -1, dtype=np.int16), []

    oy, ox = np.divmod(outlets, w)
    pts = np.stack([ox, oy], 1).astype(np.float64)

    rng = np.random.default_rng(seed * 7919 + 31)
    if len(outlets) <= k:
        mouth_region = np.arange(len(outlets))
    else:
        mouth_region = _weighted_kmeans(pts, counts.astype(np.float64), k, rng)

    region_of_root = np.full(n, -1, dtype=np.int16)
    region_of_root[outlets] = mouth_region.astype(np.int16)

    flat = region_of_root[r]
    flat = np.where(land, flat, -1).astype(np.int16)

    mouths = [(int(ox[i]), int(oy[i]), int(counts[i])) for i in range(len(outlets))]
    grid = _grow(flat.reshape(h, w), (height > 0), limit=160)
    return grid, mouths


def _grow(region, land, limit=160):
    """Give every land cell a province.

    A cell drains to a mouth outside the top-k, so it never got a label; on the
    political plate those cells printed as pale holes along the coasts and in
    the corners.  Grow the labels outward until nothing is left blank.
    """
    out = region.copy()
    for _ in range(limit):
        empty = land & (out < 0)
        if not empty.any():
            break
        nb = np.full(out.shape, -1, dtype=out.dtype)
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1),
                       (1, 1), (1, -1), (-1, 1), (-1, -1)):
            sh = np.roll(np.roll(out, dy, 0), dx, 1)
            nb = np.where((nb < 0) & (sh >= 0), sh, nb)
        out = np.where(empty & (nb >= 0), nb, out)
    return out


def borders(region, land):
    """Cells where the province changes: the drawn boundary."""
    out = np.zeros(region.shape, dtype=bool)
    for dy, dx in ((0, 1), (1, 0)):
        a = region
        b = np.roll(region, (-dy, -dx), axis=(0, 1))
        m = (a != b) & land & (a >= 0) & (b >= 0)
        out |= m
    return out


def areas(region):
    """Cells per province, and the centroid of each."""
    ids = [i for i in np.unique(region) if i >= 0]
    out = {}
    for i in ids:
        ys, xs = np.where(region == i)
        out[int(i)] = (len(ys), float(xs.mean()), float(ys.mean()))
    return out
