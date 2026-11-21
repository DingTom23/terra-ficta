"""Coastal deposition: what rivers and longshore drift leave behind.

The landmass is cut from a potential field, and a potential field has no
memory of what arrived where.  So every coast in the book is the same
serrated edge -- purely erosional, as if no river had ever reached the sea.
Real coasts are shaped by what is brought to them as much as by what is taken
away:

* a large river builds a **delta** -- a fan of new land at its mouth, with the
  channels splitting across it;
* sand moved along the shore by waves builds a **spit**, a long low ridge
  parallel to the coast;
* the water the spit encloses becomes a **lagoon**.

This module adds those three things to a finished height field, after the
drainage is known (a delta needs a river) and before the final routing (so the
channels find their way across the new ground).
"""

from __future__ import annotations

import numpy as np

from .config import WorldSpec


def _shift(mask, dy, dx):
    out = np.zeros_like(mask)
    h, w = mask.shape
    ys = slice(max(0, dy), h + min(0, dy))
    xs = slice(max(0, dx), w + min(0, dx))
    yd = slice(max(0, -dy), h + min(0, -dy))
    xd = slice(max(0, -dx), w + min(0, -dx))
    out[ys, xs] = mask[yd, xd]
    return out


def _dilate(mask, n=1):
    out = mask.copy()
    for _ in range(n):
        grown = out.copy()
        grown |= _shift(out, 1, 0)
        grown |= _shift(out, -1, 0)
        grown |= _shift(out, 0, 1)
        grown |= _shift(out, 0, -1)
        out = grown
    return out


def _seaward(height, x, y, radius=3):
    """The direction of the water from a coastal cell, as a unit vector."""
    h, w = height.shape
    sx = sy = 0.0
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if not dx and not dy:
                continue
            xx, yy = x + dx, y + dy
            if not (0 <= xx < w and 0 <= yy < h):
                continue
            if height[yy, xx] <= 0.0:
                wgt = 1.0 / (1.0 + dx * dx + dy * dy)
                sx += dx * wgt
                sy += dy * wgt
    n = float(np.hypot(sx, sy))
    if n < 1e-6:
        return None
    return sx / n, sy / n


def _mouths(height, acc, limit=5):
    """Land cells that open onto the sea, ranked by the water they carry."""
    h, w = height.shape
    sea = height <= 0.0
    touch = sea.copy()
    touch = _shift(touch, 1, 0) | _shift(touch, -1, 0) | _shift(touch, 0, 1) | _shift(touch, 0, -1)
    edge = np.zeros_like(sea)
    edge[:2, :] = edge[-2:, :] = edge[:, :2] = edge[:, -2:] = True

    cand = (~sea) & touch & ~edge
    if not cand.any():
        return []
    a = np.where(cand, acc, 0.0)
    amax = float(a.max())
    if amax <= 0.0:
        return []

    order = np.argsort(-a.ravel())[: limit * 40]
    taken = []
    gap = max(6.0, min(h, w) / 22.0)
    for idx in order:
        if a.ravel()[idx] <= 0.0:
            break
        y, x = divmod(int(idx), w)
        if any((x - tx) ** 2 + (y - ty) ** 2 < gap * gap for tx, ty, _ in taken):
            continue
        d = _seaward(height, x, y)
        if d is None:
            continue
        taken.append((x, y, a[y, x]))
        if len(taken) >= limit:
            break
    return taken


def _delta(height, acc, spec, x, y, power, amax, rng):
    """A fan of new land at a river mouth, largest where the river is."""
    h, w = height.shape
    d = _seaward(height, x, y)
    if d is None:
        return
    sx, sy = d
    sea = height <= 0.0

    # Everything is measured in cells, so it has to scale with the sheet: a
    # delta is a fixed distance in kilometres, not a fixed count of pixels.
    unit = min(h, w) / 450.0
    r = (5.0 + 17.0 * float(np.sqrt(max(0.0, power) / amax))) * unit
    r *= 0.82 + 0.36 * rng.random()
    reach = int(np.ceil(2.0 * r))
    x0, x1 = max(0, x - reach), min(w, x + reach + 1)
    y0, y1 = max(0, y - reach), min(h, y + reach + 1)
    if x1 - x0 < 3 or y1 - y0 < 3:
        return

    ys, xs = np.mgrid[y0:y1, x0:x1]
    dx = (xs - x).astype(np.float64)
    dy = (ys - y).astype(np.float64)
    dist = np.hypot(dx, dy)

    # Only downstream of the mouth: a delta does not grow back into the river.
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = (dx * sx + dy * sy) / np.maximum(dist, 1e-9)
    fan = cos > 0.15

    # A delta is lobate, not a disc: the radius is modulated by two low
    # harmonics of the bearing from the mouth, and the demarais takes whatever
    # shape the river has been building for centuries.
    ang = np.arctan2(dy, dx)
    k1, k2 = rng.uniform(0.0, 6.283, 2)
    lob = 1.0 + 0.24 * np.sin(3.0 * ang + k1) + 0.13 * np.sin(5.0 * ang + k2)
    taper = np.exp(-(dist / np.maximum(r * lob, 1.0)) ** 2)

    # Deltas are built on a shelf.  Where the sea floor falls away at the mouth
    # the river has nothing to stand on and the fan dies at the shore.
    depth = np.maximum(-height[y0:y1, x0:x1], 0.0)
    shelf = np.clip(1.0 - depth / 90.0, 0.0, 1.0)

    # The fan has to *end*: a constant floor of positive height turns the
    # whole search disc into land and the delta becomes a headland.
    new = 9.0 * taper * (0.35 + 0.65 * shelf) - 0.35
    patch = sea[y0:y1, x0:x1] & fan & (dist < 1.9 * r) & (new > 0.0)
    if not patch.any():
        return
    sub = height[y0:y1, x0:x1]
    height[y0:y1, x0:x1] = np.where(patch, np.maximum(sub, new), sub)


def _spit(height, spec, rng, tries=3):
    """A low ridge of sand parallel to the shore, with a lagoon behind it."""
    h, w = height.shape
    sea = height <= 0.0
    land = ~sea
    if land.sum() < 200:
        return

    # Shallow water only: a spit is built where the sea is doing work, and
    # never out in deep water.
    for _ in range(tries):
        unit = min(h, w) / 450.0
        k = max(3, int(rng.integers(6, 15) * unit))
        # The *shell* at distance k, not everything within k: dilating a band
        # and masking with the sea keeps every layer it has already passed
        # through, so the "ring" comes out as a filled annulus and the spit
        # is a wedge of new land instead of a line of sand.
        inner = land
        for _ in range(max(1, k - 1)):
            inner = _dilate(inner, 1)
        ring = _dilate(inner, 1) & sea & ~inner
        if not ring.any():
            return
        ys, xs = np.where(ring)
        i = int(rng.integers(0, len(xs)))
        cx, cy = int(xs[i]), int(ys[i])

        # Follow the ring a little way: a spit is an arc, not a blob.
        span = max(8, int(rng.integers(18, 34) * unit))
        ys2, xs2 = np.where(ring)
        keep = (xs2 - cx) ** 2 + (ys2 - cy) ** 2 <= span * span

        height[ys2[keep], xs2[keep]] = np.maximum(
            height[ys2[keep], xs2[keep]],
            1.0 + 1.8 * rng.random(int(keep.sum())),
        )

        # The water it encloses is shallower than the open sea: a lagoon.
        # `land` is the mask from before the ridge was raised, so it has to be
        # read again -- otherwise the lagoon step finds the new spit inside
        # its own band and drowns what it has just built.
        sea_now = height <= 0.0
        inner = _dilate(ring, 2) & sea_now
        ys3, xs3 = np.where(inner)
        if len(xs3):
            near = (xs3 - cx) ** 2 + (ys3 - cy) ** 2 <= (span * 1.15) ** 2
            height[ys3[near], xs3[near]] = np.minimum(
                height[ys3[near], xs3[near]], -2.0
            )


def deposit(height, acc, spec: WorldSpec, deltas=4, spits=2):
    """Add deltas and spits to a finished height field, in place.

    Both raise cells that were sea into land, which is exactly what they are:
    deposition is the only process in this engine that creates land after the
    sea level has been set.
    """
    if deltas <= 0 and spits <= 0:
        return height

    rng = np.random.default_rng(spec.seed + 4021)
    a = np.asarray(acc, dtype=np.float64)
    land = height > 0.0
    a = np.where(land, a, 0.0)
    amax = float(a.max()) if a.size else 0.0

    if amax > 0.0 and deltas > 0:
        for x, y, power in _mouths(height, a, limit=deltas):
            _delta(height, a, spec, x, y, power, amax, rng)

    if spits > 0:
        for _ in range(spits):
            _spit(height, spec, rng)

    return height
