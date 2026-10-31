"""Landmass construction.

Two things that are easy to conflate and must not be:

* *Where the land is* -- a continental potential built from tectonic nuclei
  plus warped noise.  This is a mask, nothing more.
* *How high it is* -- hills, belts and plains, built from noise and shaped by
  distance from the shore.

Conflating them produces the classic procedural-geography tell: a continent
that is a smooth dome.  A dome lit from the north-west reads as a balloon,
not as a landscape.

Output is a signed field in normalised units: positive above sea level,
negative below, zero at the shore.  `spec.meters` scales it to a real
elevation in metres.
"""

from __future__ import annotations

import numpy as np

from .noise import fbm, ridged, smoothstep, warp
from .config import WorldSpec
from .filters import blur


def _grid(spec: WorldSpec):
    w, h = spec.width, spec.height
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
    # x spans [0, aspect], y spans [0, 1]: features stay isotropic in map
    # space regardless of canvas proportions.
    return xs / h, ys / h


def _blobs(rng, spec: WorldSpec):
    """Elliptical nuclei.  Deliberately irregular: real continents are not
    disc-shaped, so the main body is elongated and the companions are pulled
    far enough out to make peninsulas rather than merge into a potato."""
    aspect = spec.aspect
    out = []

    cx = aspect * 0.5 + rng.uniform(-0.05, 0.05) * aspect
    cy = 0.5 + rng.uniform(-0.04, 0.04)
    main_r = 0.22 + rng.uniform(0.0, 0.05)
    stretch = rng.uniform(0.55, 1.15)
    out.append((cx, cy, main_r * stretch, main_r / stretch, rng.uniform(0, np.pi), 1.0))

    for _ in range(max(0, spec.blobs - 1)):
        ang = rng.uniform(0, 2 * np.pi)
        dist = rng.uniform(0.22, 0.40)
        bx = cx + np.cos(ang) * dist * aspect
        by = cy + np.sin(ang) * dist
        r = 0.13 + rng.uniform(0.0, 0.09)
        out.append((bx, by, r * rng.uniform(0.6, 1.7), r / rng.uniform(0.6, 1.7),
                    rng.uniform(0, np.pi), rng.uniform(0.85, 1.0)))

    for _ in range(spec.island_blobs):
        ang = rng.uniform(0, 2 * np.pi)
        dist = rng.uniform(0.44, 0.60)
        bx = float(np.clip(cx + np.cos(ang) * dist * aspect, 0.05, aspect - 0.05))
        by = float(np.clip(cy + np.sin(ang) * dist, 0.07, 0.93))
        r = 0.020 + rng.uniform(0.0, 0.028)
        out.append((bx, by, r * rng.uniform(0.7, 1.5), r / rng.uniform(0.7, 1.5),
                    rng.uniform(0, np.pi), 0.75))
    return out


def _blob_distance(x, y, blobs):
    """Union of rotated-ellipse distance fields; 1.0 on an outline, <1 inside."""
    d = np.full_like(x, 1e9)
    for cx, cy, rx, ry, ang, weight in blobs:
        ca, sa = np.cos(ang), np.sin(ang)
        dx = x - cx
        dy = y - cy
        xr = dx * ca + dy * sa
        yr = -dx * sa + dy * ca
        r = np.sqrt((xr / rx) ** 2 + (yr / ry) ** 2) / weight
        np.minimum(d, r, out=d)
    return d


def _ridge_axes(rng, spec: WorldSpec, blobs):
    aspect = spec.aspect
    n = max(1, spec.blobs)
    cx = float(np.mean([b[0] for b in blobs[:n]]))
    cy = float(np.mean([b[1] for b in blobs[:n]]))
    axes = []
    base = rng.uniform(0, np.pi)
    for i in range(spec.ridge_count):
        ang = base + i * np.pi / max(1, spec.ridge_count) + rng.uniform(-0.45, 0.45)
        length = rng.uniform(0.55, 1.0) * aspect
        ox = cx + rng.uniform(-0.10, 0.10) * aspect
        oy = cy + rng.uniform(-0.12, 0.12)
        dx, dy = np.cos(ang) * length * 0.5, np.sin(ang) * length * 0.5
        axes.append((ox - dx, oy - dy, ox + dx, oy + dy,
                     spec.ridge_width * rng.uniform(0.65, 1.35)))
    return axes


def _segment_distance(x, y, seg):
    x0, y0, x1, y1, _ = seg
    vx, vy = x1 - x0, y1 - y0
    wx, wy = x - x0, y - y0
    t = np.clip((wx * vx + wy * vy) / max(vx * vx + vy * vy, 1e-9), 0.0, 1.0)
    return np.hypot(x - (x0 + t * vx), y - (y0 + t * vy))


def build_base(spec: WorldSpec):
    """Return (height, meta); sea level is 0, land is positive."""
    rng = np.random.default_rng(spec.seed)
    x, y = _grid(spec)
    blobs = _blobs(rng, spec)
    body = 1.0 - _blob_distance(x, y, blobs)

    # --- continental potential: shape only ------------------------------
    warper = warp(spec.seed + 11, amp=1.0, octaves=4)
    wx, wy = warper(x, y, freq=spec.coast_freq * 0.45, strength=0.38)
    coast = fbm(spec.seed + 23, octaves=4)(wx, wy, freq=spec.coast_freq, gain=0.46)

    band = np.exp(-(body / 0.22) ** 2)
    potential = body + spec.coast_noise * band * coast

    # Keep the continent off the edge of the sheet: a map whose land runs off
    # the paper reads as a crop, not as a place.
    aspect = spec.aspect
    edge = np.minimum(np.minimum(x, aspect - x), np.minimum(y, 1.0 - y))
    border = smoothstep(0.0, 0.075, edge)
    potential = potential * (0.25 + 0.75 * border) - (1.0 - border) * 0.75

    # Smooth before thresholding.  Without this the level set grows thin
    # fingers wherever the noise gradient rivals the body gradient.
    potential = blur(potential, max(1.5, spec.width / 150.0), passes=3)

    sea = float(np.quantile(potential, 1.0 - spec.land_fraction))
    relief = potential - sea

    mask = smoothstep(-0.010, 0.010, relief)          # 0 water, 1 land
    inland = smoothstep(0.0, 0.42, relief)            # 0 at the shore

    # --- relief: height only --------------------------------------------
    # Broad undulation, then fine texture.  Both stay low: the height is in
    # the mountains, not in the plains, which is what makes an elevation
    # ramp read as terrain instead of as a plateau.
    hills = (0.038 * fbm(spec.seed + 61, octaves=5)(x, y, freq=6.0, gain=0.52)
             + 0.018 * fbm(spec.seed + 67, octaves=4)(x, y, freq=15.0, gain=0.45))

    axes = _ridge_axes(rng, spec, blobs)
    belt = np.zeros_like(x)
    for seg in axes:
        belt = np.maximum(belt, np.exp(-(_segment_distance(x, y, seg) / max(seg[4], 1e-6)) ** 2))
    belt = belt ** 1.8

    big = ridged(spec.seed + 37, octaves=5)(x, y, freq=spec.mountain_freq * 0.5, gain=0.55)
    fine = ridged(spec.seed + 41, octaves=4)(x, y, freq=spec.mountain_freq * 1.8, gain=0.45)
    raw = 0.70 * big + 0.30 * fine
    # Emphasise the crests smoothly.  A hard threshold here leaves isolated
    # spikes that render as worms; a power curve keeps the range connected.
    mount = raw ** 1.6

    uplift = (0.10 + 0.90 * belt) * inland ** 0.45
    mountains = spec.mountain_amp * 0.95 * uplift * mount

    elev = mask * (0.012 + 0.075 * inland + hills + mountains)
    elev = np.maximum(elev, 0.0)

    # --- sea floor ------------------------------------------------------
    offshore = np.maximum(-relief, 0.0)
    depth = 0.34 * (1.0 - np.exp(-offshore / 0.055))
    depth = depth * (1.0 - mask) * (1.0 + 0.05 * fbm(spec.seed + 71, octaves=3)(x, y, freq=4.0))

    h = (mask * elev - depth) * spec.meters
    h = h.astype(np.float32)

    meta = {
        "blobs": [tuple(map(float, b)) for b in blobs],
        "ridge_axes": [tuple(map(float, s)) for s in axes],
        "sea_level_raw": sea,
        "land_fraction": float((h > 0).mean()),
        "max_height": float(h.max()),
        "mask": mask.astype(np.float32),
        "relief": relief.astype(np.float32),
        "inland": inland.astype(np.float32),
        "belt": belt.astype(np.float32),
    }
    return h, meta


def add_detail(spec: WorldSpec, h):
    """Fine relief, added *after* the drainage has been carved.

    Order matters.  High-frequency noise put in before routing fragments the
    network into thousands of competing puddles; the same noise added
    afterwards gives the surface its grain without touching where the water
    goes.
    """
    x, y = _grid(spec)
    land = h > 0
    shore = smoothstep(0.0, 90.0, h)
    r1 = ridged(spec.seed + 101, octaves=4)(x, y, freq=30.0, gain=0.50)
    r2 = ridged(spec.seed + 103, octaves=3)(x, y, freq=70.0, gain=0.45)
    d = (0.016 * r1 + 0.007 * r2) * spec.meters * spec.detail
    return np.where(land, h + d * shore, h)
