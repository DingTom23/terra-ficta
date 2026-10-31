"""Inks and tints.

Colours are chosen as if printed on toned paper: nothing fully saturated,
nothing fully black, everything a little warm.  A map that uses pure white and
pure black looks like a screenshot; a map that uses bone and sepia looks like
it was pulled out of a drawer.
"""

from __future__ import annotations

import numpy as np

# --- paper --------------------------------------------------------------
PAPER = (0.925, 0.895, 0.820)
PAPER_DARK = (0.870, 0.830, 0.740)
INK = (0.180, 0.150, 0.125)
INK_SOFT = (0.320, 0.280, 0.230)
INK_WATER = (0.230, 0.330, 0.400)
INK_RIVER = (0.240, 0.360, 0.430)

# --- land: elevation in metres -----------------------------------------
HYPSOMETRIC = [
    (0.0, (0.720, 0.775, 0.565)),
    (150.0, (0.775, 0.790, 0.535)),
    (400.0, (0.820, 0.785, 0.490)),
    (800.0, (0.845, 0.755, 0.450)),
    (1300.0, (0.845, 0.715, 0.420)),
    (1900.0, (0.790, 0.625, 0.400)),
    (2500.0, (0.685, 0.530, 0.390)),
    (3000.0, (0.590, 0.475, 0.400)),
    (3600.0, (0.700, 0.665, 0.635)),
    (4300.0, (0.870, 0.865, 0.850)),
]

# --- sea: depth in metres (negative elevation) --------------------------
BATHYMETRIC = [
    (0.0, (0.720, 0.800, 0.795)),
    (40.0, (0.665, 0.765, 0.775)),
    (150.0, (0.590, 0.710, 0.740)),
    (500.0, (0.505, 0.640, 0.690)),
    (1200.0, (0.430, 0.575, 0.645)),
    (2200.0, (0.375, 0.520, 0.605)),
]


def _ramp(values, stops):
    """Piecewise-linear colour ramp; `stops` is [(x, rgb), ...] ascending."""
    values = np.asarray(values, dtype=np.float64)
    out = np.zeros(values.shape + (3,), dtype=np.float64)
    xs = [s[0] for s in stops]
    out[..., :] = stops[0][1]
    for i in range(len(stops) - 1):
        x0, c0 = stops[i]
        x1, c1 = stops[i + 1]
        t = np.clip((values - x0) / max(x1 - x0, 1e-9), 0.0, 1.0)[..., None]
        seg = np.array(c0) * (1.0 - t) + np.array(c1) * t
        out = np.where((values >= x0)[..., None], seg, out)
    return out


def hypsometric(elev_m):
    return _ramp(elev_m, HYPSOMETRIC)


def bathymetric(depth_m):
    return _ramp(depth_m, BATHYMETRIC)


def hexc(rgb):
    return "#%02x%02x%02x" % tuple(int(round(c * 255)) for c in rgb)
