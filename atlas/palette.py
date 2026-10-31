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
    (0.0, (0.868, 0.862, 0.786)),
    (150.0, (0.856, 0.846, 0.752)),
    (400.0, (0.842, 0.822, 0.706)),
    (800.0, (0.832, 0.796, 0.652)),
    (1300.0, (0.822, 0.764, 0.596)),
    (1900.0, (0.788, 0.706, 0.548)),
    (2500.0, (0.726, 0.632, 0.508)),
    (3000.0, (0.652, 0.562, 0.478)),
    (3600.0, (0.706, 0.678, 0.648)),
    (4300.0, (0.892, 0.888, 0.874)),
]

# --- sea: depth in metres (negative elevation) --------------------------
# A period plate does not paint the sea blue; it leaves it almost paper and
# only lets the blue in where the water gets deep.  That is what puts the
# land forward: the coast is the contrast, not the ocean.
BATHYMETRIC = [
    (0.0, (0.906, 0.906, 0.862)),
    (150.0, (0.882, 0.894, 0.870)),
    (500.0, (0.840, 0.868, 0.868)),
    (1000.0, (0.788, 0.834, 0.850)),
    (1800.0, (0.716, 0.786, 0.820)),
    (2800.0, (0.640, 0.730, 0.784)),
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
