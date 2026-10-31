"""Small numpy filters.  No scipy in this environment, so we roll our own."""

from __future__ import annotations

import numpy as np


def _box_1d(a, r, axis):
    if r < 1:
        return a
    pad = [(0, 0), (0, 0)]
    pad[axis] = (r + 1, r)
    ap = np.pad(a, pad, mode="reflect")
    c = np.cumsum(ap, axis=axis)
    if axis == 0:
        out = c[2 * r + 1:] - c[:-(2 * r + 1)]
    else:
        out = c[:, 2 * r + 1:] - c[:, :-(2 * r + 1)]
    return out / (2 * r + 1)


def blur(a, sigma, passes: int = 3):
    """Approximate Gaussian blur by repeated box blurs."""
    if sigma <= 0:
        return a
    # Radius for `passes` box blurs approximating a Gaussian of this sigma.
    r = max(1, int(round(sigma * np.sqrt(12.0 / passes + 1.0) / 2.0 - 0.5)))
    out = a.astype(np.float64)
    for _ in range(passes):
        out = _box_1d(out, r, 0)
        out = _box_1d(out, r, 1)
    return out


def blur_fast(a, sigma):
    return blur(a, sigma, passes=2)
