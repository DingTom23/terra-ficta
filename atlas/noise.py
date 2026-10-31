"""Deterministic gradient noise and its usual descendants.

Everything here is seeded and pure: the same seed and the same coordinate
always produce the same value, on any machine, forever.  That property is the
whole point -- a map that cannot be reproduced is a drawing, not a place.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "GradientNoise",
    "fbm",
    "ridged",
    "warp",
    "smoothstep",
    "clamp",
]


def clamp(x, lo=0.0, hi=1.0):
    return np.clip(x, lo, hi)


def smoothstep(edge0, edge1, x):
    """Hermite ramp between two edges."""
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


class GradientNoise:
    """Classic 2-D gradient noise (Perlin), vectorised over numpy arrays."""

    def __init__(self, seed: int):
        rng = np.random.default_rng(seed)
        p = rng.permutation(256).astype(np.int64)
        self.perm = np.concatenate([p, p])
        ang = rng.random(256) * (2.0 * np.pi)
        # Unit gradients on a circle: no axis bias, no degenerate corners.
        self.gx = np.cos(ang).astype(np.float64)
        self.gy = np.sin(ang).astype(np.float64)

    def __call__(self, x, y):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        xi = np.floor(x)
        yi = np.floor(y)
        xf = x - xi
        yf = y - yi

        X = xi.astype(np.int64) & 255
        Y = yi.astype(np.int64) & 255

        u = xf * xf * xf * (xf * (xf * 6.0 - 15.0) + 10.0)
        v = yf * yf * yf * (yf * (yf * 6.0 - 15.0) + 10.0)

        perm = self.perm
        A = perm[X] + Y
        B = perm[X + 1] + Y

        h00 = perm[A]
        h01 = perm[A + 1]
        h10 = perm[B]
        h11 = perm[B + 1]

        gx, gy = self.gx, self.gy

        n00 = gx[h00] * xf + gy[h00] * yf
        n10 = gx[h10] * (xf - 1.0) + gy[h10] * yf
        n01 = gx[h01] * xf + gy[h01] * (yf - 1.0)
        n11 = gx[h11] * (xf - 1.0) + gy[h11] * (yf - 1.0)

        nx0 = n00 + u * (n10 - n00)
        nx1 = n01 + u * (n11 - n01)
        return (nx0 + v * (nx1 - nx0)) * 1.45


class OctaveSet:
    """A bundle of gradient noise instances, one per octave.

    Each octave gets its own permutation so that octaves are statistically
    independent rather than scaled copies of one another.
    """

    def __init__(self, seed: int, octaves: int):
        self.layers = [GradientNoise(seed * 7919 + i * 104729 + 17) for i in range(octaves)]

    def __call__(self, x, y, freq=1.0, lacunarity=2.0, gain=0.5, rotation=0.0):
        total = np.zeros_like(x, dtype=np.float64)
        amp = 1.0
        norm = 0.0
        f = freq
        ca, sa = np.cos(rotation), np.sin(rotation)
        for layer in self.layers:
            if rotation:
                xr = x * ca - y * sa
                yr = x * sa + y * ca
            else:
                xr, yr = x, y
            total += amp * layer(xr * f, yr * f)
            norm += amp
            amp *= gain
            f *= lacunarity
        return total / norm


def fbm(seed: int, octaves: int = 6):
    """Return a callable fBm field over (x, y)."""
    octs = OctaveSet(seed, octaves)

    def field(x, y, freq=1.0, lacunarity=2.0, gain=0.5, rotation=0.0):
        return octs(x, y, freq, lacunarity, gain, rotation)

    return field


def ridged(seed: int, octaves: int = 6):
    """Ridged multifractal -- the shape of mountain crests.

    Returns values in roughly [0, 1] where 1 is a knife-edge ridge.
    """
    octs = OctaveSet(seed, octaves)

    def field(x, y, freq=1.0, lacunarity=2.05, gain=0.5, sharpness=2.0):
        total = np.zeros_like(x, dtype=np.float64)
        weight = np.ones_like(x, dtype=np.float64)
        f = freq
        amp = 1.0
        norm = 0.0
        for layer in octs.layers:
            n = 1.0 - np.abs(layer(x * f, y * f))
            n = n * n
            n = n ** (sharpness * 0.5)
            total += n * weight * amp
            weight = np.clip(n * 1.6, 0.0, 1.0)
            norm += amp
            amp *= gain
            f *= lacunarity
        return np.clip(total / norm, 0.0, 1.0)

    return field


def warp(seed: int, amp: float = 1.0, octaves: int = 4):
    """Domain warping: displace sample coordinates by another noise field.

    This is what turns a smooth blob into something that looks eroded by
    weather rather than drawn by a formula.
    """
    octs = OctaveSet(seed, octaves)

    def field(x, y, freq=1.0, strength=1.0, gain=0.5):
        qx = octs(x, y, freq, 2.0, gain, 0.0)
        qy = octs(x + 5.2, y + 1.3, freq, 2.0, gain, 0.0)
        rx = octs(x + 1.7 * qx, y + 9.2 * qy, freq, 2.0, gain, 0.0)
        ry = octs(x + 8.3 * qy, y + 2.8 * qx, freq, 2.0, gain, 0.0)
        return x + amp * strength * rx, y + amp * strength * ry

    return field
