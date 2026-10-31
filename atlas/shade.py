"""Hillshading and related relief treatments."""

from __future__ import annotations

import numpy as np


def surface_normal(h, z_factor=1.0):
    gy, gx = np.gradient(h.astype(np.float64) * z_factor)
    nx = -gx
    ny = -gy
    nz = np.ones_like(h, dtype=np.float64)
    norm = np.sqrt(nx * nx + ny * ny + nz * nz)
    return nx / norm, ny / norm, nz / norm


def hillshade(h, z_factor=260.0, azimuth=315.0, altitude=42.0):
    """Lambertian shading with the light from the north-west.

    The convention is not arbitrary: cartographers settled on NW illumination
    centuries ago because the human visual system reads it as relief rather
    than as a hollow.
    """
    nx, ny, nz = surface_normal(h, z_factor)
    az = np.radians(azimuth)
    alt = np.radians(altitude)
    lx = np.sin(az) * np.cos(alt)
    ly = -np.cos(az) * np.cos(alt)
    lz = np.sin(alt)
    return np.clip(nx * lx + ny * ly + nz * lz, 0.0, 1.0)


def curvature(h, z_factor=1.0):
    """Laplacian -- used to darken valleys and lighten crests."""
    z = h.astype(np.float64) * z_factor
    lap = (
        np.roll(z, 1, 0) + np.roll(z, -1, 0) + np.roll(z, 1, 1) + np.roll(z, -1, 1) - 4.0 * z
    )
    return lap


def ambient_occlusion(h, z_factor=1.0, radius=6):
    """Cheap multi-scale occlusion: how much of the sky a cell can see."""
    z = h.astype(np.float64) * z_factor
    acc = np.zeros_like(z)
    count = 0
    for r in (radius, radius * 2, radius * 4):
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)):
            shifted = np.roll(np.roll(z, dy * r, 0), dx * r, 1)
            acc += np.clip(shifted - z, 0.0, None)
            count += 1
    occ = acc / max(count, 1)
    return 1.0 - np.clip(occ, 0.0, 1.0)
