"""Rendering: turning a height field into something worth framing.

The look aimed at is a nineteenth-century relief atlas -- toned paper, muted
hypsometric bands, fine engraved contours, coastal hachures that fade with
depth, and a dark, slightly irregular coastline.  Nothing fully saturated,
nothing fully black.
"""

from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .config import WorldSpec
from .climate import BIOMES
from .filters import blur
from .noise import fbm, smoothstep
from .palette import (PAPER, PAPER_DARK, INK, INK_SOFT, INK_WATER, INK_RIVER,
                      hypsometric, bathymetric)
from .shade import hillshade, ambient_occlusion


def upsample(a, w, h, mode=Image.BICUBIC):
    im = Image.fromarray(a.astype(np.float32), mode="F")
    return np.asarray(im.resize((w, h), mode), dtype=np.float32)


def upsample_rgb(rgb, w, h, mode=Image.BILINEAR):
    out = np.empty((h, w, rgb.shape[2]), dtype=np.float32)
    for c in range(rgb.shape[2]):
        out[..., c] = upsample(rgb[..., c], w, h, mode)
    return out


def _mix(dst, src, alpha):
    """Composite src over dst with an alpha array (h, w) or (h, w, 1)."""
    if alpha.ndim == 2:
        alpha = alpha[..., None]
    return dst * (1.0 - alpha) + src * alpha


def paper(size, seed=3, tone=1.0):
    """Aged paper: broad mottling, fine grain, and a soft vignette."""
    w, h = size
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    x = xs / h
    y = ys / h

    mottle = fbm(seed, octaves=5)(x, y, freq=2.2, gain=0.55)
    grain = fbm(seed + 7, octaves=3)(x, y, freq=180.0, gain=0.5)
    fibre = fbm(seed + 13, octaves=2)(x * 1.0, y * 6.0, freq=90.0, gain=0.5)

    base = np.array(PAPER, dtype=np.float32)
    lum = 1.0 + 0.030 * mottle + 0.018 * grain + 0.010 * fibre
    rgb = base[None, None, :] * lum[..., None]

    # Vignette: the edges of an old sheet are darker.
    cx, cy = w * 0.5, h * 0.5
    r = np.sqrt(((xs - cx) / (w * 0.5)) ** 2 + ((ys - cy) / (h * 0.5)) ** 2)
    vig = 1.0 - 0.10 * np.clip(r - 0.45, 0.0, None) ** 1.6
    rgb = rgb * vig[..., None] * tone
    return np.clip(rgb, 0.0, 1.0)


def coastal_hachures(depth, spacing=26.0, max_depth=520.0, width=1.1):
    """Engraved lines that follow the shore and fade into deep water.

    Antique charts shaded the sea by drawing bands parallel to the coast,
    closer together in the shallows.  Depth contours give the same thing for
    free, because the shelf slopes gently away from land.
    """
    d = np.maximum(depth, 0.0)
    f = (d / spacing) % 1.0
    dist_m = np.minimum(f, 1.0 - f) * spacing
    gy, gx = np.gradient(d.astype(np.float32))
    grad = np.hypot(gx, gy) + 1e-4
    dist_px = dist_m / grad
    alpha = np.clip(1.0 - dist_px / width, 0.0, 1.0)
    fade = np.clip(1.0 - d / max_depth, 0.0, 1.0) ** 0.7
    return alpha * fade


def contours(elev, interval=100.0, width=0.9, index_every=5, index_width=1.7):
    """Elevation contours, analytically anti-aliased from the gradient.

    The gradient is floored: on dead-flat ground the true spacing tends to
    infinity, and an unclamped value renders the occasional straight scratch
    across a plain instead of a contour.
    """
    e = np.maximum(elev, 0.0)
    f = (e / interval) % 1.0
    dist_m = np.minimum(f, 1.0 - f) * interval
    gy, gx = np.gradient(e.astype(np.float32))
    grad = np.maximum(np.hypot(gx, gy), 1.2)
    dist_px = dist_m / grad

    alpha = np.clip(1.0 - dist_px / width, 0.0, 1.0)
    band = (e / (interval * index_every)) % 1.0
    idx_dist = np.minimum(band, 1.0 - band) * interval * index_every
    idx_px = idx_dist / grad
    idx_alpha = np.clip(1.0 - idx_px / index_width, 0.0, 1.0)
    return alpha, idx_alpha


def render_terrain(world, scale=2, z_factor=340.0, contour_interval=200.0):
    spec = world.spec
    hh, ww = world.height.shape
    RW, RH = ww * scale, hh * scale

    h = upsample(world.height, RW, RH)
    hn = h / spec.meters

    shade = hillshade(hn, z_factor=z_factor * scale, azimuth=315.0, altitude=40.0)
    ao = ambient_occlusion(hn, z_factor=z_factor * scale, radius=4 * scale)

    gy, gx = np.gradient(h.astype(np.float32))
    slope = np.hypot(gx, gy)
    # Engraved darkening: steep ground goes dark whatever the light does.
    # This is what makes a relief plate read as cut metal rather than as a
    # photograph of a sandpit.
    engrave = np.clip(slope / 42.0, 0.0, 1.0) ** 0.85

    land_m = upsample((world.height > 0).astype(np.float32), RW, RH, Image.BILINEAR)
    land_m = np.clip(land_m, 0.0, 1.0)

    sea_rgb = bathymetric(np.maximum(-h, 0.0))
    land_rgb = hypsometric(np.maximum(h, 0.0))
    land_rgb = land_rgb * (0.52 + 0.74 * shade * (0.74 + 0.26 * ao))[..., None]
    land_rgb = land_rgb * (1.0 - 0.34 * engrave)[..., None]

    # Biomes colour the lowlands and fade out at the treeline, where the
    # hypsometric ramp takes over again.  Above about 1400 m the rock is the
    # land cover, so a vegetation tint there would be a lie.
    biome = world.biome
    if biome is not None:
        b_rgb = np.zeros((world.height.shape[0], world.height.shape[1], 3), dtype=np.float32)
        for k, (_name, colour) in BIOMES.items():
            b_rgb[biome == k] = colour
        b_hi = upsample_rgb(b_rgb, RW, RH)
        a = (0.52 * np.exp(-np.maximum(h, 0.0) / 1700.0) * land_m)[..., None]
        land_rgb = land_rgb * (1.0 - a) + b_hi * a

    img = sea_rgb * (1.0 - land_m[..., None]) + land_rgb * land_m[..., None]

    # --- ink ------------------------------------------------------------
    ink = np.array(INK, dtype=np.float32)

    hach = coastal_hachures(np.maximum(-h, 0.0), spacing=13.0, max_depth=1600.0)
    hach = hach * (1.0 - land_m)
    img = _mix(img, np.array(INK_WATER, dtype=np.float32), hach * 0.50)

    alpha, idx = contours(np.maximum(h, 0.0), interval=contour_interval)
    img = _mix(img, ink, alpha * 0.34 * land_m)
    img = _mix(img, ink, idx * 0.46 * land_m)

    return img, h, land_m


def draw_rivers(img, world, scale=2, min_cells=400, colour=INK_RIVER, alpha=0.85):
    """Rivers as tapered polylines along the D8 receiver graph."""
    acc = world.accumulation
    rec = world.drainage["receivers"]
    hh, ww = acc.shape
    RH, RW = hh * scale, ww * scale

    mask = Image.new("L", (RW, RH), 0)
    d = ImageDraw.Draw(mask)

    ys, xs = np.where(acc > min_cells)
    order = np.argsort(-acc[ys, xs])
    ys, xs = ys[order], xs[order]
    amax = float(acc.max())

    for y, x in zip(ys, xs):
        r = rec[y * ww + x]
        if r < 0:
            continue
        ry, rx = divmod(int(r), ww)
        a = acc[y, x]
        w = 1.0 + 2.6 * (np.log1p(a / min_cells) / np.log1p(amax / min_cells))
        w = max(1.0, w * scale * 0.62)
        d.line([(x * scale, y * scale), (rx * scale, ry * scale)],
               fill=255, width=int(round(w)))

    a = np.asarray(mask, dtype=np.float32)[..., None] / 255.0
    return _mix(img, np.array(colour, dtype=np.float32), a * alpha)


def _morph(mask, r, mode):
    """Binary erosion ('min') or dilation ('max') with a 3x3 element, r times."""
    a = mask
    for _ in range(r):
        m = a.copy()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                sh = np.roll(np.roll(a, dy, 0), dx, 1)
                m = (m & sh) if mode == "min" else (m | sh)
        a = m
    return a


def draw_lakes(img, world, scale=2, min_depth=6.0, min_size=4.0, colour=INK_WATER):
    """Lakes, with the puddles dropped.

    The terrain is full of shallow closed basins; rendering every one of them
    gives a continent that looks diseased.  A blurred occupancy test keeps
    only water wide enough to be worth drawing -- which is also what a
    cartographer would do: a pond two millimetres across has no business on a
    sheet at this scale.
    """
    lake = (world.lake_depth > min_depth)
    if not lake.any():
        return img
    lake = lake & (blur(lake.astype(np.float32), min_size) > 0.45)
    if not lake.any():
        return img

    RH, RW = lake.shape[0] * scale, lake.shape[1] * scale
    m = upsample(lake.astype(np.float32), RW, RH, Image.BILINEAR)
    m = np.clip(m * 1.6, 0.0, 1.0)[..., None]
    return _mix(img, np.array(colour, dtype=np.float32) * 0.9, m * 0.55)


def draw_coastline(img, land_m, scale=2, width=0.42, strength=0.88):
    """Ink the shore.

    `land_m` is a bilinear mask, so the boundary is wherever it crosses 0.5.
    The band must be narrow: a generous falloff here inks the entire ocean,
    which is exactly the kind of bug that makes a whole map look muddy.
    """
    edge = np.clip(1.0 - np.abs(land_m - 0.5) / width, 0.0, 1.0)
    return _mix(img, np.array(INK, dtype=np.float32), edge * strength)


def finish_paper(img, world, scale=2, paper_tone=1.0):
    """Lay the printed image onto the sheet."""
    base = paper((img.shape[1], img.shape[0]), seed=world.spec.seed + 5, tone=paper_tone)
    # Texture only modulates luminance; tinting by the paper's hue would wash
    # the blues out of the water and turn the whole sheet to mud.
    tex = base.mean(axis=2) / max(float(base.mean()), 1e-6)
    img = img * (0.95 + 0.05 * tex)[..., None]
    img = img * 0.92 + base * 0.08
    return np.clip(img, 0.0, 1.0)


def draw_borders(img, region, land, scale=2, colour=(0.46, 0.30, 0.24), alpha=0.62,
                 dash=3, gap=3, weight=1):
    """Provincial boundaries, drawn as a dotted rule along the watershed.

    A raster boundary cannot be stroked with a real dash pattern without
    tracing it, so the line is modulated by a diagonal comb instead: on any
    boundary the comb crosses it repeatedly and leaves evenly spaced dots.
    """
    if region is None:
        return img
    b = np.zeros(region.shape, dtype=bool)
    for dy, dx in ((0, 1), (1, 0)):
        shifted = np.roll(region, (-dy, -dx), axis=(0, 1))
        b |= (region != shifted) & land & (region >= 0) & (shifted >= 0)

    for _ in range(weight):
        grown = b.copy()
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            grown |= np.roll(np.roll(b, dy, 0), dx, 1)
        b = grown

    # A border that runs along the shore is redundant with the coastline and
    # reads as a second, smudged coast.  Keep boundaries inland.
    near = ~land
    for _ in range(3):
        grown = near.copy()
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            grown |= np.roll(np.roll(near, dy, 0), dx, 1)
        near = grown
    b = b & ~near

    hh, ww = b.shape
    RH, RW = hh * scale, ww * scale
    m = upsample(b.astype(np.float32), RW, RH, Image.BILINEAR)
    m = np.clip(m * 1.4, 0.0, 1.0)

    ys, xs = np.mgrid[0:RH, 0:RW]
    comb = ((xs + ys) % (dash + gap)) < dash
    m = m * comb

    c = np.array(colour, dtype=np.float32)
    return _mix(img, c, m[..., None] * alpha)


def draw_roads(img, roads, scale=2, colour=(0.40, 0.27, 0.18), alpha=0.78, width=1.6):
    """Roads: a thin warm line, lighter than a river and never straight."""
    if not roads:
        return img
    RH, RW = img.shape[0], img.shape[1]
    mask = Image.new("L", (RW, RH), 0)
    d = ImageDraw.Draw(mask)
    for path in roads:
        if len(path) < 2:
            continue
        pts = [(x * scale, y * scale) for x, y in path]
        d.line(pts, fill=255, width=max(1, int(round(width * scale))),
               joint="curve")
    a = np.asarray(mask, dtype=np.float32)[..., None] / 255.0
    c = np.array(colour, dtype=np.float32)
    return _mix(img, c, a * alpha)


def draw_hachures(img, world, scale=2, spacing=3, colour=None, alpha=0.55,
                  lo_pct=68.0, hi_pct=97.0):
    """Engraved slope strokes.

    Hachures are the one thing that separates an engraved plate from a shaded
    relief.  They are short strokes drawn down the fall line, longer and denser
    where the ground is steeper, and they stop on the flats -- which is why an
    old map has soft plains and hatched mountains.

    Drawn as real segments rather than faked with a texture: a raster trick
    always leaves moire, and moire on a mountain reads as a printing fault.
    """
    RH, RW = img.shape[0], img.shape[1]
    h = upsample(world.height, RW, RH)
    gy, gx = np.gradient(h)
    mag = np.hypot(gx, gy)
    land_f = upsample((world.height > 0).astype(np.float32), RW, RH, Image.BILINEAR)
    land = land_f > 0.5
    if not land.any():
        return img
    # A stroke runs several pixels past its sample point, so sampling right up
    # to the shore leaves a fringe of ticks standing in the water.  Back the
    # sampling mask off the coast, and clip the finished strokes to the land.
    land = _morph(land, max(2, int(scale * 1.6)), "min")

    lo = float(np.percentile(mag[land], lo_pct))
    hi = float(np.percentile(mag[land], hi_pct))
    if hi <= lo:
        return img

    mask = Image.new("L", (RW, RH), 0)
    d = ImageDraw.Draw(mask)
    step = max(2, int(spacing * scale))
    hi_lo = hi - lo
    rng = np.random.default_rng(20240531)

    for y0 in range(step // 2, RH, step):
        # Jitter the lattice.  An even grid of strokes reads as woven cloth;
        # engravers never spaced their hachures regularly, and the eye notices
        # that long before it can say why.
        ox = float(rng.uniform(0, step))
        for x0 in range(step // 2, RW, step):
            x = int(x0 + ox + rng.uniform(-step * 0.4, step * 0.4))
            y = int(y0 + rng.uniform(-step * 0.4, step * 0.4))
            if not (0 <= x < RW and 0 <= y < RH) or not land[y, x]:
                continue
            m = mag[y, x]
            if m < lo:
                continue
            t = min((m - lo) / hi_lo, 1.0)
            g = gx[y, x]
            e = gy[y, x]
            n = math.hypot(g, e)
            if n < 1e-6:
                continue
            length = (3.0 + 8.0 * t) * scale * 0.62 * rng.uniform(0.75, 1.25)
            d.line([(x, y), (x - g / n * length, y - e / n * length)],
                   fill=int(255 * (0.30 + 0.70 * t) * rng.uniform(0.80, 1.0)),
                   width=max(1, int(scale * 0.55)))

    a = np.asarray(mask, dtype=np.float32)[..., None] / 255.0
    a = a * np.clip(land_f * 1.6, 0.0, 1.0)[..., None]
    c = np.array(colour if colour is not None else INK, dtype=np.float32)
    return _mix(img, c, a * alpha)


def render(world, scale=2, paper_tone=1.0):
    img, h, land_m = render_terrain(world, scale=scale)
    img = draw_hachures(img, world, scale=scale)
    img = draw_lakes(img, world, scale=scale)
    img = draw_rivers(img, world, scale=scale)
    img = draw_coastline(img, land_m, scale=scale)
    return finish_paper(img, world, scale=scale, paper_tone=paper_tone)

