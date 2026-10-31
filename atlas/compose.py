"""Assembling the sheet: relief, borders, roads, grid, names, and title."""

from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw

from . import decor, render, roads as roadmod
from .labels import build_labels, avoid
from .naming import Namer, roman
from .climate import BIOMES
from .palette import INK, INK_SOFT, INK_WATER, PAPER, HYPSOMETRIC

# kind -> (font face, point size as a fraction of the sheet width, tracking as
# a fraction of the point size).  Sizing type against the sheet rather than
# against the terrain grid is what keeps a 3000-px plate and a 1500-px preview
# looking like the same map.
STYLES = {
    "region": ("antiqua-bold", 0.0165, 0.22),
    "water": ("antiqua-italic", 0.0165, 0.12),
    "range": ("antiqua", 0.0115, 0.18),
    "river": ("antiqua-italic", 0.0095, 0.00),
    "town": ("serif", 0.0095, 0.00),
}


def _sea_spots(land, blocks=30, count=4, separation=0.34, threshold=0.35):
    """Open water, in map-normalised coordinates, most open first."""
    hh, ww = land.shape
    bh, bw = max(1, hh // blocks), max(1, ww // blocks)
    frac = land[: bh * blocks, : bw * blocks].reshape(blocks, bh, blocks, bw).mean(axis=(1, 3))

    big = 1e9
    dist = np.where(frac > 0.10, 0.0, big)
    ys, xs = np.mgrid[0:blocks, 0:blocks]
    for _ in range(blocks):
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            sy = np.clip(ys + dy, 0, blocks - 1)
            sx = np.clip(xs + dx, 0, blocks - 1)
            np.minimum(dist, dist[sy, sx] + 1.0, out=dist)
    dist[dist > big / 2] = 0.0

    edge = np.minimum(np.minimum(xs, blocks - 1 - xs), np.minimum(ys, blocks - 1 - ys)) / (blocks - 1)
    score = dist * np.clip(edge / 0.16, 0, 1) ** 0.5

    spots = []
    work = score.copy()
    for _ in range(count):
        j = int(np.argmax(work))
        y, x = divmod(j, blocks)
        if work[y, x] <= 0.6:
            break
        spots.append(((x + 0.5) / blocks, (y + 0.5) / blocks))
        d = np.hypot((xs - x) / blocks, (ys - y) / blocks)
        work[d < separation] = 0
    return spots


def _soundings(depth, count=24, rng=None):
    """Depth figures in the sea: the oldest way to make water look measured."""
    rng = rng or np.random.default_rng(0)
    hh, ww = depth.shape
    ys, xs = np.where((depth > 20) & (depth < 340))
    if len(ys) == 0:
        return []
    pick = rng.choice(len(ys), size=min(count * 8, len(ys)), replace=False)
    out = []
    min_d = (min(hh, ww) * 0.075) ** 2
    for i in pick:
        y, x = int(ys[i]), int(xs[i])
        if any((x - px) ** 2 + (y - py) ** 2 < min_d for px, py, _ in out):
            continue
        out.append((x, y, int(depth[y, x])))
        if len(out) >= count:
            break
    return out


def _measure(draw, text, fnt, tracking):
    w = sum(draw.textlength(ch, font=fnt) for ch in text) + tracking * max(0, len(text) - 1)
    return w, fnt.size * 1.35


def compose(world, namer: Namer, scale=2, margin_frac=0.10, title=None, subtitle=None,
            plate=None):
    """Render one world onto a finished sheet.  Returns (page, placed labels)."""
    spec = world.spec
    land = world.height > 0

    img, h, land_m = render.render_terrain(world, scale=scale)
    img = render.draw_borders(img, getattr(world, "provinces", None), land,
                              scale=scale, weight=2, alpha=0.75)
    img = render.draw_lakes(img, world, scale=scale)
    img = render.draw_rivers(img, world, scale=scale)
    img = render.draw_coastline(img, land_m, scale=scale)

    hh, ww = world.height.shape
    RW, RH = ww * scale, hh * scale
    M = max(28, int(RH * margin_frac))
    PW, PH = RW + 2 * M, RH + 2 * M

    page = Image.new("RGB", (PW, PH), tuple(int(c * 255) for c in PAPER))
    draw = ImageDraw.Draw(page)

    ink = tuple(int(c * 255) for c in INK)
    ink_soft = tuple(int(c * 255) for c in INK_SOFT)
    ink_water = tuple(int(c * 255) for c in INK_WATER)
    paper_rgb = tuple(int(c * 255) for c in PAPER)
    box = (M, M, M + RW, M + RH)

    aspect = spec.aspect
    lon_span = (spec.lat_top - spec.lat_bottom) * aspect
    lon_left = 10.0

    # --- furniture, placed before the names so the names can dodge it -----
    legend_w = RW * 0.15
    legend_box = (M + RW * 0.795, M + RH * 0.555,
                  M + RW * 0.795 + legend_w, M + RH * 0.555 + RH * 0.180)
    biome_box = (M + RW * 0.795, M + RH * 0.750,
                 M + RW * 0.795 + legend_w, M + RH * 0.750 + RH * 0.215)

    def hits(a, b, pad=0.0):
        return not (a[2] + pad < b[0] or b[2] + pad < a[0]
                    or a[3] + pad < b[1] or b[3] + pad < a[1])

    spots = _sea_spots(land, count=6)
    page_spots = [(M + sx * RW, M + sy * RH) for sx, sy in spots]

    name = title or namer.title(namer.cultures[0])
    sub = subtitle or f"surveyed in the year {roman(1800 + (spec.seed % 99))}"

    obstacles = [tuple(v / scale for v in (legend_box[0] - M, legend_box[1] - M,
                                           legend_box[2] - M, legend_box[3] - M)),
                 tuple(v / scale for v in (biome_box[0] - M, biome_box[1] - M,
                                           biome_box[2] - M, biome_box[3] - M))]
    bar_box = (M + RW * 0.03, M + RH * 0.90, M + RW * 0.31, M + RH * 0.99)
    obstacles.append(tuple(v / scale for v in (bar_box[0] - M, bar_box[1] - M,
                                               bar_box[2] - M, bar_box[3] - M)))

    cart_box = None
    for cx, cy in page_spots:
        cw, ch = RW * 0.30, RH * 0.13
        x0 = min(max(cx - cw / 2, M + RW * 0.04), M + RW * 0.96 - cw)
        y0 = min(max(cy - ch / 2, M + RH * 0.04), M + RH * 0.96 - ch)
        cand = (x0, y0, x0 + cw, y0 + ch)
        if hits(cand, legend_box, pad=RH * 0.02) or hits(cand, biome_box, pad=RH * 0.02):
            continue
        cart_box = cand
        obstacles.append(tuple(v / scale for v in (x0 - M, y0 - M, x0 + cw - M, y0 + ch - M)))
        break

    compass_box = None
    for cx, cy in page_spots:
        r = RH * 0.050
        cand = (cx - r * 1.5, cy - r * 1.5, cx + r * 1.5, cy + r * 1.5)
        if cart_box and hits(cand, cart_box, pad=RH * 0.02):
            continue
        if hits(cand, legend_box, pad=RH * 0.02) or hits(cand, biome_box, pad=RH * 0.02):
            continue
        if cand[0] < M + RW * 0.02 or cand[2] > M + RW * 0.98:
            continue
        compass_box = cand
        obstacles.append(tuple(v / scale for v in (cand[0] - M, cand[1] - M,
                                                   cand[2] - M, cand[3] - M)))
        break

    # --- names ----------------------------------------------------------
    labels = build_labels(world, namer)
    for lb in labels:
        face, frac, track_f = STYLES[lb.kind]
        px = max(10, int(frac * RW))
        f = decor.font(face, px)
        lb.tracking = px * track_f
        wpx, hpx = _measure(draw, lb.text, f, lb.tracking)
        lb.w, lb.h = wpx / scale, hpx / scale

    placed = avoid(labels, ww, hh, obstacles)

    # --- roads, from the towns that survived placement -------------------
    towns = [(lb.x, lb.y) for lb in placed if lb.kind == "town"]
    paths = roadmod.build_roads(towns, world.height, land)
    paths = [roadmod.smooth(p, window=7) for p in paths]
    img = render.draw_roads(img, paths, scale=scale)

    img = render.finish_paper(img, world, scale=scale)
    page.paste(Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)), (M, M))

    halo = max(1, int(scale * 0.8))

    for lb in placed:
        x = M + lb.x * scale
        y = M + lb.y * scale
        face, frac, _ = STYLES[lb.kind]
        f = decor.font(face, max(10, int(frac * RW)))
        if lb.kind == "town":
            r = max(2, int(f.size * 0.15))
            draw.ellipse([x - r, y - r, x + r, y + r], fill=ink)
            decor.tracked_text(draw, (x + r * 2.4, y), lb.text, f, ink,
                               anchor_center=False, stroke_width=halo,
                               stroke_fill=paper_rgb)
        elif lb.kind == "river" and abs(lb.angle) > 8:
            decor.rotated_text(page, lb.text, (x, y), f, ink_water, lb.angle,
                               stroke_width=halo, stroke_fill=paper_rgb)
        else:
            fill = ink_water if lb.kind in ("water", "river") else ink_soft
            decor.tracked_text(draw, (x, y), lb.text, f, fill,
                               tracking=lb.tracking, stroke_width=halo,
                               stroke_fill=paper_rgb)

    # --- soundings -------------------------------------------------------
    depth = np.maximum(-world.height, 0.0)
    rng = np.random.default_rng(spec.seed * 17 + 5)
    fs = decor.font("antiqua-italic", max(7, int(RW * 0.0028)))
    for sx, sy, val in _soundings(depth, count=24, rng=rng):
        px, py = M + sx * scale, M + sy * scale
        draw.text((px, py), str(val), font=fs,
                  fill=tuple(int(c * 255) for c in INK_WATER), anchor="mm")

    # --- the rest of the furniture --------------------------------------
    decor.draw_graticule(draw, box, spec.lat_top, spec.lat_bottom, lon_left,
                         lon_left + lon_span, ink, step=10.0, alpha=0.15)
    decor.draw_frame(draw, box, M, ink, paper_rgb)
    decor.draw_graticule_labels(draw, box, M, spec.lat_top, spec.lat_bottom,
                                lon_left, lon_left + lon_span, ink_soft, step=10.0,
                                size=max(11, int(M * 0.16)))

    if plate:
        fp = decor.font("antiqua-bold", max(10, int(M * 0.19)))
        draw.text((PW - M * 0.28, M * 0.30), plate, font=fp, fill=ink_soft, anchor="ra")

    if cart_box:
        decor.draw_cartouche(draw, cart_box, name, sub, ink, paper_rgb,
                             title_size=max(18, int((cart_box[3] - cart_box[1]) * 0.26)),
                             sub_size=max(10, int((cart_box[3] - cart_box[1]) * 0.13)))
    if compass_box:
        cx = (compass_box[0] + compass_box[2]) / 2
        cy = (compass_box[1] + compass_box[3]) / 2
        decor.draw_compass(draw, cx, cy, RH * 0.055, INK, PAPER)

    mid_lat = 0.5 * (spec.lat_top + spec.lat_bottom)
    km_across = lon_span * 111.0 * math.cos(math.radians(mid_lat))
    leagues_across = km_across / 5.56
    bar_w = RW * 0.20
    leagues_shown = max(100, round(leagues_across * (bar_w / RW) / 100.0) * 100)
    decor.draw_scale_bar(draw, M + RW * 0.05, M + RH * 0.955, bar_w, ink, paper_rgb,
                         units=f"leagues ({leagues_shown})", segments=4,
                         height=max(7, int(M * 0.10)))

    entries = []
    for i in range(0, len(HYPSOMETRIC) - 1, 2):
        lo = int(HYPSOMETRIC[i][0])
        hi = int(HYPSOMETRIC[min(i + 2, len(HYPSOMETRIC) - 1)][0])
        entries.append((HYPSOMETRIC[i][1], f"{lo}–{hi} m"))
    decor.draw_legend(draw, legend_box, entries[:5], ink, paper_rgb,
                      size=max(10, int(M * 0.14)))

    biome = world.biome
    if biome is not None:
        present = []
        for k, (bname, colour) in BIOMES.items():
            frac = float((biome == k).mean())
            if frac > 0.008:
                present.append((frac, colour, bname))
        present.sort(key=lambda t: -t[0])
        if present:
            decor.draw_legend(draw, biome_box, [(c, n) for _f, c, n in present[:6]],
                              ink, paper_rgb, title="BIOMES",
                              size=max(9, int(M * 0.115)))

    return page, placed
