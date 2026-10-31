"""Assembling the sheet: relief, borders, roads, grid, names, and title."""

from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw

from . import decor, layout, render, roads as roadmod
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
    "town": ("serif", 0.0108, 0.00),
}


def _nice_step(span: float, target: float = 5.0) -> float:
    """The coarsest round step that still gives about ``target`` grid lines."""
    for step in (20.0, 10.0, 5.0, 2.0, 1.0, 0.5):
        if span / step >= target - 0.5:
            return step
    return 0.5


def _soundings(depth, count=24, rng=None):
    """Depth figures in the sea: the oldest way to make water look measured."""
    rng = rng or np.random.default_rng(0)
    hh, ww = depth.shape
    ys, xs = np.where((depth > 15) & (depth < 520))
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


def _star(draw, x, y, r, fill):
    """A five-pointed star: the mark a period plate uses for a capital."""
    pts = []
    for i in range(10):
        a = -math.pi / 2 + i * math.pi / 5
        rr = r if i % 2 == 0 else r * 0.42
        pts.append((x + rr * math.cos(a), y + rr * math.sin(a)))
    draw.polygon(pts, fill=fill)


def _furniture_slots():
    """What has to be printed on the map, in the order it gets to choose.

    The cartouche goes first because it is the one piece that cannot be
    squeezed into a gap: it needs a lake of open water.  The scale bar goes
    last and is the most willing to sit against the frame.
    """
    return [
        layout.Slot("cartouche", 0.315, 0.130,
                    bias=[(0.27, 0.155), (0.73, 0.155), (0.27, 0.80)],
                    bias_gain=0.50, sigma=0.24),
        layout.Slot("elevation", 0.148, 0.185,
                    bias=[(0.855, 0.72), (0.145, 0.72)],
                    bias_gain=0.45, sigma=0.26),
        layout.Slot("biome", 0.148, 0.215,
                    bias=[(0.855, 0.455), (0.145, 0.455)],
                    bias_gain=0.30, sigma=0.28,
                    after=("elevation", -0.225)),
        layout.Slot("compass", 0.115, 0.115,
                    bias=[(0.86, 0.145), (0.14, 0.145), (0.86, 0.85)],
                    bias_gain=0.40, sigma=0.20),
        layout.Slot("scale", 0.230, 0.052,
                    bias=[(0.17, 0.925), (0.83, 0.925)],
                    bias_gain=0.50, sigma=0.24, min_open=0.55),
    ]


def _political_slots():
    return [
        layout.Slot("cartouche", 0.315, 0.130,
                    bias=[(0.27, 0.155), (0.73, 0.155), (0.27, 0.80)],
                    bias_gain=0.50, sigma=0.24),
        layout.Slot("provinces", 0.205, 0.250,
                    bias=[(0.845, 0.60), (0.155, 0.60)],
                    bias_gain=0.45, sigma=0.26, min_open=0.72),
        layout.Slot("compass", 0.115, 0.115,
                    bias=[(0.86, 0.145), (0.14, 0.145), (0.86, 0.85)],
                    bias_gain=0.40, sigma=0.20),
        layout.Slot("scale", 0.230, 0.052,
                    bias=[(0.17, 0.925), (0.83, 0.925)],
                    bias_gain=0.50, sigma=0.24, min_open=0.55),
    ]


def compose(world, namer: Namer, scale=None, margin_frac=0.10, title=None, subtitle=None,
            plate=None, framing=True, target_px=2600, mode="physical"):
    """Render one world onto a finished sheet.

    Returns (page, placed labels, view).  The view is the cropped world the
    labels were laid out on, and it is the one the text must describe: label
    coordinates are frame coordinates, so measuring them against the uncropped
    grid silently reports the wrong river, the wrong province, the wrong peak.
    """
    spec = world.spec

    if framing:
        fr = layout.choose_frame(world.height > 0)
        view = world.cropped(fr)
    else:
        fr = layout.Frame(0, 0, spec.width, spec.height)
        view = world

    vspec = view.spec
    s = int(scale) if scale else layout.fit_scale(fr, target_px)
    land = view.height > 0

    prov = getattr(view, "provinces", None)
    if mode == "political":
        img, h, land_m = render.render_political(view, scale=s)
        # A whisper of hachure under the wash.  Without it the provinces read
        # as flat polygons laid on a drawing; with it the country looks
        # engraved first and coloured afterwards, which is the right order.
        img = render.draw_hachures(img, view, scale=s, alpha=0.15)
        img = render.draw_lakes(img, view, scale=s, colour=(0.42, 0.56, 0.64), alpha=0.30)
        img = render.draw_rivers(img, view, scale=s, alpha=0.55)
        img = render.draw_coastline(img, land_m, scale=s, strength=1.0)
        img = render.draw_borders(img, prov, land, scale=s, weight=1, alpha=0.52,
                                  dash=0, colour=(0.40, 0.28, 0.22))
    else:
        img, h, land_m = render.render_terrain(view, scale=s)
        img = render.draw_hachures(img, view, scale=s)
        # No political limits here: on a physical sheet they read as hatching
        # along every ridge.  The political plate is where they belong.
        img = render.draw_lakes(img, view, scale=s)
        img = render.draw_rivers(img, view, scale=s, alpha=0.66)
        img = render.draw_coastline(img, land_m, scale=s)

    hh, ww = view.height.shape
    RW, RH = ww * s, hh * s
    M = max(28, int(RH * margin_frac))
    PW, PH = RW + 2 * M, RH + 2 * M

    page = Image.new("RGB", (PW, PH), tuple(int(c * 255) for c in PAPER))
    draw = ImageDraw.Draw(page)

    ink = tuple(int(c * 255) for c in INK)
    ink_soft = tuple(int(c * 255) for c in INK_SOFT)
    ink_water = tuple(int(c * 255) for c in INK_WATER)
    paper_rgb = tuple(int(c * 255) for c in PAPER)
    box = (M, M, M + RW, M + RH)

    # --- furniture, placed before the names so the names can dodge it -----
    open_field = layout.openness(land)
    spots = layout.place(_political_slots() if mode == "political" else _furniture_slots(),
                         open_field)

    def to_page(b):
        return (M + b[0] * RW, M + b[1] * RH, M + b[2] * RW, M + b[3] * RH)

    def to_cell(b):
        return ((b[0] * RW) / s, (b[1] * RH) / s, (b[2] * RW) / s, (b[3] * RH) / s)

    cart_box = to_page(spots["cartouche"]) if "cartouche" in spots else None
    elev_box = to_page(spots["elevation"]) if "elevation" in spots else None
    biome_box = to_page(spots["biome"]) if "biome" in spots else None
    comp_box = to_page(spots["compass"]) if "compass" in spots else None
    scale_box = to_page(spots["scale"]) if "scale" in spots else None
    prov_box = to_page(spots["provinces"]) if "provinces" in spots else None

    obstacles = [to_cell(spots[k]) for k in ("cartouche", "elevation", "biome",
                                             "provinces", "compass", "scale")
                 if k in spots]

    name = title or namer.title(namer.cultures[0])
    if subtitle:
        sub = subtitle
    elif mode == "political":
        sub = f"shewing the political divisions, MDCCC{roman(1 + spec.seed % 9)}"
    else:
        sub = f"surveyed in the year {roman(1800 + (spec.seed % 99))}"

    # --- names ----------------------------------------------------------
    labels = build_labels(view, namer, avoid=(name,))
    for lb in labels:
        face, frac, track_f = STYLES[lb.kind]
        px = max(10, int(frac * RW))
        f = decor.font(face, px)
        lb.tracking = px * track_f
        wpx, hpx = _measure(draw, lb.text, f, lb.tracking)
        lb.w, lb.h = wpx / s, hpx / s

    placed = avoid(labels, ww, hh, obstacles)

    # --- seats of government --------------------------------------------
    # A political plate that marks every town alike says nothing about where
    # the country is governed.  One town per province is promoted to its seat:
    # the one nearest the province's own name, which is already sited near the
    # middle of the province.
    prov_grid = getattr(view, "provinces", None)
    if prov_grid is not None:
        seats = {}
        for lb in placed:
            if lb.kind != "region" or lb.prov < 0:
                continue
            best, best_key = None, (9, 1e18)
            for t in placed:
                if t.kind != "town":
                    continue
                ty, tx = int(t.y), int(t.x)
                if not (0 <= ty < hh and 0 <= tx < ww):
                    continue
                if int(prov_grid[ty, tx]) != lb.prov:
                    continue
                d = (t.x - lb.x) ** 2 + (t.y - lb.y) ** 2
                key = (t.rank, d)
                if key < best_key:
                    best, best_key = t, key
            if best is not None:
                seats[lb.prov] = best
        for t in seats.values():
            t.capital = True

    # --- roads, from the towns that survived placement -------------------
    towns = [(lb.x, lb.y) for lb in placed if lb.kind == "town"]
    paths = roadmod.build_roads(towns, view.height, land)
    paths = [roadmod.smooth(p, window=7) for p in paths]
    img = render.draw_roads(img, paths, scale=s)

    img = render.finish_paper(img, view, scale=s)
    page.paste(Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)), (M, M))

    # --- rhumb lines -----------------------------------------------------
    water = Image.new("L", page.size, 0)
    water.paste(Image.fromarray(((view.height <= 0) * 255).astype(np.uint8))
                .resize((ww * s, hh * s), Image.NEAREST), (M, M))
    if comp_box:
        page = decor.draw_rhumbs(
            page, ((comp_box[0] + comp_box[2]) / 2, (comp_box[1] + comp_box[3]) / 2),
            box, water, ink_soft, alpha=0.22, count=16)
        draw = ImageDraw.Draw(page)

    halo = max(1, int(s * 0.8))

    for lb in placed:
        x = M + lb.x * s
        y = M + lb.y * s
        face, frac, _ = STYLES[lb.kind]
        f = decor.font(face, max(10, int(frac * RW)))
        if lb.kind == "town":
            boost = {0: 1.14, 1: 1.00, 2: 0.92}.get(lb.rank, 1.0)
            f = decor.font("serif-bold" if lb.capital else "serif",
                           max(10, int(frac * RW * boost)))
            if lb.capital:
                _star(draw, x, y, max(4.0, f.size * 0.40), ink)
                r = f.size * 0.40
            elif lb.rank == 1:
                r = max(3.0, f.size * 0.21)
                draw.ellipse([x - r, y - r, x + r, y + r], outline=ink,
                             width=max(1, int(round(f.size * 0.07))))
                draw.ellipse([x - r * 0.34, y - r * 0.34,
                              x + r * 0.34, y + r * 0.34], fill=ink)
            else:
                r = max(2, int(f.size * 0.15))
                draw.ellipse([x - r, y - r, x + r, y + r], fill=ink)
            gap = max(r * 2.4, f.size * 0.52)
            tx = x - gap - lb.w * s if lb.flip else x + gap
            decor.tracked_text(draw, (tx, y), lb.text, f, ink,
                               anchor_center=False, stroke_width=halo,
                               stroke_fill=paper_rgb)
        elif lb.kind == "river" and abs(lb.angle) > 8:
            decor.rotated_text(page, lb.text, (x, y), f, ink_water, lb.angle,
                               stroke_width=halo, stroke_fill=paper_rgb)
        else:
            fill = ink_water if lb.kind in ("water", "river") else ink_soft
            # A range name is the one label set over hillshade and hachures
            # both, so it gets one extra pixel of halo -- not more: a halo
            # wider than the letter's stem reads as a pale letter with a dark
            # edge, which is the opposite of what a name should look like.
            stroke = halo + 1 if lb.kind == "range" else halo
            decor.tracked_text(draw, (x, y), lb.text, f, fill,
                               tracking=lb.tracking, stroke_width=stroke,
                               stroke_fill=paper_rgb)

    # --- soundings -------------------------------------------------------
    depth = np.maximum(-view.height, 0.0)
    rng = np.random.default_rng(spec.seed * 17 + 5)
    fs = decor.font("antiqua-italic", max(11, int(RW * 0.0050)))
    for sx, sy, val in _soundings(depth, count=44, rng=rng):
        px, py = M + sx * s, M + sy * s
        draw.text((px, py), str(val), font=fs,
                  fill=tuple(int(c * 255) for c in INK_WATER), anchor="mm")

    # --- the rest of the furniture --------------------------------------
    lon_span = (vspec.lat_top - vspec.lat_bottom) * (ww / hh)
    lon_left = vspec.lon_left
    step = _nice_step(lon_span, target=5.0)

    page = decor.draw_graticule(page, box, vspec.lat_top, vspec.lat_bottom, lon_left,
                                lon_left + lon_span, ink, step=step, alpha=0.11)
    draw = ImageDraw.Draw(page)
    decor.draw_frame(draw, box, M, ink, paper_rgb, vspec.lat_top, vspec.lat_bottom,
                        lon_left, lon_left + lon_span, step=step)
    decor.draw_graticule_labels(draw, box, M, vspec.lat_top, vspec.lat_bottom,
                                lon_left, lon_left + lon_span, ink_soft, step=step,
                                size=max(11, int(M * 0.16)))

    if plate:
        fp = decor.font("antiqua-bold", max(10, int(M * 0.19)))
        draw.text((PW - M * 0.70, M * 0.14), plate, font=fp, fill=ink_soft, anchor="ra")

    if cart_box:
        decor.draw_cartouche(draw, cart_box, name, sub, ink, paper_rgb,
                             title_size=max(18, int((cart_box[3] - cart_box[1]) * 0.26)),
                             sub_size=max(10, int((cart_box[3] - cart_box[1]) * 0.13)))
    if comp_box:
        cx = (comp_box[0] + comp_box[2]) / 2
        cy = (comp_box[1] + comp_box[3]) / 2
        r = 0.5 * min(comp_box[2] - comp_box[0], comp_box[3] - comp_box[1]) * 0.92
        decor.draw_compass(draw, cx, cy, r, INK, PAPER)

    mid_lat = 0.5 * (vspec.lat_top + vspec.lat_bottom)
    km_across = lon_span * 111.0 * math.cos(math.radians(mid_lat))
    leagues_across = km_across / 5.56
    bar_w = RW * 0.20
    leagues_shown = max(50, round(leagues_across * (bar_w / RW) / 50.0) * 50)
    bar_x = scale_box[0] if scale_box else M + RW * 0.05
    bar_y = scale_box[1] if scale_box else M + RH * 0.955
    bar_h = max(7, int(M * 0.10))
    decor.draw_scale_bar(draw, bar_x, bar_y + bar_h * 1.6, bar_w, ink, paper_rgb,
                         units=f"leagues ({leagues_shown})", segments=4, height=bar_h)

    if elev_box:
        entries = []
        for i in range(0, len(HYPSOMETRIC) - 1, 2):
            lo = int(HYPSOMETRIC[i][0])
            hi = int(HYPSOMETRIC[min(i + 2, len(HYPSOMETRIC) - 1)][0])
            entries.append((HYPSOMETRIC[i][1], f"{lo}–{hi} m"))
        decor.draw_legend(draw, elev_box, entries[:5], ink, paper_rgb,
                          size=max(10, int(M * 0.14)))

    biome = view.biome
    if biome is not None and biome_box:
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

    if mode == "political" and prov_box:
        entries = [(render.province_swatch(lb.prov), lb.text)
                   for lb in placed if lb.kind == "region" and lb.prov >= 0]
        entries.sort(key=lambda e: e[1])
        if entries:
            decor.draw_legend(draw, prov_box, entries[:8], ink, paper_rgb,
                              title="PROVINCES", size=max(9, int(M * 0.105)))

    return page, placed, view
