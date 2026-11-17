"""Cross-sections: the country cut open along a line.

A map answers "where"; a section answers "how high, and how far".  Period
atlases carried them beside the plates -- a line of section drawn through the
highest ground, with the sea at both ends -- because a hillshade can suggest a
mountain but only a profile can state its height against its width.

Everything here is measured from the same height field the plate is drawn
from.  The vertical scale is exaggerated, as it must be: a country sixteen
degrees wide is a hundred times wider than its mountains are high, and at true
scale the range would be a scratch on the paper.  The exaggeration is printed
on the sheet, because an unlabelled exaggeration is a lie about shape.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from . import decor
from .features import cell_km
from .palette import INK, INK_SOFT, INK_WATER, PAPER

LAND_FILL = (0.858, 0.806, 0.700)
SEA_FILL = (0.842, 0.874, 0.884)
GRID = (0.760, 0.716, 0.628)


def _sample(height, xs, ys):
    """Bilinear height at fractional grid coordinates."""
    hh, ww = height.shape
    xs = np.clip(xs, 0.0, ww - 1.0)
    ys = np.clip(ys, 0.0, hh - 1.0)
    x0 = np.floor(xs).astype(np.int32)
    y0 = np.floor(ys).astype(np.int32)
    x1 = np.minimum(x0 + 1, ww - 1)
    y1 = np.minimum(y0 + 1, hh - 1)
    tx = xs - x0
    ty = ys - y0
    h00 = height[y0, x0]
    h10 = height[y0, x1]
    h01 = height[y1, x0]
    h11 = height[y1, x1]
    return (h00 * (1.0 - tx) * (1.0 - ty) + h10 * tx * (1.0 - ty)
            + h01 * (1.0 - tx) * ty + h11 * tx * ty)


def line_profile(world, p0, p1, n=1200):
    """Distance in kilometres and height in metres along a straight line."""
    h = world.height
    km_x, km_y = cell_km(world)
    t = np.linspace(0.0, 1.0, n)
    xs = p0[0] + (p1[0] - p0[0]) * t
    ys = p0[1] + (p1[1] - p0[1]) * t
    elev = _sample(h, xs, ys)
    dx = np.diff(xs) * km_x
    dy = np.diff(ys) * km_y
    step = np.hypot(dx, dy)
    dist = np.concatenate([[0.0], np.cumsum(step)])
    return dist, elev


def choose(world, feats):
    """The lines of section: through the highest ground, and through the seat.

    Two lines are enough to describe a country: one across its grain, one
    along it.  The first is drawn west to east through the highest cell, so it
    must cross the main range; the second north to south through the capital,
    so it passes the place the country is governed from.
    """
    h = world.height
    hh, ww = h.shape
    land = h > 0
    if not land.any():
        return []

    def span_along(values, limit):
        """First and last occupied index, with a little sea at each end."""
        idx = np.flatnonzero(values)
        if not idx.size:
            return 0, limit
        pad = max(2, int(round(0.020 * limit)))
        return max(0, int(idx[0]) - pad), min(limit, int(idx[-1]) + pad)

    py, px = np.unravel_index(int(np.argmax(np.where(land, h, -1e9))), h.shape)
    # A section runs from sea to sea, but only just: a strip of water at each
    # end frames the country, while the open ocean beyond it would take two
    # thirds of the width and flatten the mountains into a scratch.
    xa, xb = span_along(land[py, :], ww - 1)
    lines = [{
        "label": "A\u2014A\u2032",
        "title": "west to east through the highest ground",
        "p0": (float(xa), float(py)), "p1": (float(xb), float(py)),
        "ends": ("WEST", "EAST"),
    }]
    chief = feats.chief_town()
    if chief is not None:
        cx = min(max(int(round(float(chief.x))), 0), ww - 1)
        ya, yb = span_along(land[:, cx], hh - 1)
        lines.append({
            "label": "B\u2014B\u2032",
            "title": f"north to south through {chief.name.title()}",
            "p0": (float(cx), float(ya)), "p1": (float(cx), float(yb)),
            "ends": ("NORTH", "SOUTH"),
        })
    return lines


def _nice_step(span, target=6):
    """A round tick interval that gives roughly ``target`` ticks.

    Every 1-2-5 multiple is considered before choosing, so a span of 1,200
    gets a 200-kilometre interval and not the 1,000 the first candidate
    happens to offer.
    """
    raw = max(span / max(1, target), 1e-6)
    best = None
    for mult in (1, 2, 5):
        for power in range(-2, 7):
            step = mult * 10.0 ** power
            if step >= raw and (best is None or step < best):
                best = step
    return best if best is not None else raw


def _rgb(c):
    return tuple(int(round(v * 255)) for v in c)


def _draw_band(draw, box, prof, line, world, feats, font_small, font_tiny):
    """One section: header, axes, sea, land, profile line and labels."""
    x0, y0, x1, y1 = box
    dist, elev = prof
    total = float(dist[-1]) if len(dist) else 0.0

    # The ocean beyond the shelf is not the subject.  A section that spends
    # two thirds of its height on abyssal water flattens the mountains into a
    # scratch, so the sea floor is flattened at half a kilometre down.
    vmin = max(-500.0, min(0.0, float(np.floor(elev.min() / 200.0) * 200.0)))
    vmax = max(200.0, float(np.ceil(elev.max() / 500.0) * 500.0))
    span = vmax - vmin

    def ey(v):
        return y1 - (v - vmin) / span * (y1 - y0)

    def ex(d):
        return x0 + (d / total) * (x1 - x0) if total else x0

    water_y = ey(0.0)
    elev_orig = elev
    elev = np.maximum(elev, vmin)

    # Water only where the ground is under it: a section draws the sea at its
    # two ends, not a blue sheet under the whole country.
    i = 0
    n = len(elev_orig)
    while i < n:
        if elev_orig[i] >= 0:
            i += 1
            continue
        j = i
        while j < n and elev_orig[j] < 0:
            j += 1
        top = [(ex(dist[k]), water_y) for k in range(i, j)]
        bottom = [(ex(dist[k]), ey(float(elev[k]))) for k in range(j - 1, i - 1, -1)]
        draw.polygon(top + bottom, fill=_rgb(SEA_FILL))
        i = j

    # The land silhouette, clamped at the waterline: a section fills the
    # ground, it does not fill the sea.
    top = [(ex(dist[i]), ey(max(float(elev[i]), 0.0))) for i in range(len(dist))]
    draw.polygon(top + [(x1, water_y), (x0, water_y)], fill=_rgb(LAND_FILL))

    # Grid and axes.
    step_v = _nice_step(span, 5)
    v = 0.0
    while v <= vmax + 1e-6:
        yy = ey(v)
        draw.line([(x0, yy), (x1, yy)], fill=_rgb(GRID), width=1)
        draw.text((x0 - 10, yy), f"{int(v):,}", font=font_tiny,
                  fill=_rgb(INK_SOFT), anchor="rm")
        v += step_v
    if abs(vmax % step_v) > 1e-6:
        draw.text((x0 - 10, y0), f"{int(vmax):,} m", font=font_tiny,
                  fill=_rgb(INK_SOFT), anchor="rm")
    draw.line([(x0, water_y), (x1, water_y)], fill=_rgb(GRID), width=1)
    step_d = _nice_step(total, 6)
    d = 0.0
    while d <= total + 1e-6:
        xx = ex(d)
        draw.line([(xx, y1), (xx, y1 + 5)], fill=_rgb(INK_SOFT), width=1)
        draw.text((xx, y1 + 9), f"{int(d):,}", font=font_tiny,
                  fill=_rgb(INK_SOFT), anchor="mt")
        d += step_d
    draw.text((x1 + 24, y1 + 9), "km", font=font_tiny, fill=_rgb(INK_SOFT), anchor="lm")

    # The profile itself, over the fill: sea floor and all.
    pts = [(ex(dist[i]), ey(float(elev[i]))) for i in range(len(dist))]
    draw.line(pts, fill=_rgb(INK), width=2, joint="curve")

    # Horizontal exaggeration.  px per metre of height divided by px per
    # metre of ground distance.
    if total > 0 and span > 0:
        px_per_m = (y1 - y0) / span
        px_per_ground_m = (x1 - x0) / (total * 1000.0)
        ve = px_per_m / px_per_ground_m if px_per_ground_m else 0.0
        draw.text((x1, y0 - 6), f"vertical exaggeration \u00d7{ve:.0f}",
                  font=font_tiny, fill=_rgb(INK_SOFT), anchor="rb")

    # The highest point on the line.
    i_peak = int(np.argmax(elev))
    if float(elev[i_peak]) > 0:
        px, py = ex(dist[i_peak]), ey(float(elev[i_peak]))
        label = f"{int(round(float(elev[i_peak]), -1)):,} m"
        if px > x1 - 150:
            draw.line([(px, py), (px - 8, py - 8)], fill=_rgb(INK), width=1)
            draw.text((px - 10, py - 8), label, font=font_tiny, fill=_rgb(INK),
                      anchor="rm", stroke_width=2, stroke_fill=_rgb(PAPER))
        else:
            draw.line([(px, py), (px + 8, py - 8)], fill=_rgb(INK), width=1)
            draw.text((px + 10, py - 8), label, font=font_tiny, fill=_rgb(INK),
                      anchor="lm", stroke_width=2, stroke_fill=_rgb(PAPER))

    # Rivers the line crosses, named only where they are within reach of it.
    km_x, km_y = cell_km(world)
    p0 = np.array(line["p0"], dtype=float)
    p1 = np.array(line["p1"], dtype=float)
    vec = p1 - p0
    denom = float(vec @ vec) or 1.0
    seen = 0
    for river in feats.rivers:
        q = np.array([float(river.x), float(river.y)])
        t = float((q - p0) @ vec) / denom
        if not (0.02 <= t <= 0.98):
            continue
        foot = p0 + t * vec
        dx_km = (q[0] - foot[0]) * km_x
        dy_km = (q[1] - foot[1]) * km_y
        if float(np.hypot(dx_km, dy_km)) > 18.0:
            continue
        xx = ex(t * total)
        base = ey(max(float(elev[int(t * (len(dist) - 1))]), 0.0))
        draw.line([(xx, base), (xx, base - 9)], fill=_rgb(INK_WATER), width=1)
        draw.text((xx, base - 12 - (seen % 2) * 20), river.name.title(),
                  font=font_tiny, fill=_rgb(INK_WATER), anchor="mb",
                  stroke_width=2, stroke_fill=_rgb(PAPER))
        seen += 1

    draw.text((x0, y1 + 40), line["ends"][0], font=font_tiny,
              fill=_rgb(INK_SOFT), anchor="lt")
    draw.text((x1, y1 + 40), line["ends"][1], font=font_tiny,
              fill=_rgb(INK_SOFT), anchor="rt")


def page(size, world, feats, name, seed=0):
    """A page of sections: two bands, each a measured profile with its axes."""
    w, h = size
    page_img = Image.new("RGB", size, _rgb(PAPER))
    draw = ImageDraw.Draw(page_img)

    ink = _rgb(INK)
    soft = _rgb(INK_SOFT)
    margin = int(w * 0.105)

    ft = decor.font("antiqua-bold", int(w * 0.030))
    head = f"SECTIONS ACROSS {name.upper()}"
    track = ft.size * 0.18
    limit = w * (1.0 - 2 * 0.105) * 0.98
    while ft.size > 14 and (sum(draw.textlength(ch, font=ft) for ch in head)
                            + track * (len(head) - 1)) > limit:
        ft = decor.font("antiqua-bold", ft.size - 2)
        track = ft.size * 0.18
    decor.tracked_text(draw, (w / 2, margin * 0.72), head, ft, ink, tracking=track)

    fs = decor.font("antiqua-italic", int(w * 0.0155))
    draw.text((w / 2, margin * 0.72 + ft.size * 1.45),
              "shewing the profile of the ground along two lines of section",
              font=fs, fill=soft, anchor="mm")

    rule_y = margin * 0.72 + ft.size * 2.10
    draw.line([(w * 0.30, rule_y), (w * 0.70, rule_y)], fill=ink, width=1)

    lines = choose(world, feats)
    if not lines:
        return page_img

    top = rule_y + margin * 0.55
    bottom = h - margin * 0.62
    band_h = (bottom - top) / len(lines)
    font_head = decor.font("serif-bold", int(w * 0.0125))
    font_small = decor.font("serif-italic", int(w * 0.0100))
    font_tiny = decor.font("antiqua", int(w * 0.0088))

    for i, line in enumerate(lines):
        by0 = top + i * band_h
        header = f"{line['label']}   {line['title'].upper()}"
        draw.text((margin, by0), header, font=font_head, fill=ink, anchor="lt")
        prof = line_profile(world, line["p0"], line["p1"])
        box = (margin + 74, by0 + 74, w - margin, by0 + band_h - 70)
        _draw_band(draw, box, prof, line, world, feats, font_small, font_tiny)

    return page_img
