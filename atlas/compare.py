"""The six countries compared, all drawn at one scale.

A book of plates shows each country by itself, at whatever scale fits its
sheet.  That is the right way to read a country and the wrong way to
understand a survey: the reader never learns whether Khuvarurkath is half
Chouchuldora or twice it.  This plate answers that by drawing every country at
the same number of pixels to the kilometre and stacking them in order of area.

No country is enlarged or reduced for effect.  The scale is set by the largest,
and the smallest is as small as it really is.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from . import decor
from .config import WorldSpec
from .features import cell_km
from .palette import PAPER, INK, INK_SOFT
from .world import build_world
from . import worldmap

_TINT = (0.870, 0.826, 0.720)
_TINT_ALT = (0.820, 0.800, 0.735)


def _rgb(c):
    return tuple(int(round(v * 255)) for v in c)


def _silhouette(seed, lat_top, lat_bottom):
    """A whole-country silhouette, its true area, and the scale to draw it by.

    The plate world is cut by its own neat line, so the comparison uses the
    same higher sea level as the world sheet: the whole country, not a crop.
    The mask is then stretched by the square root of the area it lost, so the
    countries still compare by their true measured area.
    """
    base = WorldSpec(seed=seed, width=240, height=163,
                     lat_top=lat_top, lat_bottom=lat_bottom)
    truth = build_world(base)
    true_mask = truth.height > 0
    tkm_x, tkm_y = cell_km(truth)
    true_area = float(true_mask.sum()) * tkm_x * tkm_y
    peak = float(truth.height.max())

    frac = worldmap._fit_fraction(seed, lat_top, lat_bottom)
    fit = WorldSpec(seed=seed, width=240, height=163,
                    lat_top=lat_top, lat_bottom=lat_bottom,
                    land_fraction=frac)
    world = build_world(fit)
    mask = world.height > 0
    km_x, km_y = cell_km(world)
    mask_area = max(1.0, float(mask.sum()) * km_x * km_y)
    k = (true_area / mask_area) ** 0.5
    return mask, km_x * k, km_y * k, peak, true_area


def _edge(mask):
    e = np.zeros_like(mask)
    e[:-1, :] |= mask[:-1, :] != mask[1:, :]
    e[-1, :] |= mask[-1, :]
    e[:, :-1] |= mask[:, :-1] != mask[:, 1:]
    e[:, -1] |= mask[:, -1]
    return e


def _draw_one(page, draw, box, mask, km_x, km_y, scale, name, area, peak, river, index):
    x0, y0, x1, y1 = box
    ys, xs = np.where(mask)
    if not len(xs):
        return
    m = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    wpx = max(1, int(round(m.shape[1] * km_x * scale)))
    hpx = max(1, int(round(m.shape[0] * km_y * scale)))
    # Leave the silhouette in its cell, low in the cell so the caption has air.
    ox = int(x0 + (x1 - x0 - wpx) / 2)
    oy = int(y0 + (y1 - y0) * 0.52 - hpx * 0.55)

    im = Image.fromarray((m.astype(np.uint8) * 255), mode="L").resize(
        (max(1, wpx), max(1, hpx)), Image.LANCZOS)
    fill = Image.new("RGB", (wpx, hpx), _rgb(_TINT if index % 2 == 0 else _TINT_ALT))
    page.paste(fill, (ox, oy), im)
    edge = Image.fromarray((_edge(np.asarray(im) > 110).astype(np.uint8) * 255), mode="L")
    page.paste(Image.new("RGB", (wpx, hpx), _rgb(INK)), (ox, oy), edge)

    # Name and measured facts.
    f = decor.font("antiqua-bold", max(15, int((x1 - x0) * 0.040)))
    draw.text(((x0 + x1) / 2, oy + hpx + 22), name.upper(), font=f,
              fill=_rgb(INK), anchor="ma")
    fs = decor.font("serif-italic", max(12, int((x1 - x0) * 0.031)))
    facts = f"{area:,} km²   ·   peak {peak:,} m   ·   river {river:,} km"
    draw.text(((x0 + x1) / 2, oy + hpx + 22 + f.size * 1.5), facts, font=fs,
              fill=_rgb(INK_SOFT), anchor="ma")


def page(size=(2400, 1700), countries=None):
    """Draw the comparison plate.  ``countries`` carries seed, latitudes, name,
    river and the pre-measured area and peak (or None to measure them here)."""
    countries = countries or []
    w, h = size
    page_img = Image.new("RGB", size, _rgb(PAPER))
    draw = ImageDraw.Draw(page_img)
    m = int(w * 0.095)

    measured = []
    for c in countries:
        mask, km_x, km_y, peak, area = _silhouette(c["seed"], c["lat_top"], c["lat_bottom"])
        measured.append(dict(c, mask=mask, km_x=km_x, km_y=km_y,
                             area=int(round(area)), peak=int(round(peak / 10) * 10)))
    measured.sort(key=lambda d: -d["area"])
    if not measured:
        return page_img

    f = decor.font("antiqua-bold", max(22, int(w * 0.030)))
    title = "THE SIX COUNTRIES COMPARED"
    decor.tracked_text(draw, (w / 2, m * 0.82), title, f, _rgb(INK),
                       tracking=f.size * 0.16, anchor_center=True)
    fs = decor.font("serif-italic", max(13, int(w * 0.0122)))
    draw.text((w / 2, m * 0.82 + f.size * 1.55),
              "each drawn at the same scale, in order of area", font=fs,
              fill=_rgb(INK_SOFT), anchor="ma")
    rule = m * 0.82 + f.size * 2.25
    draw.line([m, rule, w - m, rule], fill=(120, 108, 86), width=1)

    top = rule + m * 0.52
    bottom = h - m * 1.05
    cols, rows = 3, 2
    gap_x, gap_y = w * 0.025, h * 0.035
    cw = (w - 2 * m - gap_x * (cols - 1)) / cols
    ch = (bottom - top - gap_y * (rows - 1)) / rows

    # One scale for all: the widest country may use 84% of its cell.
    widest = max(d["mask"].shape[1] * d["km_x"] for d in measured)
    scale = (cw * 0.84) / widest

    for i, d in enumerate(measured):
        col, row = i % cols, i // cols
        box = (m + col * (cw + gap_x), top + row * (ch + gap_y),
               m + col * (cw + gap_x) + cw, top + row * (ch + gap_y) + ch)
        _draw_one(page_img, draw, box, d["mask"], d["km_x"], d["km_y"], scale,
                  d["name"], d["area"], d["peak"],
                  d.get("river", 0), i)

    # A scale bar in the foot, so the comparison can be checked.
    bar_km = 500
    bar_px = bar_km * scale
    bx, by = m, h - m * 0.60
    draw.line([bx, by, bx + bar_px, by], fill=_rgb(INK), width=2)
    for k in range(0, 6):
        xx = bx + bar_px * k / 5
        draw.line([(xx, by - 7), (xx, by + 7)], fill=_rgb(INK), width=1)
    fbar = decor.font("serif-italic", max(11, int(w * 0.0095)))
    draw.text((bx + bar_px + 14, by), "500 km", font=fbar, fill=_rgb(INK), anchor="lm")
    draw.text((w - m, by), "the same scale for every country", font=fbar,
              fill=_rgb(INK_SOFT), anchor="rm")
    return page_img


def stamp_folio(img, folio):
    draw = ImageDraw.Draw(img)
    w, h = img.size
    f = decor.font("antiqua", max(10, int(w * 0.0105)))
    draw.text((w / 2, int(h * 0.945)), str(folio), font=f,
              fill=_rgb(INK_SOFT), anchor="mm")
    return img
