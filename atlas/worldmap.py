"""The known world: the six countries placed on one sheet.

Each country is generated on its own grid and knows only its own latitude
band.  This module puts them together: it takes each country's land mask and
relief, scales it to the width its latitude span implies on an equirectangular
sheet, and sets it at a longitude of its own.  The result is not a claim that
the six are one continent -- it is the chart that says where each of them lies,
which the book has been missing.

Nothing here invents geography.  Every coastline is the same coastline the
plates draw, sampled from the same height field; only the longitudes are
arbitrary, and the sheet says so in its subtitle.
"""

from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw

from . import decor
from .config import WorldSpec
from .palette import PAPER, INK, INK_SOFT, INK_WATER, hypsometric
from .shade import hillshade
from .world import build_world

SEA = (0.858, 0.882, 0.888)
SEA_DEEP = (0.812, 0.850, 0.864)
COAST = (0.278, 0.256, 0.216)

# The world sheet is 180 degrees of longitude by 90 of latitude, at twenty
# pixels to the degree, with a margin for the graduated border.
WORLD_LON = 180.0
WORLD_LAT = 90.0
PPD = 20.0
MARGIN = 150.0

# Where each country is set.  The order is the order of the plates; the
# longitudes are spaced so no two grids overlap.
LON_STARTS = (4.0, 33.0, 62.0, 91.0, 120.0, 149.0)


def _rgb(c):
    return tuple(int(round(v * 255)) for v in c)


def _fit_fraction(seed, lat_top, lat_bottom):
    """Find a sea level that leaves the whole country inside its grid.

    On the plates a country is deliberately cropped by the neat line; on the
    world sheet a country cut by a ruler-straight edge would read as a torn
    map.  Raising the sea level a little shrinks the land until it clears the
    frame, which is the small-scale generalisation a world chart would use.
    """
    hh, ww = 163, 240
    for frac in np.arange(0.34, 0.155, -0.02):
        w = build_world(WorldSpec(seed=seed, width=ww, height=hh,
                                  lat_top=lat_top, lat_bottom=lat_bottom,
                                  land_fraction=float(frac)))
        land = w.height > 0
        edge = int(land[0].sum() + land[-1].sum() + land[:, 0].sum() + land[:, -1].sum())
        if edge == 0:
            return float(frac)
    return 0.18


def _country_image(seed, lat_top, lat_bottom, tw, th):
    """Render one country's land as an RGBA patch at its world size."""
    # Drawn at twice the resolution it is used at: a country is a few hundred
    # pixels wide on the world sheet, and a 480-cell height field upsampled to
    # that width shows its cells.  The terrain is scale-free, so the shape is
    # the same; only the grain is finer.
    frac = _fit_fraction(seed, lat_top, lat_bottom)
    spec = WorldSpec(seed=seed, width=960, height=652,
                     lat_top=lat_top, lat_bottom=lat_bottom,
                     land_fraction=frac)
    world = build_world(spec)
    h = world.height
    shade = hillshade(h / spec.meters, z_factor=130.0, azimuth=315.0, altitude=40.0)
    land = h > 0

    rgb = hypsometric(np.maximum(h, 0.0)) * (0.60 + 0.48 * shade)[..., None]
    ice = (world.climate or {}).get("ice")
    if ice is not None:
        cold = np.array([0.952, 0.960, 0.962], dtype=np.float32)
        m = ice[..., None].astype(np.float32)
        rgb = rgb * (1.0 - m) + cold * m

    def resize(a, mode=Image.LANCZOS):
        im = Image.fromarray(np.asarray(a, dtype=np.float32), mode="F")
        return np.asarray(im.resize((tw, th), mode), dtype=np.float32)

    rgb = np.stack([resize(rgb[..., k]) for k in range(3)], axis=-1)
    m = resize(land.astype(np.float32))
    alpha = np.clip((m - 0.45) * 5.0, 0.0, 1.0)
    # A country whose land runs to the edge of its own survey grid would be
    # printed with a ruler-straight cut.  Feather the last few per cent so the
    # sheet says "the survey stops here", not "the country was torn".
    yy = np.minimum(np.arange(th), np.arange(th)[::-1])[:, None].astype(np.float32)
    xx = np.minimum(np.arange(tw), np.arange(tw)[::-1])[None, :].astype(np.float32)
    feather = np.clip(np.minimum(xx, yy) / max(6.0, 0.045 * min(tw, th)), 0.0, 1.0)
    alpha = alpha * feather

    patch = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8), mode="RGB")
    amask = Image.fromarray((alpha * 255).astype(np.uint8), mode="L")
    patch.putalpha(amask)
    return patch, alpha


def _coast_mask(alpha, tw, th):
    """One-pixel outline around the land, from the same resampled mask."""
    m = alpha > 0.5
    edge = np.zeros_like(m)
    edge[:-1, :] |= m[:-1, :] != m[1:, :]
    edge[-1, :] |= m[-1, :]
    edge[:, :-1] |= m[:, :-1] != m[:, 1:]
    edge[:, -1] |= m[:, -1]
    # Do not outline the grid edge: the sea beyond it is the world ocean.
    edge[0, :] = edge[-1, :] = False
    edge[:, 0] = edge[:, -1] = False
    return edge


def _nice_league_bar(leagues_across, avail_px, content_px):
    target = avail_px / content_px * leagues_across
    for mult in (1, 2, 5):
        for power in range(0, 6):
            count = mult * 10 ** power
            if count >= target * 0.55 and count <= target * 1.05:
                return count
    return max(10, int(round(target / 10.0)) * 10)


def page(size=(3600, 1950), countries=None):
    """Draw the world sheet.

    ``countries`` is a list of dicts with ``seed``, ``lat_top``, ``lat_bottom``
    and ``name``, in plate order.
    """
    countries = countries or []
    plate_w, plate_h = size
    page_img = Image.new("RGB", size, _rgb(PAPER))
    draw = ImageDraw.Draw(page_img)

    # The map box: equal pixels per degree in both directions.
    box_l, box_t, box_r, box_b = MARGIN, MARGIN, plate_w - MARGIN, plate_h - MARGIN
    bw, bh = box_r - box_l, box_b - box_t
    px, py = bw / WORLD_LON, bh / WORLD_LAT
    assert abs(px - py) < 0.02 * px, "the world sheet must be equirectangular"

    def wx(lon):
        return box_l + lon * px

    def wy(lat):
        return box_b - lat * py

    # Furniture is fixed before anything is drawn, so the country names can be
    # kept out of it.  The ocean is the only place any of it can go.
    cart = (wx(3.0), wy(89.0), wx(52.0), wy(79.0))
    comp_r = int(plate_w * 0.016)
    comp_c = (wx(12.0), wy(14.0))
    key_rect = (wx(122.0), wy(24.0), wx(178.0), wy(6.0))
    bar_rect = (wx(60.0), wy(7.0), wx(99.0), wy(2.0))
    obstacles = [cart, key_rect, bar_rect,
                 (comp_c[0] - comp_r, comp_c[1] - comp_r,
                  comp_c[0] + comp_r, comp_c[1] + comp_r)]

    draw.rectangle([box_l, box_t, box_r, box_b], fill=_rgb(SEA))

    # Each country, scaled to the width its latitude span implies.
    items = []
    for i, c in enumerate(countries):
        if i >= len(LON_STARTS):
            break
        span_lat = c["lat_top"] - c["lat_bottom"]
        span_lon = span_lat * (480.0 / 326.0)
        tw = int(round(span_lon * px))
        th = int(round(span_lat * py))
        patch, alpha = _country_image(c["seed"], c["lat_top"], c["lat_bottom"], tw, th)
        left = int(round(wx(LON_STARTS[i])))
        top = int(round(wy(c["lat_top"])))
        page_img.paste(patch, (left, top), patch)

        edge = _coast_mask(alpha, tw, th)
        coast = Image.fromarray((edge * 255).astype(np.uint8), mode="L")
        page_img.paste(Image.new("RGB", (tw, th), _rgb(COAST)), (left, top), coast)
        items.append({"c": c, "left": left, "top": top, "alpha": alpha})

    # Rhumb lines across the open water: the web of a portolan chart is what
    # makes an empty ocean look surveyed rather than unfinished.  They stop at
    # every coast.
    water = Image.new("L", size, 255)
    wdraw = ImageDraw.Draw(water)
    for it in items:
        mask = Image.fromarray(((it["alpha"] > 0.45) * 255).astype(np.uint8), mode="L")
        wdraw.bitmap((it["left"], it["top"]), mask, fill=0)
    page_img = decor.draw_rhumbs(page_img, comp_c, (box_l, box_t, box_r, box_b),
                                 water, _rgb(INK_SOFT), alpha=0.085, count=16)
    draw = ImageDraw.Draw(page_img)

    # Where the land actually lies, so a name does not sit on a neighbour's
    # coast: on a crowded sheet the label of one country can touch the shore
    # of another even when it clears its own.
    land_boxes = []
    for it in items:
        ys, xs = np.where(it["alpha"] > 0.5)
        it["bbox"] = (it["left"] + int(xs.min()), it["top"] + int(ys.min()),
                      it["left"] + int(xs.max()), it["top"] + int(ys.max())) if len(xs) else None
        if it["bbox"]:
            land_boxes.append(it["bbox"])

    # Names, placed afterwards with every obstacle on the sheet known.
    label_rects = []
    for i, it in enumerate(items):
        if not it["bbox"]:
            continue
        c = it["c"]
        name = f"{_roman(i + 1)}  {c['name'].upper()}"
        bx0, by0, bx1, by1 = it["bbox"]
        cx = (bx0 + bx1) / 2.0
        base = max(12, int(round(plate_w * 0.0067)))
        for fsize in (base, int(base * 0.88), int(base * 0.77)):
            f = decor.font("antiqua-bold", fsize)
            tw_name = sum(draw.textlength(ch, font=f) for ch in name)
            tw_name += fsize * 0.18 * max(0, len(name) - 1)
            ov, un = fsize * 1.15, fsize * 1.35
            over = (by0 - ov, by1 + un, by0 - ov * 2, by1 + un * 2, by0 - ov * 3)
            under = (by1 + un, by0 - ov, by1 + un * 2, by0 - ov * 2, by1 + un * 3)
            for ty in (over if i % 2 == 0 else under):
                tx = min(max(cx, box_l + 8 + tw_name / 2), box_r - 8 - tw_name / 2)
                rect = (tx - tw_name / 2, ty - fsize * 0.75,
                        tx + tw_name / 2, ty + fsize * 0.75)
                if rect[1] < box_t + 8 or rect[3] > box_b - 8:
                    continue
                bad = list(obstacles) + [b for b in land_boxes if b != it["bbox"]]
                if any(not (rect[2] < r[0] or rect[0] > r[2] or
                            rect[3] < r[1] or rect[1] > r[3]) for r in bad):
                    continue
                # Two names may not merely fail to overlap; they must read as
                # two names, so leave a few characters of air between them.
                if any(not (rect[2] + 30 < r[0] or rect[0] - 30 > r[2] or
                            rect[3] + 8 < r[1] or rect[1] - 8 > r[3])
                       for r in label_rects):
                    continue
                decor.tracked_text(draw, (tx, ty), name, f, _rgb(INK),
                                   tracking=f.size * 0.18, anchor_center=True)
                label_rects.append(rect)
                break
            else:
                continue
            break

    # Graticule and frame.  The graticule is composited so it sits in the sea
    # at a few per cent, not painted over it.
    page_img = decor.draw_graticule(page_img, (box_l, box_t, box_r, box_b),
                                    WORLD_LAT, 0.0, 0.0, WORLD_LON,
                                    _rgb(INK), step=15.0, alpha=0.09)
    draw = ImageDraw.Draw(page_img)
    decor.draw_frame(draw, (box_l, box_t, box_r, box_b), MARGIN * 1.0,
                     _rgb(INK), _rgb(PAPER),
                     lat_top=WORLD_LAT, lat_bottom=0.0,
                     lon_left=0.0, lon_right=WORLD_LON, step=30.0, sub=4)
    decor.draw_graticule_labels(draw, (box_l, box_t, box_r, box_b), MARGIN * 1.0,
                                WORLD_LAT, 0.0, 0.0, WORLD_LON,
                                _rgb(INK), step=30.0, size=17)

    # The empty water is named too: a period chart never leaves a blank sea
    # without telling you which sea it is.
    sea_f = decor.font("antiqua-italic", max(15, int(plate_w * 0.0062)))
    for text, sx, sy in (("THE NORTHERN OCEAN", wx(118.0), wy(82.0)),
                         ("THE WESTERN OCEAN", wx(24.0), wy(30.0)),
                         ("THE SOUTHERN OCEAN", wx(66.0), wy(9.2))):
        decor.tracked_text(draw, (sx, sy), text, sea_f,
                           _rgb((0.545, 0.585, 0.610)),
                           tracking=sea_f.size * 0.42, anchor_center=True)

    decor.draw_cartouche(draw, cart, "THE KNOWN WORLD",
                         "the six countries of the survey, set at their latitudes",
                         _rgb(INK), _rgb(PAPER),
                         title_size=int(plate_w * 0.0135), sub_size=int(plate_w * 0.0052))

    decor.draw_compass(draw, comp_c[0], comp_c[1], comp_r, _rgb(INK), _rgb(PAPER))

    # Scale of leagues at the latitude of the middle of the sheet.
    mid_lat = 45.0
    leagues_across = WORLD_LON * 111.32 * math.cos(math.radians(mid_lat)) / 5.556
    count = _nice_league_bar(leagues_across, bw * 0.22, bw)
    bar_w = bw * (count / leagues_across)
    bx, by = wx(62.0), wy(4.2)
    bh_px = max(9, int(plate_w * 0.004))
    decor.draw_scale_bar(draw, bx, by, bar_w, _rgb(INK), _rgb(PAPER),
                         units=f"leagues ({count})", segments=4, height=bh_px)
    f = decor.font("serif-italic", max(11, int(plate_w * 0.0046)))
    draw.text((bx, by + bh_px + 8), "at 45 degrees north", font=f,
              fill=_rgb(INK_SOFT), anchor="lt")

    # Key to the countries, in the southern ocean at the right.
    if countries:
        fk = decor.font("serif", max(14, int(plate_w * 0.0060)))
        kx, ky = wx(124.0), wy(20.0)
        draw.text((kx, ky - 26), "THE SIX COUNTRIES",
                  font=decor.font("antiqua-bold", max(14, int(plate_w * 0.0060))),
                  fill=_rgb(INK))
        for i, c in enumerate(countries):
            draw.text((kx, ky + i * (fk.size * 1.75)),
                      f"{_roman(i + 1)}  {c['name']}", font=fk, fill=_rgb(INK))

    return page_img


def _roman(n):
    vals = ((10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I"))
    out = ""
    for v, s in vals:
        while n >= v:
            out += s
            n -= v
    return out


def locator(lat_top, lat_bottom, lon_start, size=96, paper=PAPER):
    """A small globe showing where one country sits on the world sheet.

    An orthographic view of the northern hemisphere is more honest than a
    world map shrunk to a stamp: it shows the country as a patch on a curve,
    which is what it is.
    """
    s = size
    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    cx = cy = s / 2.0
    r = s / 2.0 - 2
    d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=_rgb(INK), width=2,
              fill=_rgb(SEA))
    # Latitude and longitude rings of the orthographic hemisphere.
    for lat in (0, 30, 60):
        rr = r * math.cos(math.radians(lat))
        yy = cy - r * math.sin(math.radians(lat))
        d.ellipse([cx - rr, yy - rr * 0.18, cx + rr, yy + rr * 0.18],
                  outline=_rgb(INK_SOFT), width=1)
    for lon in range(-60, 61, 30):
        x = cx + r * math.sin(math.radians(lon)) * 0.9
        d.line([(x, cy - r * 0.95), (x, cy + r * 0.95)], fill=_rgb(INK_SOFT), width=1)
    # The country: a small filled wedge at its latitude.
    lat_mid = (lat_top + lat_bottom) / 2.0
    yy = cy - r * math.sin(math.radians(lat_mid))
    half = max(4.0, (lat_top - lat_bottom) / 90.0 * r * 0.55)
    xx = cx + r * math.sin(math.radians(lon_start - 45.0)) * 0.55
    d.ellipse([xx - half, yy - half * 0.55, xx + half, yy + half * 0.55],
              fill=_rgb((0.62, 0.38, 0.28)), outline=_rgb(INK), width=1)
    return img
