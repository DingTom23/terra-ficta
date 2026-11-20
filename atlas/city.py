"""The capital, drawn as a town plan.

A country plate shows a capital as a dot.  This module draws the dot.  The
plan is invented, but it is not arbitrary: the town's site -- a river mouth, a
crossing, a lake outlet, a gap in the hills -- decides what the plan has to
contain, and the river, coast and hill in it are the same ones the country
plate and the dictionary describe.

The drawing is vector work at a scale no other page uses: walls with bastions,
a radial street net, a market square, a cathedral, a citadel, a harbour.  It
exists because a book of maps that never changes scale is a book of diagrams.
"""

from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw

from . import decor
from .palette import PAPER, INK, INK_SOFT, INK_WATER

# The plan covers 3.6 km east-west.  A town of this period is a twenty-minute
# walk across, and the sheet should show the fields beyond the wall.
SPAN_M = 3600.0
WATER = (0.760, 0.822, 0.842)
WATER_DEEP = (0.688, 0.772, 0.802)
FIELD = (0.895, 0.878, 0.800)


def _rgb(c):
    return tuple(int(round(v * 255)) for v in c)


def _rng(seed, town):
    return np.random.default_rng((int(seed) * 7919 + int(town.x) * 131
                                  + int(town.y) * 17) % (2 ** 31))


def _smooth(points, passes=2):
    """Chaikin corner cutting; keeps a wall or a road from looking drafted."""
    pts = [tuple(p) for p in points]
    for _ in range(passes):
        out = []
        for i in range(len(pts)):
            a, b = pts[i], pts[(i + 1) % len(pts)]
            out.append((0.75 * a[0] + 0.25 * b[0], 0.75 * a[1] + 0.25 * b[1]))
            out.append((0.25 * a[0] + 0.75 * b[0], 0.25 * a[1] + 0.75 * b[1]))
        pts = out
    return pts


def _wall(rng, radius, n=44):
    """An irregular enceinte, with the trace of eight bastions."""
    pts = []
    for i in range(n):
        a = 2 * math.pi * i / n
        r = radius * (1.0 + 0.055 * math.sin(3 * a + rng.uniform(0, 6))
                      + 0.035 * math.sin(7 * a + rng.uniform(0, 6)))
        pts.append((r * math.cos(a), r * math.sin(a)))
    return pts


def _star_fort(draw, x, y, r, ink, paper, angle=0.0, points=5):
    """A bastioned citadel: a star polygon with a dark trace."""
    pts = []
    for i in range(points * 2):
        a = angle + math.pi * i / points
        rad = r if i % 2 == 0 else r * 0.52
        pts.append((x + rad * math.cos(a), y + rad * math.sin(a)))
    draw.polygon(pts, fill=_rgb(paper), outline=_rgb(ink))
    draw.line(pts + [pts[0]], fill=_rgb(ink), width=3)


def _polyline(draw, pts, fill, width=1):
    draw.line([tuple(p) for p in pts], fill=fill, width=width, joint="curve")


def plan(world, feats, country, seed=0, size=(2400, 1700)):
    """Draw the capital of one country as a town plan."""
    town = feats.chief_town()
    if town is None:
        return None

    pw, ph = size
    margin = int(pw * 0.055)
    img = Image.new("RGB", size, _rgb(PAPER))
    draw = ImageDraw.Draw(img)
    rng = _rng(seed, town)

    # Map box: the plan is a rectangle, and the legend sits outside it.
    mx0, my0 = int(pw * 0.065), int(ph * 0.155)
    mx1, my1 = int(pw * 0.665), int(ph * 0.925)
    mw, mh = mx1 - mx0, my1 - my0
    scale = min(mw, mh) / (2.0 * SPAN_M)
    cx, cy = (mx0 + mx1) / 2.0, (my0 + my1) / 2.0

    def sx(xm):
        return cx + xm * scale

    def sy(ym):
        return cy - ym * scale

    tags = set(town.site_tags or ())
    coastal = bool(tags & {"harbour", "river_mouth"})
    has_river = bool(tags & {"river_mouth", "lake_outlet", "confluence",
                             "crossing", "harbour"}) or bool(town.river)
    lake = "lake_outlet" in tags
    hilly = "gate" in tags

    # --- ground ---------------------------------------------------------
    draw.rectangle([mx0, my0, mx1, my1], fill=_rgb(FIELD))
    # A faint contour or two under the town, if it stands in a gap.
    if hilly:
        for rr in (1500, 1180, 900):
            yy = sy(300) - rr * scale * 0.18
            draw.arc([sx(-rr), yy - rr * scale * 0.55, sx(rr), yy + rr * scale * 0.55],
                     200, 340, fill=_rgb((0.78, 0.74, 0.66)), width=1)

    # Water: sea first, then a river through the town.
    if coastal:
        # The sea lies beyond the south-east wall, not across it: a town wall
        # that runs into open water reads as a drawing error, not a harbour.
        sea_m = [(SPAN_M, -SPAN_M), (1500, -SPAN_M), (1750, -2600),
                 (2050, -2000), (2600, -1500), (SPAN_M, -900)]
        sea = [(sx(x), sy(y)) for x, y in sea_m]
        draw.polygon(sea, fill=_rgb(WATER))
        draw.line(sea, fill=_rgb(INK_WATER), width=2)

    river_pts = []
    if has_river:
        # The river enters from the west, bends through the town and leaves
        # at the sea if there is one, otherwise to the east.
        pts_m = [(-SPAN_M * 0.55, 330), (-1250, 290), (-820, 210), (-430, 120),
                 (-120, 30), (180, -30), (520, -120), (980, -210), (SPAN_M * 0.55, -280)]
        if coastal:
            pts_m = pts_m[:4] + [(900, -420), (1500, -1200), (1900, -1700), (2150, -1980)]
        pts = [(sx(x), sy(y)) for x, y in pts_m]
        river_pts = _smooth(pts, passes=2)
        width = max(8, int(34 * scale))
        _polyline(draw, river_pts, _rgb(WATER_DEEP), width + 4)
        _polyline(draw, river_pts, _rgb(WATER), width)
        # banks
        _polyline(draw, river_pts, _rgb(INK_WATER), 1)

    if lake:
        lx, ly = sx(-1150), sy(600)
        draw.ellipse([lx - 300 * scale, ly - 150 * scale,
                      lx + 300 * scale, ly + 150 * scale], fill=_rgb(WATER),
                     outline=_rgb(INK_WATER), width=2)

    # --- wall trace -------------------------------------------------------
    R = 2400.0
    wall = _smooth(_wall(rng, R), passes=1)
    wall_px = [(sx(x), sy(y)) for x, y in wall]

    # Citadel in the north-west, inside the wall.
    cit = (sx(-R * 0.58), sy(R * 0.50))
    _star_fort(draw, cit[0], cit[1], 90 * scale, INK, PAPER, angle=0.2)

    # --- streets ---------------------------------------------------------
    gates = [30, 90, 150, 210, 270, 330]
    gate_pts = []
    for a in gates:
        ang = math.radians(a)
        gate_pts.append((math.cos(ang) * R, math.sin(ang) * R))

    primary = []
    for (gx, gy) in gate_pts:
        pts = []
        for t in np.linspace(0.0, 1.0, 16):
            bend = 0.06 * math.sin(math.pi * t) * (-gy if abs(gx) > abs(gy) else gx)
            pts.append((sx(gx * t + bend), sy(gy * t)))
        primary.append(pts)

    rings = []
    for frac in (0.34, 0.62, 0.86):
        ring = []
        for i in range(48):
            a = 2 * math.pi * i / 48
            rr = R * frac * (1 + 0.035 * math.sin(5 * a + frac * 9))
            ring.append((sx(rr * math.cos(a)), sy(rr * math.sin(a))))
        rings.append(ring)

    spokes = []
    for a in np.linspace(0, 360, 18, endpoint=False):
        if min(abs(a - g) for g in gates) < 8:
            continue
        ang = math.radians(a)
        spokes.append(((sx(math.cos(ang) * R * 0.18), sy(math.sin(ang) * R * 0.18)),
                       (sx(math.cos(ang) * R * 0.94), sy(math.sin(ang) * R * 0.94))))

    # --- buildings -------------------------------------------------------
    # A polar field of small blocks, drawn in a soft ink so the street net
    # can be cut back through it afterwards.  Solid black confetti reads as
    # a stain, not as a town.
    ink = _rgb(INK)
    body_ink = _rgb((0.545, 0.515, 0.460))
    for r in np.arange(120.0, R * 0.965, 50.0):
        nang = max(24, int(2 * math.pi * r / 36.0))
        for j in range(nang):
            a = 2 * math.pi * j / nang
            if min(abs(math.degrees(a) - g) for g in gates) < 3.0:
                continue
            if min(abs(r - R * f) for f in (0.34, 0.62, 0.86)) < 20:
                continue
            if rng.random() > 0.88 - 0.34 * (r / R):
                continue
            xm, ym = r * math.cos(a), r * math.sin(a)
            px, py = sx(xm), sy(ym)
            w = max(2.0, (9.0 + rng.uniform(0, 9)) * scale)
            h = max(2.0, (6.0 + rng.uniform(0, 7)) * scale)
            ca, sa = math.cos(a), math.sin(a)
            rect = [(px + w * ca - h * sa, py + w * sa + h * ca),
                    (px - w * ca - h * sa, py - w * sa + h * ca),
                    (px - w * ca + h * sa, py - w * sa - h * ca),
                    (px + w * ca + h * sa, py + w * sa - h * ca)]
            draw.polygon(rect, fill=body_ink)

    # Suburbs along the roads outside the wall: a few clusters, not a smear.
    for (gx, gy) in gate_pts:
        for t in (1.08, 1.18, 1.30, 1.45):
            bx, by = sx(gx * t), sy(gy * t)
            for _ in range(7):
                ox, oy = rng.normal(0, 16 * scale, 2)
                w = rng.uniform(4, 10) * scale
                h = rng.uniform(4, 8) * scale
                draw.rectangle([bx + ox, by + oy, bx + ox + w, by + oy + h],
                               fill=body_ink)

    # --- cut the streets back through the blocks -------------------------
    paper = _rgb(PAPER)
    for pts in primary:
        _polyline(draw, pts, paper, max(7, int(23 * scale)))
        _polyline(draw, pts, _rgb(INK_SOFT), 1)
    for ring in rings:
        _polyline(draw, ring + [ring[0]], paper, max(5, int(15 * scale)))
        _polyline(draw, ring + [ring[0]], _rgb((0.62, 0.58, 0.52)), 1)
    for p0, p1 in spokes:
        draw.line([p0, p1], fill=paper, width=max(2, int(8 * scale)))
        draw.line([p0, p1], fill=_rgb((0.64, 0.60, 0.54)), width=1)

    # The river cuts the town as well; draw it again over the blocks.
    if has_river and river_pts:
        width = max(8, int(34 * scale))
        _polyline(draw, river_pts, paper, width + 8)
        _polyline(draw, river_pts, _rgb(WATER_DEEP), width + 4)
        _polyline(draw, river_pts, _rgb(WATER), width)
        _polyline(draw, river_pts, _rgb(INK_WATER), 1)

    # --- wall, drawn last so it is never buried --------------------------
    for w, c in ((9, (0.86, 0.84, 0.76)), (6, (0.70, 0.66, 0.57)), (2, INK)):
        draw.line(wall_px + [wall_px[0]], fill=_rgb(c), width=w, joint="curve")

    # Bastions: triangular salients sitting on the trace, eight of them.
    for k in range(8):
        a = 2 * math.pi * k / 8 + 0.15
        ca, sa = math.cos(a), math.sin(a)
        bx, by = sx(ca * R * 0.985), sy(sa * R * 0.985)
        half = 34 * scale
        depth = 62 * scale
        pts = [(bx - half * sa, by + half * ca),
               (bx + depth * ca, by - depth * sa),
               (bx + half * sa, by - half * ca)]
        draw.polygon(pts, fill=_rgb(PAPER), outline=ink)
        draw.line(pts + [pts[0]], fill=ink, width=2)

    # --- landmarks -------------------------------------------------------
    # Market square: an open, slightly irregular place at the centre.
    sq = [(sx(-95 + 8 * math.sin(i)), sy(-60 + 8 * math.cos(i * 1.3)))
          for i in range(7)]
    draw.polygon(sq, fill=_rgb(PAPER), outline=_rgb(INK))
    fx, fy = sx(0), sy(0)
    draw.ellipse([fx - 3 * scale, fy - 3 * scale, fx + 3 * scale, fy + 3 * scale],
                 outline=_rgb(INK), width=2)

    # Cathedral: a cross-shaped block east of the market.
    cat = (sx(260), sy(70))
    cw, ch = 48 * scale, 15 * scale
    draw.rectangle([cat[0] - cw, cat[1] - ch, cat[0] + cw, cat[1] + ch], fill=ink)
    draw.rectangle([cat[0] - 10 * scale, cat[1] - 34 * scale,
                    cat[0] + 10 * scale, cat[1] + 34 * scale], fill=ink)
    draw.rectangle([cat[0] - 10 * scale, cat[1] - 34 * scale,
                    cat[0] + 10 * scale, cat[1] + 34 * scale], fill=ink)

    # Town hall, west of the market.
    th = (sx(-210), sy(-110))
    draw.rectangle([th[0] - 34 * scale, th[1] - 16 * scale,
                    th[0] + 34 * scale, th[1] + 16 * scale], fill=ink)

    # Harbour: piers reaching into the sea.
    piers = []
    if coastal:
        for k in range(3):
            px = sx(2050 + 130 * k)
            py = sy(-700 - 20 * k)
            draw.line([(px, py), (px + 10 * scale, py + 95 * scale)],
                      fill=_rgb(INK), width=3)
            piers.append((px + 5 * scale, py + 58 * scale))

    # Bridge: where the main north-south road crosses the river.
    if has_river and river_pts:
        # Find the river point nearest the centre line.
        best = min(river_pts, key=lambda p: abs(p[0] - sx(0)))
        bx, by = best
        draw.rectangle([bx - 11 * scale, by - 26 * scale,
                        bx + 11 * scale, by + 26 * scale], fill=_rgb(PAPER))
        draw.rectangle([bx - 11 * scale, by - 26 * scale,
                        bx + 11 * scale, by + 26 * scale], outline=ink, width=2)
        for k in range(-3, 4):
            draw.line([(bx + k * 3 * scale, by - 26 * scale),
                       (bx + k * 3 * scale, by + 26 * scale)], fill=ink, width=1)

    # --- fields and trees outside the wall --------------------------------
    for k in range(120):
        a = rng.uniform(0, 2 * math.pi)
        r = rng.uniform(R * 1.03, min(R * 1.28, SPAN_M * 0.94))
        x, y = sx(r * math.cos(a)), sy(r * math.sin(a))
        if not (mx0 < x < mx1 and my0 < y < my1):
            continue
        if rng.random() < 0.55:
            draw.arc([x - 7, y - 3, x + 7, y + 3], 0, 180,
                     fill=_rgb((0.74, 0.70, 0.60)), width=1)

    # --- furniture -------------------------------------------------------
    # Double rule around the plate.
    for inset, w in ((margin, 2), (margin + 9, 1)):
        draw.rectangle([inset, inset, pw - inset, ph - inset],
                       outline=ink, width=w)

    # Title block above the plan.
    f_title = decor.font("antiqua-bold", int(pw * 0.026))
    f_sub = decor.font("serif-italic", int(pw * 0.0105))
    title = f"THE CITY OF {town.name.upper()}"
    draw.text((mx0, my0 - 126), title, font=f_title, fill=ink)
    draw.text((mx0, my0 - 46), f"the capital of {country}, shewing the streets and principal buildings",
              font=f_sub, fill=_rgb(INK_SOFT))
    draw.line([(mx0, my0 - 16), (mx1, my0 - 16)], fill=ink, width=1)

    # Compass rose in the bottom-left field, outside the town.
    decor.draw_compass(draw, mx0 + 70, my1 - 80, int(pw * 0.020), ink, _rgb(PAPER))

    # Scale bar: 500 metres in four parts.
    bar_m = 500.0
    bar_w = bar_m * scale
    bx, by = mx0 + 150, my1 - 34
    decor.draw_scale_bar(draw, bx, by, bar_w, ink, _rgb(PAPER), units="500 metres",
                         segments=4, height=max(8, int(pw * 0.0045)))

    # Legend panel at the right.
    lx0, ly0 = int(pw * 0.700), int(ph * 0.140)
    lx1, ly1 = int(pw * 0.935), int(ph * 0.860)
    draw.rectangle([lx0, ly0, lx1, ly1], outline=ink, width=2)
    draw.rectangle([lx0 + 5, ly0 + 5, lx1 - 5, ly1 - 5], outline=ink, width=1)
    fh = decor.font("antiqua-bold", int(pw * 0.013))
    fb = decor.font("serif", int(pw * 0.0095))
    draw.text((lx0 + 26, ly0 + 24), "A KEY TO THE CITY", font=fh, fill=ink)
    draw.line([(lx0 + 26, ly0 + 24 + fh.size + 10), (lx1 - 26, ly0 + 24 + fh.size + 10)],
              fill=ink, width=1)

    legend = [
        ("1", "The cathedral"),
        ("2", "The market square"),
        ("3", "The town hall"),
        ("4", "The citadel"),
        ("5", "The river bridge" if has_river else "The north gate"),
        ("6", "The harbour" if coastal else "The west gate"),
        ("7", "The walls and bastions"),
        ("8", "The suburb beyond the gate"),
    ]
    ly = ly0 + 78
    for num, text in legend:
        draw.ellipse([lx0 + 30, ly + 1, lx0 + 30 + fb.size * 1.25, ly + 1 + fb.size * 1.25],
                     outline=ink, width=1)
        draw.text((lx0 + 30 + fb.size * 0.62, ly + 2 + fb.size * 0.62), num,
                  font=decor.font("serif", max(8, int(pw * 0.006))), fill=ink,
                  anchor="mm")
        draw.text((lx0 + 30 + fb.size * 2.0, ly), text, font=fb, fill=ink)
        ly += fb.size * 2.4

    # A locator: the country's own silhouette with the capital marked.
    iw, ih = 190, 128
    land = (world.height > 0)
    sil = Image.fromarray((land * 255).astype(np.uint8)).resize((iw, ih), Image.BILINEAR)
    arr = np.asarray(sil)
    patch = np.full((ih, iw, 3), np.array(PAPER) * 255.0)
    patch[arr > 110] = np.array(INK) * 255.0
    px0, py0 = int(lx0 + 30), int(ly + 26)
    draw.text((px0, py0 - 22), "THE COUNTRY", font=decor.font("antiqua-bold", int(pw * 0.0095)),
              fill=_rgb(INK_SOFT))
    img.paste(Image.fromarray(patch.astype(np.uint8)), (px0, py0))
    hh, ww = world.height.shape
    dx = int(px0 + (town.x / max(1, ww - 1)) * iw)
    dy = int(py0 + (town.y / max(1, hh - 1)) * ih)
    draw.ellipse([dx - 3, dy - 3, dx + 3, dy + 3], outline=ink, width=2)

    if "river_mouth" in tags:
        site_line = "It stands where the river reaches the sea."
    elif "confluence" in tags:
        site_line = "It stands where two rivers meet."
    elif "lake_outlet" in tags:
        site_line = "It stands at the outlet of the lake."
    elif "crossing" in tags:
        site_line = "It stands at the crossing of the river."
    elif "gate" in tags:
        site_line = "It stands at a gap in the hills."
    else:
        site_line = "It stands in open country."
    notes = [
        site_line,
        f"The enceinte encloses {math.pi * (R * R) / 10000.0:.0f} hectares.",
        f"The six gates open on the roads to the country.",
    ]
    fy0 = py0 + ih + 30
    fnote = decor.font("serif-italic", int(pw * 0.0092))
    for k, line in enumerate(notes):
        draw.text((px0, fy0 + k * fnote.size * 1.8), line, font=fnote,
                  fill=_rgb(INK_SOFT))

    # The panel has room below the key; fill it with the town's own numbers,
    # so the sheet agrees with the dictionary entry for the same place.
    fs = decor.font("serif-italic", int(pw * 0.0095))
    province = ""
    if int(town.prov) >= 0:
        province = next((p.name for p in feats.provinces
                         if p.index == int(town.prov)), "")
    facts = [
        f"population {int(round(town.pop, -2)):,}" if town.pop else "",
        f"{int(round(town.elev_m))} m above the sea" if town.elev_m else "",
        f"on the {town.river}" if town.river else "",
        f"the seat of {province.title()}" if province else "",
    ]
    fy = ly1 - 150
    draw.line([(lx0 + 26, fy - 18), (lx1 - 26, fy - 18)], fill=ink, width=1)
    for line in [v for v in facts if v][:4]:
        draw.text((lx0 + 30, fy), line, font=fs, fill=_rgb(INK_SOFT))
        fy += fs.size * 1.9

    # Numbered marks on the plan.
    marks = [(1, cat), (2, (sx(0), sy(0))), (3, th), (4, cit)]
    if has_river and river_pts:
        best = min(river_pts, key=lambda p: abs(p[0] - sx(0)))
        marks.append((5, (best[0] + 20, best[1] - 16)))
    if coastal and piers:
        marks.append((6, piers[0]))
    marks.append((7, (sx(-R * 0.98), sy(R * 0.10))))
    for num, (x, y) in marks:
        draw.ellipse([x - 9, y - 9, x + 9, y + 9], fill=_rgb(PAPER), outline=ink, width=1)
        draw.text((x, y), str(num), font=decor.font("serif", max(8, int(pw * 0.006))),
                  fill=ink, anchor="mm")

    # The map has its own neat line; the page frame is outside it.
    draw.rectangle([mx0, my0, mx1, my1], outline=_rgb(INK_SOFT), width=2)

    # Plate number in the top right, if the book supplies one later.
    return img


def stamp_folio(img, folio):
    """Print a folio on a copy of the plan, in the same place as other pages."""
    out = img.copy()
    d = ImageDraw.Draw(out)
    f = decor.font("serif", max(14, int(out.size[0] * 0.009)))
    d.text((out.size[0] / 2, out.size[1] * 0.955), str(folio), font=f,
           fill=_rgb(INK), anchor="mm")
    return out
