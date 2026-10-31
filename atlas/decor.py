"""Cartographic furniture: the frame, the grid, the compass, the title.

A relief image becomes a map at the moment it acquires a border, a scale, and
a name.  Everything in this module exists to make that transition happen.
"""

from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFont

FONT_DIR = "C:/Windows/Fonts"
_SERIF = "GARA.TTF"
_SERIF_BOLD = "GARABD.TTF"
_SERIF_ITALIC = "GARAIT.TTF"
_ANTIQUA = "BOOKOS.TTF"
_ANTIQUA_BOLD = "BOOKOSB.TTF"
_ANTIQUA_ITALIC = "BOOKOSI.TTF"

_cache = {}


def font(kind: str, size: int):
    path = {
        "serif": _SERIF,
        "serif-bold": _SERIF_BOLD,
        "serif-italic": _SERIF_ITALIC,
        "antiqua": _ANTIQUA,
        "antiqua-bold": _ANTIQUA_BOLD,
        "antiqua-italic": _ANTIQUA_ITALIC,
    }[kind]
    key = (kind, size)
    if key not in _cache:
        _cache[key] = ImageFont.truetype(f"{FONT_DIR}/{path}", size)
    return _cache[key]


def _rgb(c):
    """Accept a 0..1 triple or an already-quantised 0..255 one."""
    if max(c) > 1.5:
        return tuple(int(round(v)) for v in c)
    return tuple(int(round(v * 255)) for v in c)


def tracked_text(draw, xy, text, fnt, fill, tracking=0.0, anchor_center=True,
                 stroke_width=0, stroke_fill=None):
    """Draw text with extra letter spacing -- the look of engraved titling."""
    widths = [draw.textlength(ch, font=fnt) for ch in text]
    total = sum(widths) + tracking * max(0, len(text) - 1)
    x, y = xy
    if anchor_center:
        x -= total / 2.0
    for ch, w in zip(text, widths):
        # Two passes, not one.  Pillow strokes a glyph by widening its outline
        # on both sides, so a stroke as wide as the letter's stem eats the fill
        # and the letter comes out hollow.  Draw the halo first, then the ink.
        if stroke_width and stroke_fill is not None:
            draw.text((x, y), ch, font=fnt, fill=stroke_fill, anchor="lm",
                      stroke_width=stroke_width, stroke_fill=stroke_fill)
        draw.text((x, y), ch, font=fnt, fill=fill, anchor="lm")
        x += w + tracking
    return total


def rotated_text(page, text, xy, fnt, fill, angle, stroke_width=0, stroke_fill=None):
    """Draw text at an angle -- for names that follow a river or a ridge.

    PIL cannot rotate a glyph run in place, so the run is set on a transparent
    patch, rotated, and pasted back.  At the supersampled sizes this engine
    works at, the resampling is invisible.
    """
    pad = 6 + stroke_width * 2
    w = int(fnt.getlength(text)) + pad * 2
    h = int(fnt.size * 1.6) + pad * 2
    tmp = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(tmp)
    sf = (tuple(stroke_fill) + (255,)) if stroke_fill is not None else None
    if stroke_width and sf is not None:
        d.text((pad, h / 2), text, font=fnt, fill=sf, anchor="lm",
               stroke_width=stroke_width, stroke_fill=sf)
    d.text((pad, h / 2), text, font=fnt, fill=tuple(fill) + (255,), anchor="lm")
    rot = tmp.rotate(angle, resample=Image.BICUBIC, expand=True)
    x, y = xy
    page.paste(rot, (int(x - rot.width / 2), int(y - rot.height / 2)), rot)


def draw_frame(draw, box, margin, ink, paper, lat_top=0.0, lat_bottom=0.0,
               lon_left=0.0, lon_right=0.0, step=10.0, sub=4):
    """Neat line, graduated chequer border, and outer rule.

    The chequer is the cheapest thing on the sheet and the one that dates it:
    a plain rule says "printed", alternating blocks say "surveyed".  One
    black block per half-degree of a ten-degree step is about the scale of a
    plate whose border is read at arm's length.
    """
    x0, y0, x1, y1 = box
    # A full-contrast zebra pulls the eye off the map.  The blocks are printed
    # in ink let down toward the paper, the way a graduated border actually
    # looks on a plate that has been pulled twice.
    block = tuple(int(round(i * 0.68 + p * 0.32)) for i, p in zip(ink, paper))
    bw = max(3, int(round(margin * 0.15)))
    thin = margin * 0.42
    thick = margin * 0.56

    def comb(lo, hi, a, b, vertical):
        span = hi - lo
        if span <= 0.0:
            return
        # Subdivide each degree cell until the blocks come out roughly square.
        # A fixed subdivision gives blocks twice as wide as they are tall on a
        # long edge, and the border stops reading as a scale and starts
        # reading as a picture frame.
        cells = max(1, int(round(span / step)))
        per = max(2, int(round((b - a) / (cells * bw * 1.05))))
        n = cells * per
        for k in range(n):
            if k % 2:
                continue
            t0, t1 = k / n, (k + 1) / n
            if vertical:
                ya = a + (b - a) * t0
                yb = a + (b - a) * t1
                draw.rectangle([x0 - bw, ya, x0, yb], fill=block)
                draw.rectangle([x1, ya, x1 + bw, yb], fill=block)
            else:
                xa = a + (b - a) * t0
                xb = a + (b - a) * t1
                draw.rectangle([xa, y0 - bw, xb, y0], fill=block)
                draw.rectangle([xa, y1, xb, y1 + bw], fill=block)

    comb(lon_left, lon_right, x0, x1, False)
    comb(lat_bottom, lat_top, y0, y1, True)

    draw.rectangle([x0, y0, x1, y1], outline=ink, width=max(1, int(margin * 0.012)))
    draw.rectangle([x0 - bw, y0 - bw, x1 + bw, y1 + bw], outline=ink,
                   width=max(1, int(margin * 0.010)))
    draw.rectangle([x0 - thin, y0 - thin, x1 + thin, y1 + thin], outline=ink,
                   width=max(1, int(margin * 0.008)))
    draw.rectangle([x0 - thick, y0 - thick, x1 + thick, y1 + thick], outline=ink,
                   width=max(2, int(margin * 0.038)))


def draw_graticule(page, box, lat_top, lat_bottom, lon_left, lon_right, ink, step=10.0,
                   alpha=0.10):
    """The ruled grid of the projection.

    Drawn on its own layer and composited, not painted into the page.  A
    graticule has to sit *in* the map at a few percent; pre-blending the ink
    toward white and drawing that gives a near-white rule instead, which is
    the opposite of subtle.
    """
    x0, y0, x1, y1 = box
    w = x1 - x0
    h = y1 - y0
    layer = Image.new("RGBA", page.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    col = tuple(int(round(c * 255)) for c in ink) + (int(round(alpha * 255)),)

    lat = math.ceil(lat_bottom / step) * step
    while lat <= lat_top:
        y = y0 + (lat_top - lat) / (lat_top - lat_bottom) * h
        d.line([(x0, y), (x1, y)], fill=col, width=1)
        lat += step

    lon = math.ceil(lon_left / step) * step
    while lon <= lon_right:
        x = x0 + (lon - lon_left) / (lon_right - lon_left) * w
        d.line([(x, y0), (x, y1)], fill=col, width=1)
        lon += step

    return Image.alpha_composite(page.convert("RGBA"), layer).convert("RGB")


def draw_graticule_labels(draw, box, margin, lat_top, lat_bottom, lon_left, lon_right,
                          ink, step=10.0, size=16):
    """Degree figures sit in the band between the two neat lines."""
    x0, y0, x1, y1 = box
    w = x1 - x0
    h = y1 - y0
    f = font("serif", size)
    band = margin * 0.415

    lat = math.ceil(lat_bottom / step) * step
    while lat <= lat_top:
        y = y0 + (lat_top - lat) / (lat_top - lat_bottom) * h
        txt = f"{abs(int(lat))}°{'N' if lat >= 0 else 'S'}"
        draw.text((x0 - band, y), txt, font=f, fill=ink, anchor="rm")
        draw.text((x1 + band, y), txt, font=f, fill=ink, anchor="lm")
        lat += step

    lon = math.ceil(lon_left / step) * step
    while lon <= lon_right:
        x = x0 + (lon - lon_left) / (lon_right - lon_left) * w
        txt = f"{abs(int(lon))}°{'E' if lon >= 0 else 'W'}"
        draw.text((x, y0 - band), txt, font=f, fill=ink, anchor="mb")
        draw.text((x, y1 + band), txt, font=f, fill=ink, anchor="mt")
        lon += step


def draw_rhumbs(page, origin, box, water, ink, alpha=0.10, count=32):
    """Navigational rays radiating from the compass, laid over water only.

    A sea chart is a measuring instrument before it is a picture: the web of
    rhumbs is what tells a navigator the bearing between two roads.  They are
    drawn on their own layer and cut by the water mask so that no ray ever
    crosses a headland.
    """
    layer = Image.new("RGBA", page.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    x0, y0, x1, y1 = box
    cx, cy = origin
    reach = int(math.hypot(x1 - x0, y1 - y0)) * 2
    col = _rgb(ink) + (int(round(alpha * 255)),)
    for i in range(count):
        a = i * (2.0 * math.pi / count)
        d.line([(cx, cy), (cx + math.cos(a) * reach, cy + math.sin(a) * reach)],
               fill=col, width=1)
    layer.putalpha(ImageChops.multiply(layer.getchannel("A"), water))
    return Image.alpha_composite(page.convert("RGBA"), layer).convert("RGB")


def draw_compass(draw, cx, cy, r, ink, paper):
    """Eight-point rose, the north point filled."""
    pts = 8
    for i in range(pts):
        a = -math.pi / 2 + i * (2 * math.pi / pts)
        length = r if i % 2 == 0 else r * 0.62
        half = (2 * math.pi / pts) * 0.5 * 0.55
        tip = (cx + math.cos(a) * length, cy + math.sin(a) * length)
        left = (cx + math.cos(a - half) * r * 0.20, cy + math.sin(a - half) * r * 0.20)
        right = (cx + math.cos(a + half) * r * 0.20, cy + math.sin(a + half) * r * 0.20)
        fill = _rgb(ink) if i == 0 else _rgb(tuple(c * 0.25 + 1.0 * 0.75 for c in ink))
        draw.polygon([tip, left, (cx, cy), right], fill=fill, outline=_rgb(ink))

    draw.ellipse([cx - r * 0.10, cy - r * 0.10, cx + r * 0.10, cy + r * 0.10],
                 fill=_rgb(paper), outline=_rgb(ink), width=max(1, int(r * 0.03)))
    draw.ellipse([cx - r * 1.06, cy - r * 1.06, cx + r * 1.06, cy + r * 1.06],
                 outline=_rgb(ink), width=max(1, int(r * 0.025)))

    f = font("serif", max(12, int(r * 0.30)))
    for letter, (dx, dy) in (("N", (0, -1.30)), ("E", (1.30, 0)),
                             ("S", (0, 1.30)), ("W", (-1.30, 0))):
        draw.text((cx + dx * r, cy + dy * r), letter, font=f, fill=_rgb(ink), anchor="mm")


def draw_scale_bar(draw, x, y, width, ink, paper, units="leagues", segments=4, height=None):
    """Alternating bar; the zero is at the left end."""
    h = height or max(6, int(width * 0.022))
    seg = width / segments
    for i in range(segments):
        x0 = x + i * seg
        fill = _rgb(ink) if i % 2 == 0 else _rgb(paper)
        draw.rectangle([x0, y, x0 + seg, y + h], fill=fill, outline=_rgb(ink), width=1)

    f = font("serif", max(10, int(h * 1.5)))
    fi = font("serif-italic", max(10, int(h * 1.5)))
    draw.text((x, y - h * 1.7), "0", font=f, fill=_rgb(ink), anchor="mb")
    # Right-anchored, not centred: a centred label hangs half its width past
    # the end of the bar, and on a bar placed near the frame that half lands
    # outside the plate.
    draw.text((x + width, y - h * 1.7), units.capitalize(), font=fi, fill=_rgb(ink), anchor="rb")


def draw_cartouche(draw, box, title, subtitle, ink, paper, title_size, sub_size):
    x0, y0, x1, y1 = box
    draw.rectangle([x0, y0, x1, y1], fill=_rgb(paper), outline=_rgb(ink), width=2)
    pad = (x1 - x0) * 0.06
    draw.rectangle([x0 + pad * 0.5, y0 + pad * 0.5, x1 - pad * 0.5, y1 - pad * 0.5],
                   outline=_rgb(ink), width=1)

    # Fit the title: a cartouche that overflows its own frame is worse than a
    # small one.  Shrink the type until the tracked line clears the border.
    avail = (x1 - x0) * 0.74
    size = title_size
    while size > 7:
        f = font("antiqua-bold", size)
        track = size * 0.16
        w = sum(draw.textlength(c, font=f) for c in title) + track * max(0, len(title) - 1)
        if w <= avail:
            break
        size -= 1

    cx = (x0 + x1) / 2.0
    tracked_text(draw, (cx, y0 + (y1 - y0) * 0.30), title.upper(), f, _rgb(ink),
                 tracking=size * 0.16)
    draw.line([(cx - (x1 - x0) * 0.22, y0 + (y1 - y0) * 0.55),
               (cx + (x1 - x0) * 0.22, y0 + (y1 - y0) * 0.55)], fill=_rgb(ink), width=1)
    fs = font("antiqua-italic", sub_size)
    draw.text((cx, y0 + (y1 - y0) * 0.74), subtitle, font=fs, fill=_rgb(ink), anchor="mm")


def draw_legend(draw, box, entries, ink, paper, title="ELEVATIONS", size=15):
    """A small key: swatch plus label per band."""
    x0, y0, x1, y1 = box
    draw.rectangle([x0, y0, x1, y1], fill=_rgb(paper), outline=_rgb(ink), width=1)
    f = font("serif", size)
    ft = font("antiqua-bold", int(size * 0.95))
    tracked_text(draw, ((x0 + x1) / 2, y0 + size * 0.7), title, ft, _rgb(ink),
                 tracking=size * 0.18)
    row = (y1 - y0 - size * 2.0) / max(len(entries), 1)
    for i, (colour, label) in enumerate(entries):
        cy = y0 + size * 1.7 + i * row
        draw.rectangle([x0 + size * 0.8, cy, x0 + size * 2.4, cy + row * 0.52],
                       fill=_rgb(colour), outline=_rgb(ink), width=1)
        draw.text((x0 + size * 3.0, cy + row * 0.26), label, font=f, fill=_rgb(ink), anchor="lm")
