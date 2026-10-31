"""Prose for the atlas.

A map without a gazetteer is a picture of a place.  A map with one is a
document about a place -- even if the place never existed.  These paragraphs
are assembled from the same facts the drawing uses: the extent, the highest
ground, the principal river, the named regions.  Nothing is invented twice.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from . import decor
from .palette import PAPER, INK, INK_SOFT


def _article(word: str) -> str:
    return "an" if word[:1].lower() in "aeiou" else "a"


def describe(world, labels, name: str):
    """Return (title, subtitle, paragraphs) for one sheet."""
    spec = world.spec
    land = world.height > 0
    h = world.height

    regions = _unique(lb.text.title() for lb in labels if lb.kind == "region")
    seas = _unique(lb.text.title() for lb in labels if lb.kind == "water")
    ranges = _unique(lb.text.title() for lb in labels if lb.kind == "range")
    rivers = _unique(lb.text.title() for lb in labels if lb.kind == "river")
    towns = _unique(lb.text.title() for lb in labels if lb.kind == "town")

    peak = float(h.max())
    lake_cells = int((world.lake_depth > 6).sum())
    lat_top, lat_bot = spec.lat_top, spec.lat_bottom

    paras = []
    opening = (
        f"{name.title()} lies between the {int(lat_bot)}th and {int(lat_top)}th parallels, "
        f"and holds {int(land.mean() * 100)} parts land to {100 - int(land.mean() * 100)} of water."
    )
    if seas:
        opening += f" The {seas[0]} breaks upon its western shore"
        if len(seas) > 1:
            opening += f", and the {seas[1]} closes it to the east"
        opening += "."
    paras.append(opening)

    relief = f"The ground rises to {int(round(peak, -2))} metres."
    if ranges:
        relief += f" The {ranges[0]}"
        if len(ranges) > 1:
            relief += f" and the {ranges[1]}"
        relief += " carry the highest of it, and the waters that leave them run clear to the sea."
    paras.append(relief)

    if rivers:
        body = f"The {rivers[0]} is the longest of the rivers shown"
        others = [r for r in rivers[1:] if r != rivers[0]]
        if others:
            body += f", with the {others[0]}"
            if len(others) > 1:
                body += f" and the {others[1]}"
            body += " among its tributaries"
        body += "."
        if lake_cells > 400:
            body += (f" Lakes take up {lake_cells / max(land.sum(), 1) * 100:.1f} parts in a "
                     "hundred of the land, most of them strung along the greater courses.")
        paras.append(body)

    if regions:
        body = "The country is divided into "
        if len(regions) == 1:
            body += f"the single province of {regions[0]}"
        elif len(regions) == 2:
            body += f"{regions[0]} and {regions[1]}"
        else:
            body += ", ".join(regions[:-1]) + f" and {regions[-1]}"
        body += "."
        if towns:
            body += f" Of the towns, {towns[0]} is the largest and {towns[-1]} the most remote."
        body += " No part of this country has been visited twice."
        paras.append(body)

    return name.title(), f"a survey of the country, in {max(len(regions), 1)} provinces", paras


def _unique(items):
    seen, out = set(), []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def text_page(title, subtitle, paragraphs, size=(1600, 2000), seed=0):
    """Set a gazetteer page: title, rule, and prose sized to fit the sheet.

    The body is set by measurement, not by hope.  A page whose last paragraph
    runs off the bottom, or whose colophon lands on top of it, is worse than a
    page in smaller type.
    """
    w, h = size
    page = Image.new("RGB", size, tuple(int(c * 255) for c in PAPER))
    draw = ImageDraw.Draw(page)
    ink = tuple(int(c * 255) for c in INK)
    soft = tuple(int(c * 255) for c in INK_SOFT)

    margin = int(w * 0.12)
    ft = decor.font("antiqua-bold", int(w * 0.036))
    decor.tracked_text(draw, (w / 2, margin * 0.92), title.upper(), ft, ink,
                       tracking=ft.size * 0.18)
    fs = decor.font("antiqua-italic", int(w * 0.0185))
    draw.text((w / 2, margin * 0.92 + ft.size * 1.55), subtitle, font=fs, fill=soft, anchor="mm")

    rule_y = margin * 0.92 + ft.size * 2.30
    draw.line([(w * 0.26, rule_y), (w * 0.74, rule_y)], fill=ink, width=1)

    col_w = w - 2 * margin
    top = rule_y + margin * 0.42
    bottom = h - margin * 0.62
    avail = bottom - top

    # Fit: the largest body size whose measured column clears the foot.
    size_px = max(11, int(w * 0.0165))
    floor = max(10, int(w * 0.0122))
    while True:
        fb = decor.font("serif", size_px)
        lead = int(size_px * 1.58)
        gap = int(lead * 0.46)
        blocks = [_wrap(draw, para, fb, col_w) for para in paragraphs]
        total = sum(len(b) for b in blocks) * lead + gap * max(0, len(blocks) - 1)
        if total <= avail or size_px <= floor:
            break
        size_px -= 1

    y = top
    for block in blocks:
        for line in block:
            draw.text((margin, y), line, font=fb, fill=ink)
            y += lead
        y += gap

    draw.text((w / 2, h - margin * 0.26), "— finis —", font=fs, fill=soft, anchor="mm")
    return page


def _wrap(draw, text, fnt, width):
    words = text.split()
    lines, cur = [], ""
    for word in words:
        trial = f"{cur} {word}".strip()
        if draw.textlength(trial, font=fnt) <= width:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines
