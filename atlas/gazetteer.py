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
        f"a country of {int(land.mean() * 100)} parts land to {100 - int(land.mean() * 100)} of water."
    )
    if seas:
        opening += f" Its western approaches are held by the {seas[0]}"
        if len(seas) > 1:
            opening += f", and the {seas[1]} closes it to the east"
        opening += "."
    paras.append(opening)

    relief = f"The interior rises to {int(round(peak, -2))} metres."
    if ranges:
        relief += f" The {ranges[0]}"
        if len(ranges) > 1:
            relief += f" and the {ranges[1]}"
        relief += " carry the highest ground, and the rivers that leave them run clear to the sea."
    paras.append(relief)

    if rivers:
        body = f"The {rivers[0]} is the longest of the waters shown here"
        others = [r for r in rivers[1:] if r != rivers[0]]
        if others:
            body += f"; the {others[0]}"
            if len(others) > 1:
                body += f" and the {others[1]}"
            body += " are its chief tributaries"
        body += "."
        paras.append(body)

    if lake_cells > 400:
        paras.append(
            f"Lakes occupy {lake_cells / max(land.sum(), 1) * 100:.1f} parts in a hundred of the "
            "land, most of them strung along the courses of the greater rivers."
        )

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
            body += f" Of the towns, {towns[0]} is the largest, and {towns[-1]} the most remote."
        paras.append(body)

    paras.append(
        "The coast is drawn at the survey of the year given in the title, and the "
        "soundings are in fathoms. No part of this country has been visited twice."
    )

    return name.title(), f"a survey of the country, in {max(len(regions), 1)} provinces", paras


def _unique(items):
    seen, out = set(), []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def text_page(title, subtitle, paragraphs, size=(1600, 2000), seed=0):
    """Set a gazetteer page: title, rule, justified-looking prose."""
    w, h = size
    page = Image.new("RGB", size, tuple(int(c * 255) for c in PAPER))
    draw = ImageDraw.Draw(page)
    ink = tuple(int(c * 255) for c in INK)
    soft = tuple(int(c * 255) for c in INK_SOFT)

    margin = int(w * 0.12)
    ft = decor.font("antiqua-bold", int(w * 0.040))
    decor.tracked_text(draw, (w / 2, margin * 1.4), title.upper(), ft, ink,
                       tracking=ft.size * 0.18)
    fs = decor.font("antiqua-italic", int(w * 0.020))
    draw.text((w / 2, margin * 1.4 + ft.size * 1.7), subtitle, font=fs, fill=soft, anchor="mm")

    rule_y = margin * 1.4 + ft.size * 2.6
    draw.line([(w * 0.26, rule_y), (w * 0.74, rule_y)], fill=ink, width=1)

    fb = decor.font("serif", int(w * 0.019))
    y = rule_y + int(w * 0.05)
    lead = int(fb.size * 1.72)
    col_w = w - 2 * margin

    for para in paragraphs:
        for line in _wrap(draw, para, fb, col_w):
            draw.text((margin, y), line, font=fb, fill=ink)
            y += lead
        y += int(lead * 0.42)

    draw.text((w / 2, h - margin * 0.6), "— finis —", font=fs, fill=soft, anchor="mm")
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
