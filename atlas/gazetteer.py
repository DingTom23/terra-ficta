"""Prose for the atlas.

A map without a gazetteer is a picture of a place.  A map with one is a
document about a place -- even if the place never existed.  These paragraphs
are assembled from the same facts the drawing uses: the extent, the highest
ground, the principal river, the named regions.  Nothing is invented twice.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from . import decor, features
from .palette import PAPER, INK, INK_SOFT


def _article(word: str) -> str:
    return "an" if word[:1].lower() in "aeiou" else "a"


def _count(value, step=3):
    """A measured quantity, rounded to `step` places and grouped: 3,654,000."""
    return f"{int(round(value, -step)):,}"


def describe(world, labels, name: str):
    """Return (title, subtitle, paragraphs) for one sheet.

    Every claim below is measured in `features` before it is written.  The
    prose used to call the first river in a list "the longest" and the last
    town "the most remote" without computing either; a sentence that happens
    to be true is indistinguishable from one that is not.
    """
    spec = world.spec
    land = world.height > 0
    feats = features.measure(world, labels)

    seas = [lb for lb in labels if lb.kind == "water"]
    provinces = sorted(feats.provinces, key=lambda p: -p.area_km2)
    rivers = sorted(feats.rivers, key=lambda r: -r.length_km)

    peak = feats.peak_m
    lake_cells = int((world.lake_depth > 6).sum())
    lat_top, lat_bot = spec.lat_top, spec.lat_bottom
    share = int(round(float(land.mean()) * 100))

    paras = []
    opening = (
        f"{name.title()} lies between the {int(lat_bot)}th and {int(lat_top)}th parallels, "
        f"and holds {share} parts land to {100 - share} of water."
    )
    if seas:
        # Which shore a sea breaks upon is a fact about where the label sits,
        # not about the order the labels were made in.
        cx = float(np.where(land)[1].mean())
        west = [lb for lb in seas if lb.x < cx]
        east = [lb for lb in seas if lb.x > cx]
        if west:
            opening += f" The {west[0].text.title()} breaks upon its western shore"
            if east:
                opening += f", and the {east[0].text.title()} closes it to the east"
        elif east:
            opening += f" The {east[0].text.title()} lies off its eastern shore"
        else:
            opening += f" The {seas[0].text.title()} lies off its coast"
        opening += "."
    paras.append(opening)

    relief = f"The ground rises to {_count(peak, 0)} metres."
    top = feats.highest_range()
    if top is not None:
        if top.holds_peak:
            relief += f" The {top.name.title()} carry the highest of it."
        else:
            relief += (f" The {top.name.title()} are the highest ground the survey names, "
                       f"reaching {_count(top.high_m, 0)} metres.")
    paras.append(relief)

    if rivers:
        first = rivers[0]
        where = "the sea" if first.reaches_sea else "the marshes"
        body = (f"The {first.name.title()} is the longest of the rivers shown, "
                f"{_count(first.length_km, 1)} kilometres from its source to {where}.")
        if len(rivers) > 1:
            second = rivers[1]
            body += (f" The {second.name.title()} is next at "
                     f"{_count(second.length_km, 1)} kilometres, draining "
                     f"{_count(second.drained_km2)} square kilometres.")
        if lake_cells > 400:
            body += (f" Lakes take up {lake_cells / max(land.sum(), 1) * 100:.1f} parts in a "
                     "hundred of the land, most of them strung along the greater courses.")
        paras.append(body)

    if provinces:
        if len(provinces) == 1:
            body = f"The country is the single province of {provinces[0].name.title()}"
        elif len(provinces) == 2:
            body = (f"The country is divided between {provinces[0].name.title()} "
                    f"and {provinces[1].name.title()}")
        else:
            names = [p.name.title() for p in provinces]
            body = "The country is divided into " + ", ".join(names[:-1]) + f" and {names[-1]}"
        big = provinces[0]
        body += (f", of which {big.name.title()} is the largest at "
                 f"{_count(big.area_km2)} square kilometres.")
        chief = feats.chief_town()
        remote = feats.remotest_town()
        if chief is not None:
            body += f" The seat is at {chief.name.title()}"
            if remote is not None and remote is not chief:
                body += (f", and {remote.name.title()} stands furthest from its neighbours, "
                         f"{_count(remote.nearest_km, 0)} kilometres.")
            else:
                body += "."
        body += " No part of this country has been visited twice."
        paras.append(body)

    return (name.title(),
            f"a survey of the country, in {max(len(provinces), 1)} provinces",
            paras)


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

    # "finis" belongs at the end of the text, not at the foot of the page: a
    # short survey leaves half a page of air, and a rule at the bottom of it
    # reads as a printing error rather than a full stop.
    y = min(y + gap * 0.2, h - margin * 0.30)
    draw.text((w / 2, y), "— finis —", font=fs, fill=soft, anchor="mm")
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
