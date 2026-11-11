"""Prose for the atlas.

A map without a gazetteer is a picture of a place.  A map with one is a
document about a place -- even if the place never existed.  These paragraphs
are assembled from the same facts the drawing uses: the extent, the highest
ground, the principal river, the named regions.  Nothing is invented twice.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw

from . import decor, features
from .palette import PAPER, INK, INK_SOFT


def _article(word: str) -> str:
    return "an" if word[:1].lower() in "aeiou" else "a"


def _count(value, step=3):
    """A measured quantity, rounded to `step` places and grouped: 3,654,000."""
    return f"{int(round(value, -step)):,}"


def describe(world, labels, name: str, feats=None):
    """Return (title, subtitle, paragraphs) for one sheet.

    Every claim below is measured in `features` before it is written.  The
    prose used to call the first river in a list "the longest" and the last
    town "the most remote" without computing either; a sentence that happens
    to be true is indistinguishable from one that is not.
    """
    spec = world.spec
    land = world.height > 0
    if feats is None:
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
        f"{name.title()} lies between the {_ordinal(int(lat_bot))} and "
        f"{_ordinal(int(lat_top))} parallels.  Of the ground within the neat "
        f"line, {share} parts in a hundred are land."
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
    # A heading that overruns the sheet is worse than one set smaller: measure
    # the tracked line and bring it in until it clears the margins.
    head = title.upper()
    track = ft.size * 0.18
    limit = w * (1.0 - 2 * 0.12) * 0.98
    while ft.size > 12 and (
            sum(draw.textlength(ch, font=ft) for ch in head)
            + track * (len(head) - 1)) > limit:
        ft = decor.font("antiqua-bold", ft.size - 2)
        track = ft.size * 0.18
    decor.tracked_text(draw, (w / 2, margin * 0.92), head, ft, ink, tracking=track)
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

# ---------------------------------------------------------------------------
# The dictionary of places
# ---------------------------------------------------------------------------

@dataclass
class Entry:
    head: str
    kind: str
    text: str


_RANK_WORD = {0: "city", 1: "market town", 2: "village", 3: "hamlet"}


def _m(value, step=0):
    """Metres, grouped."""
    return f"{int(round(value, -step)):,}"


def _km(value, step=0):
    return f"{int(round(value, -step)):,}"


def _article(word):
    return "an" if word[0].lower() in "aeiou" else "a"


def _town_entry(t, prov_name):
    head = _RANK_WORD.get(t.rank, "village")
    where = f" in {prov_name.title()}" if prov_name else ""
    if t.river:
        site = f", on the {t.river.title()}"
    elif 0 <= t.coast_km < 15:
        site = ", on the shore"
    elif 0 <= t.coast_km < 260:
        site = f", {_km(t.coast_km)} km inland"
    else:
        site = ""
    text = f"{_article(head)} {head}{where}{site}"
    if t.elev_m >= 3:
        text += f", {_m(t.elev_m)} m above the sea"
    text += f"; population {t.pop:,}."
    if t.capital:
        text += " The seat of government."
    if t.nearest and t.nearest_km < 1200:
        text += f" The nearest town is {t.nearest.title()}, {_km(t.nearest_km)} km away."
    return text


def _river_entry(r, towns=()):
    rise = f"a river rising at {_m(r.source_m)} m"
    if r.range_name:
        rise += f" in the {r.range_name.title()}"
    if r.mouth_sea:
        where = f"the {r.mouth_sea.title()}"
    elif r.reaches_sea:
        where = "the sea"
    else:
        where = "the marshes"
    text = (f"{rise}, running {_km(r.length_km, 1)} km {r.flow} to {where}, "
            f"and draining {_count(r.drained_km2)} square kilometres.")
    if len(towns) == 1:
        text += f" {towns[0].title()} stands upon it."
    elif len(towns) > 1:
        text += f" {towns[0].title()} and {towns[1].title()} stand upon it."
    return text


def _range_entry(r):
    text = f"a range running {_km(r.extent_km, 0)} km through {r.provinces or 'the interior'}"
    text += f"; the highest point is {_m(r.high_m, 0)} m"
    if r.holds_peak:
        text += ", the highest in the country"
    return text + "."


def _province_entry(p):
    plural = "town" if p.towns == 1 else "towns"
    text = f"a province of {_count(p.area_km2)} square kilometres, {p.towns} {plural}"
    if p.seat:
        text += f", the seat at {p.seat.title()}"
    text += "."
    if p.biome:
        text += f" Mostly {p.biome.lower()}."
    if p.coast_km > 40:
        text += f" {_km(p.coast_km)} km of coast."
    return text


def _ordinal(n: int) -> str:
    """63 -> 'the 63rd'.  A book that says '63th' is a book nobody trusts."""
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _sea_entry(s):
    text = f"a sea washing the {s.shore} shore"
    if s.rivers == 1:
        text += "; one river shown reaches it"
    elif s.rivers > 1:
        text += f"; {s.rivers} rivers shown reach it"
    return text + "."


def entries(world, labels, feats=None):
    """One entry per named feature, alphabetical -- the dictionary itself."""
    if feats is None:
        feats = features.measure(world, labels)
    prov_name = {p.index: p.name for p in feats.provinces}

    on_river = {}
    for t in feats.towns:
        if t.river:
            on_river.setdefault(t.river.lower(), []).append(t.name)

    out = []
    for t in feats.towns:
        out.append(Entry(t.name.title(), "town", _town_entry(t, prov_name.get(t.prov, ""))))
    for r in feats.rivers:
        out.append(Entry(r.name.title(), "river",
                         _river_entry(r, on_river.get(r.name.lower(), ()))))
    for r in feats.ranges:
        out.append(Entry(r.name.title(), "range", _range_entry(r)))
    for p in feats.provinces:
        out.append(Entry(p.name.title(), "province", _province_entry(p)))
    for s in feats.seas:
        out.append(Entry(s.name.title(), "sea", _sea_entry(s)))

    # Two places can share a name; the dictionary keeps the first and marks
    # the second, because an index that points twice at one word is useless.
    seen, kept = set(), []
    for e in sorted(out, key=lambda e: (e.head.lower(), e.kind)):
        if e.head.lower() in seen:
            continue
        seen.add(e.head.lower())
        kept.append(e)
    return kept


def _run_in(draw, head, text, fh, fb, col_w, indent):
    """Head and text on one line, the rest hanging under an indent."""
    hw = draw.textlength(head, font=fh) + draw.textlength(" \u2014 ", font=fb)
    avail = col_w - hw
    lines, cur, first = [], "", True
    if avail < col_w * 0.25:          # the head alone is most of the column
        lines.append("")
        first = False
        avail = col_w - indent
    for word in text.split():
        trial = f"{cur} {word}".strip()
        if draw.textlength(trial, font=fb) <= avail:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = word
            avail = col_w - indent
            first = False
    if cur:
        lines.append(cur)
    return lines, hw


def entry_pages(size, head, entries_list, per_page=None):
    """Set the dictionary in two columns; return (pages, page_of_entry)."""
    w, h = size
    paper = tuple(int(c * 255) for c in PAPER)
    ink = tuple(int(c * 255) for c in INK)
    soft = tuple(int(c * 255) for c in INK_SOFT)

    margin = int(w * 0.105)
    gutter = int(w * 0.045)
    col_w = (w - 2 * margin - gutter) // 2
    top = int(h * 0.135)
    bottom = int(h * 0.895)

    fs = max(10, int(w * 0.0128))
    fb = decor.font("serif", fs)
    fh = decor.font("serif-bold", fs)
    fsm = decor.font("antiqua", int(w * 0.0092))
    lead = int(fs * 1.52)
    indent = int(fs * 2.6)
    gap = int(lead * 0.55)

    probe = ImageDraw.Draw(Image.new("RGB", (8, 8)))

    # Measure every entry once.
    blocks = []
    for e in entries_list:
        lines, hw = _run_in(probe, e.head.upper(), e.text, fh, fb, col_w, indent)
        blocks.append((e, lines, hw))

    heights = [len(lines) * lead + gap for _, lines, _ in blocks]
    col_h = bottom - top

    def draw_block(draw, x, y, e, lines, hw):
        draw.text((x, y), e.head.upper(), font=fh, fill=ink)
        hwid = draw.textlength(e.head.upper(), font=fh)
        draw.text((x + hwid, y), " \u2014 ", font=fb, fill=ink)
        if lines and lines[0]:
            draw.text((x + hw, y), lines[0], font=fb, fill=ink)
            rest = lines[1:]
        else:
            rest = lines
        yy = y + lead
        for line in rest:
            draw.text((x + indent, yy), line, font=fb, fill=ink)
            yy += lead

    pages, page_of = [], {}
    i = 0
    while i < len(blocks):
        # Everything that fits on this page, both columns together.
        j, used = i, 0.0
        while j < len(blocks) and used + heights[j] <= 2 * col_h:
            used += heights[j]
            j += 1
        if j == i:                      # a single entry taller than a page
            j = i + 1
        chunk = list(range(i, j))

        # Cut the chunk where the two columns come out closest in height: a
        # dictionary page whose right column stops halfway looks like a page
        # that ran out of words rather than a page that ended.
        total = sum(heights[k] for k in chunk)
        acc, cut = 0.0, len(chunk)
        for n, k in enumerate(chunk):
            if acc + heights[k] * 0.5 >= total * 0.5:
                cut = n + 1
                break
            acc += heights[k]
        while cut > 1 and sum(heights[k] for k in chunk[:cut]) > col_h:
            cut -= 1

        page = Image.new("RGB", size, paper)
        draw = ImageDraw.Draw(page)
        draw.text((margin, int(h * 0.088)), "GAZETTEER", font=fsm, fill=soft)
        tw = draw.textlength(head.upper(), font=fsm)
        draw.text((w - margin - tw, int(h * 0.088)), head.upper(), font=fsm, fill=soft)
        rule = int(h * 0.100)
        draw.line([(margin, rule), (w - margin, rule)], fill=soft, width=1)

        for col, part in ((0, chunk[:cut]), (1, chunk[cut:])):
            x = margin + col * (col_w + gutter)
            y = top
            for k in part:
                e, lines, hw = blocks[k]
                draw_block(draw, x, y, e, lines, hw)
                y += heights[k]
                page_of.setdefault(e.head.lower(), len(pages))
        pages.append(page)
        i = j

    # folios
    for n, pg in enumerate(pages):
        d = ImageDraw.Draw(pg)
        d.text((w / 2, int(h * 0.945)), str(n + 1), font=fsm, fill=soft, anchor="mm")
    return pages, page_of
