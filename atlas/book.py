"""Binding the plates into a book.

A stack of images is a folder.  A book has a title page, a colophon that says
how it was made, a table of contents, numbered plates, and an index -- and the
index is what turns a set of pictures into something you can look things up in.
"""

from __future__ import annotations

import math

from PIL import Image, ImageDraw

from . import decor
from .palette import PAPER, INK, INK_SOFT


def _blank(size):
    page = Image.new("RGB", size, tuple(int(c * 255) for c in PAPER))
    return page, ImageDraw.Draw(page)


PAGE = (2400, 1700)


def place_plate(plate, size=PAGE, margin=0.035):
    """Set a plate on a page of the standard size, centred.

    Plates are framed on their own subjects and therefore come in different
    shapes.  A bound atlas does not resize its page to suit each map; it
    prints the map at whatever size the page allows and lets the margin absorb
    the difference.  That is what this does.
    """
    w, h = size
    pw, ph = plate.size
    k = min(w * (1 - 2 * margin) / pw, h * (1 - 2 * margin) / ph)
    if abs(k - 1.0) > 2e-3:
        nw, nh = max(1, int(pw * k)), max(1, int(ph * k))
        plate = plate.resize((nw, nh), Image.LANCZOS)
    page, _ = _blank(size)
    page.paste(plate, ((w - plate.size[0]) // 2, (h - plate.size[1]) // 2))
    return page


def _rule(draw, y, width, x0, x1, ink, weight=1):
    draw.line([(x0, y), (x1, y)], fill=ink, width=weight)


def title_page(size, title, subtitle, year, plates):
    w, h = size
    page, draw = _blank(size)
    ink = tuple(int(c * 255) for c in INK)
    soft = tuple(int(c * 255) for c in INK_SOFT)

    m = int(min(w, h) * 0.075)
    draw.rectangle([m, m, w - m, h - m], outline=ink, width=max(2, int(m * 0.045)))
    draw.rectangle([m * 1.35, m * 1.35, w - m * 1.35, h - m * 1.35], outline=ink, width=1)

    f1 = decor.font("antiqua-bold", int(w * 0.085))
    decor.tracked_text(draw, (w / 2, h * 0.30), title.upper(), f1, ink, tracking=f1.size * 0.34)

    _rule(draw, h * 0.40, 0, w * 0.36, w * 0.64, ink, 1)
    draw.ellipse([w / 2 - 4, h * 0.40 - 4, w / 2 + 4, h * 0.40 + 4], fill=ink)
    _rule(draw, h * 0.40, 0, w * 0.30, w * 0.70, ink, 1)

    f2 = decor.font("antiqua-italic", int(w * 0.021))
    draw.text((w / 2, h * 0.455), subtitle, font=f2, fill=soft, anchor="mm")

    f3 = decor.font("serif", int(w * 0.016))
    draw.text((w / 2, h * 0.58), f"{plates} plates", font=f3, fill=soft, anchor="mm")
    draw.text((w / 2, h * 0.625), year, font=decor.font("antiqua", int(w * 0.018)),
              fill=ink, anchor="mm")

    f4 = decor.font("serif-italic", int(w * 0.012))
    draw.text((w / 2, h * 0.84), "drawn by a machine from a single number",
              font=f4, fill=soft, anchor="mm")
    return page


def colophon_page(size, lines):
    w, h = size
    page, draw = _blank(size)
    ink = tuple(int(c * 255) for c in INK)
    soft = tuple(int(c * 255) for c in INK_SOFT)

    m = int(w * 0.18)
    f = decor.font("antiqua-italic", int(w * 0.0135))
    lead = int(f.size * 1.9)
    y = h * 0.20

    head = decor.font("antiqua-bold", int(w * 0.016))
    decor.tracked_text(draw, (w / 2, y), "COLOPHON", head, ink, tracking=head.size * 0.3)
    y += head.size * 2.2

    for para in lines:
        for line in _wrap(draw, para, f, w - 2 * m):
            draw.text((m, y), line, font=f, fill=soft)
            y += lead
        y += int(lead * 0.5)
    return page


def contents_page(size, entries):
    w, h = size
    page, draw = _blank(size)
    ink = tuple(int(c * 255) for c in INK)
    soft = tuple(int(c * 255) for c in INK_SOFT)

    head = decor.font("antiqua-bold", int(w * 0.024))
    decor.tracked_text(draw, (w / 2, h * 0.13), "CONTENTS", head, ink, tracking=head.size * 0.3)
    _rule(draw, h * 0.175, 0, w * 0.36, w * 0.64, ink, 1)

    f = decor.font("serif", int(w * 0.015))
    fy = decor.font("antiqua", int(w * 0.015))
    fnum = decor.font("antiqua", int(w * 0.014))
    x0, x1 = w * 0.20, w * 0.80
    y = h * 0.24
    lead = int(f.size * 2.1)

    # The plate column is as wide as the widest plate label, measured, not
    # guessed: "PLATE III" is nearly twice the width of "PLATE I", and a fixed
    # offset ran the two columns together on every third line.
    col = max((draw.textlength(num, font=fnum) for num, _, _ in entries), default=0)
    tx = x0 + col + f.size * 0.9

    for num, name, folio in entries:
        draw.text((x0, y), num, font=fnum, fill=soft)
        draw.text((tx, y), name, font=f, fill=ink)
        tw = draw.textlength(name, font=f)
        dots_from = tx + tw + f.size * 0.4
        dots_to = x1 - f.size * 1.4
        if dots_to > dots_from:
            step = f.size * 0.42
            x = dots_from
            while x < dots_to:
                draw.point((x, y + f.size * 0.62), fill=soft)
                x += step
        draw.text((x1, y), str(folio), font=fy, fill=ink, anchor="ra")
        y += lead
    return page


def index_pages(size, entries, per_page=64):
    """An index set in two columns, as an index should be.

    One narrow column leaves two thirds of the page blank and forces the list
    to run to several pages for no reason.
    """
    w, h = size
    pages = []
    ink = tuple(int(c * 255) for c in INK)
    soft = tuple(int(c * 255) for c in INK_SOFT)

    chunks = [entries[i:i + per_page * 2] for i in range(0, len(entries), per_page * 2)]
    for page_no, chunk in enumerate(chunks or [[]]):
        page, draw = _blank(size)
        head = decor.font("antiqua-bold", int(w * 0.024))
        label = "INDEX OF NAMES" if page_no == 0 else "INDEX OF NAMES, CONTINUED"
        decor.tracked_text(draw, (w / 2, h * 0.11), label, head, ink,
                           tracking=head.size * 0.3)
        _rule(draw, h * 0.155, 0, w * 0.36, w * 0.64, ink, 1)

        f = decor.font("serif", int(w * 0.0142))
        half = min(per_page, (len(chunk) + 1) // 2) if chunk else 0
        top = h * 0.215
        lead = min(int(f.size * 1.85), int((h * 0.80 - top) / max(half, 1)))

        for col in range(2):
            part = chunk[col * half:(col + 1) * half] if col else chunk[:half]
            x0 = w * (0.12 + 0.42 * col)
            x1 = w * (0.46 + 0.42 * col)
            y = top
            for name, folio in part:
                draw.text((x0, y), name, font=f, fill=ink)
                draw.text((x1, y), folio, font=f, fill=soft, anchor="ra")
                y += lead
        pages.append(page)
    return pages


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
