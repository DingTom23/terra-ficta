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
    x0, x1 = w * 0.20, w * 0.80
    y = h * 0.24
    lead = int(f.size * 2.1)

    for num, name, folio in entries:
        draw.text((x0, y), num, font=decor.font("antiqua", int(w * 0.014)), fill=soft)
        draw.text((x0 + w * 0.055, y), name, font=f, fill=ink)
        tw = draw.textlength(name, font=f)
        dots_from = x0 + w * 0.055 + tw + f.size * 0.4
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


def index_pages(size, entries, per_page=72):
    """Alphabetical index of every name in the book, with its plate."""
    w, h = size
    pages = []
    head = decor.font("antiqua-bold", int(w * 0.024))
    f = decor.font("serif", int(w * 0.0125))
    cols = 3
    col_w = (w * 0.72) / cols
    lead = int(f.size * 1.85)
    rows = per_page // cols

    for start in range(0, max(len(entries), 1), per_page):
        chunk = entries[start:start + per_page]
        page, draw = _blank(size)
        ink = tuple(int(c * 255) for c in INK)
        soft = tuple(int(c * 255) for c in INK_SOFT)
        decor.tracked_text(draw, (w / 2, h * 0.10), "INDEX OF NAMES", head, ink,
                           tracking=head.size * 0.28)
        _rule(draw, h * 0.145, 0, w * 0.30, w * 0.70, ink, 1)

        x0 = w * 0.14
        for i, (name, plate) in enumerate(chunk):
            c = i // rows
            r = i % rows
            if c >= cols:
                break
            x = x0 + c * col_w
            y = h * 0.21 + r * lead
            draw.text((x, y), name, font=f, fill=ink)
            draw.text((x + col_w - f.size * 0.6, y), str(plate), font=f, fill=soft, anchor="ra")
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
