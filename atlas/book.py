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


def _tracked_width(draw, text, fnt, tracking):
    """Width of a letter-spaced run, in pixels -- what tracked_text will ink."""
    if not text:
        return 0.0
    return sum(draw.textlength(ch, font=fnt) for ch in text) + tracking * (len(text) - 1)


def title_page(size, title, subtitle, year, plates):
    w, h = size
    page, draw = _blank(size)
    ink = tuple(int(c * 255) for c in INK)
    soft = tuple(int(c * 255) for c in INK_SOFT)

    m = int(min(w, h) * 0.075)
    draw.rectangle([m, m, w - m, h - m], outline=ink, width=max(2, int(m * 0.045)))
    draw.rectangle([m * 1.35, m * 1.35, w - m * 1.35, h - m * 1.35], outline=ink, width=1)

    # A title is set to the frame, not to a fixed size: "TERRA FICTA" at 8.5 per
    # cent of the sheet runs straight through both borders.
    f1 = decor.font("antiqua-bold", int(w * 0.085))
    track = f1.size * 0.34
    while f1.size > 24 and _tracked_width(draw, title.upper(), f1, track) > w * 0.78:
        f1 = decor.font("antiqua-bold", f1.size - 2)
        track = f1.size * 0.34
    decor.tracked_text(draw, (w / 2, h * 0.30), title.upper(), f1, ink, tracking=track)

    _rule(draw, h * 0.40, 0, w * 0.36, w * 0.64, ink, 1)
    draw.ellipse([w / 2 - 4, h * 0.40 - 4, w / 2 + 4, h * 0.40 + 4], fill=ink)
    _rule(draw, h * 0.40, 0, w * 0.30, w * 0.70, ink, 1)

    f2 = decor.font("antiqua-italic", int(w * 0.021))
    draw.text((w / 2, h * 0.455), subtitle, font=f2, fill=soft, anchor="mm")

    f3 = decor.font("serif", int(w * 0.016))
    draw.text((w / 2, h * 0.58), f"{plates} plate" if plates == 1 else f"{plates} plates",
              font=f3, fill=soft, anchor="mm")
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

    x0, x1 = w * 0.20, w * 0.80
    top, bottom = h * 0.22, h * 0.92

    rows = max(1, len(entries))
    avail = bottom - top
    floor = int(w * 0.0075)
    base_size = int(w * 0.015)
    gap = w * 0.035

    def fits(candidate_cols, candidate_size):
        """Whether every entry fits both vertically and beside its folio."""
        per = (rows + candidate_cols - 1) // candidate_cols
        if per * int(candidate_size * 1.45) > avail:
            return False
        f = decor.font("serif", candidate_size)
        fy = decor.font("antiqua", candidate_size)
        fnum = decor.font("antiqua", int(candidate_size * 0.93))
        span = (x1 - x0 - (candidate_cols - 1) * gap) / candidate_cols
        for c in range(candidate_cols):
            cx0 = x0 + c * (span + gap)
            cx1 = cx0 + span
            chunk = entries[c * per:(c + 1) * per]
            col = max((draw.textlength(num, font=fnum)
                       for num, _, _ in chunk), default=0)
            tx = cx0 + col + candidate_size * 0.9
            for _num, name, folio in chunk:
                page_width = draw.textlength(str(folio), font=fy)
                name_right = cx1 - page_width - candidate_size * 0.75
                if draw.textlength(name, font=f) > name_right - tx:
                    return False
        return True

    # Choose the largest type that satisfies both constraints.  The old layout
    # only checked the row count, so a long entry could run through its folio
    # and into the next column even though the last row fitted vertically.
    cols, size = 2, floor
    for candidate_cols in (1, 2):
        candidate_size = base_size
        while candidate_size >= floor and not fits(candidate_cols, candidate_size):
            candidate_size -= 1
        if candidate_size >= floor:
            cols, size = candidate_cols, candidate_size
            break

    f = decor.font("serif", size)
    fy = decor.font("antiqua", size)
    fnum = decor.font("antiqua", int(size * 0.93))
    per = (rows + cols - 1) // cols
    lead = min(int(size * 2.0), max(int(size * 1.35), avail // per))
    span = (x1 - x0 - (cols - 1) * gap) / cols

    def fit_name(name, font, width):
        """Keep a pathological title inside the folio gutter as a last resort."""
        if draw.textlength(name, font=font) <= width:
            return name
        ellipsis = "…"
        while name and draw.textlength(name + ellipsis, font=font) > width:
            name = name[:-1]
        return name + ellipsis if name else ellipsis

    for c in range(cols):
        cx0 = x0 + c * (span + gap)
        cx1 = cx0 + span
        chunk = entries[c * per:(c + 1) * per]
        # The plate column is as wide as the widest label in it, measured and
        # not guessed: "PLATE III" is nearly twice the width of "PLATE I", and
        # a fixed offset ran the two columns together on every third line.
        col = max((draw.textlength(num, font=fnum) for num, _, _ in chunk), default=0)
        tx = cx0 + col + f.size * 0.9
        y = top
        for num, name, folio in chunk:
            draw.text((cx0, y), num, font=fnum, fill=soft)
            page_width = draw.textlength(str(folio), font=fy)
            page_x0 = cx1 - page_width
            name_right = page_x0 - f.size * 0.75
            name = fit_name(name, f, max(1.0, name_right - tx))
            draw.text((tx, y), name, font=f, fill=ink)
            tw = draw.textlength(name, font=f)
            dots_from = tx + tw + f.size * 0.4
            dots_to = page_x0 - f.size * 0.35
            if dots_to > dots_from:
                step = f.size * 0.42
                x = dots_from
                while x < dots_to:
                    draw.point((x, y + f.size * 0.62), fill=soft)
                    x += step
            draw.text((cx1, y), str(folio), font=fy, fill=ink, anchor="ra")
            y += lead
    return page


def index_pages(size, entries, per_page=None):
    """An index set in two columns, as an index should be.

    One narrow column leaves two thirds of the page blank and forces the list
    to run to several pages for no reason.
    """
    w, h = size
    pages = []
    ink = tuple(int(c * 255) for c in INK)
    soft = tuple(int(c * 255) for c in INK_SOFT)

    f = decor.font("serif", int(w * 0.0135))
    top = h * 0.205
    bottom = h * 0.875
    # Leading comes from the type, not from the number of entries: squeezing
    # rows to fit a fixed count is what made the index collide with itself.
    lead = int(f.size * 1.62)
    rows = max(1, int((bottom - top) // lead))
    if per_page:
        rows = max(1, min(rows, per_page))
    per_page = rows * 2
    chunks = [entries[i:i + per_page] for i in range(0, len(entries), per_page)]

    for page_no, chunk in enumerate(chunks or [[]]):
        page, draw = _blank(size)
        head = decor.font("antiqua-bold", int(w * 0.024))
        label = "INDEX OF NAMES" if page_no == 0 else "INDEX OF NAMES, CONTINUED"
        decor.tracked_text(draw, (w / 2, h * 0.11), label, head, ink,
                           tracking=head.size * 0.3)
        _rule(draw, h * 0.155, 0, w * 0.36, w * 0.64, ink, 1)

        half = min(rows, (len(chunk) + 1) // 2) if chunk else 0
        for col in range(2):
            part = chunk[col * half:(col + 1) * half]
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
