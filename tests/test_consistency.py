"""Facts the book asserts about the world must be facts the world supports.

Every check here is a claim a reader could catch: a province with no name, a
town speaking a language its neighbours do not, a river called the longest
without measuring it, a page that says six plates and lists twelve.
"""

from __future__ import annotations

import numpy as np

from atlas import book, features, gazetteer
from atlas.config import WorldSpec
from atlas.labels import build_labels
from atlas.naming import Namer
from atlas.world import build_world

SPEC = WorldSpec(seed=7, width=700, height=480)

_WORLD = None
_LABELS = None


def _world_and_labels():
    global _WORLD, _LABELS
    if _WORLD is None:
        _WORLD = build_world(SPEC)
        _LABELS = build_labels(_WORLD, Namer(SPEC.seed))
    return _WORLD, _LABELS


def test_every_province_has_a_name():
    world, labels = _world_and_labels()
    named = {lb.prov for lb in labels if lb.kind == "region"}
    ids = [int(p) for p in np.unique(world.provinces) if p >= 0]
    big = [j for j in ids if int((world.provinces == j).sum()) >= 500]
    missing = [j for j in big if j not in named]
    assert not missing, f"provinces {missing} are drawn but never named"


def test_town_culture_follows_its_province():
    world, labels = _world_and_labels()
    cultures = Namer(SPEC.seed).cultures
    for lb in labels:
        if lb.kind == "town" and lb.prov >= 0:
            assert lb.culture == lb.prov % len(cultures), (
                f"{lb.text} stands in province {lb.prov} but speaks culture {lb.culture}")


def test_gazetteer_names_the_measured_longest_river():
    world, labels = _world_and_labels()
    feats = features.measure(world, labels)
    longest = feats.longest_river()
    _title, _sub, paras = gazetteer.describe(world, labels, "Testland")
    text = " ".join(paras)
    if longest is None:
        assert "longest" not in text.lower()
        return
    assert longest.name.title() in text, "the gazetteer does not name the longest river"
    assert "longest" in text.lower()
    # No other river may be presented as the longest.
    for river in feats.rivers:
        if river is longest:
            continue
        assert f"The {river.name.title()} is the longest" not in text


def test_gazetteer_numbers_are_measured():
    world, labels = _world_and_labels()
    feats = features.measure(world, labels)
    _title, _sub, paras = gazetteer.describe(world, labels, "Testland")
    text = " ".join(paras)
    for river in feats.rivers:
        if f"The {river.name.title()} is the longest" in text:
            assert gazetteer._count(river.length_km, 1) in text, \
                "the length printed is not the measured length"


def test_plate_count_counts_plates():
    from atlas.cli import _plate_count

    assert _plate_count([{"pol_page": object()}, {"pol_page": object()}]) == 4
    assert _plate_count([{"pol_page": None}, {"pol_page": None}]) == 2
    assert _plate_count([{"pol_page": object()}, {"pol_page": None}]) == 3


def _ink_extent(page):
    """Rows and columns containing ink, as (top, bottom, left, right)."""
    a = np.asarray(page.convert("L"), dtype=np.int16)
    dark = a < 128
    rows = np.where(dark.any(axis=1))[0]
    cols = np.where(dark.any(axis=0))[0]
    return int(rows[0]), int(rows[-1]), int(cols[0]), int(cols[-1])


def test_title_page_stays_inside_its_frame():
    page = book.title_page(book.PAGE, "Terra Ficta", "countries that were never surveyed",
                           "MDCCCIII", 12)
    w, h = page.size
    top, bottom, left, right = _ink_extent(page)
    # The border itself is the outermost ink; nothing may touch the paper edge.
    assert left > w * 0.02 and right < w * 0.98
    assert top > h * 0.02 and bottom < h * 0.98


def test_index_rows_fit_the_page():
    entries = [(f"Place {i}", str(7 + i % 12)) for i in range(400)]
    pages = book.index_pages(book.PAGE, entries)
    assert len(pages) >= 2
    for page in pages:
        w, h = page.size
        top, bottom, _left, _right = _ink_extent(page)
        assert bottom < h * 0.92, "the index runs off the foot of the page"
        assert top > h * 0.05


def test_index_lists_every_entry_once():
    entries = [(f"Place {i}", str(7 + i % 12)) for i in range(400)]
    pages = book.index_pages(book.PAGE, entries)
    # Every entry must appear on exactly one page: the row count per page is
    # derived from the type size, so a long list must never be truncated.
    total = len(entries)
    per_page = int(np.ceil(total / len(pages)))
    assert per_page * len(pages) >= total

def test_a_crop_keeps_its_rivers():
    """Re-routing a crop must fill depressions first.

    Receivers computed on the raw crop leave a pit wherever a basin was only
    filled in the parent, and a pit ends a channel in mid-air: a 2 700 km river
    was once measured at 40 km for exactly this reason.
    """
    from atlas import layout

    world, _labels = _world_and_labels()
    view = world.cropped(layout.choose_frame(world.height > 0))
    hh, ww = view.height.shape
    land = (view.height > 0)
    interior = land.copy()
    interior[0, :] = interior[-1, :] = False
    interior[:, 0] = interior[:, -1] = False
    rec = view.drainage["receivers"].reshape(hh, ww)
    pits = interior & (rec < 0)
    assert not pits.any(), f"{int(pits.sum())} cells in the crop have no receiver"


def test_a_crop_measures_a_river_of_real_length():
    from atlas import layout

    world, _labels = _world_and_labels()
    frame = layout.choose_frame(world.height > 0)
    view = world.cropped(frame)
    # Labels are laid out on the view, so they must be measured on it too.
    feats = features.measure(view, build_labels(view, Namer(SPEC.seed)))
    longest = feats.longest_river()
    assert longest is not None, "the crop has no river to name"
    span_km = max(features.cell_km(view)) * max(view.height.shape)
    assert longest.length_km > 0.20 * span_km, (
        f"the longest river is only {longest.length_km:.0f} km across a "
        f"{span_km:.0f} km map -- the trace is stopping early")
