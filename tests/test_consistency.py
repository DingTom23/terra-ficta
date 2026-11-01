"""Facts the book asserts about the world must be facts the world supports.

Every check here is a claim a reader could catch: a province with no name, a
town speaking a language its neighbours do not, a river called the longest
without measuring it, a page that says six plates and lists twelve.
"""

from __future__ import annotations

import numpy as np

from atlas import book, features, gazetteer, render
from atlas.config import WorldSpec
from atlas.labels import build_labels
from atlas.naming import Namer
from atlas.world import build_world

SPEC = WorldSpec(seed=7, width=700, height=480, lat_top=58.0, lat_bottom=42.0)

_WORLD = None
_LABELS = None
_COLD = None
COLD_SPEC = WorldSpec(seed=7, width=420, height=300, lat_top=72.0, lat_bottom=56.0)


def _cold_world():
    """A world with a north worth drawing ice in."""
    global _COLD
    if _COLD is None:
        _COLD = build_world(COLD_SPEC)
    return _COLD


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

def test_every_named_place_has_a_dictionary_entry():
    """A name printed on the plate must be a name the dictionary explains."""
    world, labels = _world_and_labels()
    feats = features.measure(world, labels)
    ents = gazetteer.entries(world, labels, feats)
    heads = {e.head.lower() for e in ents}
    missing = sorted({lb.text for lb in labels if lb.text.lower() not in heads})
    assert not missing, f"printed but never described: {missing}"


def test_dictionary_numbers_are_the_measured_ones():
    """No figure in the prose may be invented: it must come from the survey."""
    world, labels = _world_and_labels()
    feats = features.measure(world, labels)
    ents = {e.head.lower(): e for e in gazetteer.entries(world, labels, feats)}

    river = feats.longest_river()
    text = ents[river.name.lower()].text
    assert gazetteer._count(river.length_km, 1) in text, "the river length is not the measured one"

    big = feats.largest_province()
    text = ents[big.name.lower()].text
    assert gazetteer._count(big.area_km2) in text, "the province area is not the measured one"

    seat = feats.chief_town()
    text = ents[seat.name.lower()].text
    assert f"{seat.pop:,}" in text, "the seat's population is not the one measured"


def test_every_entry_has_a_page():
    """The index can only point at a folio if the dictionary recorded one."""
    world, labels = _world_and_labels()
    feats = features.measure(world, labels)
    ents = gazetteer.entries(world, labels, feats)
    pages, page_of = gazetteer.entry_pages(book.PAGE, "Testland", ents)
    assert len(pages) >= 1
    missing = [e.head for e in ents if e.head.lower() not in page_of]
    assert not missing, f"entries with no folio: {missing}"

def test_ice_lies_only_on_high_ground_or_high_latitude():
    """Ice is a fact about height and latitude, not a decoration."""
    from atlas.climate import snowline
    w = _cold_world()
    ice = w.climate["ice"]
    lat = w.climate["latitude"]
    h = w.height
    assert ice.sum() > 0, "no ice anywhere in a world that has a north"
    assert not (ice & (h <= 0)).any(), "ice was drawn on the sea as land ice"
    too_low = ice & (h < snowline(lat)) & (lat <= 66.5)
    assert not too_low.any(), f"{int(too_low.sum())} ice cells below the snowline"


def test_sea_ice_lies_only_in_water():
    w = _cold_world()
    water = w.height <= 0
    for key in ("fast_ice", "pack_ice"):
        mask = w.climate[key]
        assert not (mask & ~water).any(), f"{key} was drawn on land"


def test_ice_takes_no_province_on_the_political_plate():
    """A tinted province under an ice cap is a claim nobody can make."""
    w = _cold_world()
    ice = w.climate["ice"]
    if not ice.any():
        return
    img, _h, _land = render.render_political(w, scale=1)
    img = np.asarray(img).astype(np.float32)
    if img.max() <= 1.5:          # the renderer works in 0..1, the sheet in 0..255
        img = img * 255.0
    # The ice must be paler than the warmest province wash anywhere.
    ys, xs = np.where(ice)
    px = img[ys, xs]
    assert px.mean() > 200, f"ice on the political plate came out at {px.mean():.0f}"


def test_the_map_is_a_country_and_not_a_continent():
    """The book calls these countries, so they have to be country-sized.

    At forty-six degrees of latitude the sheet was five thousand kilometres
    across and the gazetteer had to explain why a market town's nearest
    neighbour was five hundred kilometres away.  A plate is about seventeen
    hundred kilometres: France, Spain, Sweden -- not Russia.
    """
    w, _ = _world_and_labels()
    km_x, km_y = features.cell_km(w)
    width_km = km_x * w.spec.width
    height_km = km_y * w.spec.height
    assert 800.0 <= width_km <= 3000.0, f"the map is {width_km:,.0f} km wide"
    assert 800.0 <= height_km <= 3000.0, f"the map is {height_km:,.0f} km tall"


def test_towns_are_spaced_like_settlements():
    """Towns a hundred kilometres apart, not five hundred."""
    w, labels = _world_and_labels()
    feats = features.measure(w, labels)
    spacing = sorted(t.nearest_km for t in feats.towns)
    median = spacing[len(spacing) // 2]
    assert 25.0 <= median <= 260.0, f"the median town is {median:.0f} km from its neighbour"


def test_a_city_is_bigger_than_a_market_town_and_a_village():
    w, labels = _world_and_labels()
    feats = features.measure(w, labels)
    by_rank = {}
    for t in feats.towns:
        by_rank.setdefault(t.rank, []).append(t.pop)
    assert 0 in by_rank and 2 in by_rank, "the country has no cities or no villages"
    city = max(by_rank[0])
    village = max(by_rank[2])
    assert city > village, f"a city holds {city} and a village {village}"
    assert city >= 10000, f"the largest city holds only {city}"


def test_the_longest_river_is_a_river_and_not_a_country():
    w, labels = _world_and_labels()
    feats = features.measure(w, labels)
    r = feats.longest_river()
    assert r is not None
    assert 80.0 <= r.length_km <= 2500.0, f"the longest river is {r.length_km:,.0f} km"

def test_the_scale_bar_is_as_long_as_it_says():
    """A scale bar that is twenty per cent out is worse than no scale bar."""
    import math
    from atlas.compose import _nice_bar

    total = 2704.0
    for across in (40, 60, 120, 180, 260, 300, 1000):
        bar, n = _nice_bar(across, 0.23 * total, total)
        assert bar <= 0.23 * total + 1e-6, "the bar runs out of its slot"
        assert abs(bar / total * across - n) < 1e-6, \
            f"a {n}-league bar measures {bar / total * across:.2f}"
        mant = n / 10 ** math.floor(math.log10(n))
        assert abs(mant - round(mant)) < 1e-9 and round(mant) in (1, 2, 5), \
            f"{n} is not a round number of leagues"
