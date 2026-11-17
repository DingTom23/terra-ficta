"""What kind of place a cell is.

A settlement is not a dot with a score.  It stands somewhere for a reason: at
the mouth of a river, where two channels meet, on sheltered water, in a gap
between hills, or beside a ford.  A map whose names are drawn from a phonology
but whose suffixes are drawn from a hat can print *Achmouth* on a hilltop --
and a reader who looks will find the lie.

Everything here is measured from the same arrays the drawing uses: the height
field, the D8 receivers, flow accumulation and the lake depths.  The names
module then lets a suffix follow the ground, so a place called *-mouth* is at a
river mouth and a place called *-gate* stands in a gap.

The taxonomy is deliberately small.  These are the kinds of site a period
sheet can show with one word:

    river_mouth   where a channel enters the sea
    lake_outlet   where a lake drains
    harbour       sheltered water with land on both sides
    confluence    where a significant branch joins its parent
    crossing      on a channel, away from the estuary
    gate          a low way between higher ground
    plain         none of the above

A site can satisfy several descriptions at once; ``tags_at`` returns them in
priority order and the name pool takes the union.
"""

from __future__ import annotations

import numpy as np

SITE_ORDER = ("river_mouth", "lake_outlet", "harbour", "confluence",
              "crossing", "gate")

# A suffix is a claim.  Only the suffixes below make a geographical claim, so
# only these are checked by ``tests/test_consistency.py``.  The rest are the
# ordinary vocabulary of settlement (farm, village, fortified town) and can
# stand anywhere.
SUFFIX_SITES = {
    "mouth": ("river_mouth",),
    "haven": ("harbour", "river_mouth"),
    "ford": ("crossing",),
    "bridge": ("crossing",),
    "gate": ("gate",),
}

# Suffixes that assert nothing about the ground.  They are always available,
# which also keeps a name from being forced into a claim the site cannot
# support.
GENERIC_SUFFIXES = ("stead", "thorpe", "burg", "wick")


def suffixes_for(tags) -> tuple[str, ...]:
    """The suffixes a site can truthfully wear."""
    pool = list(GENERIC_SUFFIXES)
    present = set(tags or ())
    for suffix, allowed in SUFFIX_SITES.items():
        if present.intersection(allowed):
            pool.append(suffix)
    return tuple(pool)


def _shift(a, dy, dx, fill=0):
    """Move an array by (dy, dx) without wrapping at the edges.

    ``np.roll`` would bring the far shore round to the near one: a harbour
    test at the edge of the sheet would find land off the map and invent a
    cove.  The border is an outlet, not a mirror.
    """
    out = np.full_like(a, fill)
    h, w = a.shape
    sy0, sy1 = max(0, -dy), min(h, h - dy)
    sx0, sx1 = max(0, -dx), min(w, w - dx)
    dy0, dy1 = max(0, dy), min(h, h + dy)
    dx0, dx1 = max(0, dx), min(w, w + dx)
    if dy0 < dy1 and dx0 < dx1:
        out[dy0:dy1, dx0:dx1] = a[sy0:sy1, sx0:sx1]
    return out


def _dilate(mask, r):
    """Square growth by ``r`` cells -- the zone around a site."""
    a = mask
    for _ in range(max(0, int(r))):
        grown = a.copy()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy or dx:
                    grown |= _shift(a, dy, dx, False)
        a = grown
    return a


def _erode(mask, r):
    """The dual of ``_dilate``: a cell survives only if its whole square does."""
    a = mask
    for _ in range(max(0, int(r))):
        shrunk = a.copy()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy or dx:
                    shrunk &= _shift(a, dy, dx, False)
        a = shrunk
    return a


def _open(mask, r):
    """Remove puddles: erase anything thinner than its own square, then grow back.

    The median "lake" on this terrain is two cells.  A town called "beside a
    lake outlet" because a pond drains there is a true statement that explains
    nothing, so the outlet test runs on lakes that survive an opening.
    """
    return _dilate(_erode(mask, r), r)


class SiteFields:
    """Boolean masks, one per site kind, on the world grid."""

    def __init__(self, masks):
        self.masks = masks


def fields(world) -> SiteFields:
    """Classify every cell of a world.  Cached on the world object.

    The masks are expensive enough to compute once and cheap enough to keep:
    four plates of one country share a world, and a crop has its own grid and
    therefore its own fields.
    """
    cached = getattr(world, "_site_fields", None)
    if cached is not None:
        return cached

    h = world.height
    land = h > 0
    acc = np.asarray(world.accumulation, dtype=np.float64)
    rec = world.drainage["receivers"].reshape(h.shape)
    lake = world.lake_depth > 2.0
    hh, ww = h.shape

    # Distances are ground distances, not pixels.  A 1400-cell sheet and a
    # 700-cell sheet show the same country at different resolutions, so a
    # radius that is four cells on one is eight on the other.  Without this,
    # a full-resolution plate has no settlements at river mouths at all: its
    # towns stand a few cells further from the sea than the small preview's.
    unit = max(1.0, max(hh, ww) / 700.0)

    def zone(cells):
        return max(1, int(round(cells * unit)))

    # A channel worth naming a town after.  The 97th percentile is about the
    # same weight the settlement score already requires of a river.
    if land.any():
        threshold = max(200.0, float(np.percentile(acc[land], 97.0)))
    else:
        threshold = float("inf")
    channel = land & (acc >= threshold)

    # --- river mouth -----------------------------------------------------
    # A mouth is a land cell whose receiver is sea.  A receiver of -1 is the
    # edge of the sheet, not a coast: a river leaving the neat line has not
    # reached the sea and must not be called a mouth.
    rec_flat = rec.ravel()
    water_flat = ~land.ravel()
    valid = rec_flat >= 0
    receiver_water_flat = np.zeros(h.size, dtype=bool)
    receiver_water_flat[valid] = water_flat[rec_flat[valid]]
    receiver_water = receiver_water_flat.reshape(h.shape)
    mouth_core = channel & receiver_water
    mouth = _dilate(mouth_core, zone(4)) & land

    # --- lake outlet -----------------------------------------------------
    # Follow the receiver of each lake cell; if it leaves the lake, that is
    # where the lake drains.  Only lakes that survive an opening count: a
    # two-cell pond has an outlet too, and it is not a reason for a town.
    big_lake = _open(lake, zone(2))
    lake_flat = big_lake.ravel()
    outlet_core = np.zeros(h.size, dtype=bool)
    lake_cells = np.flatnonzero(lake_flat)
    if lake_cells.size:
        targets = rec_flat[lake_cells]
        leaves = targets < 0
        inside = targets >= 0
        leaves[inside] = ~lake_flat[targets[inside]]
        outlet_core[lake_cells[leaves]] = True
    outlet_core = outlet_core.reshape(h.shape)
    lake_outlet = _dilate(outlet_core, zone(3)) & land

    # --- confluence ------------------------------------------------------
    # A cell where at least two incoming branches each carry a real share of
    # the flow.  Counting every donor would make a raindrop a confluence.
    index = np.arange(h.size).reshape(h.shape)
    donors = np.zeros(h.shape, dtype=np.int8)
    branch_floor = np.maximum(100.0, 0.22 * acc)
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1),
                   (1, 1), (1, -1), (-1, 1), (-1, -1)):
        donor = (_shift(rec, dy, dx, -2) == index)
        donor &= _shift(land, dy, dx, False)
        donor &= _shift(acc, dy, dx, 0.0) >= branch_floor
        donors += donor.astype(np.int8)
    confluence_core = channel & (donors >= 2)
    confluence = _dilate(confluence_core, zone(3)) & land

    # --- crossing --------------------------------------------------------
    # A crossing is a channel reach away from the estuary.  The raster does
    # not record bank width, so the honest claim is "on a river, not at its
    # mouth"; that is what the suffix is allowed to say.
    crossing = _dilate(channel & ~mouth_core & ~lake, zone(2)) & land

    # --- harbour ---------------------------------------------------------
    # Sheltered water: a water cell that can see land on opposite sides at a
    # boat's radius, and is deep enough to float one.  A straight coast has
    # land on one side only and cannot pass this test; a bay or an estuary
    # can.
    water = ~land
    depth = np.maximum(-h, 0.0)
    broad = zone(8)
    opposite = ((_shift(land, 0, broad, False) & _shift(land, 0, -broad, False))
                | (_shift(land, broad, 0, False) & _shift(land, -broad, 0, False)))
    harbour_core = water & opposite & (depth > 8.0)
    harbour = _dilate(harbour_core, zone(3)) & land

    # --- gate ------------------------------------------------------------
    # A low way between high ground: two opposite directions, a ridge apart,
    # both stand a good deal higher than the cell.  This finds valley gaps and
    # saddles without pretending to know the difference on a raster this
    # coarse.
    reach = zone(8)
    rise = 120.0
    dirs = ((reach, 0), (-reach, 0), (0, reach), (0, -reach),
            (reach, reach), (-reach, -reach), (reach, -reach), (-reach, reach))
    shoulders = [_shift(h, dy, dx, 1e9) for dy, dx in dirs]
    gate_core = land & (h > 20.0) & (
        ((shoulders[0] > h + rise) & (shoulders[1] > h + rise))
        | ((shoulders[2] > h + rise) & (shoulders[3] > h + rise))
        | ((shoulders[4] > h + rise) & (shoulders[5] > h + rise))
        | ((shoulders[6] > h + rise) & (shoulders[7] > h + rise))
    )
    gate = _dilate(gate_core, zone(2)) & land

    result = SiteFields({
        "river_mouth": mouth,
        "lake_outlet": lake_outlet,
        "harbour": harbour,
        "confluence": confluence,
        "crossing": crossing,
        "gate": gate,
    })
    world._site_fields = result
    return result


def tags_at(site_fields: SiteFields, x: float, y: float) -> tuple[str, ...]:
    """Every site description that holds at (x, y), in priority order."""
    masks = site_fields.masks
    h, w = next(iter(masks.values())).shape
    xi = min(max(int(round(x)), 0), w - 1)
    yi = min(max(int(round(y)), 0), h - 1)
    return tuple(kind for kind in SITE_ORDER if bool(masks[kind][yi, xi]))


def primary(tags) -> str:
    """The first description, or 'plain' when the ground makes no claim."""
    return tags[0] if tags else "plain"
