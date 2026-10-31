"""Measured facts about a world.

The gazetteer used to describe the first river in its label list as "the
longest", the first town as "the largest" and the last as "the most remote",
without measuring any of the three.  A sentence that is true by accident is
worse than no sentence: it teaches the reader to trust the rest.

Everything here is computed from the same arrays the drawing uses -- the
receiver network, the accumulation grid, the province grid -- so a claim in
the prose can always be traced back to a number.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .climate import BIOMES
from .hydro import OFFSETS


@dataclass
class River:
    name: str
    x: int
    y: int
    prov: int
    length_km: float
    source: tuple[int, int]
    mouth: tuple[int, int]
    reaches_sea: bool
    drained_km2: float
    source_m: float = 0.0
    range_name: str = ""
    mouth_sea: str = ""
    flow: str = ""


@dataclass
class Town:
    name: str
    x: int
    y: int
    prov: int
    rank: int
    nearest_km: float
    capital: bool = False
    nearest: str = ""
    elev_m: float = 0.0
    river: str = ""
    to_seat_km: float = 0.0
    coast_km: float = -1.0
    pop: int = 0


@dataclass
class Range:
    name: str
    x: int
    y: int
    prov: int
    high_m: float
    holds_peak: bool
    extent_km: float = 0.0
    provinces: str = ""


@dataclass
class Province:
    index: int
    name: str
    cells: int
    area_km2: float
    seat: str = ""
    towns: int = 0
    biome: str = ""
    coast_km: float = 0.0


@dataclass
class Sea:
    name: str
    x: int
    y: int
    shore: str = ""
    rivers: int = 0


@dataclass
class Features:
    rivers: list = field(default_factory=list)
    towns: list = field(default_factory=list)
    ranges: list = field(default_factory=list)
    provinces: list = field(default_factory=list)
    seas: list = field(default_factory=list)
    peak_m: float = 0.0
    land_km2: float = 0.0

    def longest_river(self):
        return max(self.rivers, key=lambda r: r.length_km) if self.rivers else None

    def chief_town(self):
        """The seat of government, or the highest-ranked town if none is marked."""
        if not self.towns:
            return None
        seats = [t for t in self.towns if t.capital]
        if seats:
            return seats[0]
        seats = [t for t in self.towns if t.rank == 0]
        return seats[0] if seats else min(self.towns, key=lambda t: t.rank)

    def remotest_town(self):
        return max(self.towns, key=lambda t: t.nearest_km) if self.towns else None

    def highest_range(self):
        return max(self.ranges, key=lambda r: r.high_m) if self.ranges else None

    def largest_province(self):
        return max(self.provinces, key=lambda p: p.cells) if self.provinces else None


def cell_km(world) -> tuple[float, float]:
    """Ground size of one cell, east and north, in kilometres."""
    spec = world.spec
    hh, ww = world.height.shape
    span = abs(spec.lat_top - spec.lat_bottom)
    km_y = span * 111.32 / max(hh, 1)
    mid = math.radians(0.5 * (spec.lat_top + spec.lat_bottom))
    return km_y * math.cos(mid), km_y


_COMPASS = ("east", "north-east", "north", "north-west",
            "west", "south-west", "south", "south-east")
_SHORE = ("eastern", "north-eastern", "northern", "north-western",
          "western", "south-western", "southern", "south-eastern")


def _bearing(dx: float, dy: float) -> int:
    """Index into the compass tables. +x is east, +y is south."""
    a = math.atan2(-dy, dx)
    return int(round(a / (math.pi / 4.0))) % 8


def _compass(dx: float, dy: float) -> str:
    return _COMPASS[_bearing(dx, dy)]


def _seed_of(name: str) -> int:
    """A stable small integer from a name (Python's hash() is salted)."""
    return sum((i + 1) * ord(c) for i, c in enumerate(name)) % (2 ** 31)


def _dilate(mask):
    """One cell of 4-connected growth."""
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out


def _provinces_phrase(counts: dict, names: dict) -> str:
    """'A', 'A and B', 'A, B and C' -- in order of how much ground each holds."""
    parts = [names[j].title() for j in sorted(counts, key=lambda j: -counts[j])
             if j in names]
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return f"{parts[0]} and {parts[1]}"
    return ", ".join(parts[:-1]) + f" and {parts[-1]}"


def _nearest(items, x, y, km_x, km_y, limit_km):
    """The nearest item with .x/.y, and its distance; None beyond the limit."""
    best, best_d = None, float("inf")
    for it in items:
        d = math.hypot((it.x - x) * km_x, (it.y - y) * km_y)
        if d < best_d:
            best, best_d = it, d
    if best is None or best_d > limit_km:
        return None, best_d
    return best, best_d


def _neighbour_index(i, ww, hh):
    y, x = divmod(i, ww)
    for dy, dx, _dist in OFFSETS:
        ny, nx = y + dy, x + dx
        if 0 <= ny < hh and 0 <= nx < ww:
            yield ny * ww + nx


def _trace(rec, acc, start, ww, hh, km_x, km_y, land_flat):
    """Walk a channel both ways from a point: (length_km, source, mouth, sea)."""
    step_km = 0.5 * (km_x + km_y)

    # Downstream: follow the receiver until the sea or the grid edge.
    i = start
    down = 0.0
    seen = {i}
    while True:
        r = int(rec[i])
        if r < 0 or r in seen:
            break
        seen.add(r)
        y0, x0 = divmod(i, ww)
        y1, x1 = divmod(r, ww)
        down += math.hypot((x1 - x0) * km_x, (y1 - y0) * km_y)
        i = r
        if not land_flat[i]:
            break
    mouth = divmod(i, ww)

    # Upstream: at every fork take the branch that drains the most ground.
    i = start
    up = 0.0
    seen = {i}
    while True:
        best, best_acc = -1, -1.0
        for j in _neighbour_index(i, ww, hh):
            if int(rec[j]) == i and j not in seen and acc[j] > best_acc:
                best, best_acc = j, acc[j]
        if best < 0:
            break
        seen.add(best)
        y0, x0 = divmod(i, ww)
        y1, x1 = divmod(best, ww)
        up += math.hypot((x1 - x0) * km_x, (y1 - y0) * km_y)
        i = best
    source = divmod(i, ww)
    return up + down, source, mouth, not land_flat[mouth[0] * ww + mouth[1]], step_km


def measure(world, labels) -> Features:
    h = world.height
    hh, ww = h.shape
    land = h > 0
    land_flat = land.ravel()
    acc = world.accumulation.ravel()
    rec = world.drainage["receivers"]
    prov_grid = getattr(world, "provinces", None)
    inland = world.meta.get("inland")
    if inland is None:
        inland = np.zeros_like(h)
    km_x, km_y = cell_km(world)
    cell_area = km_x * km_y

    feats = Features(peak_m=float(h.max()), land_km2=float(land.sum() * cell_area))

    def prov_at(x, y):
        return int(prov_grid[int(y), int(x)]) if prov_grid is not None else -1

    for lb in labels:
        if lb.kind == "river":
            x, y = int(lb.x), int(lb.y)
            length, src, mouth, sea, _ = _trace(
                rec, acc, y * ww + x, ww, hh, km_x, km_y, land_flat)
            feats.rivers.append(River(
                name=lb.text, x=x, y=y, prov=prov_at(x, y), length_km=length,
                source=src, mouth=mouth, reaches_sea=sea,
                drained_km2=float(acc[y * ww + x]) * cell_area,
            ))

    for lb in labels:
        if lb.kind == "town":
            x, y = int(lb.x), int(lb.y)
            feats.towns.append(Town(
                name=lb.text, x=x, y=y, prov=prov_at(x, y),
                rank=getattr(lb, "rank", 2), nearest_km=float("inf"),
                capital=bool(getattr(lb, "capital", False)),
                elev_m=float(max(h[y, x], 0.0)),
                coast_km=float(inland[y, x]) * 4200.0,
            ))
    for t in feats.towns:
        for other in feats.towns:
            if other is t:
                continue
            d = math.hypot((other.x - t.x) * km_x, (other.y - t.y) * km_y)
            if d < t.nearest_km:
                t.nearest_km, t.nearest = d, other.name
        if not math.isfinite(t.nearest_km):
            t.nearest_km = 0.0

    names = {int(lb.prov): lb.text for lb in labels
             if lb.kind == "region" and getattr(lb, "prov", -1) >= 0}

    peak_flat = int(np.argmax(h))
    peak_y, peak_x = divmod(peak_flat, ww)
    radius = max(6, int(0.09 * max(ww, hh)))
    for lb in labels:
        if lb.kind != "range":
            continue
        x, y = int(lb.x), int(lb.y)
        y0, y1 = max(0, y - radius), min(hh, y + radius)
        x0, x1 = max(0, x - radius), min(ww, x + radius)
        win = h[y0:y1, x0:x1]
        high = float(win.max())
        holds = (x0 <= peak_x < x1) and (y0 <= peak_y < y1)
        # How far the high ground runs: the diagonal of every cell standing
        # within a quarter of the local maximum.  A ridge is long and thin, so
        # this measures its length rather than the window's width.
        hi = win >= high * 0.75
        if hi.any():
            ys_, xs_ = np.where(hi)
            extent = math.hypot((xs_.max() - xs_.min()) * km_x,
                                (ys_.max() - ys_.min()) * km_y)
        else:
            extent = 0.0
        crossed = {}
        if prov_grid is not None:
            ids, counts = np.unique(prov_grid[y0:y1, x0:x1][hi], return_counts=True)
            crossed = {int(i): int(c) for i, c in zip(ids, counts) if int(i) >= 0}
        feats.ranges.append(Range(
            name=lb.text, x=x, y=y, prov=prov_at(x, y), high_m=high,
            holds_peak=holds, extent_km=extent,
            provinces=_provinces_phrase(crossed, names)))

    water = h <= 0
    shore = water | _dilate(water)
    biome = getattr(world, "biome", None)
    if prov_grid is not None:
        for j in sorted(names):
            cells = int((prov_grid == j).sum())
            prov = Province(index=j, name=names[j], cells=cells,
                            area_km2=cells * cell_area)
            inside = [t for t in feats.towns if t.prov == j]
            prov.towns = len(inside)
            if inside:
                prov.seat = min(inside, key=lambda t: (not t.capital, t.rank)).name
            if biome is not None:
                vals, counts = np.unique(biome[prov_grid == j], return_counts=True)
                if len(vals):
                    prov.biome = BIOMES[int(vals[int(np.argmax(counts))])][0]
            prov.coast_km = int(((prov_grid == j) & shore).sum()) * 0.5 * (km_x + km_y)
            feats.provinces.append(prov)

    land_cx = float(np.where(land)[1].mean()) if land.any() else ww / 2.0
    land_cy = float(np.where(land)[0].mean()) if land.any() else hh / 2.0
    for lb in labels:
        if lb.kind != "water":
            continue
        # The shore the sea washes is the shore the land is on.
        feats.seas.append(Sea(
            name=lb.text, x=int(lb.x), y=int(lb.y),
            shore=_SHORE[_bearing(land_cx - lb.x, land_cy - lb.y)],
        ))

    # --- second pass: facts that need more than one list -----------------
    for t in feats.towns:
        near, _ = _nearest(feats.rivers, t.x, t.y, km_x, km_y, 45.0)
        if near is not None:
            t.river = near.name

    chief = feats.chief_town()
    for t in feats.towns:
        if chief is not None and t is not chief:
            t.to_seat_km = math.hypot((chief.x - t.x) * km_x, (chief.y - t.y) * km_y)
        base = {0: 2500, 1: 700, 2: 90}.get(t.rank, 90)
        rng = np.random.default_rng(_seed_of(t.name))
        t.pop = int(round(base * (0.6 + 1.1 * rng.random()) / 10.0)) * 10

    for r in feats.rivers:
        r.source_m = float(max(h[r.source[0], r.source[1]], 0.0))
        r.flow = _compass(r.mouth[1] - r.source[1], r.mouth[0] - r.source[0])
        near, _ = _nearest(feats.ranges, r.source[1], r.source[0], km_x, km_y, 180.0)
        r.range_name = near.name if near is not None else ""
        if r.reaches_sea:
            sea, _ = _nearest(feats.seas, r.mouth[1], r.mouth[0], km_x, km_y, 1e9)
            r.mouth_sea = sea.name if sea is not None else ""

    for s in feats.seas:
        s.rivers = sum(1 for r in feats.rivers if r.mouth_sea == s.name)

    return feats
