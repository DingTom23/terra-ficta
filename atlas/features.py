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


@dataclass
class Range:
    name: str
    x: int
    y: int
    prov: int
    high_m: float
    holds_peak: bool


@dataclass
class Province:
    index: int
    name: str
    cells: int
    area_km2: float


@dataclass
class Features:
    rivers: list = field(default_factory=list)
    towns: list = field(default_factory=list)
    ranges: list = field(default_factory=list)
    provinces: list = field(default_factory=list)
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
            feats.towns.append(Town(
                name=lb.text, x=int(lb.x), y=int(lb.y), prov=prov_at(int(lb.x), int(lb.y)),
                rank=getattr(lb, "rank", 2), nearest_km=float("inf"),
                capital=bool(getattr(lb, "capital", False)),
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

    peak_flat = int(np.argmax(h))
    peak_y, peak_x = divmod(peak_flat, ww)
    radius = max(6, int(0.09 * max(ww, hh)))
    for lb in labels:
        if lb.kind != "range":
            continue
        x, y = int(lb.x), int(lb.y)
        y0, y1 = max(0, y - radius), min(hh, y + radius)
        x0, x1 = max(0, x - radius), min(ww, x + radius)
        high = float(h[y0:y1, x0:x1].max())
        holds = (x0 <= peak_x < x1) and (y0 <= peak_y < y1)
        feats.ranges.append(Range(
            name=lb.text, x=x, y=y, prov=prov_at(x, y), high_m=high, holds_peak=holds))

    names = {int(lb.prov): lb.text for lb in labels
             if lb.kind == "region" and getattr(lb, "prov", -1) >= 0}
    if prov_grid is not None:
        for j in sorted(names):
            cells = int((prov_grid == j).sum())
            feats.provinces.append(Province(
                index=j, name=names[j], cells=cells, area_km2=cells * cell_area))

    return feats
