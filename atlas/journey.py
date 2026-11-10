"""A journey across a country, taken along the roads that are drawn on it.

The survey tells you what a country is made of and the dictionary tells you
what is in it; neither tells you what it is like to cross.  This module walks
a route between two towns -- along the least-cost road network that the plate
already draws -- and records what the road passes: where it crosses a river,
where it climbs, which province it enters, how far it has come.

Nothing here is invented except the manner of speaking.  Every distance,
height, river and province in the text is read off the terrain the sheets are
drawn from, so the journey and the map cannot disagree.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from . import roads

_COMPASS = ("north", "north-east", "east", "south-east",
            "south", "south-west", "west", "north-west")

# A horse and cart, with the horses changed: about a degree and a half a day.
KM_PER_DAY = 30.0


def _bearing(dx, dy) -> str:
    """Compass word for a step.  +x is east, +y is south."""
    a = float(np.degrees(np.arctan2(dx, -dy))) % 360.0
    return _COMPASS[int((a + 22.5) // 45.0) % 8]


def _article(word: str) -> str:
    return "an" if word[:1].lower() in "aeiou" else "a"


@dataclass
class Station:
    km: float
    elev: float
    kind: str
    name: str = ""
    prov: int = -1


@dataclass
class Journey:
    start: str
    end: str
    stations: list = field(default_factory=list)
    total_km: float = 0.0
    high_m: float = 0.0
    rivers: list = field(default_factory=list)
    provinces: list = field(default_factory=list)
    towns_passed: list = field(default_factory=list)
    heading: str = ""

    @property
    def days(self) -> int:
        return max(2, int(round(self.total_km / KM_PER_DAY)))


def _route(towns, start, end, height, land, factor=3):
    """The least-cost way from town `start` to town `end`.

    This is deliberately not the path through the road tree.  The drawn
    network is a spanning tree, and the tree path between two towns visits
    every town in between -- an itinerary of eight thousand kilometres across
    a country fifteen hundred wide.  A traveller takes the shortest road, so
    the route is one least-cost search on the same cost field the roads are
    built from.
    """
    cost, f = roads.cost_field(height, land, factor)
    a = roads._nearest_passable(cost, int(towns[start][0] // f), int(towns[start][1] // f))
    b = roads._nearest_passable(cost, int(towns[end][0] // f), int(towns[end][1] // f))
    if a is None or b is None:
        return []
    leg = roads._astar(cost, a, b)
    if not leg or len(leg) < 2:
        return []
    return [(x * f, y * f) for x, y in leg]


def plan(world, feats, factor=3):
    """Walk from the seat of government to the town furthest from anywhere."""
    chief = feats.chief_town()
    remote = feats.remotest_town()
    if chief is None or remote is None or chief.name == remote.name:
        return None

    towns = [(t.x, t.y) for t in feats.towns]
    start = next((i for i, t in enumerate(feats.towns) if t.name == chief.name), None)
    end = next((i for i, t in enumerate(feats.towns) if t.name == remote.name), None)
    if start is None or end is None:
        return None

    path = _route(towns, start, end, world.height, world.land, factor)
    if len(path) < 4:
        return None

    hh, ww = world.height.shape
    km_x, km_y = _cell_km(world)
    prov = world.provinces
    rec = world.drainage.get("receivers")

    # A channel is judged against the country's own drainage, not against the
    # largest river in it: the trunk is orders of magnitude above a tributary,
    # and any absolute threshold either drowns the map or finds nothing.
    acc = world.accumulation
    channel = 0.0
    if acc is not None:
        channel = float(np.percentile(acc[world.land], 99.2))
    last_cross = -1000.0
    prev_heavy = False
    j = Journey(start=chief.name.title(), end=remote.name.title())
    total = 0.0
    last = path[0]
    elevs = []
    seen_rivers = set()
    seen_provs = set()
    seen_towns = set()

    # Which river is nearest to a point?  The labels are the only names we have.
    rivers = [(r.name, r.x, r.y) for r in feats.rivers]

    for i, (x, y) in enumerate(path):
        cx, cy = int(round(x)), int(round(y))
        cx = min(max(cx, 0), ww - 1)
        cy = min(max(cy, 0), hh - 1)
        total += float(np.hypot((x - last[0]) * km_x, (y - last[1]) * km_y))
        last = (x, y)
        e = float(world.height[cy, cx])
        elevs.append(e)

        if i == 0:
            j.stations.append(Station(0.0, e, "leave", j.start))
        p = int(prov[cy, cx]) if prov is not None else -1
        if p >= 0 and p not in seen_provs:
            seen_provs.add(p)
            if i > 0:
                j.stations.append(Station(total, e, "enter", prov=p))

        # A river is a channel, not a label: the road crosses water where the
        # drainage is heavy, whatever the label happens to be sitting next to.
        heavy = acc is not None and float(acc[cy, cx]) > channel
        if (heavy and not prev_heavy and total - last_cross > 25.0
                and total > 15.0):
                last_cross = total
                near = None
                for name, rx, ry in rivers:
                    d = np.hypot((rx - x) * km_x, (ry - y) * km_y)
                    if d < 160.0 and (near is None or d < near[0]):
                        near = (d, name)
                label = near[1].title() if near else ""
                j.stations.append(Station(total, e, "river", name=label, prov=p))
        prev_heavy = heavy

        for t in feats.towns:
            if t.name in seen_towns or t.name in (chief.name, remote.name):
                continue
            if abs(t.x - x) <= 6.0 and abs(t.y - y) <= 6.0:
                seen_towns.add(t.name)
                j.stations.append(Station(total, e, "town", name=t.name.title(), prov=p))

    if len(path) > 3:
        j.heading = _bearing(path[3][0] - path[0][0], path[3][1] - path[0][1])
    j.stations.append(Station(total, elevs[-1], "arrive", j.end))
    j.total_km = total
    j.high_m = max(elevs) if elevs else 0.0
    j.rivers = [s.name for s in j.stations if s.kind == "river"]
    j.provinces = [s.prov for s in j.stations if s.kind == "enter"]
    j.towns_passed = [s.name for s in j.stations if s.kind == "town"]

    # The pass: the highest point, if the road really goes over a ridge rather
    # than merely running along high ground.
    k = int(np.argmax(elevs)) if elevs else 0
    ends = max(elevs[0], elevs[-1]) if elevs else 0.0
    if elevs and elevs[k] > ends + 180.0:
        km_at = total * k / max(1, len(elevs) - 1)
        j.stations.append(Station(km_at, elevs[k], "pass", prov=-1))
        j.stations.sort(key=lambda s: s.km)
    return j


def _cell_km(world):
    from .features import cell_km
    return cell_km(world)


def _ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _count(value, step=1):
    return f"{int(round(value, -step)):,}"


def _river_phrase(name: str) -> str:
    return f"the {name}" if name else "a watercourse"


def text(journey: Journey, world, feats, name: str):
    """The journal of the journey, in the order it was travelled.

    Every clause is backed by a station, and every station was measured on the
    way: the distances, the heights, the rivers and the provinces are read off
    the same terrain the plates are drawn from.
    """
    title = f"FROM {journey.start.upper()} TO {journey.end.upper()}"
    subtitle = (f"a journey across {name} in {journey.days} days, "
                f"set down in the order it was travelled")

    heading = f" to the {journey.heading}" if journey.heading else ""
    lead = (f"The road leaves {journey.start} and strikes out{heading}. "
            f"It is {_count(journey.total_km)} kilometres to {journey.end}, "
            f"which at a cart's pace is {journey.days} days.")

    # A journal needs to breathe, not to enumerate: one sentence of each kind,
    # in the order the road meets them, and the count goes in the closing.
    body, seen = [], set()
    for s in journey.stations[1:-1]:
        if s.kind in seen:
            continue
        if s.kind == "enter":
            sentence = (f"At {_count(s.km)} kilometres the road enters "
                        f"{_province_name(s.prov, feats)}.")
        elif s.kind == "pass":
            sentence = (f"The highest ground on it comes at {_count(s.km)} kilometres, "
                        f"{_count(s.elev)} metres above the sea.")
        elif s.kind == "town":
            sentence = f"The road runs through {s.name}."
        elif s.kind == "river":
            sentence = (f"At {_count(s.km)} kilometres it crosses "
                        f"{_river_phrase(s.name)}.")
        else:
            continue
        seen.add(s.kind)
        body.append(sentence)
    body = body[:6]

    crossed = sum(1 for s in journey.stations if s.kind == "river")
    named = sum(1 for s in journey.stations if s.kind == "river" and s.name)
    rivers = (f"{named} of them named" if named and named < crossed else "")
    close = (f"{journey.end} is reached on the {_ordinal(journey.days)} day, "
             f"{_count(journey.total_km)} kilometres from {journey.start}, "
             f"having crossed {crossed} watercourse{'s' if crossed != 1 else ''}"
             f"{' (' + rivers + ')' if rivers else ''}. The highest ground on the "
             f"road is {_count(journey.high_m)} metres.")

    paras = [lead]
    if body:
        half = max(1, len(body) // 2)
        paras.append(" ".join(body[:half]))
        if body[half:]:
            paras.append(" ".join(body[half:]))
    paras.append(close)
    return title, subtitle, paras


def _province_name(index: int, feats) -> str:
    for p in feats.provinces:
        if p.index == index:
            return p.name.title()
    return "the next province"
