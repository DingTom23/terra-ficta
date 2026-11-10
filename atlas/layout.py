"""Framing and furniture: deciding what the plate shows, and where the rest goes.

Two separate problems live here.

The *frame* is the crop of the world that the sheet actually shows.  An atlas
plate is not a photograph of a whole map sheet; it is a plate cut to its
subject, with the sea left where the plate needs margin.  So the land is
measured first and the window is fitted to it, rather than the continent being
grown into a canvas of a fixed shape.

The *furniture* is everything printed on the map that is not terrain: the
cartouche, the compass, the two legends, the scale bar.  Their positions were
once constants, which works until a continent grows into the corner the legend
wanted.  Here they are placed by scoring candidate positions against open
water, so the layout bends to the land.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Frame:
    """A crop window on the terrain grid, in cells."""

    x0: int
    y0: int
    w: int
    h: int

    @property
    def aspect(self) -> float:
        return self.w / self.h

    def slices(self):
        return (slice(self.y0, self.y0 + self.h), slice(self.x0, self.x0 + self.w))


def _components(mask: np.ndarray) -> list[tuple[int, int, int, int, int]]:
    """4-connected components of a boolean mask: (area, x0, y0, x1, y1)."""
    h, w = mask.shape
    parent = list(range(h * w))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    flat = mask.ravel()
    for y in range(h):
        row = y * w
        for x in range(w):
            i = row + x
            if not flat[i]:
                continue
            if x and flat[i - 1]:
                ra, rb = find(i), find(i - 1)
                if ra != rb:
                    parent[ra] = rb
            if y and flat[i - w]:
                ra, rb = find(i), find(i - w)
                if ra != rb:
                    parent[ra] = rb

    stats: dict[int, list[int]] = {}
    for y in range(h):
        row = y * w
        for x in range(w):
            if not flat[row + x]:
                continue
            r = find(row + x)
            s = stats.get(r)
            if s is None:
                stats[r] = [1, x, y, x, y]
            else:
                s[0] += 1
                if x < s[1]:
                    s[1] = x
                if y < s[2]:
                    s[2] = y
                if x > s[3]:
                    s[3] = x
                if y > s[4]:
                    s[4] = y
    return [tuple(v) for v in stats.values()]


def main_mass(land: np.ndarray, factor: int = 4, keep: float = 0.0015):
    """Bounding box of the land that matters, in full-grid cells.

    Labelled on a coarse grid: the answer only has to be right to a few cells,
    and the coarse pass costs a hundredth of what a full one would.  Components
    smaller than ``keep`` of the largest are dropped, which is what lets a
    plate frame its continent instead of being held hostage by a rock.
    """
    hh, ww = land.shape
    f = max(1, int(factor))
    ch, cw = (hh + f - 1) // f, (ww + f - 1) // f
    pad = np.zeros((ch * f, cw * f), dtype=bool)
    pad[:hh, :ww] = land
    coarse = pad.reshape(ch, f, cw, f).any(axis=(1, 3))
    if not coarse.any():
        return 0, 0, ww, hh

    comps = _components(coarse)
    comps.sort(key=lambda t: -t[0])
    big = comps[0][0]

    # Start from the largest body, then pull in every island that is already
    # mostly inside the frame.  A far-off islet is left out on purpose: taking
    # it would double the sheet to show one rock, and a frame that reaches the
    # edge of the grid is no frame at all.
    _, a, b, c, d = comps[0]
    x0, y0, x1, y1 = a, b, c, d
    for area, a, b, c, d in comps[1:]:
        if area < keep * big:
            continue
        ox0, oy0 = max(a, x0), max(b, y0)
        ox1, oy1 = min(c, x1), min(d, y1)
        if ox1 <= ox0 or oy1 <= oy0:
            continue
        inside = (ox1 - ox0) * (oy1 - oy0) / max(1.0, float((c - a) * (d - b)))
        if inside >= 0.55:
            x0, y0, x1, y1 = min(a, x0), min(b, y0), max(c, x1), max(d, y1)

    return (x0 * f, y0 * f, min(ww, (x1 + 1) * f), min(hh, (y1 + 1) * f))


def choose_frame(land: np.ndarray, pad: float = 0.16, extra=None) -> Frame:
    """Crop the map to its land, keeping a margin of sea around it.

    ``pad`` is the margin as a fraction of the land's own size, so a wide
    continent keeps a wide margin and a compact one keeps a tight one.

    ``extra`` is accepted and deliberately ignored.  Reaching for the pack ice
    looked right and was wrong: ice lies across the whole northern sea, so a
    frame that grew toward it grew to the edge of the grid and the country
    became a detail in the middle of an ocean.  Where the land runs far enough
    north to matter, the margin already carries the ice.

    The plate is *not* forced to the aspect of the terrain grid.  A continent
    that is tall gets a tall plate, the way an atlas would print it; the book
    is what puts every plate on a page of one size.
    """
    hh, ww = land.shape

    x0, y0, x1, y1 = main_mass(land)
    if x1 <= x0 or y1 <= y0:
        return Frame(0, 0, ww, hh)

    cw = min((x1 - x0) * (1.0 + 2.0 * pad), float(ww))
    ch = min((y1 - y0) * (1.0 + 2.0 * pad), float(hh))

    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    fx0 = float(np.clip(cx - cw / 2.0, 0.0, ww - cw))
    fy0 = float(np.clip(cy - ch / 2.0, 0.0, hh - ch))

    fw = min(int(round(cw)), ww)
    fh = min(int(round(ch)), hh)
    return Frame(int(round(fx0)), int(round(fy0)), fw, fh)


def fit_scale(frame: Frame, target_px: float, lo: int = 1, hi: int = 4) -> int:
    """Supersampling that makes the crop come out about ``target_px`` wide."""
    raw = target_px / max(1, frame.w)
    return int(np.clip(int(raw + 0.5), lo, hi))


# ---------------------------------------------------------------------------
# openness: how far is the nearest land, on a coarse grid
# ---------------------------------------------------------------------------

def openness(land: np.ndarray, blocks: int = 44) -> np.ndarray:
    """Distance to the nearest land cell, in block units, on a coarse grid.

    Used to score furniture positions: a cartouche printed on a mountain range
    is a mistake that no amount of kerning fixes.
    """
    hh, ww = land.shape
    bh = max(1, hh // blocks)
    bw = max(1, ww // blocks)
    ny = max(1, hh // bh)
    nx = max(1, ww // bw)
    frac = land[: ny * bh, : nx * bw].reshape(ny, bh, nx, bw).mean(axis=(1, 3))

    big = 1e6
    d = np.where(frac > 0.04, 0.0, big)
    ys, xs = np.mgrid[0:ny, 0:nx]
    # A few sweeps of a 4-neighbour relaxation is plenty at this resolution.
    for _ in range(max(ny, nx)):
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            np.minimum(d, d[np.clip(ys + dy, 0, ny - 1), np.clip(xs + dx, 0, nx - 1)] + 1.0,
                       out=d)
    d[d > big / 2] = 0.0
    return d / max(1.0, float(d.max()))


# ---------------------------------------------------------------------------
# furniture
# ---------------------------------------------------------------------------

@dataclass
class Slot:
    """One piece of furniture, sized in fractions of the map, with preferences."""

    key: str
    w: float
    h: float
    bias: list[tuple[float, float]] = field(default_factory=list)
    bias_gain: float = 0.55
    sigma: float = 0.22
    shrink: bool = True
    min_scale: float = 0.62
    after: tuple[str, float] | None = None   # (key, dy): stack on that piece
    min_open: float = 0.0                    # refuse to print it on dry land


def _hits(a, b, gap=0.0) -> bool:
    return not (a[2] + gap < b[0] or b[2] + gap < a[0]
                or a[3] + gap < b[1] or b[3] + gap < a[1])


def _openness_of(box, open_field) -> float:
    """How much open water a box sits on.

    The mean is not enough: a key lying mostly over the sea with one corner
    over a headland still averages high, and the corner prints on the land.
    Score the mean but weigh the worst tenth of the box as heavily, so a box
    that touches the coast at all is pushed back into open water.
    """
    x0, y0, x1, y1 = box
    ny, nx = open_field.shape
    jx0 = int(np.clip(x0 * nx, 0, nx - 1))
    jx1 = int(np.clip(x1 * nx, 0, nx))
    jy0 = int(np.clip(y0 * ny, 0, ny - 1))
    jy1 = int(np.clip(y1 * ny, 0, ny))
    patch = open_field[jy0:max(jy1, jy0 + 1), jx0:max(jx1, jx0 + 1)]
    if not patch.size:
        return 0.0
    return 0.55 * float(patch.mean()) + 0.45 * float(np.percentile(patch, 10))


def _score(box, open_field, bias_points, bias_gain: float, sigma: float,
           taken, gap: float) -> float:
    """Higher is better."""
    x0, y0, x1, y1 = box
    ny, nx = open_field.shape
    # mean openness under the box, sampled on the coarse grid
    jx0 = int(np.clip(x0 * nx, 0, nx - 1))
    jx1 = int(np.clip(x1 * nx, 0, nx))
    jy0 = int(np.clip(y0 * ny, 0, ny - 1))
    jy1 = int(np.clip(y1 * ny, 0, ny))
    patch = open_field[jy0:max(jy1, jy0 + 1), jx0:max(jx1, jx0 + 1)]
    op = float(patch.mean()) if patch.size else 0.0

    # distance from the frame edge: furniture crowded against the rule reads as
    # an accident, but a legend floating mid-sea reads as a mistake too
    edge = min(x0, y0, 1.0 - x1, 1.0 - y1)
    edge_score = float(np.clip(edge / 0.06, 0.0, 1.0))

    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    bias = 0.0
    for bx, by in bias_points:
        d2 = ((cx - bx) ** 2 + (cy - by) ** 2) / max(1e-6, sigma ** 2)
        bias = max(bias, float(np.exp(-d2)))

    sep = 1.0
    for other in taken:
        if _hits(box, other, gap):
            return -1e9
        ox = 0.5 * (other[0] + other[2])
        oy = 0.5 * (other[1] + other[3])
        sep = min(sep, float(np.hypot(cx - ox, cy - oy)))

    return 1.35 * op + 0.45 * edge_score + bias_gain * bias + 0.25 * min(1.0, sep / 0.3)


def place(slots: list[Slot], open_field: np.ndarray, gap: float = 0.022,
          steps: int = 11) -> dict[str, tuple[float, float, float, float]]:
    """Greedy placement: each slot takes the best free position it can find.

    Slots are considered in the order given, so put the piece that most needs
    open water -- the cartouche -- first.  A slot that cannot be placed at its
    stated size is retried smaller, because a small title block beats a title
    block printed across a mountain.
    """
    placed: dict[str, tuple[float, float, float, float]] = {}
    taken: list[tuple[float, float, float, float]] = []

    def scan(w: float, h: float, bias, gain: float, sigma: float, need: float = 0.0):
        best = None
        best_score = -1e8
        fallback = None
        fallback_score = -1e8
        for j in range(steps):
            for i in range(steps):
                px = (i + 0.5) / steps
                py = (j + 0.5) / steps
                box = (px - w / 2.0, py - h / 2.0, px + w / 2.0, py + h / 2.0)
                if box[0] < 0.012 or box[1] < 0.012 or box[2] > 0.988 or box[3] > 0.988:
                    continue
                sc = _score(box, open_field, bias, gain, sigma, taken, gap)
                if sc <= -1e8:
                    continue
                if sc > fallback_score:
                    fallback_score, fallback = sc, box
                if need > 0.0 and _openness_of(box, open_field) < need:
                    continue
                if sc > best_score:
                    best_score = sc
                    best = box
        # A piece that only belongs on water still has to appear somewhere:
        # losing the scale bar entirely is worse than printing it near a coast.
        if best is None:
            return fallback, fallback_score
        return best, best_score

    for slot in slots:
        scale = 1.0
        while True:
            w, h = slot.w * scale, slot.h * scale
            best, best_score = scan(w, h, slot.bias, slot.bias_gain, slot.sigma,
                                    slot.min_open)

            # A piece that belongs under another one gets a second chance: it
            # tries to stack, and if the stack lands on land it goes back to
            # choosing for itself rather than vanishing.
            if slot.after and slot.after[0] in placed:
                ref = placed[slot.after[0]]
                rx = 0.5 * (ref[0] + ref[2])
                ry = 0.5 * (ref[1] + ref[3]) + slot.after[1]
                alt, alt_score = scan(w, h, [(rx, ry)], 0.95, 0.085, slot.min_open)
                if alt is not None and alt_score > best_score:
                    best, best_score = alt, alt_score

            if best is not None and best_score > -1e8:
                placed[slot.key] = best
                taken.append(best)
                break
            if not slot.shrink or scale * 0.86 < slot.min_scale:
                break
            scale *= 0.86
    return placed
