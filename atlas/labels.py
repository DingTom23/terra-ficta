"""Finding things worth naming, and deciding where their names go.

Two problems, in order:

1. *Which* features deserve a name at this scale -- a continent has thousands
   of hills and four or five things a cartographer would actually label.
2. *Where* the label goes, given that labels must not overlap each other, must
   not sit on a mountain they do not name, and must follow the feature they
   describe.

Both are solved greedily.  Greedy is not optimal, but a map is read by a
human, and a human only notices the labels that collide.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .naming import Namer, Culture


@dataclass
class Label:
    text: str
    x: float
    y: float
    kind: str
    size: int
    tracking: float = 0.0
    angle: float = 0.0
    priority: int = 0
    dot: bool = False
    w: float = 0.0
    h: float = 0.0
    prov: int = -1

    def box(self):
        return (self.x - self.w / 2, self.y - self.h / 2,
                self.x + self.w / 2, self.y + self.h / 2)


def _kmeans(points, k, rng, iters=14):
    """Plain Lloyd's algorithm; enough for scattering labels over a map."""
    if len(points) < k:
        return points, np.arange(len(points))
    idx = rng.choice(len(points), size=k, replace=False)
    cent = points[idx].astype(np.float64)
    labels = np.zeros(len(points), dtype=np.int64)
    for _ in range(iters):
        d = ((points[:, None, :] - cent[None, :, :]) ** 2).sum(-1)
        labels = d.argmin(1)
        for j in range(k):
            m = labels == j
            if m.any():
                cent[j] = points[m].mean(0)
    return cent, labels


def _snap(ys, xs, mask, weights):
    """Pick the member cell with the highest weight."""
    if len(ys) == 0:
        return None
    w = weights[ys, xs]
    i = int(np.argmax(w))
    return float(xs[i]), float(ys[i])


def _river_angle(rec, x, y, ww, steps=12):
    """Direction of flow at a point, in degrees counter-clockwise (PIL's sense)."""
    idx = int(y) * ww + int(x)
    pts = []
    for _ in range(steps):
        r = rec[idx]
        if r < 0:
            break
        pts.append(divmod(int(r), ww))
        idx = int(r)
    if len(pts) < 3:
        return 0.0
    y0, x0 = pts[0]
    y1, x1 = pts[-1]
    ang = math.degrees(math.atan2(-(y1 - y0), (x1 - x0)))
    if ang > 90.0:
        ang -= 180.0
    elif ang < -90.0:
        ang += 180.0
    return ang


def build_labels(world, namer: Namer, n_regions=5, n_seas=3, n_ranges=3,
                 n_towns=12, n_rivers=2, width=None, height=None):
    """Return a priority-ordered list of labels in map-cell coordinates."""
    h = world.height
    hh, ww = h.shape
    land = h > 0
    inland = world.meta.get("inland")
    if inland is None:
        inland = np.clip(h / max(h.max(), 1.0), 0, 1).astype(np.float32)
    rng = np.random.default_rng(world.spec.seed + 4242)

    cultures = namer.cultures
    labels: list[Label] = []

    # --- regions --------------------------------------------------------
    prov = getattr(world, "provinces", None)
    if prov is not None:
        # Place the biggest province first, then let each one dodge the names
        # already on the sheet.  Without the dodge, six provinces whose
        # centroids happen to lie in the same latitude band stack their names
        # into a column, and a column of capitals reads as a table of contents
        # rather than as a map.
        ids = [int(p) for p in np.unique(prov) if p >= 0]
        ids.sort(key=lambda j: -int((prov == j).sum()))
        taken: list[tuple[float, float]] = []
        spread = (ww * 0.16) ** 2
        for j in ids:
            ys, xs = np.where(prov == j)
            if len(ys) < 500:
                continue
            cx, cy = float(xs.mean()), float(ys.mean())
            d2 = (xs - cx) ** 2 + (ys - cy) ** 2
            score = inland[ys, xs] - 2.0e-5 * d2
            for px, py in taken:
                score = score - 0.85 * np.exp(
                    -((xs - px) ** 2 + (ys - py) ** 2) / (2.0 * spread))
            i = int(np.argmax(score))
            if inland[int(ys[i]), int(xs[i])] < 0.10:
                continue
            taken.append((float(xs[i]), float(ys[i])))
            culture = cultures[len(taken) % len(cultures)]
            labels.append(Label(
                text=namer.place(culture, "region").upper(),
                x=float(xs[i]), y=float(ys[i]), kind="region", size=30,
                tracking=5.0, priority=3, prov=int(j),
            ))
    elif len(ys) > n_regions:
        pts = np.stack([xs, ys], 1).astype(np.float64)
        cent, lab = _kmeans(pts, n_regions, rng)
        for j in range(n_regions):
            m = lab == j
            if m.sum() < 400:
                continue
            # Weight by depth inland, squared: a region name belongs in the
            # middle of its region, not on the first headland that qualifies.
            spot = _snap(ys[m], xs[m], land, inland ** 2)
            if spot is None or inland[int(spot[1]), int(spot[0])] < 0.12:
                continue
            culture = cultures[j % len(cultures)]
            labels.append(Label(
                text=namer.place(culture, "region").upper(),
                x=spot[0], y=spot[1], kind="region", size=30,
                tracking=5.0, priority=3,
            ))

    # --- seas -----------------------------------------------------------
    oys, oxs = np.where(~land)
    if len(oys) > n_seas:
        pts = np.stack([oxs, oys], 1).astype(np.float64)
        cent, lab = _kmeans(pts, n_seas, rng)
        depth = np.maximum(-h, 0)
        for j in range(n_seas):
            m = lab == j
            if m.sum() < 3000:
                continue
            spot = _snap(oys[m], oxs[m], ~land, depth)
            if spot is None:
                continue
            culture = cultures[(j + 2) % len(cultures)]
            labels.append(Label(
                text=namer.place(culture, "water", suffix=False),
                x=spot[0], y=spot[1], kind="water", size=26,
                tracking=3.0, priority=2,
            ))

    # --- mountain ranges ------------------------------------------------
    if land.any():
        high = h > max(0.55 * h.max(), 400.0)
        hys, hxs = np.where(high)
        if len(hys) > 200:
            pts = np.stack([hxs, hys], 1).astype(np.float64)
            k = min(n_ranges, max(1, len(hys) // 400))
            cent, lab = _kmeans(pts, k, rng)
            for j in range(k):
                m = lab == j
                if m.sum() < 120:
                    continue
                spot = _snap(hys[m], hxs[m], land, h)
                if spot is None:
                    continue
                culture = cultures[(j + 1) % len(cultures)]
                labels.append(Label(
                    text=namer.place(culture, "range", suffix=False).upper(),
                    x=spot[0], y=spot[1], kind="range", size=20,
                    tracking=4.0, priority=4,
                ))

    # --- towns ----------------------------------------------------------
    acc = world.accumulation
    coast = np.zeros_like(land)
    coast[1:-1, 1:-1] = (
        land[1:-1, 1:-1] & ~(land[:-2, 1:-1] & land[2:, 1:-1] & land[1:-1, :-2] & land[1:-1, 2:])
    )
    near_water = np.zeros_like(land, dtype=np.float32)
    for r in (2, 4, 8):
        near_water = np.maximum(near_water, _dilate(coast, r).astype(np.float32))
    river = acc > max(40.0, np.percentile(acc[land], 97.0)) if land.any() else np.zeros_like(land)

    score = (
        np.clip(np.log1p(acc) / 9.0, 0, 1) * 0.45
        + np.clip(h / 900.0, 0, 1) * -0.20
        + near_water * 0.25
        + river * 0.30
        + land * 0.10
    )
    score = score * land
    flat = score.ravel()
    order = np.argsort(-flat)
    picked = []
    for idx in order[:4000]:
        y, x = divmod(int(idx), ww)
        if not land[y, x]:
            continue
        if any((x - px) ** 2 + (y - py) ** 2 < (ww * 0.085) ** 2 for px, py in picked):
            continue
        picked.append((x, y))
        if len(picked) >= n_towns:
            break
    for i, (x, y) in enumerate(picked):
        culture = cultures[i % len(cultures)]
        labels.append(Label(
            text=namer.place(culture, "town", suffix=True),
            x=float(x), y=float(y), kind="town", size=15, priority=6, dot=True,
        ))

    # --- rivers ---------------------------------------------------------
    rec = world.drainage["receivers"].reshape(hh, ww)
    thr = max(200.0, np.percentile(acc[land], 99.5)) if land.any() else 1e9
    rys, rxs = np.where((acc > thr) & land)
    if len(rys) > 50:
        pts = np.stack([rxs, rys], 1).astype(np.float64)
        k = min(n_rivers, max(1, len(rys) // 300))
        cent, lab = _kmeans(pts, k, rng)
        for j in range(k):
            m = lab == j
            if m.sum() < 40:
                continue
            spot = _snap(rys[m], rxs[m], land, acc)
            if spot is None:
                continue
            culture = cultures[(j + 3) % len(cultures)]
            labels.append(Label(
                text=namer.place(culture, "river", suffix=False),
                x=spot[0], y=spot[1], kind="river", size=15,
                tracking=1.5,
                angle=_river_angle(rec.ravel(), spot[0], spot[1], ww),
                priority=5,
            ))

    for lb in labels:
        lb.w = len(lb.text) * lb.size * 0.62 + lb.tracking * len(lb.text)
        lb.h = lb.size * 1.45
    labels.sort(key=lambda a: a.priority)
    return labels


def avoid(labels, ww, hh, obstacles=()):
    """Greedy placement: keep a label if it fits, drop it if it does not.

    Boxes must already be in map-cell coordinates and measured with the font
    that will actually draw them -- an estimate that disagrees with the type
    is how labels end up on top of each other.
    """
    placed = []
    boxes = list(obstacles)
    for lb in labels:
        x0, y0, x1, y1 = lb.box()
        if x0 < ww * 0.012 or x1 > ww * 0.988 or y0 < hh * 0.012 or y1 > hh * 0.988:
            continue
        b = lb.box()
        if any(_overlap(b, o) for o in boxes):
            continue
        boxes.append(b)
        placed.append(lb)
    return placed


def _dilate(mask, r):
    a = mask
    for _ in range(r):
        m = a.copy()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                m |= np.roll(np.roll(a, dy, 0), dx, 1)
        a = m
    return a


def _overlap(a, b, pad=4.0):
    """True if two boxes overlap once each is grown by `pad` map cells.

    The padding has to be applied to the edges, not multiplied into the
    coordinates: scaling an absolute coordinate moves the box across the
    sheet instead of fattening it.
    """
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ax0 -= pad
    ay0 -= pad
    ax1 += pad
    ay1 += pad
    return not (ax1 < bx0 or bx1 < ax0 or ay1 < by0 or by1 < ay0)
