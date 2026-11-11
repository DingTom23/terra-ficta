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
from dataclasses import dataclass, field, replace

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
    culture: int = -1
    capital: bool = False
    rank: int = 2
    flip: bool = False
    # Where the name sits relative to the dot it belongs to, in cells.  A name
    # that has been pushed aside to fit still points at its town; the line
    # drawn from the dot to the name is the whole reason it is allowed to move.
    ox: float = 0.0
    oy: float = 0.0

    def box(self):
        # A town's name is set beside its dot, not centred on it.
        # Collision-testing it as a centred box let names run under the legends.
        if self.dot:
            gap = self.h * 1.1
            px, py = self.x + self.ox, self.y + self.oy
            if self.flip:
                return (px - self.w - gap, py - self.h / 2,
                        px, py + self.h / 2)
            return (px, py - self.h / 2,
                    px + self.w + gap, py + self.h / 2)
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


def _culture_index(prov, cultures, x, y, fallback=0):
    """Index of the culture of the province containing (x, y).

    One country, one language per province: the province name, its towns and
    its rivers are all drawn from the same phonology.  Where there is no
    province grid the fallback index keeps the choice deterministic.
    """
    if prov is None:
        return fallback % len(cultures)
    p = int(prov[int(y), int(x)])
    return (p if p >= 0 else fallback) % len(cultures)


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
                 n_towns=42, n_rivers=5, n_hamlets=40, width=None, height=None, avoid=(),
                 name_prov=None):
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

    # Nothing on the sheet may share a name with anything else on it.  A
    # province called what the country is called is the one a reader notices.
    used = {a.strip().upper() for a in avoid}

    def unique(fn):
        # `fn` takes the attempt number, so a retry draws a different name
        # from a generator that is otherwise fixed by the feature's identity.
        name = ""
        for k in range(60):
            name = fn(k)
            if name.upper() not in used:
                used.add(name.upper())
                return name
        return name

    def named(key, culture, kind, **kw):
        """One name, from a generator that depends only on what is named."""
        return namer.derive(key)

    # --- regions --------------------------------------------------------
    prov = getattr(world, "provinces", None)
    # Towns and rivers take their language from `name_prov`, which on an
    # older sheet is the modern partition: the borders move, the places do
    # not.
    naming = prov if name_prov is None else name_prov
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
                # A province must have a name even when it is a coastal strip.
                # Leaving it out paints a coloured area that appears in neither
                # the map nor the key, which reads as a mistake.  Take the
                # deepest point it has, and only give up on a true sliver.
                i = int(np.argmax(inland[ys, xs]))
                if inland[int(ys[i]), int(xs[i])] < 0.02:
                    continue
            taken.append((float(xs[i]), float(ys[i])))
            # The province's own id picks its language, not the order it was
            # named in: everything inside province j must speak culture j.
            culture = cultures[int(j) % len(cultures)]
            labels.append(Label(
                text=unique(lambda k: namer.derive(("region", int(j), k)).place(culture, "region").upper()),
                x=float(xs[i]), y=float(ys[i]), kind="region", size=30,
                tracking=5.0, priority=3, prov=int(j), culture=int(j) % len(cultures),
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
                text=unique(lambda k: namer.derive(("region", int(j), k)).place(culture, "region").upper()),
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
                text=namer.derive(("water", round(spot[0]), round(spot[1])))
                     .place(culture, "water", suffix=False),
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
                    text=namer.derive(("range", round(spot[0]), round(spot[1])))
                     .place(culture, "range", suffix=False).upper(),
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

    # A site at one metre above the tide is a mudflat, not a market.  The
    # shelf term pulls the choice a little upstream of every river mouth,
    # which is where a town that lives by water actually stands.
    shelf = np.exp(-((np.log1p(np.maximum(h, 0.0)) - np.log(140.0)) / 1.5) ** 2)
    score = (
        np.clip(np.log1p(acc) / 9.0, 0, 1) * 0.38
        + near_water * 0.16
        + river * 0.24
        + shelf * 0.26
        + land * 0.06
    )
    score = score * land
    flat = score.ravel()
    order = np.argsort(-flat)
    picked = []
    for idx in order[:4000]:
        y, x = divmod(int(idx), ww)
        if not land[y, x]:
            continue
        if any((x - px) ** 2 + (y - py) ** 2 < (ww * 0.052) ** 2 for px, py in picked):
            continue
        picked.append((x, y))
        if len(picked) >= n_towns:
            break

    # Rank the sites by the score that chose them: the great towns stand where
    # water, low ground and the sea all meet, and a plate should say so.
    n = len(picked)
    by_score = sorted(range(n), key=lambda i: -float(score[picked[i][1], picked[i][0]]))
    n_city = max(1, n // 6)
    n_market = max(n_city + 1, n // 2)
    ranks = [2] * n
    for pos, i in enumerate(by_score):
        ranks[i] = 0 if pos < n_city else (1 if pos < n_market else 2)

    for i, (x, y) in enumerate(picked):
        # A town belongs to the province it stands in, and takes that
        # province's language.  Numbering the towns and cycling the cultures
        # gave a country where every neighbouring village spoke a different
        # tongue and the province name matched neither of them.
        ci = _culture_index(naming, cultures, x, y, fallback=i)
        labels.append(Label(
            text=unique(lambda k: namer.derive(("town", round(x), round(y), k))
                               .place(cultures[ci], "town", suffix=True)),
            x=float(x), y=float(y), kind="town", size=15, priority=6, dot=True,
            rank=ranks[i], prov=int(prov[int(y), int(x)]) if prov is not None else -1,
            culture=ci,
        ))

    # Hamlets.  A country of this size has far more places than a sheet can
    # name, so the smallest are drawn as a dot and named only where a name
    # fits.  They are still places: they keep their entry in the dictionary and
    # their line in the index whatever the engraver had room for.
    picked_h = []
    for idx in order[:9000]:
        y, x = divmod(int(idx), ww)
        if not land[y, x]:
            continue
        if any((x - px) ** 2 + (y - py) ** 2 < (ww * 0.030) ** 2
               for px, py in picked + picked_h):
            continue
        picked_h.append((x, y))
        if len(picked_h) >= n_hamlets:
            break
    for i, (x, y) in enumerate(picked_h):
        ci = _culture_index(naming, cultures, x, y, fallback=n + i)
        labels.append(Label(
            text=unique(lambda k: namer.derive(("hamlet", round(x), round(y), k))
                               .place(cultures[ci], "town", suffix=True)),
            x=float(x), y=float(y), kind="town", size=15, priority=6, dot=True,
            rank=3, prov=int(prov[int(y), int(x)]) if prov is not None else -1,
            culture=ci,
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
            ci = _culture_index(naming, cultures, spot[0], spot[1], fallback=j + 3)
            labels.append(Label(
                text=unique(lambda k: namer.derive(("river", round(spot[0]), round(spot[1]), k))
                               .place(cultures[ci], "river", suffix=False)),
                x=spot[0], y=spot[1], kind="river", size=15,
                tracking=1.5,
                angle=_river_angle(rec.ravel(), spot[0], spot[1], ww),
                priority=5, culture=ci,
                prov=int(prov[int(spot[1]), int(spot[0])]) if prov is not None else -1,
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

    A region name is treated as compulsory.  A province whose name falls off
    the edge or lands on a town used to lose its label outright, which paints a
    coloured area that appears in neither the map nor the key -- the reader
    takes that for a mistake, and rightly.  So a region name is first clamped
    inside the margin, then nudged around the collision, and only a province
    with nowhere at all to write keeps silent.
    """
    mx, my = ww * 0.012, hh * 0.012
    placed = []
    boxes = list(obstacles)

    def fits(b):
        if b[0] < mx or b[2] > ww - mx or b[1] < my or b[3] > hh - my:
            return False
        return not any(_overlap(b, o) for o in boxes)

    for lb in labels:
        if fits(lb.box()):
            boxes.append(lb.box())
            placed.append(lb)
            continue
        # A town name that will not fit on the right of its dot is set on the
        # left instead -- the way an engraver would have done it, and the only
        # way to keep towns near the frame from disappearing altogether.
        # A town name is allowed to move off its dot, and the plate draws a
        # hairline back to the town.  Without this a crowd of names along a
        # coast simply vanishes: the tighter the settlement, the fewer towns
        # the sheet shows, which is the opposite of what density should do.
        if lb.kind == "town":
            cands = [replace(lb, flip=not lb.flip)]
            for dx, dy in ((0.0, -0.95), (0.0, 0.95), (-0.70, -0.95),
                           (0.70, 0.95), (-0.70, 0.0), (0.70, 0.0),
                           (0.0, -1.80), (0.0, 1.80)):
                cands.append(replace(lb, ox=dx * lb.w, oy=dy * lb.h))
            for cand in cands:
                if fits(cand.box()):
                    boxes.append(cand.box())
                    placed.append(cand)
                    break
            continue
        if lb.kind != "region":
            continue
        cx = min(max(lb.x, mx + lb.w / 2), ww - mx - lb.w / 2)
        cy = min(max(lb.y, my + lb.h / 2), hh - my - lb.h / 2)
        tries = [replace(lb, x=cx, y=cy)]
        for dx, dy in ((-0.30, 0.0), (0.30, 0.0), (0.0, -0.55), (0.0, 0.55),
                       (-0.30, -0.55), (0.30, 0.55), (-0.30, 0.55), (0.30, -0.55),
                       (0.0, -1.05), (0.0, 1.05)):
            tries.append(replace(lb, x=cx + dx * lb.w, y=cy + dy * lb.h))
        for cand in tries:
            cb = cand.box()
            if fits(cb):
                boxes.append(cb)
                placed.append(cand)
                break
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
