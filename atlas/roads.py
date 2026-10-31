"""Roads.

A road is not a straight line between two towns: it is the cheapest way over
the ground, which means it follows valleys and refuses slopes.  So the network
is built in two steps -- a minimum spanning tree decides *which* towns are
connected, and an A* search over a terrain cost field decides *where* each road
actually goes.
"""

from __future__ import annotations

import heapq
import math

import numpy as np

SQRT2 = math.sqrt(2.0)


def cost_field(height, land, factor=3):
    """Terrain cost at reduced resolution: flat valley floor is cheap."""
    hh, ww = height.shape
    H, W = max(1, hh // factor), max(1, ww // factor)
    h = height[: H * factor, : W * factor].reshape(H, factor, W, factor).mean(axis=(1, 3))
    l = land[: H * factor, : W * factor].reshape(H, factor, W, factor).mean(axis=(1, 3)) > 0.35
    gy, gx = np.gradient(h.astype(np.float64))
    slope = np.hypot(gx, gy)

    c = 1.0 + np.clip(np.maximum(h, 0) / 450.0, 0, 8) ** 1.4 + np.clip(slope / 35.0, 0, 5)
    return np.where(l, c, 1e6), factor


def _astar(cost, start, goal):
    H, W = cost.shape
    if not (0 <= start[0] < W and 0 <= start[1] < H and 0 <= goal[0] < W and 0 <= goal[1] < H):
        return None
    sx, sy = start
    gx, gy = goal
    if cost[sy, sx] >= 1e5 or cost[gy, gx] >= 1e5:
        return None

    def h(x, y):
        dx, dy = abs(x - gx), abs(y - gy)
        return max(dx, dy) + (SQRT2 - 1.0) * min(dx, dy)

    open_heap = [(h(sx, sy), 0.0, sx, sy)]
    best = {}
    came = {}
    while open_heap:
        f, g, x, y = heapq.heappop(open_heap)
        if (x, y) == (gx, gy):
            path = [(x, y)]
            while (x, y) in came:
                x, y = came[(x, y)]
                path.append((x, y))
            return path[::-1]
        if (x, y) in best and best[(x, y)] <= g:
            continue
        best[(x, y)] = g
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
            nx, ny = x + dx, y + dy
            if not (0 <= nx < W and 0 <= ny < H):
                continue
            step = SQRT2 if dx and dy else 1.0
            ng = g + step * cost[ny, nx]
            if (nx, ny) in best and best[(nx, ny)] <= ng:
                continue
            came[(nx, ny)] = (x, y)
            heapq.heappush(open_heap, (ng + h(nx, ny), ng, nx, ny))
    return None


def _mst(points):
    """Prim's algorithm on Euclidean distance; returns index pairs."""
    n = len(points)
    if n < 2:
        return []
    in_tree = [False] * n
    best = [float("inf")] * n
    parent = [-1] * n
    best[0] = 0.0
    edges = []
    for _ in range(n):
        u, bu = -1, float("inf")
        for i in range(n):
            if not in_tree[i] and best[i] < bu:
                u, bu = i, best[i]
        if u < 0:
            break
        in_tree[u] = True
        if parent[u] >= 0:
            edges.append((parent[u], u))
        for v in range(n):
            if in_tree[v]:
                continue
            d = math.dist(points[u], points[v])
            if d < best[v]:
                best[v], parent[v] = d, u
    return edges


def _nearest_passable(cost, x, y, radius=8):
    """A coastal town can land in a mostly-water cell once downsampled."""
    H, W = cost.shape
    x = min(max(x, 0), W - 1)
    y = min(max(y, 0), H - 1)
    if cost[y, x] < 1e5:
        return (x, y)
    best, bestd = None, 1e18
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            nx, ny = x + dx, y + dy
            if 0 <= nx < W and 0 <= ny < H and cost[ny, nx] < 1e5:
                d = dx * dx + dy * dy
                if d < bestd:
                    best, bestd = (nx, ny), d
    return best


def build_roads(towns, height, land, factor=3):
    """Connect towns by a least-cost network.  `towns` are (x, y) in cells."""
    if len(towns) < 2:
        return []
    cost, f = cost_field(height, land, factor)
    H, W = cost.shape

    pts = []
    for x, y in towns:
        p = _nearest_passable(cost, int(x // f), int(y // f))
        if p is None:
            continue
        pts.append(p)
    if len(pts) < 2:
        return []

    paths = []
    for a, b in _mst(pts):
        p = _astar(cost, pts[a], pts[b])
        if p and len(p) > 1:
            paths.append([(x * f, y * f) for x, y in p])
    return paths


def smooth(path, window=5):
    """Take the staircase out of a grid path."""
    if len(path) < window:
        return path
    a = np.array(path, dtype=np.float64)
    k = np.ones(window) / window
    x = np.convolve(a[:, 0], k, mode="same")
    y = np.convolve(a[:, 1], k, mode="same")
    x[:window] = a[:window, 0]
    y[:window] = a[:window, 1]
    x[-window:] = a[-window:, 0]
    y[-window:] = a[-window:, 1]
    return list(zip(x.tolist(), y.tolist()))
