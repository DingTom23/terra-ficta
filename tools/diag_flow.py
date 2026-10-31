"""Diagnose the flow graph: how much of the land actually drains anywhere?"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from atlas.config import WorldSpec
from atlas.terrain import build_base
from atlas import hydro

spec = WorldSpec(seed=7, width=600, height=400)
h, meta = build_base(spec)
land = h > 0
print(f"land cells            {land.sum()}  ({100 * land.mean():.1f}%)")

filled = hydro.fill_depressions(h)
print(f"raised by fill        {((filled - h) > 1e-6).sum()} cells")
print(f"max fill depth        {(filled - h).max():.1f} m")

rec, slope = hydro.receivers(filled, land)
rec2 = rec.reshape(h.shape)
sink = (rec2 < 0) & land
print(f"land cells with no receiver: {sink.sum()}  ({100 * sink.sum() / land.sum():.1f}%)")

acc = hydro.accumulate(rec, h.size).reshape(h.shape)
a = acc[land]
print("accumulation percentiles:", np.percentile(a, [50, 90, 99, 99.9]).round(1), "max", acc.max())

# How many distinct outlets does the land drain to?
outlets = np.unique(rec2[sink])
print(f"distinct outlet cells:  {len(outlets)}")

# Basin sizes: follow every cell to its terminal sink
print("flat cells (drop==0):", int((slope.reshape(h.shape) == 0).sum()))
