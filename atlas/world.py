"""Assemble a world from a seed.

The pipeline is deliberately ordered, because each stage depends on the one
before it being finished:

    landmass -> drainage -> valleys -> fine detail -> drainage again

Routing water before the valleys exist gives a network that ignores the
topography; adding grain before routing gives a network that is all noise.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import WorldSpec
from .terrain import build_base, add_detail
from . import climate, hydro

CACHE_DIR = Path(__file__).resolve().parents[1] / "cache"


@dataclass
class World:
    spec: WorldSpec
    height: np.ndarray            # metres, sea level at 0
    meta: dict
    drainage: dict
    climate: dict = field(default_factory=dict)
    extra: dict = field(default_factory=dict)

    @property
    def land(self):
        return self.height > 0

    @property
    def accumulation(self):
        return self.drainage["accumulation"]

    @property
    def lake_depth(self):
        return self.drainage["lake_depth"]

    @property
    def biome(self):
        return self.climate.get("biome")


def _cache_key(spec: WorldSpec) -> str:
    payload = json.dumps(
        {
            "seed": spec.seed,
            "width": spec.width,
            "height": spec.height,
            "land": round(spec.land_fraction, 4),
            "blobs": spec.blobs,
            "islands": spec.island_blobs,
            "coast": round(spec.coast_noise, 4),
            "ridges": spec.ridge_count,
            "mamp": round(spec.mountain_amp, 4),
            "carve": round(spec.carve_depth, 2),
            "detail": round(spec.detail, 4),
            "meters": round(spec.meters, 2),
        },
        sort_keys=True,
    )
    return hashlib.sha1(payload.encode()).hexdigest()[:16]


def build_world(spec: WorldSpec, use_cache: bool = True) -> World:
    CACHE_DIR.mkdir(exist_ok=True)
    key = _cache_key(spec)
    path = CACHE_DIR / f"world_{key}.npz"

    if use_cache and path.exists():
        with np.load(path) as z:
            height = z["height"]
            acc = z["accumulation"]
            lake = z["lake_depth"]
            rec = z["receivers"]
            relief = z["relief"] if "relief" in z else None
        meta = {"relief": relief} if relief is not None else {}
        drainage = {"accumulation": acc, "lake_depth": lake, "receivers": rec}
        world = World(spec, height, meta, drainage)
        world.climate = climate.derive(world)
        return world

    base, meta = build_base(spec)

    first = hydro.drainage(base)
    carved = hydro.carve(base, first["accumulation"], depth=spec.carve_depth)

    height = add_detail(spec, carved).astype(np.float32)

    dr = hydro.drainage(height)

    if use_cache:
        np.savez_compressed(
            path,
            height=height,
            accumulation=dr["accumulation"].astype(np.float32),
            lake_depth=dr["lake_depth"].astype(np.float32),
            receivers=dr["receivers"],
            relief=meta["relief"],
        )

    world = World(spec, height, meta, dr)
    world.climate = climate.derive(world)
    return world
