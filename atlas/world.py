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
from . import climate, hydro, regions

CACHE_DIR = Path(__file__).resolve().parents[1] / "cache"


@dataclass
class World:
    spec: WorldSpec
    height: np.ndarray            # metres, sea level at 0
    meta: dict
    drainage: dict
    climate: dict = field(default_factory=dict)
    provinces: np.ndarray | None = None
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

    def cropped(self, frame) -> "World":
        """A view of part of this world, as a world of its own.

        The frame cuts the terrain, but it also has to cut the *drainage*: a
        receiver is a flat index into the full grid, so a crop cannot inherit
        the network.  It is cheaper to re-route the crop (a fraction of a
        second) than to translate every index, and the result is a correct
        network whose outlets are the crop's own edges.
        """
        from dataclasses import replace

        sl = frame.slices()
        height = self.height[sl]
        acc = self.accumulation[sl]
        lake = self.lake_depth[sl]

        spec = self.spec
        hh, ww = self.height.shape
        dh = spec.lat_bottom - spec.lat_top
        lat_top = spec.lat_top + dh * (frame.y0 / hh)
        lat_bottom = spec.lat_top + dh * ((frame.y0 + frame.h) / hh)
        lon_span = dh * (ww / hh) * -1.0
        lon_left = spec.lon_left + lon_span * (frame.x0 / ww)

        sub = replace(spec, width=frame.w, height=frame.h,
                      lat_top=lat_top, lat_bottom=lat_bottom, lon_left=lon_left)

        meta = {}
        for k, v in self.meta.items():
            if isinstance(v, np.ndarray) and v.shape == self.height.shape:
                meta[k] = v[sl]
            else:
                meta[k] = v

        world = World(sub, height, meta, {"accumulation": acc, "lake_depth": lake})
        world.drainage["receivers"] = hydro.receivers(height, height > 0)[0]
        if self.provinces is not None:
            world.provinces = self.provinces[sl]
        world.climate = dict(self.climate)
        for k, v in self.climate.items():
            if isinstance(v, np.ndarray) and v.shape == self.height.shape:
                world.climate[k] = v[sl]
        world.extra = self.extra
        return world


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
            prov = z["provinces"] if "provinces" in z else None
        meta = {"relief": relief} if relief is not None else {}
        drainage = {"accumulation": acc, "lake_depth": lake, "receivers": rec}
        world = World(spec, height, meta, drainage)
        world.provinces = prov
        world.climate = climate.derive(world)
        return world

    base, meta = build_base(spec)

    first = hydro.drainage(base)
    carved = hydro.carve(base, first["accumulation"], depth=spec.carve_depth)

    height = add_detail(spec, carved).astype(np.float32)

    dr = hydro.drainage(height)
    prov, _mouths = regions.provinces(height, dr["receivers"], k=spec.provinces, seed=spec.seed)

    if use_cache:
        np.savez_compressed(
            path,
            height=height,
            accumulation=dr["accumulation"].astype(np.float32),
            lake_depth=dr["lake_depth"].astype(np.float32),
            receivers=dr["receivers"],
            relief=meta["relief"],
            provinces=prov,
        )

    world = World(spec, height, meta, dr)
    world.provinces = prov
    world.climate = climate.derive(world)
    return world
