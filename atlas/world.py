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
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .config import WorldSpec
from .terrain import build_base, add_detail
from . import climate, hydro, regions

CACHE_DIR = Path(__file__).resolve().parents[1] / "cache"

# Presentation only: changing these must not rebuild the terrain.
_CACHE_IGNORE = {"name", "title", "style", "projection", "supersample", "extra"}


@dataclass
class World:
    spec: WorldSpec
    height: np.ndarray            # metres, sea level at 0
    meta: dict
    drainage: dict
    climate: dict = field(default_factory=dict)
    provinces: np.ndarray | None = None
    extra: dict = field(default_factory=dict)
    _crops: dict = field(default_factory=dict, repr=False, compare=False)

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

        key = (frame.x0, frame.y0, frame.w, frame.h)
        if key in self._crops:
            return self._crops[key]

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
        # Re-route, but fill first: receivers computed on the raw crop leave a
        # pit wherever a basin was only filled in the parent, and a pit is a
        # channel that ends in mid-air -- which is how a 2 700 km river came to
        # be described as 40 km long.  The accumulation is deliberately *not*
        # recomputed: a river entering the frame is already carrying its whole
        # catchment, and restarting the count at the neat line would thin it.
        filled = hydro.fill_depressions(height)
        world.drainage["receivers"] = hydro.receivers(filled, height > 0)[0]
        if self.provinces is not None:
            world.provinces = self.provinces[sl]
        world.climate = dict(self.climate)
        for k, v in self.climate.items():
            if isinstance(v, np.ndarray) and v.shape == self.height.shape:
                world.climate[k] = v[sl]
        world.extra = self.extra
        self._crops[key] = world
        return world


def _cache_key(spec: WorldSpec) -> str:
    """Everything that can change the cached arrays, and nothing else.

    This used to be a hand-written list, which quietly omitted ``coast_freq``,
    ``ridge_width``, ``mountain_freq``, ``relief_bias`` and ``provinces``: two
    worlds that differed in those dials shared a cache entry.  Listing the
    fields that are *ignored* instead means a new dial is covered by default --
    the failure mode is a needless rebuild, not a wrong world.
    """
    payload = {}
    for key, value in asdict(spec).items():
        if key in _CACHE_IGNORE:
            continue
        payload[key] = round(value, 6) if isinstance(value, float) else value
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def _pack_meta(meta) -> dict:
    """Flatten a meta dict into npz entries: arrays as arrays, the rest as JSON."""
    packed = {}
    plain = {}
    for key, value in meta.items():
        if isinstance(value, np.ndarray):
            packed[f"meta_{key}"] = value
        else:
            plain[key] = value
    packed["meta_json"] = np.array(json.dumps(plain))
    return packed


def _load_meta(z) -> dict:
    meta = {}
    if "meta_json" in z.files:
        meta.update(json.loads(str(z["meta_json"])))
    for key in z.files:
        if key.startswith("meta_"):
            meta[key[len("meta_"):]] = z[key]
    return meta


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
            prov = z["provinces"] if "provinces" in z else None
            # The whole meta dict, not just relief: labels place themselves by
            # `inland`, and a warm cache used to fall back to a different field.
            meta = _load_meta(z)
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
    # Cast here, not at the save call: the cache stores float32, so a cold world
    # and a warm one must hold float32 too or the two disagree in the last bits.
    dr["accumulation"] = dr["accumulation"].astype(np.float32)
    dr["lake_depth"] = dr["lake_depth"].astype(np.float32)
    prov, _mouths = regions.provinces(height, dr["receivers"], k=spec.provinces, seed=spec.seed)

    if use_cache:
        np.savez_compressed(
            path,
            height=height,
            accumulation=dr["accumulation"],
            lake_depth=dr["lake_depth"],
            receivers=dr["receivers"],
            provinces=prov,
            **_pack_meta(meta),
        )

    world = World(spec, height, meta, dr)
    world.provinces = prov
    world.climate = climate.derive(world)
    return world
