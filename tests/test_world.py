"""Same seed, same world -- cold cache or warm.

The cache exists so a rebuild costs seconds instead of minutes.  It must not
be able to change the answer: a warm run has to produce the same arrays, the
same metadata and the same labels as a cold one.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from atlas.config import WorldSpec
from atlas.world import _cache_key, build_world

SPEC = WorldSpec(seed=7, width=420, height=280)


def test_cache_key_covers_every_terrain_dial():
    ref = _cache_key(SPEC)
    dials = {
        "coast_freq": 3.9,
        "ridge_width": 0.22,
        "mountain_freq": 7.5,
        "relief_bias": 0.80,
        "provinces": 3,
        "land_fraction": 0.40,
        "island_blobs": 6,
        "carve_depth": 300.0,
        "meters": 5200.0,
    }
    for name, value in dials.items():
        other = replace(SPEC, **{name: value})
        assert _cache_key(other) != ref, f"changing {name} leaves the cache key unchanged"


def test_cache_key_ignores_presentation():
    ref = _cache_key(SPEC)
    for name, value in {"name": "x", "title": "y", "style": "z", "supersample": 3}.items():
        assert _cache_key(replace(SPEC, **{name: value})) == ref, \
            f"{name} is presentation only but rebuilds the world"


def _plain(value):
    """Tuples become lists: the cache stores meta as JSON, which has no tuples."""
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def test_cold_and_warm_worlds_agree():
    cold = build_world(SPEC, use_cache=False)
    build_world(SPEC, use_cache=True)          # computes and writes
    warm = build_world(SPEC, use_cache=True)   # reads back

    assert np.array_equal(cold.height, warm.height)
    assert np.array_equal(cold.accumulation, warm.accumulation)
    assert np.array_equal(cold.lake_depth, warm.lake_depth)
    assert np.array_equal(cold.drainage["receivers"], warm.drainage["receivers"])
    assert np.array_equal(cold.provinces, warm.provinces)

    for key in ("mask", "inland", "belt", "relief"):
        assert key in warm.meta, f"{key} did not survive the cache"
        assert np.allclose(cold.meta[key], warm.meta[key]), f"{key} differs after a round trip"

    for key in ("blobs", "ridge_axes", "sea_level_raw", "max_height"):
        assert key in warm.meta, f"{key} did not survive the cache"
        assert _plain(cold.meta[key]) == _plain(warm.meta[key]), \
            f"{key} differs after a round trip"


def test_labels_do_not_depend_on_the_cache():
    from atlas.labels import build_labels
    from atlas.naming import Namer

    cold = build_world(SPEC, use_cache=False)
    warm = build_world(SPEC, use_cache=True)
    a = build_labels(cold, Namer(SPEC.seed))
    b = build_labels(warm, Namer(SPEC.seed))

    assert [(l.kind, l.text, round(l.x), round(l.y)) for l in a] == \
           [(l.kind, l.text, round(l.x), round(l.y)) for l in b], \
           "the same seed lays out different labels warm and cold"
