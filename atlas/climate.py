"""Climate and biomes.

Two fields decide where a forest is a forest and a desert is a desert:

* **Temperature** -- mostly latitude, minus a lapse rate for altitude.  A
  mountain at 40 degrees north has the climate of the arctic at its summit.
* **Moisture** -- distance from the sea, plus the orographic effect.  Air that
  has crossed a range has dropped its water on the windward side and arrives
  dry on the lee; that asymmetry is what puts a desert behind a mountain.

The biome is then a lookup on the (temperature, precipitation) plane -- the
Whittaker diagram, which is about as old as the maps this engine imitates.
"""

from __future__ import annotations

import numpy as np

from .config import WorldSpec

# biome id -> (name, colour)
BIOMES = {
    0: ("Ice", (0.855, 0.875, 0.885)),
    1: ("Tundra", (0.760, 0.775, 0.715)),
    2: ("Boreal forest", (0.505, 0.585, 0.455)),
    3: ("Temperate forest", (0.555, 0.665, 0.420)),
    4: ("Grassland", (0.760, 0.755, 0.470)),
    5: ("Steppe", (0.805, 0.755, 0.505)),
    6: ("Desert", (0.880, 0.805, 0.600)),
    7: ("Savanna", (0.775, 0.715, 0.405)),
    8: ("Tropical forest", (0.425, 0.580, 0.375)),
    9: ("Alpine", (0.720, 0.705, 0.680)),
}


def latitude_grid(spec: WorldSpec):
    """Latitude in degrees for every row, north at the top."""
    h = spec.height
    ys = np.arange(h, dtype=np.float64)
    lat = spec.lat_top - (ys / max(h - 1, 1)) * (spec.lat_top - spec.lat_bottom)
    return np.repeat(lat[:, None], spec.width, axis=1)


def temperature(lat, elev_m):
    """Degrees Celsius at the surface."""
    base = 31.0 - 0.62 * np.abs(lat)
    return base - elev_m * 0.0065


def moisture(spec: WorldSpec, height, relief, wind_from=250.0):
    """Precipitation in millimetres, from distance to sea and orography."""
    gy, gx = np.gradient(height.astype(np.float64))

    # Where the wind goes, in image coordinates (x east, y south).
    az = np.radians(wind_from + 180.0)
    wx, wy = np.sin(az), -np.cos(az)

    # Upslope lifts air and wrings it out; downslope is the rain shadow.
    along = gx * wx + gy * wy
    scale = np.percentile(np.abs(along[height > 0]), 92) if (height > 0).any() else 1.0
    along = np.clip(along / max(scale, 1e-6), -1.0, 1.0)
    lift = np.clip(along, 0.0, 1.0)
    shadow = np.clip(-along, 0.0, 1.0)

    # The continental potential is positive inland and falls away offshore, so
    # it doubles as a proxy for distance from the coast: 0 at the shore,
    # growing toward the interior.
    inland_km = np.clip(relief, 0.0, None) * 4200.0
    decay = np.exp(-inland_km / 900.0)

    base = 1300.0
    p = base * (0.24 + 0.76 * decay)
    p = p * (1.0 + 1.9 * lift) * (1.0 - 0.66 * shadow)
    return np.clip(p, 12.0, 4200.0)


def classify(temp, precip, elev):
    """Whittaker lookup, with altitude overriding everything."""
    t = temp
    p = precip
    biome = np.full(t.shape, 4, dtype=np.uint8)          # grassland default

    biome[p < 180] = 6                                    # desert
    biome[(p >= 180) & (p < 380) & (t > 4)] = 5           # steppe
    biome[(p >= 380) & (p < 700) & (t > 12)] = 7          # savanna
    biome[(p >= 380) & (p < 700) & (t <= 12)] = 4         # grassland
    biome[(p >= 700) & (t > 20)] = 8                      # tropical forest
    biome[(p >= 700) & (t > 6) & (t <= 20)] = 3           # temperate forest
    biome[(p >= 450) & (t > -2) & (t <= 6)] = 2           # boreal forest
    biome[(t <= -2) & (t > -9)] = 1                       # tundra
    biome[t <= -9] = 0                                    # ice
    biome[(elev > 2400) & (t > -9)] = 9                   # alpine
    return biome


def derive(world):
    """Compute the climate stack for a world and attach it."""
    spec = world.spec
    h = world.height
    relief = world.meta.get("relief")
    if relief is None:
        # Rebuild a stand-in from the height field when the cache has no meta.
        relief = np.where(h > 0, np.clip(h / 4600.0, 0, 1) * 0.4, np.clip(h / 4600.0, -1, 0))

    lat = latitude_grid(spec)
    temp = temperature(lat, np.maximum(h, 0.0))
    precip = moisture(spec, h, relief)
    biome = classify(temp, precip, np.maximum(h, 0.0))
    biome = np.where(h > 0, biome, 255)
    return {"latitude": lat, "temperature": temp, "precipitation": precip, "biome": biome}
