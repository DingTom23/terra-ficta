"""World specification: every dial that shapes a world."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
import json


@dataclass
class WorldSpec:
    # --- identity -------------------------------------------------------
    seed: int = 1
    name: str = ""

    # --- canvas ---------------------------------------------------------
    width: int = 1500          # terrain grid width, in cells
    height: int = 1000         # terrain grid height, in cells
    supersample: int = 2       # render multiplier over the terrain grid

    # --- landmass -------------------------------------------------------
    land_fraction: float = 0.34
    blobs: int = 3             # number of tectonic nuclei
    island_blobs: int = 4      # small detached islands
    coast_noise: float = 0.55  # how fractal the coastline is
    coast_freq: float = 2.4

    # --- relief ---------------------------------------------------------
    ridge_count: int = 3       # orogenic belts
    ridge_width: float = 0.10  # belt half-width, in map widths
    mountain_amp: float = 0.85
    mountain_freq: float = 5.0
    relief_bias: float = 0.55  # how much of the interior is uplifted
    meters: float = 4600.0     # elevation that maps to normalised height 1.0

    # --- hydrology ------------------------------------------------------
    erosion_iterations: int = 22
    erosion_rate: float = 0.035
    talus: float = 0.006       # thermal erosion threshold per cell
    river_min_cells: int = 320  # accumulation threshold for a drawn stream
    carve_depth: float = 420.0  # metres of valley incision
    detail: float = 1.0        # multiplier on the fine relief added last

    # --- climate --------------------------------------------------------
    lat_top: float = 58.0      # degrees north at the top edge
    lat_bottom: float = 12.0   # degrees north at the bottom edge
    lon_left: float = 10.0     # degrees east at the left edge
    wind_azimuth: float = 250.0  # prevailing wind, degrees clockwise from N

    # --- presentation ---------------------------------------------------
    projection: str = "equirectangular"
    style: str = "antique"
    title: str = ""
    provinces: int = 5         # number of provinces (drawn as watershed borders)

    extra: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, ensure_ascii=False)

    @property
    def aspect(self) -> float:
        return self.width / self.height
