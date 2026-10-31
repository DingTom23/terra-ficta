"""Quick contact sheet of raw landmass shapes -- for eyeballing composition."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from atlas.config import WorldSpec
from atlas.terrain import build_base
from atlas.shade import hillshade, ambient_occlusion
from atlas.palette import hypsometric, bathymetric

W, H = 760, 500
SEEDS = [1, 2, 3, 4, 5, 6]


def tint(h, meters):
    hn = h / meters
    land = h > 0
    shade = hillshade(hn, z_factor=340.0)
    ao = ambient_occlusion(hn, z_factor=340.0, radius=5)

    sea = bathymetric(np.maximum(-h, 0.0))
    landc = hypsometric(np.maximum(h, 0.0))
    s = (0.40 + 0.78 * shade * (0.72 + 0.28 * ao))[..., None]
    out = np.where(land[..., None], landc * s, sea)
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


def main():
    font = ImageFont.truetype("C:/Windows/Fonts/GARA.TTF", 22)
    cols, rows = 3, 2
    sheet = Image.new("RGB", (cols * W + (cols + 1) * 12, rows * H + (rows + 1) * 12), (30, 30, 32))
    for i, seed in enumerate(SEEDS):
        spec = WorldSpec(seed=seed, width=W, height=H)
        h, meta = build_base(spec)
        img = Image.fromarray(tint(h, spec.meters))
        d = ImageDraw.Draw(img)
        d.text((10, 6), f"seed {seed}  land {meta['land_fraction']:.2f}  max {meta['max_height']:.0f} m",
               fill=(20, 20, 20), font=font)
        sheet.paste(img, (12 + (i % cols) * (W + 12), 12 + (i // cols) * (H + 12)))
    out = Path(__file__).resolve().parents[1] / "out" / "preview_landmass.png"
    out.parent.mkdir(exist_ok=True)
    sheet.save(out)
    print(out)


if __name__ == "__main__":
    main()
