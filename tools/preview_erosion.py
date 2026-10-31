"""Erosion test bench: base / carved / drainage overlay / lakes."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from atlas.config import WorldSpec
from atlas.terrain import build_base
from atlas.shade import hillshade, ambient_occlusion
from atlas.palette import hypsometric, bathymetric
from atlas import hydro

W, H = 900, 600
SEED = 7


def tint(h, meters):
    hn = h / meters
    land = h > 0
    shade = hillshade(hn, z_factor=340.0)
    ao = ambient_occlusion(hn, z_factor=340.0, radius=4)
    sea = bathymetric(np.maximum(-h, 0.0))
    landc = hypsometric(np.maximum(h, 0.0))
    s = (0.40 + 0.78 * shade * (0.72 + 0.28 * ao))[..., None]
    return np.where(land[..., None], landc * s, sea)


def draw_streams(img, acc, rec, thr, colour=(30, 70, 110)):
    d = ImageDraw.Draw(img)
    hh, ww = acc.shape
    ys, xs = np.where(acc > thr)
    order = np.argsort(-acc[ys, xs])
    ys, xs = ys[order], xs[order]
    for y, x in zip(ys, xs):
        r = rec[y * ww + x]
        if r < 0:
            continue
        ry, rx = divmod(int(r), ww)
        width = 1 if acc[y, x] < thr * 6 else (2 if acc[y, x] < thr * 30 else 3)
        d.line([(x, y), (rx, ry)], fill=colour, width=width)
    return img


def main():
    spec = WorldSpec(seed=SEED, width=W, height=H)
    t0 = time.time()
    h0, meta = build_base(spec)
    print(f"base      {time.time() - t0:6.2f}s   max {h0.max():.0f} m")

    t0 = time.time()
    dr0 = hydro.drainage(h0)
    print(f"drainage  {time.time() - t0:6.2f}s")

    t0 = time.time()
    h1 = hydro.carve(h0, dr0["accumulation"], depth=260.0)
    print(f"carve     {time.time() - t0:6.2f}s   max {h1.max():.0f} m")

    dr1 = hydro.drainage(h1)
    print(f"lakes>5m  {int((dr1['lake_depth'] > 5).sum())} cells "
          f"({100 * (dr1['lake_depth'] > 5).mean():.2f}% of map)")

    img0 = Image.fromarray((np.clip(tint(h0, spec.meters), 0, 1) * 255).astype(np.uint8))
    img1 = Image.fromarray((np.clip(tint(h1, spec.meters), 0, 1) * 255).astype(np.uint8))

    thr = 700
    img2 = img1.copy()
    draw_streams(img2, dr1["accumulation"], dr1["receivers"], thr)

    arr3 = np.clip(tint(h1, spec.meters), 0, 1)
    lake = dr1["lake_depth"] > 2.0
    arr3[lake] = np.array([0.35, 0.55, 0.70])
    img3 = Image.fromarray((arr3 * 255).astype(np.uint8))

    font = ImageFont.truetype("C:/Windows/Fonts/GARA.TTF", 20)
    sheet = Image.new("RGB", (2 * W + 36, 2 * H + 60), (28, 28, 30))
    for i, (im, lab) in enumerate([(img0, "base"), (img1, "carved valleys"),
                                   (img2, f"streams (A>{thr})"), (img3, "lakes > 2 m")]):
        x = 12 + (i % 2) * (W + 12)
        y = 12 + (i // 2) * (H + 12)
        sheet.paste(im, (x, y))
        ImageDraw.Draw(sheet).text((x + 8, y + 6), lab, fill=(240, 230, 210), font=font)

    out = Path(__file__).resolve().parents[1] / "out" / "preview_erosion.png"
    out.parent.mkdir(exist_ok=True)
    sheet.save(out)
    print(out)


if __name__ == "__main__":
    main()
