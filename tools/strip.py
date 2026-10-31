"""A contact sheet: how one plate is made, from a height array to an atlas page."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PIL import Image, ImageDraw

from atlas import layout, shade, decor
from atlas.config import WorldSpec
from atlas.world import build_world

SEED = 3
W, H = 1400, 950
PANEL_TOP = (953, 640)
PANEL_BOT = (1460, 990)
GAP = 60
M = 110
PAPER = (236, 231, 217)
INK = (44, 38, 32)
INK_SOFT = (110, 100, 88)
SEA = (198, 208, 210)
RIVER = (70, 100, 118)


def fit(img, size):
    """Scale to fit inside the panel, centred on paper -- never crop a finished plate."""
    tw, th = size
    w, h = img.size
    s = min(tw / w, th / h)
    img = img.resize((int(round(w * s)), int(round(h * s))), Image.LANCZOS)
    canvas = Image.new("RGB", (tw, th), PAPER)
    canvas.paste(img, ((tw - img.size[0]) // 2, (th - img.size[1]) // 2))
    return canvas


def frame_crop(a, fr, size):
    """Slice a terrain array to the chosen frame and upsample to the panel."""
    sub = a[fr.y0:fr.y0 + fr.h, fr.x0:fr.x0 + fr.w]
    img = Image.fromarray(sub.astype(np.float32), mode="F")
    return np.asarray(img.resize(size, Image.BICUBIC))


def panel_land(world, fr, size):
    land = frame_crop(world.land.astype(np.float32), fr, size)
    a = np.clip(land, 0, 1)[..., None]
    sea = np.array(SEA, dtype=np.float32)
    ground = np.array((232, 226, 210), dtype=np.float32)
    return (sea * (1 - a) + ground * a).astype(np.uint8)


def panel_drainage(world, fr, size):
    h = frame_crop(world.height, fr, size)
    acc = frame_crop(np.log1p(np.maximum(world.accumulation, 0)).astype(np.float32), fr, size)
    land = frame_crop(world.land.astype(np.float32), fr, size)

    light = shade.hillshade(h, z_factor=260.0)
    light = np.clip(light, 0, 1)
    base = 0.74 + 0.28 * light
    rgb = np.repeat(base[..., None], 3, axis=2)

    lo = np.log1p(300.0)
    t = np.clip((acc - lo) / 3.2, 0, 1) ** 1.5
    t = (t * np.clip(land, 0, 1))[..., None]
    rgb = rgb * (1 - t) + (np.array(RIVER, dtype=np.float32) / 255.0) * t

    sea = np.array(SEA, dtype=np.float32) / 255.0
    a = np.clip(land, 0, 1)[..., None]
    rgb = sea * (1 - a) + rgb * a
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


def panel_relief(world, fr, size):
    h = frame_crop(world.height, fr, size)
    land = frame_crop(world.land.astype(np.float32), fr, size)
    light = np.clip(shade.hillshade(h, z_factor=260.0), 0, 1)
    v = 0.56 + 0.56 * light
    sea = np.array(SEA, dtype=np.float32) / 255.0
    a = np.clip(land, 0, 1)[..., None]
    rgb = sea * (1 - a) + v[..., None] * a
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


def main():
    world = build_world(WorldSpec(seed=SEED, width=W, height=H))
    fr = layout.choose_frame(world.land)

    sheet_w = M * 2 + PANEL_TOP[0] * 3 + GAP * 2
    sheet_h = M + PANEL_TOP[1] + 84 + PANEL_BOT[1] + 76 + M
    sheet = Image.new("RGB", (sheet_w, sheet_h), PAPER)
    draw = ImageDraw.Draw(sheet)

    f_cap = decor.font("antiqua-bold", 30)
    f_sub = decor.font("antiqua-italic", 25)

    def put(img, x, y, cap, sub):
        sheet.paste(Image.fromarray(img) if isinstance(img, np.ndarray) else img, (x, y))
        decor.tracked_text(draw, (x, y + img_height(img) + 22), cap,
                           f_cap, INK, 0.16, anchor_center=False)
        decor.tracked_text(draw, (x, y + img_height(img) + 62), sub,
                           f_sub, INK_SOFT, 0.0, anchor_center=False)

    top = [
        (panel_land(world, fr, PANEL_TOP), "LAND", "where the continent is"),
        (panel_drainage(world, fr, PANEL_TOP), "DRAINAGE", "where the water goes"),
        (panel_relief(world, fr, PANEL_TOP), "RELIEF", "how high it stands"),
    ]
    x = M
    for img, cap, sub in top:
        put(img, x, M, cap, sub)
        x += PANEL_TOP[0] + GAP

    y2 = M + PANEL_TOP[1] + 84
    phys = fit(Image.open("out/gallery/atlas_0003.png"), PANEL_BOT)
    pol = fit(Image.open("out/gallery/atlas_0003_political.png"), PANEL_BOT)
    put(phys, M, y2, "PLATE", "the engraved sheet")
    put(pol, M + PANEL_BOT[0] + GAP, y2, "POLITICAL", "the same country, coloured by province")

    sheet.save("out/pipeline.png")
    print("-> out/pipeline.png", sheet.size)


def img_height(o):
    return o.shape[0] if isinstance(o, np.ndarray) else o.size[1]


if __name__ == "__main__":
    main()
