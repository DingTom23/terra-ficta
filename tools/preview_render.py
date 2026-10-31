"""Render one world end to end and write a PNG."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from atlas.config import WorldSpec
from atlas.world import build_world
from atlas import render

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 7
SCALE = int(sys.argv[2]) if len(sys.argv) > 2 else 1


def main():
    spec = WorldSpec(seed=SEED, width=900, height=600)
    t0 = time.time()
    world = build_world(spec)
    print(f"world   {time.time() - t0:6.2f}s  max {world.height.max():.0f} m")

    t0 = time.time()
    img = render.render(world, scale=SCALE)
    print(f"render  {time.time() - t0:6.2f}s  {img.shape}")

    out = Path(__file__).resolve().parents[1] / "out" / f"render_{SEED}.png"
    out.parent.mkdir(exist_ok=True)
    Image.fromarray((img * 255).astype(np.uint8)).save(out)
    print(out)


if __name__ == "__main__":
    main()
