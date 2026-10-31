"""Compose a finished sheet and write it out."""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from atlas.config import WorldSpec
from atlas.world import build_world
from atlas.naming import Namer
from atlas.compose import compose

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 7
W = int(sys.argv[2]) if len(sys.argv) > 2 else 760
H = int(sys.argv[3]) if len(sys.argv) > 3 else 500


def main():
    spec = WorldSpec(seed=SEED, width=W, height=H)
    t0 = time.time()
    world = build_world(spec)
    print(f"world   {time.time() - t0:6.2f}s")

    t0 = time.time()
    page, labels, view = compose(world, Namer(SEED), scale=2)
    print(f"compose {time.time() - t0:6.2f}s  {page.size}  labels {len(labels)}")

    out = Path(__file__).resolve().parents[1] / "out" / f"sheet_{SEED}.png"
    out.parent.mkdir(exist_ok=True)
    page.save(out)
    print(out)


if __name__ == "__main__":
    main()
