"""Command line: one seed in, one sheet out."""

from __future__ import annotations

import argparse
import html
import sys
import time
from pathlib import Path

from .config import WorldSpec
from .world import build_world
from .naming import Namer
from .compose import compose
from . import gazetteer


def render_sheet(seed: int, width: int = 1400, height: int = 950, scale: int = 2,
                 title: str | None = None, verbose: bool = True):
    spec = WorldSpec(seed=seed, width=width, height=height)
    t0 = time.time()
    world = build_world(spec)
    t1 = time.time()
    namer = Namer(seed)
    page, labels = compose(world, namer, scale=scale, title=title)
    t2 = time.time()
    if verbose:
        print(f"  seed {seed:>5}  terrain {t1 - t0:5.1f}s  ink {t2 - t1:5.1f}s  "
              f"{page.size[0]}x{page.size[1]}")
    return world, namer, page, labels


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="atlas",
        description="Generate a fictional continent and draw it as an antique sheet.",
    )
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--seeds", type=str, default=None,
                    help="a range, e.g. 1-8, or a comma list 3,7,11")
    ap.add_argument("--width", type=int, default=1400, help="terrain grid width")
    ap.add_argument("--height", type=int, default=950, help="terrain grid height")
    ap.add_argument("--scale", type=int, default=2, help="output pixels per grid cell")
    ap.add_argument("--title", type=str, default=None)
    ap.add_argument("--out", type=str, default=None, help="output PNG path")
    ap.add_argument("--outdir", type=str, default="out", help="directory for output")
    ap.add_argument("--pdf", type=str, default=None, help="write a multi-page PDF")
    ap.add_argument("--gallery", action="store_true", help="write an index.html gallery")
    args = ap.parse_args(argv)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    seeds = _parse_seeds(args.seeds) if args.seeds else [args.seed]

    pages = []
    cards = []
    for seed in seeds:
        world, namer, page, labels = render_sheet(
            seed, args.width, args.height, args.scale,
            args.title if len(seeds) == 1 else None)

        if args.out and len(seeds) == 1:
            path = Path(args.out)
        else:
            path = outdir / f"atlas_{seed:04d}.png"
        page.save(path)
        print(f"  -> {path}")

        title, subtitle, paras = gazetteer.describe(world, labels, namer.title(namer.cultures[0]))
        pages.append(page)
        cards.append((path, title, subtitle, paras))

        if args.pdf:
            pages.insert(len(pages) - 1,
                         gazetteer.text_page(title, subtitle, paras,
                                             size=(page.size[0], page.size[1]), seed=seed))

    if args.gallery:
        _write_gallery(outdir / "index.html", cards)
        print(f"  -> {outdir / 'index.html'}")

    if args.pdf and pages:
        pdf_path = Path(args.pdf)
        pages[0].save(pdf_path, "PDF", resolution=150.0, save_all=True,
                      append_images=pages[1:])
        print(f"  -> {pdf_path}  ({len(pages)} pages)")

    return 0


def _parse_seeds(text: str):
    seeds = []
    for part in text.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            seeds.extend(range(int(lo), int(hi) + 1))
        elif part:
            seeds.append(int(part))
    return seeds


def _write_gallery(path: Path, cards):
    rows = []
    for img_path, title, subtitle, paras in cards:
        rows.append(f"""
    <figure>
      <a href="{html.escape(img_path.name)}"><img src="{html.escape(img_path.name)}" alt="{html.escape(title)}"></a>
      <figcaption>
        <h2>{html.escape(title)}</h2>
        <p class="sub">{html.escape(subtitle)}</p>
        <p>{html.escape(paras[0])}</p>
      </figcaption>
    </figure>""")

    doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Atlas</title>
<style>
  :root {{ --paper:#efe7d3; --ink:#2e2620; --soft:#5b4e42; }}
  body {{ margin:0; background:#1d1a17; color:var(--ink);
         font-family: Georgia, 'Times New Roman', serif; }}
  header {{ padding:64px 24px 32px; text-align:center; color:#e8dfc9; }}
  header h1 {{ font-size:clamp(28px,4vw,52px); letter-spacing:.18em;
               text-transform:uppercase; margin:0 0 12px; font-weight:400; }}
  header p {{ color:#9d9184; font-style:italic; margin:0; }}
  main {{ max-width:1100px; margin:0 auto; padding:24px 24px 96px; }}
  figure {{ margin:0 0 72px; background:var(--paper); padding:14px 14px 4px;
            box-shadow:0 18px 50px rgba(0,0,0,.45); }}
  figure img {{ width:100%; display:block; }}
  figcaption {{ padding:22px 20px 26px; }}
  figcaption h2 {{ margin:0 0 4px; font-weight:400; letter-spacing:.12em;
                   text-transform:uppercase; font-size:20px; }}
  .sub {{ margin:0 0 14px; color:var(--soft); font-style:italic; font-size:15px; }}
  figcaption p {{ line-height:1.65; margin:0; font-size:16px; }}
</style>
</head>
<body>
  <header>
    <h1>Atlas</h1>
    <p>countries that were never surveyed</p>
  </header>
  <main>{''.join(rows)}
  </main>
</body>
</html>
"""
    path.write_text(doc, encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
