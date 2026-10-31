"""Command line: one seed in, one sheet out; many seeds in, one book out."""

from __future__ import annotations

import argparse
import html
import sys
import time
from pathlib import Path

from .config import WorldSpec
from .world import build_world
from .naming import Namer, roman
from .compose import compose
from . import book, gazetteer, web

COLOPHON = [
    "Every plate in this book was computed, not drawn.  A single integer is "
    "expanded into a height field by fractal noise; water is routed downhill "
    "across that field; valleys are cut along the drainage; climate is derived "
    "from latitude and altitude; provinces are grown from watersheds; and the "
    "names are coined from a phonology invented for each province.",
    "The same integer always produces the same country.  There is no other copy "
    "of it anywhere, and no one has been there.",
    "Composed with Pillow and numpy.  Set in Garamond and Book Antiqua.",
]


def render_sheet(seed: int, width: int = 1400, height: int = 950, scale: int = 0,
                 title: str | None = None, plate: str | None = None,
                 mode: str = "physical", verbose: bool = True):
    spec = WorldSpec(seed=seed, width=width, height=height)
    t0 = time.time()
    world = build_world(spec)
    t1 = time.time()
    namer = Namer(seed)
    page, labels = compose(world, namer, scale=scale or None, title=title, plate=plate,
                           mode=mode)
    t2 = time.time()
    if verbose:
        print(f"  seed {seed:>5}  terrain {t1 - t0:5.1f}s  ink {t2 - t1:5.1f}s  "
              f"{page.size[0]}x{page.size[1]}")
    return world, namer, page, labels


def _title_for(seed: int, override: str | None = None) -> str:
    """The country's name, drawn once and carried everywhere.

    The cartouche, the contents, the gazetteer and the index must all print the
    same name.  Drawing it twice from the same generator gives two different
    names, because the generator has advanced in between -- so it is drawn here,
    once, and passed down.
    """
    if override:
        return override
    namer = Namer(seed)
    return namer.title(namer.cultures[0])


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="atlas",
        description="Generate fictional continents and draw them as antique sheets.",
    )
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--seeds", type=str, default=None,
                    help="a range, e.g. 1-8, or a comma list 3,7,11")
    ap.add_argument("--width", type=int, default=1400, help="terrain grid width")
    ap.add_argument("--height", type=int, default=950, help="terrain grid height")
    ap.add_argument("--scale", type=int, default=0,
                    help="output pixels per grid cell (0 = sized to the plate)")
    ap.add_argument("--title", type=str, default=None)
    ap.add_argument("--out", type=str, default=None, help="output PNG path")
    ap.add_argument("--outdir", type=str, default="out", help="directory for output")
    ap.add_argument("--book", type=str, default=None,
                    help="bind the plates into a book PDF at this path")
    ap.add_argument("--pdf", type=str, default=None, help="alias for --book")
    ap.add_argument("--gallery", action="store_true", help="write an index.html gallery")
    ap.add_argument("--viewer", action="store_true", help="write an interactive viewer.html")
    ap.add_argument("--political", action="store_true",
                    help="also draw a political plate for every world")
    args = ap.parse_args(argv)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    seeds = _parse_seeds(args.seeds) if args.seeds else [args.seed]

    sheets = []
    cards = []
    viewer_items = []
    n_plates = len(seeds) * (2 if args.political else 1)
    counter = [0]

    def next_plate():
        counter[0] += 1
        return f"PLATE {roman(counter[0])}" if n_plates > 1 else None

    for i, seed in enumerate(seeds):
        pnum = next_plate()
        name = _title_for(seed, args.title if len(seeds) == 1 else None)
        world, namer, page, labels = render_sheet(
            seed, args.width, args.height, args.scale, name, plate=pnum)

        path = Path(args.out) if (args.out and len(seeds) == 1) else outdir / f"atlas_{seed:04d}.png"
        page.save(path)
        print(f"  -> {path}")

        pol_page = None
        pol_num = None
        if args.political:
            pol_num = next_plate()
            _, _, pol_page, _ = render_sheet(
                seed, args.width, args.height, args.scale, name,
                plate=pol_num, mode="political")
            ppath = outdir / f"atlas_{seed:04d}_political.png"
            pol_page.save(ppath)
            print(f"  -> {ppath}")

        title, subtitle, paras = gazetteer.describe(world, labels, name)
        sheets.append(dict(plate=pnum, page=page, pol_page=pol_page, pol_plate=pol_num,
                           title=title, subtitle=subtitle, paras=paras, labels=labels))
        cards.append((path, title, subtitle, paras))
        viewer_items.append({"image": path.name, "title": title,
                             "subtitle": subtitle, "paras": paras,
                             "plate": pnum or ""})
        if pol_page is not None:
            pcards = (ppath, title + " — political", subtitle,
                      "The same country drawn for the reader who wants the shape of its "
                      "provinces rather than the shape of its hills.")
            cards.append(pcards)
            viewer_items.append({"image": ppath.name, "title": title,
                                 "subtitle": subtitle, "paras": [pcards[3]],
                                 "plate": pol_num or ""})

    if args.gallery:
        _write_gallery(outdir / "index.html", cards)
        print(f"  -> {outdir / 'index.html'}")

    if args.viewer:
        vp = outdir / "viewer.html"
        vp.write_text(web.viewer_html(viewer_items), encoding="utf-8")
        print(f"  -> {vp}")

    target = args.book or args.pdf
    if target:
        pages = _assemble_book(sheets, seeds)
        Path(target).parent.mkdir(parents=True, exist_ok=True)
        pages[0].save(target, "PDF", resolution=150.0, save_all=True,
                      append_images=pages[1:])
        print(f"  -> {target}  ({len(pages)} pages)")

    return 0


def _assemble_book(sheets, seeds):
    size = book.PAGE
    pages = []
    n_plates = sum(1 + (1 if s.get("pol_page") is not None else 0) for s in sheets)
    pages.append(book.title_page(size, "Terra Ficta", "countries that were never surveyed",
                                 roman(1800 + (seeds[0] % 99)), n_plates))
    pages.append(book.colophon_page(size, COLOPHON))

    contents = []
    index = {}
    folio = 3
    for sheet in sheets:
        folio += 1
        contents.append((sheet["plate"] or "", sheet["title"], folio))
        if sheet.get("pol_page") is not None:
            folio += 1
            contents.append((sheet["pol_plate"] or "", sheet["title"] + " — political", folio))
        folio += 1
        num = (sheet["plate"] or "").replace("PLATE ", "") or "I"
        for lb in sheet["labels"]:
            index.setdefault(lb.text.title(), num)
    pages.append(book.contents_page(size, contents))

    for sheet in sheets:
        pages.append(book.place_plate(sheet["page"]))
        if sheet.get("pol_page") is not None:
            pages.append(book.place_plate(sheet["pol_page"]))
        pages.append(gazetteer.text_page(sheet["title"], sheet["subtitle"], sheet["paras"],
                                         size=size,
                                         seed=sum(ord(c) for c in sheet["title"]) % 1000))

    entries = sorted(index.items(), key=lambda kv: kv[0].lower())
    pages.extend(book.index_pages(size, entries))
    return pages


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
