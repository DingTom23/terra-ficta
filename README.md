# Atlas

**A deterministic procedural cartography engine.** Give it a seed; it gives you a
continent — terrain, drainage, climate, place names, and a finished sheet drawn in
the manner of a nineteenth-century relief atlas.

Nothing in the output is hand-drawn. Nothing is random in the sense of being
irreproducible: the same seed always produces the same world, down to the last
river bend and the last letter of the last town's name.

```bash
python -m atlas --seed 7
```

```
  seed     7  terrain  21.2s  ink  21.0s  3180x2280
  -> out/atlas_0007.png
```

---

## What it produces

| | |
|---|---|
| **Terrain** | fractal landmass, orogenic belts, carved valley networks, lakes |
| **Water** | steepest-descent drainage, flow accumulation, rivers drawn tapered downstream |
| **Climate** | latitude and altitude temperature, orographic precipitation with rain shadows, Whittaker biomes, tinted into the lowlands and keyed in the legend |
| **Provinces** | grown from watersheds, so borders follow the divides; drawn as a dotted rule |
| **Roads** | a minimum spanning tree decides which towns connect, A\* over a terrain cost field decides where each road runs |
| **Names** | a small invented phonology per region, so names within a province sound related |
| **Framing** | the plate is cut to its subject: the largest landmass plus any island that mostly falls inside it, then a furniture solver finds open water for the cartouche, compass, keys and scale bar |
| **The physical sheet** | hypsometric tints, hillshade, jittered slope hachures drawn down the fall line, contours, coastal hachures, soundings, rhumb lines, graticule, graduated chequer border, compass, scale, two keys, cartouche, plate number |
| **The political sheet** | the same country flat-washed by province, boundaries drawn solid, a numbered key, and the relief dropped to a whisper |
| **The book** | title page, colophon, contents with dot leaders, numbered plates, gazetteer pages, index of names |

Output is a PNG at roughly 2700×2000, a bound PDF of twenty-two pages, and a
static viewer you can pan and zoom.

---

## Quick start

Requires Python 3.10+, `numpy` and `Pillow`. Nothing else.

```bash
# one sheet
python -m atlas --seed 7

# a bound book of six countries, each with a physical and a political plate
python -m atlas --seeds 1-6 --political --gallery --viewer --book out/atlas.pdf

# smaller paper, faster iteration
python -m atlas --seed 7 --width 900 --height 600 --scale 2
```

| flag | meaning |
|---|---|
| `--seed` | which world |
| `--seeds` | a range (`1-8`) or a list (`3,7,11`) |
| `--width` / `--height` | terrain grid, in cells (default 1400×950) |
| `--scale` | output pixels per grid cell (default 2) |
| `--title` | override the invented country name |
| `--book` / `--pdf` | bind the results into one document |
| `--gallery` | write `index.html` beside the images |
| `--viewer` | write `viewer.html`: pan, zoom, read the gazetteer |

---

## How a world is made

The pipeline is ordered, and the order matters more than any single step.

```
landmass  →  drainage  →  carved valleys  →  fine detail  →  drainage again
```

**1. Landmass.** A handful of elliptical tectonic nuclei are unioned into a
continental potential, then displaced by domain-warped noise. Sea level is chosen
by quantile, so the land fraction is exactly what was asked for. Two things are
kept strictly apart here: the potential decides *where the land is*, and a
separate relief field decides *how high it is*. Conflating them yields the classic
procedural-geography tell — a continent that is a smooth dome.

**2. Drainage.** Closed basins are filled with a priority flood over a bucket
queue, then water is routed by steepest descent and accumulated by Kahn's
algorithm in vectorised layers.

**3. Valleys.** The drainage network is blurred at three scales and subtracted
from the terrain. Big rivers cut broad troughs, creeks cut notches. This replaced
an explicit stream-power erosion model, which is physically honest but, once
discretised on a raster with a stability clamp, converges to a staircase.

**4. Detail.** High-frequency ridges are added *after* routing. Put in earlier,
they shatter the network into thousands of competing puddles.

**5. Names.** Each region draws from its own phonology — a distinct onset/vowel/
coda inventory — and its own set of toponymic suffixes. Labels are placed
greedily, measured with the font that will actually draw them.

**6. The sheet.** Layers are composited in ink order, then laid onto toned paper.

See [DESIGN.md](DESIGN.md) for the reasoning behind each choice, including the
ones that turned out to be wrong.

---

## Repository

```
atlas/
  config.py      WorldSpec — every dial that shapes a world
  noise.py       gradient noise, fBm, ridged multifractal, domain warping
  terrain.py     landmass, orogenic belts, fine detail
  hydro.py       depression filling, flow routing, accumulation, valley carving
  climate.py     temperature, orographic moisture, Whittaker biomes
  regions.py     provinces grown from watersheds
  roads.py       least-cost roads: spanning tree plus A*
  world.py       the pipeline, plus an on-disk cache
  shade.py       hillshade, curvature, ambient occlusion
  palette.py     inks and tints
  layout.py      framing: which window of the world, and where the furniture fits
  render.py      the drawing itself, physical and political
  decor.py       frame, graticule, compass, scale bar, cartouche, legend
  naming.py      invented phonologies and toponymy
  labels.py      feature extraction and label placement
  compose.py     the finished sheet
  gazetteer.py   prose
  book.py        front matter, contents, index
  web.py         the static viewer
  cli.py         command line
tools/           development benches used while building this
out/             results
cache/           generated worlds, keyed by a hash of the spec
```

The cache is safe to delete. It exists so that re-inking a world does not mean
recomputing its rivers.

---

## Determinism

Every random draw comes from a `numpy` generator seeded from the world seed, and
every noise field is a pure function of its coordinates. There is no global RNG
state and no dependence on iteration order in a hash map. Given the same seed and
the same version of this code, the output is byte-identical.

That property is the point. A map that cannot be reproduced is a drawing; a map
that can is a place.

---

## Status

Working: terrain, hydrology, climate and biomes, provinces, roads, relief
rendering, framing, furniture placement, typography, naming, physical and
political composition, book binding, gallery, viewer.

Still wrong, and known to be wrong:

- The far north has no ice cap. Latitude cools the temperature but nothing
  actually accumulates as ice, so a polar sea is drawn as open water.
- The index gives one folio per name. A river that runs through three plates
  is listed once.
- Islands are lobed by construction, which is honest but monotonous: the
  harmonic sum is the same shape at every scale.
- Towns have exactly two ranks: a seat of government gets a star, everything
  else a dot. A real plate grades market towns, post towns, villages and
  ruins, and this one has no vocabulary for any of it.
- Region names are now compulsory, but on about one seed in six a province
  still finds nowhere to write: it is then tinted on the plate and missing from
  the key, which is the one inconsistency a reader will actually notice.
- The gazetteer is generated from the same statistics that drew the map, so it
  can describe a country accurately and still tell you nothing about it.
