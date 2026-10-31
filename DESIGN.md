# Design notes

What follows is the reasoning behind the engine, and — more usefully — the record
of what went wrong on the way. Most of the interesting decisions here were forced
by a bug that took a rendered image to notice.

---

## 1. Keep "where" and "how high" apart

The first working terrain was a smooth dome with noise on top. It looked like a
balloon lit from the north-west, and the reason is structural: a single field
`h = 1 - distance_to_nucleus + noise` was doing two jobs. Because the distance
field is a cone, the *shape* of the cone dominated the shading everywhere, and the
mountains were a texture painted on a balloon.

The fix is to split the field:

- a **continental potential** — nucleus distance plus warped noise, used only to
  decide which cells are land;
- a **relief field** — hills and belts, used only to decide elevation.

The potential is thresholded at a quantile, so the land fraction is exact. The
relief never sees the potential except as a mask.

## 2. Depression filling: order, not iteration

The obvious relaxation for the fill surface is

```
W[i] = max(h[i], min over neighbours j of (W[j] + eps))
```

It is correct in the limit and unusable in practice. Convergence is one `eps` per
pass, so a basin 300 m deep needs fifteen thousand passes; worse, a flat basin
floor has no lower neighbour, so every cell in it rises by `eps` forever. Measured:
at 200 passes, 2 322 cells raised; at 1 500 passes, 7 710 raised and *more* cells
had no downhill neighbour than at the start. It diverges.

The spill level of a basin is known the moment its rim is reached. That means the
right algorithm is ordered, not iterative: a priority flood. Implemented with a
bucket queue over quantised heights, it is O(N), needs no heap, and takes 1.8 s
for 540 000 cells:

```
fill 1.8s  raised=14791  norecv=0  maxacc=36652
```

Zero cells without a receiver, and the largest drainage basin went from 1 822
cells to 36 652 — from a rash of puddles to a river system.

## 3. Stream power terraces; blurred carving does not

`E = K · A^m · S^n` with a stability clamp `E ≤ 0.45 · drop` is the textbook
incision law. Discretised, it converges to a staircase: each cell descends toward
its receiver's level but never below, and the whole surface becomes mesas.

Replacing it with multi-scale carving — blur the channel-strength field at three
radii and subtract — gives V-shaped valleys whose width and depth follow the size
of the river. It is not a physical model. It is a model of what a physical model
looks like, which is what a map needs.

## 4. Order of operations is the whole trick

Routing water before the valleys exist gives a network that ignores the
topography. Adding fine detail before routing shatters the network, because
high-frequency noise creates a closed basin every few hundred metres.

The pipeline therefore runs: landmass → route → carve → add detail → route again.

## 5. A bug that inks the ocean

`alpha = clip(1 - |land_mask - 0.5| / width)` draws a coastline. With `width = 1.1`
the expression is still 0.55 at `land_mask = 0`, which means **the entire sea gets
painted with 54% black ink**. Every surface in the image goes grey, and the cause
is invisible unless you sample a pixel in open water and compare it with the value
the palette says it should have had.

The lesson is not "use a smaller width". It is: when a whole image looks muddy,
find the layer that is darkening *everything*, not the layer that looks wrong.

## 6. Padding must fatten, not translate

The label collision test was:

```python
ax0, ax1 = ax0 * 1.12, ax1 * 1.12
```

Multiplying absolute coordinates does not grow a box — it moves it. Two labels
that clearly overlapped in the rendered sheet were reported as disjoint, and every
region name landed on top of the next. The fix is to move the *edges*:

```python
ax0 -= pad; ay0 -= pad; ax1 += pad; ay1 += pad
```

A related trap: measuring label widths with a constant (`len(text) * size * 0.62`)
instead of with the font that will draw them. Sizing type against the sheet width
rather than the terrain grid is what makes a 3000-px plate and a 1500-px preview
look like the same map.

## 7. Colour scaling appears twice

`_rgb()` multiplied a colour triple by 255. Several callers passed a triple that
was already 0–255, so cartouche, legend and scale bar rendered as pure white
rectangles. The helper now detects which convention it was given.

## 8. A sign that turned a continent into a rainforest

Precipitation decays with distance from the sea, and the continental potential
doubles as that distance — it is zero at the shore and grows inland. The first
version wrote `-relief`, which is the convention for the *ocean* side of the same
field. Every inland cell therefore read as zero kilometres from water: the
continent came out at 672–4200 mm, and 34% of it classified as forest with not a
single desert.

The tell was in the biome census, not the picture. A rendered map with the wrong
climate still looks like a map; a histogram with no desert in a continent that
size does not.

## 9. A province needs a boundary that means something

The first version placed region names by k-means on land cells. That gives
labels, but no borders — and a border drawn as the outline of a k-means cluster
is a border with no reason behind it.

The cheapest meaningful boundary was already in the data: the drainage divide.
Every cell's terminal river mouth is found by pointer-jumping down the D8
receiver graph (`p = p[p]`, twenty-four times, which resolves any forest shallower
than sixteen million cells). River mouths are then clustered geographically,
weighted by how much land they drain, and each basin joins the province of its
mouth. Borders fall along ridges, which is where people have always put them.

One consequence had to be suppressed: a province whose mouth cluster hugs the
coast produces a border that runs along the shore, and two lines a few pixels
apart read as a smudged coastline rather than as a border. Boundaries within
three cells of water are dropped.

## 10. Roads are not lines between towns

A straight line between two towns is a diagram. A road is the cheapest way over
the ground, which means it follows valleys and refuses slopes. So the network is
built in two steps: Prim's algorithm on Euclidean distance decides *which* towns
connect, and A\* over a terrain cost field (`1 + (elevation/450)^1.4 + slope/35`)
decides *where* the road goes.

The cost field is computed at a third of the terrain resolution — 42 000 cells
instead of 1.4 million — which makes seven A\* searches take ninety milliseconds
instead of minutes, and the resulting path is smoothed so it does not stair-step.

The first attempt silently produced two roads out of seven. A town at a headland
lands, once downsampled by three, in a cell that is mostly water; the cell was
marked impassable and A\* returned `None`. The fix is to snap each endpoint to
the nearest passable cell — and to make coastal cells passable at 35% land
rather than 50%.

## 11. Hachures, and why the grid had to go

Hillshade alone gives a modern relief map wearing antique colours. What actually
reads as a copper plate is hachures: short strokes drawn down the fall line,
longer and darker where the ground is steep, absent on the flats.

Two things had to be right. First, the strokes are real segments, not a texture
lookup — a raster trick always leaves moiré, and moiré on a mountain reads as a
printing fault. Second, the sampling lattice is jittered per row and per stroke.
An even grid of parallel ticks reads as woven cloth; engravers never spaced their
hachures regularly, and the eye notices that long before it can say why.

The third thing was a bug: a stroke runs several pixels past its sample point, so
sampling right up to the shore left a fringe of ticks standing in the water
around every island. The sampling mask is now eroded a few pixels inland and the
finished stroke layer is multiplied by the land mask.

## 12. The sheet is cut to the country, not the country to the sheet

The first composition had a fixed frame: a continent was generated into a
landscape grid, and whatever the noise produced had to live inside it. A country
that grew northwards had its mountains cropped; a country that grew westwards left
an empty sea where the cartouche was supposed to go.

The frame is now chosen after the land exists. `main_mass()` labels connected
components on a quarter-resolution grid, keeps the largest, and pulls in any other
component that is at least 55 per cent inside the frame so far — an island just off
the coast belongs on the sheet, an islet half a world away does not, and must not
drag the frame out to the grid edge. The frame is expanded by 16 per cent and the
plate is rendered *from the crop*.

Cropping is not slicing. `World.cropped(frame)` rebuilds everything whose meaning
depends on the grid: the D8 receivers are flat indices into the full array, so they
are recomputed for the crop; the latitude and longitude of the corners are remapped
from the crop's position in the parent; climate, provinces and lake depths are
sliced. Get this wrong and rivers reach the sea in the wrong place — the kind of
bug that looks like art until you check.

The consequence is that the sheet has no fixed aspect. Plates in the current
edition run from 1.23 to 1.50: a tall country gets a tall plate, as it would in a
portfolio of surveys.

## 13. Furniture is a constraint problem, not a layout

A cartouche placed at a constant offset from the corner is a cartouche that lands
on a mountain half the time. Every candidate position for every piece — cartouche,
compass, elevation key, biome key, scale bar — is now scored on four terms:

```
1.35 * openness      distance from land, on a coarse distance field
0.45 * edge          preference for the margins
gain * bias          the conventional position for that piece
0.25 * separation    distance from the pieces already placed
```

Placement is greedy, largest piece first. A piece that cannot be placed at full
size shrinks to 86 per cent and tries again, down to a floor; the biome key gets a
second pass stacked under the elevation key before it falls back to its own bias.

Two bugs came out of this. A slot with a `min_open` requirement was simply dropped
when nothing qualified, so the scale bar vanished from a third of the plates —
losing a piece of furniture is worse than printing it near a coast, so the solver
keeps a fallback candidate and returns it when the requirement cannot be met. And
the openness field is computed on a coarse grid, so the score has to be taken over
the box, not at its centre, or a small piece lands on a cell that is half water.

## 14. You cannot noise an ellipse into an island

Small islands came out as eggs. The cause is wavelength: the coast noise is
comparable in scale to the whole map, so across a forty-cell island it is
essentially constant — and a constant offset applied to an ellipse is still an
ellipse. A higher octave did not help; the next one was still seventy cells long.

Two attempts at domain warping failed for the same reason, and the second failed
in a way worth recording: displacement was computed as `amp * strength * noise`,
and `strength = 0.013` on a domain of 0 to 1.47 is about six cells at plate
resolution. Six cells on a forty-cell island is not a warp, it is a rounding error.

The working fix stops pretending noise will save it and modulates the island's own
distance field, in polar coordinates around its centre:

```
lobes = 1 + 0.34 sin(3a + 1.7i) + 0.18 sin(5a + 2.9i) + 0.10 sin(9a + 0.6i)
r = r / lobes
```

Three harmonics with incommensurate phases give a shape with no axis of symmetry,
which is what a real island has. It is a fake, and it looks less fake than the
noise did.

## 15. A swatch must be the colour the map actually prints

The political plate lays a province tint over paper at 55 per cent and then drops a
hillshade on it. The key originally printed the raw tint, and the two did not
match: sampled, the key was `(164, 181, 198)` and the map under the same label was
`(182, 183, 178)`. The eye reads that as "the key is wrong" without being able to
say why.

The swatch is now produced by the same function that paints the map, at the same
opacity, over the same paper. It is a small thing that decides whether a plate
looks printed or assembled.

Two more, from the same plate. Land that drains to a mouth outside the top-k
provinces had no province at all and printed as pale holes along the coast; it is
now grown inward from labelled neighbours until none is left. And `draw_scale_bar`
centred its units label on the end of the bar, so half of "Leagues (200)" hung
outside the frame; right-anchored, it stays in.

## 16. Blending toward white is not the same as drawing faintly

The graticule was drawn by pre-blending ink toward white, `ink * a + white * (1-a)`.
At `a = 0.10` that is a line 90 per cent white: invisible on paper-coloured land, a
pale stripe on a blue sea — and on the first attempt, where the ink arrived as
0–255 and was blended as if it were 0–1, very nearly black. Two bugs in one
expression, and neither of them was "the alpha is wrong".

A faint line is not a washed-out line. It is a line drawn at low opacity over
whatever is underneath, which means an alpha channel and a composite, not
arithmetic on the colour.

---

## Smaller decisions worth recording

**The border is a measuring instrument.** A graduated chequer band outside the neat
line, one block per degree, was the single change that made the plate read as a
period chart instead of a modern map in old colours. It costs forty lines of code
and does more for the illusion than the hillshade does.

**Rhumb lines belong to the compass.** Sixteen rays at 22 per cent opacity,
multiplied by the water mask so they stop at the shore. At 8 per cent they are
invisible; at 30 they are a spiderweb. The useful range is narrow.

**The index is set in two columns.** One narrow column leaves two thirds of the
page blank and runs the list onto a second sheet for no reason.

**Prose is sized to fit, by measurement.** A gazetteer page whose last paragraph
runs off the bottom is worse than a page in smaller type, so `text_page` measures
the column at the intended size and steps down until it clears the foot.

**A political plate wants less relief, not none.** At 15 per cent the hillshade
fights the flat wash and the provinces look like crumpled paper; at 11 it reads as
a tint laid over ground, which is what it is.

**Sea level by quantile.** Rather than tuning a threshold until the land fraction
looks right, take the quantile of the potential that gives the requested fraction.
The coastline then lands wherever the noise puts it, and composition is a
parameter instead of an accident.

**Sea floor is not hillshaded.** Antique maps leave water flat and shade only the
land. Hillshading the shelf produced a bright halo around every island that read
as a printing fault.

**The map edge is an outlet.** Border cells are seeded into the priority flood, so
water can leave the sheet. Without this, the frame becomes a wall and the coastal
ring fills with a moat.

**Contours need a floored gradient.** On dead-flat ground the true contour spacing
is infinite, and the unclamped formula renders an occasional straight scratch
across a plain. Flooring the gradient at a small value keeps lines continuous.

**Lakes are filtered by area.** The terrain is full of shallow closed basins.
Rendering all of them gives a continent that looks diseased. A blurred occupancy
test keeps only water wide enough to be worth a name — which is also what a
cartographer would do.

**Names are drawn from regional phonologies.** Random syllables produce letter
soup. What makes a map feel like a place is that names within one province sound
related: the same onsets recur, the same endings, the same rhythm.

**The cartouche shrinks its own title.** A title that overflows its frame is worse
than a small one, so the type is fitted to the box at draw time.

---

## What is still wrong

- The far north has no ice cap. Latitude cools the temperature, but nothing
  accumulates as ice, so a polar sea is drawn as open water and a polar coast is
  drawn with the same greens as a temperate one.
- Islands are lobed by construction, which is honest but monotonous: the same
  three harmonics at every scale, and an archipelago reads as a family.
- Soundings are scattered by a rule, not surveyed: a real chart thins them with
  depth and lines them up along the approach to a harbour.
- On a long straight slope the jittered hachures can still fall into visible
  diagonal rows, because the jitter is bounded by the lattice spacing.
- There is no inset map and no political hierarchy: every province is the same
  size and the same kind, every town the same rank, and no road is more important
  than another.
- The index gives one folio per name, so a river that crosses three plates is
  listed once.
- The naming has no grammar. Names are phonotactically consistent within a
  province, but there is no syntax: no "Great", no "of", no river named after a
  town named after a saint.
