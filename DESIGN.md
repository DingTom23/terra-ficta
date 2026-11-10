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
bucket queue over quantised heights, it is O(N) and takes 1.8 s for 540 000 cells:

```
fill 1.8s  raised=14791  norecv=0  maxacc=36652
```

Zero cells without a receiver, and the largest drainage basin went from 1 822
cells to 36 652 — from a rash of puddles to a river system.

**The bucket queue needs a heap inside each bucket.** My first version kept a
plain list per bucket and popped with `bucket.pop()`, which is last-in-first-out.
Cells that landed in the same bucket then expanded in the order they were pushed
rather than the order of their heights, and a cell could keep an overfill. The
error is bounded by one bucket width — invisible when the field is stored in
metres and the bucket is half a millimetre — but it is still wrong, and it showed
up the moment the same code was tested on a 0–1 field: 0.04–0.06 m of error over
4–22 cells, and 2–4 cells choosing a different receiver.

The proof that a bucket index never decreases is one line: a pushed cell has
`v ≥ filled[i] + eps`, so `level(v) ≥ level(filled[i])`. Given that, a heap
inside each bucket makes the global pop order exactly non-decreasing, and the
result becomes bit-identical to a textbook heap flood. `tests/test_hydro.py`
asserts that equality on five seeds.

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

## 17. A political plate must not show depth

The political sheet reuses the same sea as the physical one, and it looked wrong
in a way that took three attempts to name. The shallow end of the bathymetric
ramp is nearly paper, so a broad continental shelf printed as a soft pale cloud
round every headland — fog lying on the water. Flooring the depth made the shelf
one tone but left a smooth gradient across it, which the eye still read as haze.
Removing the depth rings helped and did not fix it.

What fixed it was giving up the premise. A political sheet is about areas, and
depth is an area that does not matter: the sea is now a single flat wash, with
the abyss beyond the shelf break allowed to darken and nothing else. The coastal
hachures came off at the same time — on a flat sea they read as dirt on the
paper. Two layers that earn their place on the physical plate are pure noise on
the political one, and the plate only became legible when they were removed
rather than tuned.

## 18. A name is generated once

The first printed copy of the book disagreed with itself. Plate IV was headed
CHOUCHULDORA; the contents and the gazetteer called the same country
SKAISKUGARD. Nothing was random about the output — every run produced exactly
that pair of names — which is what made it hard to see.

The cause was two calls to the same generator. `compose()` fills an empty title
with `namer.title(...)`, and `cli.py` asked the same `Namer` for the title again
when it built the gazetteer page. The two calls sit a few hundred draws apart in
the stream, so the second name is not a repeat of the first; it is the next name
in a sequence nobody meant to advance. Every plate that draws a name and every
page that prints one must be handed the same string.

The same class of bug appeared twice more in the same proof. The title page
counted countries where it should have counted plates, and the index squeezed
sixty-four rows onto a page regardless of how tall the type was, so the folios
collided with the names. All three are failures of a page describing something
other than what was printed — the kind of error a reader finds in ten seconds
and the generator never reports.

---

## 19. The cache is part of the model

The world cache was the source of two bugs that looked like renderer bugs, and
both had the same shape: **the cache knew less than the world did.**

The first: `_cache_key` hashed a hand-written list of `WorldSpec` fields, so
adding a dial meant editing the key or silently reusing a stale world. Editing
`terrain.py` changed nothing for twenty minutes because the npz still held the old
continent — the render printed `terrain 0.2s` and I believed it. The fix is to
hash `asdict(spec)` and ignore only what is presentation (`name`, `title`,
`style`, `projection`, `supersample`, `extra`). The failure mode then becomes a
needless rebuild instead of a wrong world, which is the right way round: a cache
should be allowed to be slow, never to be wrong.

The second: the cache stored one field of meta (`relief`) and the loader rebuilt
the dict as `{"relief": relief}`. But `labels.py` reads `world.meta["inland"]`.
A cold world and a warm world therefore laid out different labels from the same
seed — the same country, named differently depending on whether it had been built
before. The fix stores every array as `meta_<key>` and everything else as JSON,
and rebuilds the whole dict on load.

One more, quieter: `accumulation` and `lake_depth` were computed in float64 and
cast to float32 only on the way into the cache, so a cold world held float64 and a
warm one float32. Cast at the source instead. `tests/test_world.py` now asserts
that a warm world equals a cold one array by array, and that the labels match too.

## 20. Measure the country before describing it

The gazetteer used to say "the longest river" about `rivers[0]`, "the largest
town" about `towns[0]` and "the most remote" about `towns[-1]`. All three were
just the first and last entries in a list built in raster order, which is to say
they were the topmost and bottommost, and the prose was decoration over an
unmeasured world.

`atlas/features.py` measures instead. Rivers are traced downstream through the
D8 receivers and upstream by taking, at each step, the neighbour that drains into
the current cell with the largest accumulation — the main stem, not any branch.
Length is summed in kilometres, with the east-west scale corrected by
`cos(mid_latitude)`; drained area comes from the accumulation at the mouth. Towns
get their rank from the label and their distance to the nearest other town. A
range "carries the highest" only if the peak actually falls inside it.

The prose then has nothing to invent. Every number in it is a number the world
produced, and `tests/test_consistency.py` checks that the river called longest is
the one `longest_river()` returns, printed with its measured length.

## 21. One province, one language

Towns were given `cultures[i % len(cultures)]` by their index in the town list,
and province names used a third index — the count of names already taken. So a
village spoke a language unrelated to the province it stood in, and the province
name matched neither. Nothing about the map looked wrong; it was only wrong if you
read it.

The rule is now stated once and tested: **everything inside province `j` speaks
culture `j`.** `_culture_index` returns the index, never the object, because
`cultures.index(culture)` silently returns the first equal culture and two
provinces can generate identical phonologies. `Label.culture` records the index so
the invariant is checkable rather than merely intended.

---

## 22. A gazetteer is a measurement, not a mood

The first version of the dictionary wrote sentences about a country: "the
longest river", "the largest town".  Every one of those superlatives was a
guess about list order, and list order was the order the labels happened to be
built in.  The river called longest was simply the first river label.

The rule that replaced it: **a fact in the prose must be a field on a dataclass
that something computed.**  `features.measure()` walks the receivers and the
accumulation grid once and returns `River.length_km`, `Town.pop`, `Range.high_m`,
`Province.area_km2`, `Sea.shore`.  The prose formats those fields and nothing
else.  If a sentence needs a fact that is not on the dataclass, the fact gets
measured, not invented.

Two consequences worth keeping:

- The measurement found a real bug.  A river measured 40 km where the map
  showed 2,770, because `World.cropped()` re-routed the cropped grid without
  filling its depressions first — every pit became a sink and the channel ended
  in mid-air.  The prose was right and the routing was wrong.
- The prose exposed a cartographic error nobody had noticed on the plate: every
  town sat at 1 m above the sea.  The site score rewarded `near_water` and
  `river` so strongly that the best cell in the country was always a river
  mouth.  Twelve towns, twelve mudflats.  A `shelf` term — a gaussian centred on
  140 m — moved them upstream to where a town that lives by water actually
  stands.  Nothing on the plate had looked wrong, because a dot at the coast
  looks like a port.

That is the argument for writing the book at all: prose is a test suite for
geometry.

## 23. Ice has to be whiter than the sea, not bluer

The first attempt tinted sea ice toward a cold blue-grey and it disappeared.
The reason was not subtle: the shallow sea on this palette is already a pale
blue-grey, so "a colder version of the sea" was the same colour as the sea.  Ice
reads because it is **lighter** than everything around it — `cold_sea` had to go
to `(0.962, 0.968, 0.972)`, near white, before the pack ice appeared at all.

Two further rules came out of the same afternoon:

- **Keep the hillshade under the ice.**  A flat white cap looks like a hole torn
  in the plate.  The land ice is `cold_land * (0.80 + 0.34 * shade)`: the relief
  still models the surface, it is only the colour that has been taken away.
- **Stop the water lining at the ice.**  This was the real reason the first
  attempt read as fog.  The depth rings are drawn across the whole sea, and a
  frozen sea with a hundred depth rings under it is a grey smudge, however white
  the ice is painted.  `_ice` now returns the frozen fraction and the lining is
  multiplied by `1 - frozen`.  A layer that is right for water is wrong for ice.

The margin of the ice is noise-warped rather than a parallel of latitude,
because a straight edge of ice is the one thing an arctic chart never shows.

Drawing the ice took three more attempts after that, and each failure is worth
naming:

- **A bilinear edge is fog.**  Upsampling the ice mask and blending it leaves a
  forty-pixel fade between ice and water, and no amount of white paint survives
  a forty-pixel fade.  The masks are thresholded to 0 or 1, so the edge is an
  edge.
- **A field of white shapes is camouflage.**  Thresholding a blurred noise field
  into fifty per cent ice gives an even, meaningless pattern that reads as
  dazzle paint.  Ice is a **wash inside a limit** — solid in the north, thinning
  toward the margin, with a sparse scatter of separate floes where the pack is
  breaking up (the top twenty per cent of the floe field, not half of it).
- **The limit needs a line.**  The ice edge is a coastline of its own, so it is
  drawn the way the sheet draws a border: a dotted rule, in the same comb
  pattern.  Without it the white area has no shape.

The margin itself took two more tries.  A three-octave noise field at four
cycles per map, normalised by its maximum, produced a line with sharp triangular
peaks — a mountain profile, not an ice edge.  What works is **two octaves at a
low frequency**, normalised by the field's *spread* rather than its extreme, so
the margin meanders in a few great slow embayments.  An ice margin is the
smoothest line on an arctic chart; it is the coast that is jagged.

## 24. A country needs a latitude

All six countries were between 12 and 58 degrees north.  They had different
coastlines and different names and were, climatically, the same country drawn
six times — the biome census barely changed between plates.

Latitude is now drawn from the seed (`cli.latitude_for`): the span stays wide,
because a country needs a north and a south to have a climate gradient at all,
but the band moves.  Seed 7 sits at 68°N and has a frozen sea; seed 4 sits at
52°N and has none.  This cost one line of code and made the book a book about
places rather than a book about a program.

It also broke the framing, briefly.  `choose_frame` crops to the land, so pack
ice north of the coast fell outside the plate.  Letting the frame chase the ice
made it grow to the whole grid — pack ice spans the full width of the map — and
the country became a detail in the middle of an ocean.  The frame stays on the
land; the margin carries the ice.

## 25. Two plates, one country, a century apart

A book of six countries is a catalogue.  A book that shows the same country twice
is a history -- and the reader supplies the history, because the two sheets are
the same ground and nothing else about them agrees.

The ancient plate is not a second render of the same data with a filter on it.
Three things change, and they change for a reason:

**The borders.**  Provinces are grown from watersheds: river mouths are seeded,
each is weighted by the ground that drains through it, and the land is divided
between them.  Redraw the partition from a different seed and a different count
and the new provinces still follow the same ridges -- which is what real
frontiers do.  They are not random polygons laid over the country; they are
another reading of the same geography.  That is why the plate is legible as a
*previous* political order rather than as an invented one.

**The names, but not all of them.**  A province is a government, so its name
changes with the government.  A town is a place, so its name does not: a town
called Achmouth in 1703 is Achmouth in 1803, and the reader who flips between the
sheets sees the same dots under the same names inside different frontiers.  That
single invariant is what makes the two plates read as one country.  Renaming the
towns as well would have made two unrelated maps that happened to share a
coastline.

**The paper.**  The strongest signal is not a label but a colour.  A sheet a
century older has browned: `PAPER_OLD` is warmer and darker than `PAPER`, and it
is applied to the page, the land base and the paper texture together, so the
whole sheet ages as one object.  The province tints fade with it -- the greens
go olive, the blues go to dust.  Without the paper, the plate read as the
political sheet with different colours and the eye refused to believe it was
looking at another time.

What the plate deliberately does *not* do is invent a history: no battle sites,
no ruined cities, no "here the old kingdom fell".  It shows a political
arrangement and lets the reader notice that the arrangement is not the one on the
facing plate.

## 26. A thematic plate is an argument, not a surface

The physical plate models a surface; the political plate partitions one.  The
climate plate *interprets* one, and an interpretation has to be stated as a
number -- which changes the whole grammar of the sheet.

**Bands, not a ramp.**  A hypsometric ramp says "higher" and leaves the reader to
supply the figure from the key.  A rainfall map says "between 700 and 1,000
millimetres", and the band edge is the claim.  So the land is quantised into
eight bands with hard edges, and the edges are drawn as isohyets.  Quantising a
continuous field looks like a step backwards until you notice that the step *is*
the information: the line is where the rain changes, and the reader can point at
it.

**Band the smooth field, not the grid.**  The first attempt banded the terrain
grid and upsampled the result, which gave a staircase at every boundary.  Banding
a blurred full-resolution field gives a curve, because the curve is where the
*interpolated* value crosses the break.  Same data, same breaks, different line.

**Two hands, two colours.**  Rainfall is drawn blue and temperature red, and the
isotherms are broken while the isohyets are solid.  Colour alone is not enough --
a reader with a colour-blind eye, or a reader looking at a photocopy, must still
be able to tell which system of lines is which.

**The temperature had to be fixed before it could be drawn.**  The old model was
`31 - 0.62 * latitude`, a straight line that puts the equator at 31 degrees and
sixty north at minus six: it is too cold in the tropics and far too cold in the
horse latitudes.  A hypsometric plate hides that, because nobody reads a number
off a tint.  A climate plate states the number, so the model had to become one a
reader would recognise: 26 at the equator, 18 at thirty, 10 at forty-five,
nothing at sixty, minus twelve at seventy-five -- a quadratic, because the real
curve is one.

The lesson generalises.  A new *kind* of sheet is a test of the model underneath
it: the political plate tested the province partition, the ancient plate tested
whether history could be generated rather than decorated, and the climate plate
tested the climate.  Two of the three found something.

## 27. Rainfall has to know where it is

Giving each country its own latitude band was supposed to make the six of them
different.  It did not: measured across the six, the rainfall came out at a
median of 500 mm everywhere, with the tenth percentile at 270 and the ninetieth
at 1,200 in every single country.  The reason was one missing term.  The
moisture model knew about distance from the sea, about upslope lift and about
rain shadow -- all of which are local -- and nothing at all about the general
circulation, which is the one thing that is not local.

The circulation is a banded thing: wet where the trades converge at the equator,
dry under the subtropical highs near twenty-five degrees, wet again in the
westerlies near fifty.  Adding it as a multiplier on the rainfall -- a sum of
three Gaussians in latitude -- changed every country's climate at once.  The
country at 30 to 46 degrees north became a steppe, 41 per cent of it, with a
median of 387 mm; the one at 47 to 63 became boreal forest and tundra with a
median of 519.  Same terrain model, same seed machinery, same code -- a term
that knows where the country is.

The lesson is about what a *thematic* plate is for.  A hypsometric plate cannot
tell you that the climate model is latitude-blind, because a tint carries no
number and every country's tint looks plausible on its own.  A rainfall plate
puts the number on the sheet, and the moment the six sheets are laid side by
side the missing term is obvious.  Building the plate found the bug; the plate
was not the point.

## 28. A place must not depend on the sheet

A reviewer found the deepest bug in the project, and it was architectural
rather than visual.  The order of work was: build labels, place them, and then
build the roads from the towns that survived placement.  The dictionary was
measured from the same surviving list.  So a town whose name did not fit on a
sheet had no road, no dictionary entry and no index entry -- and because each
plate carries a different set of keys, the four plates of one country could
disagree about which towns existed at all.

Two things followed from the fix, and the second was worse than the first.

The first was mechanical: roads are built from every town in the country, and
the composer returns the whole country rather than the part of it that fitted.

The second was a name bug hiding behind the first.  Every name came from one
running generator, so each name depended on how many names had been drawn
before it.  The ancient sheet partitions the same ground into seven provinces
instead of five; two extra province names advanced the generator, and every
town in the country was renamed.  The same country, a century earlier, with
different towns in it -- which is not a century earlier, it is a different
country.

The fix is that a name depends on what it names and nothing else:
`Namer.derive(key)` returns a generator seeded by the feature's own identity
-- its kind and its position on the ground.  Move a province and the towns keep
their names, because the towns did not move.

The general rule: **geography is what exists; cartography is what is printed.**
Anything downstream of "did the label fit" is a rendering artefact, and a
rendering artefact must never reach the dictionary.

## 29. The hachures are a highlight, not a relief

A reviewer asked the question the plate could not answer: switch off the
continuous shading and see whether the line work still carries the form.  It
does not, and the measurement says why.

Isolating the ink the hachures add (the shaded sheet minus the same sheet
without them), and correlating it against the true slope:

- steepest tenth of the land: 0.0081 of darkening
- middle: 0.0012
- flattest tenth: 0.0002

So the hachures *do* follow the slope -- a factor of fifty from flat to steep --
but they are gated to the steepest ground and **94.5 per cent of the land
receives no hachure at all**.  They are a highlight laid on the mountains, not a
system for reading relief.  The hillshade is what makes the plate legible; the
lines decorate the part that is already legible.

That is not a bug to fix by drawing more strokes.  A sheet where lines carry the
form needs a different system -- strokes that follow the fall line across every
slope, spaced by gradient, drawn at a constant blackness -- and that is a
different plate, not a denser version of this one.  What this plate has is an
engraved *accent*, and it should be described as one.

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

- The ice has no history.  It does not advance or retreat, no coast is drawn as
  it would have looked in a colder century, and no iceberg is ever calved.
- The scale was not honest, and now is: a country spans sixteen degrees of
  latitude — seventeen hundred kilometres — so a river runs 800 km, a market
  town holds 11,000 people and its neighbour stands 110 km away.  What is still
  missing is density: two dozen towns on a million square kilometres, and the
  countryside between them holds nothing but rivers.
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

## 30. A line plate is a different claim, not a denser one

The reviewer asked the right question and then I measured it.  Isolating the ink the
hachures add -- the same sheet with and without them -- and correlating it against true
slope gives r = 0.230, with the steepest tenth of the land receiving 0.0081 and the
flattest 0.0002.  The correlation is real.  The coverage is not: **94.5 per cent of the
land receives no hachure at all.**  On the physical sheet the hachures are an accent laid
on the mountains; the continuous hillshade is what models the ground.

The obvious repair -- draw hachures everywhere -- is wrong, and I could see why before
trying it.  To cover ninety-five per cent of the land at the same stroke weight, the
strokes must be short and pale, and a field of short pale strokes is grey fog.  Density is
what makes a hachure plate legible; spreading the same ink over ten times the area removes
exactly the signal that carried the meaning.

So the line sheet inverts the arrangement instead of intensifying it.  The **contours** are
the height system: complete, unambiguous, readable with a key, and present on flat ground
where hachures have nothing to say.  The hachures become surface texture on every slope,
dense where the ground is steep and sparse where it is gentle.  The colour is withdrawn to
a thirty per cent hypsometric wash over paper.

Measured on the same seed: ink against slope rises from r = 0.230 to r = 0.349, and the
share of land with no ink at all falls from 94.5 to 26.7 per cent.  The flattest tenth of
the land now carries *more* ink than the middle tenth (0.2012 against 0.1300) -- not
because the hachures reach it, but because the flattest land is coastal plain and the
coast is inked.  That is a reminder worth keeping: on a line sheet the ink is a sum of
several systems, and any single coefficient describes the sum, not the system.

Two plates, one country, two different claims about how a map should be read.  Neither is
a mistake; printing only one of them would be.
