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

---

## Smaller decisions worth recording

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

- Islands are smoother than they should be: the anti-fringe blur that rescues the
  continental coastline rounds small detached landmasses into eggs.
- River labels are placed at the river but not rotated along it.
- The biome classification exists and is unused.
- Composition is fixed: the continent always sits near the centre, and the empty
  sea to its west is not exploited for insets or soundings.
