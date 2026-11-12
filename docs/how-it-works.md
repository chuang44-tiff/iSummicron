# How it works

Back to the [README](../README.md). This is the long version of the README's "How it works" section.

Photographers call the look of the out-of-focus parts of a picture *bokeh*. A phone's portrait mode makes it by blurring the background with plain round discs. A real lens doesn't. Its blur spots have bright rims, and toward the frame edge they turn into eggs and cat's eyes that circle the center.

iSummicron ray-traces public lens designs to get those shapes, then paints them into your photo using its depth map. This page shows where the shapes come from and how they get into the picture.

The same material exists as an interactive page, with a draggable animation, lens switches and plain-disc/real-lens toggles: [isummicron_pipeline.html](https://github.com/chuang44-tiff/iSummicron/releases/download/v0.5.0/isummicron_pipeline.html). The link downloads one self-contained file; open the downloaded file in a browser. The figures below are still frames from that page.

Contents:

- [Why the shape matters](#why-the-shape-matters)
- [How blur forms, step by step](#how-blur-forms-step-by-step)
- [Shapes across the frame](#shapes-across-the-frame)
- [How the shapes are traced](#how-the-shapes-are-traced)
- [The render pipeline, step by step](#the-render-pipeline-step-by-step)
- [Under the hood](#under-the-hood)
- [Why HDR matters for bokeh](#why-hdr-matters-for-bokeh)
- [Subject edges and occlusion](#subject-edges-and-occlusion)
- [Objects in front of the focus](#objects-in-front-of-the-focus)
- [On a real photo](#on-a-real-photo)
- [Christmas-lights demo](#christmas-lights-demo)
- [Limits](#limits)

## Why the shape matters

Below is one synthetic scene, rendered twice. Hundreds of point lights sit far behind a gold bauble, and the bauble is in focus.

| Plain round disc | Ray-traced 50 mm f/2 |
| --- | --- |
| ![The Christmas-lights scene blurred with the same round disc everywhere](https://github.com/chuang44-tiff/iSummicron/releases/download/v0.5.0/fig_xmas_50_plain_disc.jpg) | ![The same scene blurred with the ray-traced 50 mm f/2 shapes](https://github.com/chuang44-tiff/iSummicron/releases/download/v0.5.0/fig_xmas_50_real_lens.jpg) |

Left: every light becomes the same plain round disc, like a phone's portrait mode. Right: the blur shapes come from a ray-traced 50 mm f/2 double-Gauss design.

Look at the corners and the outer rings. On the right the shapes flatten and turn there, so that they circle the center. Same scene, same depth, same light. Only the blur shape changes.

The 50 mm design is traced beyond the field it was designed for, which makes its edge shapes large by construction; see [How the shapes are traced](#how-the-shapes-are-traced). The default lens is the 35 mm.

## How blur forms, step by step

A lens bends every ray that leaves a point of light so they all meet again at one spot. The camera's sensor sits at a fixed distance behind the lens. If the rays meet exactly on the sensor, you get a sharp dot. If they meet before it, they have spread out again by the time they arrive, and the dot becomes a disc.

The pattern that one point of light leaves on the sensor has a name: the *point spread function*, or PSF. This page mostly calls it the *blur shape*. Real lenses also have *aberrations*: small imperfections in how they bend the rays.

The four steps below walk through this with the 35 mm design. The ray paths inside the lens and every blur spot come from the traced data.

![Animation: a point light moves away from the 35 mm lens, then toward the frame corner, then around the frame, with the blur shape shown beside the ray drawing](https://github.com/chuang44-tiff/iSummicron/releases/download/v0.5.0/fig_journey_35.gif)

The whole journey as a 12-second loop. The bar under the drawing follows the light: 0.7 m to 20 m straight ahead, then from the frame center to the corner, then once around the frame. The animation and the step drawings below have small labels and are best viewed on a wide screen.

How to read the drawing: the light's side is not to scale. Real distances run from 0.7 m to 20 m, while the lens is a few centimeters long. The lens, the ray paths through it and the gap to the sensor are drawn to scale from the trace. The inset magnifies the last few millimeters before the sensor. The bar is spaced by blur, not by meters, so most of its first part is spent below a few meters, where the blur changes most.

### 1. In focus

![Side view of the 35 mm lens with rays from a light at 0.7 m meeting on the sensor; the spot is a single sharp point](https://github.com/chuang44-tiff/iSummicron/releases/download/v0.5.0/fig_journey_35_s1_onaxis_0p7m_focus.png)

A point light at 0.7 m, the focus distance: the rays meet on the sensor and land as a sharp dot.

The lens is focused at 0.7 m and the sensor is fixed there. The rays that get through the lens meet almost exactly on the sensor (the lens's small leftover blur is treated as sharp). The photo already holds this detail, so iSummicron leaves it untouched.

### 2. Farther away

![The same view with the light at 20 m; the rays cross before the sensor and the spot is a bright-rimmed disc](https://github.com/chuang44-tiff/iSummicron/releases/download/v0.5.0/fig_journey_35_s2_onaxis_20m.png)

The light at 20 m, straight ahead: the rays cross in front of the sensor and the dot grows into a bright-rimmed disc.

As the light moves back, its rays meet *in front of* the sensor (see the inset), cross, and spread out again. The disc is not evenly lit: it has a bright rim, and on the 50 mm also a bright center. That pattern comes from the lens's own aberrations.

Blur follows 1/distance, so the disc grows quickly over the first meter or two and hardly changes between 5 m and 20 m. On a photo 5712 px on its long edge, the on-axis disc is:

| Light at | 35 mm | 50 mm |
| --- | --- | --- |
| 1.37 m | about 46 px (0.8% of the long edge) | about 74 px (1.3%) |
| 20 m | about 90 px (1.6%) | about 146 px (2.6%) |

### 3. Toward the edge

![The light at 20 m, two thirds of the way to the corner; some rays are stopped by the glass rims and the spot is an egg shape](https://github.com/chuang44-tiff/iSummicron/releases/download/v0.5.0/fig_journey_35_s3_20m_midring.png)

The same far light, 67% of the way to the corner: the rims clip the tilted cone (gray rays) and the spot becomes an egg.

![The light at 20 m near the frame corner; most rays are stopped and the spot is a small rounded box](https://github.com/chuang44-tiff/iSummicron/releases/download/v0.5.0/fig_journey_35_s3_20m_corner.png)

Near the corner (98%): most of the cone is clipped, and the spot shrinks into a small rounded box.

Toward the frame edge the cone is tilted, so two different rims cut it: the front element trims one side, the stop and rear lenses trim the other, by different amounts. The gray rays are the clipped part; each ends where it is stopped. What survives is a narrow, off-center slice, which becomes the cat-eye. Aberrations also bend the surviving rays unevenly, so the spot gets bright edges. It is not just a circle blur.

Shape sizes at 20 m, along the radius × across, and the share of light that gets through:

| Position | 35 mm | 50 mm |
| --- | --- | --- |
| 67% of the way to the corner | 73 × 63 px, 66% | 100 × 130 px, 75% |
| 98% | 23 × 35 px, 18% | 53 × 111 px, 27% |

### 4. Around the frame

![A photo frame filled with a 5 by 7 grid of traced shapes, each turned to face away from the center; one circled shape is enlarged at the right](https://github.com/chuang44-tiff/iSummicron/releases/download/v0.5.0/fig_journey_35_s4_around_frame.png)

Every far light gets the shape for its distance from the center, turned to face away from the center. Shapes are drawn 7.1× real size.

The circled light travels around the frame and its shape turns with it. Together the shapes circle the center. This is spinning or swirly bokeh. See it in a full scene in the [Christmas-lights demo](#christmas-lights-demo).

The real lens turns each shape to face exactly away from the center (36° for the circled light). The renderer turns shapes in 8 fixed steps, so it uses 23° for that light, 13° off. See [8 hard angular wedges and the rotation](#8-hard-angular-wedges-and-the-rotation).

## Shapes across the frame

These are the blur shapes the renderer uses: the 35 mm lens by default, the 50 mm with `--lens 50`. There is one per ring of the frame, counted from the center out to the corner, and per distance. Each is drawn with the direction away from the frame center pointing up.

![Grid of traced 35 mm blur shapes: five positions from the frame center to the corner, by five distances from 0.98 m to 20 m](https://github.com/chuang44-tiff/iSummicron/releases/download/v0.5.0/fig_psf_grid_35.png)

35 mm f/2. Rows: position from the frame center (0%) to the corner (100%). Columns: distance of the light, with focus at 0.7 m. Each cell shows 95 × 95 px of a photo 5712 px on its long edge.

![The same grid for the 50 mm lens; the shapes are larger and the corner shapes are flattened ovals](https://github.com/chuang44-tiff/iSummicron/releases/download/v0.5.0/fig_psf_grid_50.png)

50 mm f/2, same layout. Each cell shows 149 × 149 px.

Brightness in both grids uses a square-root scale so that faint parts show.

Near the center the shapes are round discs. Toward the corner, the lens rims clip the cone of light from below and above, and aberrations bunch the surviving rays into bright edges.

| At 20 m | 35 mm | 50 mm |
| --- | --- | --- |
| Disc at the frame center | about 90 px across | about 146 px across |
| Corner shape | 23 × 35 px | 53 × 111 px |

The traced 35 mm lens also has a faint horizontal streak at close range. It is a fan of rays from the edge of the *aperture*, the opening that lets light through the lens. It is real lens behavior, so it stays in.

## How the shapes are traced

The blur shapes are made once, offline, with **optiland**, an open-source Python ray tracer. A ray tracer is a program that follows rays of light through a lens, one surface at a time. Nothing is traced while you render a photo: the renderer loads a finished set of shapes, the 35 mm by default or the 50 mm with `--lens 50`.

1. **Start from a public lens design.**
   - **35 mm f/2:** an 8-element design, ray-traced from the expired US patent 3,006,249, Example II, using optiland.
   - **50 mm f/2:** the six-element double-Gauss that ships as an optiland sample (MIT license). A full-frame 50 mm field (about ±23.4°) is wider than the sample was designed for, so its off-center aberration and clipping are large by construction.

   Each is scaled to its focal length at f/2 and traced in green light (550 nm). Both are ray-traced design prescriptions, not measurements of any production lens.
2. **Give every glass surface a rim.** Patents list curvatures and spacings, but not how big each piece of glass is. So the clear opening of every surface is set by a simple stated rule, then trimmed so that each element is thick enough at its edge to be made (at least 0.3 mm of glass). These rims are what clip the light toward the frame edge. Keep in mind that they are an assumption, not a measurement.
3. **Focus at 0.7 m and freeze the sensor.** The sensor distance is solved so that a point at 0.7 m is sharpest at the center of the frame. Then it stays fixed for every other distance, just like a real camera that has focused.
4. **Trace a dense grid of rays per spot.** This is done for 12 positions from the center to the corner and 10 distances from 0.98 m to 20 m. For each one, a dense, even grid of rays is traced from the point light (about two million launched per spot). Rays that hit any rim are dropped. The pattern of where the survivors land on the sensor is the blur shape.
5. **Turn ray hits into a pixel kernel.** A *kernel* is a small grid of pixels that holds one blur shape. Hits are counted into a 183 × 183 pixel kernel at the pixel size of a photo 5712 pixels on its long edge, with the direction away from the frame center pointing up. Nothing is cropped, clipped or smoothed: the box is sized to hold the whole shape, including long flare tails.
6. **Check the result.** The disc size at the frame center is checked against an independent scan of single rays across the whole aperture (within 3%). At least 99.5% of the light must land inside the box. A fixed random seed makes the trace repeat exactly. Both lenses pass.

The focus slot (0.7 m) is stored as a single bright pixel, not a traced spot. The photo already holds the in-focus detail, so the renderer's job there is to leave it alone.

| 35 mm f/2 | 50 mm f/2 |
| --- | --- |
| [![All 132 kernels of the 35 mm table as a grid of tiles](https://github.com/chuang44-tiff/iSummicron/releases/download/v0.5.0/fig_kernels_atlas_35.png)](https://github.com/chuang44-tiff/iSummicron/releases/download/v0.5.0/fig_kernels_atlas_35.png) | [![All 132 kernels of the 50 mm table as a grid of tiles](https://github.com/chuang44-tiff/iSummicron/releases/download/v0.5.0/fig_kernels_atlas_50.png)](https://github.com/chuang44-tiff/iSummicron/releases/download/v0.5.0/fig_kernels_atlas_50.png) |

Frame center (top) to corner (bottom); 0.7 m (left) to 20 m (right). These are the complete tables: one row per ring, 12 in all, and one column per distance slot, 11 in all, starting with the single-pixel 0.7 m focus slot. Click an atlas for full size.

### Rebuilding the tables

The build scripts are in `scripts/build_psf/` in a source checkout of the release tag (they are not in the wheel), with the two lens prescriptions as JSON in `scripts/build_psf/lenses/`. optiland is not needed to render; it lives in its own environment, `environment-optiland.yml` (Python 3.11, optiland 0.6.3). Each build takes about 3–5 minutes per lens.

The 35 mm table:

```text
conda env create -f environment-optiland.yml
conda run -n optiland python scripts/build_psf/build_raytrace_tensor.py \
  --patent-lens scripts/build_psf/lenses/us3006249_ex2.json \
  --fu 0.4 --rays-grid 1600 --seed 24 \
  --gt-python "conda run -n isummicron python" \
  --out my_tensors/psf_lens35_us3006249_fu0.4.npz
```

The 50 mm table:

```text
ISUMMICRON_BUILD_PSF_EFL_MM=50 ISUMMICRON_BUILD_PSF_EPD_MM=25 ISUMMICRON_BUILD_PSF_HALF_FIELD_DEG=23.3962047 \
conda run -n optiland python scripts/build_psf/build_raytrace_tensor.py \
  --patent-lens scripts/build_psf/lenses/optiland_doublegauss_50f2.json \
  --fu 0.4 --rays-grid 1600 --seed 24 \
  --gt-python "conda run -n isummicron python" \
  --out my_tensors/psf_lens50_doublegauss_fu0.4.npz
```

Render with the result through `--tensor`. The file layout is in the usage guide under [Bring your own PSF tensor](usage.md#bring-your-own-psf-tensor).

## The render pipeline, step by step

A depth map records how far away each part of the scene is. The renderer takes a portrait photo and its depth map, decides how far out of focus each pixel is, and gives each pixel the blur of the matching lens shape. These are the steps, in order.

1. **Depth from the photo.** Portrait photos carry a depth map from the phone's depth sensor. iSummicron reads it and scales it to the full image. It then lightly smooths the depth sensor's stair-steps, guided by the photo's own edges. If a photo has no depth map, a machine-learning depth estimator fills in.
2. **Tap to focus.** You pick the focus point. Its depth (a 3 × 3 median, to ignore single noisy pixels) becomes the in-focus distance. With no tap, the depth of the nearest 2% of the frame is used.
3. **Depth becomes blur amount.** Blur depends on *diopters*, not meters. A diopter is 1 / distance. That is why the background a few meters behind a subject is already nearly as blurry as the far horizon.

   For each pixel, iSummicron takes the diopter distance from the focus point and maps it onto the lens's 11 distance slots. A pixel 1 diopter behind your subject gets the blur the lens gives a light 1 diopter behind its own 0.7 m focus.

   <picture><source media="(prefers-color-scheme: dark)" srcset="https://github.com/chuang44-tiff/iSummicron/releases/download/v0.5.0/fig_diopter_layers_dark.png"><img src="https://github.com/chuang44-tiff/iSummicron/releases/download/v0.5.0/fig_diopter_layers.png" alt="The 11 depth layers as dots on a line, spaced by diopters, from the 0.70 m focus layer to the 20 m layer"></picture>

   The 11 depth layers on a diopter scale: equal gaps mean equal changes in blur.

   By default this mapping is one-to-one (a gain of 1), so the blur is physical and not tuned per photo. `--depth-gain` scales it up or down. Content in front of the focus point and behind it gets the same blur for the same diopter distance. This applies with the depth sensor's map. With estimated depth, a simpler straight mapping in meters is used.
4. **Blur in depth layers.** Each of the 11 distances is one layer. A pixel's blur is a smooth mix of at most three neighboring layers. The weights always add up to one, so there are no hard steps between near and far. Empty layers are skipped.
5. **The subject stays sharp.** Two things guard the focus point. First, pixels at its depth map to the first layer, and that layer's "blur" is a single pixel, which returns the photo unchanged. Second, at the very end, pixels close to focus are faded back to the original photo. The fade is full at the focus depth and drops to nothing by the next layer. In between, a pixel is a mix of sharp and lightly blurred.
6. **Blend across the frame.** The frame is split into 12 rings from the center to the corner. Each pixel blends the shapes of the two nearest rings, weighted by its distance between them. Going around the circle, there is no blending (see the next step).
7. **Turn the shape to face away from the center.** The lens shapes are stored once per ring, pointing up. The renderer turns them to face away from the center, but in **8 fixed steps of 45°**. Every pixel in a 45° slice of the frame gets the same turn.

   <picture><source media="(prefers-color-scheme: dark)" srcset="https://github.com/chuang44-tiff/iSummicron/releases/download/v0.5.0/fig_wedges_8_dark.png"><img src="https://github.com/chuang44-tiff/iSummicron/releases/download/v0.5.0/fig_wedges_8.png" alt="A frame cut into 8 wedges of 45 degrees around its center, with one arrow per wedge pointing away from the center"></picture>

   The frame is cut into 8 slices of 45° around its center. Each slice uses the shape turned to the slice's middle angle (arrows).

   A shape can therefore be up to 22.5° off its true angle, and its orientation changes by 45° from one slice to the next. Tests with 16 and 32 steps reduced the angle error but made a render about 1.6× and 2.8× slower. In side-by-side renders viewed normally (tone-mapped), 8, 16 and 32 steps looked the same, so the setting stays at 8.
8. **Highlights get their own pass.** Bright points such as bulbs, reflections and sun glints are split off before blurring. A local "is this a small bright spot?" test favors isolated lights over broad bright areas.

   These highlights are painted with the same lens shape, scaled to carry a fixed 2× the light's own energy. A bulb therefore stays a bright, well-defined shape, and big and small shapes carry the same light. Everything else is blurred with the shape scaled to keep its total light.

   The two parts are added back together in *linear light* (pixel values proportional to the real amount of light, before any tone curve), before the final tone step. The tone step keeps the photo's own look.

Kernel sizes are stored for a photo 5712 pixels on its long edge and scaled with the output size, so a smaller render gets proportionally smaller shapes.

## Under the hood

The same pipeline in more detail, with the formulas the renderer uses. Distances are in meters (m), and diopters (D) are 1 / distance.

### Depth to diopters to a lens distance

How a depth-map value becomes one of the lens's distance slots.

**The focus anchor.** The anchor is the depth at your tap: the median of a 3 × 3 patch, so a single noisy pixel can't move it. With no tap, the anchor is the depth of the nearest 2% of the frame (the 2nd percentile).

**The mapping.** For every pixel at depth *z*:

```text
dp  = | 1/z − 1/anchor |             diopters away from focus
e   = min(gain × dp, 1/0.7 − 1/20)    capped at 1.3786 D
z_t = 1 / (1/0.7 − e)                 equivalent distance, 0.7 … 20 m
```

*z_t* is where a light would have to sit, in front of the traced lens focused at 0.7 m, to be just as far out of focus as this pixel is from your anchor. Everything at the anchor's depth lands exactly on 0.7 m, the in-focus slot. That includes every pixel at that depth, not only the one you tapped. Depth values that are missing or invalid are also treated as in focus.

**The gain** is 1 by default, which makes the mapping physical: one diopter in your scene is one diopter for the lens. `--depth-gain N` multiplies it.

**A worked example** with gain 1: you focus on a subject at 1 m. A wall at 3 m is 1 − 0.333 = 0.667 D away, so it gets the lens's blur for 1 / (1.4286 − 0.667) ≈ 1.31 m. A very distant background is 1.0 D away and gets the blur for about 2.33 m. Anything more than 1.3786 D from focus gets the 20 m blur, the largest the lens makes.

**Front and back fold together.** Because of the absolute value, a pixel 0.5 D in front of the anchor gets exactly the same blur as one 0.5 D behind it. There are no traced slots in front of the 0.7 m focus (see [Objects in front of the focus](#objects-in-front-of-the-focus)).

**Estimated depth.** When a photo has no depth-sensor map, the machine-learning estimate is relative, not in meters, so a simpler straight mapping in meters is used instead.

### Depth layers and their weights

11 layers, at most three per pixel, weights that always add up to one.

**11 layers, one per distance slot:** 0.70, 0.98, 1.37, 1.91, 2.68, 3.74, 5.23, 7.32, 10.23, 14.30 and 20.0 m. Each is about 1.4× the one before. Layer 0, at 0.7 m, is the in-focus layer.

**The weight curves.** Layer *d*'s weight rises from slot *d*−1 to slot *d*, stays at 1 from slot *d* to slot *d*+1, and falls back to 0 by slot *d*+2. The rises and falls are smooth S-curves (3*t*² − 2*t*³) in meters. Then each pixel's weights are divided by their sum, so they always add up to exactly one. Below 0.7 m, layer 0 takes everything; beyond 20 m, layer 10 does.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://github.com/chuang44-tiff/iSummicron/releases/download/v0.5.0/fig_layer_weights_dark.png"><img src="https://github.com/chuang44-tiff/iSummicron/releases/download/v0.5.0/fig_layer_weights.png" alt="Weight curves of the 11 layers plotted against mapped distance on a diopter-spaced axis"></picture>

Weights of the 11 layers against *z_t*, drawn on a diopter scale. The in-focus layer is teal.

**What a pixel sees.** Between slots *k* and *k*+1 (for *k* ≥ 1), three layers overlap, with weights 0.5(1−*s*), 0.5 and 0.5*s*, where *s* is the S-curve of the pixel's position between the two slots. Between slot 0 and slot 1 only two overlap: 1/(1+*s*) on the in-focus layer and *s*/(1+*s*) on layer 1.

**Shapes are not mixed; results are.** Each layer blurs the frame with its own slot's traced shapes. The blurred results are then multiplied by the layer weights and added. No in-between shape is ever made up.

**Skipping.** A layer that no pixel gives a weight of at least 0.01 is skipped entirely. If skipping leaves a pixel with less than 2% total weight, it takes the result of the nearest layer that ran.

### Rings across the frame: the radial blend

12 rings from center to corner, a linear blend between the nearest two.

**Radial position** is a pixel's distance from the image center divided by the half-diagonal, so 0 is the center and 1.0 is a corner.

**12 rings.** Their edges sit at 0, 0.151, 0.302, 0.426, 0.522, 0.603, 0.674, 0.739, 0.798, 0.853, 0.905, 0.953 and 1.0. The outer ten rings have equal areas, and the central disc is split in two. Each ring's shapes are traced at the midpoint of the ring, from 0.075 to 0.977.

**The blend.** A pixel takes the shapes of the two ring midpoints on either side of it, with straight-line weights by distance between them. Inside the first midpoint, or beyond the last, it uses that one ring alone. A ring is skipped when its weight is below 0.01 everywhere in the wedge.

**Keeping the light.** Each layer's blurred result is divided by its total ring weight, so the blend never brightens or darkens the picture.

### 8 hard angular wedges and the rotation

How one stored shape per ring is turned to face away from the center.

**One orientation is stored.** The traced shapes have no angle axis. Each is stored once per ring and distance, with the direction away from the frame center pointing up.

**Wedges.** Each pixel's angle around the image center puts it in one of 8 wedges of 45°. In image coordinates (*y* pointing down):

```text
a   = floor( (atan2(y − cy, x − cx) + π) / 2π × 8 ) mod 8
θ_a = (a + 0.5) × 45° − 180°          wedge center: −157.5°, −112.5° … +157.5°
rot = −(θ_a + 90°)                    turns "up" to point at the wedge center
```

**The rotation** turns the shape about its center with bilinear sampling, clips any negative values to zero, and re-normalizes it. Every pixel in a wedge uses the same turned shape. Wedges are not blended, so a pixel's shape is off its true angle by up to ±22.5°, and the error follows a sawtooth around the frame.

**Per pixel**, that is at most 3 layers × 2 rings × 1 wedge = 6 shapes. The highlight pass uses the same turned shape. The single-pixel in-focus shape is never turned.

**Why 8.** More wedges cost time: about 1.6× at 16 and 2.8× at 32 (35 mm lens, 2048 px; median of 3 runs on an Apple M4 Max).

| Wedges | Worst angle error | Render time (35 mm lens, 2048 px, Apple M4 Max, median of 3 runs) |
| --- | --- | --- |
| 8 (used) | ±22.5° | 18.0 s |
| 16 | ±11.25° | 28.6 s |
| 32 | ±5.6° | 49.6 s |

In side-by-side tone-mapped renders, the three settings looked the same by eye, so the renderer stays at 8.

### How the focus point stays sharp

Three locks in a row, all pointing at the untouched photo.

1. **Diopter lock.** A pixel at the anchor's depth has dp = 0, so it maps to *z_t* = 0.7 m, the in-focus slot.
2. **Layer lock.** At that slot, the in-focus layer has weight 1 and every other layer 0. Its shape is a single pixel in all 12 rings, and blurring with a single pixel returns the photo unchanged. It is applied without rotation, and it stays an exact single pixel when the output is made smaller.
3. **In-focus snap.** At the very end, pixels near focus are faded back to the original photo:

   ```text
   α   = cos²( π/2 × min(r / 0.10, 1) )
   out = α × photo + (1 − α) × blurred
   ```

   Here *r* is the pixel's place on the slot axis, running from 0 at slot 0 to 1 at slot 10 in equal steps per slot. So the snap is full at the anchor and gone by slot 1.

**How wide the sharp zone is.** The snap reaches zero at slot 1, 0.4069 D from focus. There the pixel is half sharp, half layer-1 blur. The in-focus layer's share reaches zero at slot 2, 0.6979 D from focus. Past that, only blur remains.

With the default gain of 1 those are real diopters. For a subject at 1 m, the snap ends at about 0.71 m in front and 1.69 m behind. The sharp share ends at about 0.59 m in front and 3.3 m behind. A larger `--depth-gain` narrows the zone.

**The weak point is the anchor itself.** A tap on a depth edge, or on glass that the depth sensor reads instead of what is behind it, puts the sharp zone on the wrong surface.

### Bright-stamp highlights

How small bright lights are split off and painted as lens shapes.

**The split.** On the linear image, each pixel gets a bright weight:

```text
bw     = smoothstep from T − 0.2 to T + 0.2 of brightness
       × (0.10 + 0.90 × clip(tophat / p99(tophat), 0, 1))
tophat = brightness − grey_opening(brightness, 31 px)
bright = image × bw       diffuse = image − bright
```

The top-hat keeps bright features smaller than about 31 px and drops broad bright areas such as a window or a white wall. The 0.10 floor means broad bright areas still count a little. *T* is 1.0, SDR white. When the tone step reproduces the photo's own tone curve, *T* is raised to 1.1 × the HDR brightness that the curve maps to 90% white, if that is higher than 1.0.

**The stamp.** The bright part is blurred with the same turned lens shape as the rest, times 2. Its total light is kept and doubled, so a bulb reads as a bright, well-defined shape. With `--bright-stamp peak`, the stamp is instead scaled so that its brightest point is 1.

**Putting it back.** Both parts go through the same layer and ring weights. They are added together in linear light, before the final checks, the in-focus snap and the tone step. Pixels on the in-focus layer skip the split, so in-focus highlights pass through untouched.

### Key numbers

| Item | Value |
| --- | --- |
| Shape set | 12 rings × 11 distances, each 183 × 183 px |
| Reference photo size | 5712 px on the long edge. A smaller output scales shapes by long edge / 5712 (below 0.95×), to an odd size of at least 3 px: 67 px at 2048. |
| Focus slot | 0.7 m, stored as a single pixel |
| Distance slots | 11, from 0.70 to 20.0 m, about 1.4× apart |
| Diopter cap | 1/0.7 − 1/20 = 1.3786 D |
| Depth gain | 1 by default (physical), `--depth-gain` to change |
| Layers per pixel | at most 3; a layer is skipped below a 0.01 weight |
| Ring edges | 0, 0.151, 0.302, 0.426, 0.522, 0.603, 0.674, 0.739, 0.798, 0.853, 0.905, 0.953, 1.0 of the half-diagonal |
| Rings per pixel | at most 2 (linear blend) |
| Wedges | 8 × 45°, rotation −(θ_a + 90°), no blend |
| In-focus snap | cos² fade, gone at slot 1 (0.4069 D) |
| Highlight split | threshold 1.0 (SDR white), soft knee ± 0.2, 31 px top-hat, floor 0.10, normalized at the 99th percentile |
| Highlight stamp | same shape × 2 (energy-scaled) |
| Memory budget | 12 GiB by default; above it the renderer stops and suggests a smaller `--downres` |

## Why HDR matters for bokeh

Out-of-focus lights only look like lights if the renderer knows how bright they really are. A normal (SDR) photo stores everything brighter than white as plain white, 1.0. Spread a clipped bulb over a big blur shape and its light is shared across many pixels, so it comes out as a dim, soft disc.

An iPhone Portrait photo also carries an *HDR gain map*, which records how much brighter the highlights really were. In the sample kitchen photo, the brightest input highlights reach about 3.5× white instead of stopping at 1.0, and about 20% of the pixels have a color channel brighter than white. With that light restored, the blurred lamps stay bright and keep their shape.

![Two crops of blurred pendant lamps from the kitchen photo: left without the gain map, right with it](https://github.com/chuang44-tiff/iSummicron/releases/download/v0.5.0/fig_hdr_kitchen_lamps_none_vs_auto.jpg)

Blurred pendant lamps behind the subject, at 100%. Left: `--hdr-source none` (SDR). Right: `auto` (gain map); the lamps are brighter, fuller discs.

![Two crops of the in-focus subject's hair against a bright wall: left without the gain map, right with it](https://github.com/chuang44-tiff/iSummicron/releases/download/v0.5.0/fig_hdr_kitchen_hair_none_vs_auto.jpg)

The in-focus subject against a bright wall, same order. The two are nearly the same, because the subject is sharp and the wall is broad, not a point light.

Both crops: the default 35 mm lens, rendered at 2048 px on the long edge.

**Where it sits in the pipeline:** load the photo (with its gain map) → blur in linear light → tone step back to a normal SDR picture that keeps the photo's own look. The blur happens on the restored, brighter-than-white values; only the final picture is brought back to SDR.

**The `--hdr-source` setting:**

- `auto` (the default): uses the gain map when the photo has one and the macOS HDR loader is available, otherwise falls back to `none` and says so in one line.
- `macos`: always use the macOS HDR loader, with no fallback. It needs macOS 15 or later and the optional `hdr-macos` install extra.
- `none`: plain SDR. It works on any platform.

The gain-map path needs macOS, because it reads the photo with Apple's own image framework. On other platforms renders are SDR. They still work; the brightest lights are just dimmer. The difference is modest and shows mostly on the brightest lights. The output file is a normal SDR image, so you don't need an HDR display to see it.

## Subject edges and occlusion

In a real photo, the subject blocks some of the background's blur: a light just behind someone's shoulder is partly hidden by it. The default render does not model this. Background blur is free to cross the subject's outline.

Occlusion compositing stays off by default: even with real-lens blur it only changes a hairline along the subject's outline and is not visible at normal viewing, while costing about 30–60% more render time (1.3–1.6×); guided depth and the in-focus snap already keep subject edges clean. Use `--occlusion-compositing on` for scenes with very bright light directly behind the subject's outline.

With the option on, the depth layers are stacked front to back so that nearer layers cover farther ones.

The terms used here and above:

- **Guided depth:** the depth map is smoothed with the photo's own edges as a guide, so depth steps line up with the real outline.
- **In-focus snap:** at the very end, pixels at or near the focus depth are faded back to the original photo, so the subject stays exactly as shot.
- **Linear light:** pixel values proportional to the real amount of light, before any tone curve. Blurring and adding up light is only correct in these units.

## Objects in front of the focus

iSummicron only knows how far a pixel is from focus in diopters, not which side it is on. A light that is a given number of diopters *in front of* your subject is blurred exactly like one the same number of diopters *behind* it, and with the same lens shapes: the ones traced behind the focus. Both sides stop growing at the same maximum blur, 1.38 diopters from focus. With the default settings, for a subject at 0.7 m that means anything nearer than about 0.36 m gets the maximum blur; for a subject at 1 m, anything nearer than about 0.42 m.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://github.com/chuang44-tiff/iSummicron/releases/download/v0.5.0/fig_front_vs_back_dark.png"><img src="https://github.com/chuang44-tiff/iSummicron/releases/download/v0.5.0/fig_front_vs_back.png" alt="A diopter axis centered on the in-focus subject, with blur discs growing by the same amount on the near and far sides"></picture>

The same blur on both sides of the subject, set by diopter distance alone, using the behind-focus shapes.

A real lens is different in front of focus:

- **The shapes flip.** In front of focus the image of the lens opening is turned over, so tails and cat-eye cuts point toward the frame center instead of away from it.
- **The rim character swaps.** The leftover aberrations that give a disc a bright rim on one side of focus give it a soft edge on the other. So a lens with bright-rimmed "soap-bubble" discs on one side of focus has softer discs on the other.
- **A blurry foreground drapes over the subject.** A real out-of-focus object near the camera spreads softly across the sharp subject behind it, like a veil. The renderer does not do this.

## On a real photo

The sample kitchen photo, left to right: the input, the iPhone's own Portrait mode, the 50 mm render and the default 35 mm render.

[![Kitchen photo: input, iPhone Portrait, 50 mm f/2 and 35 mm f/2](https://github.com/chuang44-tiff/iSummicron/releases/download/v0.5.0/fig_kitchen_4up.jpg)](https://github.com/chuang44-tiff/iSummicron/releases/download/v0.5.0/fig_kitchen_4up.jpg)

## Christmas-lights demo

The README hero shows the scene as a 2× crop of the bottom-left corner: the sharp input, a plain round disc, and the real 35 mm blur.

![Input, plain round disc and real-lens 35 mm blur, bottom-left corner 2 times](https://github.com/chuang44-tiff/iSummicron/releases/download/v0.5.0/fig_xmas_zoom_input_disc_35.jpg)

The plain-disc panel uses the same scene, depth and settings with every 35 mm kernel replaced by a uniform round disc whose radius matches the traced kernel's 90% encircled-energy radius, so the discs look slightly smaller than the real-lens shapes, whose tails extend further.

This is a synthetic test scene. It has about 585 point lights in rings, strands and a tree, most at 20 m and some at 1.9 m. They sit behind a gold bauble in focus at 0.7 m. The scene has a fixed depth map and no photo, so only the blur differs.

Every version runs through the same blur engine. The "plain disc" versions use round discs sized to match each real lens, which is roughly what phone portrait modes do. They are comparison images for this page; the `isummicron` command does not make them.

The demo's lights use the renderer's default highlight setting: each bulb is spread with the shape scaled to its own total light times a fixed 2×. Big and small blur shapes therefore carry the same light.

The 35 mm and 50 mm lenses on the same corner crop:

[![Christmas lights: input, 35 mm f/2 and 50 mm f/2](https://github.com/chuang44-tiff/iSummicron/releases/download/v0.5.0/fig_xmas_zoom_input_35_50.jpg)](https://github.com/chuang44-tiff/iSummicron/releases/download/v0.5.0/fig_xmas_zoom_input_35_50.jpg)

### 50 mm f/2

| Plain disc | Real lens |
| --- | --- |
| ![The scene at 50 mm framing with plain round discs](https://github.com/chuang44-tiff/iSummicron/releases/download/v0.5.0/fig_xmas_50_plain_disc.jpg) | ![The scene with the ray-traced 50 mm f/2](https://github.com/chuang44-tiff/iSummicron/releases/download/v0.5.0/fig_xmas_50_real_lens.jpg) |

What to look at: the spinning bokeh. Toward the edges the shapes flatten along the line to the center, so the rings of lights seem to swirl. This lens is traced beyond its design field, so the effect is stronger than the design intended for a 50 mm frame; see [How the shapes are traced](#how-the-shapes-are-traced).

### 35 mm f/2

| Plain disc | Real lens |
| --- | --- |
| ![The scene at 35 mm framing with plain round discs](https://github.com/chuang44-tiff/iSummicron/releases/download/v0.5.0/fig_xmas_35_plain_disc.jpg) | ![The scene with the ray-traced 35 mm f/2](https://github.com/chuang44-tiff/iSummicron/releases/download/v0.5.0/fig_xmas_35_real_lens.jpg) |

What to look at: the corners. The wider lens clips much harder at the edge (under a fifth of the light gets through at the corner). Its corner shapes shrink into small rounded boxes with bright sides.

### Top-left corner, close-up

| Plain disc | Real lens |
| --- | --- |
| ![Corner close-up, 35 mm, plain discs](https://github.com/chuang44-tiff/iSummicron/releases/download/v0.5.0/fig_corner_35_plain_disc.jpg) | ![Corner close-up, 35 mm, real lens](https://github.com/chuang44-tiff/iSummicron/releases/download/v0.5.0/fig_corner_35_real_lens.jpg) |
| ![Corner close-up, 50 mm, plain discs](https://github.com/chuang44-tiff/iSummicron/releases/download/v0.5.0/fig_corner_50_plain_disc.jpg) | ![Corner close-up, 50 mm, real lens](https://github.com/chuang44-tiff/iSummicron/releases/download/v0.5.0/fig_corner_50_real_lens.jpg) |

The top-left corner of each version: 35 mm in the top row, 50 mm in the bottom row. With the 50 mm real lens, the ovals lean along the swirl.

## Limits

- **Turning in 8 steps.** Shapes face away from the center to within ±22.5°, and the orientation changes by 45° from one slice to the next. In normal (tone-mapped) viewing, 8, 16 and 32 steps looked the same; more steps cost render time.
- **Blur crosses the subject's outline.** By default, background blur is not stopped at the subject's outline. With real-lens blur this changes only a hairline, not visible at normal viewing; see [Subject edges and occlusion](#subject-edges-and-occlusion).
- **HDR needs macOS.** The gain-map path that restores real highlight brightness runs on macOS only. Elsewhere renders are SDR and the brightest lights come out dimmer.
- **No edge darkening.** The trace records how much light gets through at each position (down to roughly 18–28% in the corners), and the clipping changes the shape. The renderer does not use this to darken the corners: each shape keeps its full brightness.
- **Front and back look the same.** Blur in front of the focus point uses the same shapes as blur behind it, so lenses whose foreground and background bokeh differ can't be shown. See [Objects in front of the focus](#objects-in-front-of-the-focus).
- **One color of light.** The trace uses green light only, so there is no color fringing in the bokeh.
- **Assumed rims.** The glass rim sizes follow a stated rule, not a measured lens. Both designs are public prescriptions, not measurements of any production lens.
- **Depth is the weak link.** The phone's depth sensor can read a window instead of what is behind it, and the blur can only be as right as the depth.
- **Photos without a depth map are unvalidated.** They render with estimated depth and the straight mapping in meters; results with the new lens tables on such photos have not been checked in 0.5.0.

## Credits

Lens designs: the expired US patent 3,006,249, Example II (public domain), and the optiland DoubleGauss sample (MIT). Ray tracing: optiland. The Christmas-lights scene and the charts are synthetic. The two HDR crops come from the project's sample kitchen photo, with metadata removed.
