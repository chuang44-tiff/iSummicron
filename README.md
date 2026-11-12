# iSummicron

**Real-lens bokeh for iPhone Portrait photos: blur shapes ray-traced from public lens designs, changing with position in the frame and with depth.**

A phone's portrait mode blurs the background with the same round disc everywhere. A real lens does not: its blur spots have bright rims, and toward the frame edge they turn into eggs and cat's eyes that circle the center. iSummicron ray-traces two lens designs to get those shapes (their point spread functions, or PSFs), then paints them into your photo using its depth map.

[![Synthetic Christmas lights: the sharp input, a plain round disc and the real-lens 35 mm f/2 blur, bottom-left corner enlarged 2 times](https://github.com/chuang44-tiff/iSummicron/releases/download/v0.5.0/fig_xmas_zoom_input_disc_35.jpg)](https://github.com/chuang44-tiff/iSummicron/releases/download/v0.5.0/fig_xmas_zoom_input_disc_35.jpg)

Input (point lights enlarged for visibility) | plain round disc of matched size | real-lens blur, 35 mm default. A 2× crop of the bottom-left corner of one synthetic scene: point lights far behind a bauble that is in focus. The real lens turns each light into an oval or rounded box with a bright rim, tilted to face away from the frame center.

**Who it's for:** lens designers and lens enthusiasts who want to see how a lens design's bokeh would look in a photo. iSummicron starts from the lens prescription itself and renders that design's look onto everyday iPhone Portrait photos, including the look of rare or historic lenses. That is why the PSFs are ray-traced from a prescription with optiland, not measured or learned: any published design can be traced, including lenses nobody owns or that are too rare to measure, and the bokeh character (cat-eye, swirl) follows from the optics.

Two designs ship. The default 35 mm f/2 (US 3,006,249, Example II) gives smaller blur and clips hard in the corners. The 50 mm f/2 double-Gauss (`--lens 50`, the optiland sample) gives larger shapes that flatten and turn along the swirl toward the edges; it is traced beyond its design field, so its edge effects are large by construction. Both are [compared against plain discs](docs/how-it-works.md#christmas-lights-demo). To render another design, see [Your own lens](#your-own-lens).

**You need:** Python 3.12 and an iPhone Portrait-mode HEIC (other photos fall back to estimated depth). It is developed and tested on macOS with Apple Silicon; Linux and Windows are untested and render without HDR highlights. The distribution and CLI are `isummicron`; the Python import name is `summicron`.

**Scope:** bokeh only: blur shape, its variation across the frame, and in-focus to out-of-focus transitions. Color science and tone curves are out of scope. This is a solo-dev portfolio project, not a commercial product and not a paper.

> Summicron and Leica are trademarks of Leica Camera AG. iSummicron is not affiliated with, sponsored by, or endorsed by Leica Camera AG. The shipped PSFs are ray-traced design prescriptions, not measurements of any Leica lens.

## Install

macOS does not ship `python3.12`; install it first (for example `brew install python@3.12`).

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install "isummicron[heic,hdr-macos] @ https://github.com/chuang44-tiff/iSummicron/releases/download/v0.5.0/isummicron-0.5.0-py3-none-any.whl"
```

Or from the tag: `pip install "isummicron[heic,hdr-macos] @ git+https://github.com/chuang44-tiff/iSummicron@v0.5.0"`. Both lenses' blur-shape tables are inside the wheel. Installing from source, the optional extras and running the tests are in the [usage guide](docs/usage.md#installing-from-source).

## Try it

Download the sample Portrait HEIC from this release and verify it:

```bash
mkdir -p iphoneImage && cd iphoneImage
curl -LO https://github.com/chuang44-tiff/iSummicron/releases/download/v0.5.0/Demo_input.HEIC
curl -LO https://github.com/chuang44-tiff/iSummicron/releases/download/v0.5.0/SHA256SUMS
shasum -a 256 -c --ignore-missing SHA256SUMS && cd ..
```

The sample has its location and time metadata stripped and shows a person; the release notes list what metadata remains. Then render it:

```bash
isummicron render --image iphoneImage/Demo_input.HEIC --focus-xy 0.40 0.39 --depth-source lidar --downres 2048 --hdr-source macos --out-dir runs/render
```

This takes about 18 seconds on an Apple M4 Max Mac and writes a `.jpg` and a `.png` into `runs/render`. The result, with the default 35 mm lens (click for full size; the second row is a 1:1 crop toward the right edge of the frame):

| Before | After |
| --- | --- |
| [![Before: the input](https://github.com/chuang44-tiff/iSummicron/releases/download/v0.5.0/Demo_input_before.jpg)](https://github.com/chuang44-tiff/iSummicron/releases/download/v0.5.0/Demo_input_before.jpg) | [![After: rendered with iSummicron](https://github.com/chuang44-tiff/iSummicron/releases/download/v0.5.0/Demo_input_after.jpg)](https://github.com/chuang44-tiff/iSummicron/releases/download/v0.5.0/Demo_input_after.jpg) |
| [![Before, edge crop](https://github.com/chuang44-tiff/iSummicron/releases/download/v0.5.0/Demo_input_before_edge.jpg)](https://github.com/chuang44-tiff/iSummicron/releases/download/v0.5.0/Demo_input_before_edge.jpg) | [![After, edge crop](https://github.com/chuang44-tiff/iSummicron/releases/download/v0.5.0/Demo_input_after_edge.jpg)](https://github.com/chuang44-tiff/iSummicron/releases/download/v0.5.0/Demo_input_after_edge.jpg) |

On your own Portrait photos on macOS, `isummicron render --image photo.HEIC --focus-xy X Y` picks the same depth and HDR settings automatically and renders at full size; add `--downres 2048` for the fast preview and `--lens 50` for the 50 mm lens. `--focus-xy X Y` is the in-focus point in normalized image coordinates (X is the column, Y the row, `(0, 0)` is the top-left); without it, the closest subject is in focus.

The [usage guide](docs/usage.md) covers the focus picker, depth, HDR, memory and the common flags; the [CLI reference](docs/cli-reference.md) lists every flag. The same photo next to the iPhone's own Portrait mode is under [On a real photo](docs/how-it-works.md#on-a-real-photo).

## How it works

There is also an interactive explainer with a draggable animation and lens switches: [isummicron_pipeline.html](https://github.com/chuang44-tiff/iSummicron/releases/download/v0.5.0/isummicron_pipeline.html), a Release asset. It is one self-contained file: download it and open it in a browser.

![Animation: a point light moves away from the 35 mm lens, then toward the frame corner, then around the frame, with the blur shape shown beside the ray drawing](https://github.com/chuang44-tiff/iSummicron/releases/download/v0.5.0/fig_journey_35.gif)

The 35 mm design, traced: a point light moves from 0.7 m (in focus) out to 20 m, then to the frame corner, then once around the frame. Left: the rays through the lens. Right: the blur shape on the sensor.

A lens bends the rays from a point of light so that they meet again at one spot. If they meet in front of the sensor, they have spread out again by the time they arrive, and the dot becomes a disc. The lens's aberrations light that disc unevenly, which gives it a bright rim. Toward the frame edge the cone of light is tilted, and the rims of the glass cut it from two sides; what gets through is an off-center slice, the cat-eye.

<img src="https://github.com/chuang44-tiff/iSummicron/releases/download/v0.5.0/fig_psf_grid_35.png" width="560" alt="Grid of traced 35 mm blur shapes: five positions from the frame center to the corner, by five distances from 0.98 m to 20 m">

The 35 mm shapes the renderer uses. Rows: frame center (top) to corner (bottom). Columns: distance of the light, 0.98 m to 20 m, with focus at 0.7 m. Up is the direction away from the frame center.

The renderer turns each shape to face away from the frame center, so across a whole picture the shapes circle the center. That is swirly bokeh. The shapes are made once, offline, with optiland, an open-source Python ray tracer; nothing is traced while you render a photo. Each photo is then blurred in depth layers, using its own depth map. On macOS the photo's HDR gain map is used, so a blurred lamp keeps its real brightness.

The trace, the render pipeline, the formulas and every figure are in [docs/how-it-works.md](docs/how-it-works.md).

## Your own lens

The renderer accepts any PSF table in the layout of the two shipped ones, each `(R=12 radial bins, D=11 distances, 183, 183)` float32 in an `.npz`: pass it with `--tensor my_tensor.npz`. In a source checkout, `scripts/build_psf/` traces a new table from a lens prescription (a JSON file) with optiland. The file layout is under [Bring your own PSF tensor](docs/usage.md#bring-your-own-psf-tensor), and the build commands are under [Rebuilding the tables](docs/how-it-works.md#rebuilding-the-tables).

## Migrating from 0.4.x

0.5.0 replaces the bundled PSF table, so every render differs from 0.4.x. `--kernel-family`, `--lens-aperture-mm` and `--psf-norm` are removed, and any of them now stops with a one-line error; drop them and choose the lens with `--lens 35` (the default) or `--lens 50`. The full table is under "Migrating to 0.5.0" in [CHANGELOG.md](CHANGELOG.md).

## Limits

- **Turning in 8 steps.** Shapes face away from the center to within ±22.5°, in 45° steps around the frame.
- **Blur crosses the subject's outline.** By default, background blur is not stopped there. This changes only a hairline; see [Occlusion compositing](docs/usage.md#occlusion-compositing).
- **HDR needs macOS.** Elsewhere renders are SDR and the brightest lights come out dimmer. Linux and Windows are untested.
- **No edge darkening.** The clipping at the frame edge changes the shape, but the renderer does not darken the corners to match.
- **Front and back look the same.** Blur in front of the focus point uses the same shapes as blur behind it. A real lens differs there.
- **One color of light.** The trace uses green light (550 nm) only, so there is no color fringing in the bokeh.
- **Assumed rims.** The glass rim sizes follow a stated rule, not a measured lens. Both designs are public prescriptions, not measurements of any production lens.
- **Depth is the weak link.** The phone's depth sensor can read a window pane instead of what is behind it, and the blur can only be as right as the depth.
- **Photos without a depth map are unvalidated.** They render with estimated depth; results on such photos have not been checked in 0.5.0.

Each limit is explained in [How it works](docs/how-it-works.md#limits).

## Documentation

[How it works](docs/how-it-works.md) · [Usage guide](docs/usage.md) · [CLI reference](docs/cli-reference.md) · [Python API](docs/python-api.md) · [Troubleshooting](docs/troubleshooting.md) · [FAQ](docs/faq.md) · [CHANGELOG.md](CHANGELOG.md) · [Releases page](https://github.com/chuang44-tiff/iSummicron/releases)

## Prior art

**Learning Lens Blur Fields** (Lin et al., TPAMI 2025: [arXiv 2310.11535](https://arxiv.org/abs/2310.11535), [project page](https://blur-fields.github.io/), [code](https://github.com/estherlin/learning-lens-blur-fields)) published measured, spatially varying lens blur fields for several smartphones and Canon EF primes. It is the direct prior art for position- and depth-dependent PSF rendering. That work measures and learns the blur of specific physical devices, so it reproduces the cameras and lenses you have.

iSummicron starts from the other end: it simulates a lens design from its prescription, with PSFs ray-traced from public prescriptions, not measured. So it is aimed at lens designers and lens enthusiasts, and at the look of lenses you may not have. It also differs in shipping a discrete, inspectable `.npz` kernel bank and in rendering iPhone Portrait captures end to end (LiDAR depth, tap-to-focus, HDR highlights, depth-layered compositing).

## Credits and license

- **35 mm f/2 lens:** ray-traced from the expired US patent 3,006,249, Example II, using optiland.
- **50 mm f/2 lens:** the double-Gauss sample prescription that ships with optiland (MIT license).
- **Ray tracing:** optiland (PyPI package `optiland`, MIT license), used offline to build the lens tables.
- **Images:** the Christmas-lights scene and the diagrams are synthetic. The kitchen crops come from the sample photo, with metadata removed.

The code and both shipped tensors are Apache-2.0. The 50 mm tensor is traced from optiland's sample prescription, which stays under the MIT license. See `LICENSE` and `NOTICE`. Depth Anything V2 Small weights (Apache-2.0) are downloaded at runtime and are not redistributed.
