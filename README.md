# iSummicron

**Real-lens bokeh for iPhone Portrait photos: a simulated lens PSF that changes with position in the frame and with depth.**

The Python import name is `summicron`. The distribution and CLI are `isummicron`.

## Gallery

Sample Portrait photo, focus `0.40 0.39`, `--downres 2048`, quick-start flags. Click an image for full size. The crops are 1:1 pixels, toward the right edge of the frame.

| Before | After |
| --- | --- |
| [![Before: the input, tonemapped](https://github.com/chuang44-tiff/iSummicron/releases/download/v0.3.0/Demo_input_before.jpg)](https://github.com/chuang44-tiff/iSummicron/releases/download/v0.3.0/Demo_input_before.jpg) | [![After: rendered with iSummicron](https://github.com/chuang44-tiff/iSummicron/releases/download/v0.3.0/Demo_input_after.jpg)](https://github.com/chuang44-tiff/iSummicron/releases/download/v0.3.0/Demo_input_after.jpg) |
| [![Before, edge crop](https://github.com/chuang44-tiff/iSummicron/releases/download/v0.3.0/Demo_input_before_edge.jpg)](https://github.com/chuang44-tiff/iSummicron/releases/download/v0.3.0/Demo_input_before_edge.jpg) | [![After, edge crop](https://github.com/chuang44-tiff/iSummicron/releases/download/v0.3.0/Demo_input_after_edge.jpg)](https://github.com/chuang44-tiff/iSummicron/releases/download/v0.3.0/Demo_input_after_edge.jpg) |

The images are attached to the v0.3.0 release. The render is darker than the input: see the [FAQ](docs/faq.md).

## Why

Phone portrait blur is the same everywhere in the frame. A real lens is not: its blur changes shape from the center to the corners (swirl, cat-eye) and with distance from the focus plane. iSummicron convolves each pixel with a point spread function (PSF) picked by the pixel's radial position, angle and depth, from a table simulated in CODE V.

In 0.3.0 the field variation is subtle: highlights render as near-round discs. Making it physically right is the goal of the next releases.

**Scope:** bokeh only: blur shape, its variation across the field, and in-focus to out-of-focus transitions. Color science and tone curves are out of scope. This is a solo-dev portfolio project, not a commercial product and not a paper.

> Summicron and Leica are trademarks of Leica Camera AG. iSummicron is not affiliated with, sponsored by, or endorsed by Leica Camera AG. The shipped PSF is a simulated generic double-Gauss used as a proxy, not a measurement of any Leica lens.

## The lens, honestly

The wheel ships `summicron/_artifacts/codev_render_v1.npz`, a 4D tensor `(R=12 radial bins, D=11 distances, 183, 183)` of float32 **CODE V-simulated** PSFs. The simulated design is a **generic ~100 mm f/2 double-Gauss** (CODE V sample design `ag_dblgauss`), focused at 0.7 m with distances out to 20 m. Its green (550 nm) channel is used for all three RGB channels. It stands in for a Summicron-style double-Gauss. It is **not** a measured Leica lens.

## Install

### From a release (recommended)

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install "isummicron[heic,hdr-macos] @ https://github.com/chuang44-tiff/iSummicron/releases/download/v0.3.0/isummicron-0.3.0-py3-none-any.whl"
```

Or from the tag: `pip install "isummicron[heic,hdr-macos] @ git+https://github.com/chuang44-tiff/iSummicron@v0.3.0"`. Add `depth` to the extras for the monocular fallback. All versions are listed at https://github.com/chuang44-tiff/iSummicron/releases.

### From source (development)

With conda:

```bash
git clone https://github.com/chuang44-tiff/iSummicron.git && cd iSummicron
git checkout v0.3.0
conda env create -f environment.yml && conda activate isummicron
pip install -e '.[heic,depth,hdr-macos,eval-quality,dev]'
```

With a plain virtual environment (Python 3.12):

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e '.[heic,hdr-macos]'
```

Core dependencies are numpy, scipy and pillow. Optional extras:

- `heic`: pillow-heif, for HEIC input.
- `depth`: torch and transformers, for the monocular Depth Anything V2 Small fallback.
- `hdr-macos`: pyobjc Quartz (which re-exports Core Image), darwin only.
- `eval-quality`: scikit-image.
- `dev`: pytest, ruff and pyyaml.

A missing extra fails with an actionable `ImportError`.

In the conda environment, import numpy before torch in your own scripts: the conda-forge numpy/scipy and the pip torch wheel each bundle an OpenMP runtime, and importing torch first aborts with `OMP: Error #15`.

Tested with: macOS 26/27 arm64, Python 3.12, torch 2.12, transformers 5.9, numpy 2.4, pyobjc 12.2.

## Try it

Download the sample Portrait HEIC from this release and verify it:

```bash
mkdir -p iphoneImage && cd iphoneImage
curl -LO https://github.com/chuang44-tiff/iSummicron/releases/download/v0.3.0/Demo_input.HEIC
curl -LO https://github.com/chuang44-tiff/iSummicron/releases/download/v0.3.0/SHA256SUMS
shasum -a 256 -c --ignore-missing SHA256SUMS && cd ..
```

The sample has its location and time metadata stripped and shows a person; the release notes list what metadata remains. Then render it:

```bash
isummicron render --image iphoneImage/Demo_input.HEIC --focus-xy 0.40 0.39 --depth-source lidar --depth-scale 10 --downres 2048 --psf-norm l1 --hdr-source macos --out-dir runs/render
```

This takes about 25–30 s on an Apple Silicon Mac (dev machine) and writes `runs/render/Demo_input_r2048_rendered_ds10.0.jpg` and a `.png`. Native 5712×4284 takes about 5–6 minutes and about 5.4 GB measured (the memory guard's conservative estimate is about 9.4 GiB, under the 12 GiB default); set `ISUMMICRON_MAX_BYTES` to change the memory guard. Without `--out-dir`, output goes to `./runs/render`.

`isummicron render --help` lists every flag; `python -m summicron.render.convolve` accepts the identical set.

## Picking a focus point

`--focus-xy X Y` is the in-focus point in normalized image coordinates: X is the column, Y the row, `(0, 0)` is the top-left. Tap the subject's eye or face.

- **With the picker:** `pip install matplotlib` (a Tk-capable Python is needed), then run `python -m summicron.tools.anchor_picker --image iphoneImage/Demo_input.HEIC`. Click the subject and copy the printed `--focus-xy` values into `isummicron render`. The picker prints a developer-style command; keep only its `--focus-xy`.
- **By hand:** read the pixel position in any viewer and divide by the image width and height.

Without `--focus-xy`, the closest subject is in focus.

## Common flags

| Flag | Meaning |
| --- | --- |
| `--focus-xy X Y` | In-focus point, normalized. |
| `--downres N` | Long edge in pixels. 2048 is the fast preview. |
| `--depth-source` | `lidar` (Portrait HEIC, default) or `monocular` (Depth Anything V2 Small). |
| `--depth-scale` | Multiplier on the depth map before focusing. The quick start uses 10. |
| `--hdr-source` | `macos` keeps highlight headroom (macOS only); `none` is the 8-bit load. |
| `--out-dir`, `--output` | Output directory, or one exact output file. |
| `--occlusion-compositing` | `on` removes halos at depth edges (off by default, slower). |
| `--tensor` | Use your own PSF tensor (see below). |

Full list: [docs/cli-reference.md](docs/cli-reference.md).

## Documentation

- [Usage guide](docs/usage.md)
- [CLI reference](docs/cli-reference.md)
- [Python API](docs/python-api.md)
- [Troubleshooting](docs/troubleshooting.md)
- [FAQ](docs/faq.md)
- [CHANGELOG.md](CHANGELOG.md) and the [Releases page](https://github.com/chuang44-tiff/iSummicron/releases)

## How it works

The pipeline loads the image and its LiDAR depth, then anchors depth so the tapped pixel lands on the tensor's in-focus slice. Each pixel gets a kernel blended from its radial position (8 angular wedges rotate each kernel with the pixel's azimuth) and its depth slab. Depth slabs are gathered and composited in linear light, and the output is tonemapped to SDR.

## Bring your own PSF tensor

A tensor is an `.npz` with these required keys (without `object_distances_m`, `isummicron render` fails with `depth_m requires tensor with object_distances_m metadata`):

| Key              | Shape          | Notes                                    |
| ---------------- | -------------- | ---------------------------------------- |
| `kernels`        | `(R, D, H, W)` | float32; H == W; odd-sided; non-negative |
| `radial_edges`   | `(R+1,)`       | float64; strictly increasing in `[0, 1]` |
| `defocus_ratios` | `(D,)`         | float64; strictly increasing in `[0, 1]` |
| `object_distances_m` | `(D,)`     | float; object distances in meters, one per D node |

Recommended extra keys:

- `focus_distance_m`: the in-focus distance in meters. It must equal `object_distances_m[0]`, or loading fails with `TensorValidationError`. If it is missing, 0.7 m is assumed silently.
- `corpus_ref_long_edge`: the image long edge in pixels that the kernels are sized for. If it is missing, 5976 is assumed silently (the shipped tensor uses 5712).
- `delta_guard_applied`: if it is missing, the loader warns that in-focus detail will smear.

Add `schema_version='1.0'` to silence the legacy-layout warning.

Pass it with `--tensor my_tensor.npz`. The loader validates the schema and never unpickles. The shipped tensor is a working example.

## Tests

```bash
python -m pytest tests/ -m "not slow" -p no:xdist -q    # FAST suite
ruff check summicron/
```

Tests that need sample images or developer-only assets skip themselves when those are absent. The suite imports `summicron` from the checkout, so install with `pip install -e`.

## Prior art

**Learning Lens Blur Fields** (Lin et al., TPAMI 2025: [arXiv 2310.11535](https://arxiv.org/abs/2310.11535), [project page](https://blur-fields.github.io/), [code](https://github.com/estherlin/learning-lens-blur-fields)) published measured, spatially varying lens blur fields for several smartphones and Canon EF primes. It is the direct prior art for position- and depth-dependent PSF rendering. iSummicron differs in shipping a discrete, inspectable `.npz` kernel bank and in rendering iPhone Portrait captures end to end (LiDAR depth, tap-to-focus, HDR highlights, depth-layered compositing). Its PSFs are simulated, not measured.

## License

Apache-2.0, for the code and for the shipped PSF tensor. See `LICENSE` and `NOTICE`. Depth Anything V2 Small weights (Apache-2.0) are downloaded at runtime and are not redistributed.
