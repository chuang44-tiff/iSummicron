# Usage guide

Back to the [README](../README.md). Every flag is listed in the [CLI reference](cli-reference.md).

## What you need

- **An iPhone Portrait HEIC with embedded depth.** This is the canonical input: the depth map comes from the HEIC's LiDAR-based depth image.
- **Any other image** (JPEG, PNG, or a HEIC without depth) uses monocular depth, which estimates relative depth with Depth Anything V2 Small. `--depth-source auto` (the default) falls back to it with a note when the file has no depth image. It needs the `depth` extra and downloads the model weights on first use.
- **HEIC support** needs the `heic` extra (pillow-heif). **HDR highlight headroom** needs macOS 15 or later and the `hdr-macos` extra.

## Installing from source

The [README](../README.md#install) installs the release wheel. To work from a source checkout instead:

With conda:

```bash
git clone https://github.com/chuang44-tiff/iSummicron.git && cd iSummicron
git checkout v0.5.2
conda env create -f environment.yml && conda activate isummicron
pip install -e '.[heic,depth,hdr-macos,eval-quality,dev]'
```

With a plain virtual environment (Python 3.12):

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e '.[heic,hdr-macos]'
```

To run the tests from a source checkout:

```bash
python -m pytest tests/ -m "not slow" -p no:xdist -q    # FAST suite
ruff check summicron/
```

Tests that need sample images or developer-only assets skip themselves when those are absent. The suite imports `summicron` from the checkout, so install with `pip install -e`.

Core dependencies are numpy, scipy and pillow. Optional extras:

- `heic`: pillow-heif, for HEIC input.
- `depth`: torch and transformers, for the monocular Depth Anything V2 Small fallback.
- `hdr-macos`: pyobjc Quartz (which re-exports Core Image), darwin only.
- `eval-quality`: scikit-image.
- `gpu`: mlx, for `--engine gpu`. Apple silicon only; on other machines the extra installs nothing.
- `dev`: pytest, ruff and pyyaml.

A missing extra fails with an actionable `ImportError`.

In the conda environment, import numpy before torch in your own scripts: the conda-forge numpy/scipy and the pip torch wheel each bundle an OpenMP runtime, and importing torch first aborts with `OMP: Error #15` (see [Troubleshooting](troubleshooting.md#omp-error-15)).

Tested with: macOS 26/27 arm64, Python 3.12, torch 2.12, transformers 5.9, numpy 2.4, pyobjc 12.2.

Add `depth` to the extras of the release install for the monocular fallback. All versions are listed at https://github.com/chuang44-tiff/iSummicron/releases.

## The quick-start flags

The README's sample command spells out three settings. On a Portrait HEIC on macOS, the defaults pick the same depth and HDR settings. The size is the exception: without `--downres` a render is full size.

- `--depth-source lidar` reads LiDAR depth from the HEIC's auxiliary image (strict: it errors without one). The default is `auto`: LiDAR when the HEIC carries a depth image, otherwise monocular depth with a note. The default look also refines the depth with a guided filter and maps it through a reciprocal (diopter) curve.
- `--downres 2048` renders a 2048 px preview (long edge). There is no default; without it the render is at the photo's own size.
- `--hdr-source macos` loads the HDR gain map, so specular highlights keep their headroom through the render (strict: it needs macOS and the `hdr-macos` extra). The default is `auto` (the gain map when the macOS loader and the data are present, otherwise SDR with a note). With `--tonemap auto` the final tone follows the input photo's own SDR tone through a measured luma lookup table (`input-lut`); pass `--tonemap ci` for the Apple Core Image tonemap instead.
- Kernels are always L1-normalized. The old `--psf-norm` flag was removed in 0.5.0.

## Common flags

| Flag | Meaning |
| --- | --- |
| `--focus-xy X Y` | In-focus point, normalized. |
| `--lens {35,50}` | Bundled lens: `35` (default; 35 mm f/2, US 3,006,249, Example II) or `50` (50 mm f/2 double-Gauss, optiland sample). Not combinable with `--tensor`. |
| `--downres N` | Long edge in pixels. 2048 is the fast preview. |
| `--depth-source` | `auto` (default: LiDAR when the HEIC has a depth image, else monocular), `lidar` (strict), or `monocular` (Depth Anything V2 Small). |
| `--depth-gain` | `auto` (default: 1.0, physical) or a number in (0, 10]. Scales the diopter mapping; larger means more blur and a narrower sharp zone. |
| `--depth-scale` | Multiplier on the depth map in meters (only the linear depth mapping uses it; the default diopter mapping ignores it). |
| `--hdr-source` | `auto` (default) or `macos`: keep highlight headroom (macOS only); `none` is the 8-bit load. |
| `--bright-stamp`, `--bright-stamp-gain` | How highlights are painted. The default paints the lens shape at 2.0× the light's own energy; the gain takes a number in (0, 16]. |
| `--out-dir`, `--output` | Output directory, or one exact output file. |
| `--engine` | `reference` (default, exact), `tiled` (opt-in CPU, several times faster) or `gpu` (opt-in Apple GPU, needs the `gpu` extra). See [Render engines](#render-engines). |
| `--occlusion-compositing` | `on` stacks the depth layers front to back (off by default; see [Occlusion compositing](#occlusion-compositing)). |
| `--tensor` | Use your own PSF tensor; the format is under [Bring your own PSF tensor](#bring-your-own-psf-tensor). Not combinable with `--lens`. |

Every flag, with defaults and ranges, is in the [CLI reference](cli-reference.md). `isummicron render --help` prints the same list; `python -m summicron.render.convolve` accepts the identical set.

## Choosing the focus point

`--focus-xy X Y` takes normalized coordinates: X is the column, Y the row, `(0, 0)` is the top-left and `(1, 1)` the bottom-right. Aim at the subject's eye or face. The renderer shifts the depth map so that point lands on the tensor's in-focus slice (0.7 m for the shipped tensors).

To find the numbers, either read a pixel position in any viewer and divide by the width and height, or use the picker:

```bash
pip install matplotlib
python -m summicron.tools.anchor_picker --image iphoneImage/Demo_input.HEIC
```

Click the subject. The picker prints a report and a developer-style command; copy only its `--focus-xy` values. It opens a Tk window, so it needs a Python with Tk.

Without `--focus-xy`, the closest subject (the 2nd percentile of the depth map) is in focus.

## Choosing the lens

`--lens 35` (the default) is a 35 mm f/2, ray-traced from the expired US patent 3,006,249, Example II, using optiland. `--lens 50` is a 50 mm f/2 double-Gauss from the optiland sample prescriptions. The 35 mm gives smaller blur and clips hard in the corners; the 50 mm gives larger blur and stronger swirl toward the edges. The lens changes the blur shapes only; it does not crop or reframe the photo.

`--lens` cannot be combined with `--tensor`. [How it works](how-it-works.md) shows both sets of shapes.

## Depth

- `--depth-source auto` (default) uses the depth image embedded in a Portrait HEIC when there is one, and otherwise falls back to monocular depth with a note.
- `--depth-source lidar` is strict: it decodes the embedded depth image, and a HEIC without it fails with the error `--depth-source=lidar but image has no embedded depth aux.`.
- `--depth-source monocular` runs Depth Anything V2 Small and produces relative depth.
- A file without a depth image renders with monocular depth and the linear mapping. Results with the 0.5.0 lens tables on such files are unvalidated.
- The default look refines the depth map with a guided filter (`--depth-guided`) and maps it through a reciprocal (diopter) curve (`--depth-mapping`), which keeps far background at a believable distance instead of stretching it. The mapping is one to one by default (`--depth-gain auto` is 1.0): one diopter in the scene is one diopter for the lens. A larger `--depth-gain` gives more blur and a narrower sharp zone.
- `--depth-scale` multiplies the depth map, in meters, before the focus point is anchored. It only applies to the old linear mapping (`--depth-mapping linear`, usually with `--depth-scale 10`); with the default diopter mapping it is ignored with a warning.

Depth errors show up as wrong blur. A LiDAR depth image measures the nearest surface, so a window pane in front of a distant scene reads as near.

## Render engines

`--engine` chooses how the blur is computed. The lens kernels, the depth layers and the highlight handling are the same in all three, so the blur shapes are the same.

- `reference` (default) is the exact full-frame path. Its output is byte-identical to 0.5.1.
- `tiled` convolves in overlapping tiles on the CPU and skips tiles where a depth layer has no pixels, and runs the highlight pass only on tiles with bright pixels. Same speed class as the GPU engine at 2048 and several times faster than `reference`.
- `gpu` runs the tile convolutions on the Apple GPU with MLX. Install it by adding `gpu` to the extras of the release install command (`isummicron[heic,hdr-macos,gpu] @ <wheel URL>`, see the [README](../README.md#install)), or `pip install -e '.[gpu]'` from a source checkout. It needs Apple silicon; on other machines the extra installs nothing and `--engine gpu` stops with an error. mlx can be kept in its own virtual environment if you want the rest of your setup untouched.

```bash
isummicron render --image photo.HEIC --focus-xy 0.40 0.39 --engine tiled
```

```python
RenderOptions(image_path=..., engine="tiled")   # "reference" (default), "tiled" or "gpu"
```

Precision. `reference` is the ground truth. `tiled` and `gpu` are checked against it in linear light, with `e = |difference| / max(reference, 1)` per pixel, on real photos and standard charts, over the whole frame and the ring near the frame edge:

| | max `e` | p99.9 `e` | total energy | 8-bit output | PSNR |
| --- | --- | --- | --- | --- | --- |
| `tiled` | at most 1e-5 | at most 1e-6 | within 1e-5 | within 1 level | |
| `gpu` | at most 1e-4 | at most 1e-5 | within 1e-5 | within 1 level | at least 70 dB |

Measured worst cases on the test photos are well inside these limits (on the sample photo, `tiled` max `e` is 6.8e-7 and `gpu` 1.6e-6). The engines are checked by tolerance, not by checksum, so their files can differ from `reference` in the last bit of some pixels; the difference is not visible. On extreme inputs (a peak-normalised bright stamp on HDR, large saturated patches, lights 50 to 1000 times white, a `--bright-stamp-gain` of 8 or more) the rule is looser: the p99.9 error against a float64 calculation is at most 1.5 times (`tiled`) or 7 times (`gpu`) the reference's own float32 error. `gpu` output is deterministic on one machine for fixed settings, and its bytes can differ across GPU families.

With `--occlusion-compositing on`, both fast engines run the reference engine instead and print `note: engine tiled -> reference (--occlusion-compositing on runs on the reference engine in 0.5.2)` (or `gpu`). An engine that cannot run stops the render with an error and exit 2; see [Troubleshooting](troubleshooting.md#engine-not-available). There is no silent fallback.

## Resolution, time and memory

| Setting | Time on an Apple Silicon Mac (M4 Max) | Peak memory |
| --- | --- | --- |
| `--downres 2048`, `reference` | about 10 seconds | about 3.8 GB |
| `--downres 2048`, `tiled` or `gpu` | about 3.4 seconds | about 3.6 GB |
| native, 5712 px on the long edge, `reference` | about 100 seconds | about 11 GB |
| native, `tiled` | about 14 seconds | about 9.0 GB |
| native, `gpu` | about 13 seconds | about 9.5 GB |

Times are the whole render, on the sample photo, with other apps running, so they are rough. Use `--downres 2048` while iterating on the focus point and flags. A built-in guard estimates peak memory before rendering and stops with an error that names a `--downres` that fits. The budget is 12 GiB; change it with `ISUMMICRON_MAX_BYTES=<bytes>`.

```bash
export ISUMMICRON_MAX_BYTES=17179869184
```

## HDR

HDR matters for bokeh: a blurred light only looks like a light if the renderer knows how bright it really was. [How it works](how-it-works.md#why-hdr-matters-for-bokeh) shows the difference on the sample photo.

`--hdr-source macos` loads the HEIC's HDR gain map through Core Image and renders in linear light with the extra headroom, so bright highlights keep their punch. `--hdr-source auto` (the default) picks `macos` when the loader and a gain map are available and otherwise loads the 8-bit SDR image with a note. `--hdr-source none` always loads the SDR image.

The final tone follows the input photo's own SDR tone (`--tonemap input-lut`, chosen by `--tonemap auto` when there is HDR headroom): a luma curve measured from the input is applied to the render. `--tonemap ci` uses the Apple Core Image tonemap instead.

## Outputs

By default a render writes two files to `./runs/render`: `<stem>[_r<N>]_rendered[_ds<scale>].jpg` and a `.png`. The file name records the options used (for the quick start, `Demo_input_r2048_rendered_gd_dio1.00_tlut.jpg`; `--lens 50` adds `_lens50`). `--out-dir DIR` changes the directory. `--output FILE` writes exactly one file: PNG if it ends in `.png`, otherwise JPEG at quality 95.

## Occlusion compositing

Occlusion compositing stays off by default: even with real-lens blur it only changes a hairline along the subject's outline and is not visible at normal viewing, while costing about 30–60% more render time (1.3–1.6×); guided depth and the in-focus snap already keep subject edges clean. Use `--occlusion-compositing on` for scenes with very bright light directly behind the subject's outline.

With the option on, the depth layers are stacked front to back so that nearer layers cover farther ones.

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

Pass it with `--tensor my_tensor.npz`. The loader validates the schema and never unpickles. The two shipped tensors in `summicron/_artifacts/` are working examples: each is `(R=12 radial bins, D=11 distances, 183, 183)` float32. `scripts/build_psf/` builds new ones with optiland; the commands are under [Rebuilding the tables](how-it-works.md#rebuilding-the-tables).

## Related

- [How it works](how-it-works.md) for the lens shapes and the pipeline
- [Python API](python-api.md) for scripting
- [Troubleshooting](troubleshooting.md) when something fails
- [FAQ](faq.md)
