# Usage guide

Back to the [README](../README.md). Every flag is listed in the [CLI reference](cli-reference.md).

## What you need

- **An iPhone Portrait HEIC with embedded depth.** This is the canonical input: the depth map comes from the HEIC's LiDAR-based depth image.
- **Any other image** (JPEG, PNG, or a HEIC without depth) uses monocular depth, which estimates relative depth with Depth Anything V2 Small. `--depth-source auto` (the default) falls back to it with a note when the file has no depth image. It needs the `depth` extra and downloads the model weights on first use.
- **HEIC support** needs the `heic` extra (pillow-heif). **HDR highlight headroom** needs macOS 15 or later and the `hdr-macos` extra.

## Choosing the focus point

`--focus-xy X Y` takes normalized coordinates: X is the column, Y the row, `(0, 0)` is the top-left and `(1, 1)` the bottom-right. Aim at the subject's eye or face. The renderer shifts the depth map so that point lands on the tensor's in-focus slice (0.7 m for the shipped tensor).

To find the numbers, either read a pixel position in any viewer and divide by the width and height, or use the picker:

```bash
pip install matplotlib
python -m summicron.tools.anchor_picker --image iphoneImage/Demo_input.HEIC
```

Click the subject. The picker prints a report and a developer-style command; copy only its `--focus-xy` values. It opens a Tk window, so it needs a Python with Tk.

Without `--focus-xy`, the closest subject (the 2nd percentile of the depth map) is in focus.

## Depth

- `--depth-source auto` (default) uses the depth image embedded in a Portrait HEIC when there is one, and otherwise falls back to monocular depth with a note.
- `--depth-source lidar` is strict: it decodes the embedded depth image, and a HEIC without it fails with the error `--depth-source=lidar but image has no embedded depth aux.`.
- `--depth-source monocular` runs Depth Anything V2 Small and produces relative depth.
- The default look refines the depth map with a guided filter (`--depth-guided`) and maps it through a reciprocal (diopter) curve (`--depth-mapping`), which keeps far background at a believable distance instead of stretching it.
- `--depth-scale` multiplies the depth map, in meters, before the focus point is anchored. It only applies to the old linear mapping (`--depth-mapping linear`, usually with `--depth-scale 10`); with the default diopter mapping it is ignored with a warning.

The earlier look (linear depth, Core Image tonemap) stays reachable with `--depth-guided off --depth-mapping linear --tonemap ci --bright-pass-thresh 1.0 --depth-scale 10`.

Depth errors show up as wrong blur. A LiDAR depth image measures the nearest surface, so a window pane in front of a distant scene reads as near.

## Resolution, time and memory

| Setting | Time on an Apple Silicon Mac (dev machine) | Memory |
| --- | --- | --- |
| `--downres 2048` | about 35 s | not measured |
| native 5712×4284 | about 5–6 minutes | about 5.4 GB measured (guard estimate about 9.4 GiB) |

Use `--downres 2048` while iterating on the focus point and flags. A built-in guard estimates peak memory before rendering and stops with an error that names a `--downres` that fits. The budget is 12 GiB; change it with `ISUMMICRON_MAX_BYTES=<bytes>`.

```bash
export ISUMMICRON_MAX_BYTES=17179869184
```

## HDR

`--hdr-source macos` loads the HEIC's HDR gain map through Core Image and renders in linear light with the extra headroom, so bright highlights keep their punch. `--hdr-source auto` (the default) picks `macos` when the loader and a gain map are available and otherwise loads the 8-bit SDR image with a note. `--hdr-source none` always loads the SDR image.

The final tone follows the input photo's own SDR tone (`--tonemap input-lut`, chosen by `--tonemap auto` when there is HDR headroom): a luma curve measured from the input is applied to the render. `--tonemap ci` uses the Apple Core Image tonemap instead.

## Outputs

By default a render writes two files to `./runs/render`: `<stem>[_r<N>]_rendered[_ds<scale>].jpg` and a `.png`. The file name records the options used (for the quick start, `Demo_input_r2048_rendered_gd_dio1.12_tlut.jpg`). `--out-dir DIR` changes the directory. `--output FILE` writes exactly one file: PNG if it ends in `.png`, otherwise JPEG at quality 95.

## Occlusion compositing

`--occlusion-compositing on` swaps the default depth-blended gather for a layered compositor that removes the milky halo at depth edges, such as a head against a distant background. It is off by default and costs about 1.4–1.6× the render time. With `--psf-norm peak` it falls back to `off` with a warning.

## Related

- [Python API](python-api.md) for scripting
- [Troubleshooting](troubleshooting.md) when something fails
- [FAQ](faq.md)
