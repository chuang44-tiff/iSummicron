# summicron/render/ -- Summicron PSF Renderer

Apply ray-traced 35 mm f/2 point-spread functions (US patent 3,006,249,
Example II, via optiland) to any input photo. Center stays sharp; edges and
corners pick up the blur character of the traced lens. `--lens 50` selects the
50 mm f/2 double-Gauss instead.

## Quick start

```bash
# Render an iPhone photo (HEIC, JPEG, or PNG). --lens defaults to 35 (wheel-shipped
# summicron/_artifacts/psf_lens35_us3006249_fu0.4.npz).
isummicron render --image ~/Photos/IMG_1234.HEIC --out-dir ./runs/render
```

Output lands in `runs/render/` as both JPEG (q95) and PNG.

## Pipeline overview

1. **Tensor build** (`scripts/build_psf/`, offline, optiland)
   -- ray-traces per-distance, per-radius PSFs into the
   `(R=12, D=11, H, W)` render tensors shipped in `../_artifacts/`.
2. **Render** (`convolve.py`) -- loads the tensor and an input image, builds
   a defocus map (synthetic radial quadratic, depth-map, or `depth_m` in
   meters from LiDAR/DA model), partitions the image into 8 defocus bins x
   12 radial slabs x 8 angular wedges, FFT-convolves each partition with a
   rotated kernel (P1.0), composites via cos^2 slab taper with partition-of-unity
   normalization (P1.1), and applies mask weights after convolution to avoid
   ghost-copy artifacts on extended objects (P1.3).

## CLI commands

```bash
# Render (CLI subcommand; also `python -m summicron.render.convolve`)
isummicron render --image INPUT --out-dir ./runs/render/

# Rebuild a tensor (build-time tooling, needs the optiland env)
conda run -n optiland python scripts/build_psf/build_raytrace_tensor.py --help
```

## Direct Python usage

```python
from summicron.render._artifacts import default_tensor_path
from summicron.render.convolve import load_image, load_tensor, render, save_results
from pathlib import Path

image_srgb = load_image("photo.jpg")        # float32 [0,1] sRGB
tensor = load_tensor(default_tensor_path())  # wheel-shipped canonical tensor

# depth_m (meters, (H, W)) is REQUIRED and keyword-only. The tensor must
# carry object_distances_m metadata (the wheel-shipped canonical one does).
import numpy as np
H, W = image_srgb.shape[:2]
depth_meters = np.full((H, W), 2.0, dtype=np.float32)   # 2m flat scene
output_u8, output_linear = render(image_srgb, tensor, depth_m=depth_meters)

save_results(output_u8, "photo", Path("./output"))
```

### Typed API and probes

`RenderOptions.canonical_bench(image_path, focus_xy)` is the blessed
in-process entry for probes: it builds the exact canonical bench options
(same flags as `runs/vibe_bench/README.md`). Pass overrides as kwargs, e.g.
`return_linear=True` for linear-domain probes and `save=False` to skip
writing to disk:

```python
from summicron.render.api import RenderOptions, render_image
opts = RenderOptions.canonical_bench("IMG_6915.HEIC", (0.40, 0.45),
                                     return_linear=True, save=False)
result = render_image(opts)   # result.output_linear / result.output_u8
```

Since Run 6 the canonical bench is the new default look (guided LiDAR depth,
diopter mapping, input-lut tone). `old_look=True` builds the pre-flip look
(`CANONICAL_FLAGS_OLD`: linear depth x10, Core Image tone):

```python
opts = RenderOptions.canonical_bench("IMG_6915.HEIC", (0.40, 0.45),
                                     old_look=True, return_linear=True, save=False)
```

`RenderResult.diagnostics.resolved` reports what the `auto` modes resolved to.

When `--out-dir` is omitted, output goes to `./runs/render` under the
current working directory.

The renderer works in linear light internally (sRGB decode on input, re-encode
on output) and uses `scipy.signal.fftconvolve` per depth slab.

## What the output looks like

- **Center** stays sharp -- the defocus map is 0 at image center, so the
  kernel there is a near-delta (identity).
- **Edges and corners** pick up progressive blur shaped by the ray-traced PSF kernels
  (optiland). The blur is not a
  uniform Gaussian; it carries the asymmetric coma character of the traced lens,
  with tails pointing radially outward (P1.0 kernel rotation).
- Transitions are smooth thanks to triangular-weight blending across 8 defocus
  bins combined with cos^2 slab taper across 12 radial slabs (P1.1), both with
  partition-of-unity normalization.

## Tensor evolution

**Current (0.5.0):** ray-traced (optiland) PSF tensors, shape
`kernels(R, D, H, W)` = `(12, 11, 183, 183)` float32 at iPhone pixel pitch
(6.30 µm/px, `corpus_ref_long_edge=5712`). RAW (not OTF-normalized);
center `(r=0, d=0)` is an exact single-pixel delta. Single focus
distance (0.70 m). Tap-to-focus uses a wrap-around hack for near-focus.

**Note:** an earlier measured-PSF tensor `(6, 4, 129, 129)` came from the
removed measured-PSF pipeline (recoverable from tag `shelved-corpus-pipeline-v1`).
The live tensors are the two optiland ones in `_artifacts/`.

**Future — multi-focus-distance bank (P1.6):** 5 tensors at different
focus distances (0.7 m, 1.5 m, 3 m, 5 m, infinity). Tap-to-focus picks the
nearest tensor; wrap-around hack goes away. Each tensor is
`(12, 11, 183, 183)` at ship pitch. See (dev-only planning doc, not shipped)
for the iPhone on-device plan (~125 MB bank at fp16).
(Note: P1.3 is the FFT convolve-then-blend fix, not this item. Multi-distance
bank was reprioritized to P1.6 after P1.2 distance-indexed renderer shipped.)

## Pixel-pitch kernel scaling (Day 6)

The renderer auto-scales PSF kernels to match the input image resolution.
The 4D tensor stores `corpus_ref_long_edge`. The live tensors use
5712 (iPhone long edge); the legacy corpus tensor used 5976 (M10 long edge).
At render time, `pix_scale = max(H, W) / corpus_ref_long_edge` shrinks the
183x183 (shipped) or 129x129 (legacy) kernels to the correct size for the working
image (e.g. ~44x44 at 2048px). The fallback default of 5976 fires a `load_tensor`
warning when the tensor lacks the `corpus_ref_long_edge` key. The FFT pad size
is derived dynamically from the scaled kernel rather than hardcoded.

## Known limitations

- **Simulated PSFs.** Kernels are ray-traced design prescriptions, not measured from a
  physical lens.
- **No LoCA (longitudinal chromatic aberration).** The same grayscale kernel
  is applied to R, G, and B identically. The green/magenta fringing
  Summicron shows at wide apertures is not modeled.
- **Per-pixel kernel rotation (P1.0, shipped 2026-04-15).** Each radial slab is
  partitioned into N_ANG=8 angular wedges (45° each). The kernel for each wedge
  is rotated by `-(theta + 90°)` from the tensor's native UP direction to follow
  the pixel's azimuth. Delta kernels (peak > 0.5 of L1) skip
  rotation; all 8 angular masks collapse into one FFT (angular merge, P1.1).
  Render time vs. pre-rotation: ~6× at 5712 px (334s vs 55s); working res
  (~2000×3000) is ~28s.
- **Slab-boundary ring artifact (fixed P1.1, shipped 2026-04-15).** Hard
  binary slab masks produced brightness/sharpness steps at each radial boundary.
  Replaced with cos^2 taper (half-width 0.5 slab-widths) with partition-of-unity
  normalization. A small residual seam (~38× 2nd-deriv vs median, pre-existing)
  remains at slab edges; tracked for P2 per-pixel interpolation.
- **Convolve-then-blend order (fixed P1.3, shipped 2026-04-16).** Prior
  architecture multiplied the mask into the image before FFT, cutting extended
  objects along slab/wedge boundaries and producing ghost copies ("double oval"
  artifact on bokeh discs). Fixed by applying all masks after convolution.
  See (dev-only planning doc, not shipped).
- **Off-axis in-focus Wiener artifacts (P0.1 F1).** The v3 `[r≥1, d=0]`
  Wiener-normalized kernels have peaks at the 183² patch edge rather than
  center. These are now selected for off-axis pixels on flat-defocus fixtures
  after P0.2. A `RuntimeWarning` fires when a flat-zero defocus map is
  detected. Fix tracked as P0.1 follow-up F1.
- **Memory guard.** Images above 36 MP (6000x6000) are rejected to keep
  RAM usage reasonable on the reference NumPy path.
