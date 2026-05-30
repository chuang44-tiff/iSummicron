# iSummicron

**Spatially-varying PSF bokeh renderer.** Apply a 4D PSF tensor to an iPhone Portrait HEIC and get organic, position-dependent, naturally-transitioning bokeh — the kind real lenses produce, not the symmetric Gaussian blur most software ships.

Ships with a simulated double-gauss 50mm f/2 PSF tensor so you can clone, run, and render. Bring your own measured or simulated PSF kernels for a different lens character.

## Install

```bash
git clone <this-repo>
cd isummicron
conda env create -f environment.yml
conda activate isummicron
pip install -e .                # puts the `isummicron` console script on PATH
```

One env file installs the renderer (numpy/scipy/pillow + pillow-heif for HEIC), monocular depth (torch + Depth Anything V2), and macOS HDR (pyobjc Quartz on darwin). macOS-only for the canonical HDR path.

## Render

The canonical command — Portrait HEIC + click-to-focus + macOS HDR tonemap:

```bash
isummicron render \
    --image photo.HEIC \
    --focus-xy 0.5 0.7 \
    --hdr-source macos \
    --output bokeh.jpg
```

That's the workflow this project is built around: an iPhone Portrait HEIC carries the LiDAR depth map natively, you tap a normalized image coordinate to set the in-focus subject, and the HDR path tonemaps highlights through Apple Core Image so streetlights and specular bokeh balls render with real-lens roll-off rather than clipped white discs.

`--focus-xy X Y` is the load-bearing flag. `(0.0, 0.0)` is top-left, `(1.0, 1.0)` is bottom-right; tap your subject (eye, face center, the part you want sharp). LiDAR depth is automatically extracted from the HEIC's auxiliary image.

`isummicron render --help` lists every flag (custom tensor, depth scale, monocular fallback for non-HEIC images, K11 shadow-match for color-temperature-correct shadow lift, slab-batching for tight-RAM machines).

## How it works

1. **Load** image + LiDAR depth from the HEIC's auxiliary image.
2. **Anchor** the depth map so the `--focus-xy` pixel lands on the in-focus tensor slice.
3. **Render**: spatially-varying convolution with the 4D PSF tensor `(R radial bins, D defocus steps, kH, kW)`. Each pixel picks its kernel by radial position (center is sharper, corners get coma + cat's-eye) and local depth (in-focus vs blurred). 8 angular wedges add radially-outward kernel rotation for natural bokeh shape transitions.
4. **HDR tonemap** the linear-light output through Apple Core Image's tone-mapping filter — preserves highlight headroom that LDR-only paths crush to white.

## Bring your own PSF tensor

Write a 4D tensor as an `.npz` with these required keys:

| Key             | Shape           | Notes                                             |
| --------------- | --------------- | ------------------------------------------------- |
| `kernels`       | `(R, D, H, W)`  | float32; H == W; odd-sided; non-negative          |
| `radial_edges`  | `(R+1,)`        | float64; strictly increasing in `[0, 1]`          |
| `defocus_ratios`| `(D,)`          | float64; strictly increasing in `[0, 1]`          |

Optional keys (sensible defaults if absent):

| Key                    | Shape | Notes                                                        |
| ---------------------- | ----- | ------------------------------------------------------------ |
| `object_distances_m`   | `(D,)`| Per-pixel slab lookup for LiDAR-depth renders                |
| `focus_distance_m`     | scalar| In-focus distance; must equal `object_distances_m[0]`        |
| `corpus_ref_long_edge` | int   | Reference long-edge px for kernel pitch (default 5976)       |
| `schema_version`       | str   | `"1.0"` to silence the legacy-format warning                 |

The shipped tensor `summicron/_artifacts/codev_render_v1.npz` is a working example: `R=12, D=11, kernels (12, 11, 47, 47)`, simulated from a double-gauss 50mm f/2 design.

## Performance

On Apple Silicon (12-core M-series, 18 GB RAM):

| Resolution | Wall-clock | Peak RSS |
| --- | ---: | ---: |
| `--downres 2048` (iteration) | ~1 min | 1.5 GB |
| Native (5712×4284, no flag) | ~8 min | 5 GB |

Render path is parallel CPU FFT (`scipy.fft` with `set_workers(-1)` on pocketfft). Slab-batching processes depth slabs in serial chunks of 4 by default so peak memory doesn't scale with the number of defocus levels — pass `--slab-batch-size N` to tune for tighter RAM envelopes.

## Status

Alpha. The canonical HDR + anchor path is the supported workflow. Other depth backends (monocular Depth Anything V2) and tonemap modes work but are secondary; check `--help` for flags.

## License

Apache-2.0.
