# Changelog

All notable changes to **iSummicron** are documented in this file. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning follows [SemVer](https://semver.org/).

## [0.4.0] — 2026-11-12

### Changed

- **Default look changed.** The defaults now use guided depth refinement (`--depth-guided`), reciprocal-depth (diopter) mapping (`--depth-mapping`), input-LUT tone and white balance (`--tonemap input-lut`), and `--hdr-source auto` (HDR gain map when the loader and the data are present, otherwise SDR with a note). Renders differ from earlier versions. The previous look stays reachable with `--depth-guided off --depth-mapping linear --tonemap ci --bright-pass-thresh 1.0` (plus `--depth-scale 10`).
- `--depth-source` now defaults to `auto`: LiDAR when the HEIC carries a depth image, otherwise monocular depth with a note (it was `lidar`, which errors without a depth image). Pass `--depth-source lidar` for the strict behaviour.

### Added

- Experimental flags `--kernel-family {codev,synthetic}`, `--lens-aperture-mm`, `--bright-stamp` and `--bright-stamp-gain` (synthetic defocus-disc kernel family, run-time cache). These are experimental and may change or be removed in a later release. The default output is unchanged.

## [0.3.0] — 2026-11-12

### Breaking

- `--dof-m` is removed from the CLI.
- The Python `render()` function now requires the keyword-only argument `depth_m=`. The kwargs `defocus_map`, `n_bins`, `layered_gather`, `r_blend_strength` and `radial_blend_mode` are removed.
- `RenderOptions` validates its fields in `__post_init__`: a bad value raises `ValueError` at construction time.

### Output change

- The in-focus kernel is now an exact single-pixel delta, unconditionally. Previously the render-time cubic zoom smeared it at reduced resolutions. **Renders are therefore not bit-identical to 0.2.0.** The shipped PSF tensor itself is unchanged (md5 `e6f50154…`).

### Changed

- The `--out-dir` default is now `./runs/render` under the current directory (it was a directory inside the installed package).
- The default depth-slab batch size is 1 (was 4), which lowers peak RAM.
- The memory guard's estimate is now structural and conservative: about 9.4 GiB at native iPhone resolution, under the 12 GiB default. The measured footprint is about 5.4 GB, down from 7.0 GB. The 12 GiB budget can be overridden with `ISUMMICRON_MAX_BYTES=<bytes>`. Guard messages now say what to change.
- `ISUMMICRON_DEBUG=1` prints a full traceback for CLI errors.
- `isummicron render` and `python -m summicron.render.convolve` share one argument parser and report identical one-line errors.
- The `hdr-macos` extra lists only `pyobjc-framework-Quartz` (it re-exports Core Image).
- The `depth` extra is bounded: `torch>=2.4,<3`, `transformers>=4.45,<6`.

### Added

- `--occlusion-compositing {off,on}` (default `off`): a premultiplied OVER compositor that removes milky halos at depth silhouettes. It survives infinite depth values; with `--psf-norm peak` it falls back to off with a warning.
- `--hdr-source macos` works on macOS 27.2 with pyobjc 12.2.
- The test suite, the `summicron/eval` metrics and the `summicron/tools` anchor picker now ship in the repository.
- A GitHub Actions workflow runs the FAST test suite on pushes to `main`, on pull requests and on manual runs.
- `docs/` (usage guide, CLI reference, Python API, troubleshooting, FAQ) and `scripts/gen_cli_reference.py`, which generates the CLI reference from the real argument definitions.
- The README now has a before/after gallery and a section on picking a focus point.
- `NOTICE`, and `LICENSE` / `NOTICE` inside the wheel metadata.

### Documentation

- The README now states the lens correctly: a generic ~100 mm f/2 double-Gauss (CODE V sample design `ag_dblgauss`), kernels 183×183 (not 50 mm and 47×47). This corrects the text only; the tensor file is the same as in 0.2.0.
- Trademark disclaimer and prior-art citation (Lin et al.) added.


## [0.2.0] — 2026-05-06

### Added

- `--slab-batch-size N` CLI flag — process depth slabs in serial batches; lower N = lower peak RAM, no speed cost. Default 4. Pass `D_total` to collapse to all-at-once (legacy single-batch behavior).
- Image-FFT hoist inside the renderer: per-channel `rfft2` of the padded image buffers is computed once per render and reused across every slab × radial-bin × angular-wedge inner-loop iteration. Cuts FFT work by ~3×.
- Parallel pocketfft via `scipy.fft.set_workers(-1)` wrapping the inner-loop FFTs. Scales smoothly across Apple Silicon P+E core asymmetry; sustained-workload tests show no thermal throttling.

### Changed

- Streaming holes-fill replaces the `(D, H, W, 3)` per-slab output buffers with a running `(H, W, 3)` `best_d_color` + `(H, W) int32 best_d_dist` argmin updated per batch. Bit-exact to the prior global-argmin path; saves ~1× `(D, H, W, 3)` of working memory at peak.
- `_estimate_peak_bytes` formula updated from `D × H × W × 4 × 8` (pre-batching envelope) to `min(_SLAB_BATCH_SIZE, D) × H × W × 4 × 8 + 2 × H × W × 4 × 4` (post-batching). Native iPhone resolution (5712×4284) projects 8.6 GB → 3.9 GB; previously cap-blocked, now feasible.
- README focuses on the canonical workflow: Portrait HEIC + `--focus-xy` + `--hdr-source macos`. Secondary use cases (monocular depth, K11 shadow-match, custom tensors) documented under `--help`.

### Performance

Canonical bench (M-series 12-core, 18 GB RAM):

| Resolution | 0.1.0 wall | 0.2.0 wall | speedup |
| --- | ---: | ---: | ---: |
| `--downres 2048` (iteration) | 231 s (3 min 51 s) | **63 s (1 min 3 s)** | **3.65×** |
| native 5712×4284 | cap-blocked (8 GB rejection) | **486 s (8 min 6 s)** | newly-feasible |

JPG output is bit-exact at md5 level across 0.1.0 ↔ 0.2.0 on the canonical bench (so the speedup is pure compute reorganization, not a quality compromise).

## [0.1.0] — 2026-05-04

### Added

- Initial public release.
- `isummicron render` console script.
- Spatially-varying convolution path: `(R=12, D=11, kH=47, kW=47)` 4D PSF tensor per pixel + 8 angular wedges.
- Wheel-shipped simulated double-gauss 50mm f/2 PSF tensor (`summicron/_artifacts/codev_render_v1.npz`).
- LiDAR depth extraction from iPhone Portrait HEIC auxiliary image.
- Monocular depth fallback via Depth Anything V2 (`--depth-source monocular`).
- macOS HDR tonemap via Apple Core Image (`--hdr-source macos`).
- Tap-to-focus via `--focus-xy X Y` (normalized image coordinates).
- Bring-your-own-tensor format documented in README.
- Memory guard rejects pathological renders with actionable `--downres N` suggestion.

[0.4.0]: https://github.com/chuang44-tiff/iSummicron/releases/tag/v0.4.0
[0.3.0]: https://github.com/chuang44-tiff/iSummicron/releases/tag/v0.3.0
[0.2.0]: https://github.com/chuang44-tiff/iSummicron/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/chuang44-tiff/iSummicron/releases/tag/v0.1.0
