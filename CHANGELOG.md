# Changelog

All notable changes to **iSummicron** are documented in this file. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning follows [SemVer](https://semver.org/).

## [0.5.0] — 2026-11-12

### Breaking

- **The bundled PSF tensor is replaced.** The wheel now ships two ray-traced tensors built with optiland (PyPI package `optiland`): `psf_lens35_us3006249_fu0.4.npz` (default; a 35 mm f/2 ray-traced from the expired US patent 3,006,249, Example II) and `psf_lens50_doublegauss_fu0.4.npz` (a 50 mm f/2 double-Gauss from the optiland sample prescription). The previous bundled tensor is gone. **All render output changes.**
- `--kernel-family` (any value), `--lens-aperture-mm` (any value, including the presets `35/2`, `50/2` and `apple`) and `--psf-norm` (any value) are removed. They fail with exit 2 and a one-line error that points to this note. The run-time synthetic disc family they selected is gone.
- `RenderOptions(kernel_family=..., lens_aperture_mm=..., psf_norm=...)` raises `TypeError`; use `lens=`. `RenderOptions.canonical_bench(old_look=...)` is removed.
- The auto depth gain under `--depth-mapping diopter` is now a fixed 1.0 (physical), and `--bright-stamp auto` is now `l1` (gain 2.0), including monocular inputs. The 0.4 data-dependent auto gain is not reproducible; the nearest flags are `--bright-stamp peak --depth-gain N`.

### Added

- `--lens {35,50}` (default `35`) selects the bundled tensor. `--lens` and `--tensor` are mutually exclusive.
- `--tensor PATH` takes a tensor built with the new reproducible build in `scripts/build_psf/` (environment: `environment-optiland.yml`; run it with `conda run -n optiland python scripts/build_psf/build_raytrace_tensor.py --help`). optiland is used offline only; the render path never imports it.
- The two lens prescriptions as JSON in `scripts/build_psf/lenses/` and the field-dependent PSF tracing gates in `scripts/build_psf/_gates.py`.

### Migrating to 0.5.0

| If you used | Now |
|---|---|
| `--kernel-family` (any value, or no flag) | Remove the flag. The default is `--lens 35`. |
| `--lens-aperture-mm MM` or a preset (`35/2`, `50/2`, `apple`) | Removed. The two lenses are fixed ray-traced designs; pick `--lens 35` or `--lens 50`. |
| `--tensor` pointing at the old bundled tensor file | Fails with "tensor not found". Drop the flag, or build a tensor with `scripts/build_psf/`. A copy of the old tensor kept at another path may still load, but it is not supported. |
| `--psf-norm l1` / `--psf-norm peak` | Removed (L1 is the only normalisation; `peak` rendered dark with a halo). Any use exits 2 with one line pointing here. `RenderOptions(psf_norm=...)` and `render(psf_norm=...)` raise `TypeError`. Just drop the flag. |
| `--bright-stamp auto` | Now `l1` (gain 2.0). For the old peak stamp pass `--bright-stamp peak`. |
| `--depth-gain auto` (diopter) | Now 1.0. Pass `--depth-gain N` for another value. |
| `RenderOptions(kernel_family=..., lens_aperture_mm=...)` | `RenderOptions(lens="35"|"50")`. |
| The old synthetic cache `~/Library/Caches/isummicron/synth` | Safe to delete. `ISUMMICRON_CACHE_DIR` is no longer read. |

- Pre-0.5.0 pins and renders no longer reproduce; the pins are re-taken on the new tensors.
- Images without a LiDAR depth image use monocular depth with linear mapping. Results with the new tensors on such images are unvalidated in 0.5.0.

### Notes

- Occlusion compositing stays off by default: even with real-lens blur it only changes a hairline along the subject's outline and is not visible at normal viewing, while costing about 30–60% more render time (1.3–1.6×); guided depth and the in-focus snap already keep subject edges clean. Use `--occlusion-compositing on` for scenes with very bright light directly behind the subject's outline.
- The README and `docs/` are rewritten around the new lenses, with a new `docs/how-it-works.md`. The figures and an interactive page, `isummicron_pipeline.html`, are attached to the release.

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

[0.5.0]: https://github.com/chuang44-tiff/iSummicron/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/chuang44-tiff/iSummicron/releases/tag/v0.4.0
[0.3.0]: https://github.com/chuang44-tiff/iSummicron/releases/tag/v0.3.0
[0.2.0]: https://github.com/chuang44-tiff/iSummicron/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/chuang44-tiff/iSummicron/releases/tag/v0.1.0
