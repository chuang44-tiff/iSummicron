# Changelog

All notable changes to **iSummicron** are documented in this file. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning follows [SemVer](https://semver.org/).

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

Canonical bench (`IMG_6915.HEIC`, M-series 12-core, 18 GB RAM):

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

[0.2.0]: https://github.com/<user>/iSummicron/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/<user>/iSummicron/releases/tag/v0.1.0
