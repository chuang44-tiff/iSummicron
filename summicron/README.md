# summicron/ -- Python package

Source code for the iSummicron renderer. Import name `summicron`, distribution and CLI `isummicron` (entry point `cli.py:main`).

The shipped lens is a CODE V-simulated generic ~100 mm f/2 double-Gauss (sample design `ag_dblgauss`), a Summicron-style proxy and not a measured Leica lens.

| Directory | Contents |
|---|---|
| `render/` | The renderer, the shared CLI argument definitions and the `RenderOptions` / `render_image` API. |
| `depth/` | LiDAR depth decode from Portrait HEIC, the Depth Anything V2 fallback, tap-to-focus. |
| `io/` | macOS HDR load and tonemap (Core Image). |
| `psf_model/` | Tools that turn CODE V PSFs into the render tensor. |
| `eval/` | Image-quality metrics. |
| `tools/` | The focus-point picker (`anchor_picker`). |
| `_artifacts/` | The shipped PSF tensor, `codev_render_v1.npz`. |

See the repository README for installation and usage.
