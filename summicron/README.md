# summicron/ -- Python package

Source code for the iSummicron renderer. Import name `summicron`, distribution and CLI `isummicron` (entry point `cli.py:main`).

The default lens (`--lens 35`) is a 35 mm f/2 ray-traced from the expired US patent 3,006,249, Example II, using optiland; `--lens 50` is a 50 mm f/2 double-Gauss. Both are traced design prescriptions, not measured Leica lenses.

| Directory | Contents |
|---|---|
| `render/` | The renderer, the shared CLI argument definitions and the `RenderOptions` / `render_image` API. |
| `depth/` | LiDAR depth decode from Portrait HEIC, the Depth Anything V2 fallback, tap-to-focus. |
| `io/` | macOS HDR load and tonemap (Core Image). |
| `psf_model/` | Ray-traced PSFs to render tensor (`ray_kernel.py`); the build is `scripts/build_psf/`. |
| `eval/` | Image-quality metrics. |
| `tools/` | The focus-point picker (`anchor_picker`). |
| `_artifacts/` | The shipped PSF tensors, `psf_lens35_us3006249_fu0.4.npz` and `psf_lens50_doublegauss_fu0.4.npz`. |

See the repository README for installation and usage.
