# summicron/psf_model/ — CodeV PSFs -> render tensor

Build-time tooling that turns the CodeV-simulated PSF files into the 4D
tensor that `../render/` consumes (shipped as
`../_artifacts/codev_render_v1.npz`).

## Files

| File | Purpose |
|------|---------|
| `codev_to_render.py` | Reads the 11 per-distance CodeV `.npz` files, extracts the green-channel (550 nm) PSFs, L1-normalizes, and emits a RAW 4D tensor `(R=12, D=11, H, W)`. |
| `downscale_tensor.py` | Downscales that tensor to iPhone pixel pitch, densifies R/D (R-cubic + D-linear in `log1p(PSF x 1e4)` space), and installs the S2 delta guard at every `kernels[r, 0]`. |

## Usage

```bash
conda run -n isummicron python -m summicron.psf_model.codev_to_render --help
conda run -n isummicron python -m summicron.psf_model.downscale_tensor --help
```

The outputs are the raw `(R, D, H, W)` render tensor and the downscaled, delta-guarded tensor shipped as `_artifacts/codev_render_v1.npz`.

## Shelved corpus pipeline

The measured-PSF modules (build / cherry / tensor4d / viz) and the
`harvest*` / `psf_extract` packages were removed from `main` in SHIP-3.
They are not part of this distribution.

## Related

- `../render/` — consumes the tensor
- `../_artifacts/` — where the wheel-shipped `.npz` tensor lives (`codev_render_v1.npz`)
