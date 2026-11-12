# summicron/psf_model/ -- ray-traced PSFs -> render tensor

The render tensor is built offline with optiland (PyPI package `optiland`)
by `scripts/build_psf/` (see `environment-optiland.yml`). This package holds only
the run-time-safe helpers that describe and assemble that tensor.

## Files

| File | Purpose |
|------|---------|
| `ray_kernel.py` | Tensor schema constants (object distances, radial edges, focus distance) and `assemble_tensor`, which packs traced per-(radius, distance) PSFs into the `(R=12, D=11, H, W)` render tensor with the single-pixel delta at `kernels[r, 0]`. |

## Shipped tensors

Two tensors ship in `../_artifacts/` (selected with `--lens 35|50`):

- `psf_lens35_us3006249_fu0.4.npz` -- 35 mm f/2, ray-traced from the expired
  US patent 3,006,249, Example II, using optiland (default).
- `psf_lens50_doublegauss_fu0.4.npz` -- 50 mm f/2 double-Gauss from the optiland sample prescription.

Rebuild: `scripts/build_psf/build_raytrace_tensor.py` (the module docstring lists the exact command per lens).

## Shelved measured-PSF pipeline

The measured-PSF modules (build / cherry / tensor4d / viz) and the
`harvest*` / `psf_extract` packages were removed from `main` in SHIP-3.
They are not part of this distribution.

## Related

- `../render/` -- consumes the tensor
- `../_artifacts/` -- where the wheel-shipped `.npz` tensors live
