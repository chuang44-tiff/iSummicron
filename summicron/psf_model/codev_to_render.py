"""Convert CodeV raw PSF .npz files to a render-ready 4D tensor.

Reads 11 per-distance .npz files from the ag_dblgauss pipeline,
extracts the green-channel (w1, 550 nm) PSFs for all 12 radial bins
(including on-axis r00), L1-normalizes each kernel, and writes a
single render-ready .npz compatible with
``summicron.render.convolve.load_tensor()``.

Per locked-spec-otf-norm-v3.md, this step emits a RAW (L1-normalized
only) tensor at the native 2048 grid. OTF-normalization is applied
downstream by ``downscale_tensor`` at the target iPhone pixel pitch.
P1.10.2 Cut 2: the ``otf_norm_applied`` flag is no longer written
(OTF path removed in P1.9; load_tensor no longer reads it).

P1.9.4 (L0): the raw is **R=12 PSFs** sampled at ``radial_bin_centers_mm``
(``psf_r00..psf_r11``). The synthetic midpoint inserted at line ~120 is a
**bin BOUNDARY**, NOT a kernel sample — R-axis interpolation operates on
the 12 raw r-centers, and the midpoint edge is re-derived AFTER interp
from the new (interpolated) r-center positions.

P1.9.4 R5 PIVOT (2026-04-25, user-authorized post-probe-fail): R/D
interpolation has MOVED to ``downscale_tensor.py`` so it operates on the
~44×44 downscaled kernels rather than the 2048×2048 raw grid (~2000×
cheaper compute, identical algorithmic content). This module is
restored to P1.9.3-equivalent simplicity: produce the raw tensor only;
all densification happens downstream.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = ROOT / "runs" / "ag_dblgauss"
DEFAULT_OUTPUT = DEFAULT_INPUT_DIR / "ag_dblgauss_f2.0_render.npz"

# P1.9.3: R and D are derived at runtime from the CodeV input files
# (D = len(files); R = len(radial_edges) - 1 post-midpoint-split).
# Module-level R=12 / D=11 literals were removed so future CodeV runs
# with different bin counts Just Work — the file count and the
# radial_bin_edges_hmax length are the upstream truth.


def _compute_corpus_ref(pixel_pitch_um: np.ndarray) -> int:
    """Compute corpus_ref_long_edge from (R,D) pixel_pitch_um array.

    ``corpus_ref_long_edge`` is a scalar derived from the mean over
    (R, D) ``pixel_pitch_um``. Future rounds that change R or D MUST
    re-compute this value from the new tensor — do not pin from an old
    tensor's metadata. The mean already auto-recomputes from the derived
    ``pixel_pitch_um`` array when shape changes.

    ``corpus_ref_long_edge`` is the image long-edge (in iPhone pixels) at
    which kernel pitch equals one corpus pixel, assuming a 36x24 mm
    full-frame sensor. Uses the scalar mean over (R, D) bins (pitch
    varies ~10% across bins; scalar is acceptable for v0).

    Parameters
    ----------
    pixel_pitch_um : np.ndarray, shape (R, D), float64

    Returns
    -------
    int
    """
    return int(round(36000.0 / float(np.asarray(pixel_pitch_um).mean())))


def convert(
    input_dir: Path = DEFAULT_INPUT_DIR,
    out_path: Path = DEFAULT_OUTPUT,
    wavelength_idx: int = 1,
) -> Path:
    """Convert CodeV raw PSF .npz files to render-ready raw tensor.

    Output tensor has the original CodeV (R, D) shape — typically (12, 11).
    R/D densification is performed downstream by
    ``summicron.psf_model.downscale_tensor`` at the iPhone pixel pitch
    (R5 pivot, 2026-04-25).
    """
    input_dir, out_path = Path(input_dir), Path(out_path)

    # --- Load & sort files by distance_index ---
    # P1.9.3: D is derived from the file count (the upstream truth).
    files = sorted(input_dir.glob("ag_dblgauss_f2.0_focus*_d*.npz"))
    if len(files) == 0:
        raise RuntimeError(f"No input files found in {input_dir}")
    D = len(files)

    datasets = []
    for f in files:
        d = np.load(str(f), allow_pickle=True)
        datasets.append((int(d["distance_index"]), d))
    datasets.sort(key=lambda x: x[0])

    # --- Radial edges (identical across files) ---
    # Raw has N+1 entries: [0.0, e_1, ..., 1.0] defining N off-axis bins.
    # We insert a synthetic midpoint edge to create the on-axis bin boundary,
    # yielding N+2 edges for N+1 bins (R = N+1).
    # P1.9.4 L0: any R-axis interp must operate on the raw R=12 PSFs at
    # radial_bin_centers_mm (NOT the post-midpoint edges). The synthetic
    # midpoint is re-derived AFTER interp from the new centers — see
    # downscale_tensor._interp_rd for the post-pivot location.
    raw_edges = datasets[0][1]["radial_bin_edges_hmax"].astype(np.float64)
    midpoint = (raw_edges[0] + raw_edges[1]) / 2.0
    radial_edges = np.concatenate([[raw_edges[0], midpoint], raw_edges[1:]])
    # P1.9.3: R is derived from radial_edges length post-midpoint-split.
    R = len(radial_edges) - 1
    assert np.all(np.diff(radial_edges) > 0), "radial_edges not monotonically increasing"

    # --- Extract metadata from first file ---
    first_data = datasets[0][1]
    focus_distance_m = float(first_data["focus_distance_m"])
    h_max_mm = float(first_data["h_max_mm"])
    wavelengths_nm = first_data["wavelengths_nm"]
    wavelength_nm = float(wavelengths_nm[wavelength_idx])

    # P1.9.4 L23 NOTE: ``radial_bin_centers_mm`` (length R, kernel-sample
    # positions) is NOT serialized into the raw tensor — the committed raw
    # tensor md5 lock pre-dates this field. ``downscale_tensor`` reads it
    # directly from the CodeV ``.npz`` files when R/D interp is requested
    # (see ``downscale_tensor._read_radial_centers_from_codev_dir``).

    # --- Object distances (from raw data, sorted) ---
    object_distances_m = first_data["object_distances_m"].astype(np.float64)
    assert len(object_distances_m) == D

    # --- Build kernels tensor + pixel_pitch_um ---
    first_psf = datasets[0][1][f"psf_r00_w{wavelength_idx}"]
    H, W = first_psf.shape
    kernels = np.zeros((R, D, H, W), dtype=np.float32)
    pixel_pitch_um = np.zeros((R, D), dtype=np.float64)

    for di, (_didx, data) in enumerate(datasets):
        raw_pitch = data["pixel_pitch_um"]  # shape [R, 3]
        for ri in range(R):
            key = f"psf_r{ri:02d}_w{wavelength_idx}"
            k = data[key].astype(np.float32)
            assert k.sum() > 0, f"Kernel sum <= 0 for {key} in distance index {_didx}"
            # L1 normalize
            k /= k.sum()
            assert abs(float(k.sum()) - 1.0) < 1e-6, f"Post-norm sum != 1.0 for {key}"
            kernels[ri, di] = k
            pixel_pitch_um[ri, di] = raw_pitch[ri, wavelength_idx]

    # Tensor shape/metadata sanity.
    assert kernels.shape == (R, D, H, W)
    assert pixel_pitch_um.shape == (R, D)
    assert np.all(pixel_pitch_um > 0), "pixel_pitch_um has non-positive values"

    # --- Full-frame dimensions ---
    # [W_px, H_px] for a 36x24mm sensor at each kernel's pixel pitch
    full_frame_dim_px = np.zeros((R, D, 2), dtype=np.int32)
    full_frame_dim_px[:, :, 0] = np.round(36000.0 / pixel_pitch_um).astype(np.int32)
    full_frame_dim_px[:, :, 1] = np.round(24000.0 / pixel_pitch_um).astype(np.int32)

    # --- Defocus ratios (L16 — linspace, axis label not physical) ---
    defocus_ratios = np.linspace(0.0, 1.0, D, dtype=np.float64)

    # --- Corpus reference long edge (for kernel pixel-scale computation) ---
    corpus_ref_long_edge = _compute_corpus_ref(pixel_pitch_um)

    # --- Write ---
    # Per v3 spec S2/S4: this is the raw (L1-only) tensor; OTF-norm is
    # deferred to downscale_tensor. Flag is False.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        str(out_path),
        kernels=kernels,
        radial_edges=radial_edges,
        defocus_ratios=defocus_ratios,
        pixel_pitch_um=pixel_pitch_um,
        full_frame_dim_px=full_frame_dim_px,
        object_distances_m=object_distances_m,
        focus_distance_m=np.float64(focus_distance_m),
        h_max_mm=np.float64(h_max_mm),
        wavelength_nm=np.float64(wavelength_nm),
        corpus_ref_long_edge=np.int64(corpus_ref_long_edge),
    )
    print(f"Wrote render-ready tensor {kernels.shape} -> {out_path}")
    return out_path


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(
        description="Convert CodeV raw PSF .npz files to render-ready tensor.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing ag_dblgauss_f2.0_focus*_d*.npz files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output .npz path (default: ag_dblgauss_f2.0_render.npz).",
    )
    args = parser.parse_args(argv)
    return convert(input_dir=args.input_dir, out_path=args.output)


if __name__ == "__main__":
    main()
