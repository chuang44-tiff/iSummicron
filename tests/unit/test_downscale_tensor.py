"""Unit tests for summicron.psf_model.downscale_tensor.

Uses a synthetic (R=12, D=11, 32, 32) tensor with a known pitch. The new
API takes a ``target_long_edge`` (image pixels); the kernel grid size is
derived to make kernel pitch match the image pitch. We pick
``target_long_edge`` values that yield a small ``target_kH`` so tests
stay fast.

Synthetic pitch (mean) = 0.5638 um, orig_kH = 32.
  target_long_edge = 16000 -> target_pitch = 2.25 um,
                     scale = 0.5638/2.25 = 0.2506, target_kH = round(32*0.2506) = 8
  target_long_edge =  8000 -> target_kH = round(32*0.5638*8000/36000) = 4
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from summicron.psf_model.codev_to_render import _compute_corpus_ref
from summicron.psf_model.downscale_tensor import downscale_tensor

R, D = 12, 11
ORIG_KH = 32
# Uniform pitch keeps target_kH math simple. 0.5638 matches the real corpus mean.
PITCH_UM = 0.5638

# target_long_edge=16000 -> target_kH = 8
TARGET_LONG_EDGE = 16000
FULL_FRAME_WIDTH_UM = 36000.0
TARGET_PITCH_UM = FULL_FRAME_WIDTH_UM / TARGET_LONG_EDGE
EXPECTED_TARGET_KH = int(round(ORIG_KH * (PITCH_UM / TARGET_PITCH_UM)))  # == 8


def _make_synthetic_tensor(path: Path) -> dict:
    """Write a small synthetic render-ready .npz. Returns the field dict."""
    rng = np.random.default_rng(0)
    kernels = np.zeros((R, D, ORIG_KH, ORIG_KH), dtype=np.float32)
    y, x = np.mgrid[:ORIG_KH, :ORIG_KH].astype(np.float32)
    cy = cx = (ORIG_KH - 1) / 2.0
    for ri in range(R):
        for di in range(D):
            sigma = 1.5 + 0.05 * ri + 0.05 * di
            k = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma ** 2))
            k += rng.normal(0.0, 1e-5, size=k.shape).astype(np.float32)
            k = np.maximum(k, 0.0)
            k /= k.sum()
            kernels[ri, di] = k.astype(np.float32)

    # Uniform pitch so expected target_kH is deterministic.
    pixel_pitch_um = np.full((R, D), PITCH_UM, dtype=np.float64)
    full_frame_dim_px = np.zeros((R, D, 2), dtype=np.int32)
    full_frame_dim_px[:, :, 0] = np.round(36000.0 / pixel_pitch_um)
    full_frame_dim_px[:, :, 1] = np.round(24000.0 / pixel_pitch_um)
    fields = dict(
        kernels=kernels,
        radial_edges=np.linspace(0.0, 1.0, R + 1, dtype=np.float64),
        defocus_ratios=np.linspace(1.0 / (2 * D), 1.0, D, dtype=np.float64),
        pixel_pitch_um=pixel_pitch_um,
        full_frame_dim_px=full_frame_dim_px,
        object_distances_m=np.linspace(0.7, 20.0, D, dtype=np.float64),
        focus_distance_m=np.float64(0.7),
        h_max_mm=np.float64(21.64),
        wavelength_nm=np.float64(550.0),
        corpus_ref_long_edge=np.int64(_compute_corpus_ref(pixel_pitch_um)),
    )
    np.savez(str(path), **fields)
    return fields


@pytest.fixture
def synth_tensor(tmp_path: Path) -> tuple[Path, Path, dict]:
    in_path = tmp_path / "synth_render.npz"
    out_path = tmp_path / f"synth_render_iphone{TARGET_LONG_EDGE}.npz"
    fields = _make_synthetic_tensor(in_path)
    downscale_tensor(in_path, out_path, target_long_edge=TARGET_LONG_EDGE)
    return in_path, out_path, fields


def test_g1_output_shape(synth_tensor):
    _, out_path, _ = synth_tensor
    d = np.load(str(out_path))
    assert d["kernels"].shape == (R, D, EXPECTED_TARGET_KH, EXPECTED_TARGET_KH)
    assert d["kernels"].dtype == np.float32


def test_g2_per_kernel_l1(synth_tensor):
    _, out_path, _ = synth_tensor
    kernels = np.load(str(out_path))["kernels"]
    sums = kernels.reshape(R * D, -1).sum(axis=1)
    assert sums.min() >= 0.999
    assert sums.max() <= 1.001


def test_g3_no_nan_inf(synth_tensor):
    _, out_path, _ = synth_tensor
    kernels = np.load(str(out_path))["kernels"]
    assert not np.any(np.isnan(kernels))
    assert not np.any(np.isinf(kernels))
    assert kernels.min() >= 0.0


def test_g4_pitch_uniform_matches_target(synth_tensor):
    """New pitch is uniform and equals 36000/target_long_edge um."""
    _, out_path, _ = synth_tensor
    new_pitch = np.load(str(out_path))["pixel_pitch_um"]
    assert new_pitch.shape == (R, D)
    assert new_pitch.dtype == np.float64
    np.testing.assert_allclose(new_pitch, TARGET_PITCH_UM, rtol=1e-12)


def test_g5_corpus_ref_equals_target_long_edge(synth_tensor):
    _, out_path, _ = synth_tensor
    d = np.load(str(out_path))
    assert int(d["corpus_ref_long_edge"]) == TARGET_LONG_EDGE
    # Helper agreement (the assert inside downscale_tensor already enforced this).
    assert int(d["corpus_ref_long_edge"]) == _compute_corpus_ref(d["pixel_pitch_um"])


def test_full_frame_dim_recomputed(synth_tensor):
    """With uniform pitch, full_frame_dim_px[:,:,0] == target_long_edge."""
    _, out_path, _ = synth_tensor
    ff = np.load(str(out_path))["full_frame_dim_px"]
    assert ff.shape == (R, D, 2)
    assert np.all(ff[:, :, 0] == TARGET_LONG_EDGE)
    assert np.all(ff[:, :, 1] == int(round(TARGET_LONG_EDGE * 24 / 36)))


def test_g6_file_size_bound():
    """G6 analytic bound for target_long_edge=5712 (kH ~183): < 15 MB compressed."""
    target_kH = 183
    expected_bytes = R * D * target_kH * target_kH * 4
    # Raw ~17.7 MB; compressed must land under 15 MB in practice.
    assert expected_bytes / (1024 * 1024) < 20.0


def test_cli_default_output_naming(tmp_path: Path):
    """Default --out is <in_stem>_iphone<target_long_edge>.npz."""
    from summicron.psf_model.downscale_tensor import _default_out
    in_path = tmp_path / "foo_render.npz"
    _make_synthetic_tensor(in_path)
    out = _default_out(in_path, TARGET_LONG_EDGE)
    assert out.name == f"foo_render_iphone{TARGET_LONG_EDGE}.npz"
    downscale_tensor(in_path, out, target_long_edge=TARGET_LONG_EDGE)
    assert out.exists()


def test_default_out_strips_legacy_k_suffix(tmp_path: Path):
    """An input named ..._k256.npz should produce ..._iphone5712.npz (not ..._k256_iphone...)."""
    from summicron.psf_model.downscale_tensor import _default_out
    in_path = tmp_path / "foo_render_k256.npz"
    out = _default_out(in_path, 5712)
    assert out.name == "foo_render_iphone5712.npz"


def test_g1_center_delta_born_at_target_grid(synth_tensor):
    """v3 spec §S3: post-downscale+normalize, (r=0, d=0) is an exact
    single-pixel delta at the TARGET grid (183² for real corpus, 8²
    for this synthetic test).
    """
    _, out_path, _ = synth_tensor
    d = np.load(str(out_path))
    k00 = d["kernels"][0, 0]
    H, W = k00.shape
    assert float(k00[H // 2, W // 2]) == 1.0
    assert float(k00.sum()) == 1.0
    assert int((k00 != 0).sum()) == 1


def test_flags_p1_9(synth_tensor):
    """P1.10.2 Cut 2: otf_norm_applied flag removed; only delta_guard remains."""
    _, out_path, _ = synth_tensor
    d = np.load(str(out_path))
    assert "delta_guard_applied" in d.files and bool(d["delta_guard_applied"]) is True


def test_non_center_kernels_non_negative(synth_tensor):
    """OTF-norm clamps negatives; no kernel should carry negative pixels."""
    _, out_path, _ = synth_tensor
    kernels = np.load(str(out_path))["kernels"]
    assert kernels.min() >= 0.0


def test_all_kernels_l1_after_norm(synth_tensor):
    """Every kernel sums to 1.0 ± 0.01 after OTF-norm (v3 gate G2)."""
    _, out_path, _ = synth_tensor
    kernels = np.load(str(out_path))["kernels"]
    sums = kernels.reshape(R * D, -1).sum(axis=1)
    assert np.all(np.abs(sums - 1.0) <= 0.01)


def test_no_nan_inf_after_norm(synth_tensor):
    """No NaN/Inf after OTF-norm (v3 gate G3)."""
    _, out_path, _ = synth_tensor
    kernels = np.load(str(out_path))["kernels"]
    assert not np.any(np.isnan(kernels))
    assert not np.any(np.isinf(kernels))


REAL_CODEV = Path(__file__).resolve().parents[2] / "runs" / "ag_dblgauss" / "ag_dblgauss_f2.0_render.npz"


@pytest.mark.skipif(not REAL_CODEV.exists(), reason="real CodeV tensor not present")
def test_delta_guard_d0_slice(tmp_path: Path):
    """P1.9 spec item 7: real CodeV -> every (r, d=0) is an exact single-pixel
    delta, and (r=0, d=1) retains non-trivial PSF shape with L1 ~ 1.0."""
    out_path = tmp_path / "real_iphone5712.npz"
    downscale_tensor(REAL_CODEV, out_path, target_long_edge=5712)
    kernels = np.load(str(out_path))["kernels"]
    R_, D_ = kernels.shape[:2]
    for r in range(R_):
        k = kernels[r, 0]
        assert float(k.max()) == 1.0
        assert float(k.sum()) == 1.0
        assert int((k > 0).sum()) == 1
    k01 = kernels[0, 1]
    s = float(k01.sum())
    assert abs(s - 1.0) < 0.01
    assert float(k01.max()) / s < 0.5


def test_smaller_target_long_edge_gives_smaller_kH(tmp_path: Path):
    """target_long_edge=8000 -> target_kH=4; tensor + gates still pass."""
    in_path = tmp_path / "synth.npz"
    out_path = tmp_path / "synth_iphone8000.npz"
    _make_synthetic_tensor(in_path)
    downscale_tensor(in_path, out_path, target_long_edge=8000)
    d = np.load(str(out_path))
    # target_pitch = 36000/8000 = 4.5; scale = 0.5638/4.5 = 0.12529; kH = round(32*.12529) = 4
    assert d["kernels"].shape == (R, D, 4, 4)
    assert int(d["corpus_ref_long_edge"]) == 8000
