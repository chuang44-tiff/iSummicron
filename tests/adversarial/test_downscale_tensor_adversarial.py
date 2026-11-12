"""Adversarial tests for summicron.psf_model.downscale_tensor.

Targets edge cases and failure modes missed by the happy-path unit tests:
all-zero kernels, tiny/upscale targets, non-square input, degenerate pitch,
corrupt npz, odd target sizes. Also round-trips through load_tensor() to
verify downstream compatibility.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from summicron.psf_model.codev_to_render import _compute_corpus_ref
from summicron.psf_model.downscale_tensor import (
    _center_crop_or_pad,
    downscale_tensor,
)

R, D = 12, 11
ORIG_KH = 32
# Map a desired target_kH -> target_long_edge for our synthetic fixture
# (uniform pitch 0.5638 um, orig_kH=32). Formula:
#   target_pitch = 36000 / target_long_edge
#   target_kH    = round(orig_kH * orig_pitch / target_pitch)
#                = round(orig_kH * orig_pitch * target_long_edge / 36000)
# So for orig_pitch=0.5638, orig_kH=32:
#   target_long_edge = round(target_kH * 36000 / (0.5638 * 32))
def _tle_for_kH(target_kH: int, pitch: float = 0.5638) -> int:
    return int(round(target_kH * 36000.0 / (pitch * ORIG_KH)))


def _make_tensor(
    path: Path,
    orig_kH: int = ORIG_KH,
    orig_kW: int | None = None,
    zero_bin: tuple[int, int] | None = None,
    pitch_value: float = 0.5638,
) -> dict:
    """Build a minimal synthetic render-ready .npz. Supports non-square kernels
    and a deliberately all-zero (R,D) bin."""
    kW = orig_kW if orig_kW is not None else orig_kH
    kernels = np.zeros((R, D, orig_kH, kW), dtype=np.float32)
    y, x = np.mgrid[:orig_kH, :kW].astype(np.float32)
    cy, cx = (orig_kH - 1) / 2.0, (kW - 1) / 2.0
    sigma = max(orig_kH, kW) / 10.0
    for ri in range(R):
        for di in range(D):
            if zero_bin is not None and (ri, di) == zero_bin:
                continue  # leave it all-zero
            k = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma ** 2))
            kernels[ri, di] = (k / k.sum()).astype(np.float32)

    pixel_pitch_um = np.full((R, D), pitch_value, dtype=np.float64)
    full_frame_dim_px = np.zeros((R, D, 2), dtype=np.int32)
    if pitch_value > 0:
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
        corpus_ref_long_edge=np.int64(
            _compute_corpus_ref(pixel_pitch_um) if pitch_value > 0 else 1
        ),
    )
    np.savez(str(path), **fields)
    return fields


# ---------------------------------------------------------------------------
# Item 1 — downscale_tensor edge cases
# ---------------------------------------------------------------------------

def test_all_zero_bin_raises_runtime_error(tmp_path: Path):
    """An all-zero (R,D) PSF bin must raise RuntimeError with a clear message,
    not silently produce NaN via L1 renorm divide-by-zero."""
    in_path = tmp_path / "zero_bin.npz"
    out_path = tmp_path / "zero_bin_k16.npz"
    _make_tensor(in_path, zero_bin=(3, 5))
    with pytest.raises(RuntimeError, match=r"non-positive sum.*R=3.*D=5"):
        downscale_tensor(in_path, out_path, target_long_edge=_tle_for_kH(16))


def test_tiny_target_kH_4(tmp_path: Path):
    """target_kH=4 (via small target_long_edge): cubic zoom + renorm should
    still pass gates; extreme blur but L1=1 and no NaN must hold."""
    in_path = tmp_path / "tiny.npz"
    out_path = tmp_path / "tiny_small.npz"
    _make_tensor(in_path)
    downscale_tensor(in_path, out_path, target_long_edge=_tle_for_kH(4))
    d = np.load(str(out_path))
    assert d["kernels"].shape == (R, D, 4, 4)
    sums = d["kernels"].reshape(R * D, -1).sum(axis=1)
    assert sums.min() >= 0.999 and sums.max() <= 1.001
    assert not np.any(np.isnan(d["kernels"]))


def test_target_kH_1_degenerate(tmp_path: Path):
    """target_kH=1 is pathological but must not crash with divide-by-zero.
    After zoom+crop every kernel has a single pixel whose L1 is either 1 or 0.
    If the cubic zoom somehow produces zero sum, gate should fire cleanly."""
    in_path = tmp_path / "deg1.npz"
    out_path = tmp_path / "deg1_k1.npz"
    _make_tensor(in_path)
    # Must either succeed (all-ones kernels after renorm) or raise RuntimeError,
    # never a ZeroDivisionError / NaN.
    try:
        downscale_tensor(in_path, out_path, target_long_edge=_tle_for_kH(1))
    except RuntimeError:
        return
    d = np.load(str(out_path))
    assert d["kernels"].shape == (R, D, 1, 1)
    # Every single-pixel kernel must be exactly 1.0 after L1 renorm.
    np.testing.assert_allclose(d["kernels"], 1.0, atol=1e-6)


def test_upscale_target_larger_than_orig_raises(tmp_path: Path):
    """target_long_edge implying target_kH > orig_kH must RAISE ValueError.

    The CodeV tensor is the upper bound on kernel resolution; upscaling
    the PSF is physically meaningless (it just interpolates).
    """
    in_path = tmp_path / "up.npz"
    out_path = tmp_path / "up_big.npz"
    _make_tensor(in_path, orig_kH=32)
    # target_kH=128 requires target_long_edge ~ 255k; must be rejected.
    with pytest.raises(ValueError, match=r"upscaling is forbidden|> orig_kH"):
        downscale_tensor(in_path, out_path, target_long_edge=_tle_for_kH(128))


def test_non_square_input_rejected(tmp_path: Path):
    """Non-square (kH != kW) input must be rejected cleanly, not crash later."""
    in_path = tmp_path / "nonsq.npz"
    out_path = tmp_path / "nonsq_k16.npz"
    _make_tensor(in_path, orig_kH=32, orig_kW=48)
    with pytest.raises(RuntimeError, match=r"Non-square kernel grid"):
        downscale_tensor(in_path, out_path, target_long_edge=_tle_for_kH(16))


def test_odd_target_kH_via_long_edge(tmp_path: Path):
    """A target_long_edge that produces an odd target_kH (not dividing 32
    evenly) must still pass all gates. Pick one that yields kH=7 with
    pitch 0.5638: target_long_edge = round(7 * 36000 / (0.5638*32)) = 13973."""
    in_path = tmp_path / "odd.npz"
    out_path = tmp_path / "odd_out.npz"
    _make_tensor(in_path, orig_kH=32)
    tle = _tle_for_kH(7)
    downscale_tensor(in_path, out_path, target_long_edge=tle)
    d = np.load(str(out_path))
    assert d["kernels"].shape == (R, D, 7, 7)
    sums = d["kernels"].reshape(R * D, -1).sum(axis=1)
    assert sums.min() >= 0.999 and sums.max() <= 1.001
    assert int(d["corpus_ref_long_edge"]) == tle


def test_file_not_found(tmp_path: Path):
    """Nonexistent input file must raise FileNotFoundError (or similar IO
    error), not a cryptic numpy error."""
    out_path = tmp_path / "missing_out.npz"
    with pytest.raises((FileNotFoundError, OSError)):
        downscale_tensor(
            tmp_path / "does_not_exist.npz", out_path,
            target_long_edge=_tle_for_kH(16),
        )


def test_corrupt_npz(tmp_path: Path):
    """Corrupt .npz (random bytes) must raise a clean error."""
    in_path = tmp_path / "corrupt.npz"
    in_path.write_bytes(b"this is not an npz file " * 100)
    out_path = tmp_path / "corrupt_out.npz"
    with pytest.raises(Exception):
        downscale_tensor(in_path, out_path, target_long_edge=_tle_for_kH(16))


def test_center_crop_or_pad_symmetry():
    """Direct unit test of the center crop/pad helper on known patterns."""
    # Pad: 2x2 -> 4x4 centers the original (offsets 1,1).
    k = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    out = _center_crop_or_pad(k, 4)
    assert out.shape == (4, 4)
    np.testing.assert_array_equal(out[1:3, 1:3], k)
    assert out[0, 0] == 0 and out[3, 3] == 0
    # Crop: 6x6 -> 2x2 takes center.
    big = np.arange(36, dtype=np.float32).reshape(6, 6)
    out = _center_crop_or_pad(big, 2)
    assert out.shape == (2, 2)
    np.testing.assert_array_equal(out, big[2:4, 2:4])


# ---------------------------------------------------------------------------
# Item 2 — _compute_corpus_ref edge cases
# ---------------------------------------------------------------------------

def test_compute_corpus_ref_empty_array():
    """Empty pitch array: mean of empty -> NaN -> int cast should ValueError.
    Documents current behaviour so callers aren't surprised."""
    with pytest.raises((ValueError, RuntimeWarning)):
        # Python raises ValueError on int(NaN); a numpy warning may fire first.
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            _compute_corpus_ref(np.array([], dtype=np.float64))


def test_compute_corpus_ref_zero_pitch():
    """Zero-mean pitch divides by zero -> ZeroDivisionError expected."""
    with pytest.raises(ZeroDivisionError):
        _compute_corpus_ref(np.zeros((2, 2), dtype=np.float64))


@pytest.mark.xfail(
    reason="BUG: _compute_corpus_ref returns a negative integer for negative "
           "pitch instead of raising; callers baking this into npz will poison "
           "downstream render.convolve."
)
def test_compute_corpus_ref_negative_pitch_should_reject():
    """Negative pitch is physically impossible; the helper should raise,
    not silently return a negative int."""
    result = _compute_corpus_ref(np.full((2, 2), -0.5638, dtype=np.float64))
    # If we reach this line, the helper returned a value. Assert it's positive
    # (it won't be: current impl returns -63851, flipping xfail to fail-as-xfail).
    assert result > 0, f"Got non-positive corpus_ref {result} for negative pitch"


def test_compute_corpus_ref_agreement_across_callers(tmp_path: Path):
    """Anti-drift: the value baked by codev_to_render must equal the value
    recomputed inside downscale_tensor for the same pitch array."""
    in_path = tmp_path / "drift.npz"
    out_path = tmp_path / "drift_out.npz"
    _make_tensor(in_path)
    downscale_tensor(in_path, out_path, target_long_edge=_tle_for_kH(16))
    d_out = np.load(str(out_path))
    # Helper recomputed on the output's new_pitch must agree with baked value.
    assert int(d_out["corpus_ref_long_edge"]) == _compute_corpus_ref(
        d_out["pixel_pitch_um"]
    )


def test_load_tensor_back_compat_missing_corpus_ref(tmp_path: Path):
    """Old npz files without corpus_ref_long_edge must still load with the
    documented 5976 default."""
    from summicron.render.convolve import load_tensor
    in_path = tmp_path / "old.npz"
    fields = _make_tensor(in_path)
    fields.pop("corpus_ref_long_edge")
    # Rewrite without the field.
    old_path = tmp_path / "old_no_ref.npz"
    np.savez(str(old_path), **fields)
    t = load_tensor(old_path)
    assert t["corpus_ref_long_edge"] == 5976


# ---------------------------------------------------------------------------
# Cross-cutting: round-trip through load_tensor
# ---------------------------------------------------------------------------

def test_round_trip_load_tensor(tmp_path: Path):
    """convert→downscale→load_tensor round trip: invariants hold on output."""
    from summicron.render.convolve import load_tensor
    in_path = tmp_path / "rt.npz"
    out_path = tmp_path / "rt_out.npz"
    _make_tensor(in_path)
    downscale_tensor(in_path, out_path, target_long_edge=_tle_for_kH(16))
    t = load_tensor(out_path)
    assert t["kernels"].shape == (R, D, 16, 16)
    # L1 preserved.
    sums = t["kernels"].reshape(R * D, -1).sum(axis=1)
    assert sums.min() >= 0.999 and sums.max() <= 1.001
    # corpus_ref_long_edge plumbed through.
    assert isinstance(t["corpus_ref_long_edge"], int)
    assert t["corpus_ref_long_edge"] > 0
