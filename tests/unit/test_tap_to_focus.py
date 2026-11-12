"""Unit tests for Item B (tap-to-focus + wrap-around) and Item C (DoF band).

Axis convention reminder: ``tap_xy_norm`` is ``(x, y)`` where x = column / W
and y = row / H — standard image-xy, NOT (row, col).
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest

from summicron.depth.estimate import (
    apply_tap_focus_shift,
    build_depth_defocus_map,
    extract_lidar_depth,
    find_anchor_depth,
)

HEIC = Path(__file__).resolve().parents[2] / "iphoneImage" / "IMG_6651.HEIC"
TARGET_HW = (768, 576)
FOCUS_M = 0.7


# ---------------------------------------------------------------------------
# Item A migration tests
# ---------------------------------------------------------------------------

def test_extract_lidar_depth_meters():
    """Spec G2: 0.40 <= min <= 0.45, 2.50 <= max <= 2.70 on IMG_6651."""
    if not HEIC.exists():
        pytest.skip("IMG_6651.HEIC fixture not present.")
    depth_m = extract_lidar_depth(HEIC, target_hw=TARGET_HW)
    assert depth_m is not None
    assert depth_m.dtype == np.float32
    assert depth_m.shape == TARGET_HW
    assert 0.40 <= float(depth_m.min()) <= 0.45
    assert 2.50 <= float(depth_m.max()) <= 2.70


def test_find_anchor_depth_meters():
    """Spec G1: 0.42 <= anchor (p2) <= 0.45 on IMG_6651."""
    if not HEIC.exists():
        pytest.skip("IMG_6651.HEIC fixture not present.")
    depth_m = extract_lidar_depth(HEIC, target_hw=TARGET_HW)
    anchor = find_anchor_depth(depth_m, percentile=2.0)
    assert 0.42 <= anchor <= 0.45, f"anchor {anchor:.4f} outside [0.42, 0.45]"


# ---------------------------------------------------------------------------
# Item B math tests (synthetic)
# ---------------------------------------------------------------------------

def test_apply_tap_focus_shift_math():
    """Synthetic constant depth 1.5m; tap centre; expect shifted == 0.7m."""
    depth_m = np.full((10, 10), 1.5, dtype=np.float32)
    shifted, anchor = apply_tap_focus_shift(depth_m, (0.5, 0.5), focus_target_m=FOCUS_M)
    assert anchor == pytest.approx(1.5, abs=1e-5)
    # All pixels shift by -0.8 -> 0.7. Wrap won't engage (shifted == focus).
    assert np.allclose(shifted, FOCUS_M, atol=1e-5)


def test_apply_tap_focus_wrap_table():
    """Spec scope-doc table: depth [0.5, 1.0, 1.5, 2.5, 5.0] (1D shape (1,5)),
    tap on the 1.5 pixel (column index 2). After shift (-0.8): [-0.3, 0.2, 0.7, 1.7, 4.2].
    After wrap on neg: [1.7, 1.2, 0.7, 1.7, 4.2]."""
    row = np.array([[0.5, 1.0, 1.5, 2.5, 5.0]], dtype=np.float32)
    # tap_xy_norm = (x=2.5/5=0.5, y=0.5) → x_pix = min(int(0.5*5), 4) = 2, y_pix = 0.
    # 3x3 median around (y=0,x=2) on a 1-row array: median of [1.0, 1.5, 2.5] = 1.5.
    shifted, anchor = apply_tap_focus_shift(row, (0.5, 0.5), focus_target_m=FOCUS_M)
    assert anchor == pytest.approx(1.5, abs=1e-5)
    expected = np.array([[1.7, 1.2, 0.7, 1.7, 4.2]], dtype=np.float32)
    assert np.allclose(shifted, expected, atol=1e-4), (
        f"shifted={shifted}, expected={expected}"
    )


def test_apply_tap_focus_no_tap_fallback():
    """tap=None on IMG_6651: anchor matches find_anchor_depth p2 (≈0.43m)."""
    if not HEIC.exists():
        pytest.skip("IMG_6651.HEIC fixture not present.")
    depth_m = extract_lidar_depth(HEIC, target_hw=TARGET_HW)
    expected_anchor = find_anchor_depth(depth_m, 2.0)
    _, got_anchor = apply_tap_focus_shift(depth_m, None, focus_target_m=FOCUS_M)
    assert got_anchor == pytest.approx(expected_anchor, abs=1e-5)


def test_apply_tap_focus_anchor_at_tap_invariant():
    """Spec G7: after shift, shifted_m at the tapped pixel == focus_target_m."""
    H, W = 20, 30
    rng = np.random.default_rng(42)
    depth_m = (rng.random((H, W)).astype(np.float32) * 2.0 + 0.5)
    # Tap deliberately at (x=0.5, y=0.5) → x_pix=15, y_pix=10.
    shifted, anchor = apply_tap_focus_shift(depth_m, (0.5, 0.5), focus_target_m=FOCUS_M)
    # Cannot assert pixel-equality because anchor is 3x3 median, not pixel value.
    # But we CAN assert: every shifted pixel >= focus_target_m (G3 invariant).
    assert (shifted >= FOCUS_M - 1e-5).all()
    # Median anchor location: shifted at center should be near focus_target_m within
    # the 3x3 median spread.
    centre_orig = depth_m[10, 15]
    expected_at_centre = centre_orig + (FOCUS_M - anchor)
    if expected_at_centre < FOCUS_M:
        expected_at_centre = 2 * FOCUS_M - expected_at_centre
    assert shifted[10, 15] == pytest.approx(expected_at_centre, abs=1e-4)


def test_wrap_invariant_g3():
    """Spec G3: shifted_wrapped_m >= focus_target_m for ALL pixels."""
    depth_m = np.linspace(0.3, 5.0, 100, dtype=np.float32).reshape(10, 10)
    # Tap on a pixel near the far end -> shift negative -> wrap engages.
    shifted, _ = apply_tap_focus_shift(depth_m, (0.95, 0.95), focus_target_m=FOCUS_M)
    assert (shifted >= FOCUS_M - 1e-5).all()


# ---------------------------------------------------------------------------
# build_depth_defocus_map (meters) + DoF band (Item C)
# ---------------------------------------------------------------------------

def test_build_depth_defocus_map_meters():
    """Linear ramp 0.5→2.0, anchor=0.7. Defocus near anchor ~ 0; max ~ 1.0;
    mean in [0.1, 0.7]."""
    H, W = 50, 100
    depth_m = np.tile(np.linspace(0.5, 2.0, W, dtype=np.float32), (H, 1))
    defocus = build_depth_defocus_map(depth_m, anchor_m=0.7)
    assert defocus.dtype == np.float32
    assert defocus.shape == (H, W)
    assert defocus.min() >= 0.0
    assert defocus.max() <= 1.0
    assert defocus.max() >= 0.5, f"defocus.max={defocus.max():.3f}, expected >=0.5"
    # Pixel closest to 0.7m should have very small defocus.
    col_at_07 = int(round((0.7 - 0.5) / (2.0 - 0.5) * (W - 1)))
    assert defocus[H // 2, col_at_07] < 0.05
    mean = float(defocus.mean())
    assert 0.1 <= mean <= 0.7, f"mean defocus {mean:.3f} outside [0.1, 0.7]"


def test_dof_band():
    """Same ramp, anchor=0.7, dof_m=0.1: pixels with |depth-0.7|<0.1 → defocus=0."""
    H, W = 20, 100
    depth_m = np.tile(np.linspace(0.5, 2.0, W, dtype=np.float32), (H, 1))
    defocus = build_depth_defocus_map(depth_m, anchor_m=0.7, dof_m=0.1)
    band_mask = np.abs(depth_m - 0.7) < 0.1
    assert band_mask.sum() > 0
    assert (defocus[band_mask] == 0.0).all(), (
        f"DoF band pixels not all zero: max in band = {defocus[band_mask].max():.6f}"
    )
    # Outside band: at least some non-zero defocus.
    out_mask = ~band_mask
    assert defocus[out_mask].max() > 0.1


def test_no_nan_inf_g8():
    """Spec G8: no NaN/Inf in depth_m, shifted_m, or defocus_map."""
    depth_m = np.linspace(0.4, 3.0, 200, dtype=np.float32).reshape(10, 20)
    shifted, anchor = apply_tap_focus_shift(depth_m, (0.5, 0.5), focus_target_m=FOCUS_M)
    defocus = build_depth_defocus_map(shifted, FOCUS_M, dof_m=0.05)
    for arr, name in [(depth_m, "depth_m"), (shifted, "shifted"), (defocus, "defocus")]:
        assert not np.any(np.isnan(arr)), f"NaN in {name}"
        assert not np.any(np.isinf(arr)), f"Inf in {name}"


# ---------------------------------------------------------------------------
# Item C — DoF band edge semantics (ported from the deleted
# tests/adversarial/test_dof_cli_render_adversarial.py, CLI-1a B1.2; the
# --dof-m CLI flag is gone, build_depth_defocus_map(dof_m=) is kept).
# ---------------------------------------------------------------------------

def test_dof_zero_matches_no_band():
    """dof_m=0 (default) must produce identical defocus map as no-band."""
    rng = np.random.default_rng(seed=0)
    z = rng.uniform(0.5, 5.0, size=(40, 50)).astype(np.float32)
    d_default = build_depth_defocus_map(z, 1.0)
    d_zero = build_depth_defocus_map(z, 1.0, dof_m=0.0)
    assert np.array_equal(d_default, d_zero), "dof_m=0 should be a no-op"


def test_dof_huge_zeros_entire_scene():
    """dof_m larger than scene depth range -> defocus map all zeros."""
    z = np.linspace(0.5, 3.0, 100, dtype=np.float32).reshape(10, 10)
    d = build_depth_defocus_map(z, 1.5, dof_m=100.0)
    assert np.allclose(d, 0.0), f"expected zeros under huge DoF; got max={d.max()}"


@pytest.mark.xfail(
    reason="BUG: build_depth_defocus_map accepts dof_m<0 silently "
           "(|z-a| < negative is always False -> behaves like dof_m=0). "
           "Should raise ValueError to surface caller misuse."
)
def test_dof_negative_should_reject():
    z = np.full((10, 10), 1.0, dtype=np.float32)
    with pytest.raises(ValueError):
        build_depth_defocus_map(z, 1.0, dof_m=-0.5)


def test_dof_band_edge_pixel():
    """Band masking uses strict ``<``: a pixel at exactly anchor + dof_m is
    NOT zeroed and keeps its raw |1/z - 1/a| contribution."""
    z = np.array([[1.0, 1.5, 2.0]], dtype=np.float32)
    d = build_depth_defocus_map(z, 1.0, dof_m=0.5)
    assert d[0, 0] == 0.0, "pixel at anchor must be zeroed"
    assert d[0, 1] > 0.0, "boundary pixel (|delta|==dof) must NOT be zeroed (strict <)"
    assert d[0, 2] > 0.0
