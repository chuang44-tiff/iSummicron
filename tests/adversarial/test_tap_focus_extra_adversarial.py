"""Adversarial tests for Item B (tap-to-focus + wrap-around) — EXTENDED.

Targets edge cases the Coder's happy-path file (test_tap_to_focus_adversarial)
does NOT cover:

  - MANDATORY wrong-polarity fixture (Day-1 post-mortem rule)
  - Wrap-around invariant on extreme synthetic ranges (0.01m to 1000m)
  - Wrap math correctness with hand-computed expected values
  - Corner taps (0,0), (1,1)
  - tap = (Inf, Inf) coords
  - focus_target_m = 0 (degenerate)
  - focus_target_m negative
  - Real-data reproducibility: closest-pixel-tap == None-fallback
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
FOCUS_M = 0.7


# ---------------------------------------------------------------------------
# MANDATORY: wrong-polarity fixture (Day-1 post-mortem #1)
# ---------------------------------------------------------------------------

@pytest.mark.xfail(
    reason="BUG: apply_tap_focus_shift trusts polarity convention (small=near, "
           "large=far). Given an inverted depth_m (small=far, large=near), the "
           "wrap math still 'succeeds' silently and produces wrong focus "
           "geometry. There is no polarity sniffer; consumer must trust caller."
)
def test_wrong_polarity_depth_should_warn():
    """Build a depth map where polarity is inverted (large value = NEAR pixel).
    apply_tap_focus_shift should detect or warn."""
    H, W = 60, 80
    # Conventional: smaller = nearer. Inverted: 8.0 is *near* the camera.
    inverted = np.full((H, W), 8.0, dtype=np.float32)
    inverted[10:20, 30:50] = 0.5  # in inverted convention: this region is FAR
    # Tap on the "near" region (large values 8.0).
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _shifted, anchor = apply_tap_focus_shift(
            inverted, (0.5, 0.5), focus_target_m=FOCUS_M
        )
    msgs = " ".join(str(w.message).lower() for w in caught)
    assert "polarity" in msgs or "invert" in msgs or "convention" in msgs, (
        "Expected polarity/inversion warning when depth_m has flipped sign convention"
    )


# ---------------------------------------------------------------------------
# Wrap-around invariant: extreme range
# ---------------------------------------------------------------------------

def test_wrap_invariant_extreme_range():
    """Synthetic depth from 0.01m to 1000m — after shift, ALL pixels >= focus."""
    rng = np.random.default_rng(seed=1)
    # Log-spaced extreme range
    depth_m = np.exp(rng.uniform(np.log(0.01), np.log(1000.0), size=(50, 50))).astype(np.float32)
    for tap_x in (0.1, 0.5, 0.9):
        shifted, anchor = apply_tap_focus_shift(
            depth_m, (tap_x, 0.5), focus_target_m=FOCUS_M
        )
        assert (shifted >= FOCUS_M - 1e-5).all(), (
            f"wrap invariant violated at tap_x={tap_x}: min={shifted.min()}"
        )
        assert np.isfinite(shifted).all()


# ---------------------------------------------------------------------------
# Wrap math: hand-computed expected values (single deterministic case)
# ---------------------------------------------------------------------------

def test_wrap_math_hand_computed():
    """Construct a 3x3 with known depths; tap on the center value 1.5.
    focus_target=0.7. Verify wrap math at two specific pixels."""
    # 3x3 with center 1.5 (tap target). Surrounding values 1.5 so 3x3 median == 1.5.
    depth_m = np.full((3, 3), 1.5, dtype=np.float32)
    # Now build a larger map that includes a 0.4m pixel and a 1.5m pixel
    # away from the tap, so we can verify the wrap math on those.
    H, W = 5, 5
    big = np.full((H, W), 2.0, dtype=np.float32)
    big[2, 2] = 1.5         # tap target
    big[1:4, 1:4] = 1.5     # 3x3 around tap is uniform 1.5 -> median = 1.5
    big[0, 0] = 0.4         # known near pixel
    big[0, 4] = 1.5         # equals anchor
    # Tap at (x=0.5, y=0.5) -> x_pix = int(0.5*5)=2, y_pix=2.
    shifted, anchor = apply_tap_focus_shift(big, (0.5, 0.5), focus_target_m=0.7)
    assert anchor == pytest.approx(1.5, abs=1e-5)
    # shift = 0.7 - 1.5 = -0.8
    # pixel 0.4: shifted_raw = 0.4 - 0.8 = -0.4 -> wrap = 2*0.7 - (-0.4) = 1.8
    assert shifted[0, 0] == pytest.approx(1.8, abs=1e-4)
    # pixel 1.5: shifted_raw = 1.5 - 0.8 = 0.7 -> not < 0.7 (not strict) -> kept 0.7
    # depending on float rounding, may wrap. Accept either 0.7 (no wrap) or
    # ~0.7 (wrapped to 0.7 too, since 2*0.7-0.7=0.7).
    assert shifted[0, 4] == pytest.approx(0.7, abs=1e-4)
    # background pixel 2.0: shifted = 2.0 - 0.8 = 1.2 (no wrap)
    assert shifted[2, 0] == pytest.approx(1.2, abs=1e-4)


# ---------------------------------------------------------------------------
# Corner taps: (0,0) and (1,1) — 3x3 patch must handle borders
# ---------------------------------------------------------------------------

def test_tap_corner_top_left():
    depth_m = np.full((10, 10), 1.0, dtype=np.float32)
    depth_m[0, 0] = 0.5  # corner pixel known
    shifted, anchor = apply_tap_focus_shift(depth_m, (0.0, 0.0), focus_target_m=FOCUS_M)
    # 3x3 median around (0,0) = median of [0.5, 1.0, 1.0, 1.0] (mirror-clamped) = 1.0
    assert 0.5 <= anchor <= 1.0
    assert (shifted >= FOCUS_M - 1e-5).all()
    assert np.isfinite(shifted).all()


def test_tap_corner_bottom_right():
    depth_m = np.full((12, 18), 2.0, dtype=np.float32)
    depth_m[-1, -1] = 0.8  # corner pixel
    shifted, anchor = apply_tap_focus_shift(depth_m, (1.0, 1.0), focus_target_m=FOCUS_M)
    # x_clip=1.0 -> x_pix = min(int(1.0*18), 17) = 17. y_pix=11.
    # Patch [10:13, 16:19] = [10:12, 16:18] (clipped) of all-2.0 except corner 0.8.
    # Median of [2,2,2,0.8] = 2.0 (4 elements, median is mean of 2 middles = (2+2)/2 = 2)
    assert anchor == pytest.approx(2.0, abs=1e-4) or 0.8 <= anchor <= 2.0
    assert (shifted >= FOCUS_M - 1e-5).all()


# ---------------------------------------------------------------------------
# Inf tap coords -> non-finite warning + fallback
# ---------------------------------------------------------------------------

def test_tap_with_inf_coord():
    depth_m = np.full((10, 10), 1.7, dtype=np.float32)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _, anchor = apply_tap_focus_shift(
            depth_m, (float("inf"), 0.5), focus_target_m=FOCUS_M
        )
    msgs = " ".join(str(w.message).lower() for w in caught)
    assert "non-finite" in msgs or "nan" in msgs or "inf" in msgs
    assert anchor == pytest.approx(1.7, abs=1e-4)


# ---------------------------------------------------------------------------
# focus_target_m = 0 (degenerate)
# ---------------------------------------------------------------------------

@pytest.mark.xfail(
    reason="BUG: apply_tap_focus_shift accepts focus_target_m=0 silently. "
           "shift = 0 - anchor = -anchor; everything wraps to <= 0; clip to "
           "[0, 1e3] floor at 0 produces nonsensical zero-distance focus."
)
def test_focus_target_zero_should_reject():
    depth_m = np.full((10, 10), 1.0, dtype=np.float32)
    with pytest.raises((ValueError, AssertionError)):
        apply_tap_focus_shift(depth_m, (0.5, 0.5), focus_target_m=0.0)


@pytest.mark.xfail(
    reason="BUG: apply_tap_focus_shift accepts negative focus_target_m silently."
)
def test_focus_target_negative_should_reject():
    depth_m = np.full((10, 10), 1.0, dtype=np.float32)
    with pytest.raises((ValueError, AssertionError)):
        apply_tap_focus_shift(depth_m, (0.5, 0.5), focus_target_m=-0.3)


# ---------------------------------------------------------------------------
# All-NaN depth map -> safe fallback (covered in production code path).
# ---------------------------------------------------------------------------

def test_all_nan_depth_returns_flat_focus():
    depth_m = np.full((20, 20), np.nan, dtype=np.float32)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        shifted, anchor = apply_tap_focus_shift(depth_m, (0.5, 0.5), focus_target_m=FOCUS_M)
    msgs = " ".join(str(w.message).lower() for w in caught)
    assert "non-finite" in msgs or "nan" in msgs
    assert anchor == pytest.approx(FOCUS_M, abs=1e-5)
    assert np.allclose(shifted, FOCUS_M, atol=1e-5)


# ---------------------------------------------------------------------------
# Closest-pixel-tap reproducibility vs. None-fallback
# Spec: "with --focus-xy omitted (closest-subject fallback), output should
#  match what probe-round-4 reports". So: tap=None should equal tap @ p2 location.
# ---------------------------------------------------------------------------

def test_none_tap_matches_p2_anchor_value():
    """tap=None -> anchor=find_anchor_depth(depth_m, 2.0). Verify equivalence."""
    if not HEIC.exists():
        pytest.skip("IMG_6651.HEIC fixture not present.")
    depth_m = extract_lidar_depth(HEIC, target_hw=(768, 576))
    shifted_none, anchor_none = apply_tap_focus_shift(depth_m, None, focus_target_m=FOCUS_M)
    expected_anchor = find_anchor_depth(depth_m, percentile=2.0)
    assert anchor_none == pytest.approx(expected_anchor, abs=1e-5)
    # Direct synthesis: shift by (FOCUS_M - expected_anchor), then wrap.
    raw = depth_m + (FOCUS_M - expected_anchor)
    raw[raw < FOCUS_M] = 2.0 * FOCUS_M - raw[raw < FOCUS_M]
    raw = np.clip(raw, FOCUS_M, 1e3).astype(np.float32)
    assert np.allclose(shifted_none, raw, atol=1e-4), (
        f"None-tap output diverged from manual synthesis; max|Δ|={np.abs(shifted_none - raw).max()}"
    )


# ---------------------------------------------------------------------------
# Library-API drift sentinel: scipy.ndimage.median_filter signature pinned
# (production currently uses np.median, not median_filter, but spec mentions it.)
# ---------------------------------------------------------------------------

def test_scipy_median_filter_signature_stable():
    import inspect
    import scipy.ndimage as ndi
    sig = inspect.signature(ndi.median_filter)
    params = list(sig.parameters.keys())
    # Required+common kwargs that the codebase or spec might rely on:
    for required in ("input", "size", "footprint", "mode"):
        assert required in params, (
            f"scipy.ndimage.median_filter dropped {required!r}; signature now: {params}"
        )
