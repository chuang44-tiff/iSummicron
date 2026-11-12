"""P1.10 region-aware bright-pass — adversarial tests T8-T12.

Spec: ``doc/plan/phase1/section16/locked-spec-P1_10.md`` § 7.

T8: NaN/Inf safety (no crash; finite at non-affected pixels).
T9: single-pixel input (1×1) does not crash.
T10: deleted in HYG-0 item 14 (dead duplicate of G1; passed removed --layered-gather).
T11: wall-clock < 500ms on 2856×2142 float32 (slow).
T12: chromaticity gate — region detector operates on luminance only.
"""
from __future__ import annotations

import time

import numpy as np
import pytest

from summicron.render.convolve import _compute_region_aware_bright_weight


# T8 — NaN/Inf safety
@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_T8_nonfinite_pixel_does_not_crash(bad_value):
    """A single NaN/Inf pixel must not crash the function; pixels far from
    the contaminated neighborhood remain finite (nanpercentile guards the
    global anchor)."""
    lum = np.full((64, 64), 1.5, dtype=np.float32)
    lum[32, 32] = bad_value
    bw = _compute_region_aware_bright_weight(lum, 1.0, 15, 0.10, 99.0)
    # Corner is far enough from (32, 32) that grey_opening's region_size=15
    # neighborhood does not reach it; bw_lum at corner is finite, top-hat
    # is finite, and the global anchor uses nanpercentile.
    assert np.isfinite(bw[0, 0]), f"corner pixel must be finite, got {bw[0, 0]}"


# T9 — single-pixel input
def test_T9_single_pixel_input():
    """A 1×1 luminance array must not crash; returns a valid scalar weight."""
    lum = np.array([[1.5]], dtype=np.float32)
    bw = _compute_region_aware_bright_weight(lum, 1.0, 31, 0.10, 99.0)
    assert bw.shape == (1, 1)
    assert np.isfinite(bw[0, 0])
    assert 0.0 <= float(bw[0, 0]) <= 1.0


# T11 — wall-clock < 500ms (slow)
@pytest.mark.slow
def test_T11_wallclock_under_500ms_on_full_res():
    """Detector cost on a 2856×2142 float32 luminance map: median(5) < 500ms."""
    rng = np.random.default_rng(42)
    lum = rng.uniform(0.0, 2.0, (2856, 2142)).astype(np.float32)
    _compute_region_aware_bright_weight(lum, 1.0, 31, 0.10, 99.0)  # warm-up
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        _compute_region_aware_bright_weight(lum, 1.0, 31, 0.10, 99.0)
        times.append(time.perf_counter() - t0)
    median_ms = float(np.median(times)) * 1000.0
    assert median_ms < 500.0, (
        f"detector cost {median_ms:.1f}ms > 500ms budget; "
        f"times_ms={[round(t*1000, 1) for t in times]}"
    )


# T12 — chromaticity gate: detector operates on luminance only
def test_T12_chromaticity_gate_luminance_only():
    """The detector consumes a single 2D luminance map (Rec.709 RGB→Y is done
    upstream by render()); two RGB fixtures with identical luminance produce
    identical bright_weight regardless of chroma. Mandatory per
    ``feedback_adversarial_chromaticity_gate.md``: warm + cool saturated
    HDR fixtures must produce the same multiplier."""
    rec709 = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    target_lum = 2.0
    # Warm (R-dominant) and cool (B-dominant) RGBs scaled to identical lum.
    warm_rgb = np.array([5.0, 1.5, 0.3], dtype=np.float32)
    warm_rgb *= target_lum / float(np.dot(rec709, warm_rgb))
    cool_rgb = np.array([0.3, 1.5, 5.0], dtype=np.float32)
    cool_rgb *= target_lum / float(np.dot(rec709, cool_rgb))
    lum_w = float(np.dot(rec709, warm_rgb))
    lum_c = float(np.dot(rec709, cool_rgb))
    assert abs(lum_w - lum_c) < 1e-5

    # Build matched bright-spot fixtures (different chroma, identical lum).
    H, W = 96, 96
    img_warm = np.full((H, W, 3), 0.2, dtype=np.float32)
    img_cool = np.full((H, W, 3), 0.2, dtype=np.float32)
    img_warm[40:48, 40:48, :] = warm_rgb
    img_cool[40:48, 40:48, :] = cool_rgb
    lum_warm = (img_warm * rec709).sum(axis=-1)
    lum_cool = (img_cool * rec709).sum(axis=-1)
    np.testing.assert_allclose(lum_warm, lum_cool, atol=1e-5)

    bw_warm = _compute_region_aware_bright_weight(
        lum_warm, 1.0, 15, 0.10, 99.0,
    )
    bw_cool = _compute_region_aware_bright_weight(
        lum_cool, 1.0, 15, 0.10, 99.0,
    )
    np.testing.assert_allclose(bw_warm, bw_cool, atol=1e-6)
