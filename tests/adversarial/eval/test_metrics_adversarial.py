"""Adversarial tests for metric implementations (LOCKED SPEC §5 items 7-13).

Fixture-level adversarial tests (items 1-6) live in Coder A's
``test_fixtures_adversarial.py``.
"""
from __future__ import annotations

import numpy as np
import pytest

from summicron.eval.metrics import (
    energy_ratio,
    m2_roundness,
    m3_sag_tan,
    m7_wrap_probe,
)
from tests.unit.eval.conftest import (
    bubble_grid_annotation_stub,
    mono_rgb,
    rasterize_disc,
    standard_grid_discs,
)


# ---------- Item 7: NaN/Inf in rendered input -> typed error ----------

def test_item7_nan_in_rendered_rejected_by_energy():
    rgb = mono_rgb(np.ones((64, 64), dtype=np.float32))
    bad = rgb.copy()
    bad[0, 0, 0] = np.nan
    with pytest.raises((ValueError, AssertionError)):
        energy_ratio(rgb, bad)


def test_item7_inf_in_rendered_rejected_by_m2():
    mono = rasterize_disc(1024, 1024, 512, 512, radius=20.0)
    mono[512, 512] = np.inf
    rgb = mono_rgb(mono)
    ann = bubble_grid_annotation_stub(standard_grid_discs())
    with pytest.raises(ValueError):
        m2_roundness(rgb, ann)


# ---------- Item 8: all-zero rendered -> NaN with typed flag ----------

def test_item8_zero_input_too_dark():
    rgb = np.zeros((64, 64, 3), dtype=np.float32)
    ratio, flag = energy_ratio(rgb, rgb)
    assert np.isnan(ratio)
    assert flag == "input_too_dark"


def test_item8_zero_rendered_m2_no_blob():
    """Rendered all-zero ROI triggers 'input_flat' or 'no_blob'."""
    mono = np.zeros((1024, 1024), dtype=np.float32)
    rgb = mono_rgb(mono)
    ann = bubble_grid_annotation_stub(standard_grid_discs())
    val, flag = m2_roundness(rgb, ann)
    assert np.isnan(val)
    assert flag in ("no_blob", "input_flat")


# ---------- Item 9: sub-pixel disc < 5 px -> low_support ----------

def test_item9_subpixel_disc_m2_low_support():
    # A single 1-px "disc" (radius 0.7 -> few pixels), well below 20 px support
    mono = np.zeros((1024, 1024), dtype=np.float32)
    mono[512, 512] = 1.0
    rgb = mono_rgb(mono)
    ann = bubble_grid_annotation_stub(
        [
            {
                "idx": 0,
                "center_xy_px": [512, 512],
                "r_frac": 0.0,
                "depth_m": 0.70,
                "wrap_applied": False,
            }
        ]
    )
    val, flag = m2_roundness(rgb, ann)
    assert np.isnan(val)
    assert flag == "low_support"


# ---------- Item 10: forced r_frac < 0.3 -> near_center_degenerate ----------

def test_item10_near_center_degenerate():
    mono = rasterize_disc(1024, 1024, 512, 600, radius=20.0)
    rgb = mono_rgb(mono)
    ann = bubble_grid_annotation_stub(
        [
            {
                "idx": 0,
                "center_xy_px": [512, 600],
                "r_frac": 0.1,  # forced near-center
                "depth_m": 1.91,
                "wrap_applied": False,
            }
        ]
    )
    results = m3_sag_tan(rgb, ann)
    assert len(results) == 1
    assert np.isnan(results[0]["sag_tan"])
    assert results[0]["flag"] == "near_center_degenerate"


# ---------- Item 11: wrong-polarity (dark-on-bright vs bright-on-dark) ----

def test_item11_wrong_polarity_identical_magnitude():
    """Dark disc on bright field vs bright disc on dark field.

    Per the Day-1 post-mortem standing order: wrong-polarity must produce
    identical-magnitude results OR identical NaN reasons. If this fails on
    the real pipeline, mark xfail linking to the LOCKED SPEC line. Here we
    test the metric functions themselves.
    """
    h = w = 1024
    # bright-on-dark
    bright = rasterize_disc(h, w, 512, 512, radius=20.0)
    # dark-on-bright: 1.0 - disc, plus we threshold on ROI.max() so a 1.0
    # background with a 0.0 hole produces no "blob" above the threshold.
    # That's the expected NaN-same-reason branch. Assert both paths reach
    # either "ok with same roundness" or "no_blob" (identical failure mode).
    rgb_bright = mono_rgb(bright)
    rgb_dark = mono_rgb(1.0 - bright)

    ann = bubble_grid_annotation_stub(
        [
            {
                "idx": 0,
                "center_xy_px": [512, 512],
                "r_frac": 0.0,
                "depth_m": 0.70,
                "wrap_applied": False,
            }
        ]
    )
    v1, f1 = m2_roundness(rgb_bright, ann)
    v2, f2 = m2_roundness(rgb_dark, ann)
    # Either both succeed with identical magnitude, or both fail same way.
    if f1 is None and f2 is None:
        assert abs(v1 - v2) < 0.02
    else:
        # same failure reason (polarity invariance on the flag dimension)
        assert f1 == f2, f"polarity flag drift: {f1!r} vs {f2!r}"


# ---------- Item 12: wrap-probe disc gone missing -> wrap_no_blob ----

def test_item12_wrap_probe_zeroed_out():
    h = w = 1024
    # reference disc #3 present; disc #11 region empty.
    mono = rasterize_disc(h, w, 800, 512, radius=20.0)  # disc #3
    rgb = mono_rgb(mono)
    ann = bubble_grid_annotation_stub(standard_grid_discs())
    result = m7_wrap_probe(rgb, ann)
    assert result["finite_ok"] is True
    assert result["support_px"] == 0
    assert result["flag"] == "wrap_no_blob"


# ---------- Item 13: wrap-probe disc NaN -> wrap_produced_nonfinite ----

def test_item13_wrap_probe_nan_roi():
    h = w = 1024
    mono = rasterize_disc(h, w, 800, 512, radius=20.0)  # ref disc #3
    mono[500:525, 700:740] = np.nan  # poison disc #11 ROI
    rgb = mono_rgb(mono)
    ann = bubble_grid_annotation_stub(standard_grid_discs())
    result = m7_wrap_probe(rgb, ann)
    assert result["finite_ok"] is False
    assert result["flag"] == "wrap_produced_nonfinite"


# ---------- Extras required by LOCKED SPEC §5: wrong-shape annotation ----

def test_wrong_shape_annotation_missing_scene():
    rgb = mono_rgb(np.ones((64, 64), dtype=np.float32))
    depth = np.ones((64, 64), dtype=np.float32)
    from summicron.eval.score import score
    with pytest.raises(ValueError, match="scene"):
        score(rgb, depth, {}, input_rgb=rgb)


def test_zero_disc_annotation():
    """Annotation with empty discs list -> M2 returns no_center_disc."""
    rgb = mono_rgb(np.zeros((1024, 1024), dtype=np.float32))
    ann = bubble_grid_annotation_stub([])
    val, flag = m2_roundness(rgb, ann)
    assert np.isnan(val)
    assert flag == "no_center_disc"
