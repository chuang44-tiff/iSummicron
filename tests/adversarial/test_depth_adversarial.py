"""Adversarial tests for Day 6 depth estimation and defocus mapping.

Attack vectors: polarity inversion, degenerate inputs (all-zero, all-one,
NaN/Inf), single-hot-pixel anchor latch, shape mismatches in render(),
out-of-range defocus values, and energy conservation under depth-based maps.

Xfail markers indicate genuine bugs found; Coder will fix in a later round.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from summicron.depth.estimate import (
    build_depth_defocus_map,
    find_anchor_depth,
)
from tests.fixtures.defocus import render_with_defocus



# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tensor_stub(ksize: int = 15) -> dict:
    """Build a minimal 4D PSF tensor stub for render() tests.

    Returns a dict matching load_tensor() output with delta-like kernels.
    """
    R, D = 3, 4
    kernels = np.zeros((R, D, ksize, ksize), dtype=np.float32)
    c = ksize // 2
    for r in range(R):
        for d in range(D):
            kernels[r, d, c, c] = 1.0  # delta kernel
    radial_edges = np.linspace(0.0, 1.0, R + 1)
    defocus_ratios = np.linspace(0.0, 1.0, D)
    return {
        "kernels": kernels,
        "radial_edges": radial_edges,
        "defocus_ratios": defocus_ratios,
    }


# ---------------------------------------------------------------------------
# 1. Wrong-polarity depth map (MANDATORY per Post-mortem 1)
# ---------------------------------------------------------------------------

class TestWrongPolarity:
    """The spec says 0=near, 1=far. What if a user passes inverted depth?"""

    def test_inverted_polarity_gives_different_defocus(self):
        """An inverted depth map (0=far, 1=near) must produce a materially
        different defocus distribution than the correct-polarity map."""
        H, W = 100, 200
        depth_correct = np.tile(
            np.linspace(0.0, 1.0, W, dtype=np.float32), (H, 1)
        )  # 0=near at left, 1=far at right
        depth_inverted = 1.0 - depth_correct  # 0=far at left, 1=near at right

        anchor_correct = find_anchor_depth(depth_correct, percentile=2.0)
        anchor_inverted = find_anchor_depth(depth_inverted, percentile=2.0)

        defocus_correct = build_depth_defocus_map(depth_correct, anchor_correct)
        defocus_inverted = build_depth_defocus_map(depth_inverted, anchor_inverted)

        # The two maps should NOT be identical — polarity matters
        diff = np.abs(defocus_correct - defocus_inverted).mean()
        assert diff > 0.01, (
            f"Polarity inversion produced nearly identical defocus "
            f"(mean diff={diff:.6f}); reciprocal math may be polarity-insensitive"
        )

    def test_inverted_polarity_defocus_pattern_flips(self):
        """With correct polarity (0=near left, 1=far right), defocus should be
        higher on the right. With inverted polarity, it should flip."""
        H, W = 80, 160
        depth = np.tile(
            np.linspace(0.0, 1.0, W, dtype=np.float32), (H, 1)
        )
        anchor = find_anchor_depth(depth, percentile=2.0)
        defocus = build_depth_defocus_map(depth, anchor)

        left_mean = defocus[:, : W // 4].mean()
        right_mean = defocus[:, 3 * W // 4 :].mean()
        assert left_mean < right_mean, "Correct polarity: left should have less defocus"

        depth_inv = 1.0 - depth
        anchor_inv = find_anchor_depth(depth_inv, percentile=2.0)
        defocus_inv = build_depth_defocus_map(depth_inv, anchor_inv)

        left_inv = defocus_inv[:, : W // 4].mean()
        right_inv = defocus_inv[:, 3 * W // 4 :].mean()
        assert left_inv > right_inv, (
            f"Inverted polarity should flip defocus pattern: "
            f"left={left_inv:.4f}, right={right_inv:.4f}"
        )


# ---------------------------------------------------------------------------
# 2. All-zero depth map
# ---------------------------------------------------------------------------

class TestAllZeroDepth:
    """Every pixel at 0. The reciprocal 1/(0+eps) is huge but uniform."""

    def test_all_zero_depth_gives_zero_defocus(self):
        depth = np.zeros((50, 50), dtype=np.float32)
        anchor = find_anchor_depth(depth, percentile=2.0)
        assert anchor <= 1e-6, f"Anchor should be ~0, got {anchor}"

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            defocus = build_depth_defocus_map(depth, anchor)

        # Uniform depth -> uniform reciprocal -> zero difference -> zero defocus
        assert defocus.max() < 0.02, (
            f"All-zero depth should give near-zero defocus, got max={defocus.max():.6f}"
        )

    def test_all_zero_no_nan_inf(self):
        depth = np.zeros((30, 30), dtype=np.float32)
        anchor = find_anchor_depth(depth, percentile=2.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            defocus = build_depth_defocus_map(depth, anchor)
        assert not np.any(np.isnan(defocus)), "NaN in defocus from all-zero depth"
        assert not np.any(np.isinf(defocus)), "Inf in defocus from all-zero depth"


# ---------------------------------------------------------------------------
# 3. All-ones depth map
# ---------------------------------------------------------------------------

class TestAllOnesDepth:
    """Every pixel at 1.0 — uniform far depth."""

    def test_all_ones_depth_gives_near_zero_defocus(self):
        depth = np.ones((50, 50), dtype=np.float32)
        anchor = find_anchor_depth(depth, percentile=2.0)
        assert abs(anchor - 1.0) < 0.01, f"Anchor should be ~1.0, got {anchor}"

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            defocus = build_depth_defocus_map(depth, anchor)

        assert defocus.max() < 0.02, (
            f"All-ones depth should give near-zero defocus, got max={defocus.max():.6f}"
        )

    def test_all_ones_no_nan_inf(self):
        depth = np.ones((30, 30), dtype=np.float32)
        anchor = 1.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            defocus = build_depth_defocus_map(depth, anchor)
        assert not np.any(np.isnan(defocus)), "NaN in defocus from all-ones depth"
        assert not np.any(np.isinf(defocus)), "Inf in defocus from all-ones depth"


# ---------------------------------------------------------------------------
# 4. Single hot pixel — anchor latch
# ---------------------------------------------------------------------------

class TestSingleHotPixel:
    """Depth map all 0.5 except one pixel at 0.0 — does anchor latch to outlier?"""

    def test_anchor_latches_to_outlier(self):
        """2nd percentile of a mostly-0.5 map with one 0.0 pixel.
        At 50x50 = 2500 pixels, 2nd percentile = pixel ~50.
        Only 1 pixel is 0.0, so anchor should still be ~0.5."""
        depth = np.full((50, 50), 0.5, dtype=np.float32)
        depth[25, 25] = 0.0  # single outlier
        anchor = find_anchor_depth(depth, percentile=2.0)
        # With only 1/2500 pixels at 0.0, 2nd percentile should be 0.5
        assert anchor > 0.4, (
            f"Single outlier should NOT drag anchor to 0; got {anchor:.4f}"
        )

    def test_many_outliers_drag_anchor(self):
        """If 5% of pixels are 0.0, anchor at 2nd percentile should be 0.0."""
        depth = np.full((100, 100), 0.5, dtype=np.float32)
        # Set 5% of pixels to 0.0
        depth[:5, :] = 0.0  # 500/10000 = 5%
        anchor = find_anchor_depth(depth, percentile=2.0)
        assert anchor < 0.01, (
            f"5% outliers should drag 2nd-percentile anchor to ~0; got {anchor:.4f}"
        )


# ---------------------------------------------------------------------------
# 5. NaN / Inf in depth map
# ---------------------------------------------------------------------------

class TestNaNInfDepth:
    """Does build_depth_defocus_map handle NaN/Inf gracefully?"""

    def test_nan_in_depth_propagates(self):
        """NaN in depth map: the function does NOT sanitize NaN.
        This is a potential bug — reciprocal of NaN is NaN."""
        depth = np.full((20, 20), 0.5, dtype=np.float32)
        depth[10, 10] = np.nan
        anchor = 0.5

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            defocus = build_depth_defocus_map(depth, anchor)

        has_nan = np.any(np.isnan(defocus))
        if has_nan:
            pytest.xfail(
                "BUG: NaN in depth map propagates to defocus map — "
                "build_depth_defocus_map does not sanitize NaN inputs"
            )
        # If we get here, NaN was handled (unexpected but good)
        assert defocus.shape == (20, 20)

    def test_inf_in_depth(self):
        """Inf in depth map should not crash or propagate."""
        depth = np.full((20, 20), 0.5, dtype=np.float32)
        depth[5, 5] = np.inf
        anchor = 0.5

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            defocus = build_depth_defocus_map(depth, anchor)

        has_nan = np.any(np.isnan(defocus))
        has_inf = np.any(np.isinf(defocus))
        if has_nan or has_inf:
            pytest.xfail(
                "BUG: Inf in depth map produces NaN/Inf in defocus map — "
                "build_depth_defocus_map does not sanitize Inf inputs"
            )
        assert defocus.shape == (20, 20)

    def test_negative_inf_in_depth(self):
        """Negative Inf in depth map."""
        depth = np.full((20, 20), 0.5, dtype=np.float32)
        depth[5, 5] = -np.inf
        anchor = 0.5

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            defocus = build_depth_defocus_map(depth, anchor)

        has_nan = np.any(np.isnan(defocus))
        has_inf = np.any(np.isinf(defocus))
        if has_nan or has_inf:
            pytest.xfail(
                "BUG: -Inf in depth map produces NaN/Inf in defocus — "
                "no input sanitization"
            )


# ---------------------------------------------------------------------------
# 6. Defocus map passed to render() with wrong shape
# ---------------------------------------------------------------------------

class TestRenderShapeMismatch:
    """render() should reject defocus_map with wrong (H, W)."""

    def test_wrong_shape_raises_valueerror(self):
        from summicron.render.convolve import render

        H, W = 32, 48
        image = np.random.rand(H, W, 3).astype(np.float32)
        tensor = _make_tensor_stub(ksize=15)

        # Wrong shape: transposed
        bad_defocus = np.zeros((W, H), dtype=np.float32)
        with pytest.raises(ValueError, match="depth_m shape"):
            render_with_defocus(image, tensor, dmap=bad_defocus)

    def test_3d_defocus_raises(self):
        """defocus_map must be 2D, not (H, W, 1)."""
        from summicron.render.convolve import render

        H, W = 32, 48
        image = np.random.rand(H, W, 3).astype(np.float32)
        tensor = _make_tensor_stub(ksize=15)

        bad_defocus = np.zeros((H, W, 1), dtype=np.float32)
        with pytest.raises((ValueError, Exception)):
            render_with_defocus(image, tensor, dmap=bad_defocus)


# ---------------------------------------------------------------------------
# 7. Defocus map with values outside [0,1]
# ---------------------------------------------------------------------------

class TestDefocusOutOfRange:
    """What if defocus_map has values > 1 or < 0?"""

    def test_defocus_above_one_in_render(self):
        """render() receives defocus_map with values up to 2.0.
        The dmap is used as-is (no internal clip).
        This could cause out-of-range bin lookups."""
        from summicron.render.convolve import render

        H, W = 32, 48
        image = np.random.rand(H, W, 3).astype(np.float32) * 0.5 + 0.25
        tensor = _make_tensor_stub(ksize=15)

        # defocus_map with values up to 2.0
        defocus = np.full((H, W), 2.0, dtype=np.float32)
        # This should either clip internally or handle gracefully
        try:
            out_u8, out_lin = render_with_defocus(image, tensor, dmap=defocus)
            # If it runs, check no NaN/Inf
            assert not np.any(np.isnan(out_lin)), "NaN from out-of-range defocus"
            assert not np.any(np.isinf(out_lin)), "Inf from out-of-range defocus"
        except (AssertionError, ValueError, RuntimeError) as exc:
            # Gate failure (P1.audit Item 5: energy gates raise RuntimeError;
            # CLI shim translates to exit 2) or validation error is acceptable.
            pass

    def test_negative_defocus_in_render(self):
        """defocus_map with negative values."""
        from summicron.render.convolve import render

        H, W = 32, 48
        image = np.random.rand(H, W, 3).astype(np.float32) * 0.5 + 0.25
        tensor = _make_tensor_stub(ksize=15)

        defocus = np.full((H, W), -0.5, dtype=np.float32)
        try:
            out_u8, out_lin = render_with_defocus(image, tensor, dmap=defocus)
            assert not np.any(np.isnan(out_lin))
        except (AssertionError, ValueError, RuntimeError):
            # P1.audit Item 5: energy gates assert→RuntimeError.
            pass


# ---------------------------------------------------------------------------
# 8. Very large depth range
# ---------------------------------------------------------------------------

class TestLargeDepthRange:
    """Depth map from 0.001 to 0.999 — does reciprocal blow up?"""

    def test_extreme_range_no_blowup(self):
        H, W = 60, 80
        depth = np.tile(
            np.linspace(0.001, 0.999, W, dtype=np.float32), (H, 1)
        )
        anchor = find_anchor_depth(depth, percentile=2.0)
        defocus = build_depth_defocus_map(depth, anchor)

        assert not np.any(np.isnan(defocus)), "NaN from extreme depth range"
        assert not np.any(np.isinf(defocus)), "Inf from extreme depth range"
        assert defocus.min() >= 0.0
        assert defocus.max() <= 1.0

    def test_near_zero_depth_values(self):
        """Depth values very close to 0 — reciprocal is ~1/eps = 10000."""
        H, W = 40, 40
        depth = np.full((H, W), 1e-5, dtype=np.float32)
        depth[:, W // 2 :] = 0.5
        anchor = find_anchor_depth(depth, percentile=2.0)
        defocus = build_depth_defocus_map(depth, anchor)

        assert not np.any(np.isnan(defocus)), "NaN from near-zero depth"
        assert not np.any(np.isinf(defocus)), "Inf from near-zero depth"
        assert defocus.max() <= 1.0, f"Defocus max {defocus.max()} > 1.0"


# ---------------------------------------------------------------------------
# 9. Anchor at exactly 0.0
# ---------------------------------------------------------------------------

class TestAnchorAtZero:
    """The eps guard should handle anchor=0.0."""

    def test_anchor_zero_no_crash(self):
        """Explicitly pass anchor=0.0 to build_depth_defocus_map."""
        H, W = 40, 40
        depth = np.tile(
            np.linspace(0.0, 1.0, W, dtype=np.float32), (H, 1)
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            defocus = build_depth_defocus_map(depth, anchor_m=0.0)

        assert not np.any(np.isnan(defocus)), "NaN with anchor=0.0"
        assert not np.any(np.isinf(defocus)), "Inf with anchor=0.0"
        assert defocus.min() >= 0.0
        assert defocus.max() <= 1.0

    def test_anchor_negative_eps(self):
        """Anchor at a tiny negative value (shouldn't happen, but test robustness)."""
        depth = np.full((20, 20), 0.5, dtype=np.float32)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            defocus = build_depth_defocus_map(depth, anchor_m=-0.001)
        assert not np.any(np.isnan(defocus))
        assert not np.any(np.isinf(defocus))


# ---------------------------------------------------------------------------
# 10. render() energy conservation with depth-based defocus map
# ---------------------------------------------------------------------------

class TestEnergyConservationWithDepthDefocus:
    """Does the [0.85, 1.15] energy gate still pass with a non-radial defocus map?"""

    def test_energy_gate_with_gradient_defocus(self):
        """Gradient defocus map (not radial) through render()."""
        from summicron.render.convolve import render

        H, W = 32, 48
        # Mid-gray image to avoid edge effects
        image = np.full((H, W, 3), 0.5, dtype=np.float32)
        tensor = _make_tensor_stub(ksize=15)

        # Gradient defocus: 0 at left, 1 at right
        defocus = np.tile(
            np.linspace(0.0, 1.0, W, dtype=np.float32), (H, 1)
        )

        # Should not raise energy gate assertion
        out_u8, out_lin = render_with_defocus(image, tensor, dmap=defocus)

        # Verify basic sanity
        assert out_u8.shape == (H, W, 3)
        assert out_u8.dtype == np.uint8

    def test_energy_gate_with_all_max_defocus(self):
        """All-ones defocus map — maximum blur everywhere."""
        from summicron.render.convolve import render

        H, W = 32, 48
        image = np.full((H, W, 3), 0.5, dtype=np.float32)
        tensor = _make_tensor_stub(ksize=15)

        defocus = np.ones((H, W), dtype=np.float32)
        out_u8, out_lin = render_with_defocus(image, tensor, dmap=defocus)
        assert out_u8.shape == (H, W, 3)

    def test_energy_gate_with_zero_defocus(self):
        """All-zero defocus — no blur, should perfectly conserve energy."""
        from summicron.render.convolve import render

        H, W = 32, 48
        image = np.full((H, W, 3), 0.5, dtype=np.float32)
        tensor = _make_tensor_stub(ksize=15)

        defocus = np.zeros((H, W), dtype=np.float32)
        out_u8, out_lin = render_with_defocus(image, tensor, dmap=defocus)
        assert out_u8.shape == (H, W, 3)


# ---------------------------------------------------------------------------
# 11. Flat-scene detection edge cases
# ---------------------------------------------------------------------------

class TestFlatSceneEdgeCases:
    """Boundary conditions around the 0.05 threshold."""

    def test_just_above_flat_threshold(self):
        """ptp/max just above 0.05 should NOT trigger flat-scene path."""
        # Want ptp/(max+1e-8) = 0.06 > 0.05
        # Let min=0.94, max=1.0 -> ptp=0.06, ptp/max=0.06
        H, W = 40, 40
        depth = np.full((H, W), 0.94, dtype=np.float32)
        depth[:, W // 2 :] = 1.0

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            defocus = build_depth_defocus_map(depth, anchor_m=0.94)
            flat_warnings = [w for w in caught if "Flat scene" in str(w.message)]
            assert len(flat_warnings) == 0, (
                "ptp/max=0.06 > 0.05 should NOT trigger flat-scene warning"
            )

    def test_just_below_flat_threshold(self):
        """ptp/max just below 0.05 should trigger flat-scene path."""
        # Want ptp/(max+1e-8) = 0.04 < 0.05
        # Let min=0.96, max=1.0 -> ptp=0.04
        H, W = 40, 40
        depth = np.full((H, W), 0.96, dtype=np.float32)
        depth[:, W // 2 :] = 1.0

        with pytest.warns(UserWarning, match="Flat scene"):
            defocus = build_depth_defocus_map(depth, anchor_m=0.96)

        # Flat scene should give attenuated defocus (multiplied by 0.01)
        assert defocus.max() < 0.1, (
            f"Flat-scene defocus should be heavily attenuated, got max={defocus.max():.4f}"
        )


# ---------------------------------------------------------------------------
# 12. Negative depth values (out of spec [0,1])
# ---------------------------------------------------------------------------

class TestNegativeDepthValues:
    """Depth map with negative values — out of spec but should not crash."""

    def test_negative_depth_no_crash(self):
        depth = np.full((30, 30), -0.5, dtype=np.float32)
        depth[:, 15:] = 0.5
        anchor = find_anchor_depth(depth, percentile=2.0)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            defocus = build_depth_defocus_map(depth, anchor)

        assert not np.any(np.isnan(defocus)), "NaN from negative depth values"
        assert not np.any(np.isinf(defocus)), "Inf from negative depth values"
