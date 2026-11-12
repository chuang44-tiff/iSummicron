"""Unit tests for summicron.depth.estimate (Day 5 depth-based defocus).

Tests S8, S9, S10 from the spec. No OpenCV. Float32 throughout.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# S8: Flat depth map -> near-zero defocus
# ---------------------------------------------------------------------------

class TestBuildDepthDefocusMap:
    """Tests for build_depth_defocus_map (S4)."""

    def test_flat_depth_returns_near_zero_defocus(self):
        """S8: constant depth 0.5 everywhere -> max defocus < 0.02."""
        from summicron.depth.estimate import build_depth_defocus_map

        depth = np.full((100, 100), 0.5, dtype=np.float32)
        anchor = 0.5
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # flat-scene warning expected
            defocus = build_depth_defocus_map(depth, anchor)

        assert defocus.shape == (100, 100)
        assert defocus.dtype == np.float32
        assert defocus.max() < 0.02, (
            f"Flat depth should give near-zero defocus, got max={defocus.max():.6f}"
        )

    def test_flat_scene_warning_emitted(self):
        """Flat scene triggers a warning."""
        from summicron.depth.estimate import build_depth_defocus_map

        depth = np.full((50, 50), 0.3, dtype=np.float32)
        anchor = 0.3
        with pytest.warns(UserWarning, match="Flat scene"):
            build_depth_defocus_map(depth, anchor)


# ---------------------------------------------------------------------------
# S9: Gradient depth map -> defocus increases away from anchor
# ---------------------------------------------------------------------------

class TestGradientDefocus:
    """Tests for gradient depth -> defocus pattern (S9)."""

    def test_gradient_defocus_increases_from_anchor(self):
        """S9: linear 0->1 left-to-right, anchor near left -> defocus grows rightward."""
        from summicron.depth.estimate import (
            build_depth_defocus_map,
            find_anchor_depth,
        )

        H, W = 100, 200
        # Depth gradient: 0 at left, 1 at right
        depth = np.tile(
            np.linspace(0.0, 1.0, W, dtype=np.float32), (H, 1)
        )
        anchor = find_anchor_depth(depth, percentile=2.0)

        defocus = build_depth_defocus_map(depth, anchor)

        assert defocus.shape == (H, W)
        assert defocus.dtype == np.float32

        # Mean defocus in left quarter (near anchor) should be less than right quarter
        left_mean = defocus[:, : W // 4].mean()
        right_mean = defocus[:, 3 * W // 4 :].mean()
        assert left_mean < right_mean, (
            f"Defocus should increase away from anchor: "
            f"left_mean={left_mean:.4f} >= right_mean={right_mean:.4f}"
        )

    def test_gradient_defocus_range(self):
        """Gradient depth produces defocus in [0, 1]."""
        from summicron.depth.estimate import build_depth_defocus_map

        H, W = 80, 160
        depth = np.tile(
            np.linspace(0.0, 1.0, W, dtype=np.float32), (H, 1)
        )
        anchor = 0.02  # near the closest end

        defocus = build_depth_defocus_map(depth, anchor)
        assert defocus.min() >= 0.0
        assert defocus.max() <= 1.0


# ---------------------------------------------------------------------------
# S10: estimate_depth post-processing with synthetic input (mock-based)
# ---------------------------------------------------------------------------

class TestEstimateDepthPostProcessing:
    """S10: Test estimate_depth output shape and range via mock pipeline."""

    def test_estimate_depth_shape_and_range(self, monkeypatch):
        """Mock the HF pipeline and verify post-processing produces correct
        shape, dtype, and [0,1] range."""
        import torch
        from summicron.depth import estimate as est_mod

        H, W = 64, 80

        # Synthetic disparity: higher in center (closer), lower at edges
        ys = np.linspace(-1, 1, H, dtype=np.float32)
        xs = np.linspace(-1, 1, W, dtype=np.float32)
        yy, xx = np.meshgrid(ys, xs, indexing="ij")
        disp_np = 10.0 - 3.0 * (xx ** 2 + yy ** 2)  # center = 10, corners ~4
        disp_tensor = torch.from_numpy(disp_np)

        class FakePipeline:
            def __call__(self, pil_img):
                # Return at inference resolution, estimate_depth should upsample
                return {"predicted_depth": disp_tensor, "depth": pil_img}

        fake_bundle = {"model": FakePipeline(), "device": "cpu"}

        image_rgb = np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)
        depth = est_mod.estimate_depth(image_rgb, fake_bundle)

        assert depth.shape == (H, W), f"Expected ({H}, {W}), got {depth.shape}"
        assert depth.dtype == np.float32
        assert depth.min() >= 0.0, f"Depth min {depth.min()} < 0"
        assert depth.max() <= 1.0, f"Depth max {depth.max()} > 1"
        # Center should be farther (lower disparity -> higher depth after inversion?
        # No: center has HIGHEST disparity=10, so after inversion depth=0 (near).
        # Corners have lowest disparity=~4, so depth=1 (far). 0=near, 1=far.
        assert depth[H // 2, W // 2] < depth[0, 0], (
            "Center (high disp) should map to lower depth (nearer) than corner"
        )

    def test_estimate_depth_flat_disparity(self, monkeypatch):
        """Flat disparity -> depth = 0.5 everywhere."""
        import torch
        from summicron.depth import estimate as est_mod

        H, W = 32, 32
        disp_tensor = torch.full((H, W), 5.0)

        class FakePipeline:
            def __call__(self, pil_img):
                return {"predicted_depth": disp_tensor, "depth": pil_img}

        fake_bundle = {"model": FakePipeline(), "device": "cpu"}
        image_rgb = np.zeros((H, W, 3), dtype=np.uint8)
        depth = est_mod.estimate_depth(image_rgb, fake_bundle)

        assert depth.shape == (H, W)
        assert np.allclose(depth, 0.5), f"Flat disp should give 0.5, got {depth.mean()}"


# ---------------------------------------------------------------------------
# S3: find_anchor_depth
# ---------------------------------------------------------------------------

class TestFindAnchorDepth:

    def test_anchor_at_percentile(self):
        from summicron.depth.estimate import find_anchor_depth

        depth = np.linspace(0.1, 0.9, 1000, dtype=np.float32).reshape(25, 40)
        anchor = find_anchor_depth(depth, percentile=2.0)
        expected = float(np.percentile(depth, 2.0))
        assert abs(anchor - expected) < 1e-5

    def test_degenerate_anchor_warns(self):
        from summicron.depth.estimate import find_anchor_depth

        depth = np.zeros((10, 10), dtype=np.float32)
        with pytest.warns(UserWarning, match="Degenerate anchor"):
            anchor = find_anchor_depth(depth)
        assert anchor <= 1e-6


# ---------------------------------------------------------------------------
# S7: estimate_and_build_defocus convenience wrapper
# ---------------------------------------------------------------------------

class TestEstimateAndBuildDefocus:

    def test_convenience_returns_correct_structure(self, monkeypatch):
        """Verify the wrapper chains correctly and returns expected types."""
        import torch
        from summicron.depth import estimate as est_mod

        H, W = 48, 64
        # Gradient disparity
        disp_np = np.tile(
            np.linspace(5.0, 10.0, W, dtype=np.float32), (H, 1)
        )
        disp_tensor = torch.from_numpy(disp_np)

        class FakePipeline:
            def __call__(self, pil_img):
                return {"predicted_depth": disp_tensor, "depth": pil_img}

        fake_bundle = {"model": FakePipeline(), "device": "cpu"}
        image_rgb = np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)

        defocus_map, info = est_mod.estimate_and_build_defocus(image_rgb, fake_bundle)

        assert defocus_map.shape == (H, W)
        assert defocus_map.dtype == np.float32
        assert isinstance(info, dict)
        assert "anchor_depth" in info
        assert "depth_range" in info
        assert "flat_scene" in info
        assert isinstance(info["flat_scene"], bool)
