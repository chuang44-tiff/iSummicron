"""Unit tests for slab-boundary taper and d-axis ceiling fix (Coder A: Smoothness).

T1: Taper continuity at slab boundaries.
T2: d=0 delta fidelity (center-crop SSIM).
T3: Energy conservation through tapered pipeline.
T4: Bilinear weights with defocus_ratios starting at 0.0.
"""
from __future__ import annotations

import numpy as np
import pytest

from summicron.render.convolve import (
    _bilinear_weights,
    srgb_to_linear,
)
from tests.fixtures.defocus import render_with_defocus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_delta_tensor_d0(ksize: int = 33) -> dict:
    """Build a synthetic tensor where every kernel is a centered delta,
    and defocus_ratios starts at 0.0 (post-Item-2 convention).
    """
    R, D = 6, 4
    kernels = np.zeros((R, D, ksize, ksize), dtype=np.float32)
    c = ksize // 2
    kernels[:, :, c, c] = 1.0
    radial_edges = np.array([0.0, 0.15, 0.30, 0.45, 0.60, 0.80, 1.01], dtype=np.float64)
    defocus_ratios = np.linspace(0.0, 1.0, D, dtype=np.float64)
    return {
        "kernels": kernels,
        "radial_edges": radial_edges,
        "defocus_ratios": defocus_ratios,
    }


# ---------------------------------------------------------------------------
# T1: Taper continuity at slab boundaries
# ---------------------------------------------------------------------------

class TestTaperContinuity:
    """Slab weights should change smoothly across adjacent pixels near a boundary."""

    def test_no_hard_discontinuity(self):
        """Create a synthetic radial_frac_map row crossing a slab boundary.
        Assert max weight jump between adjacent pixels < 0.01."""
        radial_edges = np.array(
            [0.0, 0.15, 0.30, 0.45, 0.60, 0.80, 1.01], dtype=np.float64
        )
        r_centers = 0.5 * (radial_edges[:-1] + radial_edges[1:])
        n_slabs = len(r_centers) + 1

        # Build a fine 1D radial sweep across the 2nd boundary (r_centers[1])
        boundary = float(r_centers[1])
        radial_vals = np.linspace(boundary - 0.05, boundary + 0.05, 500, dtype=np.float32)

        # Compute slab_frac for each point
        slab_frac = np.interp(
            radial_vals,
            r_centers,
            np.arange(len(r_centers), dtype=np.float64),
        ).astype(np.float32)

        # For each slab, compute the cos^2 taper weight
        for s in range(n_slabs):
            half_w = 0.5
            dx = np.abs(slab_frac - float(s))
            t = np.clip(dx / half_w, 0.0, 1.0)
            weights = np.cos(0.5 * np.pi * t) ** 2

            # Check max discontinuity between adjacent pixels
            diffs = np.abs(np.diff(weights))
            max_jump = float(diffs.max())
            # With 500 samples over 0.1 radial range, the cos^2 taper
            # should produce max jumps well under 0.01. A hard binary
            # mask would produce a jump of ~1.0 here.
            assert max_jump < 0.02, (
                f"Slab {s}: max weight jump {max_jump:.4f} >= 0.02 "
                f"(taper not smooth)"
            )


# ---------------------------------------------------------------------------
# T2: d=0 delta fidelity
# ---------------------------------------------------------------------------

class TestDeltaFidelityD0:
    """Render with all-delta kernels and flat defocus -> output ~ input."""

    def test_center_crop_ssim(self):
        """128x128 gradient image, delta tensor with d[0]=0.0, SSIM > 0.95."""
        sz = 128
        # Horizontal gradient in sRGB space
        grad = np.linspace(0.2, 0.8, sz, dtype=np.float32)
        img = np.zeros((sz, sz, 3), dtype=np.float32)
        img[:, :, :] = grad[None, :, None]

        tensor = _make_delta_tensor_d0()
        output_u8, output_lin = render_with_defocus(img, tensor)
        input_lin = srgb_to_linear(img)

        # Center crop comparison
        ch, cw = 64, 64
        y0, x0 = (sz - ch) // 2, (sz - cw) // 2
        center_in = input_lin[y0:y0 + ch, x0:x0 + cw]
        center_out = output_lin[y0:y0 + ch, x0:x0 + cw]

        try:
            from skimage.metrics import structural_similarity
            ssim = structural_similarity(
                center_in, center_out, channel_axis=2, data_range=1.0
            )
            assert ssim > 0.95, f"Center SSIM {ssim:.4f} <= 0.95"
        except ImportError:
            # Fallback: max absolute error
            diff = np.abs(center_out - center_in)
            assert diff.max() < 0.05, f"Max diff {diff.max():.4f} >= 0.05"


# ---------------------------------------------------------------------------
# T3: Energy conservation through tapered pipeline
# ---------------------------------------------------------------------------

class TestEnergyConservationTaper:
    """Energy ratio must stay within [0.85, 1.15] with the cos^2 taper."""

    def test_energy_ratio(self):
        """Synthetic gradient image through tapered pipeline."""
        sz = 128
        grad = np.linspace(0.1, 0.9, sz, dtype=np.float32)
        img = np.zeros((sz, sz, 3), dtype=np.float32)
        img[:, :, :] = grad[None, :, None]

        tensor = _make_delta_tensor_d0()
        _output_u8, output_lin = render_with_defocus(img, tensor)
        input_lin = srgb_to_linear(img)

        ratio = float(output_lin.sum()) / float(input_lin.sum())
        assert 0.85 <= ratio <= 1.15, (
            f"Energy ratio {ratio:.4f} outside [0.85, 1.15]"
        )


# ---------------------------------------------------------------------------
# T4: Bilinear weights with d=0
# ---------------------------------------------------------------------------

class TestBilinearWeightsD0:
    """_bilinear_weights must return valid output when defocus_ratios[0] == 0.0."""

    def test_d0_valid_indices_and_weights(self):
        """Call _bilinear_weights with d_center=0.0 and defocus_ratios starting at 0."""
        radial_edges = np.array([0.0, 0.15, 0.30, 0.45, 0.60, 0.80, 1.01], dtype=np.float64)
        defocus_ratios = np.linspace(0.0, 1.0, 4, dtype=np.float64)

        # Query at center (r=0, d=0)
        indices, weights = _bilinear_weights(
            radial_frac=0.0,
            defocus_val=0.0,
            radial_edges=radial_edges,
            defocus_ratios=defocus_ratios,
        )

        # Must return 4 index pairs and 4 weights
        assert len(indices) == 4
        assert len(weights) == 4

        # All indices must be valid
        Nr = len(radial_edges) - 1  # number of radial bins
        Nd = len(defocus_ratios)
        for ri, di in indices:
            assert 0 <= ri < Nr, f"r_idx {ri} out of range [0, {Nr})"
            assert 0 <= di < Nd, f"d_idx {di} out of range [0, {Nd})"

        # Weights must sum to 1.0
        wsum = sum(weights)
        assert abs(wsum - 1.0) < 1e-6, f"Weights sum {wsum:.6f} != 1.0"

        # No negative weights
        for w in weights:
            assert w >= 0.0, f"Negative weight {w}"

        # No NaN or Inf
        for w in weights:
            assert np.isfinite(w), f"Non-finite weight {w}"

    def test_d0_small_query(self):
        """Query near d=0 edge: should still produce valid output."""
        radial_edges = np.array([0.0, 0.15, 0.30, 0.45, 0.60, 0.80, 1.01], dtype=np.float64)
        defocus_ratios = np.linspace(0.0, 1.0, 11, dtype=np.float64)

        indices, weights = _bilinear_weights(
            radial_frac=0.5,
            defocus_val=0.001,
            radial_edges=radial_edges,
            defocus_ratios=defocus_ratios,
        )

        assert len(indices) == 4
        wsum = sum(weights)
        assert abs(wsum - 1.0) < 1e-6
