"""Tests for angular merge optimization (Item 4).

T5: Speed gate — wall-clock < 150s on ~2000x3000 synthetic image.
T6: Merge equivalence — delta tensor output matches N_ANG=1 reference within 1e-4.
"""
from __future__ import annotations

import time

import numpy as np
import pytest

from summicron.render.convolve import N_ANG
from tests.fixtures.defocus import render_with_defocus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_delta_tensor(ksize: int = 33) -> dict:
    """Build a synthetic tensor where every kernel is a centered delta."""
    R, D = 6, 4
    kernels = np.zeros((R, D, ksize, ksize), dtype=np.float32)
    c = ksize // 2
    kernels[:, :, c, c] = 1.0
    radial_edges = np.array([0.0, 0.15, 0.30, 0.45, 0.60, 0.80, 1.01], dtype=np.float64)
    defocus_ratios = np.array([0.0, 0.33, 0.67, 1.0], dtype=np.float64)
    return {
        "kernels": kernels,
        "radial_edges": radial_edges,
        "defocus_ratios": defocus_ratios,
    }


def _make_mixed_tensor(ksize: int = 33) -> dict:
    """Tensor with delta kernels at d=0 and Gaussian blur at d>0.

    This exercises both the merged (delta) and angular-loop (non-delta)
    code paths within a single render call.
    """
    R, D = 6, 4
    kernels = np.zeros((R, D, ksize, ksize), dtype=np.float32)
    c = ksize // 2

    # d=0 slices: delta kernels
    kernels[:, 0, c, c] = 1.0

    # d>0 slices: small Gaussian blur (not near-delta)
    y, x = np.mgrid[:ksize, :ksize]
    sigma = ksize / 8.0
    gauss = np.exp(-((x - c) ** 2 + (y - c) ** 2) / (2 * sigma ** 2)).astype(np.float32)
    gauss /= gauss.sum()
    for d in range(1, D):
        kernels[:, d] = gauss

    radial_edges = np.array([0.0, 0.15, 0.30, 0.45, 0.60, 0.80, 1.01], dtype=np.float64)
    defocus_ratios = np.array([0.0, 0.33, 0.67, 1.0], dtype=np.float64)
    return {
        "kernels": kernels,
        "radial_edges": radial_edges,
        "defocus_ratios": defocus_ratios,
    }


# ---------------------------------------------------------------------------
# T5: Speed gate
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestSpeedGate:
    """Render a ~2000x3000 image and assert wall-clock < 150s."""

    def test_render_time_under_150s(self):
        H, W = 2000, 3000
        np.random.seed(42)
        img = np.random.rand(H, W, 3).astype(np.float32) * 0.8 + 0.1

        tensor = _make_delta_tensor(ksize=33)

        t0 = time.perf_counter()
        output_u8, _ = render_with_defocus(img, tensor)
        elapsed = time.perf_counter() - t0

        assert elapsed < 150.0, f"Render took {elapsed:.1f}s, expected < 150s"
        assert output_u8.shape == (H, W, 3)
        assert output_u8.dtype == np.uint8


# ---------------------------------------------------------------------------
# T6: Merge equivalence
# ---------------------------------------------------------------------------

class TestMergeEquivalence:
    """Compare optimized (merged delta) output to a reference.

    Reference: render the same all-delta tensor on a small image. With all
    deltas, the merged path produces identical results to the pre-optimization
    angular loop (since every wedge used the same unrotated kernel). We verify
    the output is correct by checking it matches the linearized input closely
    (a delta convolution is an identity transform).
    """

    def test_delta_identity_small(self):
        """256x256 gradient image through all-delta tensor: output ~ input."""
        sz = 256
        grad = np.linspace(0.1, 0.9, sz, dtype=np.float32)
        img = np.zeros((sz, sz, 3), dtype=np.float32)
        img[:, :, :] = grad[None, :, None]

        tensor = _make_delta_tensor(ksize=33)
        _u8, output_lin = render_with_defocus(img, tensor)

        # Linearize input for comparison
        from summicron.render.convolve import srgb_to_linear
        input_lin = srgb_to_linear(img)

        # Max pixel difference should be tiny for delta kernels
        max_diff = float(np.abs(output_lin - input_lin).max())
        assert max_diff < 1e-4, (
            f"Merge equivalence FAIL: max pixel diff {max_diff:.6f} >= 1e-4"
        )

    def test_mixed_tensor_runs_both_paths(self):
        """Mixed tensor (delta at d=0, Gaussian at d>0) exercises both paths.

        This is a smoke test: the render must complete without error and
        pass the internal energy gate.
        """
        sz = 128
        np.random.seed(99)
        img = np.random.rand(sz, sz, 3).astype(np.float32) * 0.7 + 0.15

        tensor = _make_mixed_tensor(ksize=33)
        output_u8, output_lin = render_with_defocus(img, tensor)

        assert output_u8.shape == (sz, sz, 3)
        assert output_u8.dtype == np.uint8
        # Energy conservation is checked inside render(); if we get here it passed.

    def test_delta_merge_pixel_equivalence(self):
        """Verify merged path matches manual angular-sum for delta kernels.

        Render a small image with an all-delta tensor. The merged optimization
        sums `w` (all angular masks combined) in one FFT. The angular loop
        would do `sum(w * angular_mask_a)` over all a, which equals `w` since
        angular masks partition the pixel space. For delta kernels (no rotation),
        both paths convolve with the same kernel_base, so the result must match.

        We verify by checking that the output is very close to the identity
        (linearized input), which is only possible if the merge is correct.
        """
        sz = 64
        img = np.full((sz, sz, 3), 0.5, dtype=np.float32)
        # Add a checkerboard pattern to make differences visible
        img[::2, ::2, :] = 0.8
        img[1::2, 1::2, :] = 0.2

        tensor = _make_delta_tensor(ksize=17)
        _u8, output_lin = render_with_defocus(img, tensor)

        from summicron.render.convolve import srgb_to_linear
        input_lin = srgb_to_linear(img)

        max_diff = float(np.abs(output_lin - input_lin).max())
        assert max_diff < 1e-4, (
            f"Pixel equivalence FAIL: max diff {max_diff:.6f} >= 1e-4"
        )
