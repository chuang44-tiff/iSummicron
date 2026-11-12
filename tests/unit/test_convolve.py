"""Unit tests for the bin-and-convolve renderer."""
from __future__ import annotations

import numpy as np
import pytest

from summicron.render.convolve import (
    linear_to_srgb,
    srgb_to_linear,
    _memory_guard,
)
from tests.fixtures.defocus import radial_dmap, render_with_defocus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_delta_tensor(ksize: int = 129) -> dict:
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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestDeltaKernelIdentity:
    """Convolving with a delta kernel should approximately reproduce the input."""

    def test_white_dot_ssim(self):
        """128x128 white-dot image -> delta kernel -> output ~ input (SSIM > 0.99)."""
        sz = 128
        # Create a simple white-dot image in sRGB space
        img = np.zeros((sz, sz, 3), dtype=np.float32)
        img[sz // 2, sz // 2, :] = 1.0  # white dot

        tensor = _make_delta_tensor()
        output_u8, output_lin = render_with_defocus(img, tensor)

        # Linearize the input for comparison
        input_lin = srgb_to_linear(img)

        # SSIM check
        try:
            from skimage.metrics import structural_similarity
            ssim = structural_similarity(
                input_lin, output_lin, channel_axis=2, data_range=1.0
            )
            assert ssim > 0.99, f"SSIM {ssim:.4f} <= 0.99"
        except ImportError:
            # Fallback: check max absolute error
            diff = np.abs(output_lin - input_lin)
            assert diff.max() < 0.05, f"Max diff {diff.max():.4f} >= 0.05"


class TestEnergyConservation:
    """Energy should be conserved within [0.85, 1.15]."""

    def test_gradient_image(self):
        """Synthetic gradient image energy is preserved."""
        sz = 128
        # Horizontal gradient in sRGB space
        grad = np.linspace(0.1, 0.9, sz, dtype=np.float32)
        img = np.zeros((sz, sz, 3), dtype=np.float32)
        img[:, :, :] = grad[None, :, None]

        tensor = _make_delta_tensor()
        _output_u8, output_lin = render_with_defocus(img, tensor)
        input_lin = srgb_to_linear(img)

        ratio = float(output_lin.sum()) / float(input_lin.sum())
        assert 0.85 <= ratio <= 1.15, f"Energy ratio {ratio:.4f} outside [0.85, 1.15]"


class TestMemoryGuard:
    """Images exceeding the byte-budget envelope must be rejected.

    P1.audit Item 6: ``_memory_guard`` signature now takes a third ``d``
    (D-axis size from the loaded tensor) for byte-budget projection.
    The 36 MP fast-path early-gate is preserved.
    """

    def test_oversized_rejected(self):
        with pytest.raises(RuntimeError, match="Image too large"):
            # 7000² = 49 MP > 36 MP fast-path gate; D doesn't matter.
            _memory_guard(7000, 7000, d=11)

    def test_acceptable_size(self):
        # R-native-mem (locked-spec-mem.md §8): the honest estimator admits
        # 4096x3072 D=11 (the old 6000x6000 D=1 case was deleted; its estimate
        # is recorded in coder-mem-report.md).
        _memory_guard(4096, 3072, d=11)


class TestDefocusMap:
    """Defocus map sanity checks."""

    def test_center_zero(self):
        dmap = radial_dmap(101, 101)
        assert dmap[50, 50] == pytest.approx(0.0, abs=1e-5)

    def test_corners_one(self):
        dmap = radial_dmap(101, 101)
        assert dmap[0, 0] == pytest.approx(1.0, abs=1e-3)
        assert dmap[0, 100] == pytest.approx(1.0, abs=1e-3)
        assert dmap[100, 0] == pytest.approx(1.0, abs=1e-3)
        assert dmap[100, 100] == pytest.approx(1.0, abs=1e-3)


class TestSrgbRoundtrip:
    """sRGB -> linear -> sRGB should be near-identity."""

    def test_roundtrip(self):
        x = np.linspace(0, 1, 256, dtype=np.float32)
        y = linear_to_srgb(srgb_to_linear(x))
        assert np.allclose(x, y, atol=1e-5)
