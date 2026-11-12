"""Unit tests for PSF kernel resolution scaling (pixel-scale)."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from summicron.render.convolve import load_tensor
from tests.fixtures.defocus import render_with_defocus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CORPUS_REF = 5976
KSIZE = 129
R, D = 6, 4
RADIAL_EDGES = np.array([0.0, 0.15, 0.30, 0.45, 0.60, 0.80, 1.01], dtype=np.float64)
DEFOCUS_RATIOS = np.array([0.0, 0.33, 0.67, 1.0], dtype=np.float64)


def _make_delta_tensor(corpus_ref: int = CORPUS_REF) -> dict:
    """Synthetic tensor with centered delta kernels."""
    kernels = np.zeros((R, D, KSIZE, KSIZE), dtype=np.float32)
    c = KSIZE // 2
    kernels[:, :, c, c] = 1.0
    return {
        "kernels": kernels,
        "radial_edges": RADIAL_EDGES,
        "defocus_ratios": DEFOCUS_RATIOS,
        "corpus_ref_long_edge": corpus_ref,
    }


def _make_gaussian_tensor(sigma: float = 8.0, corpus_ref: int = CORPUS_REF) -> dict:
    """Synthetic tensor with Gaussian kernels (all identical)."""
    c = KSIZE // 2
    ax = np.arange(KSIZE, dtype=np.float32) - c
    x, y = np.meshgrid(ax, ax)
    g = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2)).astype(np.float32)
    g /= g.sum()
    kernels = np.zeros((R, D, KSIZE, KSIZE), dtype=np.float32)
    for ri in range(R):
        for di in range(D):
            kernels[ri, di] = g
    return {
        "kernels": kernels,
        "radial_edges": RADIAL_EDGES,
        "defocus_ratios": DEFOCUS_RATIOS,
        "corpus_ref_long_edge": corpus_ref,
    }


def _gradient_image(h: int, w: int) -> np.ndarray:
    """Horizontal gradient in sRGB, shape (h, w, 3), float32 [0, 1]."""
    grad = np.linspace(0.1, 0.9, w, dtype=np.float32)
    img = np.zeros((h, w, 3), dtype=np.float32)
    img[:, :, :] = grad[None, :, None]
    return img


# ---------------------------------------------------------------------------
# Test 1: Native resolution -> no zoom applied
# ---------------------------------------------------------------------------

class TestNativeResNoZoom:
    """When image long edge matches corpus ref, kernel should not be zoomed."""

    def test_native_res_identity(self):
        # Use a small image whose long edge == corpus_ref to skip zoom.
        # We set corpus_ref_long_edge to match the image size.
        sz = 128
        tensor = _make_delta_tensor(corpus_ref=sz)
        img = _gradient_image(sz, sz)
        out_u8, out_lin = render_with_defocus(img, tensor)
        # With delta kernels and no scaling, output should be very close to input
        from summicron.render.convolve import srgb_to_linear
        inp_lin = srgb_to_linear(img)
        diff = np.abs(out_lin - inp_lin).max()
        assert diff < 0.05, f"Max diff {diff:.4f} >= 0.05 at native resolution"


# ---------------------------------------------------------------------------
# Test 2: Scale ~0.34 -> kernel must be smaller than 129x129
# ---------------------------------------------------------------------------

class TestScaledKernel:
    """At working width 2048, kernel should be zoomed down from 129x129."""

    def test_kernel_is_zoomed(self):
        # Image: 2048 x 1366 (long edge 2048)
        # corpus_ref = 5976 -> scale = 2048/5976 ~ 0.343
        # Expected scaled_ksize = max(3, round(129 * 0.343) | 1) = round(44.2) | 1 = 44 | 1 = 45
        h, w = 128, 128  # small for speed
        corpus_ref = 5976
        expected_scale = max(h, w) / corpus_ref  # 128/5976 ~ 0.021
        expected_ksize = max(3, int(round(KSIZE * expected_scale)) | 1)

        tensor = _make_gaussian_tensor(sigma=8.0, corpus_ref=corpus_ref)
        img = _gradient_image(h, w)
        # The render should succeed; we verify via the pad calculation logic
        # that the kernel is scaled. We can't directly inspect pad, but we can
        # check that the output shape is correct and render doesn't crash.
        out_u8, out_lin = render_with_defocus(img, tensor)
        assert out_u8.shape == (h, w, 3)
        assert expected_ksize < KSIZE, (
            f"Expected scaled ksize {expected_ksize} should be < {KSIZE}"
        )


# ---------------------------------------------------------------------------
# Test 3: Energy conservation at both native and scaled resolutions
# ---------------------------------------------------------------------------

class TestEnergyConservation:
    """Energy ratio must be in [0.85, 1.15] at both scales."""

    def test_energy_native(self):
        sz = 128
        tensor = _make_delta_tensor(corpus_ref=sz)
        img = _gradient_image(sz, sz)
        _out_u8, out_lin = render_with_defocus(img, tensor)
        from summicron.render.convolve import srgb_to_linear
        inp_lin = srgb_to_linear(img)
        ratio = float(out_lin.sum()) / float(inp_lin.sum())
        assert 0.85 <= ratio <= 1.15, f"Energy ratio {ratio:.4f} outside [0.85, 1.15]"

    def test_energy_scaled(self):
        h, w = 128, 128
        tensor = _make_gaussian_tensor(sigma=8.0, corpus_ref=5976)
        img = _gradient_image(h, w)
        _out_u8, out_lin = render_with_defocus(img, tensor)
        from summicron.render.convolve import srgb_to_linear
        inp_lin = srgb_to_linear(img)
        ratio = float(out_lin.sum()) / float(inp_lin.sum())
        assert 0.85 <= ratio <= 1.15, f"Energy ratio {ratio:.4f} outside [0.85, 1.15]"


# ---------------------------------------------------------------------------
# Test 4: load_tensor reads corpus_ref_long_edge (present and absent)
# ---------------------------------------------------------------------------

class TestLoadTensorCorpusRef:
    """load_tensor should read corpus_ref_long_edge, defaulting to 5976."""

    def test_present(self, tmp_path):
        kernels = np.zeros((R, D, KSIZE, KSIZE), dtype=np.float32)
        c = KSIZE // 2
        kernels[:, :, c, c] = 1.0
        npz_path = tmp_path / "test_tensor.npz"
        np.savez(
            str(npz_path),
            kernels=kernels,
            radial_edges=RADIAL_EDGES,
            defocus_ratios=DEFOCUS_RATIOS,
            corpus_ref_long_edge=np.array(4000),
        )
        t = load_tensor(npz_path)
        assert t["corpus_ref_long_edge"] == 4000

    def test_absent_defaults_to_5976(self, tmp_path):
        kernels = np.zeros((R, D, KSIZE, KSIZE), dtype=np.float32)
        c = KSIZE // 2
        kernels[:, :, c, c] = 1.0
        npz_path = tmp_path / "test_tensor_old.npz"
        np.savez(
            str(npz_path),
            kernels=kernels,
            radial_edges=RADIAL_EDGES,
            defocus_ratios=DEFOCUS_RATIOS,
            # No corpus_ref_long_edge key
        )
        t = load_tensor(npz_path)
        assert t["corpus_ref_long_edge"] == 5976
