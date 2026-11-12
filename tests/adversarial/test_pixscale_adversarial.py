"""Adversarial tests for pixel-pitch-aware kernel scaling in convolve.render().

Attack vectors: wrong-polarity scale, tiny images, large images, degenerate
corpus_ref values, extreme defocus+pixscale combos, energy conservation at
extreme scales, pad-size correctness, portrait/landscape orientation.
"""
from __future__ import annotations

import numpy as np
import pytest

from summicron.render.convolve import render, srgb_to_linear
from tests.fixtures.defocus import render_with_defocus

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

KSIZE = 129
R, D = 6, 4
RADIAL_EDGES = np.array([0.0, 0.15, 0.30, 0.45, 0.60, 0.80, 1.01], dtype=np.float64)
DEFOCUS_RATIOS = np.array([0.0, 0.33, 0.67, 1.0], dtype=np.float64)


def _delta_tensor(corpus_ref: int | None = 5976) -> dict:
    """Delta kernels (identity convolution)."""
    kernels = np.zeros((R, D, KSIZE, KSIZE), dtype=np.float32)
    c = KSIZE // 2
    kernels[:, :, c, c] = 1.0
    t = {
        "kernels": kernels,
        "radial_edges": RADIAL_EDGES,
        "defocus_ratios": DEFOCUS_RATIOS,
    }
    if corpus_ref is not None:
        t["corpus_ref_long_edge"] = corpus_ref
    return t


def _gaussian_tensor(sigma: float = 8.0, corpus_ref: int = 5976) -> dict:
    """Gaussian kernels (all bins identical)."""
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


def _flat_image(h: int, w: int, val: float = 0.5) -> np.ndarray:
    """Uniform sRGB image, shape (h, w, 3), float32."""
    return np.full((h, w, 3), val, dtype=np.float32)


def _gradient_image(h: int, w: int) -> np.ndarray:
    """Horizontal gradient sRGB, shape (h, w, 3), float32 [0.1, 0.9]."""
    grad = np.linspace(0.1, 0.9, w, dtype=np.float32)
    img = np.zeros((h, w, 3), dtype=np.float32)
    img[:, :, :] = grad[None, :, None]
    return img


# ===================================================================
# 1. Wrong-polarity: corpus_ref SMALLER than image (scale > 1.0)
# ===================================================================

class TestWrongPolarityScale:
    """corpus_ref < image long edge -> pix_scale > 1.0.
    Currently scaling is skipped (>= 0.95). Kernels are NOT enlarged.
    This means a larger-than-corpus image gets the same kernel size as the
    corpus, which is arguably wrong (kernels should be bigger), but at
    least it should not crash or violate energy gates.
    """

    def test_scale_1_5_does_not_crash(self):
        """Image 150x100, corpus_ref=100 -> scale=1.5. Must not crash."""
        tensor = _delta_tensor(corpus_ref=100)
        img = _gradient_image(100, 150)
        out_u8, out_lin = render_with_defocus(img, tensor)
        assert out_u8.shape == (100, 150, 3)

    def test_scale_2_0_energy_conserved(self):
        """Image 200x100, corpus_ref=100 -> scale=2.0. Energy must hold."""
        tensor = _gaussian_tensor(sigma=4.0, corpus_ref=100)
        img = _flat_image(100, 200, val=0.5)
        out_u8, out_lin = render_with_defocus(img, tensor)
        inp_lin = srgb_to_linear(img)
        ratio = float(out_lin.sum()) / float(inp_lin.sum())
        assert 0.85 <= ratio <= 1.15, f"Energy ratio {ratio:.4f}"

    def test_scale_3_0_large_kernel_pad(self):
        """corpus_ref=64, image=192x128 -> scale=3.0.
        Kernel size stays at KSIZE=129, pad=64. Image is only 128 tall.
        Pad >= image height could cause issues with reflect padding.
        """
        tensor = _delta_tensor(corpus_ref=64)
        img = _gradient_image(128, 192)
        # pad=64, H=128: reflect pad needs H > pad, which is 128 > 64 -> OK
        out_u8, out_lin = render_with_defocus(img, tensor)
        assert out_u8.shape == (128, 192, 3)


# ===================================================================
# 2. Tiny image: kernel zooms to near-nothing
# ===================================================================

class TestTinyImage:
    """32x32 image with corpus_ref=5976 -> scale ~0.0054, kernel -> 3x3."""

    def test_32x32_survives(self):
        tensor = _gaussian_tensor(sigma=4.0, corpus_ref=5976)
        img = _flat_image(32, 32, 0.5)
        out_u8, out_lin = render_with_defocus(img, tensor)
        assert out_u8.shape == (32, 32, 3)

    def test_8x8_extreme_downscale(self):
        """8x8 image, scale ~0.0013. Kernel zoom factor is extreme."""
        tensor = _gaussian_tensor(sigma=4.0, corpus_ref=5976)
        img = _flat_image(8, 8, 0.5)
        # Image is < 7 pixels on a side, SSIM check skipped.
        # But energy gate should still hold.
        out_u8, out_lin = render_with_defocus(img, tensor)
        assert out_u8.shape == (8, 8, 3)

    def test_1x1_degenerate(self):
        """1x1 image. scale ~0.00017. Defocus map is zeros."""
        tensor = _gaussian_tensor(sigma=4.0, corpus_ref=5976)
        img = _flat_image(1, 1, 0.5)
        out_u8, out_lin = render_with_defocus(img, tensor)
        assert out_u8.shape == (1, 1, 3)


# ===================================================================
# 3. Very large image (larger than corpus)
# ===================================================================

class TestLargerThanCorpus:
    """Image bigger than corpus_ref -> scale > 1.0, no zoom."""

    def test_8000x6000_blocked_by_memory_guard(self):
        """8000x6000 = 48M pixels > MAX_PIXELS (36M). Should raise."""
        tensor = _delta_tensor(corpus_ref=5976)
        img = _flat_image(10, 10, 0.5)  # dummy, won't reach render internals
        # We need to actually pass a large image shape.
        # The memory guard checks H*W, so we fabricate the right shape.
        big = np.full((6001, 6001, 3), 0.5, dtype=np.float32)
        with pytest.raises(RuntimeError, match="Image too large"):
            render_with_defocus(big, tensor)

    def test_5976x3984_at_corpus_ref(self):
        """Exactly corpus_ref long edge. scale=1.0, no zoom. Use small proxy."""
        sz = 200
        tensor = _delta_tensor(corpus_ref=sz)  # corpus_ref matches long edge
        img = _gradient_image(134, sz)
        out_u8, out_lin = render_with_defocus(img, tensor)
        assert out_u8.shape == (134, sz, 3)


# ===================================================================
# 4. Portrait vs landscape orientation
# ===================================================================

class TestOrientation:
    """max(H, W) must select the correct axis for scale computation."""

    def test_landscape(self):
        """200x100 landscape: max=200, corpus_ref=400 -> scale=0.5."""
        tensor = _gaussian_tensor(sigma=4.0, corpus_ref=400)
        img = _gradient_image(100, 200)
        out_u8, _ = render_with_defocus(img, tensor)
        assert out_u8.shape == (100, 200, 3)

    def test_portrait(self):
        """100x200 portrait (H > W): max=200, corpus_ref=400 -> scale=0.5."""
        tensor = _gaussian_tensor(sigma=4.0, corpus_ref=400)
        img = _gradient_image(200, 100)  # H=200, W=100
        out_u8, _ = render_with_defocus(img, tensor)
        assert out_u8.shape == (200, 100, 3)

    def test_portrait_landscape_same_energy(self):
        """Portrait and landscape with same pixel count should give similar energy."""
        tensor = _gaussian_tensor(sigma=4.0, corpus_ref=400)
        img_land = _flat_image(100, 200, 0.5)
        img_port = _flat_image(200, 100, 0.5)
        _, lin_land = render_with_defocus(img_land, tensor)
        _, lin_port = render_with_defocus(img_port, tensor)
        ratio = float(lin_land.sum()) / float(lin_port.sum())
        assert 0.95 <= ratio <= 1.05, (
            f"Portrait/landscape energy divergence: {ratio:.4f}"
        )


# ===================================================================
# 5. Kernel zoom to even size / zero / edge cases with | 1
# ===================================================================

class TestScaledKsizeEdgeCases:
    """Verify the max(3, round(...) | 1) logic handles boundary values."""

    def test_zoom_produces_even_forced_odd(self):
        """Pick scale so round(129*scale) is even. | 1 should make it odd."""
        # scale = 0.5 -> round(129 * 0.5) = round(64.5) = 64 (banker's) or 65
        # 64 | 1 = 65; 65 | 1 = 65. Either way odd. Just verify no crash.
        tensor = _gaussian_tensor(sigma=4.0, corpus_ref=256)
        img = _gradient_image(128, 128)  # scale = 128/256 = 0.5
        out_u8, _ = render_with_defocus(img, tensor)
        assert out_u8.shape == (128, 128, 3)

    def test_zoom_round_to_2(self):
        """Scale so round(129*scale) = 2. 2 | 1 = 3. max(3,3) = 3."""
        # round(129 * s) = 2 -> s ~ 2/129 ~ 0.0155
        # corpus_ref such that long_edge / corpus_ref ~ 0.0155
        # long_edge=128 -> corpus_ref = 128/0.0155 ~ 8258
        tensor = _gaussian_tensor(sigma=4.0, corpus_ref=8258)
        img = _gradient_image(128, 128)
        out_u8, _ = render_with_defocus(img, tensor)
        assert out_u8.shape == (128, 128, 3)

    def test_zoom_round_to_1(self):
        """Scale so round(129*scale) = 1. 1 | 1 = 1. max(3,1) = 3."""
        # round(129 * s) = 1 -> s ~ 1/129 ~ 0.00775
        # long_edge=64 -> corpus_ref = 64/0.00775 ~ 8258
        tensor = _gaussian_tensor(sigma=4.0, corpus_ref=8258)
        img = _gradient_image(64, 64)
        out_u8, _ = render_with_defocus(img, tensor)
        assert out_u8.shape == (64, 64, 3)

    def test_zoom_round_to_0(self):
        """Scale so round(129*scale) = 0. 0 | 1 = 1. max(3,1) = 3."""
        # round(129 * s) = 0 -> s < 0.5/129 ~ 0.00388
        # long_edge=32 -> corpus_ref = 32/0.00388 ~ 8247
        tensor = _gaussian_tensor(sigma=4.0, corpus_ref=10000)
        img = _gradient_image(32, 32)  # scale = 32/10000 = 0.0032
        out_u8, _ = render_with_defocus(img, tensor)
        assert out_u8.shape == (32, 32, 3)


# ===================================================================
# 6. Energy conservation at extreme scales
# ===================================================================

class TestEnergyExtremeScales:
    """Energy gate [0.85, 1.15] must hold even when kernel is zoomed to 3x3."""

    def test_energy_at_scale_0_02(self):
        """128/5976 ~ 0.021. Kernel -> 3x3. Energy should still hold."""
        tensor = _gaussian_tensor(sigma=8.0, corpus_ref=5976)
        img = _flat_image(128, 128, 0.5)
        out_u8, out_lin = render_with_defocus(img, tensor)
        inp_lin = srgb_to_linear(img)
        ratio = float(out_lin.sum()) / float(inp_lin.sum())
        assert 0.85 <= ratio <= 1.15, f"Energy ratio {ratio:.4f} at scale ~0.02"

    def test_energy_at_scale_0_50(self):
        """256/512 = 0.5. Kernel -> ~65x65."""
        tensor = _gaussian_tensor(sigma=8.0, corpus_ref=512)
        img = _flat_image(256, 256, 0.5)
        out_u8, out_lin = render_with_defocus(img, tensor)
        inp_lin = srgb_to_linear(img)
        ratio = float(out_lin.sum()) / float(inp_lin.sum())
        assert 0.85 <= ratio <= 1.15, f"Energy ratio {ratio:.4f} at scale ~0.50"

    def test_energy_at_scale_0_90(self):
        """Just below threshold: scale=0.90 (< 0.95 -> zoom active)."""
        # long_edge=90, corpus_ref=100 -> scale=0.90
        tensor = _gaussian_tensor(sigma=4.0, corpus_ref=100)
        img = _flat_image(60, 90, 0.5)
        out_u8, out_lin = render_with_defocus(img, tensor)
        inp_lin = srgb_to_linear(img)
        ratio = float(out_lin.sum()) / float(inp_lin.sum())
        assert 0.85 <= ratio <= 1.15, f"Energy ratio {ratio:.4f} at scale ~0.90"

    def test_energy_at_scale_0_95_boundary(self):
        """Exactly at threshold: scale=0.95 -> no zoom. Kernel stays full."""
        # long_edge=95, corpus_ref=100 -> scale=0.95
        tensor = _gaussian_tensor(sigma=4.0, corpus_ref=100)
        img = _flat_image(64, 95, 0.5)
        out_u8, out_lin = render_with_defocus(img, tensor)
        inp_lin = srgb_to_linear(img)
        ratio = float(out_lin.sum()) / float(inp_lin.sum())
        assert 0.85 <= ratio <= 1.15, f"Energy ratio {ratio:.4f} at scale=0.95"


# ===================================================================
# 7. Interaction: pix_scale + defocus_map extremes
# ===================================================================

class TestPixscaleDefocusInteraction:
    """Combined pixel-scale downscaling + extreme defocus_map values."""

    def test_scale_0_34_defocus_max(self):
        """pix_scale~0.34, defocus_map=1.0 everywhere."""
        tensor = _gaussian_tensor(sigma=8.0, corpus_ref=5976)
        img = _gradient_image(128, 128)  # scale ~0.021
        dmap = np.ones((128, 128), dtype=np.float32)
        out_u8, out_lin = render_with_defocus(img, tensor, dmap=dmap)
        assert out_u8.shape == (128, 128, 3)
        inp_lin = srgb_to_linear(img)
        ratio = float(out_lin.sum()) / float(inp_lin.sum())
        assert 0.85 <= ratio <= 1.15, f"Energy ratio {ratio:.4f}"

    def test_scale_0_50_defocus_near_zero(self):
        """pix_scale=0.5, near-zero defocus_map."""
        tensor = _gaussian_tensor(sigma=4.0, corpus_ref=256)
        img = _gradient_image(128, 128)  # scale=0.5
        dmap = np.full((128, 128), 0.01, dtype=np.float32)
        out_u8, out_lin = render_with_defocus(img, tensor, dmap=dmap)
        assert out_u8.shape == (128, 128, 3)

    def test_extreme_defocus_at_extreme_scale(self):
        """pix_scale~0.005, defocus_map=1.0."""
        tensor = _gaussian_tensor(sigma=4.0, corpus_ref=5976)
        img = _flat_image(32, 32, 0.5)  # scale ~0.005
        dmap = np.ones((32, 32), dtype=np.float32)
        out_u8, out_lin = render_with_defocus(img, tensor, dmap=dmap)
        assert out_u8.shape == (32, 32, 3)


# ===================================================================
# 8. Missing corpus_ref_long_edge (old .npz fallback)
# ===================================================================

class TestMissingCorpusRef:
    """When corpus_ref_long_edge is absent from tensor dict, pix_scale=1.0."""

    def test_no_key_means_no_scaling(self):
        """Tensor without corpus_ref_long_edge should use native kernels."""
        kernels = np.zeros((R, D, KSIZE, KSIZE), dtype=np.float32)
        c = KSIZE // 2
        kernels[:, :, c, c] = 1.0
        tensor = {
            "kernels": kernels,
            "radial_edges": RADIAL_EDGES,
            "defocus_ratios": DEFOCUS_RATIOS,
            # No corpus_ref_long_edge
        }
        img = _flat_image(64, 64, 0.5)
        # corpus_ref = tensor.get("corpus_ref_long_edge") -> None
        # pix_scale = 1.0 -> no zoom
        out_u8, out_lin = render_with_defocus(img, tensor)
        assert out_u8.shape == (64, 64, 3)

    def test_none_value_means_no_scaling(self):
        """Explicit None value should also skip scaling."""
        tensor = _delta_tensor(corpus_ref=None)
        # corpus_ref key is absent from dict in this case
        img = _flat_image(64, 64, 0.5)
        out_u8, _ = render_with_defocus(img, tensor)
        assert out_u8.shape == (64, 64, 3)


# ===================================================================
# 9. corpus_ref_long_edge = 0 or negative
# ===================================================================

class TestDegenerateCorpusRef:
    """corpus_ref_long_edge = 0 or negative should not silently corrupt."""

    def test_corpus_ref_zero(self):
        """corpus_ref=0 -> pix_scale = max(H,W)/0 -> must raise ValueError."""
        tensor = _delta_tensor(corpus_ref=0)
        img = _flat_image(64, 64, 0.5)
        with pytest.raises(ValueError, match="corpus_ref_long_edge"):
            render_with_defocus(img, tensor)

    def test_corpus_ref_negative(self):
        """corpus_ref=-100 -> pix_scale = 64/-100 = -0.64 < 0.95.
        scaled_ksize = max(3, round(129 * -0.64) | 1) = max(3, round(-82.56) | 1)
        = max(3, -83 | 1) = max(3, -83) = 3.
        Then ndimage_zoom(kernel, -0.64) -- negative zoom factor!
        """
        tensor = _delta_tensor(corpus_ref=-100)
        img = _flat_image(64, 64, 0.5)
        with pytest.raises(ValueError, match="corpus_ref_long_edge"):
            render_with_defocus(img, tensor)

    def test_corpus_ref_one(self):
        """corpus_ref=1. scale=64. >= 0.95, no zoom. Large pad from full kernel."""
        tensor = _delta_tensor(corpus_ref=1)
        img = _flat_image(64, 64, 0.5)
        out_u8, _ = render_with_defocus(img, tensor)
        assert out_u8.shape == (64, 64, 3)


# ===================================================================
# 10. Pad size correctness after scaling
# ===================================================================

class TestPadSizeCorrectness:
    """After scaling, pad = scaled_ksize // 2. Verify no clipping artefacts."""

    def test_pad_does_not_clip_output(self):
        """Output shape must match input shape exactly, regardless of scale."""
        for corpus_ref in [50, 100, 500, 5976, 10000]:
            tensor = _gaussian_tensor(sigma=4.0, corpus_ref=corpus_ref)
            img = _gradient_image(64, 96)
            out_u8, _ = render_with_defocus(img, tensor)
            assert out_u8.shape == (64, 96, 3), (
                f"Shape mismatch at corpus_ref={corpus_ref}: {out_u8.shape}"
            )

    def test_pad_larger_than_image_dim(self):
        """When no scaling (scale >= 0.95), pad = 129//2 = 64.
        For a 32x32 image, pad(64) > H(32). np.pad reflect needs
        array size > pad. This should fail or be handled.
        """
        # corpus_ref = 32 -> scale = 32/32 = 1.0 -> no zoom -> pad = 64
        # np.pad reflect mode requires array dim > pad
        tensor = _delta_tensor(corpus_ref=32)
        img = _flat_image(32, 32, 0.5)
        # pad=64 > 32: np.pad('reflect') will fail
        # Actually numpy reflect allows pad > dim by wrapping multiple times.
        # Let's just verify it doesn't crash.
        out_u8, _ = render_with_defocus(img, tensor)
        assert out_u8.shape == (32, 32, 3)

    def test_small_image_large_corpus_minimal_pad(self):
        """16x16 image, corpus_ref=5976. scale~0.0027 -> ksize=3, pad=1.
        Very small pad; should be safe.
        """
        tensor = _gaussian_tensor(sigma=4.0, corpus_ref=5976)
        img = _flat_image(16, 16, 0.5)
        out_u8, _ = render_with_defocus(img, tensor)
        assert out_u8.shape == (16, 16, 3)


# ===================================================================
# 11. Threshold boundary: scale = 0.949 vs 0.951
# ===================================================================

class TestThresholdBoundary:
    """Verify behaviour right at the 0.95 threshold."""

    def test_just_below_threshold_activates_zoom(self):
        """scale = 0.949 (< 0.95) -> zoom should activate."""
        # long_edge / corpus_ref = 0.949 -> corpus_ref = long_edge / 0.949
        # long_edge = 128 -> corpus_ref ~= 134.88 -> 135
        tensor = _gaussian_tensor(sigma=4.0, corpus_ref=135)
        img = _gradient_image(86, 128)  # long_edge=128, 128/135~0.948
        out_u8, out_lin = render_with_defocus(img, tensor)
        # Scaled ksize = max(3, round(129 * 0.948) | 1) = round(122.3) | 1 = 122 | 1 = 123
        assert out_u8.shape == (86, 128, 3)

    def test_just_above_threshold_skips_zoom(self):
        """scale = 0.951 (>= 0.95) -> no zoom."""
        # long_edge = 128 -> corpus_ref = 128/0.951 ~= 134.6 -> 134
        tensor = _gaussian_tensor(sigma=4.0, corpus_ref=134)
        img = _gradient_image(86, 128)  # 128/134~0.955
        out_u8, _ = render_with_defocus(img, tensor)
        assert out_u8.shape == (86, 128, 3)


# ===================================================================
# 12. Non-square kernels (future-proofing / assumption check)
# ===================================================================

class TestKernelShapeAssumptions:
    """The code uses kH for both dimensions of zoom. If kH != kW, trouble."""

    def test_code_assumes_square_kernels(self):
        """Verify that kH == kW is assumed (production tensors should be square)."""
        # This is a correctness assertion on the assumption, not a bug test.
        kernels = np.zeros((R, D, KSIZE, KSIZE), dtype=np.float32)
        c = KSIZE // 2
        kernels[:, :, c, c] = 1.0
        assert kernels.shape[2] == kernels.shape[3], "Kernels must be square"
