"""Adversarial tests for the bin-and-convolve renderer."""
from __future__ import annotations

import numpy as np
import pytest

from summicron.render.convolve import (
    render,
    srgb_to_linear,
    _memory_guard,
    _bilinear_weights,
)
from tests.fixtures.defocus import radial_dmap, render_with_defocus



# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_delta_tensor(ksize: int = 129, R: int = 6, D: int = 4) -> dict:
    """Delta-impulse tensor: every kernel is a centered delta."""
    kernels = np.zeros((R, D, ksize, ksize), dtype=np.float32)
    c = ksize // 2
    kernels[:, :, c, c] = 1.0
    radial_edges = np.linspace(0.0, 1.01, R + 1, dtype=np.float64)
    defocus_ratios = np.linspace(0.0, 1.0, D, dtype=np.float64)
    return {
        "kernels": kernels,
        "radial_edges": radial_edges,
        "defocus_ratios": defocus_ratios,
    }


def _make_unnormalized_tensor(ksize: int = 129, scale: float = 2.0) -> dict:
    """Tensor with kernels that sum to `scale` instead of 1.0."""
    R, D = 6, 4
    kernels = np.zeros((R, D, ksize, ksize), dtype=np.float32)
    c = ksize // 2
    kernels[:, :, c, c] = scale
    radial_edges = np.linspace(0.0, 1.01, R + 1, dtype=np.float64)
    defocus_ratios = np.linspace(0.0, 1.0, D, dtype=np.float64)
    return {
        "kernels": kernels,
        "radial_edges": radial_edges,
        "defocus_ratios": defocus_ratios,
    }


# ---------------------------------------------------------------------------
# MANDATORY: Wrong-polarity fixtures
# ---------------------------------------------------------------------------

class TestAllBlackImage:
    """All-black (zero signal) must produce all-black, no NaN/garbage.

    BUG: energy_ratio = 0/epsilon = 0.0 fails the [0.85, 1.15] gate.
    The energy gate must special-case zero-energy inputs (0/0 -> ratio 1.0).
    """

    def test_all_black_output_is_black(self):
        sz = 64
        img = np.zeros((sz, sz, 3), dtype=np.float32)
        tensor = _make_delta_tensor()
        output_u8, output_lin = render_with_defocus(img, tensor)
        assert output_u8.max() == 0, f"Expected all-black output, got max={output_u8.max()}"
        assert not np.any(np.isnan(output_lin)), "NaN in output from all-black input"

    def test_all_black_no_inf(self):
        sz = 64
        img = np.zeros((sz, sz, 3), dtype=np.float32)
        tensor = _make_delta_tensor()
        _output_u8, output_lin = render_with_defocus(img, tensor)
        assert not np.any(np.isinf(output_lin)), "Inf in output from all-black input"


class TestAllWhiteImage:
    """All-white (saturated) must conserve energy."""

    def test_all_white_energy_conservation(self):
        sz = 64
        img = np.ones((sz, sz, 3), dtype=np.float32)
        tensor = _make_delta_tensor()
        _output_u8, output_lin = render_with_defocus(img, tensor)
        input_lin = srgb_to_linear(img)
        ratio = float(output_lin.sum()) / float(input_lin.sum())
        assert 0.85 <= ratio <= 1.15, (
            f"Energy ratio {ratio:.4f} outside [0.85, 1.15] for all-white input"
        )

    def test_all_white_no_nan(self):
        sz = 64
        img = np.ones((sz, sz, 3), dtype=np.float32)
        tensor = _make_delta_tensor()
        _output_u8, output_lin = render_with_defocus(img, tensor)
        assert not np.any(np.isnan(output_lin)), "NaN in output from all-white input"


# ---------------------------------------------------------------------------
# Attack vector 1: Single-pixel image
# ---------------------------------------------------------------------------

class TestSinglePixelImage:
    """1x1 image must not crash the pipeline.

    BUG: radial_dmap(1,1) divides by (0^2 + 0^2) = 0 -> NaN.
    The defocus map must handle degenerate sizes (cy=0, cx=0).
    """

    def test_single_pixel_no_crash(self):
        img = np.array([[[0.5, 0.5, 0.5]]], dtype=np.float32)
        tensor = _make_delta_tensor()
        output_u8, output_lin = render_with_defocus(img, tensor)
        assert output_u8.shape == (1, 1, 3)
        assert not np.any(np.isnan(output_lin))


# ---------------------------------------------------------------------------
# Attack vector 2: Non-square (wide panorama)
# ---------------------------------------------------------------------------

class TestNonSquareImage:
    """Wide panorama: defocus map and convolution must handle non-square."""

    def test_panorama_defocus_map(self):
        dmap = radial_dmap(64, 512)
        # Center should be ~0
        assert dmap[32, 256] == pytest.approx(0.0, abs=0.02)
        # Corners should be ~1
        assert dmap[0, 0] == pytest.approx(1.0, abs=0.02)
        assert dmap[63, 511] == pytest.approx(1.0, abs=0.02)

    def test_panorama_render(self):
        img = np.full((32, 256, 3), 0.5, dtype=np.float32)
        tensor = _make_delta_tensor()
        output_u8, output_lin = render_with_defocus(img, tensor)
        assert output_u8.shape == (32, 256, 3)
        assert not np.any(np.isnan(output_lin))


# ---------------------------------------------------------------------------
# Attack vector 3: Kernel sum != 1.0 (un-normalized)
# ---------------------------------------------------------------------------

class TestUnnormalizedKernel:
    """Kernels summing to 2.0 should violate energy conservation gate."""

    def test_double_energy_kernel_fails_gate(self):
        sz = 64
        img = np.full((sz, sz, 3), 0.5, dtype=np.float32)
        tensor = _make_unnormalized_tensor(scale=2.0)
        # P1.audit Item 5: energy gate assert→RuntimeError.
        with pytest.raises((AssertionError, RuntimeError), match="energy ratio"):
            render_with_defocus(img, tensor)


# ---------------------------------------------------------------------------
# Attack vector 4: HEIC without pillow-heif
# ---------------------------------------------------------------------------

class TestHeicWithoutPillowHeif:
    """Loading a .heic without pillow-heif should give a clear error."""

    def test_heic_missing_lib(self, tmp_path):
        from summicron.render.convolve import load_image
        fake_heic = tmp_path / "fake.heic"
        fake_heic.write_bytes(b"\x00" * 128)
        with pytest.raises(Exception):
            load_image(fake_heic)


# ---------------------------------------------------------------------------
# Attack vector 5: Defocus map all zeros (d=0 everywhere => identity)
# ---------------------------------------------------------------------------

class TestDefocusAllZeros:
    """When defocus is 0 everywhere (1x1 image center), output ~ input."""

    def test_identity_at_zero_defocus(self):
        # Use a small even image; center defocus will be ~0
        sz = 64
        img = np.full((sz, sz, 3), 0.5, dtype=np.float32)
        tensor = _make_delta_tensor()
        _output_u8, output_lin = render_with_defocus(img, tensor)
        input_lin = srgb_to_linear(img)
        # At center, output should closely match input
        ch = sz // 4
        c = sz // 2
        center_in = input_lin[c - ch:c + ch, c - ch:c + ch]
        center_out = output_lin[c - ch:c + ch, c - ch:c + ch]
        diff = np.abs(center_out - center_in).max()
        assert diff < 0.05, f"Center region max diff {diff:.4f} >= 0.05 with delta kernel"


# ---------------------------------------------------------------------------
# Attack vector 6: Defocus map all ones (maximum blur)
# ---------------------------------------------------------------------------

class TestDefocusAllOnes:
    """Maximum defocus (corners of large image) should not crash."""

    def test_max_blur_no_crash(self):
        # Even with max defocus, pipeline must not crash
        sz = 64
        img = np.full((sz, sz, 3), 0.4, dtype=np.float32)
        tensor = _make_delta_tensor()
        output_u8, output_lin = render_with_defocus(img, tensor)
        assert output_u8.shape == (sz, sz, 3)
        assert not np.any(np.isnan(output_lin))


# ---------------------------------------------------------------------------
# Attack vector 7: Integer overflow / uint8 input handling
# ---------------------------------------------------------------------------

class TestUint8Input:
    """Renderer expects float32 [0,1]. What if someone passes uint8?

    BUG: uint8 values (0-255) are silently coerced to float32 by srgb_to_linear
    but NOT divided by 255. Values like 200.0 pass through sRGB linearization
    and convolution without any range check. The energy gate doesn't catch it
    because ratio of large numbers can still be ~1.0.
    render() should validate input range [0, 1] or reject non-float input.
    """

    def test_uint8_input_coerced_or_errors(self):
        sz = 64
        img_u8 = np.full((sz, sz, 3), 200, dtype=np.uint8)
        tensor = _make_delta_tensor()
        # uint8 input should be auto-coerced to float32 / 255
        output_u8, output_lin = render_with_defocus(img_u8, tensor)
        assert output_u8.shape == (sz, sz, 3)
        assert not np.any(np.isnan(output_lin))


# ---------------------------------------------------------------------------
# Attack vector 8: Energy after mirror-pad + crop
# ---------------------------------------------------------------------------

class TestMirrorPadEnergy:
    """Padding must not inject extra energy into the output."""

    def test_uniform_image_pad_energy(self):
        sz = 128
        val = 0.3
        img = np.full((sz, sz, 3), val, dtype=np.float32)
        tensor = _make_delta_tensor()
        _output_u8, output_lin = render_with_defocus(img, tensor)
        input_lin = srgb_to_linear(img)
        ratio = float(output_lin.sum()) / float(input_lin.sum())
        assert 0.95 <= ratio <= 1.05, (
            f"Mirror-pad energy ratio {ratio:.4f} outside tight [0.95, 1.05] "
            f"for uniform image with delta kernel"
        )


# ---------------------------------------------------------------------------
# Edge case: 2x2 image (minimal non-trivial)
# ---------------------------------------------------------------------------

class TestTinyImage:
    """2x2 image -- smallest non-trivial size.

    BUG: skimage.structural_similarity requires images >= 7x7 by default.
    The center SSIM check in render() crashes on images smaller than 7x7.
    Should guard with min(ch, cw) >= 7 or skip SSIM for tiny images.
    """

    def test_2x2_no_crash(self):
        img = np.array([
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            [[0.7, 0.8, 0.9], [0.2, 0.3, 0.4]],
        ], dtype=np.float32)
        tensor = _make_delta_tensor()
        output_u8, output_lin = render_with_defocus(img, tensor)
        assert output_u8.shape == (2, 2, 3)
        assert not np.any(np.isnan(output_lin))


# ---------------------------------------------------------------------------
# Edge case: snap_indices boundary
# ---------------------------------------------------------------------------

class TestBilinearWeightsBoundary:
    """Bilinear weights at extreme values must produce valid indices and weights."""

    def test_snap_at_zero(self):
        edges = np.array([0.0, 0.5, 1.01], dtype=np.float64)
        ratios = np.array([0.0, 0.5, 1.0], dtype=np.float64)
        indices, weights = _bilinear_weights(0.0, 0.0, edges, ratios)
        Nr, Nd = len(edges) - 1, len(ratios)
        for ri, di in indices:
            assert 0 <= ri < Nr, f"r_idx {ri} out of range [0, {Nr})"
            assert 0 <= di < Nd, f"d_idx {di} out of range [0, {Nd})"
        assert abs(sum(weights) - 1.0) < 1e-6, f"Weights sum to {sum(weights)}"

    def test_snap_at_one(self):
        edges = np.array([0.0, 0.5, 1.01], dtype=np.float64)
        ratios = np.array([0.0, 0.5, 1.0], dtype=np.float64)
        indices, weights = _bilinear_weights(1.0, 1.0, edges, ratios)
        Nr, Nd = len(edges) - 1, len(ratios)
        for ri, di in indices:
            assert 0 <= ri < Nr, f"r_idx {ri} out of range [0, {Nr})"
            assert 0 <= di < Nd, f"d_idx {di} out of range [0, {Nd})"
        assert abs(sum(weights) - 1.0) < 1e-6, f"Weights sum to {sum(weights)}"

    def test_snap_beyond_one(self):
        edges = np.array([0.0, 0.5, 1.01], dtype=np.float64)
        ratios = np.array([0.0, 0.5, 1.0], dtype=np.float64)
        indices, weights = _bilinear_weights(1.5, 1.5, edges, ratios)
        Nr, Nd = len(edges) - 1, len(ratios)
        for ri, di in indices:
            assert 0 <= ri < Nr, f"r_idx {ri} out of range [0, {Nr})"
            assert 0 <= di < Nd, f"d_idx {di} out of range [0, {Nd})"


# ===================================================================
# Day 5 Adversarial: Bilinear interpolation attack vectors
# ===================================================================

from summicron.render.convolve import _bilinear_weights

_RADIAL_EDGES = np.array([0.0, 0.15, 0.30, 0.45, 0.60, 0.80, 1.01], dtype=np.float64)
_DEFOCUS_RATIOS = np.array([0.0, 0.33, 0.67, 1.0], dtype=np.float64)


def _make_smooth_blur_tensor(ksize: int = 129) -> dict:
    """Tensor with Gaussian kernels that grow with defocus; d=0 is delta."""
    R, D = 6, 4
    kernels = np.zeros((R, D, ksize, ksize), dtype=np.float32)
    c = ksize // 2
    for ri in range(R):
        kernels[ri, 0, c, c] = 1.0
        for di in range(1, D):
            sigma = 1.0 + di * 1.5
            y, x = np.mgrid[-c:c + 1, -c:c + 1].astype(np.float32)
            g = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
            g /= g.sum()
            kernels[ri, di] = g.astype(np.float32)
    return {
        "kernels": kernels,
        "radial_edges": _RADIAL_EDGES.copy(),
        "defocus_ratios": _DEFOCUS_RATIOS.copy(),
    }


# ---------------------------------------------------------------------------
# AV1: Bilinear weight edge cases
# ---------------------------------------------------------------------------

class TestBilinearWeightEdgeCases:
    """Attack bilinear weight computation at exact grid points and boundaries."""

    def test_weights_at_exact_bin_center(self):
        """r_frac exactly at a bin center should give weight=1 on one r-axis side."""
        r_centers = 0.5 * (_RADIAL_EDGES[:-1] + _RADIAL_EDGES[1:])
        # Pick 3rd bin center, defocus at 0.67 (exact ratio)
        r = float(r_centers[2])
        d = 0.67
        indices, weights = _bilinear_weights(r, d, _RADIAL_EDGES, _DEFOCUS_RATIOS)
        wsum = sum(weights)
        assert abs(wsum - 1.0) < 1e-6, f"Weights sum to {wsum} at exact grid point"

    def test_weights_r_frac_zero(self):
        """r_frac=0.0 must not produce negative indices."""
        indices, weights = _bilinear_weights(0.0, 0.5, _RADIAL_EDGES, _DEFOCUS_RATIOS)
        for ri, di in indices:
            assert ri >= 0, f"Negative radial index {ri}"
            assert di >= 0, f"Negative defocus index {di}"
        assert abs(sum(weights) - 1.0) < 1e-6

    def test_weights_r_frac_one(self):
        """r_frac=1.0 must clamp to valid range."""
        indices, weights = _bilinear_weights(1.0, 0.5, _RADIAL_EDGES, _DEFOCUS_RATIOS)
        Nr = len(_RADIAL_EDGES) - 1
        for ri, di in indices:
            assert 0 <= ri < Nr, f"r_idx {ri} out of range [0, {Nr})"
        assert abs(sum(weights) - 1.0) < 1e-6

    def test_weights_d_val_zero_returns_four_corners(self):
        """Post v2 OTF-norm (spec §6), d_val=0.0 returns all 4 bilinear
        corners; the former delta-kernel guard has been removed because
        ``kernels[0, 0]`` is now an exact delta by construction.
        """
        indices, weights = _bilinear_weights(0.5, 0.0, _RADIAL_EDGES, _DEFOCUS_RATIOS)
        assert len(indices) == 4
        assert len(weights) == 4
        assert abs(sum(weights) - 1.0) < 1e-6

    def test_weights_d_val_between_zero_and_first_ratio(self):
        """d_val=0.15 (between the delta row and first non-delta ratio) is
        now a regular four-corner bilinear blend; the delta corner
        participates with its exact-delta kernel.
        """
        indices, weights = _bilinear_weights(0.5, 0.15, _RADIAL_EDGES, _DEFOCUS_RATIOS)
        assert len(indices) == 4
        assert abs(sum(weights) - 1.0) < 1e-6
        # d_idx=0 IS permitted in the blend (it is a real delta kernel).
        d_indices = [idx[1] for idx in indices]
        assert 0 in d_indices


# ---------------------------------------------------------------------------
# AV2: Kernel blending with zero-energy bin
# ---------------------------------------------------------------------------

class TestKernelBlendZeroEnergy:
    """What if one of the 4 bilinear corners has an all-zeros kernel?"""

    def test_zero_kernel_corner_no_nan(self):
        """One radial bin has all-zero kernels. Blend must not produce NaN."""
        tensor = _make_smooth_blur_tensor()
        # Zero out bin r=2, all defocus
        tensor["kernels"][2, :, :, :] = 0.0
        sz = 64
        img = np.full((sz, sz, 3), 0.5, dtype=np.float32)
        # This may fail energy gate because some kernels are zero
        try:
            _out_u8, out_lin = render_with_defocus(img, tensor)
            assert not np.any(np.isnan(out_lin)), "NaN from zero-kernel bin blend"
        except (AssertionError, RuntimeError) as e:
            # P1.audit Item 5: energy/NaN gates assert→RuntimeError.
            if "energy ratio" in str(e) or "NaN" in str(e):
                pass  # Expected: zero kernels break conservation OR trip NaN gate
            else:
                raise


# ---------------------------------------------------------------------------
# AV3: MANDATORY wrong-polarity fixtures
# ---------------------------------------------------------------------------

class TestWrongPolarityBrightOnDim:
    """Bright point source on dim background (typical). Baseline sanity."""

    def test_bright_on_dim_no_artifacts(self):
        sz = 128
        img = np.full((sz, sz, 3), 0.05, dtype=np.float32)  # dim background
        img[sz // 2, sz // 2, :] = 0.95  # bright point
        tensor = _make_smooth_blur_tensor()
        _out_u8, out_lin = render_with_defocus(img, tensor)
        assert not np.any(np.isnan(out_lin))
        assert not np.any(np.isinf(out_lin))
        # Output should be non-negative everywhere
        assert out_lin.min() >= -0.01, f"Negative pixel {out_lin.min():.4f}"


class TestWrongPolarityDimOnBright:
    """MANDATORY wrong-polarity: dark subject on bright background.

    Bilinear blending of blur kernels on a mostly-bright image with a small
    dark region tests whether the convolution + normalization pipeline
    handles inverted contrast correctly (no bright halos around dark subject).
    """

    def test_dim_on_bright_no_artifacts(self):
        sz = 128
        img = np.full((sz, sz, 3), 0.9, dtype=np.float32)  # bright bg
        # Dark square in center
        img[56:72, 56:72, :] = 0.05
        tensor = _make_smooth_blur_tensor()
        _out_u8, out_lin = render_with_defocus(img, tensor)
        assert not np.any(np.isnan(out_lin))
        assert not np.any(np.isinf(out_lin))
        # Output should be non-negative everywhere
        assert out_lin.min() >= -0.01, f"Negative pixel {out_lin.min():.4f}"

    def test_dim_on_bright_energy_conservation(self):
        """Dim-on-bright must also conserve energy."""
        sz = 128
        img = np.full((sz, sz, 3), 0.9, dtype=np.float32)
        img[56:72, 56:72, :] = 0.05
        tensor = _make_smooth_blur_tensor()
        _out_u8, out_lin = render_with_defocus(img, tensor)
        input_lin = srgb_to_linear(img)
        ratio = float(out_lin.sum()) / float(input_lin.sum())
        assert 0.85 <= ratio <= 1.15, (
            f"Dim-on-bright energy ratio {ratio:.4f} outside [0.85, 1.15]"
        )

    def test_dim_on_bright_no_bright_halo(self):
        """Dark disk on bright background: pixels just outside the dark disk
        should not be brighter than the original background (halo artifact)."""
        sz = 128
        img = np.full((sz, sz, 3), 0.8, dtype=np.float32)
        # Dark disk radius 10 at center
        cy, cx = sz // 2, sz // 2
        for y in range(sz):
            for x in range(sz):
                if (y - cy) ** 2 + (x - cx) ** 2 < 100:
                    img[y, x, :] = 0.02
        tensor = _make_smooth_blur_tensor()
        _out_u8, out_lin = render_with_defocus(img, tensor)
        input_lin = srgb_to_linear(img)
        bg_linear = srgb_to_linear(np.array([0.8], dtype=np.float32))[0]
        # Check a ring just outside the dark disk
        halo_max = 0.0
        for y in range(sz):
            for x in range(sz):
                r2 = (y - cy) ** 2 + (x - cx) ** 2
                if 120 < r2 < 200:  # ring just outside disk
                    halo_max = max(halo_max, float(out_lin[y, x, 0]))
        # Allow 5% overshoot from convolution ringing
        assert halo_max < bg_linear * 1.05, (
            f"Bright halo detected: {halo_max:.4f} > {bg_linear * 1.05:.4f}"
        )


# ---------------------------------------------------------------------------
# AV4: Defocus map extremes (flat zero / flat 1.0)
# ---------------------------------------------------------------------------

class TestDefocusMapExtremes:
    """Flat-zero defocus_map forces d=0 everywhere (delta guard territory).
    Flat-one defocus_map forces d=1.0 everywhere."""

    def test_defocus_map_zero(self):
        """All pixels at d=0. Delta guard fires for every bin."""
        sz = 64
        img = np.full((sz, sz, 3), 0.5, dtype=np.float32)
        tensor = _make_delta_tensor()
        dmap = np.zeros((sz, sz), dtype=np.float32)
        _out_u8, out_lin = render_with_defocus(img, tensor, dmap=dmap)
        assert not np.any(np.isnan(out_lin))
        # With all-delta, output should closely match input
        input_lin = srgb_to_linear(img)
        diff = np.abs(out_lin - input_lin).max()
        assert diff < 0.05, f"flat-zero dmap identity broken: max diff {diff:.4f}"

    def test_defocus_map_one(self):
        """All pixels at d=1.0. No crash, energy OK."""
        sz = 64
        img = np.full((sz, sz, 3), 0.5, dtype=np.float32)
        tensor = _make_smooth_blur_tensor()
        dmap = np.ones((sz, sz), dtype=np.float32)
        _out_u8, out_lin = render_with_defocus(img, tensor, dmap=dmap)
        assert not np.any(np.isnan(out_lin))
        assert not np.any(np.isinf(out_lin))


# ---------------------------------------------------------------------------
# AV5: Single-bin tensor (R=1, D=1)
# ---------------------------------------------------------------------------

class TestSingleBinTensor:
    """Degenerate tensor with only 1 radial bin and 1 defocus level.
    _bilinear_weights must not index out-of-bounds when Nr=1, Nd=1.
    The code clamps to [0, N-2] which would be [0, -1] -- potential bug.
    """

    def test_single_bin_no_crash(self):
        ksize = 129
        R, D = 1, 1
        kernels = np.zeros((R, D, ksize, ksize), dtype=np.float32)
        c = ksize // 2
        kernels[0, 0, c, c] = 1.0
        tensor = {
            "kernels": kernels,
            "radial_edges": np.array([0.0, 1.01], dtype=np.float64),
            "defocus_ratios": np.array([0.5], dtype=np.float64),  # single non-delta
        }
        sz = 32
        img = np.full((sz, sz, 3), 0.5, dtype=np.float32)
        _out_u8, out_lin = render_with_defocus(img, tensor)
        assert not np.any(np.isnan(out_lin))

    def test_single_radial_bin_two_defocus(self):
        """R=1, D=2: radial interpolation degenerates, defocus still works."""
        ksize = 129
        R, D = 1, 2
        kernels = np.zeros((R, D, ksize, ksize), dtype=np.float32)
        c = ksize // 2
        kernels[0, 0, c, c] = 1.0  # delta
        sigma = 2.0
        y, x = np.mgrid[-c:c + 1, -c:c + 1].astype(np.float32)
        g = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
        g /= g.sum()
        kernels[0, 1] = g.astype(np.float32)
        tensor = {
            "kernels": kernels,
            "radial_edges": np.array([0.0, 1.01], dtype=np.float64),
            "defocus_ratios": np.array([0.0, 1.0], dtype=np.float64),
        }
        sz = 32
        img = np.full((sz, sz, 3), 0.5, dtype=np.float32)
        _out_u8, out_lin = render_with_defocus(img, tensor)
        assert not np.any(np.isnan(out_lin))


# ---------------------------------------------------------------------------
# AV6: Non-square tall narrow image with bilinear
# ---------------------------------------------------------------------------

class TestTallNarrowImageBilinear:
    """Tall narrow image: extreme aspect ratio stress-tests defocus map + bilinear."""

    def test_tall_narrow_render(self):
        img = np.full((256, 16, 3), 0.4, dtype=np.float32)
        tensor = _make_smooth_blur_tensor()
        _out_u8, out_lin = render_with_defocus(img, tensor)
        assert out_lin.shape == (256, 16, 3)
        assert not np.any(np.isnan(out_lin))

    def test_wide_short_render(self):
        img = np.full((16, 256, 3), 0.4, dtype=np.float32)
        tensor = _make_smooth_blur_tensor()
        _out_u8, out_lin = render_with_defocus(img, tensor)
        assert out_lin.shape == (16, 256, 3)
        assert not np.any(np.isnan(out_lin))


# ---------------------------------------------------------------------------
# P1.7 Adversarial: peak-norm "stamp" mode
# ---------------------------------------------------------------------------


class TestPeakNormStampAdversarial:
    """Adversarial coverage for ``psf_norm="peak"`` (stamp) mode.

    Targets thin spots the Coder's happy-path acceptance suite did not
    cover:
      1. Wrong-polarity fixture (dim-on-bright) per Post-mortem 1.
      2. Depth-varied (not flat-max) defocus and its bloom ratio.
      3. Near-delta vs wide-blur step-function seam.
      4. Very small image (sz < 32) partition-of-unity exposure.
      5. Single-color-channel extreme (no cross-channel contamination).
      6. Zero-input edge case (no NaN / div-by-zero on skipped renorm).
      7. Invalid psf_norm string via direct render() call.
    """

    # --- 1. wrong-polarity: dim pixel on bright background ---------------
    def test_peak_mode_dim_on_bright_no_nan_no_inf(self):
        """Dim (0.1) pixel at centre on bright (0.95) background.

        Spec §3 claims "most pixels are dim, ratio stays modest". Here the
        opposite polarity: almost every pixel is bright. In peak mode,
        ``output[i,j] ~ sum_{kernel}(img * kernel_peak_normed)``, and
        because the input is nearly uniform ~0.95 with kernel.sum() ~ 5
        for wide Gaussians, output is expected to clip hard to white.
        The invariants we assert here are structural: no NaN/Inf, no
        negatives, and the final uint8 doesn't blow past 255. Spatial
        variation from per-wedge/slab kernel-sum drift is
        spec-consistent (LDR clip absorbs it) — documented, not asserted
        as a failure.
        """
        sz = 128
        img = np.full((sz, sz, 3), 0.95, dtype=np.float32)
        img[sz // 2, sz // 2, :] = 0.1  # single dim pixel on bright bg
        tensor = _make_smooth_blur_tensor()
        flat_dmap = np.full((sz, sz), 1.0, dtype=np.float32)
        out_u8, out_lin = render_with_defocus(
            img, tensor, dmap=flat_dmap, psf_norm="peak"
        )
        assert not np.any(np.isnan(out_lin)), "NaN in wrong-polarity peak-mode output"
        assert not np.any(np.isinf(out_lin)), "Inf in wrong-polarity peak-mode output"
        assert out_lin.min() >= -1e-5, (
            f"Negative pixel {out_lin.min():.6f} in peak-mode dim-on-bright"
        )
        # uint8 must not overflow beyond clip
        assert out_u8.max() <= 255 and out_u8.min() >= 0

    def test_peak_mode_dim_on_bright_saturates_to_white(self):
        """Bright uniform bg × peak-normed kernel.sum() >> 1 -> clip to white.

        Spec-consistent behaviour: post-conv renorm is gone, so almost
        every pixel receives ~(bg_linear * kernel.sum()) which exceeds 1.
        The linear-domain output therefore exceeds 1 in most places, and
        after clip+sRGB the uint8 should be at/near 255 for most pixels.
        This *documents* the LDR-clip absorption claim in the spec.
        """
        sz = 128
        img = np.full((sz, sz, 3), 0.95, dtype=np.float32)
        img[sz // 2, sz // 2, :] = 0.1
        tensor = _make_smooth_blur_tensor()
        flat_dmap = np.full((sz, sz), 1.0, dtype=np.float32)
        out_u8, _out_lin = render_with_defocus(
            img, tensor, dmap=flat_dmap, psf_norm="peak"
        )
        # Overwhelming majority saturates to white after clip; allow a
        # small margin for the dim pixel's neighbourhood and edge-pad
        # effects.
        frac_saturated = float((out_u8 >= 250).mean())
        assert frac_saturated > 0.80, (
            f"Expected majority white from LDR clip (documents spec §3 "
            f"absorb claim); got frac_saturated={frac_saturated:.3f}"
        )

    # --- 2. depth-varied (not flat-max) defocus --------------------------
    def test_peak_mode_depth_varied_ratio_modest(self):
        """Linearly-varying defocus across the image keeps bloom modest.

        Spec §3: "most pixels are dim, ratio stays modest". The Coder's
        gate-trip test used a flat-max defocus and hit 153×. Here we use
        a realistic smooth defocus ramp on a mid-grey image; bloom ratio
        should be well below the 500× safety ceiling and — on mid-grey
        input that does NOT saturate — closer to the low end of the
        10–500× range in spec §3.
        """
        sz = 128
        img = np.full((sz, sz, 3), 0.5, dtype=np.float32)  # mid-grey
        # Realistic ramp (foreground sharp → background blurred).
        dmap = np.linspace(0.0, 1.0, sz, dtype=np.float32)
        dmap = np.broadcast_to(dmap[None, :], (sz, sz)).astype(np.float32).copy()
        tensor = _make_smooth_blur_tensor()
        _out_u8, out_lin = render_with_defocus(
            img, tensor, dmap=dmap, psf_norm="peak"
        )
        input_lin = srgb_to_linear(img)
        ratio = float(out_lin.sum()) / float(input_lin.sum())
        # Upper bound: well below the 500× ceiling so we have headroom.
        # Lower bound: peak mode should still bloom somewhat on mid-grey.
        assert ratio <= 200.0, (
            f"Depth-varied bloom ratio {ratio:.2f} exceeds 200× — spec "
            f"§3 'most pixels dim, ratio stays modest' is violated."
        )
        assert ratio >= 0.5, (
            f"Depth-varied bloom ratio {ratio:.3f} unexpectedly low "
            f"(peak mode should amplify, not average away)."
        )

    # --- 3. near-delta vs wide-blur step seam ----------------------------
    def test_peak_mode_step_defocus_seam(self):
        """Half-d=0 / half-d=1 step: measure brightness discontinuity.

        Core "averaging-is-what-smoothed-the-seam-in-L1" concern.
        In l1 mode, post-conv ``/ weight_sum`` averages kernels of
        differing sum into a per-pixel normalised blend; in peak mode
        each pixel gets stamp-summed directly. Uniform mid-grey input
        should therefore show a visible brightness jump at the seam in
        peak mode — we document this is the case (it is spec-consistent
        behaviour) by asserting the jump exists and characterising its
        magnitude. Spec-consistent, not a bug.
        """
        sz = 128
        img = np.full((sz, sz, 3), 0.5, dtype=np.float32)
        dmap = np.zeros((sz, sz), dtype=np.float32)
        dmap[:, sz // 2:] = 1.0  # right half is max blur
        tensor = _make_smooth_blur_tensor()
        _out_u8, out_lin = render_with_defocus(
            img, tensor, dmap=dmap, psf_norm="peak"
        )
        # Sample brightness well away from the seam so kernel footprint
        # doesn't smear the step into adjacent bins.
        left_mid = float(
            out_lin[sz // 2 - 4: sz // 2 + 4, 8:24, 1].mean()
        )
        right_mid = float(
            out_lin[sz // 2 - 4: sz // 2 + 4, sz - 24: sz - 8, 1].mean()
        )
        # Peak-mode seam is expected — right (d=1, wide kernel) much
        # brighter than left (d=0, delta). Document the expected sign.
        assert right_mid > left_mid, (
            f"Step-defocus seam sign wrong: left={left_mid:.4f}, "
            f"right={right_mid:.4f}"
        )
        # No NaN/Inf even with the discontinuous dmap.
        assert not np.any(np.isnan(out_lin))
        assert not np.any(np.isinf(out_lin))

    # --- 4. very small image (sz < 32) -----------------------------------
    def test_peak_mode_small_image_sz16(self):
        """sz=16 delta-tensor / flat-zero defocus: identity within tolerance.

        The Coder's TestPeakNormStamp.test_in_focus_peak_preservation
        bumped to sz=128 citing partition-of-unity holes at sz<96.
        Confirm what actually happens at sz=16. Delta + d=0 has no slab-
        or wedge-sensitivity because the kernel is a single pixel, so
        the partition-of-unity concern is moot here — output should be
        bit-near identical to input.
        """
        sz = 16
        rng = np.random.default_rng(0)
        img = rng.uniform(0.1, 0.9, size=(sz, sz, 3)).astype(np.float32)
        tensor = _make_delta_tensor()
        flat_dmap = np.zeros((sz, sz), dtype=np.float32)
        _out_u8, out_lin = render_with_defocus(
            img, tensor, dmap=flat_dmap, psf_norm="peak"
        )
        input_lin = srgb_to_linear(img)
        # Delta at d=0 should preserve input even at sz=16 (single-pixel
        # kernel has no spatial extent to expose P-of-U holes).
        max_diff = float(np.max(np.abs(out_lin - input_lin)))
        assert max_diff < 5e-3, (
            f"sz=16 delta-identity broken: max diff {max_diff:.6f}; "
            f"documents whether partition-of-unity hole exposes here"
        )

    def test_peak_mode_small_image_sz16_smooth_tensor(self):
        """sz=16 smooth-blur tensor / flat-max defocus: confirm no NaN.

        With a wide Gaussian kernel at sz=16 the partition-of-unity
        holes flagged by the Coder would manifest as isolated dim
        pixels. In peak mode the post-conv renorm is gone, so such
        holes are not hidden. We do NOT assert uniformity (that would
        codify a known artifact); we assert structural invariants.
        """
        sz = 16
        img = np.full((sz, sz, 3), 0.5, dtype=np.float32)
        tensor = _make_smooth_blur_tensor()
        flat_dmap = np.full((sz, sz), 1.0, dtype=np.float32)
        _out_u8, out_lin = render_with_defocus(
            img, tensor, dmap=flat_dmap, psf_norm="peak"
        )
        assert not np.any(np.isnan(out_lin))
        assert not np.any(np.isinf(out_lin))
        assert out_lin.min() >= -1e-5

    # --- 5. single-color-channel extreme ---------------------------------
    def test_peak_mode_single_channel_no_crosstalk(self):
        """Red-only bright pixel → output's G and B channels stay at 0.

        Peak mode does not change the per-channel independent FFT
        convolve, but we verify no cross-channel contamination was
        introduced by the renorm-skip refactor.
        """
        sz = 64
        img = np.zeros((sz, sz, 3), dtype=np.float32)
        img[sz // 2, sz // 2, 0] = 1.0  # red only
        tensor = _make_smooth_blur_tensor()
        flat_dmap = np.full((sz, sz), 1.0, dtype=np.float32)
        _out_u8, out_lin = render_with_defocus(
            img, tensor, dmap=flat_dmap, psf_norm="peak"
        )
        g_max = float(out_lin[:, :, 1].max())
        b_max = float(out_lin[:, :, 2].max())
        assert g_max < 1e-5, (
            f"Cross-channel contamination: G channel max = {g_max:.6f}"
        )
        assert b_max < 1e-5, (
            f"Cross-channel contamination: B channel max = {b_max:.6f}"
        )
        # Red channel must bloom meaningfully (this is the point of peak mode).
        r_sum = float(out_lin[:, :, 0].sum())
        assert r_sum > 0.5, (
            f"Red channel lost its energy in peak mode: sum={r_sum:.4f}"
        )

    # --- 6. zero-input edge case -----------------------------------------
    def test_peak_mode_all_zeros_input(self):
        """All-zero input: output must be all zeros, no NaN/Inf from skipped renorm.

        In l1 mode ``output /= weight_sum`` handles the 0/0 case via the
        ``np.maximum(weight_sum, 1e-8)`` guard. In peak mode that divide
        is skipped entirely; the accumulator starts at 0 and only gets
        additions of ``w_ang * fftconvolve(zeros, kernel)`` which stays
        zero. Verify no NaN/Inf and exact zero output.
        """
        sz = 64
        img = np.zeros((sz, sz, 3), dtype=np.float32)
        tensor = _make_smooth_blur_tensor()
        flat_dmap = np.full((sz, sz), 1.0, dtype=np.float32)
        out_u8, out_lin = render_with_defocus(
            img, tensor, dmap=flat_dmap, psf_norm="peak"
        )
        assert not np.any(np.isnan(out_lin)), "NaN in peak-mode zero-input output"
        assert not np.any(np.isinf(out_lin)), "Inf in peak-mode zero-input output"
        assert out_u8.max() == 0, (
            f"Expected all-zero uint8 output, got max={out_u8.max()}"
        )
        assert float(out_lin.max()) == 0.0, (
            f"Expected all-zero linear output, got max={float(out_lin.max()):.6f}"
        )

    # --- 7. invalid psf_norm via direct render() call --------------------
    def test_peak_mode_invalid_psf_norm_direct(self):
        """Direct render() call with bogus psf_norm raises ValueError.

        The Coder's CLI test covers argparse rejection. Here verify the
        render() API-level guard fires with a clear message naming the
        parameter.
        """
        sz = 32
        img = np.full((sz, sz, 3), 0.5, dtype=np.float32)
        tensor = _make_delta_tensor()
        with pytest.raises(ValueError, match="psf_norm"):
            render_with_defocus(img, tensor, psf_norm="stamp")  # close-but-wrong
        with pytest.raises(ValueError, match="psf_norm"):
            render_with_defocus(img, tensor, psf_norm="")
        with pytest.raises(ValueError, match="psf_norm"):
            render_with_defocus(img, tensor, psf_norm="L1")  # case-sensitive
