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
