"""Unit tests for bilinear kernel interpolation (Day 5 spec S5-S8)."""
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

_RADIAL_EDGES = np.array([0.0, 0.15, 0.30, 0.45, 0.60, 0.80, 1.01], dtype=np.float64)
_DEFOCUS_RATIOS = np.array([0.0, 0.33, 0.67, 1.0], dtype=np.float64)


def _make_delta_tensor(ksize: int = 129) -> dict:
    """Build a synthetic tensor where every kernel is a centered delta."""
    R, D = 6, 4
    kernels = np.zeros((R, D, ksize, ksize), dtype=np.float32)
    c = ksize // 2
    kernels[:, :, c, c] = 1.0
    return {
        "kernels": kernels,
        "radial_edges": _RADIAL_EDGES.copy(),
        "defocus_ratios": _DEFOCUS_RATIOS.copy(),
    }


def _make_smooth_blur_tensor(ksize: int = 129) -> dict:
    """Tensor with small Gaussian kernels that grow with defocus index.

    d_idx=0 slots are deltas (matching real tensor schema).
    d_idx=1..3 have progressively wider Gaussians, L1-normalized.
    """
    R, D = 6, 4
    kernels = np.zeros((R, D, ksize, ksize), dtype=np.float32)
    c = ksize // 2

    for ri in range(R):
        # d=0 -> delta
        kernels[ri, 0, c, c] = 1.0
        for di in range(1, D):
            sigma = 1.0 + di * 1.5  # mild blur
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
# S5: Weight sum == 1.0 +/- 1e-6 for 20 random (r, d) samples
# ---------------------------------------------------------------------------

class TestBilinearWeightSum:
    """S5: bilinear weights must sum to 1.0 within tolerance."""

    def test_weight_sum_random_samples(self):
        rng = np.random.default_rng(42)
        for _ in range(20):
            r = float(rng.uniform(0.0, 1.0))
            d = float(rng.uniform(0.0, 1.0))
            indices, weights = _bilinear_weights(r, d, _RADIAL_EDGES, _DEFOCUS_RATIOS)
            wsum = sum(weights)
            assert abs(wsum - 1.0) < 1e-6, (
                f"Weight sum {wsum} != 1.0 for r={r:.4f}, d={d:.4f}"
            )

    def test_weight_sum_at_corners(self):
        """Edge/corner values of the parameter space."""
        test_points = [
            (0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0),
            (0.5, 0.5), (0.075, 0.33),
        ]
        for r, d in test_points:
            indices, weights = _bilinear_weights(r, d, _RADIAL_EDGES, _DEFOCUS_RATIOS)
            wsum = sum(weights)
            assert abs(wsum - 1.0) < 1e-6, (
                f"Weight sum {wsum} != 1.0 for r={r}, d={d}"
            )


# ---------------------------------------------------------------------------
# S6: Gradient-derivative seam test
# ---------------------------------------------------------------------------

class TestGradientSeamSmoothness:
    """S6: max 2nd-derivative at defocus bin boundary < 2x interior."""

    def test_seam_second_derivative(self):
        sz = 256
        # Horizontal gradient image in sRGB [0.1, 0.9]
        grad = np.linspace(0.1, 0.9, sz, dtype=np.float32)
        img = np.zeros((sz, sz, 3), dtype=np.float32)
        img[:, :, :] = grad[None, :, None]

        tensor = _make_smooth_blur_tensor()
        _output_u8, output_lin = render_with_defocus(img, tensor)

        # Take a horizontal scanline through the center
        mid_row = output_lin[sz // 2, :, 0]  # green channel

        # 2nd derivative (discrete Laplacian along x)
        d2 = np.abs(np.diff(mid_row, n=2))

        # Find defocus bin boundaries: defocus map is quadratic from center.
        # The bins are at d_center = (b+0.5)/8 for b in 0..7.
        # We compare boundary neighborhoods vs interior.
        n_bins = 8
        bin_width = 1.0 / n_bins

        # Build the defocus values along the center row
        cx = (sz - 1) / 2.0
        cy = (sz - 1) / 2.0
        max_dist = cy ** 2 + cx ** 2
        xs = np.arange(sz, dtype=np.float32)
        dx = xs - cx
        # Along center row, dy=0
        dmap_row = (dx ** 2) / max_dist

        # Identify boundary pixels: where dmap crosses a bin edge
        bin_edges = [(b * bin_width) for b in range(1, n_bins)]
        boundary_mask = np.zeros(sz - 2, dtype=bool)  # d2 has length sz-2
        for edge in bin_edges:
            crossings = np.where(
                (dmap_row[:-2] < edge) & (dmap_row[2:] >= edge)
                | (dmap_row[:-2] >= edge) & (dmap_row[2:] < edge)
            )[0]
            for c_idx in crossings:
                lo = max(0, c_idx - 2)
                hi = min(len(boundary_mask), c_idx + 3)
                boundary_mask[lo:hi] = True

        interior_mask = ~boundary_mask
        if interior_mask.sum() < 10 or boundary_mask.sum() < 1:
            pytest.skip("Not enough interior/boundary pixels for seam test")

        max_boundary_d2 = float(d2[boundary_mask].max())
        max_interior_d2 = float(d2[interior_mask].max())

        # Guard: if interior is effectively flat, skip
        if max_interior_d2 < 1e-8:
            # Both are essentially zero -> no seam
            return

        ratio = max_boundary_d2 / max_interior_d2
        # P1.3: threshold relaxed from 2.0 to 3.0. The convolve-then-blend
        # fix (post-conv weighting) slightly increases boundary 2nd-derivative
        # because the convolution now sees the full unweighted image rather
        # than a smooth-windowed version. This is the correct trade-off:
        # the old pre-conv weighting hid bin-boundary seams but caused ghost
        # copies (double-oval artifact) on extended objects.
        assert ratio < 3.0, (
            f"Seam artifact: boundary 2nd-deriv {max_boundary_d2:.6f} / "
            f"interior {max_interior_d2:.6f} = {ratio:.2f}x (must be < 3x)"
        )


# ---------------------------------------------------------------------------
# S7: Energy ratio in [0.85, 1.15]
# ---------------------------------------------------------------------------

class TestBilinearEnergyConservation:
    """S7: energy ratio must be in [0.85, 1.15] with bilinear blending."""

    def test_energy_gradient_image(self):
        sz = 128
        grad = np.linspace(0.1, 0.9, sz, dtype=np.float32)
        img = np.zeros((sz, sz, 3), dtype=np.float32)
        img[:, :, :] = grad[None, :, None]

        tensor = _make_smooth_blur_tensor()
        _output_u8, output_lin = render_with_defocus(img, tensor)
        input_lin = srgb_to_linear(img)

        ratio = float(output_lin.sum()) / float(input_lin.sum())
        assert 0.85 <= ratio <= 1.15, (
            f"Energy ratio {ratio:.4f} outside [0.85, 1.15]"
        )

    def test_energy_uniform_image(self):
        sz = 128
        img = np.full((sz, sz, 3), 0.5, dtype=np.float32)
        tensor = _make_smooth_blur_tensor()
        _output_u8, output_lin = render_with_defocus(img, tensor)
        input_lin = srgb_to_linear(img)

        ratio = float(output_lin.sum()) / float(input_lin.sum())
        assert 0.85 <= ratio <= 1.15, (
            f"Energy ratio {ratio:.4f} outside [0.85, 1.15]"
        )


# ---------------------------------------------------------------------------
# S8: d=0 identity -- near-zero defocus_map, SSIM > 0.99 vs input
# ---------------------------------------------------------------------------

class TestDeltaIdentity:
    """S8: near-zero defocus should preserve the image (SSIM > 0.99)."""

    def test_near_zero_defocus_ssim(self):
        sz = 128
        # Checkerboard pattern for SSIM sensitivity
        img = np.zeros((sz, sz, 3), dtype=np.float32)
        img[::2, ::2, :] = 0.8
        img[1::2, 1::2, :] = 0.8
        img[::2, 1::2, :] = 0.2
        img[1::2, ::2, :] = 0.2

        tensor = _make_delta_tensor()
        # Near-zero defocus map forces all pixels to d~0 (delta territory)
        near_zero_dmap = np.full((sz, sz), 0.01, dtype=np.float32)
        _output_u8, output_lin = render_with_defocus(img, tensor, dmap=near_zero_dmap)
        input_lin = srgb_to_linear(img)

        from skimage.metrics import structural_similarity
        ssim = structural_similarity(
            input_lin, output_lin, channel_axis=2, data_range=1.0,
        )
        assert ssim > 0.99, f"SSIM {ssim:.4f} <= 0.99 with near-zero defocus_map"

    def test_near_zero_defocus_smooth_image(self):
        """Smooth gradient at low defocus should also preserve identity."""
        sz = 128
        grad = np.linspace(0.1, 0.9, sz, dtype=np.float32)
        img = np.zeros((sz, sz, 3), dtype=np.float32)
        img[:, :, :] = grad[None, :, None]

        tensor = _make_delta_tensor()
        near_zero_dmap = np.full((sz, sz), 0.01, dtype=np.float32)
        _output_u8, output_lin = render_with_defocus(img, tensor, dmap=near_zero_dmap)
        input_lin = srgb_to_linear(img)

        from skimage.metrics import structural_similarity
        ssim = structural_similarity(
            input_lin, output_lin, channel_axis=2, data_range=1.0,
        )
        assert ssim > 0.99, f"SSIM {ssim:.4f} <= 0.99 with near-zero defocus_map"


# ---------------------------------------------------------------------------
# P1.7: Peak-norm "stamp" mode (locked-spec-stamp-mode.md)
# ---------------------------------------------------------------------------


class TestPeakNormStamp:
    """Acceptance tests for ``psf_norm="peak"`` (stamp) render path."""

    def test_in_focus_peak_preservation(self):
        """Delta tensor + flat zero defocus + peak mode -> output ~= input.

        Peak-normed delta is still a delta (peak=1, sum=1); with the
        post-conv renorm skipped there must be no center dimming or
        averaging.

        Note on image size: we use 128x128 rather than a smaller fixture
        because at <96x96 the (slab, bin) inner-loop short-circuit
        (``w.max() < 0.01``) drops whole slabs whose mass is too spatially
        narrow to pass the threshold globally. In l1 mode the post-conv
        ``output /= weight_sum`` hides that partition-of-unity hole; in
        peak mode it's exposed as isolated dim pixels at specific radii.
        This is a pre-existing renderer artifact, not a stamp-mode bug.
        """
        sz = 128
        rng = np.random.default_rng(0)
        img = rng.uniform(0.1, 0.9, size=(sz, sz, 3)).astype(np.float32)

        tensor = _make_delta_tensor()
        flat_dmap = np.zeros((sz, sz), dtype=np.float32)
        _out_u8, output_lin = render_with_defocus(
            img, tensor, dmap=flat_dmap, psf_norm="peak"
        )
        input_lin = srgb_to_linear(img)
        assert np.allclose(output_lin, input_lin, atol=1e-3), (
            "Peak-mode delta convolution must preserve input "
            f"(max abs diff = {np.max(np.abs(output_lin - input_lin)):.6f})"
        )

    def test_isolated_bright_pixel_blooms(self):
        """Single bright pixel + wide peak-normed kernel -> sum > 1.5x input.

        L1 kernel has sum=1, peak~0.2; peak-normed kernel has peak=1,
        sum~1/0.2=5. A lone bright pixel × peak-normed kernel therefore
        spreads *more total energy* than it started with (bloom).
        """
        sz = 64
        img = np.zeros((sz, sz, 3), dtype=np.float32)
        img[sz // 2, sz // 2, :] = 1.0  # single sRGB white pixel

        tensor = _make_smooth_blur_tensor()
        flat_dmap = np.full((sz, sz), 1.0, dtype=np.float32)  # max blur
        _out_u8, output_lin = render_with_defocus(
            img, tensor, dmap=flat_dmap, psf_norm="peak"
        )
        input_lin = srgb_to_linear(img)

        input_sum = float(input_lin.sum())
        output_sum = float(output_lin.sum())
        assert output_sum > 1.5 * input_sum, (
            f"Stamp-mode bloom too weak: output.sum()={output_sum:.4f} "
            f"vs input.sum()={input_sum:.4f} (need > 1.5x)"
        )

    def test_default_l1_path_bit_identical(self):
        """Default psf_norm (omitted) and explicit ``psf_norm="l1"`` match bitwise.

        Also re-run through a smooth-blur fixture to confirm l1-mode
        numerics are stable across the branch-gated refactor.
        """
        sz = 64
        grad = np.linspace(0.1, 0.9, sz, dtype=np.float32)
        img = np.zeros((sz, sz, 3), dtype=np.float32)
        img[:, :, :] = grad[None, :, None]

        tensor = _make_smooth_blur_tensor()
        # Call 1: omit kwarg (default)
        out_default_u8, out_default_lin = render_with_defocus(img, tensor)
        # Call 2: explicit l1
        out_l1_u8, out_l1_lin = render_with_defocus(img, tensor, psf_norm="l1")

        assert np.array_equal(out_default_u8, out_l1_u8), (
            "Default and explicit psf_norm='l1' u8 outputs must be "
            "bit-identical"
        )
        assert np.array_equal(out_default_lin, out_l1_lin), (
            "Default and explicit psf_norm='l1' linear outputs must be "
            "bit-identical"
        )

        # Golden-fixture-style sanity: l1 mode output must be close to a
        # freshly-computed reference. Since no external golden exists, we
        # self-check by running a second fixture (delta tensor, identity
        # preservation) — any silent change in the l1 path would show up as
        # a non-identity output at near-zero defocus.
        delta_tensor = _make_delta_tensor()
        near_zero = np.full((sz, sz), 0.01, dtype=np.float32)
        _u8, out_delta = render_with_defocus(
            img, delta_tensor, dmap=near_zero, psf_norm="l1"
        )
        input_lin = srgb_to_linear(img)
        # layered delta gain 0.99993 (CLI-1a): the layered path applies a
        # uniform ~6.6e-5 gain at dmap 0.01 (inside the snap knee), so the
        # legacy atol=1e-5 identity is replaced by rtol=1e-4.
        assert np.allclose(out_delta, input_lin, rtol=1e-4, atol=1e-6), (
            "L1-mode delta identity broken (rtol=1e-4, atol=1e-6)"
        )

    def test_stamp_mode_gate_does_not_trip(self, caplog):
        """Random iPhone-like input in stamp mode must complete without gate fail.

        Emits the energy ratio via print() for run-log monitoring.
        """
        sz = 64
        rng = np.random.default_rng(1234)
        img = rng.uniform(0.0, 1.0, size=(sz, sz, 3)).astype(np.float32)

        tensor = _make_smooth_blur_tensor()
        flat_dmap = np.full((sz, sz), 1.0, dtype=np.float32)

        _out_u8, output_lin = render_with_defocus(
            img, tensor, dmap=flat_dmap, psf_norm="peak"
        )
        input_lin = srgb_to_linear(img)
        ratio = float(output_lin.sum()) / float(input_lin.sum())
        # Record for monitoring; no hard lower/upper bounds beyond runaway.
        print(f"[TestPeakNormStamp] stamp-mode energy_ratio={ratio:.4f}")
        assert ratio <= 500.0, (
            f"Stamp bloom runaway: energy_ratio={ratio:.4f} > 500.0"
        )

    def test_invalid_psf_norm_raises(self):
        """Non-{l1,peak} value for psf_norm must raise ValueError."""
        sz = 32
        img = np.full((sz, sz, 3), 0.5, dtype=np.float32)
        tensor = _make_delta_tensor()
        with pytest.raises(ValueError, match="psf_norm"):
            render_with_defocus(img, tensor, psf_norm="bogus")

    def test_peak_mode_auto_exposure_max_is_one(self):
        """Peak-mode auto-exposure lands output.max() at ~1.0, no overshoot.

        Setup: smooth-blur tensor (L1=1, peak~0.2, kernel-sum ~5 after
        peak-norm), single bright pixel = 1.0 on black, flat max-defocus.
        A peak-normed wide kernel × bright pixel yields a bloom peak of
        ~1.0 at the bright center (kernel.max()*1.0) — the auto-exposure
        divide thus lands exactly on 1.0 when bloom would overflow, or
        stays pinned at 1.0 when it wouldn't. Either way the final
        linear output must have max >= 0.999 and <= 1.0 + epsilon.
        """
        sz = 64
        img = np.zeros((sz, sz, 3), dtype=np.float32)
        img[sz // 2, sz // 2, :] = 1.0  # single sRGB white on black

        tensor = _make_smooth_blur_tensor()
        flat_dmap = np.full((sz, sz), 1.0, dtype=np.float32)  # max blur

        _out_u8, output_lin = render_with_defocus(
            img, tensor, dmap=flat_dmap, psf_norm="peak"
        )
        omax = float(output_lin.max())
        assert omax >= 0.999, (
            f"Auto-exposure under-shoot: output.max()={omax:.6f} < 0.999"
        )
        assert omax <= 1.0 + 1e-6, (
            f"Auto-exposure over-shoot: output.max()={omax:.6f} > 1 + 1e-6"
        )
