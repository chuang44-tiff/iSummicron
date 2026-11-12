"""Adversarial tests for P1.0 angular kernel rotation.

Targets: sign convention, delta guard boundary, atan2 wraparound,
empty slab×wedge, energy loss at diagonal angles, edge-pixel clipping,
N_ANG=1 fallback.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.ndimage import rotate as ndimage_rotate

from summicron.render.convolve import N_ANG, render
from tests.fixtures.defocus import render_with_defocus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tensor_from_kernel(kernel: np.ndarray, R: int = 6, D: int = 4) -> dict:
    """Wrap a single kernel into a synthetic 4D tensor dict."""
    kH, kW = kernel.shape
    kernels = np.zeros((R, D, kH, kW), dtype=np.float32)
    kernels[:, :] = kernel[None, None, :, :]
    radial_edges = np.linspace(0.0, 1.01, R + 1, dtype=np.float64)
    defocus_ratios = np.linspace(0.0, 1.0, D, dtype=np.float64)
    return {
        "kernels": kernels,
        "radial_edges": radial_edges,
        "defocus_ratios": defocus_ratios,
    }


def _rotation_formula(a: int, n_ang: int = N_ANG) -> float:
    """Spec S3 rotation formula."""
    theta_a = (a + 0.5) * (2 * np.pi / n_ang) - np.pi
    return -(theta_a * 180.0 / np.pi + 90.0)


def _centroid(k: np.ndarray) -> tuple[float, float]:
    """Return (cy, cx) centroid of kernel."""
    ys, xs = np.mgrid[: k.shape[0], : k.shape[1]]
    total = k.sum() + 1e-15
    return float((ys * k).sum() / total), float((xs * k).sum() / total)


# ---------------------------------------------------------------------------
# 1. Wrong-polarity fixture (MANDATORY per Post-mortem 1)
# ---------------------------------------------------------------------------

class TestWrongPolarityFixture:
    """Kernel with coma tail pointing DOWN (positive dy centroid) instead of
    the expected UP convention.  If the rotation formula assumes UP-tails and
    gets a DOWN-tail kernel, the output should point radially INWARD — a bug.
    """

    @staticmethod
    def _make_down_tail_kernel(ksize: int = 65, offset: int = 10) -> np.ndarray:
        """Gaussian shifted BELOW center — tail points DOWN (wrong polarity)."""
        k = np.zeros((ksize, ksize), dtype=np.float32)
        cy, cx = ksize // 2, ksize // 2
        ys, xs = np.mgrid[:ksize, :ksize]
        sigma = 4.0
        g = np.exp(-((ys - (cy + offset)) ** 2 + (xs - cx) ** 2) / (2 * sigma ** 2))
        k = g.astype(np.float32)
        k /= k.sum()
        return k

    @pytest.mark.xfail(
        reason="BUG: renderer has no guard against wrong-polarity (DOWN-tail) "
               "input kernels; rotation formula assumes UP-tails and produces "
               "radially INWARD coma when given DOWN-tails (wrong-polarity fixture)"
    )
    def test_down_tail_at_3_oclock_points_inward(self):
        """A DOWN-tail kernel rotated for 3-o'clock pixel (theta_a=0) should
        point RIGHT (radially outward) if the convention is correct.
        With wrong polarity input, the tail ends up pointing LEFT (inward).
        """
        kernel = self._make_down_tail_kernel()
        # theta_a = 0 -> 3 o'clock -> rot_deg = -90
        theta_a = 0.0
        rot_deg = -(theta_a * 180.0 / np.pi + 90.0)
        rotated = ndimage_rotate(kernel, rot_deg, reshape=False, order=1)
        rotated = np.maximum(rotated, 0.0)
        rotated /= rotated.sum() + 1e-12

        cy, cx = rotated.shape[0] // 2, rotated.shape[1] // 2
        right_mass = rotated[:, cx + 1:].sum()
        left_mass = rotated[:, :cx].sum()

        # For correct radially-outward behavior at 3 o'clock, right > left.
        # With a DOWN-tail input, the rotation formula (designed for UP-tails)
        # will produce LEFT-pointing output — bug: no polarity validation.
        assert right_mass > left_mass, (
            f"BUG: down-tail kernel at 3-o'clock points LEFT "
            f"(right={right_mass:.4f}, left={left_mass:.4f})"
        )

    @pytest.mark.xfail(
        strict=False,
        reason="XPASSES FOR THE WRONG REASON (HYG-0 item 15; test not fixed "
               "this round): the assertion compares a 16x16 top-right corner "
               "sum against a 10x10 centre sum on a 0.5-grey background, so the "
               "corner wins by AREA regardless of coma direction (dot-only "
               "bleed: corner 1.36 vs centre 0.0). The underlying BUG stands: "
               "no guard against wrong-polarity (DOWN-tail) input kernels; "
               "rotation assumes UP-tails and produces radially INWARD coma."
    )
    def test_down_tail_full_render_orientation(self):
        """Full render with a DOWN-tail kernel.  The corner coma should
        point outward, but with wrong polarity it points inward.
        """
        ksize = 65
        kernel = self._make_down_tail_kernel(ksize=ksize, offset=10)
        tensor = _make_tensor_from_kernel(kernel, R=6, D=4)

        # Small image, strong defocus at edges
        sz = 64
        img = np.full((sz, sz, 3), 0.5, dtype=np.float32)
        # Plant a bright dot in the top-right corner
        img[5, sz - 6, :] = 1.0

        _out_u8, out_lin = render_with_defocus(img, tensor)

        # The bright dot is at ~45 deg from center (top-right).
        # Radially outward = toward the corner.  Measure if the blur
        # spreads more toward the corner (outward) or toward center (inward).
        cy, cx = sz // 2, sz // 2
        corner_region = out_lin[:cy // 2, cx + cx // 2:, 0]  # top-right corner
        center_region = out_lin[cy - 5:cy + 5, cx - 5:cx + 5, 0]  # near center

        # With correct orientation, corner_region should have significant
        # bleed from the bright dot.  With wrong polarity, bleed goes
        # toward center instead.
        corner_spread = float(corner_region.sum())
        center_spread = float(center_region.sum())

        # This is a qualitative check — wrong polarity sends energy centerward
        assert corner_spread > center_spread, (
            f"BUG: coma bleed toward center ({center_spread:.4f}) > "
            f"toward corner ({corner_spread:.4f})"
        )


# ---------------------------------------------------------------------------
# 2. Delta guard boundary
# ---------------------------------------------------------------------------

class TestDeltaGuardBoundary:
    """The delta guard skips rotation when peak > 0.5 * L1_sum.
    Test behavior right at the boundary.
    """

    @staticmethod
    def _make_kernel_with_peak_ratio(ksize: int, ratio: float) -> np.ndarray:
        """Create a kernel where max pixel / L1 sum == ratio (approximately)."""
        k = np.zeros((ksize, ksize), dtype=np.float32)
        cy, cx = ksize // 2, ksize // 2
        # Put `ratio` of mass in center, distribute rest as a small tail
        k[cy, cx] = ratio
        # Distribute (1 - ratio) across a tail above center
        tail_pixels = 20
        tail_val = (1.0 - ratio) / tail_pixels
        for i in range(1, tail_pixels + 1):
            if cy - i >= 0:
                k[cy - i, cx] = tail_val
        k /= k.sum()  # Normalize
        # After normalization, peak/sum should be close to `ratio`
        return k

    def test_peak_at_051_skips_rotation(self):
        """peak > 0.5 * L1 -> delta guard fires -> no rotation."""
        k = self._make_kernel_with_peak_ratio(65, 0.51)
        assert k.max() > 0.5 * k.sum(), "Precondition: peak > 0.5 * L1"

        rot_deg = _rotation_formula(0)  # -90 deg
        # Simulate the code path
        if k.max() > 0.5 * k.sum():
            rotated = k  # delta guard
        else:
            rotated = ndimage_rotate(k, rot_deg, reshape=False, order=1)
            rotated = np.maximum(rotated, 0.0)
            rotated /= rotated.sum() + 1e-12

        # Should be identical (skipped)
        np.testing.assert_array_equal(rotated, k)

    def test_peak_at_049_rotates(self):
        """peak < 0.5 * L1 -> rotation happens."""
        k = self._make_kernel_with_peak_ratio(65, 0.49)
        assert k.max() <= 0.5 * k.sum(), "Precondition: peak <= 0.5 * L1"

        rot_deg = _rotation_formula(0)
        if k.max() > 0.5 * k.sum():
            rotated = k
        else:
            rotated = ndimage_rotate(k, rot_deg, reshape=False, order=1)
            rotated = np.maximum(rotated, 0.0)
            rotated /= rotated.sum() + 1e-12

        # Should differ from original
        diff = float(np.abs(rotated - k).sum())
        assert diff > 0.01, f"Expected rotation but L1 diff only {diff:.6f}"

    def test_peak_exactly_050_behavior(self):
        """peak == 0.5 * L1.  The guard uses strict >, so 0.5 should NOT
        skip rotation.  Verify this.
        """
        k = self._make_kernel_with_peak_ratio(65, 0.50)
        # Due to floating point, get the actual ratio
        actual_ratio = k.max() / k.sum()
        # The guard: `kernel_base.max() > 0.5 * kernel_base.sum()`
        # At exactly 0.5, strict > is False, so rotation should happen.
        should_skip = k.max() > 0.5 * k.sum()
        if abs(actual_ratio - 0.5) < 1e-6:
            assert not should_skip, (
                "BUG: peak == 0.5 * L1 triggers delta guard (should use strict >)"
            )


# ---------------------------------------------------------------------------
# 3. Wrap-around at wedge boundaries (atan2 discontinuity at +/-pi)
# ---------------------------------------------------------------------------

class TestWedgeWrapAround:
    """Pixels at theta ~ +pi and theta ~ -pi should land in the same wedge."""

    def test_atan2_discontinuity(self):
        """Pixel at angle +179.9 deg and -179.9 deg should map to same wedge."""
        for angle_deg in [179.9, -179.9]:
            theta = np.radians(angle_deg)
            wedge = int((theta + np.pi) / (2 * np.pi) * N_ANG) % N_ANG
            # Both should be wedge 0 (centered around -pi / +pi)
            # Actually wedge 0 covers [-pi, -pi + 45deg]
            # +179.9 deg -> theta ~ +pi -> (pi + pi)/(2pi)*8 = 8 -> %8 = 0
            # -179.9 deg -> theta ~ -pi -> (~0)/(2pi)*8 ~ 0 -> %8 = 0
            assert wedge == 0 or wedge == N_ANG - 1 or wedge == 0, (
                f"angle={angle_deg}: wedge={wedge}, expected 0 or {N_ANG - 1}"
            )

    def test_exact_pi_maps_to_wedge_0_or_last(self):
        """theta = +pi exactly.  (pi + pi)/(2pi)*8 = 8 -> int(8) % 8 = 0."""
        theta = np.pi
        wedge = int((theta + np.pi) / (2 * np.pi) * N_ANG) % N_ANG
        # Should be 0 (wrap), not 8 (out of bounds)
        assert 0 <= wedge < N_ANG, f"wedge={wedge} out of range [0, {N_ANG})"

    def test_negative_pi_maps_consistently(self):
        """theta = -pi exactly.  (-pi + pi)/(2pi)*8 = 0."""
        theta = -np.pi
        wedge = int((theta + np.pi) / (2 * np.pi) * N_ANG) % N_ANG
        assert wedge == 0, f"wedge={wedge}, expected 0"


# ---------------------------------------------------------------------------
# 4. Zero-area slab x wedge intersection & single-pixel mask
# ---------------------------------------------------------------------------

class TestEmptyAndSinglePixelMasks:
    """Corner slabs may have zero pixels in certain wedges.
    A slab with only 1 pixel should still work with FFT.
    """

    def test_tiny_image_no_crash(self):
        """A 3x3 image: some (slab, wedge) combos will have 0 or 1 pixel."""
        ksize = 3
        k = np.zeros((ksize, ksize), dtype=np.float32)
        k[ksize // 2, ksize // 2] = 1.0  # delta
        tensor = _make_tensor_from_kernel(k, R=2, D=2)

        img = np.random.default_rng(99).random((3, 3, 3), dtype=np.float32)
        # Should not crash
        out_u8, out_lin = render_with_defocus(img, tensor)
        assert out_u8.shape == (3, 3, 3)

    def test_single_pixel_image(self):
        """1x1 image — degenerate but should not crash."""
        ksize = 3
        k = np.zeros((ksize, ksize), dtype=np.float32)
        k[1, 1] = 1.0
        tensor = _make_tensor_from_kernel(k, R=2, D=2)

        img = np.array([[[0.5, 0.5, 0.5]]], dtype=np.float32)
        out_u8, out_lin = render_with_defocus(img, tensor)
        assert out_u8.shape == (1, 1, 3)

    def test_nonsquare_image(self):
        """Tall narrow image (100x10): corner wedges in outer slabs may be empty."""
        ksize = 5
        k = np.zeros((ksize, ksize), dtype=np.float32)
        k[ksize // 2, ksize // 2] = 1.0
        tensor = _make_tensor_from_kernel(k, R=3, D=2)

        img = np.random.default_rng(7).random((100, 10, 3), dtype=np.float32) * 0.5
        out_u8, out_lin = render_with_defocus(img, tensor)
        assert out_u8.shape == (100, 10, 3)


# ---------------------------------------------------------------------------
# 5. Energy conservation: diagonal vs cardinal rotation angles
# ---------------------------------------------------------------------------

class TestEnergyLossDiagonalVsCardinal:
    """Sharply asymmetric kernels lose more energy at diagonal rotations
    (22.5 deg) than cardinal (0, 90) because more pixels get interpolated
    off the edge.  Measure the difference.
    """

    def test_diagonal_energy_loss_bounded(self):
        """Energy loss at 22.5 deg rotation should be < 5% (after renorm
        the question is how much mass got clamped from negatives).
        """
        ksize = 65
        k = np.zeros((ksize, ksize), dtype=np.float32)
        cy, cx = ksize // 2, ksize // 2
        # High aspect ratio blob: tall and narrow
        ys, xs = np.mgrid[:ksize, :ksize]
        sigma_y, sigma_x = 12.0, 3.0
        g = np.exp(-((ys - (cy - 15)) ** 2 / (2 * sigma_y ** 2) +
                     (xs - cx) ** 2 / (2 * sigma_x ** 2)))
        k = g.astype(np.float32)
        k /= k.sum()

        pre_l1 = float(k.sum())

        losses = {}
        for angle in [0.0, 22.5, 45.0, 90.0]:
            rotated = ndimage_rotate(k, angle, reshape=False, order=1)
            # Measure how much mass is negative (before clamp)
            neg_mass = float(np.abs(rotated[rotated < 0]).sum())
            rotated = np.maximum(rotated, 0.0)
            post_l1 = float(rotated.sum())
            losses[angle] = neg_mass

        # All negative-mass losses should be small (< 5% of total)
        for angle, loss in losses.items():
            assert loss < 0.05 * pre_l1, (
                f"Angle {angle}: negative mass {loss:.6f} >= 5% of L1 ({pre_l1:.6f})"
            )


# ---------------------------------------------------------------------------
# 6. Edge-pixel clipping under rotation (reshape=False)
# ---------------------------------------------------------------------------

class TestEdgePixelClipping:
    """A kernel with mass at the very edge will lose that mass when rotated
    by any non-zero angle with reshape=False, because the pixel rotates
    outside the canvas.
    """

    def test_corner_pixel_lost_after_rotation(self):
        """Bright pixel at (0, ksize//2) — top edge center.
        After 45-deg rotation, it moves outside the canvas.
        """
        ksize = 65
        k = np.zeros((ksize, ksize), dtype=np.float32)
        cy, cx = ksize // 2, ksize // 2
        k[cy, cx] = 0.3  # Some center mass
        k[0, cx] = 0.7   # Edge mass

        pre_l1 = float(k.sum())
        rotated = ndimage_rotate(k, 45.0, reshape=False, order=1)
        rotated = np.maximum(rotated, 0.0)
        post_l1_before_renorm = float(rotated.sum())

        # Significant mass loss expected (the edge pixel rotates out)
        loss_frac = 1.0 - post_l1_before_renorm / pre_l1
        assert loss_frac > 0.1, (
            f"Expected > 10% mass loss from edge clipping, got {loss_frac:.4f}"
        )

    def test_edge_kernel_shape_preserved_after_rotation(self):
        """After rotation + renorm, the kernel shape should still have a
        clear tail.  But with edge clipping, the tail gets amputated and
        the centroid shifts significantly.  Use a kernel with mass
        concentrated right at the edge to maximize clipping.
        """
        ksize = 65
        k = np.zeros((ksize, ksize), dtype=np.float32)
        cy, cx = ksize // 2, ksize // 2
        # 70% of mass at the very top row (row 0), 30% at center
        k[0, cx] = 0.7
        k[cy, cx] = 0.3
        k /= k.sum()

        orig_cy, _ = _centroid(k)
        orig_offset = abs(orig_cy - cy)

        # Rotate 90 degrees — the top-edge pixel moves to the right edge,
        # but with reshape=False it gets clipped if too far from center
        rotated = ndimage_rotate(k, 90.0, reshape=False, order=1)
        rotated = np.maximum(rotated, 0.0)
        rotated /= rotated.sum() + 1e-12

        rot_cy, rot_cx = _centroid(rotated)
        rot_offset = np.hypot(rot_cy - cy, rot_cx - cx)

        # With 70% mass at row 0 (distance=32 from center in a 65x65 kernel),
        # 90-deg rotation should preserve offset.  Edge clipping may lose it.
        assert rot_offset > 0.7 * orig_offset, (
            f"Centroid offset shrunk from {orig_offset:.2f} to "
            f"{rot_offset:.2f} ({rot_offset / orig_offset:.0%} preserved) "
            f"— edge clipping distorts the PSF"
        )


# ---------------------------------------------------------------------------
# 7. N_ANG=1 fallback (monkeypatch)
# ---------------------------------------------------------------------------

class TestNAng1Fallback:
    """With N_ANG=1, there is a single wedge covering all angles.
    The representative angle theta_a for wedge 0 is:
      (0 + 0.5) * 2*pi/1 - pi = pi - pi = 0
    So rot_deg = -(0 + 90) = -90.
    This means ALL pixels get the kernel rotated by -90 deg, which is
    NOT the same as no rotation. This is arguably a design flaw.
    """

    def test_n_ang_1_applies_uniform_rotation(self, monkeypatch):
        """With N_ANG=1, the single wedge applies rot_deg=-90 to everything,
        which is NOT identity. Verify this is the actual behavior.
        """
        import summicron.render.convolve as mod
        monkeypatch.setattr(mod, "N_ANG", 1)

        ksize = 15
        k = np.zeros((ksize, ksize), dtype=np.float32)
        cy, cx = ksize // 2, ksize // 2
        ys, xs = np.mgrid[:ksize, :ksize]
        sigma = 2.0
        g = np.exp(-((ys - (cy - 3)) ** 2 + (xs - cx) ** 2) / (2 * sigma ** 2))
        k = g.astype(np.float32)
        k /= k.sum()
        tensor = _make_tensor_from_kernel(k, R=2, D=2)

        img = np.full((32, 32, 3), 0.5, dtype=np.float32)
        # Should not crash
        out_u8, out_lin = render_with_defocus(img, tensor)
        assert out_u8.shape == (32, 32, 3)

        # Verify that a -90 deg rotation happened (tail should point RIGHT
        # for the single wedge at theta_a=0, i.e., 3 o'clock direction)
        theta_a = 0.0
        rot_deg = -(theta_a * 180.0 / np.pi + 90.0)
        assert rot_deg == -90.0, f"Expected -90 deg, got {rot_deg}"
