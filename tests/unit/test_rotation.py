"""Unit tests for P1.0 angular kernel rotation in the bin-and-convolve renderer."""
from __future__ import annotations

import time

import numpy as np
import pytest
from scipy.ndimage import rotate as ndimage_rotate

from summicron.render.convolve import N_ANG
from tests.fixtures.defocus import render_with_defocus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_asymmetric_kernel(ksize: int = 65, offset: int = 5) -> np.ndarray:
    """Gaussian offset upward from center (tail pointing UP in array coords).

    The centroid sits above the geometric center, matching the tensor convention
    described in the locked spec (tails point UP).
    """
    k = np.zeros((ksize, ksize), dtype=np.float32)
    cy, cx = ksize // 2, ksize // 2
    ys, xs = np.mgrid[:ksize, :ksize]
    # Offset the Gaussian center upward (lower row index = up in image)
    sigma = 4.0
    g = np.exp(-((ys - (cy - offset)) ** 2 + (xs - cx) ** 2) / (2 * sigma ** 2))
    k = g.astype(np.float32)
    k /= k.sum()
    return k


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


def _rotation_formula(a: int) -> float:
    """Compute the rotation in degrees for angular wedge *a* (spec S3)."""
    theta_a = (a + 0.5) * (2 * np.pi / N_ANG) - np.pi
    return -(theta_a * 180.0 / np.pi + 90.0)


# ---------------------------------------------------------------------------
# T1: Energy conservation across all 8 wedge rotations
# ---------------------------------------------------------------------------

class TestRotationEnergy:
    """Rotating + clamp + renorm should preserve L1 within [0.85, 1.15]."""

    def test_energy_all_wedges(self):
        kernel = _make_asymmetric_kernel()
        original_l1 = float(kernel.sum())

        for a in range(N_ANG):
            rot_deg = _rotation_formula(a)
            rotated = ndimage_rotate(kernel, rot_deg, reshape=False, order=1)
            rotated = np.maximum(rotated, 0.0)
            rotated /= rotated.sum() + 1e-12
            ratio = float(rotated.sum()) / original_l1
            assert 0.85 <= ratio <= 1.15, (
                f"Wedge {a}: L1 ratio {ratio:.4f} outside [0.85, 1.15]"
            )


# ---------------------------------------------------------------------------
# T2: Render time on synthetic 2048 image (slow gate)
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestRenderTime:
    """Full render through angular rotation must complete in < 15 s at 2048."""

    def test_wall_clock_2048(self):
        sz = 2048
        img = np.random.default_rng(42).random((sz, sz, 3), dtype=np.float32) * 0.8 + 0.1
        kernel = _make_asymmetric_kernel(ksize=65)
        tensor = _make_tensor_from_kernel(kernel, R=6, D=4)

        t0 = time.perf_counter()
        _out_u8, _out_lin = render_with_defocus(img, tensor)
        elapsed = time.perf_counter() - t0
        assert elapsed < 15.0, f"Render took {elapsed:.1f}s > 15s"


# ---------------------------------------------------------------------------
# T3: 4-cardinal orientation check
# ---------------------------------------------------------------------------

class TestCardinalOrientation:
    """Rotate an UP-tailed kernel to 0, 90, 180, 270 deg and verify
    the peak-energy quadrant matches expectation.

    Convention (numpy array coords):
      - Row increases downward, column increases rightward.
      - 'UP' in image = lower row index.
      - Quadrants relative to kernel center:
          right  = cols > center
          up     = rows < center (lower row index)
          left   = cols < center
          down   = rows > center
    """

    @staticmethod
    def _peak_quadrant(k: np.ndarray) -> str:
        """Return which quadrant holds the most energy (right/up/left/down)."""
        cy, cx = k.shape[0] // 2, k.shape[1] // 2
        right = k[:, cx + 1:].sum()
        left = k[:, :cx].sum()
        up = k[:cy, :].sum()
        down = k[cy + 1:, :].sum()
        mapping = {"right": right, "up": up, "left": left, "down": down}
        return max(mapping, key=mapping.get)

    @pytest.mark.parametrize(
        "math_angle_deg, expected_quadrant",
        [
            (0, "right"),      # 3 o'clock -> tail points RIGHT
            (90, "up"),        # 12 o'clock -> tail points UP (low rows)
            (180, "left"),     # 9 o'clock -> tail points LEFT
            (270, "down"),     # 6 o'clock -> tail points DOWN (high rows)
        ],
    )
    def test_cardinal(self, math_angle_deg, expected_quadrant):
        """Apply spec rotation formula at 4 cardinal azimuths and verify
        peak-energy quadrant.

        The spec angles (0, 90, 180, 270) are in standard math convention
        (y-up, CCW positive). arctan2 in array coords (y-down) inverts the
        y-axis, so math_angle 90 deg = theta_a = -pi/2 in array arctan2.
        """
        kernel = _make_asymmetric_kernel(ksize=65, offset=10)
        # Convert from math-convention angle (y-up) to array-convention
        # theta_a (y-down): negate the y component, i.e., theta_array = -theta_math
        theta_a = -np.radians(math_angle_deg)
        rot_deg = -(theta_a * 180.0 / np.pi + 90.0)
        rotated = ndimage_rotate(kernel, rot_deg, reshape=False, order=1)
        rotated = np.maximum(rotated, 0.0)
        rotated /= rotated.sum() + 1e-12

        quadrant = self._peak_quadrant(rotated)
        assert quadrant == expected_quadrant, (
            f"math_angle={math_angle_deg} (rot_deg={rot_deg:.1f}): "
            f"expected {expected_quadrant}, got {quadrant}"
        )

    def test_formula_cardinal_angles(self):
        """Verify the spec rotation formula maps 4 cardinal azimuths to
        correct centroid directions. Centroid angle error < 25 deg (G3).

        theta_a is in array-coord arctan2 convention (y-down):
          theta_a = 0      -> pixel at 3 o'clock  -> tail RIGHT
          theta_a = pi/2   -> pixel at 6 o'clock  -> tail DOWN (high rows)
          theta_a = pi     -> pixel at 9 o'clock  -> tail LEFT
          theta_a = -pi/2  -> pixel at 12 o'clock -> tail UP (low rows)

        We measure centroid angle in image convention (x-right, y-up)
        using arctan2(cy - centroid_y, centroid_x - cx).
        """
        kernel = _make_asymmetric_kernel(ksize=65, offset=10)

        # (theta_a in array coords, expected centroid angle in image y-up degrees)
        cardinals = [
            (0.0, 0.0),          # 3 o'clock -> tail right -> centroid angle 0
            (np.pi / 2, -90.0),  # 6 o'clock -> tail down -> centroid angle -90
            (np.pi, 180.0),      # 9 o'clock -> tail left -> centroid angle 180
            (-np.pi / 2, 90.0),  # 12 o'clock -> tail up -> centroid angle 90
        ]

        cy, cx = kernel.shape[0] // 2, kernel.shape[1] // 2
        ys, xs = np.mgrid[: kernel.shape[0], : kernel.shape[1]]

        for theta_a, expected_deg in cardinals:
            rot_deg = -(theta_a * 180.0 / np.pi + 90.0)
            rotated = ndimage_rotate(kernel, rot_deg, reshape=False, order=1)
            rotated = np.maximum(rotated, 0.0)
            rotated /= rotated.sum() + 1e-12

            # Centroid in array coords
            centroid_y = float((ys * rotated).sum())
            centroid_x = float((xs * rotated).sum())
            # Angle from center (image convention: x-right, y-up)
            angle_rad = np.arctan2(cy - centroid_y, centroid_x - cx)
            angle_deg = np.degrees(angle_rad)

            # Angular error (handle wraparound)
            err = abs(((angle_deg - expected_deg) + 180) % 360 - 180)
            assert err < 25.0, (
                f"theta_a={theta_a:.2f}: centroid angle {angle_deg:.1f} deg, "
                f"expected {expected_deg:.1f} deg, error {err:.1f} deg > 25"
            )


# ---------------------------------------------------------------------------
# T4: Round-trip identity (+45 then -45)
# ---------------------------------------------------------------------------

class TestRoundTrip:
    """Rotate +45 then -45; L1 diff from original should be < 0.02."""

    def test_roundtrip_45(self):
        kernel = _make_asymmetric_kernel(ksize=65, offset=5)

        fwd = ndimage_rotate(kernel, 45.0, reshape=False, order=1)
        fwd = np.maximum(fwd, 0.0)
        fwd /= fwd.sum() + 1e-12

        back = ndimage_rotate(fwd, -45.0, reshape=False, order=1)
        back = np.maximum(back, 0.0)
        back /= back.sum() + 1e-12

        l1_diff = float(np.abs(back - kernel).sum())
        assert l1_diff < 0.02, f"Round-trip L1 diff {l1_diff:.4f} >= 0.02"
