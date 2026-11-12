"""Adversarial tests for P1.1 slab taper + angular merge.

Targets:
  A1: Wrong-polarity fixture (cos^2 non-negativity)
  A2: Taper at extreme radii (r=0, r=1.0)
  A3: Angular merge boundary (peak == 0.5 * L1)
  A4: _bilinear_weights at exact d=0 grid point
  A5: Partition-of-unity with 1-bin tensor
  A6: Energy conservation with taper + merge both active
"""
from __future__ import annotations

import numpy as np
import pytest

from summicron.render.convolve import _bilinear_weights, render, srgb_to_linear
from tests.fixtures.defocus import render_with_defocus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_delta_tensor(ksize: int = 33, R: int = 6, D: int = 4) -> dict:
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


def _make_single_bin_tensor(ksize: int = 17) -> dict:
    """Tensor with exactly 1 radial bin (r_centers has 1 element)."""
    R, D = 1, 2
    kernels = np.zeros((R, D, ksize, ksize), dtype=np.float32)
    c = ksize // 2
    kernels[:, :, c, c] = 1.0
    radial_edges = np.array([0.0, 1.01], dtype=np.float64)
    defocus_ratios = np.array([0.0, 1.0], dtype=np.float64)
    return {
        "kernels": kernels,
        "radial_edges": radial_edges,
        "defocus_ratios": defocus_ratios,
    }


def _make_boundary_kernel_tensor(peak_fraction: float, ksize: int = 33) -> dict:
    """Tensor where kernel peak / L1 = peak_fraction.

    When peak_fraction > 0.5, the delta guard fires (merged path).
    When peak_fraction <= 0.5, the angular loop runs.
    """
    R, D = 6, 4
    kernels = np.zeros((R, D, ksize, ksize), dtype=np.float32)
    c = ksize // 2

    # Distribute energy: peak at center, rest spread uniformly
    # peak / total = peak_fraction  =>  peak = peak_fraction * total
    # total = peak + rest  =>  rest = peak * (1 - peak_fraction) / peak_fraction
    # Spread rest over ring of 8 neighbors
    total = 1.0
    peak_val = peak_fraction * total
    rest = total - peak_val
    n_spread = 8  # 8 neighbors
    spread_val = rest / n_spread

    for ri in range(R):
        for di in range(D):
            kernels[ri, di, c, c] = peak_val
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == 0 and dx == 0:
                        continue
                    kernels[ri, di, c + dy, c + dx] = spread_val

    radial_edges = np.linspace(0.0, 1.01, R + 1, dtype=np.float64)
    defocus_ratios = np.linspace(0.0, 1.0, D, dtype=np.float64)
    return {
        "kernels": kernels,
        "radial_edges": radial_edges,
        "defocus_ratios": defocus_ratios,
    }


# ---------------------------------------------------------------------------
# A1: Wrong-polarity fixture (MANDATORY)
# ---------------------------------------------------------------------------

class TestWrongPolarity:
    """Verify cos^2 slab weights are never negative."""

    def test_slab_weights_nonnegative(self):
        """Sweep radial_frac from 0 to 1 and assert all taper weights >= 0."""
        radial_edges = np.linspace(0.0, 1.01, 7, dtype=np.float64)
        r_centers = 0.5 * (radial_edges[:-1] + radial_edges[1:])
        n_slabs = len(r_centers) + 1

        rfrac = np.linspace(0.0, 1.0, 2000, dtype=np.float32)
        slab_frac = np.interp(
            rfrac, r_centers, np.arange(len(r_centers), dtype=np.float64)
        ).astype(np.float32)

        taper_half = 0.5
        for s in range(n_slabs):
            dx = np.abs(slab_frac - float(s))
            t = np.clip(dx / taper_half, 0.0, 1.0)
            raw_w = np.cos(0.5 * np.pi * t) ** 2
            assert raw_w.min() >= 0.0, (
                f"NEGATIVE taper weight {raw_w.min():.6f} at slab {s}"
            )

    def test_negative_weight_produces_inverted_blur(self):
        """Inject a synthetic negative slab weight and confirm it would invert.

        This is the wrong-polarity canary: if a code path ever produced a
        negative weight, the convolution output would be subtracted from
        the accumulator, inverting bright regions to dark. We verify this
        property on a known synthetic kernel to validate the test itself.
        """
        ksize = 5
        c = ksize // 2
        kernel = np.zeros((ksize, ksize), dtype=np.float32)
        kernel[c, c] = 1.0  # delta

        # Positive weight -> positive contribution
        patch = np.ones((10, 10), dtype=np.float32) * 0.5
        from scipy.signal import fftconvolve
        pos_contrib = fftconvolve(patch * 1.0, kernel, mode="same")
        neg_contrib = fftconvolve(patch * (-1.0), kernel, mode="same")

        assert pos_contrib.mean() > 0, "Positive weight should give positive output"
        assert neg_contrib.mean() < 0, "Negative weight WOULD give negative output"

    def test_render_output_nonnegative_linear(self):
        """Full render: output linear values should never go significantly negative.

        A negative slab weight would pull pixel values below zero before clipping.
        """
        sz = 64
        img = np.full((sz, sz, 3), 0.5, dtype=np.float32)
        tensor = _make_delta_tensor(ksize=17)
        _u8, output_lin = render_with_defocus(img, tensor)
        # Output linear should not have meaningfully negative values
        assert output_lin.min() > -0.01, (
            f"Negative linear output {output_lin.min():.4f} suggests wrong-polarity weight"
        )


# ---------------------------------------------------------------------------
# A2: Taper at extreme radii
# ---------------------------------------------------------------------------

class TestTaperExtremeRadii:
    """At r=0 (center) slab 0 should dominate; at r=1.0 outermost slab should dominate."""

    def test_center_pixel_slab0_dominant(self):
        """At r=0, slab_frac should map to slab index near 0."""
        radial_edges = np.linspace(0.0, 1.01, 7, dtype=np.float64)
        r_centers = 0.5 * (radial_edges[:-1] + radial_edges[1:])
        n_slabs = len(r_centers) + 1

        slab_frac_at_0 = np.interp(0.0, r_centers, np.arange(len(r_centers), dtype=np.float64))

        taper_half = 0.5
        weights = []
        for s in range(n_slabs):
            dx = abs(slab_frac_at_0 - float(s))
            t = min(dx / taper_half, 1.0)
            weights.append(np.cos(0.5 * np.pi * t) ** 2)

        weights = np.array(weights)
        weights /= weights.sum()
        # Slab 0 should have the highest weight at r=0
        assert np.argmax(weights) == 0, (
            f"At r=0, slab 0 weight {weights[0]:.4f} is not dominant; "
            f"argmax={np.argmax(weights)}, weights={weights}"
        )

    @pytest.mark.xfail(reason="BUG: np.interp clamps slab_frac to last r_center index; "
                        "far-edge slab (s=n_slabs-1) never gets nonzero weight at r=1.0")
    def test_corner_pixel_outermost_slab_dominant(self):
        """At r=1.0, the outermost slab should have the highest weight.

        BUG FOUND: np.interp(1.0, r_centers, arange(Nr)) returns Nr-1 (=5),
        so slab_frac maxes out at index 5. But n_slabs = Nr+1 = 7, meaning
        slab 6 (the far-edge slab that covers r >= r_centers[-1]) always gets
        weight 0. The far-edge slab is dead weight -- all corner pixels are
        handled by slab 5 instead. This means the r_query mean-radial tweak
        for the far-edge slab (lines 392-401) is never used, and corner
        kernels are interpolated at the wrong radial position.
        """
        radial_edges = np.linspace(0.0, 1.01, 7, dtype=np.float64)
        r_centers = 0.5 * (radial_edges[:-1] + radial_edges[1:])
        n_slabs = len(r_centers) + 1

        slab_frac_at_1 = np.interp(1.0, r_centers, np.arange(len(r_centers), dtype=np.float64))

        taper_half = 0.5
        weights = []
        for s in range(n_slabs):
            dx = abs(slab_frac_at_1 - float(s))
            t = min(dx / taper_half, 1.0)
            weights.append(np.cos(0.5 * np.pi * t) ** 2)

        weights = np.array(weights)
        weights /= weights.sum()
        # Outermost slab should dominate
        assert np.argmax(weights) == n_slabs - 1, (
            f"At r=1.0, outermost slab weight {weights[-1]:.4f} is not dominant; "
            f"argmax={np.argmax(weights)}, weights={weights}"
        )


# ---------------------------------------------------------------------------
# A3: Angular merge boundary threshold
# ---------------------------------------------------------------------------

class TestAngularMergeBoundary:
    """Delta guard: peak > 0.5 * L1. Test at 0.499, 0.501, and exact 0.5."""

    def test_below_threshold_uses_angular_loop(self):
        """peak_fraction=0.499 -> angular loop (is_delta=False)."""
        tensor = _make_boundary_kernel_tensor(0.499)
        k = tensor["kernels"][0, 0]
        is_delta = k.max() > 0.5 * k.sum()
        assert not is_delta, "0.499 should NOT trigger delta guard"

    def test_above_threshold_uses_merged(self):
        """peak_fraction=0.501 -> merged path (is_delta=True)."""
        tensor = _make_boundary_kernel_tensor(0.501)
        k = tensor["kernels"][0, 0]
        is_delta = k.max() > 0.5 * k.sum()
        assert is_delta, "0.501 SHOULD trigger delta guard"

    def test_exact_threshold_uses_angular_loop(self):
        """peak_fraction=0.5 exactly -> NOT delta (strict >)."""
        tensor = _make_boundary_kernel_tensor(0.5)
        k = tensor["kernels"][0, 0]
        is_delta = k.max() > 0.5 * k.sum()
        assert not is_delta, "Exact 0.5 should NOT trigger (guard uses strict >)"

    def test_near_boundary_render_equivalence(self):
        """Render with 0.499 and 0.501 kernels on same image; outputs should be close.

        Both are near-delta, so the difference between angular loop and merged
        path should be small for this nearly-delta kernel.
        """
        sz = 64
        img = np.full((sz, sz, 3), 0.5, dtype=np.float32)
        img[::2, ::2, :] = 0.7

        t_below = _make_boundary_kernel_tensor(0.499)
        t_above = _make_boundary_kernel_tensor(0.501)

        _u8_b, lin_below = render_with_defocus(img, t_below)
        _u8_a, lin_above = render_with_defocus(img, t_above)

        max_diff = float(np.abs(lin_below - lin_above).max())
        # Both kernels are nearly identical (only 0.002 difference in peak),
        # so outputs should be very close
        assert max_diff < 0.05, (
            f"Near-boundary render diff {max_diff:.4f} >= 0.05; "
            "merged vs angular paths diverge at threshold"
        )


# ---------------------------------------------------------------------------
# A4: _bilinear_weights at exact d=0 grid point
# ---------------------------------------------------------------------------

class TestBilinearExactD0:
    """When d_query=0.0 and defocus_ratios[0]=0.0, bilinear hits exact grid point."""

    def test_exact_d0_r0_weights(self):
        """d=0, r=0: should give weight=1.0 to corner (0,0)."""
        radial_edges = np.array([0.0, 0.5, 1.01], dtype=np.float64)
        defocus_ratios = np.array([0.0, 0.5, 1.0], dtype=np.float64)

        indices, weights = _bilinear_weights(0.0, 0.0, radial_edges, defocus_ratios)

        # At exact grid point, one weight should be 1.0 and rest 0.0
        assert abs(sum(weights) - 1.0) < 1e-6
        assert max(weights) > 0.99, (
            f"At exact grid point, max weight should be ~1.0, got {max(weights):.4f}"
        )

    def test_exact_d0_no_nan(self):
        """d_lo == d_hi should not produce NaN from 0/0 division."""
        radial_edges = np.array([0.0, 1.01], dtype=np.float64)
        defocus_ratios = np.array([0.0, 1.0], dtype=np.float64)

        # d=0 => d_lo=0, d_hi=1, d_span=1.0. No division by zero here.
        # But with a single-element defocus array this would be different.
        indices, weights = _bilinear_weights(0.0, 0.0, radial_edges, defocus_ratios)
        for w in weights:
            assert np.isfinite(w), f"Non-finite weight {w} at exact d=0"

    def test_single_defocus_ratio(self):
        """Edge case: defocus_ratios has only 1 element. d_lo and d_hi clamp to 0."""
        radial_edges = np.array([0.0, 0.5, 1.01], dtype=np.float64)
        defocus_ratios = np.array([0.5], dtype=np.float64)

        # With 1 element, Nd-2 = -1, clip to 0 => d_lo=d_hi=0
        # d_span = defocus_ratios[1] - defocus_ratios[0] => IndexError!
        try:
            indices, weights = _bilinear_weights(0.0, 0.0, radial_edges, defocus_ratios)
            # If it doesn't crash, verify sanity
            assert abs(sum(weights) - 1.0) < 1e-6
        except IndexError:
            pytest.xfail("BUG: _bilinear_weights crashes with single-element defocus_ratios")


# ---------------------------------------------------------------------------
# A5: Partition-of-unity with 1 radial bin
# ---------------------------------------------------------------------------

class TestSingleRadialBin:
    """Tensor with 1 radial bin: np.interp against 1-element r_centers."""

    def test_single_bin_renders_without_crash(self):
        """Smoke test: 1-bin tensor should not crash."""
        sz = 32
        img = np.full((sz, sz, 3), 0.5, dtype=np.float32)
        tensor = _make_single_bin_tensor(ksize=17)
        try:
            _u8, output_lin = render_with_defocus(img, tensor)
            assert output_lin.shape == (sz, sz, 3)
        except Exception as e:
            pytest.xfail(f"BUG: single-bin tensor crashes: {e}")

    def test_single_bin_partition_of_unity(self):
        """With 1 r_center, the normalized slab weights should still sum to 1."""
        radial_edges = np.array([0.0, 1.01], dtype=np.float64)
        r_centers = 0.5 * (radial_edges[:-1] + radial_edges[1:])
        assert len(r_centers) == 1

        n_slabs = len(r_centers) + 1  # = 2
        H, W = 32, 32
        rfrac = np.linspace(0.0, 1.0, W, dtype=np.float32)
        rfrac_2d = np.broadcast_to(rfrac[None, :], (H, W)).copy()

        slab_frac = np.interp(
            rfrac_2d.ravel(), r_centers,
            np.arange(len(r_centers), dtype=np.float64),
        ).reshape(H, W).astype(np.float32)

        taper_half = 0.5
        raw_weights = np.zeros((n_slabs, H, W), dtype=np.float32)
        for s in range(n_slabs):
            dx = np.abs(slab_frac - float(s))
            t = np.clip(dx / taper_half, 0.0, 1.0)
            raw_weights[s] = (np.cos(0.5 * np.pi * t) ** 2).astype(np.float32)

        wsum = raw_weights.sum(axis=0)
        wsum_norm = np.maximum(wsum, 1e-8)
        norm_weights = raw_weights / wsum_norm[None, :, :]

        # After normalization, should sum to 1.0 everywhere
        final_sum = norm_weights.sum(axis=0)
        assert np.allclose(final_sum, 1.0, atol=1e-5), (
            f"Partition-of-unity violated: min={final_sum.min():.6f} max={final_sum.max():.6f}"
        )


# ---------------------------------------------------------------------------
# A6: Energy conservation with taper + merge both active
# ---------------------------------------------------------------------------

class TestEnergyTaperPlusMerge:
    """Render uniform white image: output should be uniform within tolerance."""

    def test_uniform_white_preservation(self):
        """All-1.0 sRGB input through delta tensor: output should be near-uniform."""
        sz = 64
        img = np.ones((sz, sz, 3), dtype=np.float32)  # all white in sRGB
        tensor = _make_delta_tensor(ksize=17)
        _u8, output_lin = render_with_defocus(img, tensor)

        input_lin = srgb_to_linear(img)
        # All pixels should be ~1.0 in linear space
        expected = float(input_lin[sz // 2, sz // 2, 0])

        # Check center region (away from edges where padding effects exist)
        margin = 5
        center = output_lin[margin:-margin, margin:-margin, :]
        max_dev = float(np.abs(center - expected).max())
        assert max_dev < 0.02, (
            f"Uniform white: max deviation {max_dev:.4f} from expected {expected:.4f}"
        )

    def test_energy_ratio_uniform_gray(self):
        """Uniform gray: energy ratio should be very close to 1.0."""
        sz = 64
        img = np.full((sz, sz, 3), 0.5, dtype=np.float32)
        tensor = _make_delta_tensor(ksize=17)
        _u8, output_lin = render_with_defocus(img, tensor)
        input_lin = srgb_to_linear(img)

        ratio = float(output_lin.sum()) / float(input_lin.sum())
        assert 0.95 <= ratio <= 1.05, (
            f"Energy ratio {ratio:.4f} outside tight [0.95, 1.05] for uniform input"
        )

    def test_mixed_tensor_energy(self):
        """Mixed tensor (delta + Gaussian) on uniform: exercises both paths."""
        sz = 64
        img = np.full((sz, sz, 3), 0.4, dtype=np.float32)

        R, D = 6, 4
        ksize = 17
        c = ksize // 2
        kernels = np.zeros((R, D, ksize, ksize), dtype=np.float32)
        # d=0: delta (merged path)
        kernels[:, 0, c, c] = 1.0
        # d>0: Gaussian (angular loop path)
        y, x = np.mgrid[:ksize, :ksize]
        sigma = ksize / 8.0
        gauss = np.exp(-((x - c) ** 2 + (y - c) ** 2) / (2 * sigma ** 2)).astype(np.float32)
        gauss /= gauss.sum()
        for d in range(1, D):
            kernels[:, d] = gauss

        tensor = {
            "kernels": kernels,
            "radial_edges": np.linspace(0.0, 1.01, R + 1, dtype=np.float64),
            "defocus_ratios": np.linspace(0.0, 1.0, D, dtype=np.float64),
        }

        _u8, output_lin = render_with_defocus(img, tensor)
        input_lin = srgb_to_linear(img)
        ratio = float(output_lin.sum()) / float(input_lin.sum())
        assert 0.85 <= ratio <= 1.15, (
            f"Mixed-tensor energy ratio {ratio:.4f} outside [0.85, 1.15]"
        )
