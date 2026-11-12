"""Adversarial tests for P1.2 distance-indexed renderer."""
from __future__ import annotations

import numpy as np
import pytest

from summicron.render.convolve import load_tensor, render, srgb_to_linear


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _gaussian_kernel(ksize: int, sigma: float) -> np.ndarray:
    """Normalized 2D Gaussian kernel."""
    ax = np.arange(ksize, dtype=np.float32) - ksize // 2
    xx, yy = np.meshgrid(ax, ax)
    k = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    return (k / k.sum()).astype(np.float32)


def _make_distance_tensor(ksize: int = 33) -> dict:
    """Synthetic tensor: 6 radial bins, 5 defocus levels.

    Defocus level 0 is a delta (in-focus), higher levels get progressively
    blurrier Gaussians so that depth-dependent blur is actually visible.
    """
    R, D = 6, 5
    kernels = np.zeros((R, D, ksize, ksize), dtype=np.float32)
    c = ksize // 2
    sigmas = [0.0, 1.0, 2.0, 4.0, 6.0]  # increasing blur with defocus
    for d_idx, sigma in enumerate(sigmas):
        if sigma < 0.1:
            kern = np.zeros((ksize, ksize), dtype=np.float32)
            kern[c, c] = 1.0
        else:
            kern = _gaussian_kernel(ksize, sigma)
        for r_idx in range(R):
            kernels[r_idx, d_idx] = kern
    radial_edges = np.array([0.0, 0.15, 0.30, 0.45, 0.60, 0.80, 1.01], dtype=np.float64)
    defocus_ratios = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float64)
    object_distances_m = np.array([0.7, 1.0, 3.0, 10.0, 20.0], dtype=np.float64)
    focus_distance_m = 0.7
    return {
        "kernels": kernels,
        "radial_edges": radial_edges,
        "defocus_ratios": defocus_ratios,
        "object_distances_m": object_distances_m,
        "focus_distance_m": focus_distance_m,
        "corpus_ref_long_edge": 256,
        "otf_norm_applied": True,
    }


def _flat_image(sz: int = 64, val: float = 0.5) -> np.ndarray:
    return np.full((sz, sz, 3), val, dtype=np.float32)


# ---------------------------------------------------------------------------
# A1: Wrong-polarity depth (MANDATORY)
# ---------------------------------------------------------------------------

class TestWrongPolarity:
    """Inverted depth map: near objects get far distances and vice versa."""

    def test_inverted_depth_produces_inverted_blur(self):
        """Near=0.8m should blur more than far=20m, but with inverted depth_m
        the mapping sends 0.8m->low defocus and 20m->high defocus.
        Verify the renderer faithfully follows the depth_m values (no
        automatic inversion detection)."""
        sz = 64
        np.random.seed(99)
        img = np.random.rand(sz, sz, 3).astype(np.float32) * 0.8 + 0.1
        tensor = _make_distance_tensor()

        # Normal depth: top=near(0.8), bottom=far(20)
        depth_normal = np.zeros((sz, sz), dtype=np.float32)
        depth_normal[:sz // 2, :] = 0.8
        depth_normal[sz // 2:, :] = 20.0

        # Inverted depth: top=far(20), bottom=near(0.8)
        depth_inverted = np.zeros((sz, sz), dtype=np.float32)
        depth_inverted[:sz // 2, :] = 20.0
        depth_inverted[sz // 2:, :] = 0.8

        _, out_normal = render(img, tensor, depth_m=depth_normal)
        _, out_inverted = render(img, tensor, depth_m=depth_inverted)

        # In normal: top half gets low defocus, bottom gets high
        # In inverted: top half gets HIGH defocus, bottom gets LOW
        # So the two outputs must differ — renderer follows the map blindly
        assert not np.allclose(out_normal, out_inverted, atol=1e-4), (
            "Normal and inverted depth produced identical output — mapping is broken"
        )

        # Top half of inverted should match bottom half of normal (same defocus)
        # This is approximate due to radial position effects
        # Just verify the blur direction flipped by checking variance
        input_lin = srgb_to_linear(img)
        diff_normal_top = np.abs(out_normal[:sz // 4, sz // 4:3 * sz // 4] -
                                  input_lin[:sz // 4, sz // 4:3 * sz // 4]).mean()
        diff_normal_bot = np.abs(out_normal[3 * sz // 4:, sz // 4:3 * sz // 4] -
                                  input_lin[3 * sz // 4:, sz // 4:3 * sz // 4]).mean()
        diff_inv_top = np.abs(out_inverted[:sz // 4, sz // 4:3 * sz // 4] -
                               input_lin[:sz // 4, sz // 4:3 * sz // 4]).mean()
        diff_inv_bot = np.abs(out_inverted[3 * sz // 4:, sz // 4:3 * sz // 4] -
                               input_lin[3 * sz // 4:, sz // 4:3 * sz // 4]).mean()

        # Normal: top (near) less deviation, bottom (far) more
        # Inverted: top (far mapped) more deviation, bottom (near mapped) less
        # We just confirm the polarity flipped
        normal_gradient = diff_normal_bot - diff_normal_top
        inverted_gradient = diff_inv_bot - diff_inv_top
        assert normal_gradient * inverted_gradient < 0 or (
            abs(normal_gradient) < 1e-5 and abs(inverted_gradient) < 1e-5
        ), "Blur gradient did not flip with inverted depth"


# ---------------------------------------------------------------------------
# A2: NaN in depth_m
# ---------------------------------------------------------------------------

class TestNaNDepth:
    """np.interp propagates NaN — renderer should not silently produce garbage."""

    # HYG-0 item 15: xfail removed — XPASSed (NaN depth now raises/cleans).
    def test_nan_depth_raises_or_clean(self):
        """depth_m with NaN should either raise ValueError or produce no NaN output."""
        sz = 32
        img = _flat_image(sz, 0.5)
        tensor = _make_distance_tensor()
        depth_m = np.full((sz, sz), 5.0, dtype=np.float32)
        depth_m[10, 10] = np.nan

        # Should either raise or produce clean output
        try:
            out_u8, out_lin = render(img, tensor, depth_m=depth_m)
            # If it didn't raise, output must be NaN-free
            assert not np.any(np.isnan(out_lin)), "NaN leaked into output"
            assert not np.any(np.isnan(out_u8.astype(float))), "NaN in uint8"
        except (ValueError, RuntimeError):
            pass  # Acceptable: explicit error


# ---------------------------------------------------------------------------
# A3: Negative depth
# ---------------------------------------------------------------------------

class TestNegativeDepth:
    """depth_m = -5.0 is physically meaningless; np.interp clamps to first value."""

    def test_negative_depth_no_crash(self):
        sz = 32
        img = _flat_image(sz, 0.5)
        tensor = _make_distance_tensor()
        depth_m = np.full((sz, sz), -5.0, dtype=np.float32)
        out_u8, out_lin = render(img, tensor, depth_m=depth_m)
        assert out_u8.shape == (sz, sz, 3)
        assert not np.any(np.isnan(out_lin))

    def test_negative_depth_clamps_to_focus(self):
        """Negative depth should map to defocus_ratios[0]=0.0, same as focus distance."""
        sz = 64
        img = _flat_image(sz, 0.5)
        tensor = _make_distance_tensor()

        depth_neg = np.full((sz, sz), -5.0, dtype=np.float32)
        depth_focus = np.full((sz, sz), 0.7, dtype=np.float32)  # focus_distance_m

        out_neg_u8, _ = render(img, tensor, depth_m=depth_neg)
        out_focus_u8, _ = render(img, tensor, depth_m=depth_focus)
        np.testing.assert_array_equal(out_neg_u8, out_focus_u8)


# ---------------------------------------------------------------------------
# A4: Zero depth
# ---------------------------------------------------------------------------

class TestZeroDepth:
    """depth_m = 0.0 — closer than focus, maps to defocus_ratios[0]."""

    def test_zero_depth_same_as_focus(self):
        sz = 64
        img = _flat_image(sz, 0.5)
        tensor = _make_distance_tensor()

        depth_zero = np.full((sz, sz), 0.0, dtype=np.float32)
        depth_focus = np.full((sz, sz), 0.7, dtype=np.float32)

        out_zero_u8, _ = render(img, tensor, depth_m=depth_zero)
        out_focus_u8, _ = render(img, tensor, depth_m=depth_focus)
        np.testing.assert_array_equal(out_zero_u8, out_focus_u8)


# ---------------------------------------------------------------------------
# A5: Exact tensor distances — zero interpolation error
# ---------------------------------------------------------------------------

class TestExactTensorDistances:
    """depth_m matching object_distances_m exactly should give exact defocus_ratios."""

    def test_exact_distances_exact_ratios(self):
        """np.interp at exact xp values should return exact fp values."""
        tensor = _make_distance_tensor()
        od = tensor["object_distances_m"]
        dr = tensor["defocus_ratios"]

        # Scalar interp at each exact distance
        for dist, expected_ratio in zip(od, dr):
            result = np.interp(dist, od, dr)
            assert abs(result - expected_ratio) < 1e-12, (
                f"np.interp({dist}, ...) = {result}, expected {expected_ratio}"
            )

    def test_exact_distance_image(self):
        """Full render at each exact tensor distance produces no NaN."""
        sz = 32
        img = _flat_image(sz, 0.5)
        tensor = _make_distance_tensor()
        for d in tensor["object_distances_m"]:
            depth_m = np.full((sz, sz), float(d), dtype=np.float32)
            out_u8, out_lin = render(img, tensor, depth_m=depth_m)
            assert not np.any(np.isnan(out_lin))


# ---------------------------------------------------------------------------
# A6: depth_m renders deterministically (defocus_scale removed from render())
# ---------------------------------------------------------------------------

class TestDepthDeterministic:
    """depth_m with same inputs produces identical output."""

    def test_deterministic(self):
        sz = 32
        img = _flat_image(sz, 0.5)
        tensor = _make_distance_tensor()
        depth_m = np.full((sz, sz), 5.0, dtype=np.float32)

        out1_u8, _ = render(img, tensor, depth_m=depth_m)
        out2_u8, _ = render(img, tensor, depth_m=depth_m)
        np.testing.assert_array_equal(out1_u8, out2_u8)


# ---------------------------------------------------------------------------
# A7: Full-resolution smoke test (memory)
# ---------------------------------------------------------------------------

class TestFullResMemory:
    """5712x4284 depth_m allocation smoke test."""

    @pytest.mark.slow
    def test_full_res_allocation(self):
        """Just verify we can allocate and interp a full-res depth_m."""
        H, W = 4284, 5712
        depth_m = np.full((H, W), 5.0, dtype=np.float32)
        tensor = _make_distance_tensor()
        od = tensor["object_distances_m"]
        dr = tensor["defocus_ratios"]
        dmap = np.interp(depth_m, od, dr).astype(np.float32)
        assert dmap.shape == (H, W)
        assert not np.any(np.isnan(dmap))


# ---------------------------------------------------------------------------
# A8: Non-monotonic object_distances_m in tensor file
# ---------------------------------------------------------------------------

class TestNonMonotonicTensor:
    """Craft tensor with non-monotonic distances; load_tensor must reject."""

    def test_non_monotonic_reversed(self, tmp_path):
        ksize = 17
        R, D = 4, 3
        kernels = np.zeros((R, D, ksize, ksize), dtype=np.float32)
        c = ksize // 2
        kernels[:, :, c, c] = 1.0
        npz_path = tmp_path / "reversed.npz"
        np.savez(
            str(npz_path),
            kernels=kernels,
            radial_edges=np.array([0.0, 0.3, 0.6, 0.9, 1.01]),
            defocus_ratios=np.array([0.0, 0.5, 1.0]),
            object_distances_m=np.array([20.0, 3.0, 0.7]),  # reversed
            focus_distance_m=np.float64(20.0),
            otf_norm_applied=True,
        )
        # P1.9.4 L11: was AssertionError, now ValueError (uniform validation type).
        with pytest.raises(ValueError, match="monotonically increasing"):
            load_tensor(npz_path)

    def test_non_monotonic_duplicate(self, tmp_path):
        """Duplicate values (non-strictly-increasing) should also fail."""
        ksize = 17
        R, D = 4, 3
        kernels = np.zeros((R, D, ksize, ksize), dtype=np.float32)
        c = ksize // 2
        kernels[:, :, c, c] = 1.0
        npz_path = tmp_path / "duplicate.npz"
        np.savez(
            str(npz_path),
            kernels=kernels,
            radial_edges=np.array([0.0, 0.3, 0.6, 0.9, 1.01]),
            defocus_ratios=np.array([0.0, 0.5, 1.0]),
            object_distances_m=np.array([0.7, 3.0, 3.0]),  # duplicate!
            focus_distance_m=np.float64(0.7),
            otf_norm_applied=True,
        )
        # P1.9.4 L11: was AssertionError, now ValueError (uniform validation type).
        with pytest.raises(ValueError, match="monotonically increasing"):
            load_tensor(npz_path)

    def test_single_element_no_crash(self, tmp_path):
        """Single-element object_distances_m — np.diff returns empty, should pass."""
        ksize = 17
        R, D = 4, 1
        kernels = np.zeros((R, D, ksize, ksize), dtype=np.float32)
        c = ksize // 2
        kernels[:, :, c, c] = 1.0
        npz_path = tmp_path / "single.npz"
        np.savez(
            str(npz_path),
            kernels=kernels,
            radial_edges=np.array([0.0, 0.3, 0.6, 0.9, 1.01]),
            defocus_ratios=np.array([0.0]),
            object_distances_m=np.array([0.7]),
            focus_distance_m=np.float64(0.7),
            otf_norm_applied=True,
        )
        t = load_tensor(npz_path)
        assert len(t["object_distances_m"]) == 1


# ---------------------------------------------------------------------------
# A9: Inf in depth_m
# ---------------------------------------------------------------------------

class TestInfDepth:
    """np.inf in depth_m — np.interp clamps to last value, should produce max blur."""

    def test_inf_no_crash(self):
        sz = 32
        img = _flat_image(sz, 0.5)
        tensor = _make_distance_tensor()
        depth_m = np.full((sz, sz), np.inf, dtype=np.float32)
        out_u8, out_lin = render(img, tensor, depth_m=depth_m)
        assert out_u8.shape == (sz, sz, 3)
        assert not np.any(np.isnan(out_lin))

    def test_inf_same_as_max_distance(self):
        """Inf should clamp to same result as max object distance (20m)."""
        sz = 32
        img = _flat_image(sz, 0.5)
        tensor = _make_distance_tensor()

        depth_inf = np.full((sz, sz), np.inf, dtype=np.float32)
        depth_max = np.full((sz, sz), 20.0, dtype=np.float32)

        out_inf_u8, _ = render(img, tensor, depth_m=depth_inf)
        out_max_u8, _ = render(img, tensor, depth_m=depth_max)
        np.testing.assert_array_equal(out_inf_u8, out_max_u8)
