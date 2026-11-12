"""P1.10.1 — adversarial chromaticity gate (G10).

Per `feedback_adversarial_chromaticity_gate.md` and locked-spec § S6 item 10:
warm + cool saturated HDR fixtures must preserve achromaticity (per-channel
ratio drift ≤ 1%) through `--radial-blend-mode basis`. Achromaticity here
means: a uniform-color input field, after blur, retains its color ratio in
the output (no per-channel drift introduced by the new path).

Fixtures:
- warm: R/G/B = 1.5/0.6/0.1, lum 0.85 (orange highlight).
- cool: R/G/B = 0.3/0.6/1.5, lum 0.71 (teal highlight).
- HDR overflow safety: lum=8.0 half-frame.
"""
from __future__ import annotations

import numpy as np
import pytest

from summicron.render.convolve import render
from tests.fixtures.defocus import render_with_defocus



def _make_gaussian_tensor(R=4, D=4, ksize=33, sigma=4.0):
    cy, cx = ksize // 2, ksize // 2
    yy, xx = np.mgrid[:ksize, :ksize]
    g = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * sigma * sigma))
    g = g.astype(np.float32) / g.sum()
    kernels = np.tile(g[None, None, :, :], (R, D, 1, 1))
    delta = np.zeros((ksize, ksize), dtype=np.float32)
    delta[cy, cx] = 1.0
    kernels[:, 0] = delta
    return {
        "kernels": kernels,
        "radial_edges": np.linspace(0.0, 1.0, R + 1).astype(np.float64),
        "defocus_ratios": np.linspace(0.0, 1.0, D).astype(np.float64),
        "corpus_ref_long_edge": None,
        "otf_norm_applied": False,
        "delta_guard_applied": True,
        "object_distances_m": None,
        "focus_distance_m": None,
    }


def _channel_ratio(img: np.ndarray) -> tuple[float, float]:
    """Return mean R/G and B/G ratios of (H, W, 3) float image (linear)."""
    means = img.reshape(-1, 3).mean(axis=0)
    g = float(means[1]) + 1e-12
    return float(means[0]) / g, float(means[2]) / g


# ---------------------------------------------------------------------------
# Warm + cool HDR fixtures
# ---------------------------------------------------------------------------

class TestChromaticityWarmCool:
    def _run_basis_render(self, color_rgb, lum, defocus_level=0.6):
        """Build a uniform colored field at given lum, render with basis mode."""
        H, W = 64, 64
        # Construct fixture: full-frame uniform color (already linear, HDR ok)
        img = np.zeros((H, W, 3), dtype=np.float32)
        img[..., :] = np.array(color_rgb, dtype=np.float32) * lum
        defocus = np.full((H, W), float(defocus_level), dtype=np.float32)
        tensor = _make_gaussian_tensor()
        # HDR path skips srgb_to_linear, accepts > 1.0
        _, out_lin = render_with_defocus(
            img, tensor, dmap=defocus, hdr_source="macos",
        )
        return img, out_lin

    def test_warm_chromaticity_preserved(self):
        """Warm orange: R/G/B = 1.5/0.6/0.1, lum 0.85."""
        img_in, out = self._run_basis_render([1.5, 0.6, 0.1], lum=0.85)
        rg_in, bg_in = _channel_ratio(img_in)
        rg_out, bg_out = _channel_ratio(out)
        # Allow ≤1% drift per spec G10.
        rg_drift = abs(rg_out - rg_in) / max(rg_in, 1e-6)
        bg_drift = abs(bg_out - bg_in) / max(bg_in, 1e-6)
        assert rg_drift <= 0.01, (
            f"Warm R/G drift {rg_drift:.4%} > 1% (in={rg_in:.4f}, out={rg_out:.4f})"
        )
        assert bg_drift <= 0.01, (
            f"Warm B/G drift {bg_drift:.4%} > 1% (in={bg_in:.4f}, out={bg_out:.4f})"
        )

    def test_cool_chromaticity_preserved(self):
        """Cool teal: R/G/B = 0.3/0.6/1.5, lum 0.71."""
        img_in, out = self._run_basis_render([0.3, 0.6, 1.5], lum=0.71)
        rg_in, bg_in = _channel_ratio(img_in)
        rg_out, bg_out = _channel_ratio(out)
        rg_drift = abs(rg_out - rg_in) / max(rg_in, 1e-6)
        bg_drift = abs(bg_out - bg_in) / max(bg_in, 1e-6)
        assert rg_drift <= 0.01, (
            f"Cool R/G drift {rg_drift:.4%} > 1% (in={rg_in:.4f}, out={rg_out:.4f})"
        )
        assert bg_drift <= 0.01, (
            f"Cool B/G drift {bg_drift:.4%} > 1% (in={bg_in:.4f}, out={bg_out:.4f})"
        )


# ---------------------------------------------------------------------------
# HDR overflow safety
# ---------------------------------------------------------------------------

class TestHDROverflowSafety:
    def test_lum_8_half_frame_no_nan(self):
        """Half-frame HDR fixture (lum=8.0 left, 0.0 right): basis path stays
        finite (no NaN/Inf propagation through the new helpers)."""
        H, W = 64, 64
        img = np.zeros((H, W, 3), dtype=np.float32)
        # Left half saturated white at lum=8.0 (HDR), right half black.
        img[:, : W // 2, :] = 8.0
        defocus = np.full((H, W), 0.6, dtype=np.float32)
        tensor = _make_gaussian_tensor()
        _, out_lin = render_with_defocus(
            img, tensor, dmap=defocus, hdr_source="macos",
        )
        assert np.isfinite(out_lin).all(), (
            "HDR overflow propagated NaN/Inf through basis path"
        )

    def test_high_dynamic_range_chromaticity(self):
        """HDR warm fixture at lum=4.0 (well above SDR ceiling): chromaticity
        preserved through basis blend."""
        H, W = 64, 64
        img = np.zeros((H, W, 3), dtype=np.float32)
        img[..., :] = np.array([1.5, 0.6, 0.1], dtype=np.float32) * 4.0
        defocus = np.full((H, W), 0.6, dtype=np.float32)
        tensor = _make_gaussian_tensor()
        _, out = render_with_defocus(
            img, tensor, dmap=defocus, hdr_source="macos",
        )
        rg_in, bg_in = _channel_ratio(img)
        rg_out, bg_out = _channel_ratio(out)
        rg_drift = abs(rg_out - rg_in) / max(rg_in, 1e-6)
        bg_drift = abs(bg_out - bg_in) / max(bg_in, 1e-6)
        assert rg_drift <= 0.01, f"HDR warm R/G drift {rg_drift:.4%} > 1%"
        assert bg_drift <= 0.01, f"HDR warm B/G drift {bg_drift:.4%} > 1%"
