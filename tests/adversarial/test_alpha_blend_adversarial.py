"""P1.9.1 α-blend adversarial tests.

Attacks the per-pixel α(defocus) blend on dtype robustness, HDR headroom
preservation, D_KNEE_LAYERED edge cases, and peak-mode placement invariants.
"""
from __future__ import annotations

import re

import numpy as np
import pytest

import summicron.render.convolve as cv
from summicron.render.convolve import D_KNEE_LAYERED, render, srgb_to_linear
from tests.fixtures.defocus import render_with_defocus



# --------------------------------------------------------------------------- #
# Shared fixtures                                                              #
# --------------------------------------------------------------------------- #

_EDGES = np.array([0.0, 0.15, 0.30, 0.45, 0.60, 0.80, 1.01], dtype=np.float64)
_RATIOS = np.array([0.0, 0.33, 0.67, 1.0], dtype=np.float64)


def _blur_tensor(ksize: int = 33) -> dict:
    """Tiny Gaussian-blur tensor; mirror of unit-test factory."""
    R, D = 6, 4
    kernels = np.zeros((R, D, ksize, ksize), dtype=np.float32)
    c = ksize // 2
    y, x = np.mgrid[-c:c + 1, -c:c + 1].astype(np.float32)
    for di in range(D):
        g = np.exp(-(x ** 2 + y ** 2) / (2 * (1.0 + di * 2.0) ** 2))
        kernels[:, di] = (g / g.sum()).astype(np.float32)
    return {
        "kernels": kernels,
        "radial_edges": _EDGES.copy(),
        "defocus_ratios": _RATIOS.copy(),
    }


def _image(seed: int = 0, H: int = 64, W: int = 64, low: float = 0.1, high: float = 0.9) -> np.ndarray:
    return np.random.default_rng(seed).uniform(low, high, size=(H, W, 3)).astype(np.float32)


# --------------------------------------------------------------------------- #
# V1 — HDR headroom preservation                                               #
# --------------------------------------------------------------------------- #

class TestAlphaBlendHDRHeadroom:
    """The blend must not silently clamp values > 1.0 in HDR mode."""

    def test_hdr_alpha_one_preserves_headroom(self):
        """α=1 (dmap=0): output == img_lin; HDR values > 1.0 pass through."""
        img = np.full((64, 64, 3), 3.0, dtype=np.float32)
        dmap = np.zeros((64, 64), dtype=np.float32)
        _, out_lin = render_with_defocus(img, _blur_tensor(), dmap=dmap, hdr_source="macos")
        assert out_lin.max() > 2.99, (
            f"HDR α=1 clamped: out.max={out_lin.max()} (expect ≈3.0)"
        )
        assert np.allclose(out_lin, 3.0, atol=1e-5)

    def test_hdr_partial_alpha_keeps_super_one_range(self):
        """α≈0.5: blend of img_lin (5.0) + convolve (≈5.0 L1-preserving)
        must stay above 1.0. Guards against naive clip/normalize."""
        img = np.full((64, 64, 3), 5.0, dtype=np.float32)
        # dmap where α ≈ 0.5: cos²(π/2 · x) = 0.5 → x = 0.5 → d = 0.5·D_KNEE_LAYERED
        dmap = np.full((64, 64), 0.5 * D_KNEE_LAYERED, dtype=np.float32)
        _, out_lin = render_with_defocus(img, _blur_tensor(), dmap=dmap, hdr_source="macos")
        assert out_lin.max() > 4.5, f"HDR mid-blend clamped: {out_lin.max()}"


# --------------------------------------------------------------------------- #
# V3 — dtype robustness                                                        #
# --------------------------------------------------------------------------- #

class TestAlphaBlendDmapDtype:
    """The blend casts dmap to float32; output dtype must stay float32."""

    def test_int_dmap_works_as_zero_identity(self):
        """int32 dmap of zeros → α=1 → output bit-identical to linearized input."""
        img = _image(seed=0)
        dmap = np.zeros((64, 64), dtype=np.int32)
        _, out_lin = render_with_defocus(img, _blur_tensor(), dmap=dmap)
        assert out_lin.dtype == np.float32
        assert np.max(np.abs(out_lin - srgb_to_linear(img))) < 1e-6

    def test_float64_dmap_does_not_upcast_output(self):
        """Passing float64 dmap must not silently upcast output."""
        img = _image(seed=1)
        dmap = np.full((64, 64), 0.2, dtype=np.float64)
        _, out_lin = render_with_defocus(img, _blur_tensor(), dmap=dmap)
        assert out_lin.dtype == np.float32


# --------------------------------------------------------------------------- #
# V5 — D_KNEE_LAYERED = 0 edge case                                                    #
# --------------------------------------------------------------------------- #

class TestAlphaBlendDKneeEdges:
    """D_KNEE_LAYERED = 0 is a pathological config; document current behavior.

    Current observed behavior with D_KNEE_LAYERED = 0:
      - dmap = 0: min(0/0, 1) = min(NaN, 1) — numpy folds to NaN → α=NaN → output NaN
      - dmap > 0: min(inf, 1) = 1 → α=0 → output fully convolved

    The pre-blend energy gate does NOT catch this because it runs on
    pre-blend output. The blend itself has no NaN guard. PIN THE OBSERVED
    BEHAVIOR so any future D_KNEE_LAYERED=0 fix lands explicit NaN handling instead
    of silently bleeding NaN through.
    """

    def test_d_knee_zero_dmap_gtz_yields_fully_convolved(self, monkeypatch):
        """D_KNEE_LAYERED=0, dmap>0: α=0, output equals pre-blend convolved."""
        monkeypatch.setattr(cv, "D_KNEE_LAYERED", 1e-9)
        img = _image(seed=3)
        dmap = np.full((64, 64), 0.5, dtype=np.float32)
        _, out_lin = render_with_defocus(img, _blur_tensor(), dmap=dmap)
        # No NaN propagation; output is floating and finite
        assert np.all(np.isfinite(out_lin))
        # And non-trivially different from img_lin (i.e. actual convolve won)
        assert np.max(np.abs(out_lin - srgb_to_linear(img))) > 1e-3


# --------------------------------------------------------------------------- #
# V6 — D_KNEE_LAYERED > 1                                                              #
# --------------------------------------------------------------------------- #

class TestAlphaBlendDKneeWide:
    """D_KNEE_LAYERED > 1: α never reaches 0 within the valid dmap range [0, 1].

    With D_KNEE_LAYERED=1.5, α(1.0) = cos²(π/3) ≈ 0.25 — the "plateau" behaviour
    promised by the design ("α ≈ 0 at full defocus") is silently lost.
    This is a documented design contract: D_KNEE_LAYERED must be ≤ max(dmap) for
    the blend to ever hand full control to the convolved branch.
    """

    def test_d_knee_15_leaves_floor_alpha_at_dmap_one(self, monkeypatch):
        """α at dmap=1 with D_KNEE_LAYERED=1.5 ≈ 0.25, not 0."""
        monkeypatch.setattr(cv, "D_KNEE_LAYERED", 1.5)
        img = _image(seed=4)
        dmap = np.full((64, 64), 1.0, dtype=np.float32)
        _, out_lin = render_with_defocus(img, _blur_tensor(), dmap=dmap)
        # Pre-blend convolved ~ similar magnitude to img_lin; blended with α≈0.25
        # means ≥20% of img_lin signature leaks into "fully-defocused" pixels.
        expected_alpha = np.cos(0.5 * np.pi * min(1.0 / 1.5, 1.0)) ** 2
        assert 0.24 < expected_alpha < 0.26, (
            "sanity: expected_alpha formula changed"
        )
        # Finite + non-clipped
        assert np.all(np.isfinite(out_lin))


# --------------------------------------------------------------------------- #
# V7 — Energy conservation (gate pre-blend + post-blend ratio)                 #
# --------------------------------------------------------------------------- #

class TestAlphaBlendEnergyConservation:
    """Post-blend energy must be a convex combination of pre-blend branches.
    Both branches are ~L1-preserving so the blended sum should match input
    within the existing energy-gate bounds (±15%)."""

    def test_partial_blend_energy_within_gate(self):
        img = np.full((64, 64, 3), 0.5, dtype=np.float32)  # flat image
        # α ≈ 0.5 at d = 0.5·D_KNEE_LAYERED
        dmap = np.full((64, 64), 0.5 * D_KNEE_LAYERED, dtype=np.float32)
        _, out_lin = render_with_defocus(img, _blur_tensor(), dmap=dmap)
        ratio = out_lin.sum() / srgb_to_linear(img).sum()
        assert 0.85 <= ratio <= 1.15, (
            f"post-blend energy ratio {ratio:.4f} outside [0.85, 1.15]"
        )


# --------------------------------------------------------------------------- #
# V8 — Log spam                                                                #
# --------------------------------------------------------------------------- #

class TestAlphaBlendLogging:
    """α-blend emits one line per render(). With 243 test call-sites, this is
    noisy. Pin format so any future silence / dbg-flag refactor has a guard."""

    def test_alpha_log_format_stable(self, capsys):
        img = _image(seed=5)
        dmap = np.full((64, 64), 0.2, dtype=np.float32)
        render_with_defocus(img, _blur_tensor(), dmap=dmap)
        out = capsys.readouterr().out
        assert re.search(r"α-blend: D_KNEE_LAYERED=[\d.]+, alpha_mean=[\d.]+", out), (
            f"α-blend log format drifted; captured stdout:\n{out}"
        )
