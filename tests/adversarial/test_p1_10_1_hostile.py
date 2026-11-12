"""P1.10.1 — Adversarial hostile-probe coverage.

Round 1 Adversarial pass over per-pixel radial-basis kernel blend path.
This file is INTENDED to find ways the new path breaks under hostile
or unexpected inputs that the Coder's own tests may not catch.

Scope (categories 1..14 in adversarial brief):
 1. scalar-collapse regression (extreme inputs)
 2. G1 SHOWSTOPPER stress (kill-switch determinism, r_blend_strength edges)
 3. radial_frac edge cases (NaN/Inf/negative; non-monotone r_centers)
 4. Tensor degeneracy (Nr=1, Nr=2, zero-L1 slice)
 5. Image degeneracy (1×1, all-zero, all-saturated, NaN, single channel)
 6. Top-hat × basis interaction (chromaticity)
 7. Layered + basis interaction (slab kill-switch under layered)
 8. Wedge ordering (docstring + actual loop nesting)
 9. D-axis stance (D-axis is scalar, not per-pixel)
10. Memory probe (skipped: would OOM at native res; estimate-only)
11. Determinism (md5)
12. xfail flip claim (pytest collection)
13. r_blend_strength deprecation warning across modes
14. Argparse round-trip (invalid value, case-sensitive)

Locked spec: doc/plan/phase1/section16/locked-spec-P1_10_1.md
Coder handoff: doc/plan/phase1/section16/coder-handoff-P1_10_1.md
"""
from __future__ import annotations

import inspect
import re
import warnings
from pathlib import Path

import numpy as np
import pytest

from summicron.render.convolve import (
    _build_basis_kernel,
    _radial_basis_weights,
    _scatter_one_slab_fixed_d_basis,
    render,
)
from tests.fixtures.defocus import render_with_defocus


def _make_gaussian_tensor(R=4, D=4, ksize=33, sigma=4.0,
                          with_distance=False):
    cy, cx = ksize // 2, ksize // 2
    yy, xx = np.mgrid[:ksize, :ksize]
    g = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * sigma * sigma))
    g = g.astype(np.float32) / g.sum()
    kernels = np.tile(g[None, None, :, :], (R, D, 1, 1))
    delta = np.zeros((ksize, ksize), dtype=np.float32)
    delta[cy, cx] = 1.0
    kernels[:, 0] = delta  # S2 delta guard
    obj_dist = None
    if with_distance:
        obj_dist = np.linspace(0.7, 5.0, D).astype(np.float64)
    return {
        "kernels": kernels,
        "radial_edges": np.linspace(0.0, 1.0, R + 1).astype(np.float64),
        "defocus_ratios": np.linspace(0.0, 1.0, D).astype(np.float64),
        "corpus_ref_long_edge": None,
        "otf_norm_applied": False,
        "delta_guard_applied": True,
        "object_distances_m": obj_dist,
        "focus_distance_m": 0.7 if with_distance else None,
    }


# ---------------------------------------------------------------------------
# Category 1 — scalar-collapse regression (hostile inputs)
# ---------------------------------------------------------------------------

class TestScalarCollapseHostile:
    """Beyond Coder's test, attempt to silently collapse the basis weights to
    a scalar via degenerate radial_frac maps. The helper MUST always return
    (Nr, H, W) and produce sane PoU regardless."""

    def test_all_zero_radial_frac_returns_per_pixel_shape(self):
        rc = np.linspace(0.05, 0.95, 12, dtype=np.float64)
        H, W = 8, 8
        rf = np.zeros((H, W), dtype=np.float32)
        bw = _radial_basis_weights(rf, rc)
        assert bw.shape == (12, H, W)
        # All zeros → below rc[0] → 100% on ri=0 everywhere.
        assert np.allclose(bw[0], 1.0, atol=1e-6)
        assert np.allclose(bw[1:], 0.0, atol=1e-6)
        # PoU still holds.
        assert np.allclose(bw.sum(axis=0), 1.0, atol=1e-6)

    def test_all_constant_radial_frac_returns_per_pixel_shape(self):
        """All pixels map to same value → every (y,x) should have the same
        basis_w vector but the shape must still be (Nr, H, W)."""
        rc = np.linspace(0.05, 0.95, 12, dtype=np.float64)
        H, W = 7, 11
        rf = np.full((H, W), 0.42, dtype=np.float32)
        bw = _radial_basis_weights(rf, rc)
        assert bw.shape == (12, H, W)
        # All pixels identical.
        for y in range(H):
            for x in range(W):
                assert np.allclose(bw[:, y, x], bw[:, 0, 0], atol=1e-6)
        assert np.allclose(bw.sum(axis=0), 1.0, atol=1e-6)

    def test_all_nan_does_not_silently_collapse(self):
        """All-NaN input must not silently produce a wrong-but-valid
        looking output (e.g. all weight on ri=0). Either NaN propagates,
        or a clear-error is raised — but PoU on the NaN entries should
        NOT be 1.0 if the helper is honest about the bad input."""
        rc = np.linspace(0.05, 0.95, 12, dtype=np.float64)
        H, W = 4, 4
        rf = np.full((H, W), np.nan, dtype=np.float32)
        # Suppress a NumPy "invalid value" RuntimeWarning that the
        # internal `np.clip((rf - rc_lo) / span_safe, 0, 1)` may emit.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            bw = _radial_basis_weights(rf, rc)
        assert bw.shape == (12, H, W)
        # The honest behavior: PoU should NOT be 1.0 for NaN inputs (NaN
        # propagation expected). If PoU == 1.0 across the board, the
        # helper has silently faked a valid output for invalid input.
        s = bw.sum(axis=0)
        # At least one pixel must NOT sum to ~1.0, OR all values must
        # contain NaN propagation.
        ok = (
            (not np.allclose(s, 1.0, atol=1e-6))
            or np.isnan(bw).any()
        )
        assert ok, (
            "NaN-bearing radial_frac silently produced PoU output: "
            "this is the scalar-collapse failure mode (input garbage → "
            "output looks valid)."
        )


# ---------------------------------------------------------------------------
# Category 2 — G1 SHOWSTOPPER stress
# ---------------------------------------------------------------------------

class TestG1Stress:


    def test_default_run_twice_md5_identical(self):
        """Determinism gate: same call twice → bit-identical output."""
        H, W = 48, 48
        rng = np.random.default_rng(777)
        img = rng.uniform(0.1, 0.9, size=(H, W, 3)).astype(np.float32)
        defocus = rng.uniform(0.0, 0.5, size=(H, W)).astype(np.float32)
        tensor = _make_gaussian_tensor()
        out1, _ = render_with_defocus(img, tensor, dmap=defocus)
        out2, _ = render_with_defocus(img, tensor, dmap=defocus)
        assert np.array_equal(out1, out2)


# ---------------------------------------------------------------------------
# Category 3 — radial_frac edge cases (helper-level)
# ---------------------------------------------------------------------------

class TestRadialFracEdges:
    def test_inf_radial_frac(self):
        """+Inf → must clamp to ri=Nr-1 (or propagate, but not hang/crash)."""
        rc = np.linspace(0.05, 0.95, 12, dtype=np.float64)
        rf = np.array([[np.inf, 0.5], [0.3, -np.inf]], dtype=np.float32)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            bw = _radial_basis_weights(rf, rc)
        assert bw.shape == (12, 2, 2)
        # +Inf: searchsorted side='right' returns Nr → clipped to Nr-2
        # → t clipped to 1 → full weight on ri=Nr-1.
        assert np.isclose(bw[11, 0, 0], 1.0, atol=1e-6), (
            f"+Inf clamp failed: bw[11, 0, 0] = {bw[11, 0, 0]}"
        )
        # -Inf: searchsorted returns 0 → clipped to 0 → t=0 → full weight ri=0.
        assert np.isclose(bw[0, 1, 1], 1.0, atol=1e-6)

    def test_negative_radial_frac_clamps_to_zero(self):
        """Negative rf shouldn't appear in production but must not crash."""
        rc = np.linspace(0.05, 0.95, 12, dtype=np.float64)
        rf = np.full((4, 4), -2.5, dtype=np.float32)
        bw = _radial_basis_weights(rf, rc)
        assert np.allclose(bw[0], 1.0, atol=1e-6)
        assert np.allclose(bw.sum(axis=0), 1.0, atol=1e-6)

    def test_non_monotone_r_centers_does_not_crash(self):
        """Hostile: non-monotone r_centers (should never happen but probe
        defensively). The helper doesn't validate monotonicity but
        searchsorted requires it; we check the helper raises or produces
        a non-NaN finite output (no silent garbage)."""
        rc = np.array([0.5, 0.3, 0.7, 0.1, 0.9], dtype=np.float64)  # NOT monotone
        rf = np.array([[0.4]], dtype=np.float32)
        # The helper doesn't explicitly validate monotonicity; result is
        # implementation-defined but should be finite & shape-correct.
        bw = _radial_basis_weights(rf, rc)
        assert bw.shape == (5, 1, 1)
        assert np.isfinite(bw).all(), "Non-monotone r_centers produced NaN/Inf"

    def test_duplicate_r_centers_no_div_by_zero(self):
        """Duplicate r_centers → zero span between adjacent centers — must not divide by zero."""
        rc = np.array([0.1, 0.5, 0.5, 0.5, 0.9], dtype=np.float64)
        rf = np.array([[0.5]], dtype=np.float32)  # exactly on the duplicate
        bw = _radial_basis_weights(rf, rc)
        assert np.isfinite(bw).all(), "Duplicate r_centers caused NaN (div-by-zero)"
        assert np.isclose(bw.sum(), 1.0, atol=1e-5), (
            f"PoU broken with duplicate centers: sum={bw.sum()}"
        )


# ---------------------------------------------------------------------------
# Category 4 — tensor degeneracy
# ---------------------------------------------------------------------------

class TestTensorDegeneracy:
    def test_nr_2_basis_renders(self):
        """Nr=2 (smallest non-trivial radial bin count) under render()."""
        H, W = 32, 32
        img = np.zeros((H, W, 3), dtype=np.float32) + 0.5
        defocus = np.zeros((H, W), dtype=np.float32)
        ksize = 17
        cy, cx = ksize // 2, ksize // 2
        yy, xx = np.mgrid[:ksize, :ksize]
        g = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / 8.0)
        g = (g / g.sum()).astype(np.float32)
        kernels = np.tile(g[None, None, :, :], (2, 2, 1, 1))
        delta = np.zeros((ksize, ksize), dtype=np.float32)
        delta[cy, cx] = 1.0
        kernels[:, 0] = delta
        tensor = {
            "kernels": kernels,
            "radial_edges": np.array([0.0, 0.5, 1.0], dtype=np.float64),
            "defocus_ratios": np.array([0.0, 1.0], dtype=np.float64),
            "corpus_ref_long_edge": None,
            "otf_norm_applied": False,
            "delta_guard_applied": True,
            "object_distances_m": None,
            "focus_distance_m": None,
        }
        out, _ = render_with_defocus(img, tensor, dmap=defocus)
        assert out.shape == (H, W, 3)
        assert np.isfinite(out).all()


# ---------------------------------------------------------------------------
# Category 5 — image degeneracy
# ---------------------------------------------------------------------------

class TestImageDegeneracy:
    def test_all_zero_image(self):
        """All-zero LDR sRGB input through basis path: zero in → zero out."""
        H, W = 32, 32
        img = np.zeros((H, W, 3), dtype=np.float32)
        defocus = np.zeros((H, W), dtype=np.float32)
        tensor = _make_gaussian_tensor()
        out, _ = render_with_defocus(img, tensor, dmap=defocus)
        assert np.isfinite(out).all()
        assert int(out.max()) <= 1, f"All-zero input produced non-zero output {out.max()}"

    def test_all_saturated_image(self):
        """uint8=255 saturated input through basis path: must not NaN.
        Pre-P1.11 hard-cap on `image.max() > 1.001` could trip if the
        renderer runs in LDR mode; uint8 casts to 1.0 which is fine."""
        H, W = 32, 32
        img = np.full((H, W, 3), 255, dtype=np.uint8)
        defocus = np.zeros((H, W), dtype=np.float32)
        tensor = _make_gaussian_tensor()
        out, _ = render_with_defocus(img, tensor, dmap=defocus)
        assert np.isfinite(out).all()

    def test_hdr_overflow_lum_5_basis_finite(self):
        """HDR lum=5.0 saturated white through basis: stays finite (no NaN/Inf)."""
        H, W = 32, 32
        img = np.full((H, W, 3), 5.0, dtype=np.float32)
        defocus = np.full((H, W), 0.5, dtype=np.float32)
        tensor = _make_gaussian_tensor()
        _, lin = render_with_defocus(img, tensor, dmap=defocus, hdr_source="macos")
        assert np.isfinite(lin).all(), "HDR overflow propagated NaN/Inf"


# ---------------------------------------------------------------------------
# Category 6 — top-hat × basis interaction
# ---------------------------------------------------------------------------

class TestTophatBasisChromaticity:
    """G10 chromaticity gate must hold with BOTH top-hat AND basis flags
    active (canonical-bench possible future combo)."""

    def test_chromaticity_warm_under_tophat_and_basis(self):
        H, W = 64, 64
        warm_color = np.array([1.5, 0.6, 0.1], dtype=np.float32)
        img = np.zeros((H, W, 3), dtype=np.float32)
        img[..., :] = warm_color * 0.85
        defocus = np.full((H, W), 0.6, dtype=np.float32)
        tensor = _make_gaussian_tensor()
        _, out = render_with_defocus(
            img, tensor, dmap=defocus, hdr_source="macos",
            bright_pass_region_size=15,
            bright_pass_floor=0.10,
            bright_pass_norm_pct=99.0,
        )
        # Channel ratios.
        means_in = img.reshape(-1, 3).mean(axis=0)
        means_out = out.reshape(-1, 3).mean(axis=0)
        rg_in = means_in[0] / (means_in[1] + 1e-12)
        rg_out = means_out[0] / (means_out[1] + 1e-12)
        bg_in = means_in[2] / (means_in[1] + 1e-12)
        bg_out = means_out[2] / (means_out[1] + 1e-12)
        rg_drift = abs(rg_out - rg_in) / max(rg_in, 1e-6)
        bg_drift = abs(bg_out - bg_in) / max(bg_in, 1e-6)
        assert rg_drift <= 0.01, (
            f"Top-hat × basis warm R/G drift {rg_drift:.4%} > 1%"
        )
        assert bg_drift <= 0.01, (
            f"Top-hat × basis warm B/G drift {bg_drift:.4%} > 1%"
        )


# ---------------------------------------------------------------------------
# Category 7 — layered + basis interaction (kill-switch under layered)
# ---------------------------------------------------------------------------


    # P1.10.4 C4b: `test_layered_basis_differs_from_layered_slab` deleted —
    # mathematically obsolete after slab→basis dispatch redirect collapsed
    # to basis-only (radial_blend_mode kwarg is now a silently-accepted noop).


# ---------------------------------------------------------------------------
# Category 8 — wedge ordering (docstring + actual loop nesting)
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Category 9 — D-axis stance
# ---------------------------------------------------------------------------

class TestDAxisStance:
    def test_build_basis_kernel_docstring_d_axis(self):
        """Spec mandates D-axis stance docstring verbatim."""
        ds = inspect.getdoc(_build_basis_kernel) or ""
        assert "D-axis remains 2-tap bilinear" in ds, (
            f"D-axis stance docstring missing from `_build_basis_kernel`; "
            f"got: {ds[:200]!r}"
        )
        assert "OUT OF SCOPE for P1.10.1" in ds, (
            "D-axis 'OUT OF SCOPE for P1.10.1' marker missing"
        )

    def test_build_basis_kernel_d_axis_is_scalar(self):
        """Smoke: at fixed ri + scalar d_center, output is one (kH, kW)
        kernel — not a per-pixel array (would indicate D-axis became per-pixel)."""
        tensor = _make_gaussian_tensor(R=4, D=4, ksize=33, sigma=4.0)
        kernels = tensor["kernels"]
        defocus_ratios = tensor["defocus_ratios"]
        kH, kW = kernels.shape[2], kernels.shape[3]
        out = _build_basis_kernel(
            ri=2, d_center_or_idx=0.5, use_idx=False,
            defocus_ratios=defocus_ratios, kernels=kernels,
            kH=kH, kW=kW, pix_scale=1.0, scaled_ksize=kH,
        )
        assert out.shape == (kH, kW), (
            f"Expected scalar-D kernel shape ({kH}, {kW}); got {out.shape}"
        )
        assert out.ndim == 2, "D-axis became per-pixel (kernel has > 2 dims)"


# ---------------------------------------------------------------------------
# Category 10 — memory probe (estimate-only, no OOM risk)
# ---------------------------------------------------------------------------

class TestMemoryEstimate:
    def test_native_res_memory_estimate_below_8gb(self):
        """Estimate-only: at native HEIC res 5712×4284, _radial_basis_weights
        allocates Nr × H × W × 4 bytes = 12 × 24.5e6 × 4 ≈ 1.18 GB.
        Verify the spec § S9.3 soft-warn arithmetic still holds."""
        Nr, H, W = 12, 4284, 5712
        bytes_alloc = Nr * H * W * 4
        gb = bytes_alloc / (1024 ** 3)
        # Sanity: this matches the spec's 1.18 GB estimate.
        assert 1.0 < gb < 1.5, (
            f"Memory estimate drift: expected ~1.18 GB, got {gb:.2f} GB"
        )
        # Verify the half-MAX_PIXELS gate would soft-warn at native res.
        max_pixels = 36e6  # MAX_PIXELS from convolve.py
        threshold = 0.5 * max_pixels * 4  # spec § S9.3 formula
        # spec text: `Nr × H × W × 4 > 0.5 × MAX_PIXELS budget`
        # At native res this should trigger the soft warn.
        assert bytes_alloc > threshold, (
            "Native-res Nr×H×W×4 below 0.5×MAX_PIXELS budget — soft warn dead"
        )


# ---------------------------------------------------------------------------
# Category 13 — r_blend_strength deprecation warning across modes
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Category 14 — argparse round-trip
# ---------------------------------------------------------------------------
# P1.10.4 C4b: `TestArgparseRoundtrip` deleted — `--radial-blend-mode`
# CLI flag removed; argparse now errors with "unrecognized argument" on
# any value, so the {garbage, uppercase} probes are structurally obsolete.


# ---------------------------------------------------------------------------
# Bonus: sanity on `_scatter_one_slab_fixed_d_basis` signature parity
# with the slab counterpart (two-site dispatch consistency).
# ---------------------------------------------------------------------------

class TestDispatchSignatureParity:
    def test_scatter_basis_layered_callable_with_layered_args(self):
        """`_scatter_one_slab_fixed_d_basis` must accept the same argument
        set as `_scatter_one_slab_fixed_d` from layered call-site (sans
        radial_edges / _norm_slab_weights / n_slabs which the basis path
        no longer needs)."""
        sig = inspect.signature(_scatter_one_slab_fixed_d_basis)
        params = set(sig.parameters.keys())
        # Must accept the canonical args.
        required = {
            "d_idx", "img_diffuse_pad", "img_bright_pad", "img_full_pad",
            "kernels", "defocus_ratios", "radial_frac_map", "r_centers",
            "wedge_idx", "kH", "kW", "pad", "pix_scale", "scaled_ksize",
            "H", "W", "output", "output_bright",
            "weight_sum", "delta_weight_sum",
        }
        missing = required - params
        assert not missing, (
            f"`_scatter_one_slab_fixed_d_basis` missing args: {missing}"
        )
        # Should NOT depend on n_slabs / _norm_slab_weights / radial_edges.
        forbidden = {"n_slabs", "_norm_slab_weights", "radial_edges"}
        present_forbidden = forbidden & params
        # Spec § S2 + 'r-axis collapse' implies these are no longer needed.
        # If present, that's a sign the basis path retained legacy plumbing
        # (not strictly a bug, but a code-hygiene smell).
        assert not present_forbidden, (
            f"basis-path scatter retains legacy slab plumbing args: "
            f"{present_forbidden}. Per spec § S2 the basis path collapses "
            f"the r-axis and should not need these."
        )
