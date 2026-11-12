"""P1.10.1 — `_radial_basis_weights` PoU + identity + boundary.

Tests for the per-pixel piecewise-linear partition-of-unity over Nr
radial basis indices used by the new `--radial-blend-mode basis` path.

Locked spec: doc/plan/phase1/section16/locked-spec-P1_10_1.md § S6 item 7
+ § S8.3 (scalar-collapse guard — load-bearing).

Gate matrix:
- G5 PoU sanity:    Σ_ri basis_w[ri,y,x] = 1 ± 1e-6 for all (y,x).
- G6 r-axis ID:     basis_w[ri, y, x] == 1.0 when radial_frac == r_centers[ri].
- G6b clamp:        100% on ri=0 below r_centers[0]; 100% on ri=Nr-1 above r_centers[-1].
- § S8.3 guard:     return shape (Nr, H, W) AND distinct (y, x) with distinct
                    radial_frac yield distinct basis_w[ri, y, x] (no scalar collapse).
"""
from __future__ import annotations

import numpy as np
import pytest

from summicron.render.convolve import _radial_basis_weights


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _canonical_r_centers(Nr: int = 12) -> np.ndarray:
    """Mirror the canonical CodeV tensor's R-axis sqrt-quadrature edges → centers.

    The raw tensor's `radial_bin_edges_hmax = sqrt(linspace(0, 1, 12))` and
    `r_centers = 0.5 * (edges[:-1] + edges[1:])` (length Nr-1). For tests we
    use a length-Nr equispaced approximation since the helper is agnostic to
    the exact spacing — only requires monotone increasing.
    """
    edges = np.sqrt(np.linspace(0.0, 1.0, Nr + 1)).astype(np.float64)
    return (0.5 * (edges[:-1] + edges[1:])).astype(np.float64)


# ---------------------------------------------------------------------------
# G5 — PoU sanity
# ---------------------------------------------------------------------------

class TestPoUSanity:
    """Σ_ri basis_w[ri, y, x] = 1 ± 1e-6 for every pixel."""

    def test_pou_sum_to_one_random_field(self):
        rng = np.random.default_rng(20260427)
        H, W = 32, 64
        rf = rng.uniform(0.0, 1.0, size=(H, W)).astype(np.float32)
        rc = _canonical_r_centers(12)
        bw = _radial_basis_weights(rf, rc)
        s = bw.sum(axis=0)
        assert np.allclose(s, 1.0, atol=1e-6), (
            f"PoU violated: sum range [{s.min()}, {s.max()}]"
        )

    def test_at_most_two_nonzero_per_pixel(self):
        """Bilinear interpolation: at most 2 of Nr planes non-zero per pixel."""
        rng = np.random.default_rng(7)
        H, W = 16, 16
        rf = rng.uniform(0.05, 0.95, size=(H, W)).astype(np.float32)
        rc = _canonical_r_centers(12)
        bw = _radial_basis_weights(rf, rc)
        # Count non-zero (above tiny epsilon) planes at each pixel.
        nnz = (bw > 1e-7).sum(axis=0)
        assert int(nnz.max()) <= 2, (
            f"Expected ≤2 non-zero ri per pixel, got max={nnz.max()}"
        )

    def test_pou_with_full_range_radial(self):
        """Cover endpoints + below + above r_centers range."""
        rc = _canonical_r_centers(12)
        # rf includes: below rc[0], rc[0] exactly, midpoints, rc[-1] exactly,
        # above rc[-1].
        rf = np.array([
            [-0.1, 0.0, float(rc[0]), float(rc[0] + 1e-6)],
            [float(rc[5]), float(0.5 * (rc[5] + rc[6])), float(rc[-1]), 1.5],
        ], dtype=np.float32)
        bw = _radial_basis_weights(rf, rc)
        s = bw.sum(axis=0)
        assert np.allclose(s, 1.0, atol=1e-6), (
            f"PoU violated for boundary inputs: {s}"
        )
        assert (bw >= 0.0).all(), "Negative weights detected"


# ---------------------------------------------------------------------------
# G6 — r-axis identity at node positions
# ---------------------------------------------------------------------------

class TestRAxisIdentity:
    """`basis_w[ri, y, x] == 1` when `radial_frac == r_centers[ri]` exactly."""

    def test_identity_at_each_node(self):
        rc = _canonical_r_centers(12)
        Nr = len(rc)
        # 1 row per node: radial_frac equals rc[i] across the row.
        rf = np.tile(rc.astype(np.float32)[:, None], (1, 5))  # (Nr, 5)
        bw = _radial_basis_weights(rf, rc)  # (Nr, Nr, 5)
        for i in range(Nr):
            # Row i has rf == rc[i] → bw[i, i, :] should be 1, others ≈0.
            assert np.allclose(bw[i, i, :], 1.0, atol=1e-6), (
                f"Identity at rc[{i}]: bw[{i}, {i}, 0] = {bw[i, i, 0]} (expected 1.0)"
            )
            others = np.delete(bw[:, i, :], i, axis=0)
            assert np.all(others < 1e-6), (
                f"Non-zero on non-identity ri at node rc[{i}]: max = {others.max()}"
            )

    def test_identity_first_node_clamped(self):
        rc = _canonical_r_centers(12)
        rf = np.full((4, 4), float(rc[0]), dtype=np.float32)
        bw = _radial_basis_weights(rf, rc)
        # At rc[0], bilinear places full weight on ri=0 (since searchsorted
        # of rc[0] in rc is 0 with side='right' minus 1 = -1 clipped to 0;
        # then t = 0).
        assert np.allclose(bw[0], 1.0, atol=1e-6), (
            f"Expected bw[0] all ones at rc[0]; got max-min [{bw[0].min()}, {bw[0].max()}]"
        )


# ---------------------------------------------------------------------------
# G6b — boundary clamp
# ---------------------------------------------------------------------------

class TestBoundaryClamp:
    """At rf < rc[0]: all weight on ri=0. At rf > rc[-1]: all weight on ri=Nr-1."""

    def test_below_first_center(self):
        rc = _canonical_r_centers(12)
        rf = np.full((3, 5), -0.5, dtype=np.float32)  # well below rc[0]
        bw = _radial_basis_weights(rf, rc)
        assert np.allclose(bw[0], 1.0, atol=1e-6), (
            f"Below rc[0]: ri=0 weight not unity, got {bw[0].mean()}"
        )
        assert np.allclose(bw[1:], 0.0, atol=1e-6), (
            "Below rc[0]: non-zero weight on ri>0"
        )

    def test_above_last_center(self):
        rc = _canonical_r_centers(12)
        Nr = len(rc)
        rf = np.full((3, 5), 5.0, dtype=np.float32)  # well above rc[-1]
        bw = _radial_basis_weights(rf, rc)
        assert np.allclose(bw[Nr - 1], 1.0, atol=1e-6), (
            f"Above rc[-1]: ri=Nr-1 weight not unity, got {bw[Nr - 1].mean()}"
        )
        assert np.allclose(bw[:Nr - 1], 0.0, atol=1e-6), (
            "Above rc[-1]: non-zero weight on ri<Nr-1"
        )

    def test_no_negative_no_nan(self):
        rng = np.random.default_rng(13)
        rc = _canonical_r_centers(12)
        rf = rng.uniform(-1.0, 2.0, size=(8, 8)).astype(np.float32)
        bw = _radial_basis_weights(rf, rc)
        assert np.isfinite(bw).all(), "NaN/Inf in basis_w"
        assert (bw >= 0.0).all(), f"Negative weight: min = {bw.min()}"


# ---------------------------------------------------------------------------
# § S8.3 — scalar-collapse guard (LOAD-BEARING)
# ---------------------------------------------------------------------------

class TestScalarCollapseGuard:
    """If a future refactor accidentally collapses to a scalar `r_pix_w(ri)`
    (e.g. by using the global mean of `radial_frac_map`), the basis blend
    reduces to legacy slab structure with NO ring fix. These tests guard
    against that regression by asserting:
      1. shape is exactly (Nr, H, W).
      2. for distinct (y0, x0), (y1, x1) with distinct radial_frac, the
         per-pixel basis weights differ (i.e. weights actually depend on
         pixel position).
    """

    def test_returns_per_pixel_shape(self):
        rc = _canonical_r_centers(12)
        H, W = 7, 11
        rf = np.linspace(0.0, 1.0, H * W, dtype=np.float32).reshape(H, W)
        bw = _radial_basis_weights(rf, rc)
        assert bw.shape == (len(rc), H, W), (
            f"Expected shape ({len(rc)}, {H}, {W}); got {bw.shape}"
        )
        assert bw.dtype == np.float32

    def test_per_pixel_weights_differ(self):
        """basis_w[ri, y0, x0] != basis_w[ri, y1, x1] for distinct radial_frac."""
        rc = _canonical_r_centers(12)
        H, W = 5, 5
        # Construct a radial_frac_map with explicit distinct values.
        rf = np.zeros((H, W), dtype=np.float32)
        rf[0, 0] = 0.10  # well below rc[1]
        rf[1, 1] = 0.40  # mid range
        rf[2, 2] = 0.80  # near top
        rf[3, 3] = 0.95  # near top
        rf[4, 4] = 0.20  # second bracket
        bw = _radial_basis_weights(rf, rc)
        # For at least one ri, the weight should differ across distinct
        # radial_frac pixels.
        diffs_seen = 0
        for ri in range(len(rc)):
            vals = np.array([
                bw[ri, 0, 0], bw[ri, 1, 1], bw[ri, 2, 2], bw[ri, 3, 3], bw[ri, 4, 4]
            ])
            if vals.max() - vals.min() > 1e-6:
                diffs_seen += 1
        assert diffs_seen >= 2, (
            f"Per-pixel dependence not detected: only {diffs_seen} ri planes "
            "show variation across distinct radial_frac inputs. The helper "
            "may have collapsed to a scalar (§ S8.3 critical bug)."
        )

    def test_distinct_pixels_distinct_weights(self):
        """Direct positive assertion per § S8.3:
           basis_w[ri, y0, x0] != basis_w[ri, y1, x1] for distinct
           (y0, x0), (y1, x1) with distinct radial_frac values."""
        rc = _canonical_r_centers(12)
        rf = np.array([
            [0.10, 0.30],
            [0.60, 0.85],
        ], dtype=np.float32)
        bw = _radial_basis_weights(rf, rc)
        # Pick two pixels with distinct radial_frac and confirm the entire
        # basis_w *vector* differs (not just one ri).
        v00 = bw[:, 0, 0]
        v01 = bw[:, 0, 1]
        v10 = bw[:, 1, 0]
        v11 = bw[:, 1, 1]
        assert not np.allclose(v00, v01, atol=1e-6), (
            f"§ S8.3 SCALAR-COLLAPSE: pixels (0,0) and (0,1) have distinct "
            f"radial_frac (0.10 vs 0.30) but identical basis_w!"
        )
        assert not np.allclose(v00, v10, atol=1e-6)
        assert not np.allclose(v01, v11, atol=1e-6)


# ---------------------------------------------------------------------------
# Degenerate Nr=1 tensor smoke
# ---------------------------------------------------------------------------

def test_nr_1_returns_unit_plane():
    rc = np.array([0.5], dtype=np.float64)
    rf = np.linspace(0.0, 1.0, 16, dtype=np.float32).reshape(4, 4)
    bw = _radial_basis_weights(rf, rc)
    assert bw.shape == (1, 4, 4)
    assert np.allclose(bw, 1.0, atol=1e-6)


def test_nr_2_two_centers():
    """Smallest non-trivial case: Nr=2 → linear blend between two centers."""
    rc = np.array([0.25, 0.75], dtype=np.float64)
    # rf at midpoint -> 50/50.
    rf = np.array([[0.50]], dtype=np.float32)
    bw = _radial_basis_weights(rf, rc)
    assert bw.shape == (2, 1, 1)
    assert np.allclose(bw[0, 0, 0], 0.5, atol=1e-6)
    assert np.allclose(bw[1, 0, 0], 0.5, atol=1e-6)
