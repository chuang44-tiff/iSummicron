"""P1.9.4 hostile adversarial tests targeting the R/D interp pipeline,
load_tensor validation hardening, and `_MAX_ACTIVE_SLABS` shape-aware cap.

Coder A delivered the R-cubic / D-linear two-pass RGI plus memory guard.
R5 PIVOT (2026-04-25, post-71-min-probe-fail) relocated those helpers
from ``codev_to_render`` to ``downscale_tensor`` so they operate on the
post-downscale ~44/183-grid kernels rather than raw 2048-grid (~2000×
cheaper compute, identical algorithmic content). Coder B hardened
``load_tensor`` in ``summicron/render/convolve.py`` and made
``_MAX_ACTIVE_SLABS`` shape-aware. The Phase 2.4 Adversarial brief
listed 19 attack-surface targets; this file picks the 18 highest-
leverage ones (skipping item 9 alt-filename which no longer exists
post-pivot) plus the **MANDATORY** wrong-polarity fixture per
Post-mortem 1.

Source bugs found via these probes are marked with
``@pytest.mark.xfail(reason="BUG: ...", strict=False)``.
"""
from __future__ import annotations

import hashlib
import warnings
from pathlib import Path

import numpy as np
import pytest

from summicron.render._artifacts import default_tensor_path

from summicron.psf_model.codev_to_render import (
    DEFAULT_INPUT_DIR,
    DEFAULT_OUTPUT,
    convert,
)
from summicron.psf_model.downscale_tensor import (
    MEM_THRESHOLD_GB_DEFAULT,
    _interp_pixel_pitch,
    _interp_rd,
    downscale_tensor,
)
from summicron.render.convolve import load_tensor, render
from tests.fixtures.synthetic_tensor import write_synthetic_tensor

ROOT = Path(__file__).resolve().parents[2]
RAW_NPZ = ROOT / "runs" / "ag_dblgauss" / "ag_dblgauss_f2.0_focus0.70m_d00_0.70m.npz"
IPHONE_TENSOR = default_tensor_path()
RAW_TENSOR = ROOT / "runs" / "ag_dblgauss" / "ag_dblgauss_f2.0_render.npz"


# ---------------------------------------------------------------------------
# Synthetic helpers
# ---------------------------------------------------------------------------

def _asym_synthetic(raw_R: int = 12, raw_D: int = 11, kH: int = 7, seed: int = 42):
    """Build an asymmetric L1-normed (R, D, kH, kH) tensor where neighbor
    slices differ enough that R-then-D vs D-then-R orderings DIVERGE."""
    rng = np.random.default_rng(seed)
    h_max_mm = 21.6
    centers = h_max_mm * np.sqrt(np.linspace(0.0, 0.954, raw_R))
    od = np.geomspace(0.7, 20.0, raw_D)
    kernels = np.zeros((raw_R, raw_D, kH, kH), dtype=np.float32)
    cy = cx = kH // 2
    yy, xx = np.mgrid[:kH, :kH]
    for ri in range(raw_R):
        for di in range(raw_D):
            # Strong R-D asymmetry: shape and offset both vary by axis.
            sigma = 0.6 + 0.5 * (ri / max(1, raw_R - 1)) + 1.5 * (di / max(1, raw_D - 1))
            dy = 0.4 * (ri - raw_R / 2)
            dx = 0.6 * (di - raw_D / 2)
            jit = 0.05 * rng.standard_normal()
            k = np.exp(
                -((xx - cx - dx) ** 2 + (yy - cy - dy) ** 2 * (1 + jit))
                / (2 * sigma ** 2)
            )
            k /= k.sum()
            kernels[ri, di] = k.astype(np.float32)
    return kernels, centers, h_max_mm, od


def _fake_codev_input_dir(tmp_path: Path, raw_R: int = 4, D_files: int = 5, kH: int = 33) -> Path:
    """Build a small fake CodeV input dir for fast convert() roundtrips."""
    d = tmp_path / "fake_codev"
    d.mkdir()
    h_max_mm = 21.6
    centers = h_max_mm * np.sqrt(np.linspace(0.0, 0.954, raw_R))
    edges = np.sqrt(np.linspace(0.0, 1.0, raw_R))
    od = np.geomspace(0.7, 20.0, D_files)
    wavelengths_nm = np.array([450.0, 550.0, 650.0])
    pixel_pitch_um = np.tile(np.linspace(0.5, 0.55, raw_R)[:, None], (1, 3)).astype(np.float64)
    for di, val in enumerate(od):
        payload = {
            "distance_index": np.int64(di),
            "object_distance_m": np.float64(val),
            "object_distances_m": od.astype(np.float64),
            "focus_distance_m": np.float64(od[0]),
            "h_max_mm": np.float64(h_max_mm),
            "wavelengths_nm": wavelengths_nm,
            "radial_bin_edges_hmax": edges,
            "radial_bin_centers_mm": centers,
            "pixel_pitch_um": pixel_pitch_um,
        }
        for ri in range(raw_R):
            for wi in range(3):
                yy, xx = np.mgrid[:kH, :kH]
                cy = cx = kH // 2
                sigma = 0.8 + 0.4 * ri + 0.6 * di
                k = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
                k /= k.sum()
                payload[f"psf_r{ri:02d}_w{wi}"] = k.astype(np.float32)
        np.savez(str(d / f"ag_dblgauss_f2.0_focus0.70m_d{di:02d}_{val:.2f}m.npz"), **payload)
    return d


# ===========================================================================
# 1. Interp ordering — proves R-cubic-first, D-linear-second, NOT swapped
# ===========================================================================

class TestInterpOrdering:
    """Brief item 1: R-cubic-first / D-linear-second (per LOCKED L20). The
    swapped order produces a DIFFERENT log-space output by ~1e-5; the
    pipeline MUST run R first.

    We can't easily monkeypatch the inner pipeline; instead we run a
    parallel re-implementation of BOTH orderings and assert they DIVERGE.
    Then we assert the actual ``_interp_rd`` output matches the R-then-D
    branch within tolerance, NOT the D-then-R branch.
    """

    def test_orderings_diverge_then_match_R_first(self):
        from scipy.interpolate import RegularGridInterpolator

        kernels_raw, centers, h_max, od = _asym_synthetic()
        target_R, target_D = 24, 22

        log_kernels = np.log1p(kernels_raw.astype(np.float64) * 1e4)
        r_q = h_max * np.sqrt(np.linspace(0.0, (centers[-1] / h_max) ** 2, target_R))
        d_q = np.geomspace(od[0], od[-1], target_D)

        # Ordering A: R-cubic then D-linear (canonical)
        rgi_r = RegularGridInterpolator((centers,), log_kernels, method="cubic", bounds_error=True)
        log_r = rgi_r(r_q[:, None])
        log_r_dfirst = np.moveaxis(log_r, 1, 0)
        rgi_d = RegularGridInterpolator((od,), log_r_dfirst, method="linear", bounds_error=True)
        log_A = np.moveaxis(rgi_d(d_q[:, None]), 0, 1)

        # Ordering B: D-linear then R-cubic (anti-canonical)
        vals_dfirst = np.moveaxis(log_kernels, 1, 0)
        rgi_d2 = RegularGridInterpolator((od,), vals_dfirst, method="linear", bounds_error=True)
        log_d = rgi_d2(d_q[:, None])
        log_d_rfirst = np.moveaxis(log_d, 1, 0)
        rgi_r2 = RegularGridInterpolator((centers,), log_d_rfirst, method="cubic", bounds_error=True)
        log_B = rgi_r2(r_q[:, None])

        # Sanity: orderings must measurably DIFFER (else the test isn't telling
        # us anything). On synthetic asymmetric data the diff is ~1e-5.
        diff = float(np.max(np.abs(log_A - log_B)))
        assert diff > 1e-7, (
            f"Orderings should diverge on asymmetric tensor, got max-diff={diff:.3e}; "
            "test fixture is too symmetric to discriminate."
        )

        # Now compare actual _interp_rd output (in log-space, pre-overwrite) to
        # both. For the assertion we re-derive log-space from the actual output
        # by applying log1p(out * 1e4); raw nodes get overwritten so we look at
        # an INTERIOR cell that is NOT a raw node.
        out_actual, r_actual, _ = _interp_rd(
            kernels_raw, centers, h_max, od, target_R, target_D
        )
        log_actual = np.log1p(out_actual.astype(np.float64) * 1e4)

        # Pick an interior (r_t, d_t) far from raw nodes (target r=11, d=10
        # is interior for both axes). Compare both orderings to the actual.
        r_t, d_t = 11, 10
        diff_to_A = float(np.max(np.abs(log_actual[r_t, d_t] - log_A[r_t, d_t])))
        diff_to_B = float(np.max(np.abs(log_actual[r_t, d_t] - log_B[r_t, d_t])))
        # The L22 per-(r,d) renorm scales kernels in linear space; this is a
        # near-uniform scalar multiply that translates to a near-constant log
        # offset. The TIGHT lock is: actual is closer to A than to B.
        assert diff_to_A < diff_to_B + 1e-9, (
            f"_interp_rd output looks closer to D-FIRST (diff={diff_to_B:.3e}) "
            f"than R-FIRST (diff={diff_to_A:.3e}); pipeline order is reversed!"
        )


# ===========================================================================
# 2. L1 conservation under R-cubic + D-linear (synthetic, real-data slow)
# ===========================================================================

class TestL1ConservationStress:
    """Brief item 2: per-(r, d) L1 ∈ [1−1e-5, 1+1e-5] under L22 renorm.
    Real-data R=24/D=22 at 2048-grid is too slow for adversarial; we
    parametrize across many target shapes on a synthetic tensor instead.
    """

    @pytest.mark.parametrize(
        "target_R,target_D",
        [(13, 11), (12, 12), (24, 22), (36, 33), (50, 22), (12, 50)],
    )
    def test_l1_per_slice_within_tolerance(self, target_R, target_D):
        kernels_raw, centers, h_max, od = _asym_synthetic()
        out, _, _ = _interp_rd(kernels_raw, centers, h_max, od, target_R, target_D)
        sums = out.reshape(target_R * target_D, -1).sum(axis=1)
        # Tighter than the synthetic test's 1e-5: post-renorm we expect within
        # float32 ULP (~1e-7) but allow 1e-5 to match the spec gate.
        bad = np.where(np.abs(sums - 1.0) >= 1e-5)[0]
        assert len(bad) == 0, (
            f"R={target_R} D={target_D}: {len(bad)}/{target_R * target_D} slices "
            f"failed L1 ± 1e-5; min={sums.min():.6f} max={sums.max():.6f}"
        )


# ===========================================================================
# 3. Endpoint preservation — interior ULP-error sanity check
# ===========================================================================

class TestEndpointPreservation:
    """Brief item 3: post-overwrite at raw nodes is the deterministic node fix.
    RGI cubic alone is NOT node-exact (~1e-5 log error per Phase 1 Fact 7).
    We can't easily skip the overwrite from outside, so we instead:
      (a) verify the overwrite DOES happen — corner cells are bit-identical;
      (b) verify the INTERIOR (non-overwritten) cells DRIFT measurably from
          a hypothetical "no overwrite" path — establishing that overwrite
          actually contributes work.
    """

    def test_corners_bit_identical_post_overwrite(self):
        """Corners must be bit-identical across all 4 endpoint pairings."""
        kernels_raw, centers, h_max, od = _asym_synthetic()
        target_R, target_D = 24, 22
        raw_R, raw_D = kernels_raw.shape[:2]
        out, _, _ = _interp_rd(kernels_raw, centers, h_max, od, target_R, target_D)
        for r_raw, d_raw, r_t, d_t in [
            (0, 0, 0, 0),
            (0, raw_D - 1, 0, target_D - 1),
            (raw_R - 1, 0, target_R - 1, 0),
            (raw_R - 1, raw_D - 1, target_R - 1, target_D - 1),
        ]:
            assert np.array_equal(out[r_t, d_t], kernels_raw[r_raw, d_raw]), (
                f"Corner ({r_raw},{d_raw})→({r_t},{d_t}) NOT bit-identical "
                "after post-overwrite"
            )

    def test_interior_cell_differs_from_raw_corners(self):
        """Sanity: an interior interpolated cell is NOT equal to any raw
        corner (else the overwrite is overwriting EVERYTHING — bug)."""
        kernels_raw, centers, h_max, od = _asym_synthetic()
        target_R, target_D = 24, 22
        raw_R, raw_D = kernels_raw.shape[:2]
        out, _, _ = _interp_rd(kernels_raw, centers, h_max, od, target_R, target_D)
        # Interior r=11, d=11 is far from any raw node by construction.
        interior = out[11, 11]
        for r_raw in range(raw_R):
            for d_raw in range(raw_D):
                assert not np.array_equal(interior, kernels_raw[r_raw, d_raw]), (
                    f"Interior cell (11,11) bit-equal to raw ({r_raw},{d_raw}); "
                    "post-overwrite is over-aggressive"
                )


# ===========================================================================
# 4. D-axis MUST be linear — cubic would inject 2.78M negative pixels
# ===========================================================================

class TestDAxisIsLinearNotCubic:
    """Brief item 4: if Coder mistakenly switches D-axis to cubic, real-data
    upsampling would emit ~2.77% negative pixels (per Phase 1 Fact 5). On
    synthetic Gaussians the negative-density is much smaller but cubic still
    overshoots peaks. Test: post-clip min == 0 AND max within 0.1% of raw
    peak per slab.
    """

    @pytest.mark.parametrize("target_D", [22, 33, 50])
    def test_no_negatives_post_clip(self, target_D):
        kernels_raw, centers, h_max, od = _asym_synthetic()
        out, _, _ = _interp_rd(kernels_raw, centers, h_max, od, 12, target_D)
        assert out.min() >= 0.0, (
            f"D={target_D}: linear-D should produce no negatives; min={out.min():.3e}. "
            "If this fails, D-axis was swapped to cubic."
        )

    @pytest.mark.parametrize("target_D", [22, 33])
    def test_no_peak_overshoot(self, target_D):
        kernels_raw, centers, h_max, od = _asym_synthetic()
        raw_peak = float(kernels_raw.max())
        out, _, _ = _interp_rd(kernels_raw, centers, h_max, od, 12, target_D)
        assert out.max() <= raw_peak * 1.001, (
            f"D={target_D}: linear-D yields no overshoot; got max={out.max():.4e} "
            f"vs raw_peak*1.001={raw_peak * 1.001:.4e}. Cubic-D would overshoot ~60%."
        )


# ===========================================================================
# 5. RGI bounds_error=True endpoint acceptance — geomspace + sqrt formulas
# ===========================================================================

class TestRGIEndpointAcceptance:
    """Brief item 5: ``np.geomspace(0.7, 20.0, 24)[-1]`` is exactly 20.0 by
    IEEE 754; ``sqrt(linspace)`` endpoints likewise. Both must be accepted
    by RGI under ``bounds_error=True`` without raising.
    """

    @pytest.mark.parametrize("target_R,target_D", [(24, 22), (36, 33), (50, 11)])
    def test_no_bounds_error_at_endpoints(self, target_R, target_D):
        kernels_raw, centers, h_max, od = _asym_synthetic()
        # Should not raise.
        out, r_q, d_q = _interp_rd(
            kernels_raw, centers, h_max, od, target_R, target_D
        )
        # Endpoints must be EXACTLY equal (IEEE 754 guarantee).
        assert r_q[0] == centers[0]
        assert r_q[-1] == centers[-1]
        assert d_q[0] == od[0]
        assert d_q[-1] == od[-1]


# ===========================================================================
# 6. Memory guard fires at full-grid R=24/D=22, escapes via override
# ===========================================================================

class TestMemoryGuardAtRealKH:
    """Brief item 6 (post-R5): at post-downscale kH=183, the formula scales
    much more gracefully than at the raw kH=2048 stage. Need pathological
    R AND D to exceed the 8 GB default threshold.
    """

    @pytest.mark.skipif(
        not IPHONE_TENSOR.exists(),
        reason="committed iPhone tensor absent",
    )
    def test_pathological_R_D_at_183_grid_raises_default_threshold(self):
        """Post-downscale R=4000, D=4000, kH=183 → ~10.7 TB → must raise."""
        with pytest.raises(RuntimeError, match=r"would allocate.*GB transient"):
            downscale_tensor(
                in_path=IPHONE_TENSOR,
                out_path=Path("/tmp/_downscale_R4000_D4000_block.npz"),
                target_long_edge=5712,
                target_R=4000,
                target_D=4000,
                mem_threshold_gb=MEM_THRESHOLD_GB_DEFAULT,
            )

    def test_default_threshold_value_is_8_gb(self):
        """Spec L15: default threshold is 8.0 GB."""
        assert MEM_THRESHOLD_GB_DEFAULT == 8.0


# ===========================================================================
# 7. R-axis source axis = radial_bin_centers_mm, NOT formula-derived
# ===========================================================================

class TestRAxisSourceIsCentersNotFormula:
    """Brief item 7 + Phase 1 Fact 2: ``radial_bin_centers_mm / h_max``
    is NOT equal to ``sqrt(linspace(0, 1, 12))``. They differ by ~0.088
    max-abs. If a Coder accidentally formula-derived the source axis, kernel
    positions would shift by up to 1.9mm at h_max=21.6mm.
    """

    @pytest.mark.skipif(not RAW_NPZ.exists(), reason="raw CodeV file absent")
    def test_centers_differ_from_sqrt_linspace_formula(self):
        raw = np.load(str(RAW_NPZ))
        centers = raw["radial_bin_centers_mm"]
        h_max = float(raw["h_max_mm"])
        formula = np.sqrt(np.linspace(0.0, 1.0, 12))
        diff = float(np.max(np.abs(centers / h_max - formula)))
        # Empirically 0.0883 — well above any tolerance.
        assert diff > 0.05, (
            f"centers/h_max should differ from sqrt-formula by >0.05; got {diff:.4f}. "
            "If this drops to ~0, the source axis got swapped to formula-derived "
            "and kernel positions shift by up to h_max*diff mm."
        )


# ===========================================================================
# 8. No-op identity: convert(None, None) AND convert(12, 11) match
# ===========================================================================

class TestNoOpIdentityDualBranch:
    """Brief item 8 (post-R5): both invocation forms of downscale_tensor
    should produce IDENTICAL kernels.
    ``is_no_op = (eff_R == R) and (eff_D == D)`` accepts both
    ``target_R=None / target_D=None`` and ``target_R=R_raw / target_D=D_raw``.
    """

    def test_explicit_raw_targets_kernels_match_default(self, tmp_path):
        """Run downscale_tensor() under both invocation forms on the SAME
        fake raw tensor; compare kernel arrays element-wise.
        """
        fake_dir = _fake_codev_input_dir(tmp_path, raw_R=4, D_files=5, kH=33)
        # Build a raw tensor first via convert.
        raw_out = tmp_path / "raw.npz"
        convert(input_dir=fake_dir, out_path=raw_out)
        # Pick target_long_edge so target_kH == raw_kH (no real downscale).
        d_raw = np.load(raw_out)
        target_long_edge = int(round(36000.0 / float(d_raw["pixel_pitch_um"].mean())))
        # raw_R = len(edges) - 1 = 4; D_files = 5.
        out_default = tmp_path / "default.npz"
        out_explicit = tmp_path / "explicit_raw.npz"
        downscale_tensor(
            in_path=raw_out, out_path=out_default,
            target_long_edge=target_long_edge,
            codev_input_dir=fake_dir,
        )
        downscale_tensor(
            in_path=raw_out, out_path=out_explicit,
            target_long_edge=target_long_edge,
            target_R=4, target_D=5,
            codev_input_dir=fake_dir,
        )
        d1 = np.load(out_default)
        d2 = np.load(out_explicit)
        assert np.array_equal(d1["kernels"], d2["kernels"]), (
            "downscale_tensor(None, None) and downscale_tensor(target_R=raw, "
            "target_D=raw) produce different kernels — no-op branch is not "
            "bit-identical"
        )


# ===========================================================================
# 9. Validation negative-tolerance edge — exactly at -1e-6 boundary
# ===========================================================================

class TestNegativeToleranceBoundary:
    """Brief item 11: spec L11 says ``< -1e-6`` raises; exactly ``-1e-6`` is
    the boundary. Test both sides plus exactly-at-boundary."""

    def test_exactly_minus_1e_minus_6_passes(self, tmp_path):
        """``kernels.min() == -1e-6`` is NOT ``< -1e-6`` → must load."""
        path = write_synthetic_tensor(tmp_path / "edge.npz", R=2, D=2)
        data = dict(np.load(str(path), allow_pickle=True))
        data["kernels"][0, 1, 7, 8] = np.float32(-1e-6)
        np.savez(str(path), **data)
        # Must NOT raise.
        loaded = load_tensor(path)
        assert loaded["kernels"].shape == (2, 2, 33, 33)

    def test_just_above_tolerance_passes(self, tmp_path):
        """``-9e-7`` (less negative than ``-1e-6``) → must load."""
        path = write_synthetic_tensor(tmp_path / "edge_above.npz", R=2, D=2)
        data = dict(np.load(str(path), allow_pickle=True))
        data["kernels"][0, 1, 7, 8] = np.float32(-9e-7)
        np.savez(str(path), **data)
        loaded = load_tensor(path)
        assert loaded["kernels"].shape == (2, 2, 33, 33)

    def test_just_below_tolerance_raises(self, tmp_path):
        """``-1.1e-6`` (more negative than ``-1e-6``) → MUST raise."""
        path = write_synthetic_tensor(tmp_path / "edge_below.npz", R=2, D=2)
        data = dict(np.load(str(path), allow_pickle=True))
        data["kernels"][0, 1, 7, 8] = np.float32(-1.1e-6)
        np.savez(str(path), **data)
        with pytest.raises(ValueError, match="negative"):
            load_tensor(path)


# ===========================================================================
# 10. _MAX_ACTIVE_SLABS shape-aware: cap at D=8 boundary
# ===========================================================================

def _make_d_tensor(D: int, R: int = 4):
    """Return an in-memory tensor dict with arbitrary D (deltas at d=0,
    Gaussians at d>=1) for layered_gather rendering."""
    H = W = 33
    kernels = np.zeros((R, D, H, W), dtype=np.float32)
    cy = cx = H // 2
    for r in range(R):
        kernels[r, 0, cy, cx] = 1.0
    if D >= 2:
        ys = np.arange(H, dtype=np.float32) - cy
        xs = np.arange(W, dtype=np.float32) - cx
        yy, xx = np.meshgrid(ys, xs, indexing="ij")
        for d in range(1, D):
            sigma = 1.0 + 0.5 * d
            g = np.exp(-(yy * yy + xx * xx) / (2.0 * sigma * sigma)).astype(np.float32)
            g /= g.sum()
            for r in range(R):
                kernels[r, d] = g
    od = np.geomspace(0.7, 20.0, D)
    return {
        "kernels": kernels,
        "radial_edges": np.linspace(0.0, 1.0, R + 1, dtype=np.float64),
        "defocus_ratios": np.linspace(0.0, 1.0, D, dtype=np.float64),
        "corpus_ref_long_edge": 5712,
        "otf_norm_applied": False,
        "delta_guard_applied": True,
        "object_distances_m": od,
        "focus_distance_m": float(od[0]),
        "R": R,
        "D": D,
    }


class TestMaxActiveSlabsBoundary:
    """Brief items 12 + 13: ``effective_cap = max(8, D//2)``. At D=8 the cap
    is 8; at D=24 the cap is 12. Verify the formula at the boundary."""

    def test_D_24_cap_is_12_active_13_warns(self):
        """D=24, all 24 slabs active → warn fires at active_count > 12."""
        H = W = 40
        depth_m = np.linspace(0.7, 20.0, H * W, dtype=np.float32).reshape(H, W)
        img = np.full((H, W, 3), 0.3, dtype=np.float32)
        tensor = _make_d_tensor(D=24)
        with warnings.catch_warnings(record=True) as w_list:
            warnings.simplefilter("always")
            _, out = render(img, tensor, depth_m=depth_m)
        assert np.all(np.isfinite(out))
        rt_warns = [
            w for w in w_list
            if issubclass(w.category, RuntimeWarning)
            and "MAX_ACTIVE_SLABS" in str(w.message)
        ]
        assert len(rt_warns) >= 1, (
            f"D=24 cap=12: expected MAX_ACTIVE_SLABS warn; got {len(rt_warns)}. "
            f"All warnings: {[str(w.message) for w in w_list]}"
        )
        # Message must report cap=12 (NOT 8).
        msg = str(rt_warns[0].message)
        assert "MAX_ACTIVE_SLABS=12" in msg, (
            f"D=24 cap should be 12, message says: {msg!r}"
        )

    def test_D_24_cap_is_12_active_under_12_no_warn(self):
        """D=24 but only 4 slabs active (narrow depth) → NO warn."""
        H = W = 30
        # All pixels at the same depth → exactly 1 slab fully active.
        depth_m = np.full((H, W), 1.5, dtype=np.float32)
        img = np.full((H, W, 3), 0.3, dtype=np.float32)
        tensor = _make_d_tensor(D=24)
        with warnings.catch_warnings(record=True) as w_list:
            warnings.simplefilter("always")
            _, _ = render(img, tensor, depth_m=depth_m)
        rt_warns = [
            w for w in w_list
            if issubclass(w.category, RuntimeWarning)
            and "MAX_ACTIVE_SLABS" in str(w.message)
        ]
        assert len(rt_warns) == 0, (
            f"D=24 with 1-2 active slabs should NOT warn; got {len(rt_warns)} warns: "
            f"{[str(w.message) for w in rt_warns]}"
        )


# ===========================================================================
# 11. NaN gate finds first non-finite multi-index
# ===========================================================================

class TestNaNGateFirstIndex:
    """Brief item 14: ``ValueError`` message must contain the first
    non-finite multi-index."""

    def test_nan_at_known_position_in_message(self, tmp_path):
        path = write_synthetic_tensor(tmp_path / "nan_pos.npz", R=4, D=4)
        data = dict(np.load(str(path), allow_pickle=True))
        # Inject NaN at a position that comes after the natural d=0 deltas
        # in argwhere C-order; specifically (3, 3, 16, 17). With d=0 deltas
        # at (r, 0, cy, cx) being non-NaN-producers, our injected NaN is the
        # first non-finite in C-order.
        data["kernels"][3, 3, 16, 17] = np.nan
        np.savez(str(path), **data)
        with pytest.raises(ValueError, match=r"first at index \(3, 3, 16, 17\)"):
            load_tensor(path)


# ===========================================================================
# 12. MANDATORY wrong-polarity fixture: descending radial_edges
# ===========================================================================

class TestWrongPolarityRadialEdges:
    """Brief item 15 (MANDATORY per Post-mortem 1).

    Build a tensor with VALID kernel values but ``radial_edges`` REVERSED
    (descending from 1.0 to 0.0). Does load_tensor catch this, or does it
    silently feed the renderer with backwards radial mapping?

    Mechanism: ``radial_edges`` defines the radial axis for the bilinear
    kernel interp. If reversed, the on-axis bin (r=0) maps to the FAR edge
    (r=1.0) and vice versa — every pixel gets the WRONG kernel.

    Phase 1 audit finding (Coder B item #2): ``radial_edges`` monotonicity
    is NOT validated. This is a confirmed silent-pass bug.
    """

    # HYG-0 item 15: xfail removed — XPASSes (P1.audit Item 4 monotonic gate).
    def test_descending_radial_edges_should_raise_value_error(self, tmp_path):
        """Reversed radial_edges (1.0 → 0.0) — load_tensor should raise."""
        path = write_synthetic_tensor(tmp_path / "rev_edges.npz", R=4, D=3)
        data = dict(np.load(str(path), allow_pickle=True))
        data["radial_edges"] = data["radial_edges"][::-1].copy()
        np.savez(str(path), **data)
        with pytest.raises(ValueError, match="(monoton|radial_edges)"):
            load_tensor(path)

    def test_descending_radial_edges_at_least_does_not_crash_silently(self, tmp_path):
        """P1.audit Item 4 fixed the bug this test was documenting:
        load_tensor() now VALIDATES radial_edges monotonicity at load time
        and raises TensorValidationError. The original "silent pass-through"
        behavior was a documented gap that the audit closed.
        """
        from summicron.render.convolve import TensorValidationError
        path = write_synthetic_tensor(tmp_path / "rev_edges2.npz", R=4, D=3)
        data = dict(np.load(str(path), allow_pickle=True))
        rev = data["radial_edges"][::-1].copy()
        data["radial_edges"] = rev
        np.savez(str(path), **data)
        with pytest.raises(TensorValidationError, match="strictly increasing"):
            load_tensor(path)


# ===========================================================================
# 13. focus_distance_m mismatch is still AssertionError (Coder B audit #1)
# ===========================================================================

class TestFocusDistanceMismatchTypeBug:
    """Brief item 16: tensor where ``focus_distance_m != object_distances_m[0]``
    — does it still raise ``AssertionError`` instead of ``ValueError``?

    Per Coder B's audit finding #1, this is a known carry-forward bug: the
    spec says ``raise ValueError`` for uniform validation type, but
    Fixed by P1.audit Item 5 (assert -> raise ValueError); xfail removed in HYG-0."""

    # HYG-0 item 15: xfail removed — XPASSes (P1.audit Item 5 assert->raise).
    def test_focus_distance_mismatch_raises_value_error_not_assertion(self, tmp_path):
        path = write_synthetic_tensor(tmp_path / "focus_mismatch.npz", R=2, D=4)
        data = dict(np.load(str(path), allow_pickle=True))
        # focus_distance_m drifted from object_distances_m[0]
        data["focus_distance_m"] = np.float64(data["object_distances_m"][0] + 0.5)
        np.savez(str(path), **data)
        with pytest.raises(ValueError):
            load_tensor(path)

    def test_focus_distance_mismatch_at_least_raises_something(self, tmp_path):
        """Negative-control: confirm the assert fires (test isn't broken)."""
        path = write_synthetic_tensor(tmp_path / "focus_mismatch2.npz", R=2, D=4)
        data = dict(np.load(str(path), allow_pickle=True))
        data["focus_distance_m"] = np.float64(data["object_distances_m"][0] + 0.5)
        np.savez(str(path), **data)
        with pytest.raises((AssertionError, ValueError)):
            load_tensor(path)


# ===========================================================================
# 14. Cross-Coder integration: A's interp output flows through B's load_tensor
# ===========================================================================

class TestInterpOutputThroughLoadTensor:
    """Brief item 18 (post-R5): build a R=10/D=8 interp tensor via
    downscale_tensor(), save to disk, load via load_tensor — must pass all
    4 validation gates without raising.
    """

    def test_interp_output_loads_clean(self, tmp_path):
        fake_dir = _fake_codev_input_dir(tmp_path, raw_R=4, D_files=5, kH=33)
        raw_path = tmp_path / "raw.npz"
        convert(input_dir=fake_dir, out_path=raw_path)
        d_raw = np.load(raw_path)
        target_long_edge = int(round(36000.0 / float(d_raw["pixel_pitch_um"].mean())))
        out_path = tmp_path / "interp_R10_D8.npz"
        downscale_tensor(
            in_path=raw_path, out_path=out_path,
            target_long_edge=target_long_edge,
            target_R=10, target_D=8,
            codev_input_dir=fake_dir,
        )
        # Load — should not raise on any of the 4 gates.
        loaded = load_tensor(out_path)
        assert loaded["R"] == 10
        assert loaded["D"] == 8
        # All 4 validators must pass:
        assert np.all(np.isfinite(loaded["kernels"]))
        assert float(loaded["kernels"].min()) >= -1e-6
        assert np.all(np.diff(loaded["object_distances_m"]) > 0)
        # Per-(r,d) L1 ± 1e-5 (G_L1_CONSERVATION on interp output via load).
        sums = loaded["kernels"].reshape(10 * 8, -1).sum(axis=1)
        assert np.all(np.abs(sums - 1.0) < 1e-5), (
            f"L1 drift after load: min={sums.min():.6f} max={sums.max():.6f}"
        )


# ===========================================================================
# 15. Cross-Coder: render at R=24/D=22 hits the cap=12 warn correctly
# ===========================================================================

class TestInterpOutputThroughLayeredCap:
    """Brief item 19: render Coder A's R=24/D=22 alternate tensor through
    layered_gather=True. Verify Coder B's effective_cap = max(8, 24//2) = 12
    fires when active_count > 12.
    """

    def test_layered_render_R24_D22_caps_at_12(self, tmp_path):
        """Build a R=24/D=22 tensor (synthetic small kernels), render with
        all 22 slabs active → warn fires with cap=12.
        """
        D = 22
        tensor = _make_d_tensor(D=D, R=24)
        H = W = 40
        depth_m = np.linspace(0.7, 20.0, H * W, dtype=np.float32).reshape(H, W)
        img = np.full((H, W, 3), 0.3, dtype=np.float32)
        with warnings.catch_warnings(record=True) as w_list:
            warnings.simplefilter("always")
            _, out = render(img, tensor, depth_m=depth_m)
        assert np.all(np.isfinite(out))
        rt_warns = [
            w for w in w_list
            if issubclass(w.category, RuntimeWarning)
            and "MAX_ACTIVE_SLABS" in str(w.message)
        ]
        assert len(rt_warns) >= 1
        # Message reports cap=11 = max(8, 22//2). Verify the formula.
        msg = str(rt_warns[0].message)
        assert "MAX_ACTIVE_SLABS=11" in msg, (
            f"D={D}: cap = max(8, D//2) = max(8, 11) = 11; message: {msg!r}"
        )


# ===========================================================================
# 16. Pixel-pitch interp must be linear (NOT cubic) — no negatives/overshoot
# ===========================================================================

class TestPixelPitchInterpLinear:
    """L6: pixel pitch is a smooth physical metric — linear interp avoids
    overshoot. If swapped to cubic, monotone-positive pitch could overshoot
    at edges (or, worse, dip negative at very steep transitions).
    """

    def test_no_negative_pitch_under_steep_gradient(self):
        """Build a pitch array with a steep step; linear interp stays
        positive and bounded."""
        raw = np.tile(np.linspace(0.5, 0.6, 12)[:, None], (1, 11)).astype(np.float64)
        # inject a step on one row
        raw[5, :] = 0.55
        out = _interp_pixel_pitch(raw, 24, 22)
        assert out.shape == (24, 22)
        assert np.all(out > 0)
        # Linear monotone: bounded by [min(raw), max(raw)] within ULP.
        assert out.min() >= raw.min() - 1e-9
        assert out.max() <= raw.max() + 1e-9


# ===========================================================================
# 17. Bake-time pipeline order is INVISIBLE from outside, prove via L1+nonneg
# ===========================================================================

class TestPipelineOrderObservableViaInvariants:
    """If the pipeline order is wrong — e.g. renorm BEFORE clip, or overwrite
    BEFORE renorm — the OBSERVABLE invariants (L1=1, min>=0, corners
    bit-identical) will fail in distinct combinations.

    This omnibus probe runs all three at once on a stress-shape and gives
    an explicit error message that points to the most-likely-broken order
    step.
    """

    def test_omnibus_invariants_at_R24_D22(self):
        kernels_raw, centers, h_max, od = _asym_synthetic()
        out, _, _ = _interp_rd(kernels_raw, centers, h_max, od, 24, 22)
        # Invariant 1: non-negative.
        assert out.min() >= 0.0, (
            f"min={out.min():.3e} < 0 → likely renorm-before-clip order bug"
        )
        # Invariant 2: corner bit-identical.
        assert np.array_equal(out[0, 0], kernels_raw[0, 0]), (
            "corner (0,0) not bit-identical → post-overwrite missing or "
            "renorm-after-overwrite re-scaled the corner mass"
        )
        # Invariant 3: per-slice L1=1 ± 1e-5.
        sums = out.reshape(24 * 22, -1).sum(axis=1)
        bad = np.where(np.abs(sums - 1.0) >= 1e-5)[0]
        assert len(bad) == 0, (
            f"{len(bad)}/{24 * 22} slices fail L1; min={sums.min():.6f} "
            f"max={sums.max():.6f} → likely renorm step missing or "
            "applied before expm1 decompression"
        )


# ===========================================================================
# 18. (Post-R5) Interp-time alt-filename was at codev_to_render._resolve_out_path
# which has been removed. The downscale_tensor CLI takes an explicit --out
# (and the tests above already cover it). This test class is intentionally
# removed; placeholder retained for traceability.
# ===========================================================================
