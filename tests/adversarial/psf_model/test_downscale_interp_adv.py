"""P1.9.4 adversarial tests for the R/D interp pipeline (post-R5 PIVOT).

This is the **pre-write** Adversarial set. R5 PIVOT relocated the interp
helpers from ``codev_to_render`` to ``downscale_tensor`` (operating on
post-downscale ~44/183-grid kernels, not raw 2048-grid). Algorithmic
locks unchanged; tests follow.

- Wrong-polarity targets (target_R=1, target_D=1 — RGI cubic needs ≥2
  source points for monotone behavior; we accept that ValueError-or-
  RuntimeError-or-similar fires; the user-visible semantic is "renderer
  cannot operate at R<2 / D<2 — fail loud").
- target_R = raw_R + 1 (densify by exactly one bin — boundary case for
  the corner-overwrite logic).
- target = 2 (minimal valid densification — both endpoints overwritten).
- pathological mem (G7 — R=4000/D=4000 at kH=183 must trigger guard).
- no-op content identity (G5 — already covered in unit tests; replicated
  here as the adversarial confidence check; HYG-0: was byte-md5).
- Non-monotone object_distances → handled upstream (raw .npz contract);
  not triggered by interp.
"""
from __future__ import annotations

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
    DEFAULT_INPUT as DOWNSCALE_DEFAULT_INPUT,
    MEM_THRESHOLD_GB_DEFAULT,
    _interp_pixel_pitch,
    _interp_rd,
    downscale_tensor,
)

ROOT = Path(__file__).resolve().parents[3]
IPHONE_TENSOR = default_tensor_path()


def _assert_npz_content_equal(a_path, b_path, ignore=frozenset({"otf_norm_applied"})):
    """Same key set (modulo ``ignore``) and per-key ``np.array_equal`` (HYG-0 item 12)."""
    with np.load(str(a_path)) as a, np.load(str(b_path)) as b:
        ka, kb = set(a.files) - ignore, set(b.files) - ignore
        assert ka == kb, f"key sets differ: only-a={sorted(ka - kb)} only-b={sorted(kb - ka)}"
        bad = [
            k for k in sorted(ka)
            if a[k].dtype != b[k].dtype or not np.array_equal(a[k], b[k])
        ]
        assert not bad, f"content differs at keys {bad} ({a_path} vs {b_path})"


def _make_synthetic(raw_R: int = 12, raw_D: int = 11, kH: int = 33):
    h_max_mm = 21.6
    centers = h_max_mm * np.sqrt(np.linspace(0.0, 0.954, raw_R))
    od = np.geomspace(0.7, 20.0, raw_D)
    yy, xx = np.mgrid[:kH, :kH]
    cy = cx = kH // 2
    kernels = np.zeros((raw_R, raw_D, kH, kH), dtype=np.float32)
    for ri in range(raw_R):
        for di in range(raw_D):
            sigma = 0.8 + 0.4 * ri / max(1, raw_R - 1) + 1.5 * di / max(1, raw_D - 1)
            k = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
            k /= k.sum()
            kernels[ri, di] = k.astype(np.float32)
    return kernels, centers, h_max_mm, od


# =============================================================================
# G7 — memory guard fires on pathological R/D and is escapable via override.
# (Post-R5: kH=183 at iPhone-grid; the formula scales as R*D*kH²*4*5;
# need much larger R/D to exceed 8 GB threshold than at kH=2048.)
# =============================================================================
class TestMemoryGuardG7:
    @pytest.mark.skipif(
        not IPHONE_TENSOR.exists(),
        reason="committed iPhone tensor absent",
    )
    def test_pathological_R_D_raises(self):
        """R=4000, D=4000 at kH=183 → ~10.7 TB → must raise.

        Post-pivot: at the post-downscale kH≈183 the bytes formula
        scales much more gracefully (~9.8 GB at R=4000, D=400). To
        push above the 8 GB default we need pathological R AND D.
        """
        with pytest.raises(RuntimeError, match=r"would allocate.*GB transient"):
            downscale_tensor(
                in_path=IPHONE_TENSOR,
                out_path=Path("/tmp/_downscale_path_block.npz"),
                target_long_edge=5712,
                target_R=4000,
                target_D=4000,
                mem_threshold_gb=MEM_THRESHOLD_GB_DEFAULT,
            )

    def test_default_threshold_value_is_8_gb(self):
        """Spec L15: default threshold is 8.0 GB."""
        assert MEM_THRESHOLD_GB_DEFAULT == 8.0


# =============================================================================
# Boundary targets — minimal/+1 densification must not crash.
# =============================================================================
class TestBoundaryTargets:
    def test_target_R_equals_raw_plus_1(self):
        """target_R = raw_R + 1 (densify by one bin) — ensure post-overwrite
        logic still bit-preserves the 4 corners."""
        kernels_raw, centers, h_max, od = _make_synthetic(raw_R=12, raw_D=11)
        out, _, _ = _interp_rd(kernels_raw, centers, h_max, od, 13, 11)
        assert out.shape == (13, 11, 33, 33)
        # corners — note: target_R indices for raw r=0,11 are 0 and 12.
        assert np.array_equal(out[0, 0], kernels_raw[0, 0])
        assert np.array_equal(out[12, 0], kernels_raw[11, 0])
        assert np.array_equal(out[0, 10], kernels_raw[0, 10])
        assert np.array_equal(out[12, 10], kernels_raw[11, 10])

    def test_target_D_equals_raw_plus_1(self):
        kernels_raw, centers, h_max, od = _make_synthetic()
        out, _, _ = _interp_rd(kernels_raw, centers, h_max, od, 12, 12)
        assert out.shape == (12, 12, 33, 33)

    def test_target_R_2(self):
        """Minimal R densification: only 2 R-bins (both raw endpoints)."""
        kernels_raw, centers, h_max, od = _make_synthetic()
        out, _, _ = _interp_rd(kernels_raw, centers, h_max, od, 2, 11)
        assert out.shape == (2, 11, 33, 33)
        # Both R targets are raw endpoint corners.
        assert np.array_equal(out[0, 0], kernels_raw[0, 0])
        assert np.array_equal(out[1, 0], kernels_raw[11, 0])

    def test_target_D_2(self):
        """Minimal D densification: only 2 D-bins (both raw endpoints)."""
        kernels_raw, centers, h_max, od = _make_synthetic()
        out, _, _ = _interp_rd(kernels_raw, centers, h_max, od, 12, 2)
        assert out.shape == (12, 2, 33, 33)
        # Both D targets are raw endpoint corners.
        assert np.array_equal(out[0, 0], kernels_raw[0, 0])
        assert np.array_equal(out[0, 1], kernels_raw[0, 10])


# =============================================================================
# G5 — no-op content identity (adversarial confidence replication; HYG-0
# item 12: per-key array_equal, was byte-md5).
# =============================================================================
class TestNoOpMd5Adversarial:
    @pytest.mark.skipif(
        not DEFAULT_OUTPUT.exists(),
        reason="committed default tensor absent",
    )
    @pytest.mark.slow
    def test_md5_byte_identity_default(self, tmp_path):
        out = tmp_path / "noop_adv.npz"
        convert(input_dir=DEFAULT_INPUT_DIR, out_path=out)
        _assert_npz_content_equal(DEFAULT_OUTPUT, out)


# =============================================================================
# Wrong-polarity / value-error guards on direct _interp_rd.
# =============================================================================
class TestWrongPolarity:
    def test_target_R_1_raises_or_degenerates(self):
        """RGI cubic requires ≥2 source points; target=1 is semantically
        invalid (no R-axis variation). Implementation may raise or
        produce a degenerate single-bin tensor; either way the unit must
        not silently corrupt downstream data."""
        kernels_raw, centers, h_max, od = _make_synthetic()
        # We accept any of (ValueError, RuntimeError, AssertionError) as
        # the failure mode for target_R=1. If interp succeeds, the output
        # must at least pass L1 (sum=1) for the single bin.
        try:
            out, _, _ = _interp_rd(kernels_raw, centers, h_max, od, 1, 11)
        except (ValueError, RuntimeError, AssertionError, IndexError):
            return
        # If it didn't raise, single-bin output must still be L1=1.
        assert out.shape == (1, 11, 33, 33)
        sums = out.reshape(11, -1).sum(axis=1)
        assert np.all(np.abs(sums - 1.0) < 1e-5)

    @pytest.mark.skipif(
        not IPHONE_TENSOR.exists(),
        reason="committed iPhone tensor absent",
    )
    def test_negative_target_R_in_downscale(self):
        """Negative target_R must not silently produce a tensor."""
        with pytest.raises((ValueError, RuntimeError, AssertionError)):
            downscale_tensor(
                in_path=IPHONE_TENSOR,
                out_path=Path("/tmp/_downscale_neg_R.npz"),
                target_long_edge=5712,
                target_R=-5,
                target_D=11,
            )


# =============================================================================
# Endpoint defensive — focus_distance_m == object_distances_m[0].
# =============================================================================
class TestFocusDistanceEndpointAdversarial:
    def test_focus_endpoint_holds_for_synthetic_geomspace(self):
        kernels_raw, centers, h_max, od = _make_synthetic()
        _, _, od_interp = _interp_rd(kernels_raw, centers, h_max, od, 12, 22)
        # geomspace endpoint preservation (L17).
        assert od_interp[0] == od[0]
        assert od_interp[-1] == od[-1]


# =============================================================================
# pixel_pitch interp linearity sanity.
# =============================================================================
class TestPixelPitchLinearAdversarial:
    def test_pitch_interp_no_negative(self, tmp_path):
        """L6 — order=1 zoom on monotone-positive pitch yields no negatives."""
        # raw pitch like CodeV (small variation, all positive).
        raw = 0.5 + 0.05 * np.arange(12 * 11, dtype=np.float64).reshape(12, 11) / 132
        out = _interp_pixel_pitch(raw, 24, 22)
        assert out.shape == (24, 22)
        assert np.all(out > 0)

    def test_pitch_interp_no_op_byte_identity(self):
        """target_R=raw_R, target_D=raw_D → exact copy."""
        raw = 0.5 + 0.05 * np.arange(12 * 11, dtype=np.float64).reshape(12, 11) / 132
        out = _interp_pixel_pitch(raw, 12, 11)
        assert np.array_equal(out, raw)
