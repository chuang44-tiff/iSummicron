"""P1.9.4 unit tests for the R/D interp pipeline in ``downscale_tensor``.

Gates covered (per locked spec section 14, post-R5 PIVOT):
- G_R_QUADRATURE — R-axis sqrt source axis preserved (also pinned by
  ``test_codev_raw_contract.py`` on the raw .npz; here on interp output).
- G_ENDPOINT_PRESERVING — raw nodes bit-identical at all 4 corners after
  post-overwrite.
- G_D_AXIS_NO_OVERSHOOT — D-axis pinned linear; min ≥ 0 and max ≤ raw
  peak * 1.001 per slab.
- G_L1_CONSERVATION — per-(r, d) slice L1 sum = 1.0 ± 1e-5 on the
  CodeV-derived interpolated downscaled tensor.
- G_NO_OP_IDENTITY (L4 / G14 / G5) — content equality (per-key array_equal,
  HYG-0 item 12) with the wheel-shipped canonical tensor at default invocation.

R5 PIVOT (2026-04-25): interp helpers moved from ``codev_to_render`` to
``downscale_tensor`` and operate on the post-downscale ~44/183-grid
kernels rather than the raw 2048 grid (~2000× cheaper). Algorithmic
locks unchanged; tests follow the helper to its new module home.

To keep wall time reasonable, most tests build a small-kernel synthetic
tensor and invoke ``_interp_rd`` directly. The G_NO_OP_IDENTITY gate
runs the full ``downscale_tensor`` pipeline at 5712 long-edge.
"""
from __future__ import annotations

import os
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
    _interp_rd,
    downscale_tensor,
)

ROOT = Path(__file__).resolve().parents[3]
RAW_NPZ = ROOT / "runs" / "ag_dblgauss" / "ag_dblgauss_f2.0_focus0.70m_d00_0.70m.npz"
DEFAULT_TENSOR = ROOT / "runs" / "ag_dblgauss" / "ag_dblgauss_f2.0_render.npz"
IPHONE_TENSOR = default_tensor_path()

SLOW = pytest.mark.slow


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


# Real-data tests at 2048-grid take 30-60s each; mark slow.
# Tests using synthetic small kernels (kH=33) are fast (<1s).


# -----------------------------------------------------------------------------
# Synthetic fixture — small-kernel tensor that exercises the interp pipeline.
# -----------------------------------------------------------------------------
def _make_synthetic(raw_R: int = 12, raw_D: int = 11, kH: int = 33, seed: int = 7):
    """Build a small synthetic (raw_R, raw_D, kH, kH) L1-normed tensor.

    Each (r, d) slice is a Gaussian whose width grows with r and d (mimics
    PSF-spread-with-defocus). Returns the kernel array plus the sqrt-shaped
    centers and geomspace distances that match the raw CodeV layout.
    """
    rng = np.random.default_rng(seed)
    h_max_mm = 21.6
    radial_bin_centers_mm = h_max_mm * np.sqrt(np.linspace(0.0, 0.954, raw_R))
    object_distances_m = np.geomspace(0.7, 20.0, raw_D)

    kernels = np.zeros((raw_R, raw_D, kH, kH), dtype=np.float32)
    cy = cx = kH // 2
    yy, xx = np.mgrid[:kH, :kH]
    for ri in range(raw_R):
        for di in range(raw_D):
            sigma = 0.8 + 0.5 * ri / max(1, raw_R - 1) + 1.5 * di / max(1, raw_D - 1)
            # add tiny radial-asymmetry so corner-overwrite isn't trivially equal
            jit = 0.05 * rng.standard_normal()
            k = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2 * (1 + jit)) / (2 * sigma ** 2))
            k /= k.sum()
            kernels[ri, di] = k.astype(np.float32)
    return kernels, radial_bin_centers_mm, h_max_mm, object_distances_m


# =============================================================================
# G_NO_OP_IDENTITY — canonical tensors content must remain stable (HYG-0: per-key
# array_equal; was byte-md5).
#
# Post-R5-pivot:
#   - codev_to_render no longer interps; ``ag_dblgauss_f2.0_render.npz``
#     content is preserved by ``convert()`` running a clean P1.9.3 path.
#   - downscale_tensor at default (target_R=None, target_D=None) is
#     content-identical to the wheel-shipped canonical tensor.
# =============================================================================
class TestNoOpIdentity:
    @pytest.mark.skipif(
        not DEFAULT_TENSOR.exists(),
        reason="committed default tensor absent; skip content stability gate",
    )
    @SLOW
    def test_codev_default_md5_byte_identity(self, tmp_path):
        """Default convert() must produce content-identical output to committed
        raw tensor (codev_to_render no longer interps post-R5)."""
        out = tmp_path / "noop_default.npz"
        convert(input_dir=DEFAULT_INPUT_DIR, out_path=out)
        _assert_npz_content_equal(DEFAULT_TENSOR, out)

    @pytest.mark.skipif(
        not IPHONE_TENSOR.exists(),
        reason="committed iPhone tensor absent; skip content stability gate",
    )
    @SLOW
    def test_downscale_default_md5_byte_identity(self, tmp_path):
        """Default downscale_tensor() (target_R=None, target_D=None) must
        produce content-identical output to the wheel-shipped tensor."""
        out = tmp_path / "iphone_default.npz"
        downscale_tensor(
            in_path=DEFAULT_TENSOR, out_path=out, target_long_edge=5712
        )
        _assert_npz_content_equal(IPHONE_TENSOR, out)

    @pytest.mark.skipif(
        not IPHONE_TENSOR.exists(),
        reason="committed iPhone tensor absent; skip content stability gate",
    )
    @SLOW
    def test_downscale_explicit_raw_targets_md5_byte_identity(self, tmp_path):
        """--target-R 12 --target-D 11 (== raw shape) → content-identical too.

        Note: explicit ``target_R=R_raw`` (= 12) takes the is_no_op branch,
        which avoids running interp and preserves content identity.
        """
        out = tmp_path / "iphone_explicit.npz"
        downscale_tensor(
            in_path=DEFAULT_TENSOR, out_path=out, target_long_edge=5712,
            target_R=12, target_D=11,
        )
        _assert_npz_content_equal(IPHONE_TENSOR, out)


# =============================================================================
# G_ENDPOINT_PRESERVING — corners byte-identical after post-overwrite.
# =============================================================================
class TestEndpointPreserving:
    """After RGI + post-overwrite, raw kernels at the 4 corner nodes
    (r=0/-1 × d=0/-1) must be bit-identical to ``kernels_raw[corner]``."""

    @pytest.mark.parametrize(
        "target_R,target_D",
        [(24, 11), (12, 22), (24, 22), (36, 11), (12, 33)],
    )
    def test_corners_bit_identical_synthetic(self, target_R, target_D):
        kernels_raw, centers, h_max, od = _make_synthetic()
        raw_R, raw_D = kernels_raw.shape[:2]
        kernels_interp, _, _ = _interp_rd(
            kernels_raw, centers, h_max, od, target_R, target_D
        )
        # 4 corners (r=0,-1 × d=0,-1).
        corners = [(0, 0), (0, raw_D - 1), (raw_R - 1, 0), (raw_R - 1, raw_D - 1)]
        target_corners = [(0, 0), (0, target_D - 1),
                          (target_R - 1, 0), (target_R - 1, target_D - 1)]
        for (r_raw, d_raw), (r_t, d_t) in zip(corners, target_corners):
            assert np.array_equal(
                kernels_interp[r_t, d_t], kernels_raw[r_raw, d_raw]
            ), f"Corner ({r_raw},{d_raw})→({r_t},{d_t}) not byte-identical"

    def test_no_op_synthetic_kernels_unchanged(self):
        """target_R=raw_R, target_D=raw_D should still call _interp_rd
        and produce kernels equal to raw at every (r,d) — node-overwrite
        means every cell is a raw node."""
        kernels_raw, centers, h_max, od = _make_synthetic()
        raw_R, raw_D = kernels_raw.shape[:2]
        kernels_interp, _, _ = _interp_rd(
            kernels_raw, centers, h_max, od, raw_R, raw_D
        )
        # Every cell is a raw node → post-overwrite makes it byte-identical.
        for r in range(raw_R):
            for d in range(raw_D):
                assert np.array_equal(kernels_interp[r, d], kernels_raw[r, d])


# =============================================================================
# G_D_AXIS_NO_OVERSHOOT — linear D-axis must be monotone-safe.
# =============================================================================
class TestDAxisNoOvershoot:
    @pytest.mark.parametrize("target_D", [11, 22, 24, 33])
    def test_d_axis_no_negatives_synthetic(self, target_D):
        """D-axis linear interp produces no negative pixels post-clip."""
        kernels_raw, centers, h_max, od = _make_synthetic()
        kernels_interp, _, _ = _interp_rd(
            kernels_raw, centers, h_max, od, 12, target_D
        )
        assert kernels_interp.min() >= 0.0, (
            f"D={target_D}: min={kernels_interp.min():.3e} (linear should be >=0)"
        )

    @pytest.mark.parametrize("target_D", [22, 33])
    def test_d_axis_no_overshoot_synthetic(self, target_D):
        """Per-slab peak ≤ raw_peak * 1.001 (no Gibbs ringing on linear)."""
        kernels_raw, centers, h_max, od = _make_synthetic()
        raw_peak = float(kernels_raw.max())
        kernels_interp, _, _ = _interp_rd(
            kernels_raw, centers, h_max, od, 12, target_D
        )
        assert kernels_interp.max() <= raw_peak * 1.001, (
            f"D={target_D}: max={kernels_interp.max():.3e} > raw_peak*1.001="
            f"{raw_peak * 1.001:.3e}"
        )


# =============================================================================
# G_L1_CONSERVATION — per-(r, d) L1 sum = 1.0 ± 1e-5.
# =============================================================================
class TestL1Conservation:
    @pytest.mark.parametrize(
        "target_R,target_D",
        [(24, 11), (12, 22), (24, 22), (36, 11), (12, 33)],
    )
    def test_l1_per_slice_synthetic(self, target_R, target_D):
        kernels_raw, centers, h_max, od = _make_synthetic()
        kernels_interp, _, _ = _interp_rd(
            kernels_raw, centers, h_max, od, target_R, target_D
        )
        sums = kernels_interp.reshape(target_R * target_D, -1).sum(axis=1)
        assert np.all(np.abs(sums - 1.0) < 1e-5), (
            f"R={target_R} D={target_D}: L1 drift; "
            f"min={sums.min():.6f} max={sums.max():.6f}"
        )

    @pytest.mark.skipif(
        not DEFAULT_TENSOR.exists(),
        reason="committed raw tensor absent",
    )
    @pytest.mark.skipif(
        os.environ.get("ISUMMICRON_SKIP_FULL_RES_BAKE") == "1",
        reason="ISUMMICRON_SKIP_FULL_RES_BAKE=1 set; skip R=24/D=22 bake",
    )
    @SLOW
    def test_l1_real_data_R24_D22(self, tmp_path):
        """G_L1_CONSERVATION on the REAL CodeV-derived (24,22) interp tensor.

        Threshold = 1.0 ± 1e-5 on every (r, d) slice. Without the per-(r,d)
        L1 renorm step (L22), midpoint slices drop to L1 ≈ 0.90.

        R5 PIVOT: interp now lives at the post-downscale stage, so the
        bake is FAST (~50s at kH=183 instead of ~71min at kH=2048).
        """
        out = tmp_path / "real_R24_D22_iphone5712.npz"
        downscale_tensor(
            in_path=DEFAULT_TENSOR,
            out_path=out,
            target_long_edge=5712,
            target_R=24,
            target_D=22,
        )
        d = np.load(out)
        k = d["kernels"]
        assert k.shape[:2] == (24, 22)
        # iPhone-grid kH ~ 183.
        sums = k.reshape(24 * 22, -1).sum(axis=1)
        assert np.all(np.abs(sums - 1.0) < 1e-5), (
            f"L1 drift on real (24,22): min={sums.min():.6f} max={sums.max():.6f}"
        )


# =============================================================================
# G_R_QUADRATURE on interp output — query points lie inside source range
# and follow sqrt density.
# =============================================================================
class TestRQuadratureInterp:
    @pytest.mark.parametrize("target_R", [12, 24, 36])
    def test_r_interp_inside_source_range_synthetic(self, target_R):
        """L23 runtime assert — r_interp_mm in [centers[0], centers[-1]]."""
        kernels_raw, centers, h_max, od = _make_synthetic()
        # If query points fell outside, _interp_rd would assert/raise.
        kernels_interp, r_interp_mm, _ = _interp_rd(
            kernels_raw, centers, h_max, od, target_R, 11
        )
        assert r_interp_mm[0] == centers[0]
        assert r_interp_mm[-1] == centers[-1]
        assert np.all(r_interp_mm >= centers[0])
        assert np.all(r_interp_mm <= centers[-1])

    @pytest.mark.parametrize("target_R", [24, 36])
    def test_r_target_is_sqrt_quadrature_synthetic(self, target_R):
        """r_interp_mm follows sqrt(linspace(0, (centers[-1]/h_max)**2, N))."""
        kernels_raw, centers, h_max, od = _make_synthetic()
        _, r_interp_mm, _ = _interp_rd(
            kernels_raw, centers, h_max, od, target_R, 11
        )
        expected = h_max * np.sqrt(
            np.linspace(0.0, (centers[-1] / h_max) ** 2, target_R)
        )
        assert np.allclose(r_interp_mm, expected, atol=1e-12)


# =============================================================================
# Ancillary: object_distances_m geomspace, defocus_ratios linspace,
# focus_distance_m endpoint, alt filename, asymmetric flags.
# =============================================================================
class TestAxisMetadata:
    def test_geomspace_distances_synthetic(self):
        kernels_raw, centers, h_max, od = _make_synthetic()
        _, _, od_interp = _interp_rd(
            kernels_raw, centers, h_max, od, 12, 22
        )
        # geomspace → constant log-step.
        log_step = np.diff(np.log(od_interp))
        assert np.allclose(log_step, log_step[0], atol=1e-9)
        # endpoints exact.
        assert od_interp[0] == od[0]
        assert od_interp[-1] == od[-1]


class TestDownscaleCLIBranches:
    """Exercise the public ``downscale_tensor()`` at small-kernel scale.

    Post-R5 PIVOT: interp lives in downscale_tensor. We build a fake CodeV
    input dir, run codev_to_render to produce a small raw tensor, then
    invoke downscale_tensor with various R/D targets to exercise the
    interp branch.
    """

    @pytest.fixture(scope="class")
    def fake_codev_dir(self, tmp_path_factory):
        """Build a fake CodeV input dir with 5 distance files, 4×4 R, 33×33 PSFs."""
        d = tmp_path_factory.mktemp("fake_codev")
        raw_R = 4
        D_files = 5
        kH = 33
        h_max_mm = 21.6
        radial_bin_centers_mm = h_max_mm * np.sqrt(np.linspace(0.0, 0.954, raw_R))
        radial_bin_edges_hmax = np.sqrt(np.linspace(0.0, 1.0, raw_R))
        object_distances_m = np.geomspace(0.7, 20.0, D_files)
        wavelengths_nm = np.array([450.0, 550.0, 650.0])
        pixel_pitch_um = np.tile(
            np.linspace(0.5, 0.55, raw_R)[:, None], (1, 3)
        ).astype(np.float64)

        for di, od in enumerate(object_distances_m):
            payload = {
                "distance_index": np.int64(di),
                "object_distance_m": np.float64(od),
                "object_distances_m": object_distances_m.astype(np.float64),
                "focus_distance_m": np.float64(object_distances_m[0]),
                "h_max_mm": np.float64(h_max_mm),
                "wavelengths_nm": wavelengths_nm,
                "radial_bin_edges_hmax": radial_bin_edges_hmax,
                "radial_bin_centers_mm": radial_bin_centers_mm,
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

            fname = f"ag_dblgauss_f2.0_focus0.70m_d{di:02d}_{od:.2f}m.npz"
            np.savez(str(d / fname), **payload)
        return d

    @pytest.fixture(scope="class")
    def fake_raw_tensor(self, fake_codev_dir, tmp_path_factory):
        """Build a raw tensor from the fake CodeV dir (codev_to_render).

        kH=33 → for the downscale to be a no-op (target_kH=raw_kH) we set
        ``target_long_edge`` so 36000/target_long_edge ≈ pitch_mean ≈ 0.525,
        which gives target_long_edge ≈ 68571. Use that.
        """
        d = tmp_path_factory.mktemp("fake_raw")
        out = d / "fake_render.npz"
        convert(input_dir=fake_codev_dir, out_path=out)
        return out

    def test_no_op_downscale(self, fake_codev_dir, fake_raw_tensor, tmp_path):
        """Default downscale_tensor on fake raw tensor — shape preserved."""
        # Pick target_long_edge so target_kH equals raw_kH (no real downscale).
        # kH=33, mean_pitch ≈ 0.525 → ~ 36000 / 0.525 = ~68571.
        d_raw = np.load(fake_raw_tensor)
        target_long_edge = int(round(36000.0 / float(d_raw["pixel_pitch_um"].mean())))
        out = tmp_path / "fake_iphone.npz"
        downscale_tensor(
            in_path=fake_raw_tensor, out_path=out,
            target_long_edge=target_long_edge,
            codev_input_dir=fake_codev_dir,
        )
        d = np.load(out)
        # Raw R = len(edges) - 1 = 4; D = 5 distance files.
        assert d["kernels"].shape[:2] == (4, 5)

    def test_interp_R_only(self, fake_codev_dir, fake_raw_tensor, tmp_path):
        """Asymmetric: R only (per L9)."""
        d_raw = np.load(fake_raw_tensor)
        target_long_edge = int(round(36000.0 / float(d_raw["pixel_pitch_um"].mean())))
        out = tmp_path / "fake_R10_D5.npz"
        downscale_tensor(
            in_path=fake_raw_tensor, out_path=out,
            target_long_edge=target_long_edge,
            target_R=10, target_D=5,
            codev_input_dir=fake_codev_dir,
        )
        d = np.load(out)
        assert d["kernels"].shape[:2] == (10, 5)
        # G_L1_CONSERVATION
        sums = d["kernels"].reshape(10 * 5, -1).sum(axis=1)
        assert np.all(np.abs(sums - 1.0) < 1e-5)

    def test_interp_D_only(self, fake_codev_dir, fake_raw_tensor, tmp_path):
        """Asymmetric: D only (per L9). Geomspace check too."""
        d_raw = np.load(fake_raw_tensor)
        target_long_edge = int(round(36000.0 / float(d_raw["pixel_pitch_um"].mean())))
        out = tmp_path / "fake_R4_D10.npz"
        downscale_tensor(
            in_path=fake_raw_tensor, out_path=out,
            target_long_edge=target_long_edge,
            target_R=4, target_D=10,
            codev_input_dir=fake_codev_dir,
        )
        d = np.load(out)
        assert d["kernels"].shape[:2] == (4, 10)
        # geomspace lock (G2).
        od_int = d["object_distances_m"]
        assert len(od_int) == 10
        log_step = np.diff(np.log(od_int))
        assert np.allclose(log_step, log_step[0], atol=1e-9)
        # defocus_ratios linspace (G3).
        dr = d["defocus_ratios"]
        assert dr[0] == 0.0 and dr[-1] == 1.0
        assert np.allclose(np.diff(dr), np.diff(dr)[0], atol=1e-9)
        # focus_distance_m endpoint (G6 / L17).
        assert np.isclose(float(d["focus_distance_m"]), float(od_int[0]), rtol=1e-9)

    def test_interp_both_axes(self, fake_codev_dir, fake_raw_tensor, tmp_path):
        """Both R and D densified."""
        d_raw = np.load(fake_raw_tensor)
        target_long_edge = int(round(36000.0 / float(d_raw["pixel_pitch_um"].mean())))
        out = tmp_path / "fake_R10_D10.npz"
        downscale_tensor(
            in_path=fake_raw_tensor, out_path=out,
            target_long_edge=target_long_edge,
            target_R=10, target_D=10,
            codev_input_dir=fake_codev_dir,
        )
        d = np.load(out)
        assert d["kernels"].shape[:2] == (10, 10)
        assert d["pixel_pitch_um"].shape == (10, 10)
        assert d["full_frame_dim_px"].shape == (10, 10, 2)

    def test_radial_edges_length_on_interp(
        self, fake_codev_dir, fake_raw_tensor, tmp_path
    ):
        """L5 — radial_edges length = target_R + 1 (post-midpoint-split)."""
        d_raw = np.load(fake_raw_tensor)
        target_long_edge = int(round(36000.0 / float(d_raw["pixel_pitch_um"].mean())))
        out = tmp_path / "fake_R10_D5.npz"
        downscale_tensor(
            in_path=fake_raw_tensor, out_path=out,
            target_long_edge=target_long_edge,
            target_R=10, target_D=5,
            codev_input_dir=fake_codev_dir,
        )
        d = np.load(out)
        assert len(d["radial_edges"]) == 10 + 1
        assert np.all(np.diff(d["radial_edges"]) > 0)
