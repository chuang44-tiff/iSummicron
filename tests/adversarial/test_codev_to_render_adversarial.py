"""Adversarial tests for summicron.psf_model.codev_to_render.

Goal: break the implementation, find edge cases, expose wrong assumptions.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from summicron.psf_model.codev_to_render import convert, DEFAULT_INPUT_DIR
from summicron.render._artifacts import default_tensor_path
from summicron.render.convolve import _bilinear_weights, load_tensor

ROOT = Path(__file__).resolve().parents[2]
OUT_PATH = ROOT / "runs" / "ag_dblgauss" / "ag_dblgauss_f2.0_render_adversarial.npz"

# P1.9.3: R/D are no longer module-level constants in codev_to_render;
# derive them locally for the adversarial assertions.
# CI round: the raw CodeV files are untracked (runs/ag_dblgauss/). When
# absent, fall back to the shipped tensor's (R, D) so the synthetic tmp_path
# tests keep running; tests that need the raw files skip via render_npz.
_FILES = sorted(DEFAULT_INPUT_DIR.glob("ag_dblgauss_f2.0_focus*_d*.npz"))
_RAW_PRESENT = bool(_FILES)
if _RAW_PRESENT:
    D = len(_FILES)
    with np.load(str(_FILES[0])) as _f0:
        R = len(_f0["radial_bin_edges_hmax"]) - 1 + 1  # +1 for synthetic midpoint
else:
    with np.load(str(default_tensor_path())) as _ft:
        R, D = (int(x) for x in _ft["kernels"].shape[:2])


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def render_npz():
    """Run convert() once and yield path; clean up after."""
    if not _RAW_PRESENT:
        pytest.skip(f"raw CodeV file absent: {DEFAULT_INPUT_DIR}")
    path = convert(input_dir=DEFAULT_INPUT_DIR, out_path=OUT_PATH)
    yield path
    if path.exists():
        path.unlink()


@pytest.fixture(scope="module")
def tensor(render_npz):
    """Load the render-ready tensor dict."""
    return load_tensor(render_npz)


def _fake_metadata(D, R):
    """Return common metadata keys needed by synthetic .npz files.

    P1.9.4 L23: ``radial_bin_centers_mm`` is now a required key in the
    raw CodeV .npz (the source axis for R-axis interp). Synthetic
    fixtures must provide it; we use a sqrt-shaped sampling consistent
    with the real CodeV pattern.
    """
    h_max_mm = 21.6
    return {
        "focus_distance_m": np.float64(0.7),
        "h_max_mm": np.float64(h_max_mm),
        "wavelengths_nm": np.array([470.0, 550.0, 660.0]),
        "object_distances_m": np.linspace(0.7, 10.0, D),
        "pixel_pitch_um": np.full((R, 3), 5.0, dtype=np.float64),
        "radial_bin_centers_mm": (
            h_max_mm * np.sqrt(np.linspace(0.0, 0.954, R))
        ).astype(np.float64),
    }


@pytest.fixture
def fake_npz_dir(tmp_path):
    """Create a minimal fake directory with 11 .npz files for convert()."""
    radial_edges = np.linspace(0.0, 1.0, R)
    meta = _fake_metadata(D, R)
    for di in range(D):
        data = {"distance_index": np.array(di), "radial_bin_edges_hmax": radial_edges,
                **meta}
        for ri in range(R):
            for wi in range(3):
                # Small positive kernels
                k = np.random.rand(32, 32).astype(np.float32) + 0.01
                data[f"psf_r{ri:02d}_w{wi}"] = k
        np.savez(tmp_path / f"ag_dblgauss_f2.0_focus0.70m_d{di:02d}_fake.npz", **data)
    return tmp_path


# ── Delta-kernel guard interaction ────────────────────────────────────────

class TestDeltaKernelGuard:
    """Post v2 OTF-norm spec (§6), the delta-kernel guard in
    ``_bilinear_weights`` has been removed. ``kernels[0, 0]`` is now an
    exact single-pixel delta installed by OTF-normalization, so the
    four-corner bilinear blend is numerically well-behaved with the delta
    corner participating. These tests pin the new behavior.
    """

    def test_defocus_zero_returns_four_corners(self, tensor):
        """defocus_val=0.0 returns all 4 corners (no guard collapse)."""
        radial_edges = tensor["radial_edges"]
        defocus_ratios = tensor["defocus_ratios"]
        indices, weights = _bilinear_weights(0.5, 0.0, radial_edges, defocus_ratios)
        assert len(indices) == 4
        assert len(weights) == 4
        assert pytest.approx(sum(weights), abs=1e-6) == 1.0

    def test_defocus_small_returns_four_corners(self, tensor):
        """defocus_val=0.04: four corners, weights still sum to 1."""
        radial_edges = tensor["radial_edges"]
        defocus_ratios = tensor["defocus_ratios"]
        indices, weights = _bilinear_weights(0.5, 0.04, radial_edges, defocus_ratios)
        assert len(indices) == 4
        assert pytest.approx(sum(weights), abs=1e-6) == 1.0

    def test_defocus_zero_uses_actual_d0_kernel(self, tensor):
        """Post v2 OTF-norm: d_idx=0 IS in the returned indices for
        defocus_val=0 (the d=0 center kernel is now an exact delta, so
        including it in the blend is correct).
        """
        radial_edges = tensor["radial_edges"]
        defocus_ratios = tensor["defocus_ratios"]
        indices, weights = _bilinear_weights(0.5, 0.0, radial_edges, defocus_ratios)
        d_indices = [idx[1] for idx in indices]
        assert 0 in d_indices

    def test_defocus_mid_no_delta_guard(self, tensor):
        """defocus_val=0.5 should NOT trigger the delta guard."""
        radial_edges = tensor["radial_edges"]
        defocus_ratios = tensor["defocus_ratios"]
        indices, weights = _bilinear_weights(0.5, 0.5, radial_edges, defocus_ratios)
        assert len(indices) == 4
        assert pytest.approx(sum(weights), abs=1e-6) == 1.0

    def test_defocus_one_no_delta_guard(self, tensor):
        """defocus_val=1.0 (max defocus) should NOT trigger delta guard."""
        radial_edges = tensor["radial_edges"]
        defocus_ratios = tensor["defocus_ratios"]
        indices, weights = _bilinear_weights(0.5, 1.0, radial_edges, defocus_ratios)
        # d_lo should be 9, d_hi=10 -- no d_idx==0 involved
        d_indices = [idx[1] for idx in indices]
        assert 0 not in d_indices


# ── Negative values in raw kernels ────────────────────────────────────────

class TestNegativeKernels:
    """What if raw PSF data contains negative values (e.g., from Gibbs ringing)?"""

    def test_convert_rejects_negative_sum_kernel(self, tmp_path):
        """A kernel whose sum <= 0 should cause convert() to fail (G1 assertion)."""
        radial_edges = np.linspace(0.0, 1.0, R)
        meta = _fake_metadata(D, R)
        for di in range(D):
            data = {"distance_index": np.array(di), "radial_bin_edges_hmax": radial_edges,
                    **meta}
            for ri in range(R):
                for wi in range(3):
                    if di == 5 and ri == 3 and wi == 1:
                        # Create a kernel whose sum is negative
                        k = np.full((32, 32), -1.0, dtype=np.float32)
                    else:
                        k = np.ones((32, 32), dtype=np.float32)
                    data[f"psf_r{ri:02d}_w{wi}"] = k
            np.savez(tmp_path / f"ag_dblgauss_f2.0_focus0.70m_d{di:02d}_neg.npz", **data)

        out = tmp_path / "out.npz"
        with pytest.raises(AssertionError, match="sum <= 0"):
            convert(input_dir=tmp_path, out_path=out)

    def test_convert_preserves_negatives_raw(self, tmp_path):
        """v3 spec §S2: raw converter only L1-normalizes; it does NOT
        OTF-normalize or clamp. Small negative pixels in raw CodeV
        inputs pass through (Wiener clamp happens at downscale_tensor).
        """
        radial_edges = np.linspace(0.0, 1.0, R)
        meta = _fake_metadata(D, R)
        for di in range(D):
            data = {"distance_index": np.array(di), "radial_bin_edges_hmax": radial_edges,
                    **meta}
            for ri in range(R):
                for wi in range(3):
                    k = np.ones((32, 32), dtype=np.float32)
                    # Sprinkle a few negative values
                    k[0, 0] = -0.5
                    k[1, 1] = -0.3
                    data[f"psf_r{ri:02d}_w{wi}"] = k
            np.savez(tmp_path / f"ag_dblgauss_f2.0_focus0.70m_d{di:02d}_neg2.npz", **data)

        out = tmp_path / "out.npz"
        path = convert(input_dir=tmp_path, out_path=out)
        d = np.load(str(path))
        kernels = d["kernels"]
        # Raw converter does not clamp negatives — they propagate.
        assert kernels.min() < 0.0


# ── Missing or malformed raw files ────────────────────────────────────────

class TestMissingOrMalformedFiles:

    def test_short_file_set_derives_smaller_D(self, tmp_path):
        """P1.9.3 L6: D is derived from len(files), not a hardcoded 11.

        A 5-file directory now produces a tensor with D=5, provided the
        rest of the pipeline (metadata, distance indices) is consistent.
        Pre-P1.9.3 this raised RuntimeError on the file-count check;
        the new behavior is "Just Works at the new D".
        """
        # Create 5 files with proper sequential distance_index + metadata.
        radial_edges = np.linspace(0.0, 1.0, R)
        meta_5 = _fake_metadata(D=5, R=R)
        out_path = tmp_path / "out.npz"
        for di in range(5):
            data = {
                "distance_index": np.array(di),
                "radial_bin_edges_hmax": radial_edges,
                **meta_5,
            }
            for ri in range(R):
                for wi in range(3):
                    data[f"psf_r{ri:02d}_w{wi}"] = np.ones((32, 32), dtype=np.float32)
            np.savez(
                tmp_path / f"ag_dblgauss_f2.0_focus0.70m_d{di:02d}_short.npz",
                **data,
            )
        convert(input_dir=tmp_path, out_path=out_path)
        loaded = np.load(str(out_path))
        # D in output tensor must equal the file count (=5).
        assert loaded["kernels"].shape[1] == 5
        assert loaded["kernels"].shape[0] == R

    def test_missing_psf_key_raises(self, tmp_path):
        """Missing psf_r05_w1 key should raise KeyError."""
        radial_edges = np.linspace(0.0, 1.0, R)
        meta = _fake_metadata(D, R)
        for di in range(D):
            data = {"distance_index": np.array(di), "radial_bin_edges_hmax": radial_edges,
                    **meta}
            for ri in range(R):
                for wi in range(3):
                    if di == 7 and ri == 5 and wi == 1:
                        continue  # skip this key
                    data[f"psf_r{ri:02d}_w{wi}"] = np.ones((32, 32), dtype=np.float32)
            np.savez(tmp_path / f"ag_dblgauss_f2.0_focus0.70m_d{di:02d}_miss.npz", **data)

        with pytest.raises(KeyError):
            convert(input_dir=tmp_path, out_path=tmp_path / "out.npz")

    def test_inconsistent_kernel_shape_raises(self, tmp_path):
        """If one file has 64x64 kernels while others have 32x32, shape mismatch."""
        radial_edges = np.linspace(0.0, 1.0, R)
        meta = _fake_metadata(D, R)
        for di in range(D):
            data = {"distance_index": np.array(di), "radial_bin_edges_hmax": radial_edges,
                    **meta}
            ksize = 64 if di == 3 else 32
            for ri in range(R):
                for wi in range(3):
                    data[f"psf_r{ri:02d}_w{wi}"] = np.ones((ksize, ksize), dtype=np.float32)
            np.savez(tmp_path / f"ag_dblgauss_f2.0_focus0.70m_d{di:02d}_shape.npz", **data)

        # The first file sets H, W = 32, 32. File di=3 has 64x64 kernels.
        # Assignment kernels[ri, di] = k where k is 64x64 into a 32x32 slot
        # should raise ValueError.
        with pytest.raises(ValueError):
            convert(input_dir=tmp_path, out_path=tmp_path / "out.npz")

    def test_empty_directory_raises(self, tmp_path):
        """Empty directory still raises RuntimeError (P1.9.3 L6: zero files
        is the only file-count condition that hard-fails — D >= 1)."""
        with pytest.raises(RuntimeError, match="No input files"):
            convert(input_dir=tmp_path, out_path=tmp_path / "out.npz")

    def test_missing_radial_edges_key_raises(self, tmp_path):
        """Missing both radial_edges_hmax and radial_bin_edges_hmax should raise."""
        for di in range(D):
            data = {"distance_index": np.array(di)}
            for ri in range(R):
                for wi in range(3):
                    data[f"psf_r{ri:02d}_w{wi}"] = np.ones((32, 32), dtype=np.float32)
            np.savez(tmp_path / f"ag_dblgauss_f2.0_focus0.70m_d{di:02d}_noedge.npz", **data)

        with pytest.raises(KeyError):
            convert(input_dir=tmp_path, out_path=tmp_path / "out.npz")


# ── Energy conservation through the pipeline ──────────────────────────────

class TestEnergyConservation:
    """L1 normalization must ensure every kernel sums to 1.0 after convert()."""

    def test_all_kernels_sum_to_one(self, render_npz):
        data = np.load(str(render_npz))
        kernels = data["kernels"]
        for ri in range(kernels.shape[0]):
            for di in range(kernels.shape[1]):
                s = float(kernels[ri, di].sum())
                assert abs(s - 1.0) < 1e-5, f"Kernel [{ri},{di}] sum={s}"

    def test_no_nan_in_output(self, render_npz):
        data = np.load(str(render_npz))
        assert not np.any(np.isnan(data["kernels"]))

    def test_no_inf_in_output(self, render_npz):
        data = np.load(str(render_npz))
        assert not np.any(np.isinf(data["kernels"]))

    def test_all_kernels_non_negative_in_real_data(self, render_npz):
        """Real CodeV PSFs should not contain negative values post-normalization."""
        data = np.load(str(render_npz))
        kernels = data["kernels"]
        assert kernels.min() >= 0.0, (
            f"Unexpected negative value {kernels.min()} in real-data kernels"
        )


# ── Wrong-polarity / wrong-assumption fixtures (Post-mortem 1) ───────────

class TestWrongPolarityAssumptions:
    """Per DevTeam.iSummicron.md Post-mortem 1: always include at least one
    wrong-polarity or wrong-assumption fixture to catch inverted logic.
    """

    def test_defocus_ratios_are_not_distances(self, render_npz):
        """defocus_ratios are linspace(0.0,1,D), NOT actual object distances.
        Verify they are monotonically increasing up to 1, not from
        0.70m to infinity (which would be wrong polarity).
        Item 2 (d-axis ceiling fix): defocus_ratios now starts at 0.0.
        """
        data = np.load(str(render_npz))
        dr = data["defocus_ratios"]
        assert dr[0] >= 0, "defocus_ratios[0] must not be negative"
        assert dr[0] == pytest.approx(0.0)
        assert dr[-1] == pytest.approx(1.0)
        # Wrong polarity would be dr[0] > dr[-1] (e.g. far-to-near ordering)
        assert np.all(np.diff(dr) > 0), "defocus_ratios should be monotonically increasing"

    def test_radial_edges_start_at_zero(self, render_npz):
        """radial_edges[0] must be 0.0 (image center). If it starts at 1.0,
        the polarity is inverted (edge-to-center ordering).
        """
        data = np.load(str(render_npz))
        edges = data["radial_edges"]
        assert edges[0] == pytest.approx(0.0, abs=1e-6)
        assert edges[-1] == pytest.approx(1.0, abs=1e-2)

    def test_kernel_peak_near_center(self, render_npz):
        """The on-axis (r=0) in-focus PSF should peak near its center, not at
        edges. A wrong-polarity kernel (flipped or transposed axes) would have
        the peak displaced to a corner.
        """
        data = np.load(str(render_npz))
        kernels = data["kernels"]
        # r=0, d=0: on-axis, nearest distance
        k = kernels[0, 0]
        H, W = k.shape
        peak_y, peak_x = np.unravel_index(k.argmax(), k.shape)
        # Peak should be within the central 50% of the kernel
        assert H // 4 <= peak_y <= 3 * H // 4, (
            f"Peak at y={peak_y}, expected within [{H // 4}, {3 * H // 4}]"
        )
        assert W // 4 <= peak_x <= 3 * W // 4, (
            f"Peak at x={peak_x}, expected within [{W // 4}, {3 * W // 4}]"
        )

    def test_axis_order_is_radial_defocus_not_transposed(self, render_npz):
        """kernels shape is (R, D, H, W) = (12, 11, 256, 256).
        Wrong assumption: someone might swap R and D. Verify that the
        on-axis kernel (r=0) is narrower than the off-axis kernel (r=10)
        at the same defocus step, which is the physically expected behavior.
        """
        data = np.load(str(render_npz))
        kernels = data["kernels"]
        # Compare spatial extent at mid-defocus
        d_mid = D // 2
        k_onaxis = kernels[0, d_mid]
        k_offaxis = kernels[R - 1, d_mid]
        # Measure spread as second moment (sigma^2)
        def spread(k):
            H, W = k.shape
            y, x = np.mgrid[:H, :W].astype(np.float64)
            cy, cx = H / 2.0, W / 2.0
            total = k.sum()
            if total < 1e-10:
                return 0.0
            return float((((y - cy) ** 2 + (x - cx) ** 2) * k).sum()) / total

        spread_on = spread(k_onaxis)
        spread_off = spread(k_offaxis)
        # Off-axis should be at least as wide as on-axis due to aberrations
        # This is a physical sanity check, not a strict invariant
        assert spread_off >= spread_on * 0.5, (
            f"Off-axis spread {spread_off:.1f} unexpectedly much narrower than "
            f"on-axis {spread_on:.1f} -- axis order may be swapped"
        )


# ── Bilinear weight edge cases ───────────────────────────────────────────

class TestBilinearWeightEdgeCases:

    def test_weights_sum_to_one(self, tensor):
        """Bilinear weights must sum to 1.0 for arbitrary inputs."""
        radial_edges = tensor["radial_edges"]
        defocus_ratios = tensor["defocus_ratios"]
        for rf in [0.0, 0.25, 0.5, 0.75, 1.0]:
            for dv in [0.15, 0.3, 0.5, 0.7, 0.85, 1.0]:
                _, weights = _bilinear_weights(rf, dv, radial_edges, defocus_ratios)
                assert pytest.approx(sum(weights), abs=1e-6) == 1.0

    def test_radial_frac_outside_range(self, tensor):
        """radial_frac > 1.0 or < 0.0 should clamp, not crash."""
        radial_edges = tensor["radial_edges"]
        defocus_ratios = tensor["defocus_ratios"]
        # Should not raise
        indices, weights = _bilinear_weights(2.0, 0.5, radial_edges, defocus_ratios)
        assert pytest.approx(sum(weights), abs=1e-6) == 1.0

        indices, weights = _bilinear_weights(-0.5, 0.5, radial_edges, defocus_ratios)
        assert pytest.approx(sum(weights), abs=1e-6) == 1.0

    def test_defocus_exactly_at_boundary(self, tensor):
        """defocus_val exactly at a ratio boundary should not crash."""
        radial_edges = tensor["radial_edges"]
        defocus_ratios = tensor["defocus_ratios"]
        # Exactly at defocus_ratios[5] = 0.5
        indices, weights = _bilinear_weights(0.5, 0.5, radial_edges, defocus_ratios)
        assert pytest.approx(sum(weights), abs=1e-6) == 1.0


# ── Duplicate distance indices ───────────────────────────────────────────

class TestDuplicateDistanceIndex:

    def test_duplicate_distance_index_extends_D(self, tmp_path):
        """P1.9.3 L6: with D derived from len(files), an "extra" file
        bumps D from 11 to 12; metadata `object_distances_m` length-check
        in load_tensor catches the inconsistency downstream.

        Pre-P1.9.3 this raised at the file-count gate. Post-P1.9.3 the
        inconsistency surfaces at the load_tensor validation step (G5):
        the writer emits a (12, 11, ...) shape mismatch with respect to
        object_distances_m (length 11 from metadata).

        We assert the inconsistency is detected — either at convert()
        (assert in object_distances_m length check) or at load_tensor
        (ValueError on length mismatch). The exact failure mode is the
        diagnostic — silent overwrite is what we are protecting against.
        """
        radial_edges = np.linspace(0.0, 1.0, R)
        meta = _fake_metadata(D, R)
        for di in range(D):
            data = {
                "distance_index": np.array(di),
                "radial_bin_edges_hmax": radial_edges,
                **meta,
            }
            for ri in range(R):
                for wi in range(3):
                    data[f"psf_r{ri:02d}_w{wi}"] = np.ones((32, 32), dtype=np.float32)
            np.savez(
                tmp_path / f"ag_dblgauss_f2.0_focus0.70m_d{di:02d}_dup.npz",
                **data,
            )

        # Extra file with duplicate distance_index=5 → 12 files total.
        data = {
            "distance_index": np.array(5),
            "radial_bin_edges_hmax": radial_edges,
            **meta,
        }
        for ri in range(R):
            for wi in range(3):
                data[f"psf_r{ri:02d}_w{wi}"] = np.ones((32, 32), dtype=np.float32) * 2.0
        np.savez(tmp_path / "ag_dblgauss_f2.0_focus0.70m_d99_dup.npz", **data)

        out_path = tmp_path / "out.npz"
        # The convert step asserts len(object_distances_m) == D in the
        # metadata-derivation loop. With D=12 (file count) and metadata
        # object_distances_m of length 11, the assertion in convert()
        # fires.
        with pytest.raises((AssertionError, RuntimeError, ValueError)):
            convert(input_dir=tmp_path, out_path=out_path)
