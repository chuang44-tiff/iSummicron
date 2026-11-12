"""Tests for summicron.psf_model.codev_to_render."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from summicron.psf_model.codev_to_render import convert, DEFAULT_INPUT_DIR
from summicron.render.convolve import load_tensor

ROOT = Path(__file__).resolve().parents[2]
OUT_PATH = ROOT / "runs" / "ag_dblgauss" / "ag_dblgauss_f2.0_render_test.npz"


# P1.9.3: R/D are no longer module-level constants in codev_to_render;
# derive them from the canonical CodeV input directory.
# D = number of distance .npz files; R = post-midpoint radial bins.
# CI round: the raw CodeV files are untracked (runs/ag_dblgauss/), so this
# module must stay importable without them (other tests import
# _assert_npz_content_equal from here); the fixture skips instead.
_FILES = sorted(DEFAULT_INPUT_DIR.glob("ag_dblgauss_f2.0_focus*_d*.npz"))
_RAW_PRESENT = bool(_FILES)
if _RAW_PRESENT:
    D = len(_FILES)
    with np.load(str(_FILES[0])) as _f0:
        R = len(_f0["radial_bin_edges_hmax"]) - 1 + 1  # +1 for synthetic midpoint
        # Kernel H/W derived from the first input PSF file (not hardcoded).
        K = int(_f0["psf_r00_w1"].shape[0])
else:
    D = R = K = None


@pytest.fixture(scope="module")
def render_npz():
    """Run convert() once, yield the output path, clean up after."""
    if not _RAW_PRESENT:
        pytest.skip(f"raw CodeV file absent: {DEFAULT_INPUT_DIR}")
    path = convert(input_dir=DEFAULT_INPUT_DIR, out_path=OUT_PATH)
    yield path
    if path.exists():
        path.unlink()


class TestConvertShape:
    def test_output_shape(self, render_npz):
        data = np.load(str(render_npz))
        assert data["kernels"].shape == (R, D, K, K)

    def test_kernels_dtype(self, render_npz):
        data = np.load(str(render_npz))
        assert data["kernels"].dtype == np.float32


class TestL1Normalization:
    def test_all_kernels_sum_to_one(self, render_npz):
        data = np.load(str(render_npz))
        kernels = data["kernels"]
        for ri in range(kernels.shape[0]):
            for di in range(kernels.shape[1]):
                s = float(kernels[ri, di].sum())
                assert abs(s - 1.0) < 1e-6, (
                    f"Kernel [{ri},{di}] sum={s}"
                )


class TestRadialEdges:
    def test_shape(self, render_npz):
        data = np.load(str(render_npz))
        assert data["radial_edges"].shape == (R + 1,)  # 13

    def test_starts_at_zero(self, render_npz):
        data = np.load(str(render_npz))
        assert float(data["radial_edges"][0]) == pytest.approx(0.0)

    def test_monotonic(self, render_npz):
        data = np.load(str(render_npz))
        edges = data["radial_edges"]
        assert np.all(np.diff(edges) > 0), "radial_edges not monotonically increasing"


class TestDefocusRatios:
    def test_shape(self, render_npz):
        data = np.load(str(render_npz))
        assert data["defocus_ratios"].shape == (D,)

    def test_no_negative(self, render_npz):
        data = np.load(str(render_npz))
        dr = data["defocus_ratios"]
        assert np.all(dr >= 0), "defocus_ratios should have no negative values"

    def test_range(self, render_npz):
        data = np.load(str(render_npz))
        dr = data["defocus_ratios"]
        # Item 2 (d-axis ceiling fix): defocus_ratios now starts at 0.0
        assert float(dr[0]) == pytest.approx(0.0)
        assert float(dr[-1]) == pytest.approx(1.0)


class TestPixelPitch:
    def test_shape(self, render_npz):
        data = np.load(str(render_npz))
        assert data["pixel_pitch_um"].shape == (R, D)

    def test_all_positive(self, render_npz):
        data = np.load(str(render_npz))
        assert np.all(data["pixel_pitch_um"] > 0)

    def test_dtype(self, render_npz):
        data = np.load(str(render_npz))
        assert data["pixel_pitch_um"].dtype == np.float64


class TestFullFrameDimPx:
    def test_shape(self, render_npz):
        data = np.load(str(render_npz))
        assert data["full_frame_dim_px"].shape == (R, D, 2)

    def test_dtype(self, render_npz):
        data = np.load(str(render_npz))
        assert data["full_frame_dim_px"].dtype == np.int32

    def test_reasonable_values(self, render_npz):
        """Expect ~50K-80K pixels per dimension for these pitches."""
        data = np.load(str(render_npz))
        dims = data["full_frame_dim_px"]
        # Width (36mm sensor)
        assert np.all(dims[:, :, 0] > 40000), "Width too small"
        assert np.all(dims[:, :, 0] < 100000), "Width too large"
        # Height (24mm sensor)
        assert np.all(dims[:, :, 1] > 30000), "Height too small"
        assert np.all(dims[:, :, 1] < 70000), "Height too large"


class TestObjectDistances:
    def test_shape(self, render_npz):
        data = np.load(str(render_npz))
        assert data["object_distances_m"].shape == (D,)

    def test_sorted(self, render_npz):
        data = np.load(str(render_npz))
        od = data["object_distances_m"]
        assert np.all(np.diff(od) > 0), "object_distances_m not sorted ascending"


class TestMetadataScalars:
    def test_focus_distance_m(self, render_npz):
        data = np.load(str(render_npz))
        assert "focus_distance_m" in data
        assert float(data["focus_distance_m"]) == pytest.approx(0.7)

    def test_h_max_mm(self, render_npz):
        data = np.load(str(render_npz))
        assert "h_max_mm" in data
        assert float(data["h_max_mm"]) == pytest.approx(21.6)

    def test_wavelength_nm(self, render_npz):
        data = np.load(str(render_npz))
        assert "wavelength_nm" in data
        assert float(data["wavelength_nm"]) == pytest.approx(550.0)


class TestOtfNormIntegration:
    """v3 spec §S2/S4: raw converter output is NOT OTF-normalized.

    OTF-normalization moved to ``downscale_tensor`` (spec v3). The raw
    2048 tensor is L1-renormed only; the center kernel is the physical
    on-axis best-focus PSF, not a delta.
    """

    def test_center_kernel_is_not_delta(self, render_npz):
        """(r=0, d=0) is the raw L1-normed CodeV PSF, NOT a delta."""
        data = np.load(str(render_npz))
        k00 = data["kernels"][0, 0]
        # L1 = 1 but energy is spread (real PSF, not a delta).
        assert abs(float(k00.sum()) - 1.0) < 1e-6
        assert int((k00 > 0).sum()) > 1, (
            "Raw converter output should be a physical PSF, not a delta"
        )

    def test_kernels_are_non_negative(self, render_npz):
        """Raw CodeV PSFs are non-negative by physics."""
        data = np.load(str(render_npz))
        assert data["kernels"].min() >= 0.0

    def test_kernels_l1_is_one(self, render_npz):
        """Every kernel is L1-renormalized to 1.0."""
        data = np.load(str(render_npz))
        kernels = data["kernels"]
        for ri in range(kernels.shape[0]):
            for di in range(kernels.shape[1]):
                s = float(kernels[ri, di].sum())
                assert abs(s - 1.0) < 1e-5, f"kernel[{ri},{di}] sum={s}"


class TestLoadTensorRoundTrip:
    def test_load_tensor_succeeds(self, render_npz):
        result = load_tensor(render_npz)
        assert "kernels" in result
        assert "radial_edges" in result
        assert "defocus_ratios" in result

    def test_load_tensor_shapes(self, render_npz):
        result = load_tensor(render_npz)
        assert result["kernels"].shape == (R, D, K, K)
        assert result["radial_edges"].shape == (R + 1,)
        assert result["defocus_ratios"].shape == (D,)


class TestNoModuleLevelRDConstants:
    """P1.9.3 L6 — R/D were removed as module-level literals.

    Derivation now happens at call time inside ``convert()`` (D from
    file count, R from post-midpoint radial_edges length). A new CodeV
    run with a different bin count must Just Work without a code edit.
    """

    def test_no_module_level_r_constant(self):
        import summicron.psf_model.codev_to_render as codev_to_render

        assert "R" not in vars(codev_to_render), (
            "module-level R = <int> literal leaked; P1.9.3 L6 requires "
            "runtime derivation from len(radial_edges) - 1"
        )

    def test_no_module_level_d_constant(self):
        import summicron.psf_model.codev_to_render as codev_to_render

        assert "D" not in vars(codev_to_render), (
            "module-level D = <int> literal leaked; P1.9.3 L6 requires "
            "runtime derivation from len(files)"
        )


def _assert_npz_content_equal(a_path, b_path, ignore=frozenset({"otf_norm_applied"})):
    """Same key set (modulo ``ignore``) and per-key ``np.array_equal``.

    HYG-0 item 12: content compare instead of byte-md5 (npz zip headers /
    legacy metadata keys are not part of the contract).
    """
    with np.load(str(a_path)) as a, np.load(str(b_path)) as b:
        ka, kb = set(a.files) - ignore, set(b.files) - ignore
        assert ka == kb, f"key sets differ: only-a={sorted(ka - kb)} only-b={sorted(kb - ka)}"
        bad = [
            k for k in sorted(ka)
            if a[k].dtype != b[k].dtype or not np.array_equal(a[k], b[k])
        ]
        assert not bad, f"content differs at keys {bad} ({a_path} vs {b_path})"


class TestConvertByteIdentical:
    """P1.9.3 L6 acceptance — re-running convert() on canonical CodeV
    files must reproduce the default raw tensor's content exactly
    (per-key array equality; HYG-0 item 12 replaced the byte-md5).

    Proves the derivation path is bit-exactly equivalent to the
    deleted R=12 / D=11 literals. SLOW: writes ~2.2 GB + reads ~6.6 GB.
    """

    @pytest.mark.slow
    def test_content_matches_default_tensor(self, tmp_path):
        default_raw = (
            ROOT / "runs" / "ag_dblgauss" / "ag_dblgauss_f2.0_render.npz"
        )
        if not default_raw.exists():
            pytest.skip(
                f"default raw tensor absent ({default_raw}); run "
                "`python -m summicron.psf_model.codev_to_render` first"
            )
        out = tmp_path / "roundtrip.npz"
        convert(input_dir=DEFAULT_INPUT_DIR, out_path=out)
        _assert_npz_content_equal(default_raw, out)
