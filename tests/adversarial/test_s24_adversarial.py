"""Adversarial tests for the optiland ray-traced PSF tensors (0.5.0).

FAST, numpy-only, isummicron env, no ``runs/`` dependence. Attacks
``summicron.psf_model.ray_kernel``, the G-O orientation gate (wrong-polarity
fixture), the two SHIPPED tensors under ``summicron/_artifacts/``, the shipped
default pin, and the reproducibility of the ``scripts/build_psf`` tooling.

Real bugs are ``xfail(strict=True, reason="BUG: ...")``; production code is not
modified here.
"""
from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pytest

from summicron.psf_model import ray_kernel as rk
from summicron.render.convolve import load_tensor

from summicron.render._artifacts import LENS_ARTIFACTS, default_tensor_path

ROOT = Path(__file__).resolve().parents[2]
BUILD = ROOT / "scripts" / "build_psf"
LENSES = BUILD / "lenses"
LENS_KEYS = ["35", "50"]


def _ker(R=12, D=11, n=5):
    k = np.zeros((R, D, n, n))
    k[:, 1:, n // 2, n // 2] = 1.0
    return k


# =========================================================================== 1. ray_kernel edges
@pytest.mark.parametrize("size", [0, 1, 2, 4, 182, -3, -183])
def test_rays_to_kernel_rejects_even_or_tiny_or_negative_size(size):
    with pytest.raises(ValueError, match="odd"):
        rk.rays_to_kernel([0.0], [0.0], 1.0, size=size)


def test_rays_to_kernel_float_size_raises_not_silently_truncates():
    # 5.0 is not an int: any error is fine, a silently wrong box is not.
    with pytest.raises((TypeError, ValueError)):
        rk.rays_to_kernel([0.0], [0.0], 1.0, size=5.0)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("axis", ["out", "tan"])
def test_rays_to_kernel_rejects_nonfinite(bad, axis):
    a = np.zeros(10)
    b = np.zeros(10)
    (a if axis == "out" else b)[3] = bad
    with pytest.raises(ValueError, match="non-finite"):
        rk.rays_to_kernel(a, b, 0.01)


@pytest.mark.parametrize("pitch", [0.0, -1.0, np.nan, np.inf])
def test_rays_to_kernel_rejects_bad_pitch(pitch):
    with pytest.raises(ValueError, match="pitch"):
        rk.rays_to_kernel([0.0], [0.0], pitch)


def test_rays_to_kernel_rejects_empty_and_mismatched():
    with pytest.raises(ValueError, match="no rays"):
        rk.rays_to_kernel([], [], 1.0)
    with pytest.raises(ValueError, match="same length"):
        rk.rays_to_kernel([0.0, 1.0], [0.0], 1.0)


@pytest.mark.parametrize("where", [(1e3, 0.0), (0.0, -1e3), (-1e3, 1e3)])
def test_rays_to_kernel_all_outside_raises(where):
    n = 100
    with pytest.raises(ValueError, match="outside"):
        rk.rays_to_kernel(np.full(n, where[0]), np.full(n, where[1]), 0.01, size=31)


def test_rays_to_kernel_overflowing_displacement_counted_outside():
    # d/pitch overflows to inf: must be counted as outside, not poison the kernel.
    with np.errstate(all="ignore"):
        k, out = rk.rays_to_kernel([0.0, 1e300], [0.0, 0.0], 1e-10, size=5, return_outside=True)
    assert np.all(np.isfinite(k)) and k.sum() == pytest.approx(1.0)
    assert out == pytest.approx(0.5)
    assert k[2, 2] == 1.0


@pytest.mark.parametrize("d_out_px,d_tan_px", [(0.0, 1.0), (1.0, 0.0), (-1.0, -1.0), (0.5, 0.0), (0.0, -0.5)])
def test_single_ray_on_pixel_boundary(d_out_px, d_tan_px):
    pitch = 0.004
    k, out = rk.rays_to_kernel([d_out_px * pitch], [d_tan_px * pitch], pitch, size=7, return_outside=True)
    assert out == 0.0
    assert k.sum() == pytest.approx(1.0, abs=1e-12)
    row = 3 + rk.OUTWARD_ROW_SIGN * d_out_px
    col = 3 + d_tan_px
    ys, xs = np.indices(k.shape)
    # bilinear splat preserves the centroid exactly
    assert (k * ys).sum() == pytest.approx(row, abs=1e-9)
    assert (k * xs).sum() == pytest.approx(col, abs=1e-9)
    if float(d_out_px).is_integer() and float(d_tan_px).is_integer():
        assert np.count_nonzero(k) == 1


def test_ray_on_far_box_edge_keeps_all_energy():
    # col == size-1 exactly: the zero-weight neighbour at col == size lies outside;
    # no energy may be reported lost.
    k, out = rk.rays_to_kernel([0.0], [2.0], 1.0, size=5, return_outside=True)
    assert out == 0.0 and k[2, 4] == 1.0


def test_ray_half_outside_box_reports_half():
    k, out = rk.rays_to_kernel([0.0], [2.5], 1.0, size=5, return_outside=True)
    assert out == pytest.approx(0.5)
    assert k[2, 4] == pytest.approx(1.0)  # renormalised over in-box energy


def test_large_ray_count_l1_and_centroid():
    rng = np.random.default_rng(24)
    n = 3_000_000
    a = rng.normal(0, 0.05, n)
    b = rng.normal(0, 0.05, n)
    k, out = rk.rays_to_kernel(a, b, 0.0088, size=183, return_outside=True)
    assert out == 0.0
    assert abs(k.sum() - 1.0) < 1e-12
    ys, xs = np.indices(k.shape)
    assert abs((k * ys).sum() - 91) < 0.05 and abs((k * xs).sum() - 91) < 0.05


def test_float32_input_matches_float64():
    rng = np.random.default_rng(5)
    a = rng.normal(0, 0.03, 20000).astype(np.float32)
    b = rng.normal(0, 0.03, 20000).astype(np.float32)
    k32 = rk.rays_to_kernel(a, b, np.float32(0.0088), size=63)
    k64 = rk.rays_to_kernel(a.astype(np.float64), b.astype(np.float64), float(np.float32(0.0088)), size=63)
    assert k32.dtype == np.float64
    np.testing.assert_allclose(k32, k64, atol=1e-12)


def test_shape_metrics_tail_sign_inward_vs_outward():
    n = 101
    c = n // 2
    y, x = np.indices((n, n), dtype=np.float64)
    core = np.exp(-0.5 * (((y - c) / 3.0) ** 2 + ((x - c) / 3.0) ** 2))
    core /= core.sum()
    # a smooth one-sided wedge of mass, not a single pixel
    out_t = np.zeros_like(core)
    in_t = np.zeros_like(core)
    for s in range(5, 30):
        out_t[c + rk.OUTWARD_ROW_SIGN * s, c - 2:c + 3] = 1.0 / s
        in_t[c - rk.OUTWARD_ROW_SIGN * s, c - 2:c + 3] = 1.0 / s
    out_t /= out_t.sum()
    in_t /= in_t.sum()
    _, t_out = rk.shape_metrics(0.85 * core + 0.15 * out_t)
    _, t_in = rk.shape_metrics(0.85 * core + 0.15 * in_t)
    assert t_out > 0.3, t_out
    assert t_in < -0.3, t_in
    assert t_out == pytest.approx(-t_in, rel=1e-9)  # exact mirror


def test_shape_metrics_on_rays_kernel_tail_follows_ray_side():
    # Through rays_to_kernel: an outward ray tail must read tail > 0.
    rng = np.random.default_rng(9)
    pitch = 0.01
    core_o, core_t = rng.normal(0, 0.03, 50000), rng.normal(0, 0.03, 50000)
    tail_o = rng.exponential(0.15, 10000)  # positive = outward
    tail_t = rng.normal(0, 0.01, 10000)
    k_out = rk.rays_to_kernel(np.r_[core_o, tail_o], np.r_[core_t, tail_t], pitch, size=183)
    k_in = rk.rays_to_kernel(np.r_[core_o, -tail_o], np.r_[core_t, tail_t], pitch, size=183)
    assert rk.shape_metrics(k_out)[1] > 0.5
    assert rk.shape_metrics(k_in)[1] < -0.5


def test_shape_metrics_delta_is_degenerate_not_crash():
    a, t = rk.shape_metrics(rk.make_delta(5))
    assert a == float("inf") and t == 0.0
    with pytest.raises(ValueError):
        rk.shape_metrics(np.zeros((5, 5)))
    with pytest.raises(ValueError):
        rk.shape_metrics(np.zeros(5))


# --------------------------------------------------------------------------- assemble_tensor
@pytest.mark.parametrize("shape", [(11, 11, 5, 5), (12, 10, 5, 5), (13, 11, 5, 5), (12, 12, 5, 5)])
def test_assemble_wrong_R_or_D_rejected(shape):
    k = np.zeros(shape)
    k[:, 1:, 2, 2] = 1.0
    with pytest.raises(ValueError):
        rk.assemble_tensor(k, {})


@pytest.mark.parametrize("shape", [(12, 11, 5, 7), (12, 11, 6, 6), (12, 11, 5), (12, 11, 5, 5, 1)])
def test_assemble_bad_spatial_shape_rejected(shape):
    with pytest.raises(ValueError):
        rk.assemble_tensor(np.zeros(shape), {})


def test_assemble_nonfinite_rejected():
    k = _ker()
    k[2, 3, 0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        rk.assemble_tensor(k, {})


def test_assemble_forces_non_delta_d0_to_exact_delta():
    # A caller handing a real (blurred) d0 gets it replaced by the spec's exact delta.
    k = _ker()
    k[:, 0] = 1.0 / 25
    t = rk.assemble_tensor(k, {})
    d0 = t["kernels"][:, 0]
    assert np.all(d0[:, 2, 2] == 1.0)
    assert np.count_nonzero(d0) == 12
    assert bool(t["delta_guard_applied"]) is True


def test_assemble_float32_l1_within_1e6():
    rng = np.random.default_rng(3)
    k = rng.random((12, 11, 31, 31)) ** 12 * 1e-3  # badly scaled input
    t = rk.assemble_tensor(k, {})
    s = t["kernels"].sum(axis=(2, 3), dtype=np.float64)
    assert np.abs(s - 1.0).max() <= 1e-6


def test_assemble_rejects_all_zero_nond0_kernel():
    k = _ker()
    k[4, 6] = 0.0
    with np.errstate(all="ignore"):
        try:
            t = rk.assemble_tensor(k, {})
        except ValueError:
            return
    assert np.all(np.isfinite(t["kernels"])), "silent NaN kernel"


def test_assemble_meta_cannot_replace_kernels():
    k = _ker()
    try:
        t = rk.assemble_tensor(k, {"kernels": np.ones((2, 2))})
    except ValueError:
        return
    assert t["kernels"].shape == (12, 11, 5, 5)


def test_assemble_meta_cannot_unset_delta_guard():
    t = rk.assemble_tensor(_ker(), {"delta_guard_applied": False})
    assert bool(t["delta_guard_applied"]) is True


def test_assemble_meta_dict_value_render_safe(tmp_path):
    """A dict meta value becomes a pickled object array. load_tensor never reads
    non-schema keys, so the render path is unaffected (documented, not a bug);
    a strict np.load of that key does refuse it."""
    t = rk.assemble_tensor(_ker(), {"note": {"a": 1}})
    p = tmp_path / "t.npz"
    rk.save_tensor(p, t)
    load_tensor(p)
    with np.load(p, allow_pickle=False) as z, pytest.raises(ValueError):
        z["note"]


@pytest.mark.parametrize("n", [183, 303])
def test_load_tensor_roundtrip_sizes(tmp_path, n):
    rng = np.random.default_rng(n)
    k = np.zeros((12, 11, n, n))
    for r in range(12):
        for d in range(1, 11):
            k[r, d] = rk.rays_to_kernel(rng.normal(0, 8, 400), rng.normal(0, 8, 400), 1.0, size=n)
    t = rk.assemble_tensor(k, {"fu": 0.4, "transmission": np.ones((12, 11))})
    p = tmp_path / f"t{n}.npz"
    rk.save_tensor(p, t)
    lt = load_tensor(p)
    assert lt["kernels"].shape == (12, 11, n, n)
    assert np.all(lt["kernels"][:, 0, n // 2, n // 2] == 1.0)
    assert np.count_nonzero(lt["kernels"][:, 0]) == 12
    s = lt["kernels"].sum(axis=(2, 3), dtype=np.float64)
    assert np.abs(s - 1.0).max() <= 1e-6
    np.testing.assert_array_equal(lt["kernels"][:, 1:], t["kernels"][:, 1:])


# =========================================================================== 2. wrong polarity
def test_G_O_gate_can_fail_on_inward_tensor(tmp_path, monkeypatch):
    """MANDATORY wrong-polarity fixture: the real G-O test, fed a tensor whose mass is
    placed INWARD (opposite sign), must fail on the outward-cosine assertion."""
    import tests.unit.test_raytrace_orientation as go

    monkeypatch.setattr(go, "OUTWARD_ROW_SIGN", -rk.OUTWARD_ROW_SIGN)
    with pytest.raises(AssertionError, match="outward cosines"):
        go.test_outward_mass_moves_outward_in_all_wedges(tmp_path)


def test_G_O_gate_passes_on_correct_polarity(tmp_path):
    import tests.unit.test_raytrace_orientation as go

    assert go.OUTWARD_ROW_SIGN == rk.OUTWARD_ROW_SIGN
    go.test_outward_mass_moves_outward_in_all_wedges(tmp_path)


# =========================================================================== 3. shipped tensors
REQUIRED_META = [
    "kernels", "radial_edges", "defocus_ratios", "object_distances_m", "schema_version",
    "corpus_ref_long_edge", "focus_distance_m", "wavelength_nm", "otf_norm_applied",
    "delta_guard_applied", "transmission", "fu", "lens_citation", "prescription_variant",
    "sd_mm", "apertures_assumed", "aperture_note", "pixel_pitch_um", "framing_equiv_focal_mm",
]


@pytest.mark.parametrize("lens", LENS_KEYS)
def test_shipped_tensor_integrity(lens):
    p = default_tensor_path(lens)
    with np.load(p, allow_pickle=False) as z:
        missing = [k for k in REQUIRED_META if k not in z.files]
        assert not missing, missing
        k = z["kernels"]
        assert k.dtype == np.float32 and k.ndim == 4 and k.shape[:2] == (12, 11)
        n = k.shape[2]
        assert k.shape[3] == n and n % 2 == 1
        assert not np.isnan(k).any() and np.all(np.isfinite(k)) and k.min() >= 0
        l1 = k.sum(axis=(2, 3), dtype=np.float64)
        assert np.abs(l1 - 1.0).max() <= 1e-6
        d0 = k[:, 0]
        assert np.all(d0[:, n // 2, n // 2] == 1.0) and np.count_nonzero(d0) == 12
        # every non-d0 node is a real spread kernel, not a stray delta
        assert np.all(k[:, 1:].max(axis=(2, 3)) < 1.0)
        np.testing.assert_array_equal(z["radial_edges"], rk.RADIAL_EDGES)
        np.testing.assert_array_equal(z["object_distances_m"], rk.OBJECT_DISTANCES_M)
        np.testing.assert_array_equal(z["defocus_ratios"], rk.DEFOCUS_RATIOS)
        assert str(z["schema_version"]) == "1.0"
        assert int(z["corpus_ref_long_edge"]) == 5712
        assert float(z["focus_distance_m"]) == 0.7
        assert float(z["wavelength_nm"]) == 550.0
        assert not bool(z["otf_norm_applied"]) and bool(z["delta_guard_applied"])
        assert float(z["fu"]) == 0.4
        sd = z["sd_mm"]
        assert sd.ndim == 1 and sd.size >= 11 and np.all(np.isfinite(sd)) and np.all(sd[1:] > 0)
        assert z["transmission"].shape == (12, 11)
        cite = str(z["lens_citation"])
        if lens == "35":
            assert "US patent 3,006,249" in cite and "optiland" in cite
            assert str(z["prescription_variant"]) == "patent"
        else:
            assert "optiland" in cite and "double-Gauss" in cite
            assert str(z["prescription_variant"]) == "optiland-sample"
    lt = load_tensor(p)
    assert (lt["R"], lt["D"]) == (12, 11)


@pytest.mark.parametrize("lens", LENS_KEYS)
def test_shipped_transmission_vs_field(lens):
    """T is non-increasing with field at the far nodes (d>=5), within 1e-4."""
    with np.load(default_tensor_path(lens)) as z:
        T = z["transmission"]
    assert np.all(np.isfinite(T)) and T.min() > 0.1
    far = T[:, 5:]
    assert np.diff(far, axis=0).max() <= 1e-4, np.diff(far, axis=0).max()
    assert T.max() < 1.05
    assert T[-1, -1] < T[0, -1]  # vignetting is present (cat-eye source)


@pytest.mark.parametrize("lens", LENS_KEYS)
def test_shipped_aberrations_visible(lens):
    """The aberrations must stay visible: edge kernels at 20 m are not round."""
    with np.load(default_tensor_path(lens)) as z:
        k = z["kernels"]
    a_axis, _ = rk.shape_metrics(k[0, 10])
    a_edge, t_edge = rk.shape_metrics(k[11, 10])
    assert abs(a_axis - 1.0) < 0.15
    assert a_edge < 0.9  # radial compression (cat-eye) at the field edge
    assert t_edge > 0.05  # outward tail at the field edge


@pytest.mark.parametrize("lens", LENS_KEYS)
def test_shipped_axis_transmission_about_unity(lens):
    with np.load(default_tensor_path(lens)) as z:
        assert "epd_collimated_mm" in z.files
        T = z["transmission"]
    assert 0.97 <= T[0, 10] <= 1.03, T[0, 10]


def test_default_is_the_35_artifact():
    assert default_tensor_path().name == LENS_ARTIFACTS["35"]
    assert default_tensor_path() == default_tensor_path("35")


def test_shipped_tensor_md5_pins_unchanged():
    from summicron.render._bench import TENSOR_MD5, TENSOR_MD5_50

    assert hashlib.md5(default_tensor_path("35").read_bytes()).hexdigest() == TENSOR_MD5
    assert hashlib.md5(default_tensor_path("50").read_bytes()).hexdigest() == TENSOR_MD5_50


def test_render_args_do_not_reference_scratch_or_inventor_names():
    from summicron.render import _render_args

    src = Path(_render_args.__file__).read_text().lower()
    assert "s24_raytrace" not in src and "mand" + "ler" not in src and "lei" + "tz" not in src


def test_ray_kernel_has_no_artifact_dependency():
    assert not hasattr(rk, "DEFAULT_SOURCE")
    src = Path(rk.__file__).read_text()
    assert "_artifacts" not in src and "np.load" not in src
    assert rk.OBJECT_DISTANCES_M.shape == (11,) and rk.RADIAL_EDGES.shape == (13,)


# =========================================================================== 5. reproducibility
def test_env_file_pins_optiland():
    env = (ROOT / "environment-optiland.yml").read_text()
    assert re.search(r"optiland==\d+\.\d+\.\d+", env), "optiland not pinned to an exact version"


def test_build_scripts_carry_no_absolute_paths():
    bad = re.compile("(" + "|".join([
        "/pri" + "vate/tmp", "/tm" + "p/", "scratch" + "pad", "/Vol" + "umes/", "/Us" + "ers/",
        "claude" + "-501", "doc/" + "plan", "runs/"]) + ")")
    files = list(BUILD.glob("*.py")) + list(LENSES.glob("*.json")) + [
        ROOT / "environment-optiland.yml", ROOT / "summicron/psf_model/ray_kernel.py"]
    hits = [f"{f.name}:{i}" for f in files for i, ln in enumerate(f.read_text().splitlines(), 1)
            if bad.search(ln)]
    assert not hits, hits


@pytest.mark.parametrize("fname,needle,ref,variant", [
    ("us3006249_ex2.json", "US patent 3,006,249", "scan", "patent"),
    ("optiland_doublegauss_50f2.json", "optiland", "scan", "optiland-sample"),
])
def test_lens_json_metadata(fname, needle, ref, variant):
    d = json.loads((LENSES / fname).read_text())
    assert needle in d["citation"]
    assert "http" not in d["citation"]
    assert d["gs_reference"] == ref and d["prescription_variant"] == variant
    assert any(s.get("stop") for s in d["surfaces"]) or fname.startswith("optiland"), "no stop surface"
    assert d["modified_from_patent"] is False


def test_stop_split_preserves_a3():
    """Stop sits at 0.6 a3; the two air pieces around it must sum to a3."""
    d = json.loads((LENSES / "us3006249_ex2.json").read_text())
    s = d["surfaces"]
    i = next(j for j, x in enumerate(s) if x.get("stop"))
    a3 = s[i - 1]["thickness"] + s[i]["thickness"]
    assert a3 == pytest.approx(d["a3_patent"], abs=1e-9)
    assert s[i - 1]["thickness"] / a3 == pytest.approx(d["stop_fraction_of_a3"], abs=1e-9)


def test_gate_failure_cannot_reach_the_builder():
    """Static: either the optimiser exits before writing, or the builder refuses a
    lens carrying gate_failures."""
    opt_src = (BUILD / "optimise_lens.py").read_text()
    tree = ast.parse(opt_src)
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    src = ast.get_source_segment(opt_src, main)
    write_at = src.find("write_text(json.dumps(out")
    exit_at = src.find("sys.exit(2)") if "if not all(gates.values()):\n        sys.exit" in src else -1
    exits_first = exit_at != -1 and exit_at < write_at
    builder_checks = "gate_failures" in (BUILD / "build_raytrace_tensor.py").read_text()
    assert exits_first or builder_checks


def test_builder_has_no_hardcoded_conda_env_and_declares_gs_reference():
    src = (BUILD / "build_raytrace_tensor.py").read_text()
    assert '"conda", "run"' not in src
    assert "--gt-python" in src and "gs_reference" in src
    assert "scripts/build_psf" in src and "scripts/s24" not in src


# ---- build gate helpers (scripts/build_psf/_gates.py, numpy-only) ---------------------------------
def _gates():
    import importlib.util

    spec = importlib.util.spec_from_file_location("psf_gates", BUILD / "_gates.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_ref3_divides_by_xp_to_paraxial_image_not_sensor():
    """Cone from an exit pupil (D=20 mm, 50 mm from the paraxial image); sensor 5 mm past it:
    radius = 5 * 10 / 50 = 1 mm -> 2000 um. The old divisor (XP->sensor = 55) gave 1818 um."""
    g = _gates()
    assert g.ref3_diameter_um(5.0, 20.0, -50.0) == pytest.approx(2000.0)
    assert g.ref3_diameter_um(-5.0, 20.0, 50.0) == pytest.approx(2000.0)
    assert g.ref3_diameter_um(5.0, 20.0, -50.0) != pytest.approx(1e3 * 5 * 20 / 55, rel=1e-3)


def test_marginal_ray_gate_tolerance_and_polarity():
    g = _gates()
    m = 1000.0
    assert g.gs_ratio_ok(1029.0, m) and g.gs_ratio_ok(971.0, m)
    assert not g.gs_ratio_ok(1031.0, m) and not g.gs_ratio_ok(969.0, m)
    # a units/scale bug (x1.15, as an oversized-aperture or pitch error would give) must fail
    assert not g.gs_ratio_ok(1150.0, m) and not g.gs_ratio_ok(870.0, m)
    assert not g.gs_ratio_ok(1000.0, 0.0)        # degenerate reference never passes


def test_t_scale_makes_axis_t_unity():
    """Nominal launch disc edge lands at h1=11.66 on a lens whose real collimated beam is
    EPD 24.915: raw T = (12.4575/11.66)^2 -> rescaled to 1."""
    g = _gates()
    raw_t = (12.4575 / 11.66) ** 2
    assert raw_t * g.t_scale(11.66, 24.915) == pytest.approx(1.0, rel=1e-6)
