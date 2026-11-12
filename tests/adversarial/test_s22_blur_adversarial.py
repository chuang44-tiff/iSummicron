"""Adversarial round for S22 S1 (synthetic disc kernel family, L1 bright stamp,
physical knee, family cache).

Spec: doc/plan/phase2/section22_blur-magnitude/ (locked spec S1 rev 2).
The strict xfails from the first round were fixed in 42c725e.
"""
from __future__ import annotations

import os
import stat
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from summicron.render import _family
from summicron.render import convolve as cv
from summicron.render.api import RenderOptions, render_image
from tests.fixtures.synthetic_tensor import make_synthetic_tensor

REPO = Path(__file__).resolve().parents[2]
SYN = "synthetic_disc_v0"


def _mk(**kw):
    return RenderOptions(image_path=Path("x.jpg"), **kw)


def _cli(argv, env=None):
    e = dict(os.environ)
    e.pop("ISUMMICRON_DEBUG", None)
    e["PYTHONPATH"] = str(REPO)
    if env:
        e.update(env)
    return subprocess.run(
        [sys.executable, "-m", "summicron.render.convolve", *argv],
        capture_output=True, text=True, env=e, cwd=REPO,
    )


# ---------------------------------------------------------------- aperture validation

@pytest.mark.parametrize("a", [
    0, 0.0, -1, -17.5, 2.999999, 30.000001, 31, 1e308, 1e-300,
    float("nan"), float("inf"), float("-inf"), True, False, "17.5", "", [17.5],
    (17.5,), 1j, np.float32(17.5), np.int64(17),
])
def test_aperture_rejected_direct(a):
    with pytest.raises(ValueError, match="--lens-aperture-mm"):
        _mk(kernel_family="synthetic", lens_aperture_mm=a)


@pytest.mark.parametrize("a", [3, 3.0, 30, 30.0, 17.5, np.float64(13.5)])
def test_aperture_boundaries_accepted_and_floated(a):
    o = _mk(kernel_family="synthetic", lens_aperture_mm=a)
    assert isinstance(o.lens_aperture_mm, float) and o.lens_aperture_mm == float(a)


@pytest.mark.parametrize("a", [3.0, 17.5, float("nan")])
def test_aperture_with_codev_rejected(a):
    with pytest.raises(ValueError, match="--lens-aperture-mm"):
        _mk(lens_aperture_mm=a)


def test_aperture_bad_value_with_codev_names_flag_not_crash():
    with pytest.raises(ValueError):
        _mk(kernel_family="codev", lens_aperture_mm="x")


@pytest.mark.parametrize("fam", ["Synthetic", "SYNTHETIC", " synthetic", "", None,
                                 "synth", 1, True])
def test_kernel_family_enum_strict(fam):
    with pytest.raises(ValueError, match="--kernel-family"):
        _mk(kernel_family=fam)


@pytest.mark.parametrize("s", ["L1", "none", "", None, False, "peak ", 0])
def test_bright_stamp_enum_strict(s):
    with pytest.raises(ValueError, match="--bright-stamp"):
        _mk(bright_stamp=s)


# ---------------------------------------------------------------- stamp gain bounds

@pytest.mark.parametrize("g", [
    0, 0.0, -0.0, -1, -1e-12, 16.0000001, 17, 1e308, float("nan"), float("inf"),
    float("-inf"), True, False, "2", "", [2.0], 1j,
])
def test_gain_rejected_direct(g):
    with pytest.raises(ValueError, match="--bright-stamp-gain"):
        _mk(kernel_family="synthetic", bright_stamp_gain=g)


@pytest.mark.parametrize("g", [1e-9, 0.5, 1, 2, 16, 16.0])
def test_gain_accepted_and_floated(g):
    o = _mk(kernel_family="synthetic", bright_stamp_gain=g)
    assert type(o.bright_stamp_gain) is float


def test_gain_requires_l1_stamp():
    for kw in (
        {},  # codev default => peak
        {"kernel_family": "codev", "bright_stamp": "peak"},
        {"kernel_family": "codev", "bright_stamp": "auto"},
        {"kernel_family": "synthetic", "bright_stamp": "peak"},
    ):
        with pytest.raises(ValueError, match="--bright-stamp-gain"):
            _mk(bright_stamp_gain=2.0, **kw)
    _mk(bright_stamp="l1", bright_stamp_gain=2.0)
    _mk(kernel_family="synthetic", bright_stamp_gain=2.0)
    _mk(kernel_family="synthetic", bright_stamp="l1", bright_stamp_gain=2.0)


def test_invalid_gain_beats_peak_complaint():
    with pytest.raises(ValueError, match="--bright-stamp-gain"):
        _mk(bright_stamp_gain=float("nan"))


def test_l1_stamp_with_psf_peak_rejected():
    with pytest.raises(ValueError, match="psf-norm l1"):
        _mk(bright_stamp="l1", psf_norm="peak")
    with pytest.raises(ValueError, match="psf-norm l1"):
        _mk(kernel_family="synthetic", psf_norm="peak")
    with pytest.raises(ValueError, match="psf-norm l1"):
        _mk(kernel_family="synthetic", bright_stamp="peak", psf_norm="peak")


# ---------------------------------------------------------------- flag combinations

def test_synthetic_with_tensor_rejected():
    with pytest.raises(ValueError, match="--tensor"):
        _mk(kernel_family="synthetic", tensor_path=Path("t.npz"))


@pytest.mark.parametrize("kw", [
    {"depth_mapping": "linear"},
    {"depth_source": "monocular"},
    {"depth_source": "monocular", "depth_mapping": "diopter"},
])
def test_synthetic_needs_lidar_diopter(kw):
    with pytest.raises(ValueError):
        _mk(kernel_family="synthetic", **kw)


def test_synthetic_old_look_rejected_but_old_look_diopter_override_ok():
    with pytest.raises(ValueError, match="synthetic"):
        RenderOptions.canonical_bench(
            "x.jpg", (0.5, 0.5), old_look=True, kernel_family="synthetic")
    o = RenderOptions.canonical_bench(
        "x.jpg", (0.5, 0.5), old_look=True, kernel_family="synthetic",
        depth_mapping="diopter")
    assert o.kernel_family == "synthetic" and o.depth_mapping == "diopter"


def test_synthetic_canonical_bench_override_and_occlusion_ok():
    o = RenderOptions.canonical_bench(
        "x.jpg", (0.5, 0.5), kernel_family="synthetic",
        occlusion_compositing="on", lens_aperture_mm=25.0)
    assert o.psf_norm == "l1" and o.occlusion_compositing == "on"


def test_default_fields_untouched():
    o = _mk()
    assert (o.kernel_family, o.lens_aperture_mm, o.bright_stamp,
            o.bright_stamp_gain) == ("codev", None, "auto", None)


def _parser():
    import argparse
    from summicron.render import _render_args

    p = argparse.ArgumentParser()
    _render_args.add_render_arguments(p)
    return p


@pytest.mark.parametrize("val", ["nan", "inf", "0", "-3", "2.9", "31", "1e999", "17.5.1"])
def test_cli_parse_aperture_bad(val):
    try:
        ns = _parser().parse_args(["--image", "x.jpg", "--kernel-family", "synthetic",
                                   f"--lens-aperture-mm={val}"])
    except SystemExit:
        return
    with pytest.raises(ValueError, match="--lens-aperture-mm"):
        RenderOptions.from_argparse(ns)


@pytest.mark.parametrize("val,exp", [("35/2", 17.5), ("50/2", 25.0), ("apple", 13.5),
                                     ("17.5", 17.5), ("3", 3.0)])
def test_cli_presets(val, exp):
    ns = _parser().parse_args(["--image", "x.jpg", "--kernel-family", "synthetic",
                               "--lens-aperture-mm", val])
    assert RenderOptions.from_argparse(ns).lens_aperture_mm == exp


@pytest.mark.parametrize("val", ["Apple", "35/2 ", " 50/2", "35 / 2", "f/2", "35/2/", "1/2"])
def test_cli_preset_lookalikes_rejected(val):
    with pytest.raises(SystemExit):
        _parser().parse_args(["--image", "x.jpg", "--kernel-family", "synthetic",
                              f"--lens-aperture-mm={val}"])


def test_cli_error_contract_nan_aperture_exit2_no_traceback():
    r = _cli(["--image", "iphoneImage/IMG_6651.HEIC", "--kernel-family", "synthetic",
              "--lens-aperture-mm=nan", "--out-dir", "/nonexistent/should/not/be/used"])
    assert r.returncode == 2, (r.stdout, r.stderr)
    assert "--lens-aperture-mm" in r.stderr and "Traceback" not in r.stderr


def test_cli_error_contract_gain_without_l1():
    r = _cli(["--image", "iphoneImage/IMG_6651.HEIC", "--bright-stamp-gain", "3"])
    assert r.returncode == 2
    assert "--bright-stamp" in r.stderr and "Traceback" not in r.stderr


def test_cli_synthetic_without_lidar_fails_before_family_build(tmp_path):
    from PIL import Image

    img = tmp_path / "nolidar.jpg"
    Image.fromarray(np.full((32, 32, 3), 100, np.uint8)).save(img)
    cache = tmp_path / "cache"
    r = _cli(["--image", str(img), "--kernel-family", "synthetic",
              "--out-dir", str(tmp_path / "o"), "--hdr-source", "none"],
             env={"ISUMMICRON_CACHE_DIR": str(cache)})
    assert r.returncode == 2, (r.stdout, r.stderr)
    assert "Traceback" not in r.stderr and "render: " in r.stderr
    assert not cache.exists() or not list(cache.rglob("*.npz")), "family built before LiDAR check"


# ---------------------------------------------------------------- pipeline (fast fakes)

def _tensor_npz(path: Path, *, synth: bool, D: int = 5) -> Path:
    R, K = 3, 11
    kernels = np.zeros((R, D, K, K), dtype=np.float32)
    kernels[:, :, K // 2, K // 2] = 1.0
    extra = {"synth_family": np.array(SYN)} if synth else {}
    np.savez(
        str(path), kernels=kernels, radial_edges=np.linspace(0.0, 1.0, R + 1),
        defocus_ratios=np.linspace(0.0, 1.0, D), corpus_ref_long_edge=100,
        otf_norm_applied=True, delta_guard_applied=True,
        schema_version=np.array("1.0"),
        object_distances_m=np.linspace(0.7, 5.0, D), focus_distance_m=0.7, **extra,
    )
    return path


def _pipeline(tmp_path, **kw):
    from PIL import Image

    img = tmp_path / "t.jpg"
    Image.fromarray(np.random.default_rng(0).integers(0, 256, (64, 64, 3), dtype=np.uint8),
                    "RGB").save(str(img), "JPEG", quality=80)
    synth_npz = _tensor_npz(tmp_path / "s.npz", synth=True)
    codev_npz = _tensor_npz(tmp_path / "c.npz", synth=False)
    depth = np.linspace(0.7, 4.0, 64 * 64, dtype=np.float32).reshape(64, 64)
    cap: dict = {"fam": []}

    def fake_family(a, *, on_progress=None, **k):
        cap["fam"].append(a)
        return cv.load_tensor(synth_npz), synth_npz, True

    def fake_render(image, tensor, **k):
        cap["render"] = k
        h, w = image.shape[:2]
        return np.zeros((h, w, 3), np.uint8), np.zeros((h, w, 3), np.float32)

    def fake_save(u8, stem, out_dir, **k):
        cap["saved"] = k
        return out_dir / "x.jpg"

    if kw.get("kernel_family", "codev") == "codev":
        kw.setdefault("tensor_path", codev_npz)
    base = dict(image_path=img, focus_xy=(0.5, 0.5), out_dir=tmp_path / "o",
                depth_source="lidar", hdr_source="none")
    base.update(kw)
    opts = RenderOptions(**base)
    with patch("summicron.depth.estimate.extract_lidar_depth",
               side_effect=lambda *a, **k: depth.copy()), \
         patch("summicron.render._family.load_family_tensor", side_effect=fake_family), \
         patch("summicron.render.convolve.render", side_effect=fake_render), \
         patch("summicron.render.convolve.save_results", side_effect=fake_save):
        res = render_image(opts)
    cap["res"] = res
    return cap


def test_pipeline_synthetic_occlusion_on_threads_gain_and_mode(tmp_path):
    cap = _pipeline(tmp_path, kernel_family="synthetic", occlusion_compositing="on",
                    lens_aperture_mm=13.5, bright_stamp_gain=3.0)
    assert cap["render"]["occlusion_compositing"] == "on"
    assert cap["render"]["bright_stamp_gain"] == 3.0
    assert cap["fam"] == [13.5]


def test_pipeline_synthetic_explicit_peak_stamp_threads_none(tmp_path):
    cap = _pipeline(tmp_path, kernel_family="synthetic", bright_stamp="peak")
    assert cap["render"]["bright_stamp_gain"] is None
    assert cap["saved"]["name_suffix"].endswith("_peak")


def test_pipeline_codev_l1_stamp_threads_gain_and_no_family(tmp_path):
    cap = _pipeline(tmp_path, bright_stamp="l1", bright_stamp_gain=4.0,
                    depth_mapping="diopter")
    assert cap["fam"] == [] and cap["render"]["bright_stamp_gain"] == 4.0


def test_pipeline_two_apertures_distinct_suffixes(tmp_path):
    a = _pipeline(tmp_path, kernel_family="synthetic", lens_aperture_mm=13.5)
    b = _pipeline(tmp_path, kernel_family="synthetic", lens_aperture_mm=25.0)
    assert a["saved"]["name_suffix"] != b["saved"]["name_suffix"]


def test_pipeline_codev_default_diagnostics_have_no_family_extras(tmp_path):
    r = _pipeline(tmp_path)["res"].diagnostics.resolved
    assert "lens_aperture_mm" not in r and "bright_stamp_gain" not in r


def test_pipeline_codev_stdout_has_no_family_text(tmp_path, capsys):
    _pipeline(tmp_path)
    out = capsys.readouterr().out
    assert "synthetic" not in out and "knee=" not in out


# ---------------------------------------------------------------- render() stamp gain

def _img(h=40, w=40, seed=1):
    rng = np.random.default_rng(seed)
    img = (rng.random((h, w, 3)) * 0.5 + 0.1).astype(np.float32)
    img[10:14, 10:14] = 1.0
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def _depth(h=40, w=40, lo=0.7, hi=3.0):
    return np.linspace(lo, hi, h * w, dtype=np.float32).reshape(h, w)


def _tensor(D=11, synth=False, R=3):
    t = make_synthetic_tensor(R, D, 15, 15, corpus_ref_long_edge=40)
    t["synth_family"] = SYN if synth else None
    return t


def _render(tensor=None, depth=None, **kw):
    kw.setdefault("bright_thresh", 0.3)
    kw.setdefault("bright_pass_region_size", 3)
    stats = kw.pop("stats", {})
    out = cv.render(_img(), tensor or _tensor(),
                    depth_m=depth if depth is not None else _depth(), stats=stats, **kw)
    return out, stats


@pytest.mark.parametrize("g", [0, 0.0, -1, float("nan"), float("inf"), True, "2", [1.0]])
def test_render_gain_rejected(g):
    with pytest.raises(ValueError, match="bright_stamp_gain"):
        _render(bright_stamp_gain=g)


def test_render_gain_requires_l1():
    with pytest.raises(ValueError, match="psf_norm"):
        _render(bright_stamp_gain=2.0, psf_norm="peak")


def test_render_gain_none_equals_omitted_bitwise():
    (a, _), _ = _render()
    (b, _), _ = _render(bright_stamp_gain=None)
    assert np.array_equal(a, b)


def test_render_gain_extremes_finite_and_deterministic():
    (u1, l1), _ = _render(bright_stamp_gain=16.0)
    (u2, l2), _ = _render(bright_stamp_gain=16.0)
    assert np.isfinite(l1).all() and np.array_equal(l1, l2) and np.array_equal(u1, u2)
    (_, lt), _ = _render(bright_stamp_gain=1e-30)
    assert np.isfinite(lt).all()


def test_render_gain_linear_in_bright_term():
    ls = [np.asarray(_render(bright_stamp_gain=g)[0][1], np.float64) for g in (1.0, 2.0, 3.0)]
    d1, d2 = ls[1] - ls[0], ls[2] - ls[1]
    assert np.abs(d1).max() > 0, "gain has no effect: stamp not threaded"
    assert np.allclose(d1, d2, rtol=1e-3, atol=1e-4)


def test_render_gain_both_scatter_paths():
    (_, off), _ = _render(bright_stamp_gain=2.0, occlusion_compositing="off")
    (_, on), _ = _render(bright_stamp_gain=2.0, occlusion_compositing="on")
    assert np.isfinite(off).all() and np.isfinite(on).all()
    (_, on_pk), _ = _render(occlusion_compositing="on")
    assert not np.array_equal(on, on_pk), "ON path ignores the L1 stamp"


def test_render_gain_peak_psf_plus_on_rejected():
    with pytest.raises(ValueError):
        _render(bright_stamp_gain=2.0, psf_norm="peak", occlusion_compositing="on")


# ---------------------------------------------------------------- knee

def test_knee_codev_exactly_010_is_the_constant():
    _, st = _render()
    assert st["knee_ratio"] == 0.1 and st["knee_ratio"] == cv.D_KNEE_LAYERED


def test_knee_codev_pixels_at_exactly_node1_fully_snap():
    t = _tensor(D=11)
    od = t["object_distances_m"]
    _, st = _render(t, np.full((40, 40), od[1], np.float32))
    assert float(st["dmap"].max()) == pytest.approx(0.1, abs=1e-6)
    assert float(st["alpha"].max()) < 1e-4
    _, st2 = _render(t, np.full((40, 40), od[1] - 0.01, np.float32))
    assert 0.0 < float(st2["alpha"].max()) < 1.0


def test_knee_codev_beyond_knee_alpha_nonnegative():
    t = _tensor(D=11)
    _, st = _render(t, np.full((40, 40), t["object_distances_m"][5], np.float32))
    assert float(st["alpha"].min()) >= 0.0 and float(st["alpha"].max()) < 1e-6


def test_knee_synthetic_tag_with_geometric_axis_no_nan():
    t = _tensor(D=16, synth=True)
    t["object_distances_m"] = np.geomspace(0.7, 20.0, 16)
    _, st = _render(t)
    assert 0.0 <= st["knee_ratio"] <= 1.0 and np.isfinite(st["alpha"]).all()


def _diopter16(focus=0.7):
    from summicron.psf_model import synth_family as sf

    tt = _tensor(D=16, synth=True)
    tt["object_distances_m"] = np.asarray(
        sf.object_distances_for_axis("diopter16", np.array([0.7, 20.0]), n_d=16), np.float64)
    tt["focus_distance_m"] = focus
    return tt


def test_knee_synthetic_family_axis_at_knee_diopter():
    tt = _diopter16()
    z = 1.0 / (1.0 / 0.7 - cv.KNEE_DIOPTER)
    _, st = _render(tt, np.full((40, 40), z, np.float32))
    assert 0.2950 <= st["knee_ratio"] <= 0.2954
    assert float(st["alpha"].max()) < 1e-3


def test_knee_synthetic_focus_none_defaults_to_07():
    _, st = _render(_diopter16(focus=None))
    assert 0.2950 <= st["knee_ratio"] <= 0.2954


def test_knee_degenerate_axis_does_not_nan():
    t = _tensor(D=16, synth=True)
    t["object_distances_m"] = np.linspace(1.0, 20.0, 16)
    t["focus_distance_m"] = 0.7
    _, st = _render(t)
    assert np.isfinite(st["alpha"]).all()


def test_knee_unknown_family_tag_uses_codev_constant():
    t = _tensor(D=11)
    t["synth_family"] = "synthetic_disc_v1"
    _, st = _render(t)
    assert st["knee_ratio"] == cv.D_KNEE_LAYERED


# ---------------------------------------------------------------- slab cap (D=16)

def test_synthetic_d16_spread_never_warns():
    msgs: list[str] = []
    _render(_tensor(D=16, synth=True), _depth(hi=30.0), on_progress=msgs.append)
    assert not any("pathological" in m for m in msgs)


def test_synthetic_d16_occlusion_on_runs():
    (u, lin), _ = _render(_tensor(D=16, synth=True), _depth(hi=30.0),
                          occlusion_compositing="on")
    assert np.isfinite(lin).all() and u.shape == (40, 40, 3)


# ---------------------------------------------------------------- memory guard

def test_guard_estimate_monotone_and_native_d16_fits():
    H, W = 4284, 5712
    e11 = cv._estimate_peak_bytes(H, W, 11, kernel_size=183)
    e16 = cv._estimate_peak_bytes(H, W, 16, kernel_size=183)
    assert e16 > e11
    on16 = cv._estimate_peak_bytes(H, W, 16, kernel_size=183, occlusion=True)
    assert on16 >= e16
    assert cv._estimate_peak_bytes(H, W, 16, kernel_size=183, slab_batch_size=4) >= e16
    assert on16 < cv.MAX_BYTES and e16 < cv.MAX_BYTES


def test_guard_native_d16_fails_cleanly_under_small_budget(monkeypatch):
    monkeypatch.setenv("ISUMMICRON_MAX_BYTES", str(2 * 1024 ** 3))
    with pytest.raises(cv.MemoryBudgetError) as ei:
        cv._memory_guard(4284, 5712, 16, kernel_size=183, n_radial=12)
    assert ei.value.suggested_downres is None or ei.value.suggested_downres >= 256
    assert "downres" in str(ei.value)


def test_guard_uses_tensor_D_not_hardcoded_11(monkeypatch):
    kw = dict(kernel_size=15, n_radial=3)
    e11 = cv._estimate_peak_bytes(40, 40, 11, **kw)
    e16 = cv._estimate_peak_bytes(40, 40, 16, **kw)
    if e16 <= e11 + 1:
        pytest.skip("estimator not D-sensitive at this size")
    monkeypatch.setenv("ISUMMICRON_MAX_BYTES", str((e11 + e16) // 2))
    _render(_tensor(D=11))
    with pytest.raises(cv.MemoryBudgetError):
        _render(_tensor(D=16, synth=True))


# ---------------------------------------------------------------- family cache

@pytest.fixture(scope="module")
def built(tmp_path_factory):
    d = tmp_path_factory.mktemp("advfam")
    t, p, hit = _family.load_family_tensor(17.5, cache_dir=d)
    assert hit is False
    with np.load(str(p), allow_pickle=False) as z:
        payload = {k: z[k] for k in z.files}
    return d, t, p, payload


def _forge(payload, tmp_path, name, **repl):
    pl = dict(payload)
    pl.update(repl)
    pl["synth_kernels_sha256"] = np.array(_family._kernels_sha256(pl["kernels"]))
    path = tmp_path / name
    np.savez(path, **pl)
    return path


def test_poison_nan_kernels_valid_hash_rebuilt(built, tmp_path):
    _, _, p, payload = built
    k = payload["kernels"].copy()
    k[2, 3, 91, 91] = np.nan
    _forge(payload, tmp_path, p.name, kernels=k)
    t2, _, hit = _family.load_family_tensor(17.5, cache_dir=tmp_path)
    assert hit is False and np.isfinite(t2["kernels"]).all()


def test_poison_negative_kernels_valid_hash_rebuilt(built, tmp_path):
    _, _, p, payload = built
    k = payload["kernels"].copy()
    k[2, 3, 91, 91] = -1.0
    _forge(payload, tmp_path, p.name, kernels=k)
    t2, _, hit = _family.load_family_tensor(17.5, cache_dir=tmp_path)
    assert hit is False or (t2["kernels"] >= 0).all()


def test_poison_wrong_shape_valid_hash_rebuilt(built, tmp_path):
    _, _, p, payload = built
    _forge(payload, tmp_path, p.name, kernels=payload["kernels"][:, :11])
    t2, _, _ = _family.load_family_tensor(17.5, cache_dir=tmp_path)
    assert t2["D"] == 16 and t2["kernels"].shape == (12, 16, 183, 183)


def test_poison_missing_family_tag_must_not_hit(built, tmp_path):
    _, _, p, payload = built
    pl = {k: v for k, v in payload.items() if k != "synth_family"}
    pl["synth_kernels_sha256"] = np.array(_family._kernels_sha256(pl["kernels"]))
    np.savez(tmp_path / p.name, **pl)
    t2, _, _ = _family.load_family_tensor(17.5, cache_dir=tmp_path)
    assert t2["synth_family"] == SYN


def test_poison_object_pickle_not_executed(built, tmp_path):
    marker = tmp_path / "pwned"

    class Evil:
        def __reduce__(self):
            return (Path.touch, (marker,))

    _, _, p, _ = built
    np.savez(tmp_path / p.name, kernels=np.array([Evil()], dtype=object),
             synth_key=np.array(_family.family_key(17.5)))
    t2, _, hit = _family.load_family_tensor(17.5, cache_dir=tmp_path)
    assert not marker.exists() and hit is False and t2["D"] == 16


def test_zero_byte_cache_file_rebuilt(built, tmp_path):
    (tmp_path / built[2].name).write_bytes(b"")
    assert _family.load_family_tensor(17.5, cache_dir=tmp_path)[2] is False


def test_directory_squatting_cache_name_falls_back(built, tmp_path):
    (tmp_path / built[2].name).mkdir()
    t2, _, hit = _family.load_family_tensor(17.5, cache_dir=tmp_path)
    assert hit is False and t2["D"] == 16


def test_stale_source_md5_changes_key(built, monkeypatch):
    k0 = _family.family_key(17.5)
    monkeypatch.setattr(_family, "_source_md5", lambda: "0" * 32)
    assert _family.family_key(17.5) != k0


def test_readonly_dir_with_valid_cache_is_a_hit(built, tmp_path):
    import shutil

    ro = tmp_path / "ro"
    ro.mkdir()
    shutil.copy(built[2], ro / built[2].name)
    ro.chmod(stat.S_IRUSR | stat.S_IXUSR)
    try:
        if os.access(ro, os.W_OK):
            pytest.skip("cannot make dir read-only")
        _, _, hit = _family.load_family_tensor(17.5, cache_dir=ro)
        assert hit is True
    finally:
        ro.chmod(stat.S_IRWXU)


def test_readonly_dir_corrupt_cache_falls_back_and_leaves_no_litter(built, tmp_path):
    ro = tmp_path / "ro2"
    ro.mkdir()
    (ro / built[2].name).write_bytes(b"junk")
    ro.chmod(stat.S_IRUSR | stat.S_IXUSR)
    try:
        if os.access(ro, os.W_OK):
            pytest.skip("cannot make dir read-only")
        msgs: list[str] = []
        t, _, hit = _family.load_family_tensor(17.5, cache_dir=ro, on_progress=msgs.append)
        assert hit is False and t["D"] == 16 and any("not writable" in m for m in msgs)
        assert sorted(x.name for x in ro.iterdir()) == [built[2].name]
    finally:
        ro.chmod(stat.S_IRWXU)


def test_readonly_fallback_returned_path_exists(built, tmp_path):
    ro = tmp_path / "ro3"
    ro.mkdir()
    ro.chmod(stat.S_IRUSR | stat.S_IXUSR)
    try:
        if os.access(ro, os.W_OK):
            pytest.skip("cannot make dir read-only")
        _, path, _ = _family.load_family_tensor(17.5, cache_dir=ro / "sub")
        # the cache dir was unwritable: the path must name it, never a cleaned temp dir
        assert "isummicron_synth_" not in str(path)
        assert str(path).startswith(str(ro))
    finally:
        ro.chmod(stat.S_IRWXU)


@pytest.mark.parametrize("a", [0.0, -5.0])
def test_load_family_nonpositive_aperture_raises_and_leaves_no_files(a, tmp_path):
    with pytest.raises(ValueError):
        _family.load_family_tensor(a, cache_dir=tmp_path)
    assert not list(tmp_path.glob("*"))


def test_load_family_nan_aperture_leaves_no_cache_file(tmp_path):
    with pytest.raises(Exception):
        _family.load_family_tensor(float("nan"), cache_dir=tmp_path)
    assert not list(tmp_path.glob("*"))


def test_load_family_inf_aperture_rejected(tmp_path):
    with pytest.raises((ValueError, ArithmeticError)):
        _family.load_family_tensor(float("inf"), cache_dir=tmp_path)


def test_cache_dir_env_empty_string_is_default_not_cwd(monkeypatch):
    monkeypatch.setenv("ISUMMICRON_CACHE_DIR", "")
    assert _family.default_cache_dir().is_absolute()


def test_concurrent_builds_one_valid_file_no_litter(tmp_path):
    code = (
        "import sys;from summicron.render import _family;"
        "t,p,h=_family.load_family_tensor(7.0,cache_dir=sys.argv[1]);"
        "assert t['D']==16"
    )
    env = dict(os.environ, PYTHONPATH=str(REPO))
    procs = [subprocess.Popen([sys.executable, "-c", code, str(tmp_path)], env=env,
                              stderr=subprocess.PIPE, text=True) for _ in range(3)]
    errs = [p.communicate()[1] for p in procs]
    assert all(p.returncode == 0 for p in procs), errs
    files = sorted(x.name for x in tmp_path.iterdir())
    assert len(files) == 1 and files[0].startswith("synth_") and ".tmp" not in files[0], files
    assert _family.load_family_tensor(7.0, cache_dir=tmp_path)[2] is True


def test_concurrent_threads_same_key(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(3) as ex:
        res = list(ex.map(lambda _: _family.load_family_tensor(9.0, cache_dir=tmp_path),
                          range(3)))
    assert all(r[0]["D"] == 16 for r in res)
    assert len(list(tmp_path.glob("synth_*.npz"))) == 1
    assert not list(tmp_path.glob("*.tmp*"))
    assert np.array_equal(res[0][0]["kernels"], res[1][0]["kernels"])
    assert np.array_equal(res[1][0]["kernels"], res[2][0]["kernels"])


def test_family_key_stable_and_aperture_sensitive():
    assert _family.family_key(17.5) == _family.family_key(17.5)
    assert _family.family_key(17.5) != _family.family_key(17.51)
    assert _family.family_key(17.5) == _family.family_key(np.float64(17.5))


def test_other_aperture_build_does_not_disturb_first(built):
    before = built[1]["kernels"].copy()
    _family.load_family_tensor(25.0, cache_dir=built[0])
    t, _, hit = _family.load_family_tensor(17.5, cache_dir=built[0])
    assert hit is True and np.array_equal(t["kernels"], before)
    assert np.array_equal(built[1]["kernels"], before)


def test_shipped_artifact_still_pristine():
    import hashlib

    from summicron.render._artifacts import default_tensor_path
    from summicron.render._bench import TENSOR_MD5

    assert hashlib.md5(Path(default_tensor_path()).read_bytes()).hexdigest() == TENSOR_MD5
    assert cv.load_tensor(default_tensor_path())["synth_family"] is None


def test_resolve_family_is_none_idiom_not_truthiness():
    from summicron.render._resolve import resolve_family

    class Opts:  # bypasses validation on purpose
        kernel_family = "synthetic"
        lens_aperture_mm = None
        bright_stamp = "auto"
        bright_stamp_gain = None

    r = resolve_family(Opts())
    assert (r.lens_aperture_mm, r.bright_stamp, r.bright_stamp_gain) == (17.5, "l1", 2.0)
    Opts.lens_aperture_mm = 0.0
    Opts.bright_stamp_gain = 0.0
    r = resolve_family(Opts())
    assert r.lens_aperture_mm == 0.0 and r.bright_stamp_gain == 0.0
    Opts.kernel_family = "codev"
    Opts.lens_aperture_mm = None
    Opts.bright_stamp_gain = None
    r = resolve_family(Opts())
    assert (r.lens_aperture_mm, r.bright_stamp, r.bright_stamp_gain) == (None, "peak", None)
