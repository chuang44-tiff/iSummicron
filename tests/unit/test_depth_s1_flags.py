"""S1 plumbing: --depth-guided / --depth-mapping / --depth-gain (Run 5 part 1)."""
from __future__ import annotations

import math
import subprocess
import sys
import warnings
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from summicron.render.api import RenderOptions, render_image



def _make_tensor_npz(path: Path, focus_distance_m: float = 0.7, n_defocus: int = 4) -> Path:
    R, D, K = 3, n_defocus, 11
    kernels = np.zeros((R, D, K, K), dtype=np.float32)
    kernels[:, :, K // 2, K // 2] = 1.0
    np.savez(
        str(path), kernels=kernels,
        radial_edges=np.linspace(0.0, 1.0, R + 1),
        defocus_ratios=np.linspace(0.0, 1.0, D),
        corpus_ref_long_edge=100, otf_norm_applied=True,
        object_distances_m=np.linspace(focus_distance_m, 5.0, D),
        focus_distance_m=focus_distance_m,
    )
    return path


def _make_image(path: Path, h: int = 64, w: int = 64) -> Path:
    from PIL import Image
    arr = np.random.default_rng(0).integers(0, 256, (h, w, 3), dtype=np.uint8)
    Image.fromarray(arr, "RGB").save(str(path), "JPEG", quality=80)
    return path

_PARSE = None


def _parse(*argv):
    import argparse

    from summicron.render import _render_args

    p = argparse.ArgumentParser()
    _render_args.add_render_arguments(p)
    return p.parse_args(["--image", "x.jpg", *argv])


# ---- 1. parity / defaults --------------------------------------------------

def test_defaults_and_dests():
    ns = _parse()
    assert (ns.depth_guided, ns.depth_mapping, ns.depth_gain) == ("auto", "auto", "auto")
    o = RenderOptions.from_argparse(ns)
    assert (o.depth_guided, o.depth_mapping, o.depth_gain) == ("auto", "auto", "auto")


def test_argparse_values():
    o = RenderOptions.from_argparse(
        _parse("--depth-guided", "on", "--depth-mapping", "diopter", "--depth-gain", "1.5"))
    assert (o.depth_guided, o.depth_mapping, o.depth_gain) == ("on", "diopter", 1.5)
    assert RenderOptions.from_argparse(
        _parse("--depth-mapping", "diopter", "--depth-gain", "auto")).depth_gain == "auto"


def test_gain_type_failure_is_argparse_error():
    with pytest.raises(SystemExit) as e:
        _parse("--depth-gain", "x")
    assert e.value.code == 2


def test_canonical_bench_defaults_unchanged():
    o = RenderOptions.canonical_bench("a.jpg", (0.4, 0.4))
    assert (o.depth_guided, o.depth_mapping, o.depth_gain) == ("auto", "auto", "auto")
    assert o.depth_scale == 1.0  # Run 6: CANONICAL_FLAGS drops --depth-scale 10
    o = RenderOptions.canonical_bench("a.jpg", (0.4, 0.4), old_look=True)
    assert (o.depth_guided, o.depth_mapping, o.depth_gain) == ("off", "linear", "auto")
    assert o.depth_scale == 10.0


def test_render_args_stays_numpy_free():
    code = ("import sys; import summicron.render._render_args; "
            "sys.exit(int('numpy' in sys.modules))")
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0


# ---- 5. validation ---------------------------------------------------------

def _mk(**kw):
    return RenderOptions(image_path=Path("x.jpg"), **kw)


@pytest.mark.parametrize("kw,flag", [
    (dict(depth_guided="maybe"), "--depth-guided"),
    (dict(depth_mapping="meters"), "--depth-mapping"),
    (dict(depth_mapping="diopter", depth_gain=0), "--depth-gain"),
    (dict(depth_mapping="diopter", depth_gain=-1.0), "--depth-gain"),
    (dict(depth_mapping="diopter", depth_gain=float("nan")), "--depth-gain"),
    (dict(depth_mapping="diopter", depth_gain=float("inf")), "--depth-gain"),
    (dict(depth_mapping="diopter", depth_gain=11), "--depth-gain"),
    (dict(depth_mapping="diopter", depth_gain=True), "--depth-gain"),
    (dict(depth_mapping="diopter", depth_gain="x"), "--depth-gain"),
    (dict(depth_mapping="diopter", depth_gain=None), "--depth-gain"),
    (dict(depth_gain=1.5, depth_mapping="linear"), "--depth-gain"),  # numeric with linear
    (dict(depth_mapping="diopter", depth_source="monocular"), "--depth-mapping"),
    (dict(depth_guided="on", depth_source="monocular"), "--depth-guided"),
])
def test_validation_rejects(kw, flag):
    with pytest.raises(ValueError, match=flag):
        _mk(**kw)


def test_validation_accepts():
    assert _mk(depth_mapping="diopter", depth_gain=2).depth_gain == 2.0
    assert isinstance(_mk(depth_mapping="diopter", depth_gain=2).depth_gain, float)
    assert _mk(depth_mapping="diopter", depth_gain=10).depth_gain == 10.0
    assert _mk(depth_gain="auto").depth_gain == "auto"
    _mk(depth_guided="off", depth_source="monocular")


# ---- pipeline harness ------------------------------------------------------

def _run(tmp_path, *, scale=1.0, depth=None, **kw):
    img = _make_image(tmp_path / "t.jpg")
    ten = _make_tensor_npz(tmp_path / "t.npz")
    if depth is None:
        depth = np.linspace(0.7, 4.0, 64 * 64, dtype=np.float32).reshape(64, 64)
    cap: dict = {"lidar": [], "tap": []}

    def lidar(path, target_hw=None, **k):
        cap["lidar"].append(k)
        d = depth
        if k.get("depth_guided") == "on":
            d = depth * np.float32(1.1)  # guided stand-in: visibly different
        return d.copy()

    from summicron.depth.estimate import apply_tap_focus_shift as orig_tap

    def tap(d, fxy, ft=0.7, *a, **k):
        cap["tap"].append(np.array(d, copy=True))
        return orig_tap(d, fxy, ft, *a, **k)

    def fake_render(image, tensor, **k):
        cap["render"] = k
        h, w = image.shape[:2]
        return np.zeros((h, w, 3), np.uint8), np.zeros((h, w, 3), np.float32)

    # Run 6: this harness characterises the explicit (pre-flip) flag set unless
    # a test overrides it; the auto defaults are covered by test_default_flip*.
    kw.setdefault("depth_guided", "off")
    kw.setdefault("depth_mapping", "linear")
    opts = RenderOptions(image_path=img, tensor_path=ten, depth_scale=scale,
                         focus_xy=(0.5, 0.5), out_dir=tmp_path / "o", **kw)
    saved: dict = {}

    def fake_save(u8, stem, out_dir, **k):
        saved.update(k)
        return out_dir / "x.jpg"

    with patch("summicron.depth.estimate.extract_lidar_depth", side_effect=lidar), \
         patch("summicron.depth.estimate.apply_tap_focus_shift", side_effect=tap), \
         patch("summicron.render.convolve.render", side_effect=fake_render), \
         patch("summicron.render.convolve.save_results", side_effect=fake_save):
        res = render_image(opts)
    cap["saved"] = saved
    cap["res"] = res
    cap["depth"] = depth
    return cap


# ---- 2. threading ----------------------------------------------------------

def test_default_call_signature_has_no_guided_kwarg(tmp_path):
    cap = _run(tmp_path)
    assert cap["lidar"] == [{}]


def test_guided_on_passes_kwarg_only_then(tmp_path):
    cap = _run(tmp_path, depth_guided="on")
    assert cap["lidar"] == [{"depth_guided": "on"}]
    assert "depth_matte" not in cap["lidar"][0]
    assert "_gd" in cap["saved"].get("name_suffix", "")


def test_guided_changes_map(tmp_path):
    a = _run(tmp_path)["render"]["depth_m"]
    b = _run(tmp_path, depth_guided="on")["render"]["depth_m"]
    assert not np.array_equal(a, b)


# ---- 3/4. diopter branch ---------------------------------------------------

def test_linear_branch_unscaled_untouched(tmp_path):
    cap = _run(tmp_path, scale=10.0)
    np.testing.assert_array_equal(cap["tap"][0], cap["depth"] * np.float32(10.0))
    assert cap["saved"]["depth_scale"] == 10.0
    assert cap["saved"].get("name_suffix", "") == ""
    assert cap["res"].diagnostics.warnings == ()


def test_diopter_matches_leaf_functions(tmp_path):
    from summicron.depth.diopter import auto_gain, diopter_equivalent_depth
    from summicron.depth.estimate import apply_tap_focus_shift

    cap = _run(tmp_path, depth_mapping="diopter")
    d = cap["depth"]
    od = np.linspace(0.7, 5.0, 4).astype(np.float64)
    _, anchor = apply_tap_focus_shift(d, (0.5, 0.5), 0.7)
    g = auto_gain(d, anchor, z_focus=0.7, od=od, lo=1.0)
    want = diopter_equivalent_depth(d, anchor, gain=g, z_focus=0.7, z_far=float(od[-1]))
    np.testing.assert_array_equal(cap["render"]["depth_m"], want)
    assert cap["saved"]["name_suffix"] == f"_dio{g:.2f}"
    assert cap["saved"]["depth_scale"] == 1.0
    np.testing.assert_array_equal(cap["tap"][0], d)  # unscaled


def test_diopter_explicit_gain_verbatim_and_differs(tmp_path):
    a = _run(tmp_path, depth_mapping="diopter", depth_gain=1.0)
    b = _run(tmp_path, depth_mapping="diopter", depth_gain=2.0)
    assert a["saved"]["name_suffix"] == "_dio1.00"
    assert b["saved"]["name_suffix"] == "_dio2.00"
    assert not np.array_equal(a["render"]["depth_m"], b["render"]["depth_m"])


def test_diopter_equals_linear_distinct(tmp_path):
    a = _run(tmp_path)["render"]["depth_m"]
    b = _run(tmp_path, depth_mapping="diopter")["render"]["depth_m"]
    assert not np.array_equal(a, b)


def test_diopter_scale_warns_and_collects(tmp_path):
    with pytest.warns(RuntimeWarning, match="depth-scale"):
        cap = _run(tmp_path, scale=10.0, depth_mapping="diopter")
    assert any("depth-scale" in w for w in cap["res"].diagnostics.warnings)
    assert cap["saved"]["depth_scale"] == 1.0


def test_diopter_scale_one_silent(tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        cap = _run(tmp_path, scale=1.0, depth_mapping="diopter")
    assert cap["res"].diagnostics.warnings == ()


def test_diopter_output_independent_of_scale(tmp_path):
    a = _run(tmp_path, scale=1.0, depth_mapping="diopter")["render"]["depth_m"]
    with pytest.warns(RuntimeWarning):
        b = _run(tmp_path, scale=10.0, depth_mapping="diopter")["render"]["depth_m"]
    np.testing.assert_array_equal(a, b)


def test_combo_suffix_order(tmp_path):
    cap = _run(tmp_path, depth_guided="on", depth_mapping="diopter", depth_gain=1.12)
    assert cap["saved"]["name_suffix"] == "_gd_dio1.12"


def test_save_results_name_suffix(tmp_path):
    from PIL import Image  # noqa: F401

    from summicron.render.convolve import save_results

    p = save_results(np.zeros((4, 4, 3), np.uint8), "s", tmp_path, depth_scale=1.0,
                     name_suffix="_gd_dio1.12")
    assert p.name == "s_rendered_gd_dio1.12.jpg"
    p = save_results(np.zeros((4, 4, 3), np.uint8), "s", tmp_path)
    assert p.name == "s_rendered.jpg"


@pytest.mark.parametrize("argv", [
    ["--depth-mapping", "linear", "--depth-gain", "1.5"],
    ["--depth-mapping", "diopter", "--depth-gain", "11"],
    ["--depth-guided", "on", "--depth-source", "monocular"],
])
def test_cli_validation_error_exit2(argv):
    r = subprocess.run(
        [sys.executable, "-m", "summicron.render.convolve", "--image", "nonexistent.jpg", *argv],
        capture_output=True, text=True,
    )
    assert r.returncode == 2
    lines = [ln for ln in r.stderr.splitlines() if ": error: " in ln]
    assert len(lines) == 1 and "--depth-" in lines[0], r.stderr[-500:]
    assert "Traceback" not in r.stderr


# ---- BUG-1: focus_target_m >= tensor far end under diopter (pipeline guard) -------------

def test_diopter_focus_target_at_or_beyond_far_end_raises(tmp_path):
    # test tensor object_distances_m = linspace(0.7, 5.0): far end 5.0 m
    for ft in (5.0, 25.0):
        with pytest.raises(ValueError, match="focus-target-m"):
            _run(tmp_path, depth_mapping="diopter", focus_target_m=ft)


def test_diopter_focus_target_below_far_end_ok_and_linear_unguarded(tmp_path):
    _run(tmp_path, depth_mapping="diopter", focus_target_m=4.0)
    _run(tmp_path, depth_mapping="linear", focus_target_m=25.0)  # guard is diopter-only


def test_diopter_far_focus_cli_contract_one_line_exit2(tmp_path, capsys):
    from summicron.render.convolve import run_cli_render

    def run():
        _run(tmp_path, depth_mapping="diopter", focus_target_m=25.0)

    assert run_cli_render(run) == 2
    err = capsys.readouterr().err.strip().splitlines()
    assert len(err) == 1 and err[0].startswith("render: ") and "focus-target-m" in err[0]
