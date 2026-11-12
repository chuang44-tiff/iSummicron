"""Run 6 default flip: ``auto`` resolver, defaults, validation, pipeline fakes."""
from __future__ import annotations

import argparse
import io
import subprocess
import sys
import warnings
from contextlib import ExitStack, redirect_stdout
from itertools import product
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from summicron.render._resolve import MONOCULAR_NOTE, resolve_auto
from summicron.render.api import RenderDiagnostics, RenderOptions, render_image
from tests.unit.test_depth_s1_flags import _make_tensor_npz, _parse

darwin = pytest.mark.skipif(sys.platform != "darwin", reason="hdr macos")


def _mk(**kw):
    return RenderOptions(image_path=Path("x.jpg"), **kw)


# ---- 1. pure resolver --------------------------------------------------------

@pytest.mark.parametrize("has_lidar,hdr,headroom,shadow", list(product(
    (True, False), ("none", "macos"), (True, False), ("off", "on"),
)))
def test_resolver_truth_table_defaults(has_lidar, hdr, headroom, shadow):
    if shadow == "on" and hdr == "none":
        pytest.skip("invalid combination (rejected by validation)")
    o = _mk(hdr_source=hdr, shadow_match_mode=shadow)
    m = resolve_auto(o, has_lidar=has_lidar, hdr_headroom=headroom)
    assert m.depth_source == ("lidar" if has_lidar else "monocular")
    assert m.depth_guided == ("on" if has_lidar else "off")
    assert m.depth_mapping == ("diopter" if has_lidar else "linear")
    want_lut = hdr == "macos" and shadow == "off" and headroom
    assert m.tonemap == ("input-lut" if want_lut else "ci")
    assert m.any_auto
    mono = [n for n in m.notes if n == MONOCULAR_NOTE]
    assert len(mono) == (0 if has_lidar else 1)
    tone_notes = [n for n in m.notes if "tonemap auto" in n]
    if hdr == "macos" and not want_lut:
        assert len(tone_notes) == 1
    else:
        assert tone_notes == []


def test_resolver_explicit_never_overridden():
    o = _mk(depth_source="lidar", depth_guided="off", depth_mapping="linear",
            tonemap="ci", hdr_source="macos")
    m = resolve_auto(o, has_lidar=True, hdr_headroom=True)
    assert (m.depth_source, m.depth_guided, m.depth_mapping, m.tonemap) == (
        "lidar", "off", "linear", "ci")
    assert not m.any_auto and m.notes == ()
    o = _mk(tonemap="input-lut", hdr_source="macos")
    assert resolve_auto(o, has_lidar=True, hdr_headroom=False).tonemap == "input-lut"


@pytest.mark.parametrize("kw", [
    dict(depth_mapping="diopter"), dict(depth_guided="on"),
])
def test_resolver_explicit_lidar_only_mode_without_lidar_raises(kw):
    with pytest.raises(ValueError, match="requires LiDAR depth"):
        resolve_auto(_mk(**kw), has_lidar=False, hdr_headroom=False)


def test_resolver_explicit_lidar_without_lidar_raises():
    with pytest.raises(ValueError, match="no embedded depth aux"):
        resolve_auto(_mk(depth_source="lidar"), has_lidar=False, hdr_headroom=False)


def test_resolver_numeric_gain_with_auto_mapping_no_lidar_warns():
    m = resolve_auto(_mk(depth_gain=2.0), has_lidar=False, hdr_headroom=False)
    assert m.depth_mapping == "linear" and len(m.warnings) == 1
    m = resolve_auto(_mk(depth_gain=2.0), has_lidar=True, hdr_headroom=False)
    assert m.depth_mapping == "diopter" and m.warnings == ()


def test_resolver_bright_thresh_auto_with_ci_note():
    o = _mk(bright_pass_thresh="auto", hdr_source="macos")
    m = resolve_auto(o, has_lidar=True, hdr_headroom=False)
    assert m.tonemap == "ci"
    assert sum("bright_pass_thresh auto" in n for n in m.notes) == 1


# ---- 2. defaults / parity ----------------------------------------------------

def test_field_and_cli_defaults():
    want = dict(depth_source="auto", depth_guided="auto", depth_mapping="auto",
                tonemap="auto", depth_gain="auto", bright_pass_thresh=None,
                depth_scale=1.0, hdr_source="auto")
    o = RenderOptions(image_path=Path("x.jpg"))
    c = RenderOptions.from_argparse(_parse())
    for k, v in want.items():
        assert getattr(o, k) == v, k
        assert getattr(c, k) == v, k


def test_canonical_bench_new_and_old():
    from summicron.render import _bench
    n = RenderOptions.canonical_bench("a.heic", (0.4, 0.45))
    assert (n.depth_source, n.depth_guided, n.depth_mapping, n.tonemap,
            n.depth_scale, n.hdr_source, n.downres) == (
        "lidar", "auto", "auto", "auto", 1.0, "macos", 2048)
    assert "--depth-scale" not in _bench.CANONICAL_FLAGS
    o = RenderOptions.canonical_bench("a.heic", (0.4, 0.45), old_look=True)
    assert (o.depth_source, o.depth_guided, o.depth_mapping, o.tonemap,
            o.bright_pass_thresh, o.depth_scale) == (
        "lidar", "off", "linear", "ci", 1.0, 10.0)
    n_old = len(_bench.OLD_LOOK_FLAGS)
    assert _bench.CANONICAL_FLAGS_OLD[-n_old:] == _bench.OLD_LOOK_FLAGS


def test_argparse_choices_include_auto():
    from summicron.render import _render_args
    p = argparse.ArgumentParser()
    _render_args.add_render_arguments(p)
    ch = {a.dest: a.choices for a in p._actions if a.choices}
    for k in ("depth_source", "depth_guided", "depth_mapping", "tonemap"):
        assert "auto" in ch[k], k
    assert "auto" in ch["hdr_source"]


def test_diagnostics_resolved_hash_and_eq():
    a = RenderDiagnostics(resolved={"tonemap": "ci"})
    b = RenderDiagnostics(resolved={"tonemap": "input-lut"})
    assert hash(a) == hash(b) and a == b


# ---- 3. validation -----------------------------------------------------------

@pytest.mark.parametrize("kw,flag", [
    (dict(hdr_source="bogus"), "--hdr-source"),
    (dict(depth_source="monocular", depth_mapping="diopter"), "--depth-mapping"),
    (dict(depth_source="monocular", depth_guided="on"), "--depth-guided"),
    (dict(tonemap="input-lut", hdr_source="none"), "--tonemap"),
    (dict(tonemap="input-lut", hdr_source="macos", shadow_match_mode="on"), "--tonemap"),
    (dict(bright_pass_thresh="auto", tonemap="ci"), "--bright-pass-thresh"),
    (dict(bright_pass_thresh="auto", tonemap="ci", hdr_source="none"), "--bright-pass-thresh"),
    (dict(depth_source="monocular", depth_gain=2.0), "--depth-gain"),
    (dict(depth_gain=2.0, depth_mapping="linear"), "--depth-gain"),
    (dict(tonemap="Auto"), "--tonemap"),
])
def test_validation_rejects(kw, flag):
    with pytest.raises(ValueError, match=flag):
        _mk(**kw)


def test_validation_accepts():
    _mk(depth_gain=2.0)  # auto mapping: runtime decision
    _mk(depth_gain=2.0, depth_source="lidar")
    _mk(tonemap="auto", shadow_match_mode="on", hdr_source="macos")
    _mk(bright_pass_thresh="auto", hdr_source="macos")  # tonemap auto
    _mk(bright_pass_thresh="auto", tonemap="input-lut", hdr_source="macos")


@pytest.mark.parametrize("argv", [
    ["--hdr-source", "auto"],
    ["--tonemap", "bogus"],
    ["--depth-source", "monocular", "--depth-gain", "2"],
])
def test_cli_exit2(argv):
    r = subprocess.run(
        [sys.executable, "-m", "summicron.render.convolve", "--image", "nonexistent.jpg", *argv],
        capture_output=True, text=True,
    )
    assert r.returncode == 2 and "Traceback" not in r.stderr


# ---- 4. pipeline with fakes --------------------------------------------------

def _png(path: Path, hw=(64, 64)) -> Path:
    from PIL import Image
    g = np.linspace(0, 255, hw[1], dtype=np.float32)
    arr = np.repeat(np.repeat(g[None, :, None], hw[0], axis=0), 3, axis=2).astype(np.uint8)
    Image.fromarray(arr, "RGB").save(str(path), "PNG")
    return path


def _drive(tmp_path, *, lidar=True, hdr_gain=None, **kw):
    """Run the real pipeline with fakes; returns captured calls + stdout."""
    img = _png(tmp_path / "t.png")
    ten = _make_tensor_npz(tmp_path / "t.npz")
    cap: dict = {"lidar": [], "mono": 0, "loads": 0}
    depth = np.linspace(0.7, 4.0, 64 * 64, dtype=np.float32).reshape(64, 64)

    def fake_lidar(path, target_hw=None, **k):
        cap["lidar"].append(k)
        return depth.copy() if lidar else None

    def fake_render(image, tensor, **k):
        cap["render"] = k
        h, w = image.shape[:2]
        return np.zeros((h, w, 3), np.uint8), np.zeros((h, w, 3), np.float32)

    def fake_save(u8, stem, out_dir, **k):
        cap["saved"] = k
        return out_dir / "x.jpg"

    def mono_model(device="mps"):
        cap["mono"] += 1
        return object()

    def mono_est(image, bundle):
        return np.full(image.shape[:2], 0.5, np.float32)

    import summicron.render.convolve as cv
    orig_load = cv.load_image

    def spy_load(p):
        cap["loads"] += 1
        return orig_load(p)

    hdr_kw = {}
    patches = [
        patch("summicron.depth.estimate.extract_lidar_depth", side_effect=fake_lidar),
        patch("summicron.depth.estimate.load_depth_model", side_effect=mono_model),
        patch("summicron.depth.estimate.estimate_depth", side_effect=mono_est),
        patch("summicron.render.convolve.load_image", side_effect=spy_load),
        patch("summicron.render.convolve.render", side_effect=fake_render),
        patch("summicron.render.convolve.save_results", side_effect=fake_save),
    ]
    if hdr_gain is not None:
        hdr = (np.full((64, 64, 3), 0.5, np.float32) * hdr_gain).astype(np.float32)
        patches.append(patch("summicron.io.hdr_load_macos.load_hdr_heic", return_value=hdr))
        hdr_kw["hdr_source"] = "macos"
    opts = RenderOptions(image_path=img, tensor_path=ten, focus_xy=(0.5, 0.5),
                         out_dir=tmp_path / "o", **hdr_kw, **kw)
    buf = io.StringIO()
    with ExitStack() as st, redirect_stdout(buf):
        for p in patches:
            st.enter_context(p)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = render_image(opts)
    cap["out"] = buf.getvalue()
    cap["res"] = res
    return cap


def test_pipeline_lidar_default_is_guided_diopter(tmp_path):
    cap = _drive(tmp_path, lidar=True)
    assert cap["lidar"] == [{"depth_guided": "on"}]
    assert cap["mono"] == 0
    r = cap["res"].diagnostics.resolved
    assert (r["depth_source"], r["depth_guided"], r["depth_mapping"], r["tonemap"]) == (
        "lidar", "on", "diopter", "ci")
    assert ("Resolved: hdr_source=none depth_source=lidar depth_guided=on "
            "depth_mapping=diopter tonemap=ci") in cap["out"]
    assert cap["out"].count("note:") == 1  # hdr_source auto -> none (PNG)
    assert cap["saved"]["name_suffix"].startswith("_gd_dio")


def test_pipeline_no_lidar_falls_back_to_monocular_linear(tmp_path):
    cap = _drive(tmp_path, lidar=False)
    assert cap["mono"] == 1
    assert cap["out"].count(MONOCULAR_NOTE) == 1
    r = cap["res"].diagnostics.resolved
    assert (r["depth_source"], r["depth_guided"], r["depth_mapping"]) == (
        "monocular", "off", "linear")
    assert cap["saved"]["name_suffix"] == ""


def test_pipeline_explicit_old_flags_have_no_resolution_noise(tmp_path):
    cap = _drive(tmp_path, depth_source="lidar", depth_guided="off",
                 depth_mapping="linear", tonemap="ci", hdr_source="none")
    assert cap["lidar"] == [{}]  # no guided kwarg
    assert "Resolved:" not in cap["out"] and "note:" not in cap["out"]


@darwin
def test_pipeline_hdr_headroom_selects_input_lut_with_second_load(tmp_path):
    cap = _drive(tmp_path, hdr_gain=3.0)
    assert cap["res"].diagnostics.resolved["tonemap"] == "input-lut"
    assert cap["loads"] == 1  # SDR base for the LUT fit
    assert cap["saved"]["name_suffix"].endswith("_tlut")


@darwin
def test_pipeline_no_headroom_falls_back_to_ci_with_note(tmp_path):
    cap = _drive(tmp_path, hdr_gain=1.0)
    assert cap["res"].diagnostics.resolved["tonemap"] == "ci"
    assert cap["loads"] == 0
    assert cap["out"].count("no HDR headroom") == 1


def test_pipeline_hdr_none_is_ci_silent(tmp_path):
    cap = _drive(tmp_path)
    assert cap["res"].diagnostics.resolved["tonemap"] == "ci"
    assert cap["loads"] == 1  # only the initial image load
    assert "tonemap auto" not in cap["out"]


def test_pipeline_explicit_diopter_without_lidar_raises(tmp_path):
    with pytest.raises(ValueError, match="requires LiDAR depth"):
        _drive(tmp_path, lidar=False, depth_mapping="diopter")


def test_pipeline_numeric_gain_ignored_warning_when_no_lidar(tmp_path):
    cap = _drive(tmp_path, lidar=False, depth_gain=2.0)
    assert any("--depth-gain" in w for w in cap["res"].diagnostics.warnings)


def test_pipeline_focus_target_far_guard_only_under_diopter(tmp_path):
    with pytest.raises(ValueError, match="focus-target-m"):
        _drive(tmp_path, focus_target_m=25.0)
    _drive(tmp_path, focus_target_m=25.0, depth_mapping="linear")


# ---- R2.7 error mapping ------------------------------------------------------

@pytest.mark.parametrize("exc", [ImportError("no torch"), OSError("offline")])
def test_monocular_failure_is_one_line_exit2(tmp_path, capsys, exc):
    from summicron.cli import main as cli_main
    img = _png(tmp_path / "t.png")
    ten = _make_tensor_npz(tmp_path / "t.npz")
    with patch("summicron.depth.estimate.extract_lidar_depth", return_value=None), \
         patch("summicron.depth.estimate.load_depth_model", side_effect=exc):
        rc = cli_main(["render", "--image", str(img), "--tensor", str(ten),
                       "--out-dir", str(tmp_path / "o")])
    err = capsys.readouterr().err.strip().splitlines()
    assert rc == 2
    assert len(err) == 1 and err[0].startswith("render: image has no LiDAR depth")
    assert type(exc).__name__ in err[0]


def test_monocular_failure_debug_reraises(tmp_path, monkeypatch):
    from summicron.cli import main as cli_main
    monkeypatch.setenv("ISUMMICRON_DEBUG", "1")
    img = _png(tmp_path / "t.png")
    ten = _make_tensor_npz(tmp_path / "t.npz")
    with patch("summicron.depth.estimate.extract_lidar_depth", return_value=None), \
         patch("summicron.depth.estimate.load_depth_model", side_effect=ImportError("x")), \
         pytest.raises(ValueError, match="monocular fallback failed"):
        cli_main(["render", "--image", str(img), "--tensor", str(ten),
                  "--out-dir", str(tmp_path / "o")])


def test_explicit_monocular_message_prefix(tmp_path):
    with patch("summicron.depth.estimate.load_depth_model", side_effect=OSError("offline")):
        img = _png(tmp_path / "t.png")
        ten = _make_tensor_npz(tmp_path / "t.npz")
        with pytest.raises(ValueError, match="^monocular depth failed"):
            render_image(RenderOptions(image_path=img, tensor_path=ten,
                                       depth_source="monocular", save=False))
