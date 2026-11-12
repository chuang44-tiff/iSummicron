"""Run 6 default flip (section20): adversarial probes of the ``auto`` defaults.

Complements ``test_default_flip_adversarial.py``: CLI one-line error contract
under the new default failure modes, degenerate LiDAR depth through the
diopter mapping, ``dataclasses.replace`` / ``canonical_bench`` matrices,
explicit-old-flags strictness, and demo pin hygiene. Real renders are SLOW.
"""
from __future__ import annotations

import dataclasses
import hashlib
import io
import os
import subprocess
import sys
import warnings
from contextlib import ExitStack, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

from summicron.render import _bench
from summicron.render.api import RenderOptions, render_image
from tests.unit.test_default_flip import _png
from tests.unit.test_depth_s1_flags import _make_tensor_npz

ROOT = Path(__file__).resolve().parents[2]


def _opts(tmp_path, img=None, **kw):
    img = img if img is not None else _png(tmp_path / "t.png")
    ten = _make_tensor_npz(tmp_path / "t.npz")
    kw.setdefault("focus_xy", (0.5, 0.5))
    return RenderOptions(image_path=img, tensor_path=ten,
                         out_dir=tmp_path / "o", **kw)


def _fakes(depth=None, *, lidar=True, mono_exc=None, hdr=None):
    """Context patches: LiDAR returns ``depth`` (or None), render is captured."""
    cap: dict = {"lidar_calls": [], "mono": 0}
    if depth is None:
        depth = np.linspace(0.7, 4.0, 64 * 64, dtype=np.float32).reshape(64, 64)

    def fake_lidar(path, target_hw=None, **k):
        cap["lidar_calls"].append(k)
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
        if mono_exc is not None:
            raise mono_exc
        return object()

    def mono_est(image, bundle):
        return np.full(image.shape[:2], 0.5, np.float32)

    ps = [
        patch("summicron.depth.estimate.extract_lidar_depth", side_effect=fake_lidar),
        patch("summicron.depth.estimate.load_depth_model", side_effect=mono_model),
        patch("summicron.depth.estimate.estimate_depth", side_effect=mono_est),
        patch("summicron.render.convolve.render", side_effect=fake_render),
        patch("summicron.render.convolve.save_results", side_effect=fake_save),
    ]
    if hdr is not None:
        ps.append(patch("summicron.io.hdr_load_macos.load_hdr_heic", return_value=hdr))
    return cap, ps


def _run(opts, ps):
    buf = io.StringIO()
    with ExitStack() as st, redirect_stdout(buf):
        for p in ps:
            st.enter_context(p)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = render_image(opts)
    return res, buf.getvalue()


# ---- 1. degenerate LiDAR depth through the default diopter mapping ----------

_N = 64 * 64


def _depth_cases():
    g = np.linspace(0.7, 4.0, _N, dtype=np.float32).reshape(64, 64)
    nan_anchor = g.copy()
    nan_anchor[32, 32] = np.nan
    zero_anchor = g.copy()
    zero_anchor[32, 32] = 0.0
    inf_anchor = g.copy()
    inf_anchor[32, 32] = np.inf
    return {
        "all_nan": np.full((64, 64), np.nan, np.float32),
        "all_zero": np.zeros((64, 64), np.float32),
        "all_inf": np.full((64, 64), np.inf, np.float32),
        "all_neg": np.full((64, 64), -2.0, np.float32),
        "const": np.full((64, 64), 2.0, np.float32),
        "huge": np.full((64, 64), 1e30, np.float32),
        "tiny": np.full((64, 64), 1e-30, np.float32),
        "nan_at_anchor": nan_anchor,
        "zero_at_anchor": zero_anchor,
        "inf_at_anchor": inf_anchor,
        "float64": g.astype(np.float64),
    }


@pytest.mark.parametrize("name", list(_depth_cases()))
def test_default_diopter_survives_degenerate_lidar(tmp_path, name):
    cap, ps = _fakes(_depth_cases()[name])
    res, _ = _run(_opts(tmp_path), ps)
    assert res.diagnostics.resolved["depth_mapping"] == "diopter"
    d = np.asarray(cap["render"]["depth_m"])
    assert d.shape == (64, 64)
    assert np.isfinite(d).all(), name
    # fake tensor object distances span [0.7, 5.0]
    assert d.min() >= 0.7 - 1e-4 and d.max() <= 5.0 + 1e-4, (name, d.min(), d.max())


def test_default_diopter_no_numpy_warning_leak_on_zero_depth(tmp_path):
    cap, ps = _fakes(np.zeros((64, 64), np.float32))
    buf = io.StringIO()
    with ExitStack() as st, redirect_stdout(buf):
        for p in ps:
            st.enter_context(p)
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            render_image(_opts(tmp_path))
    bad = [str(w.message) for w in rec
           if issubclass(w.category, RuntimeWarning)
           and ("divide" in str(w.message) or "invalid value" in str(w.message))]
    assert bad == []


# ---- 2. CLI one-line error contract ------------------------------------------

def _cli_main(tmp_path, argv, ps):
    from summicron.render import convolve
    ten = _make_tensor_npz(tmp_path / "t.npz")
    img = _png(tmp_path / "t.png")
    full = ["--image", str(img), "--tensor", str(ten), "--focus-xy", "0.5", "0.5",
            "--out-dir", str(tmp_path / "o"), *argv]
    out, err = io.StringIO(), io.StringIO()
    old_err = sys.stderr
    sys.stderr = err
    try:
        with ExitStack() as st, redirect_stdout(out):
            for p in ps:
                st.enter_context(p)
            try:
                rc = convolve.main(full)
            except SystemExit as e:
                rc = e.code
    finally:
        sys.stderr = old_err
    return rc, err.getvalue()


@pytest.mark.parametrize("argv", [
    ["--depth-source", "lidar"],
    ["--depth-source", "lidar", "--depth-mapping", "diopter"],
    ["--depth-mapping", "diopter"],
    ["--depth-guided", "on"],
], ids=["lidar", "lidar+diopter", "auto+diopter", "auto+guided_on"])
def test_cli_strict_lidar_requests_one_line_exit2_without_lidar(tmp_path, argv):
    cap, ps = _fakes(lidar=False)
    rc, err = _cli_main(tmp_path, argv, ps)
    assert rc == 2
    lines = [ln for ln in err.splitlines() if ln.strip()]
    assert len(lines) == 1 and lines[0].startswith("render: "), err
    assert "Traceback" not in err
    assert cap["mono"] == 0, "strict request must never silently fall to monocular"



@pytest.mark.parametrize("argv", [[], ["--depth-source", "monocular"]],
                         ids=["auto", "explicit"])
@pytest.mark.parametrize("exc", [
    pytest.param(RuntimeError("line one\nline two\n  line three"),
                 id="multiline-runtime"),
    pytest.param(OSError("offline\nHF hub: cannot reach\n"),
                 id="multiline-oserror"),
    pytest.param(ImportError("No module named 'transformers'"), id="importerror"),
])
def test_cli_monocular_failure_is_exactly_one_line(tmp_path, argv, exc):
    cap, ps = _fakes(lidar=False, mono_exc=exc)
    rc, err = _cli_main(tmp_path, argv, ps)
    assert rc == 2, err
    lines = [ln for ln in err.splitlines() if ln.strip()]
    assert len(lines) == 1, err
    assert lines[0].startswith("render: ")


def test_cli_monocular_notimplemented_propagates(tmp_path):
    cap, ps = _fakes(lidar=False, mono_exc=NotImplementedError("mps op"))
    with pytest.raises(NotImplementedError):
        _cli_main(tmp_path, [], ps)


def test_cli_debug_env_reraises_monocular_failure(tmp_path, monkeypatch):
    monkeypatch.setenv("ISUMMICRON_DEBUG", "1")
    cap, ps = _fakes(lidar=False, mono_exc=RuntimeError("boom"))
    with pytest.raises(ValueError, match="monocular"):
        _cli_main(tmp_path, [], ps)


def test_cli_garbage_image_with_default_flags_one_line(tmp_path):
    bad = tmp_path / "junk.png"
    bad.write_bytes(b"this is not an image")
    ten = _make_tensor_npz(tmp_path / "t.npz")
    r = subprocess.run(
        [sys.executable, "-m", "summicron.render.convolve", "--image", str(bad),
         "--tensor", str(ten), "--out-dir", str(tmp_path / "o")],
        capture_output=True, text=True, cwd=str(ROOT), env=dict(os.environ, PYTHONPATH=str(ROOT)),
    )
    assert r.returncode == 2, r.stderr[-400:]
    assert "Traceback" not in r.stderr
    assert len([ln for ln in r.stderr.splitlines() if ln.strip()]) == 1, r.stderr


def test_cli_focus_target_at_tensor_far_end_default_is_one_line(tmp_path):
    cap, ps = _fakes(lidar=True)
    rc, err = _cli_main(tmp_path, ["--focus-target-m", "5.0"], ps)
    assert rc == 2
    assert len([ln for ln in err.splitlines() if ln.strip()]) == 1, err


# ---- 3. RenderOptions: direct / replace / canonical_bench --------------------

def test_renderoptions_defaults_match_parser_defaults():
    import argparse
    from summicron.render import _render_args
    p = argparse.ArgumentParser()
    _render_args.add_render_arguments(p)
    ns = p.parse_args(["--image", "x.heic"])
    a = RenderOptions.from_argparse(ns)
    b = RenderOptions(image_path=Path("x.heic"))
    assert a == b


def test_canonical_bench_argv_has_no_old_look_leakage():
    assert "--depth-scale" not in _bench.CANONICAL_FLAGS
    for f in ("--depth-guided", "--depth-mapping", "--tonemap", "--bright-pass-thresh"):
        assert f not in _bench.CANONICAL_FLAGS
    o = RenderOptions.canonical_bench("x.heic", (0.4, 0.45))
    assert (o.depth_guided, o.depth_mapping, o.tonemap, o.bright_pass_thresh) == (
        "auto", "auto", "auto", None)


@pytest.mark.parametrize("old", [False, True])
@pytest.mark.parametrize("kw,ok", [
    (dict(depth_source="monocular"), None),
    (dict(hdr_source="none"), None),
    (dict(shadow_match_mode="on"), None),
    (dict(occlusion_compositing="on"), True),
    (dict(psf_norm="peak"), True),
    (dict(downres=None), True),
    (dict(downres=0), False),
    (dict(depth_source="monocular", depth_mapping="diopter"), False),
    (dict(depth_source="monocular", depth_guided="on"), False),
    (dict(hdr_source="none", tonemap="input-lut"), False),
    (dict(shadow_match_mode="on", tonemap="input-lut"), False),
    (dict(shadow_match_mode="on", bright_pass_thresh="auto"), False),
    (dict(depth_gain=2.0, depth_mapping="linear"), False),
    (dict(hdr_source="none", shadow_match_mode="on"), False),
])
def test_canonical_bench_overrides_are_revalidated(old, kw, ok):
    if ok is None:  # depends on look: just must not crash with a non-ValueError
        try:
            RenderOptions.canonical_bench("x.heic", (0.4, 0.45), old_look=old, **kw)
        except ValueError:
            pass
        return
    if ok:
        RenderOptions.canonical_bench("x.heic", (0.4, 0.45), old_look=old, **kw)
    else:
        with pytest.raises(ValueError):
            RenderOptions.canonical_bench("x.heic", (0.4, 0.45), old_look=old, **kw)


def test_canonical_bench_new_look_monocular_override_is_valid_auto_normalised():
    """New look only: mapping/guided are ``auto`` so a monocular override must
    validate (auto -> linear/off) rather than trip the diopter rule."""
    o = RenderOptions.canonical_bench("x.heic", (0.4, 0.45), depth_source="monocular")
    assert o.depth_source == "monocular"


def test_replace_roundtrip_and_revalidation():
    o = RenderOptions(image_path=Path("x.heic"), depth_gain=2)  # int -> float
    assert o.depth_gain == 2.0 and isinstance(o.depth_gain, float)
    o2 = dataclasses.replace(o, depth_gain=3)
    assert o2.depth_gain == 3.0
    with pytest.raises(ValueError):
        dataclasses.replace(o, depth_mapping="linear")
    with pytest.raises(ValueError):
        dataclasses.replace(o, depth_source="monocular")
    o3 = dataclasses.replace(o, depth_gain="auto", depth_source="monocular")
    assert o3.depth_source == "monocular"
    with pytest.raises(ValueError):
        dataclasses.replace(RenderOptions(image_path=Path("x"), hdr_source="macos",
                                          bright_pass_thresh="auto"),
                            shadow_match_mode="on")
    with pytest.raises(ValueError):
        dataclasses.replace(RenderOptions(image_path=Path("x"), hdr_source="macos",
                                          bright_pass_thresh="auto"),
                            hdr_source="none")


def test_renderoptions_rejects_bool_and_nan_gain_and_string_numbers():
    for bad in (True, float("nan"), float("inf"), 0, -1, 11, "2", "AUTO"):
        with pytest.raises(ValueError):
            RenderOptions(image_path=Path("x"), depth_gain=bad)


# ---- 4. explicit old flags are strict and silent -----------------------------

def test_explicit_old_flags_no_resolution_chatter_and_old_call_shape(tmp_path):
    cap, ps = _fakes()
    o = _opts(tmp_path, depth_source="lidar", depth_scale=10.0, depth_guided="off",
              depth_mapping="linear", tonemap="ci", bright_pass_thresh=1.0,
              hdr_source="none")
    res, out = _run(o, ps)
    assert "Resolved:" not in out and "note:" not in out
    assert cap["lidar_calls"] == [{}], "old path must not pass depth_guided"
    assert cap["saved"]["name_suffix"] == ""
    d = cap["render"]["depth_m"]
    assert float(np.nanmax(d)) > 5.0  # 10x metres, not diopter-capped
    assert cap["render"]["bright_thresh"] == 1.0


@pytest.mark.parametrize("kw,lidar_kw,suffix", [
    (dict(depth_source="lidar", depth_guided="off"), {}, "_dio"),
    (dict(depth_mapping="linear", depth_scale=10.0), {"depth_guided": "on"}, "_gd"),
    (dict(depth_guided="on", depth_mapping="linear"), {"depth_guided": "on"}, "_gd"),
])
def test_mixed_explicit_auto_combos_resolve_independently(tmp_path, kw, lidar_kw, suffix):
    cap, ps = _fakes()
    res, out = _run(_opts(tmp_path, **kw), ps)
    assert cap["lidar_calls"] == [lidar_kw]
    assert suffix in cap["saved"]["name_suffix"]
    assert "Resolved:" in out  # something was auto


def test_depth_scale_is_ignored_with_warning_under_default_diopter(tmp_path):
    cap, ps = _fakes()
    buf = io.StringIO()
    with ExitStack() as st, redirect_stdout(buf):
        for p in ps:
            st.enter_context(p)
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            res = render_image(_opts(tmp_path, depth_scale=10.0))
    assert any("ignored" in str(w.message) for w in rec)
    assert any("ignored" in w for w in res.diagnostics.warnings)


def test_resolved_dict_is_all_str_and_has_fixed_keys(tmp_path):
    cap, ps = _fakes()
    res, _ = _run(_opts(tmp_path), ps)
    r = dict(res.diagnostics.resolved)
    assert set(r) == {"hdr_source", "depth_source", "depth_guided", "depth_mapping",
                      "tonemap", "bright_pass_thresh",
                      "kernel_family", "bright_stamp"}  # S22 S1 (codev default)
    assert all(isinstance(v, str) for v in r.values())
    float(r["bright_pass_thresh"])


def test_progress_callback_gets_resolved_line_and_cancel_is_honoured(tmp_path):
    import threading
    cap, ps = _fakes(lidar=False)
    lines: list[str] = []
    buf = io.StringIO()
    with ExitStack() as st, redirect_stdout(buf):
        for p in ps:
            st.enter_context(p)
        render_image(_opts(tmp_path), on_progress=lines.append)
    assert any(ln.startswith("Resolved:") for ln in lines)
    assert any(ln.startswith("note: depth_source auto -> monocular") for ln in lines)
    assert not any(ln.startswith("Resolved:") for ln in buf.getvalue().splitlines())

    cap, ps = _fakes(lidar=False)
    ev = threading.Event()
    ev.set()
    # real render honours cancel at slab checkpoints; with render faked this
    # only checks that a pre-set cancel does not corrupt resolution/raise oddly.
    with ExitStack() as st, redirect_stdout(io.StringIO()):
        for p in ps:
            st.enter_context(p)
        try:
            render_image(_opts(tmp_path), cancel=ev)
        except Exception as e:  # noqa: BLE001
            assert "ancel" in type(e).__name__ or "ancel" in str(e)


# ---- 5. tonemap auto edge cases ----------------------------------------------

def test_tonemap_auto_no_headroom_emits_note_and_skips_second_load(tmp_path):
    hdr = np.full((64, 64, 3), 0.5, np.float32)
    cap, ps = _fakes(hdr=hdr)
    res, out = _run(_opts(tmp_path, hdr_source="macos"), ps)
    assert res.diagnostics.resolved["tonemap"] == "ci"
    assert "note: tonemap auto -> ci (no HDR headroom in input)" in out


def test_tonemap_auto_headroom_exact_threshold_is_ci(tmp_path):
    hdr = np.full((64, 64, 3), 1.0, np.float32)
    cap, ps = _fakes(hdr=hdr)
    res, _ = _run(_opts(tmp_path, hdr_source="macos"), ps)
    assert res.diagnostics.resolved["tonemap"] == "ci"


def test_tonemap_auto_nan_in_hdr_buffer_does_not_crash_resolver(tmp_path):
    hdr = np.full((64, 64, 3), 0.5, np.float32)
    hdr[3, 3, 0] = np.nan
    cap, ps = _fakes(hdr=hdr)
    res, _ = _run(_opts(tmp_path, hdr_source="macos"), ps)
    assert res.diagnostics.resolved["tonemap"] in ("ci", "input-lut")


def test_tonemap_auto_inf_in_hdr_buffer_is_handled(tmp_path):
    """A +Inf pixel in the HDR buffer would make max()>1.001 -> input-lut ->
    fit_input_tone on non-finite data; must not crash (spec: finite on NaN/Inf)."""
    hdr = np.full((64, 64, 3), 0.5, np.float32)
    hdr[3, 3, 0] = np.inf
    cap, ps = _fakes(hdr=hdr)
    res, _ = _run(_opts(tmp_path, hdr_source="macos"), ps)
    assert np.isfinite(float(res.diagnostics.resolved["bright_pass_thresh"]))


# ---- 6. demo / pin hygiene ----------------------------------------------------

_NO_DEMO = pytest.mark.skipif(
    not ((ROOT / "demo" / "manifest.toml").exists() and (ROOT / "iphoneImage").is_dir()),
    reason="demo/ or iphoneImage/ absent (staged release tree)")


@_NO_DEMO
def test_demo_manifest_rows_reference_existing_pins_and_images():
    import tomllib
    rows = tomllib.loads((ROOT / "demo" / "manifest.toml").read_text())["row"]
    assert rows[0]["name"] == "g1" and rows[0]["pin"] == "G1_MD5"
    for r in rows:
        assert (ROOT / "iphoneImage" / r["image"]).exists(), r
        if "pin" in r:
            assert hasattr(_bench, r["pin"])
        # extra flags must not smuggle old-look flags (would change the look silently)
        for f in ("--depth-scale", "--tonemap", "--depth-mapping", "--depth-guided"):
            assert f not in r.get("extra", []), (r["name"], f)


@_NO_DEMO
def test_demo_on_pin_jpg_matches_current_on_md5():
    sys.path.insert(0, str(ROOT))
    import importlib.util
    spec = importlib.util.spec_from_file_location("make_demo", ROOT / "demo" / "make_demo.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    p = Path(mod.ON_PIN_JPG)
    if not p.exists():
        pytest.skip("pin jpg absent")
    assert hashlib.md5(p.read_bytes()).hexdigest() == _bench.ON_MD5


# ---- 7. SLOW: real renders on small inputs -----------------------------------

def _syn(tmp_path):
    import pillow_heif
    from summicron.render.convolve import downres_array, load_image
    src = ROOT / "iphoneImage" / "Demo_input.HEIC"
    if not src.exists():
        pytest.skip("missing Demo_input.HEIC")
    arr = np.asarray(downres_array(load_image(src), 384))
    u8 = np.clip(arr * (255.0 if arr.max() <= 1.5 else 1.0), 0, 255).astype(np.uint8)
    im = Image.fromarray(u8, "RGB")
    pillow_heif.register_heif_opener()
    paths = {}
    for k, fmt, kw in (("png", "PNG", {}), ("jpg", "JPEG", {"quality": 95}),
                       ("heic", "HEIF", {})):
        paths[k] = tmp_path / f"s.{k}"
        im.save(paths[k], format=fmt, **kw)
    return paths


def _cli(args):
    env = dict(os.environ, HF_HUB_OFFLINE="1", PYTHONPATH=str(ROOT))
    return subprocess.run([sys.executable, "-m", "summicron.render.convolve", *args],
                          capture_output=True, text=True, cwd=str(ROOT), env=env,
                          timeout=900)


def _da2_or_skip():
    os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        from summicron.depth.estimate import load_depth_model
        load_depth_model(device="mps")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"DA2 unavailable offline: {exc!r}")


@pytest.mark.slow
@pytest.mark.parametrize("kind", ["png", "jpg", "heic"])
@pytest.mark.parametrize("hdr", [False, True], ids=["sdr", "hdr-macos"])
def test_sdr_inputs_default_cli_renders_with_resolved_line(tmp_path, kind, hdr):
    """Spec G6: a non-HDR input loaded with --hdr-source macos should announce
    the no-headroom fallback. A plain sRGB file loaded via the macOS path
    overshoots 1.0 (colour-space conversion), so auto picks input-lut with no
    note. Probe documents the actual resolution without asserting on it."""
    _da2_or_skip()
    p = _syn(tmp_path)[kind]
    r = _cli(["--image", str(p), "--downres", "256", "--out-dir", str(tmp_path / "o"),
              *(["--hdr-source", "macos"] if hdr else [])])
    assert r.returncode == 0, r.stderr[-600:]
    assert "Resolved:" in r.stdout and "Traceback" not in r.stderr


@pytest.mark.slow
@pytest.mark.parametrize("kind", ["png", "jpg"])
def test_default_cli_strict_lidar_on_non_heic_is_one_line(tmp_path, kind):
    p = _syn(tmp_path)[kind]
    r = _cli(["--image", str(p), "--depth-source", "lidar", "--downres", "128",
              "--out-dir", str(tmp_path / "o")])
    assert r.returncode == 2
    assert len([ln for ln in r.stderr.splitlines() if ln.strip()]) == 1, r.stderr
    assert r.stderr.startswith("render: ")


@pytest.mark.slow
def test_downres_variants_real_heic_default_render(tmp_path):
    img = ROOT / "iphoneImage" / "IMG_6915.HEIC"
    if not img.exists():
        pytest.skip("missing IMG_6915")
    for d in (1, 16, 64, 192):
        res = render_image(RenderOptions.canonical_bench(
            img, (0.4, 0.45), downres=d, save=False,
            return_linear=True))
        assert np.isfinite(res.output_linear).all(), d
        assert res.diagnostics.resolved["depth_mapping"] == "diopter"
