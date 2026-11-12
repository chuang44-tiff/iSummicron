"""Rev 3 (section20) adversarial probes of ``--hdr-source auto``.

Fast, fake-render probes (no real renders). Bugs are strict xfails.
"""
from __future__ import annotations

import dataclasses
import io
import sys
from contextlib import ExitStack, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

from summicron.render import _hdr_detect as hl
from summicron.render.api import RenderOptions, render_image
from tests.unit.test_depth_s1_flags import _make_tensor_npz, _parse
from tests.unit.test_hdr_auto import _fakes, _opts, _png, _run

ROOT = Path(__file__).resolve().parents[2]
IPH = ROOT / "iphoneImage"
GM = patch("summicron.render._hdr_detect.has_hdr_gainmap", return_value=True)
LD = patch("summicron.render._hdr_detect.macos_hdr_loader_available",
           return_value=True)
LUT = patch("summicron.io.tone_lut.fit_input_tone",
            return_value=(np.array([0, 1.0]), np.array([0, 1.0]),
                          np.array([1.0]), np.array([1.0])))


def _hdr(v=0.5, peak=2.0):
    a = np.full((64, 64, 3), v, np.float32)
    a[0, 0] = peak
    return a


def _hdr_fakes(**kw):
    cap, ps = _fakes(hdr=_hdr(), **kw)
    return cap, ps + [GM, LD, LUT]


# ---- detection on real / synthetic files -------------------------------------

@pytest.mark.skipif(sys.platform != "darwin", reason="macOS")
@pytest.mark.parametrize("name", ["Demo_input.HEIC", "IMG_6915.HEIC",
                                  "IMG_4343.heic", "IMG_6651.HEIC"])
def test_real_scenes_detect_gainmap_and_loader(name):
    p = IPH / name
    if not p.exists():
        pytest.skip("scene missing")
    assert hl.has_hdr_gainmap(p) is True
    assert hl.has_hdr_gainmap(str(p)) is True
    assert hl.macos_hdr_loader_available() is True


def _sdr_heic(path):
    pillow_heif = pytest.importorskip("pillow_heif")
    pillow_heif.register_heif_opener()
    Image.fromarray(np.full((64, 64, 3), 100, np.uint8)).save(path, format="HEIF")
    return path


def test_sdr_heic_without_gainmap_is_not_hdr(tmp_path):
    p = _sdr_heic(tmp_path / "sdr.heic")
    assert hl.has_hdr_gainmap(p) is False


def test_sdr_heic_auto_pipeline_resolves_none(tmp_path):
    p = _sdr_heic(tmp_path / "sdr.heic")
    cap, ps = _fakes()
    o = dataclasses.replace(_opts(tmp_path), image_path=p)
    res, out = _run(o, ps)
    assert res.diagnostics.resolved["hdr_source"] == "none"
    assert res.mode == "ldr" and out.count("note: hdr_source auto -> none") == 1


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS")
def test_gainmap_heic_renamed_jpg_or_uppercase_suffix():
    src = IPH / "IMG_6915.HEIC"
    if not src.exists():
        pytest.skip("scene missing")
    import shutil
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        for suf, want in ((".jpg", False), (".HEIF", True), (".HeIc", True)):
            q = Path(d) / f"x{suf}"
            shutil.copy(src, q)
            assert hl.has_hdr_gainmap(q) is want, suf


@pytest.mark.parametrize("ext", ["jpg", "jpeg", "png"])
def test_sdr_raster_auto_resolves_none_even_with_loader(tmp_path, ext):
    p = tmp_path / f"a.{ext}"
    Image.fromarray(np.full((64, 64, 3), 128, np.uint8)).save(p)
    cap, ps = _fakes()
    ps += [LD]
    o = dataclasses.replace(_opts(tmp_path), image_path=p)
    res, out = _run(o, ps)
    assert res.diagnostics.resolved["hdr_source"] == "none"
    assert "no HDR gain map" in out and cap["load_hdr"] == 0


def test_detect_survives_hostile_find_spec(monkeypatch):
    import importlib.util as iu
    monkeypatch.setattr(sys, "platform", "darwin")
    for exc in (ValueError("x"), ModuleNotFoundError("Quartz"), ImportError("y")):
        monkeypatch.setattr(iu, "find_spec",
                            lambda *a, _e=exc, **k: (_ for _ in ()).throw(_e))
        assert hl.macos_hdr_loader_available() is False
    monkeypatch.setattr(iu, "find_spec", lambda n, *a, **k: None)
    assert hl.macos_hdr_loader_available() is False


def test_detect_requires_both_modules(monkeypatch):
    import importlib.util as iu
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(iu, "find_spec",
                        lambda n, *a, **k: object() if n == "Quartz" else None)
    assert hl.macos_hdr_loader_available() is False


# ---- simulated non-macOS / pyobjc missing -------------------------------------

def test_auto_on_linux_falls_back_without_touching_loader(tmp_path, monkeypatch):
    cap, ps = _fakes()
    ps += [GM]
    monkeypatch.setattr(sys, "platform", "linux")
    res, out = _run(_opts(tmp_path), ps)
    assert res.diagnostics.resolved["hdr_source"] == "none"
    assert "loader unavailable" in out and cap["load_hdr"] == 0


def test_explicit_macos_on_linux_is_one_line_value_error(tmp_path, monkeypatch):
    cap, ps = _fakes()
    monkeypatch.setattr(sys, "platform", "linux")
    with pytest.raises(ValueError, match="requires macOS") as ei:
        _run(_opts(tmp_path, hdr_source="macos"), ps)
    assert "\n" not in str(ei.value)


def _cli(tmp_path, argv, ps, img=None):
    from summicron.render import convolve
    ten = _make_tensor_npz(tmp_path / "t.npz")
    img = img or _png(tmp_path / "t.png")
    full = ["--image", str(img), "--tensor", str(ten), "--focus-xy", "0.5", "0.5",
            "--out-dir", str(tmp_path / "o"), *argv]
    out, err = io.StringIO(), io.StringIO()
    old = sys.stderr
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
        sys.stderr = old
    return rc, err.getvalue()


def test_cli_explicit_macos_on_linux_one_line(tmp_path, monkeypatch):
    cap, ps = _fakes()
    monkeypatch.setattr(sys, "platform", "linux")
    rc, err = _cli(tmp_path, ["--hdr-source", "macos"], ps)
    assert rc == 2 and "requires macOS" in err
    assert len([l for l in err.splitlines() if "requires macOS" in l]) == 1


def _import_error_loader():
    return patch("summicron.io.hdr_load_macos.load_hdr_heic", side_effect=ImportError(
        "macOS HDR loader requires `pip install isummicron[hdr-macos]`"))


def test_cli_explicit_macos_pyobjc_missing_is_one_line(tmp_path):
    cap, ps = _fakes()
    ps += [_import_error_loader()]
    rc, err = _cli(tmp_path, ["--hdr-source", "macos"], ps)
    assert rc == 2 and err.startswith("render:") and err.count("\n") == 1


def test_cli_auto_broken_pyobjc_degrades_gracefully(tmp_path):
    cap, ps = _fakes()
    ps += [GM, LD, _import_error_loader()]
    rc, err = _cli(tmp_path, [], ps)
    assert rc in (0, 2)


# ---- loader failure mid-load ---------------------------------------------------

def test_cli_auto_loader_runtimeerror_is_one_line_exit2(tmp_path):
    cap, ps = _fakes()
    ps += [GM, LD, patch("summicron.io.hdr_load_macos.load_hdr_heic",
                         side_effect=RuntimeError("CIImage failed to load: x"))]
    rc, err = _cli(tmp_path, [], ps)
    assert rc == 2 and err.startswith("render:") and err.count("\n") == 1


def test_loader_runtimeerror_propagates_via_api(tmp_path):
    cap, ps = _fakes()
    ps += [GM, LD, patch("summicron.io.hdr_load_macos.load_hdr_heic",
                         side_effect=RuntimeError("boom"))]
    with pytest.raises(RuntimeError, match="boom"):
        _run(_opts(tmp_path), ps)


def test_loader_not_implemented_propagates(tmp_path):
    cap, ps = _fakes()
    ps += [GM, LD, patch("summicron.io.hdr_load_macos.load_hdr_heic",
                         side_effect=NotImplementedError("x"))]
    with pytest.raises(NotImplementedError):
        _run(_opts(tmp_path), ps)


def test_loader_nan_inf_buffer_auto_ok(tmp_path):
    h = _hdr()
    h[1, 1, 2] = np.nan
    h[2, 2, 0] = np.inf
    cap, ps = _fakes(hdr=h)
    ps += [GM, LD]
    res, _ = _run(_opts(tmp_path), ps)
    assert res.diagnostics.resolved["tonemap"] == "ci"


# ---- explicit values --------------------------------------------------------------

def test_explicit_none_with_gainmap_never_loads_hdr(tmp_path):
    cap, ps = _hdr_fakes()
    res, out = _run(_opts(tmp_path, hdr_source="none"), ps)
    assert cap["load_hdr"] == 0 and cap["render"]["hdr_source"] == "none"
    assert "note: hdr_source" not in out and res.mode == "ldr"


def test_explicit_macos_on_png_is_strict_and_calls_loader(tmp_path):
    cap, ps = _fakes(hdr=_hdr())
    res, out = _run(_opts(tmp_path, hdr_source="macos"), ps)
    assert cap["load_hdr"] == 1 and "note: hdr_source" not in out


def test_explicit_macos_does_not_consult_detectors(tmp_path):
    cap, ps = _fakes(hdr=_hdr())
    g = patch("summicron.render._hdr_detect.has_hdr_gainmap", side_effect=AssertionError)
    l = patch("summicron.render._hdr_detect.macos_hdr_loader_available",
              side_effect=AssertionError)
    _run(_opts(tmp_path, hdr_source="macos"), ps + [g, l])


# ---- shadow match / tonemap with auto -----------------------------------------------

def test_shadow_match_on_with_auto_gainmap(tmp_path):
    cap, ps = _hdr_fakes()
    ps.append(patch("summicron.render.convolve.load_image",
                    return_value=np.full((64, 64, 3), .5, np.float32)))
    try:
        res, _ = _run(_opts(tmp_path, shadow_match_mode="on"), ps)
    except ValueError:
        pytest.fail("shadow-match on + auto + HDR input must not be rejected")
    assert res.diagnostics.resolved["hdr_source"] == "macos"
    assert res.diagnostics.resolved["tonemap"] == "ci"


@pytest.mark.parametrize("kw,pat", [({"shadow_match_mode": "on"}, "shadow-match"),
                                    ({"tonemap": "input-lut"}, "input-lut")])
def test_auto_none_with_hdr_only_feature_is_one_line_cli(tmp_path, kw, pat):
    cap, ps = _fakes()
    argv = (["--shadow-match-mode", "on"] if "shadow_match_mode" in kw
            else ["--tonemap", "input-lut"])
    rc, err = _cli(tmp_path, argv, ps)
    assert rc == 2 and err.startswith("render:") and err.count("\n") == 1
    assert pat in err


def test_input_lut_with_auto_on_linux_reason_in_message(tmp_path, monkeypatch):
    cap, ps = _fakes()
    monkeypatch.setattr(sys, "platform", "linux")
    with pytest.raises(ValueError, match="loader unavailable"):
        _run(_opts(tmp_path, tonemap="input-lut"), ps + [GM])


def test_tonemap_ci_explicit_with_auto_gainmap(tmp_path):
    cap, ps = _hdr_fakes()
    res, out = _run(_opts(tmp_path, tonemap="ci"), ps)
    assert res.diagnostics.resolved["hdr_source"] == "macos"
    assert res.diagnostics.resolved["tonemap"] == "ci"
    assert cap["load_hdr"] == 1


def test_tonemap_input_lut_explicit_with_auto_gainmap(tmp_path):
    cap, ps = _hdr_fakes()
    res, _ = _run(_opts(tmp_path, tonemap="input-lut"), ps)
    assert res.diagnostics.resolved["tonemap"] == "input-lut"


def test_auto_gainmap_headroom_selects_input_lut_else_ci(tmp_path):
    cap, ps = _hdr_fakes()
    res, _ = _run(_opts(tmp_path), ps)
    assert res.diagnostics.resolved["tonemap"] == "input-lut"
    cap, ps = _fakes(hdr=np.full((64, 64, 3), .5, np.float32))
    res, out = _run(_opts(tmp_path), ps + [GM, LD])
    assert res.diagnostics.resolved["tonemap"] == "ci"
    assert "no HDR headroom" in out


def test_bright_pass_numeric_untouched_by_auto(tmp_path):
    cap, ps = _hdr_fakes()
    _run(_opts(tmp_path, tonemap="ci", bright_pass_thresh=2.5), ps)
    assert cap["render"]["bright_thresh"] == 2.5


# ---- depth interplay ------------------------------------------------------------------

def test_gainmap_heic_without_lidar_uses_monocular(tmp_path):
    cap, ps = _hdr_fakes(lidar=False)
    ps += [
        patch("summicron.depth.estimate.load_depth_model", return_value=object()),
        patch("summicron.depth.estimate.estimate_depth",
              side_effect=lambda im, b: np.full(im.shape[:2], .5, np.float32)),
    ]
    res, out = _run(_opts(tmp_path), ps)
    assert res.diagnostics.resolved["hdr_source"] == "macos"
    assert res.diagnostics.resolved["depth_source"] == "monocular"


def test_gainmap_heic_without_lidar_explicit_lidar_strict(tmp_path):
    cap, ps = _hdr_fakes(lidar=False)
    with pytest.raises(ValueError):
        _run(_opts(tmp_path, depth_source="lidar"), ps)


# ---- old_look / defaults / determinism / ON ------------------------------------------

def test_old_look_untouched():
    from summicron.render import _bench
    o = RenderOptions.canonical_bench("x.heic", (0.4, 0.45), old_look=True)
    assert o.hdr_source == "macos" and o.tonemap == "ci"
    assert o.depth_guided == "off" and o.depth_mapping == "linear"
    assert o.depth_scale == 10.0 and o.bright_pass_thresh == 1.0
    assert "macos" in _bench.CANONICAL_FLAGS_OLD
    assert _bench.G1_MD5_OLD.startswith("a106801a")
    n = RenderOptions.canonical_bench("x.heic", (0.4, 0.45))
    assert n.hdr_source == "macos"


def test_old_look_pipeline_emits_no_auto_chatter(tmp_path):
    cap, ps = _hdr_fakes()
    o = _opts(tmp_path, depth_source="lidar", depth_scale=10.0, depth_guided="off",
              depth_mapping="linear", tonemap="ci", bright_pass_thresh=1.0,
              hdr_source="macos")
    res, out = _run(o, ps)
    assert "Resolved:" not in out and "note:" not in out


def test_renderoptions_defaults_equal_parser_defaults():
    from summicron.render.api import _ARGPARSE_ALIAS
    ns = _parse()
    o = RenderOptions(image_path=Path("x"))
    bad = []
    for f in dataclasses.fields(RenderOptions):
        a = _ARGPARSE_ALIAS.get(f.name, f.name)
        if not hasattr(ns, a):
            continue
        pv = getattr(ns, a)
        ov = getattr(o, f.name)
        if f.name in ("image_path", "tensor_path", "out_path", "out_dir"):
            continue
        if f.name == "focus_xy":
            continue
        if pv != ov and not (pv is None and ov is None):
            bad.append((f.name, pv, ov))
    assert not bad, bad
    assert RenderOptions.from_argparse(ns).hdr_source == o.hdr_source == "auto"


def test_pipeline_deterministic_and_idempotent(tmp_path):
    outs = []
    for i in range(2):
        cap, ps = _hdr_fakes()
        res, out = _run(_opts(tmp_path), ps)
        outs.append((out, dict(res.diagnostics.resolved), res.mode,
                     {k: v for k, v in cap["render"].items()
                      if not isinstance(v, np.ndarray)}))
    assert outs[0] == outs[1]
    assert hl.has_hdr_gainmap(IPH / "IMG_6915.HEIC") == hl.has_hdr_gainmap(IPH / "IMG_6915.HEIC")


def test_input_options_object_not_mutated(tmp_path):
    cap, ps = _fakes()
    o = _opts(tmp_path)
    before = dataclasses.asdict(o)
    _run(o, ps)
    assert dataclasses.asdict(o) == before and o.hdr_source == "auto"


def test_occlusion_on_with_auto_gainmap(tmp_path):
    cap, ps = _hdr_fakes()
    res, _ = _run(_opts(tmp_path, occlusion_compositing="on"), ps)
    assert res.diagnostics.resolved["hdr_source"] == "macos"
    assert cap["render"]["hdr_source"] == "macos"


def test_occlusion_on_with_auto_png(tmp_path):
    cap, ps = _fakes()
    res, _ = _run(_opts(tmp_path, occlusion_compositing="on"), ps)
    assert res.diagnostics.resolved["hdr_source"] == "none"


def test_note_text_single_line_and_before_load(tmp_path):
    cap, ps = _fakes()
    res, out = _run(_opts(tmp_path), ps)
    lines = out.splitlines()
    i = next(k for k, l in enumerate(lines) if l.startswith("note: hdr_source"))
    j = next(k for k, l in enumerate(lines) if l.startswith("Loading image"))
    assert i < j


def test_missing_image_auto_is_filenotfound_one_line(tmp_path):
    cap, ps = _fakes()
    rc, err = _cli(tmp_path, [], ps, img=tmp_path / "nope.heic")
    assert rc == 2 and err.startswith("render:") and err.count("\n") == 1


def test_directory_named_heic_auto(tmp_path):
    d = tmp_path / "dir.heic"
    d.mkdir()
    assert hl.has_hdr_gainmap(d) is False
