"""S1 tone fix G plumbing: --tonemap input-lut, --bright-pass-thresh auto."""
from __future__ import annotations

import io
import subprocess
import sys
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from summicron.render.api import RenderOptions, render_image
from tests.unit.test_depth_s1_flags import _make_tensor_npz, _parse


def _mk(**kw):
    return RenderOptions(image_path=Path("x.jpg"), **kw)


# ---- defaults / argparse -----------------------------------------------------

def test_defaults():
    ns = _parse()
    assert ns.tonemap == "auto" and ns.bright_pass_thresh is None
    o = RenderOptions.from_argparse(ns)
    assert o.tonemap == "auto" and o.bright_pass_thresh is None
    c = RenderOptions.canonical_bench("a.jpg", (0.4, 0.4))
    assert c.tonemap == "auto" and c.bright_pass_thresh is None


def test_argparse_values():
    o = RenderOptions.from_argparse(_parse("--tonemap", "input-lut", "--hdr-source", "macos",
                                           "--bright-pass-thresh", "auto"))
    assert o.tonemap == "input-lut" and o.bright_pass_thresh == "auto"
    assert RenderOptions.from_argparse(_parse("--bright-pass-thresh", "2.5")).bright_pass_thresh == 2.5
    with pytest.raises(SystemExit) as e:
        _parse("--bright-pass-thresh", "foo")
    assert e.value.code == 2
    with pytest.raises(SystemExit):
        _parse("--tonemap", "bogus")


# ---- validation --------------------------------------------------------------

@pytest.mark.parametrize("kw,flag", [
    (dict(tonemap="reinhard"), "--tonemap"),
    (dict(tonemap="input-lut", hdr_source="none"), "--tonemap"),
    (dict(tonemap="input-lut", hdr_source="macos", shadow_match_mode="on"), "--tonemap"),
    (dict(bright_pass_thresh="auto", hdr_source="none", tonemap="ci"), "--bright-pass-thresh"),
    (dict(bright_pass_thresh="auto", tonemap="ci"), "--bright-pass-thresh"),
    (dict(bright_pass_thresh=True), "--bright-pass-thresh"),
    (dict(bright_pass_thresh=float("nan")), "--bright-pass-thresh"),
    (dict(bright_pass_thresh=float("inf")), "--bright-pass-thresh"),
    (dict(bright_pass_thresh="x"), "--bright-pass-thresh"),
    (dict(bright_pass_thresh=[1.0]), "--bright-pass-thresh"),
])
def test_validation_rejects(kw, flag):
    with pytest.raises(ValueError, match=flag):
        _mk(**kw)


def test_validation_accepts():
    _mk()
    _mk(bright_pass_thresh=None)
    _mk(bright_pass_thresh=1)
    _mk(bright_pass_thresh=1e9)
    _mk(tonemap="input-lut", hdr_source="macos", bright_pass_thresh="auto")
    _mk(tonemap="input-lut", hdr_source="macos")


@pytest.mark.parametrize("argv", [
    ["--tonemap", "input-lut"],
    ["--bright-pass-thresh", "auto", "--tonemap", "ci"],
    ["--tonemap", "input-lut", "--hdr-source", "macos", "--shadow-match-mode", "on"],
])
def test_cli_validation_exit2(argv):
    r = subprocess.run(
        [sys.executable, "-m", "summicron.render.convolve", "--image", "nonexistent.jpg", *argv],
        capture_output=True, text=True,
    )
    assert r.returncode == 2
    assert "Traceback" not in r.stderr
    assert "--tonemap" in r.stderr or "--bright-pass-thresh" in r.stderr


# ---- pipeline ----------------------------------------------------------------

def _grad_image(path: Path) -> np.ndarray:
    from PIL import Image
    g = np.linspace(0, 255, 256, dtype=np.float32)
    arr = np.repeat(np.repeat(g[None, :, None], 128, axis=0), 3, axis=2).astype(np.uint8)
    Image.fromarray(arr, "RGB").save(str(path), "PNG")
    return arr


def _run(tmp_path, *, gain=3.0, **kw):
    from summicron.render.convolve import srgb_to_linear
    img = tmp_path / "t.png"
    sdr_u8 = _grad_image(img)
    hdr = (srgb_to_linear(sdr_u8.astype(np.float32) / 255.0) * gain).astype(np.float32)
    ten = _make_tensor_npz(tmp_path / "t.npz")
    cap: dict = {}

    def fake_render(image, tensor, **k):
        cap["render"] = k
        h, w = image.shape[:2]
        lin = image.astype(np.float32)
        return np.zeros((h, w, 3), np.uint8), lin

    def fake_save(u8, stem, out_dir, **k):
        cap["saved"] = k
        cap["u8"] = u8
        return out_dir / "x.jpg"

    depth = np.linspace(0.7, 4.0, 256 * 128, dtype=np.float32).reshape(128, 256)
    opts = RenderOptions(image_path=img, tensor_path=ten, focus_xy=(0.5, 0.5),
                         out_dir=tmp_path / "o", hdr_source="macos", **kw)
    sink: dict = {}
    loads = {"n": 0}
    import summicron.render.convolve as cv
    orig_load = cv.load_image

    def spy_load(p):
        loads["n"] += 1
        return orig_load(p)

    buf = io.StringIO()
    with patch("summicron.io.hdr_load_macos.load_hdr_heic", return_value=hdr), \
         patch("summicron.depth.estimate.extract_lidar_depth", return_value=depth), \
         patch("summicron.render.convolve.load_image", side_effect=spy_load), \
         patch("summicron.render.convolve.render", side_effect=fake_render), \
         patch("summicron.render.convolve.save_results", side_effect=fake_save), \
         redirect_stdout(buf):
        render_image(opts, _eval_sink=sink)
    cap.update(sink=sink, loads=loads["n"], out=buf.getvalue(), hdr=hdr, sdr_u8=sdr_u8)
    return cap


pytestmark_darwin = pytest.mark.skipif(sys.platform != "darwin", reason="hdr macos")


@pytestmark_darwin
def test_ci_default_thr_one_no_second_load_no_new_line(tmp_path):
    cap = _run(tmp_path, tonemap="ci")
    assert cap["render"]["bright_thresh"] == 1.0
    assert type(cap["render"]["bright_thresh"]) is float
    assert cap["sink"]["bright_thresh"] == 1.0
    assert cap["loads"] == 0  # hdr loaded via load_hdr_heic; no second load_image
    assert "bright-pass thr" not in cap["out"]
    assert "_tlut" not in cap["saved"]["name_suffix"]


@pytestmark_darwin
def test_ci_explicit_float_verbatim(tmp_path):
    cap = _run(tmp_path, bright_pass_thresh=2.5)
    assert cap["render"]["bright_thresh"] == 2.5


@pytestmark_darwin
def test_input_lut_auto_threshold_and_suffix(tmp_path):
    from summicron.io.tone_lut import auto_bright_threshold, fit_input_lut
    from summicron.render.convolve import srgb_to_linear
    cap = _run(tmp_path, tonemap="input-lut")
    sdr = srgb_to_linear(np.repeat(cap["sdr_u8"].astype(np.float32) / 255.0, 1, axis=2))
    xs, ys = fit_input_lut(cap["hdr"], sdr)
    thr, h = auto_bright_threshold(xs, ys)
    assert thr > 2.0  # non-identity LUT on a gain-3 scene
    assert cap["render"]["bright_thresh"] == pytest.approx(thr)
    assert cap["sink"]["bright_thresh"] == pytest.approx(thr)
    assert cap["loads"] == 1
    assert f"bright-pass thr: {thr:g} (auto" in cap["out"]
    assert cap["saved"]["name_suffix"].endswith("_tlut")
    assert cap["u8"].dtype == np.uint8


@pytestmark_darwin
def test_input_lut_auto_equals_unset_and_explicit_wins(tmp_path):
    a = _run(tmp_path, tonemap="input-lut")["render"]["bright_thresh"]
    b = _run(tmp_path, tonemap="input-lut", bright_pass_thresh="auto")["render"]["bright_thresh"]
    c = _run(tmp_path, tonemap="input-lut", bright_pass_thresh=1.7)
    assert a == b
    assert c["render"]["bright_thresh"] == 1.7
    assert "bright-pass thr" not in c["out"] or "(auto" not in c["out"]


@pytestmark_darwin
def test_input_lut_changes_output_vs_ci(tmp_path):
    d1 = tmp_path / "ci"
    d2 = tmp_path / "lut"
    d1.mkdir()
    d2.mkdir()
    ci = _run(d1, tonemap="ci")["u8"]
    lut = _run(d2, tonemap="input-lut")["u8"]
    assert lut.shape[:2] == (128, 256)
    assert not np.array_equal(ci, lut)


@pytestmark_darwin
def test_input_lut_reproduces_input_tone(tmp_path):
    """HDR = 3x linear SDR, render = identity -> LUT output ~ SDR base (T1-style)."""
    cap = _run(tmp_path, tonemap="input-lut")
    out = cap["u8"].astype(np.float32) / 255
    sdr = cap["sdr_u8"].astype(np.float32) / 255
    w = np.array([0.2126, 0.7152, 0.0722], np.float32)
    q = np.arange(1, 100)
    err = np.abs(np.percentile(out @ w, q) - np.percentile(sdr @ w, q)).mean()
    assert err < 0.03


@pytestmark_darwin
def test_no_headroom_note_and_threshold_range(tmp_path):
    cap = _run(tmp_path, gain=1.0, tonemap="input-lut")
    assert "no HDR headroom" in cap["out"]
    assert 1.0 <= cap["render"]["bright_thresh"] <= 1.12
