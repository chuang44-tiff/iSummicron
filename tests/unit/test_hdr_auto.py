"""Rev 3 (section20): ``--hdr-source auto`` default + Run 7 audit fixes."""
from __future__ import annotations

import io
import sys
import warnings
from contextlib import ExitStack, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

from summicron.render import _hdr_detect as hl
from summicron.render._resolve import resolve_auto, resolve_hdr_source
from summicron.render.api import RenderOptions, render_image
from tests.unit.test_depth_s1_flags import _make_tensor_npz, _parse

ROOT = Path(__file__).resolve().parents[2]


# ---- pure resolver -----------------------------------------------------------

@pytest.mark.parametrize("req,avail,has,want,note", [
    ("macos", False, False, "macos", False),
    ("macos", True, True, "macos", False),
    ("none", True, True, "none", False),
    ("auto", True, True, "macos", False),
    ("auto", False, True, "none", True),
    ("auto", True, False, "none", True),
    ("auto", False, False, "none", True),
])
def test_resolve_hdr_source_table(req, avail, has, want, note):
    v, n = resolve_hdr_source(req, loader_available=avail, has_hdr_data=has)
    assert v == want
    assert (n is not None) == note
    if n:
        assert n.startswith("note: hdr_source auto -> none (")
        assert "\n" not in n


def test_resolve_hdr_source_reasons():
    _, n1 = resolve_hdr_source("auto", loader_available=False, has_hdr_data=True)
    _, n2 = resolve_hdr_source("auto", loader_available=True, has_hdr_data=False)
    assert "loader unavailable" in n1
    assert "no HDR gain map" in n2


def test_resolve_auto_reports_hdr_source_and_line():
    o = RenderOptions(image_path=Path("x.heic"))
    m = resolve_auto(o, has_lidar=True, hdr_headroom=True, hdr_source="macos")
    assert m.hdr_source == "macos"
    assert m.tonemap == "input-lut"
    assert m.resolved_line().startswith("Resolved: hdr_source=macos ")
    m2 = resolve_auto(o, has_lidar=True, hdr_headroom=True, hdr_source="none")
    assert m2.tonemap == "ci" and m2.hdr_source == "none"


def test_resolve_auto_nonfinite_note():
    o = RenderOptions(image_path=Path("x.heic"))
    m = resolve_auto(o, has_lidar=True, hdr_headroom=False,
                     hdr_source="macos", nonfinite=True)
    assert m.tonemap == "ci"
    assert sum("non-finite" in n for n in m.notes) == 1
    assert not any("no HDR headroom" in n for n in m.notes)


# ---- defaults / validation -----------------------------------------------------

def test_default_hdr_source_is_auto_everywhere():
    assert RenderOptions(image_path=Path("x")).hdr_source == "auto"
    assert _parse().hdr_source == "auto"


def test_canonical_bench_keeps_explicit_macos():
    o = RenderOptions.canonical_bench("x.heic", (0.4, 0.4))
    assert o.hdr_source == "macos"
    o = RenderOptions.canonical_bench("x.heic", (0.4, 0.4), old_look=True)
    assert o.hdr_source == "macos"


def test_auto_accepts_input_lut_and_shadow_statically():
    RenderOptions(image_path=Path("x"), tonemap="input-lut")
    RenderOptions(image_path=Path("x"), shadow_match_mode="on")
    with pytest.raises(ValueError):
        RenderOptions(image_path=Path("x"), tonemap="input-lut", hdr_source="none")
    with pytest.raises(ValueError):
        RenderOptions(image_path=Path("x"), shadow_match_mode="on", hdr_source="none")
    with pytest.raises(ValueError):
        RenderOptions(image_path=Path("x"), hdr_source="bogus")


# ---- detection -----------------------------------------------------------------

def test_has_hdr_gainmap_false_for_non_heic_and_missing(tmp_path):
    p = tmp_path / "a.png"
    Image.new("RGB", (4, 4)).save(p)
    assert hl.has_hdr_gainmap(p) is False
    assert hl.has_hdr_gainmap(tmp_path / "nope.heic") is False
    bad = tmp_path / "bad.heic"
    bad.write_bytes(b"junk")
    assert hl.has_hdr_gainmap(bad) is False


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS")
def test_has_hdr_gainmap_real_scenes():
    for n in ("Demo_input.HEIC", "IMG_6915.HEIC"):
        p = ROOT / "iphoneImage" / n
        if p.exists():
            assert hl.has_hdr_gainmap(p) is True


def test_loader_available_false_off_darwin(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    assert hl.macos_hdr_loader_available() is False


# ---- pipeline with fakes ---------------------------------------------------------

def _fakes(hdr=None, lidar=True):
    cap: dict = {"load_hdr": 0}
    depth = np.linspace(0.7, 4.0, 64 * 64, dtype=np.float32).reshape(64, 64)

    def fake_render(image, tensor, **k):
        cap["render"] = k
        h, w = image.shape[:2]
        return np.zeros((h, w, 3), np.uint8), np.zeros((h, w, 3), np.float32)

    def fake_load_hdr(p):
        cap["load_hdr"] += 1
        return hdr.copy()

    ps = [
        patch("summicron.depth.estimate.extract_lidar_depth",
              side_effect=lambda *a, **k: depth.copy() if lidar else None),
        patch("summicron.render.convolve.render", side_effect=fake_render),
        patch("summicron.render.convolve.save_results",
              side_effect=lambda u8, stem, d, **k: d / "x.jpg"),
    ]
    if hdr is not None:
        ps.append(patch("summicron.io.hdr_load_macos.load_hdr_heic",
                        side_effect=fake_load_hdr))
    return cap, ps


def _png(p):
    Image.fromarray(np.full((64, 64, 3), 128, np.uint8)).save(p)
    return p


def _run(opts, ps):
    buf = io.StringIO()
    with ExitStack() as st, redirect_stdout(buf):
        for p in ps:
            st.enter_context(p)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = render_image(opts)
    return res, buf.getvalue()


def _opts(tmp_path, **kw):
    kw.setdefault("focus_xy", (0.5, 0.5))
    return RenderOptions(
        image_path=_png(tmp_path / "t.png"),
        tensor_path=_make_tensor_npz(tmp_path / "t.npz"),
        out_dir=tmp_path / "o", **kw)


def test_auto_png_resolves_none_with_one_note(tmp_path):
    cap, ps = _fakes()
    res, out = _run(_opts(tmp_path), ps)
    assert res.diagnostics.resolved["hdr_source"] == "none"
    assert cap["render"]["hdr_source"] == "none"
    assert res.mode == "ldr"
    assert out.count("note: hdr_source auto -> none") == 1
    assert "hdr_source=none" in out


def test_auto_gainmap_heic_takes_macos_path(tmp_path):
    hdr = np.full((64, 64, 3), 0.5, np.float32)
    hdr[0, 0] = 2.0
    cap, ps = _fakes(hdr=hdr)
    ps += [
        patch("summicron.render._hdr_detect.has_hdr_gainmap", return_value=True),
        patch("summicron.render._hdr_detect.macos_hdr_loader_available",
              return_value=True),
        patch("summicron.io.tone_lut.fit_input_tone",
              return_value=(np.array([0, 1.0]), np.array([0, 1.0]),
                            np.array([1.0]), np.array([1.0]))),
    ]
    res, out = _run(_opts(tmp_path), ps)
    assert cap["load_hdr"] == 1
    assert cap["render"]["hdr_source"] == "macos"
    assert res.mode == "hdr"
    assert res.diagnostics.resolved["hdr_source"] == "macos"
    assert "note: hdr_source" not in out


def test_auto_unavailable_loader_falls_back(tmp_path):
    cap, ps = _fakes()
    ps += [
        patch("summicron.render._hdr_detect.has_hdr_gainmap", return_value=True),
        patch("summicron.render._hdr_detect.macos_hdr_loader_available",
              return_value=False),
    ]
    res, out = _run(_opts(tmp_path), ps)
    assert res.diagnostics.resolved["hdr_source"] == "none"
    assert "loader unavailable" in out


def test_explicit_input_lut_with_auto_resolving_none_is_value_error(tmp_path):
    cap, ps = _fakes()
    with pytest.raises(ValueError, match="input-lut"):
        _run(_opts(tmp_path, tonemap="input-lut"), ps)


def test_explicit_shadow_match_with_auto_resolving_none_is_value_error(tmp_path):
    cap, ps = _fakes()
    with pytest.raises(ValueError, match="shadow-match"):
        _run(_opts(tmp_path, shadow_match_mode="on"), ps)


# ---- audit fixes -------------------------------------------------------------------

def test_bright_pass_auto_with_ci_tonemap_resolves_to_one(tmp_path):
    """Run 7 fix 1: 'auto' threshold + tonemap resolving to ci -> 1.0."""
    cap, ps = _fakes()
    res, _ = _run(_opts(tmp_path, bright_pass_thresh="auto"), ps)
    assert cap["render"]["bright_thresh"] == 1.0
    assert res.diagnostics.resolved["bright_pass_thresh"] == "1"


def test_nonfinite_hdr_buffer_never_reaches_fit_input_tone(tmp_path):
    hdr = np.full((64, 64, 3), 0.5, np.float32)
    hdr[3, 3, 0] = np.inf
    hdr[5, 5, 0] = 3.0  # finite headroom AND an Inf pixel
    cap, ps = _fakes(hdr=hdr)
    res, out = _run(_opts(tmp_path, hdr_source="macos"), ps)
    assert res.diagnostics.resolved["tonemap"] == "ci"
    assert "non-finite" in out


def test_load_image_undecodable_is_value_error(tmp_path):
    from summicron.render.convolve import load_image
    p = tmp_path / "junk.png"
    p.write_bytes(b"this is not an image")
    with pytest.raises(ValueError, match="cannot decode image"):
        load_image(p)


def test_monocular_error_text_is_collapsed_to_one_line():
    from summicron.render.convolve import _monocular_depth

    def boom(device="mps"):
        raise RuntimeError("a\nb\n   c")

    with pytest.raises(ValueError) as ei:
        _monocular_depth(np.zeros((4, 4, 3), np.float32), boom, None, fallback=True)
    assert "\n" not in str(ei.value)
    assert "a b c" in str(ei.value)


@pytest.mark.slow
@pytest.mark.skipif(sys.platform != "darwin", reason="macOS HDR loader")
def test_cli_gray_png_macos_bright_pass_auto_renders(tmp_path):
    """Run 7 fix 1, real pipeline: flat gray PNG, explicit macos loader,
    ``--bright-pass-thresh auto`` (tonemap resolves to ci) must render."""
    import os
    import subprocess

    img = tmp_path / "gray.png"
    Image.fromarray(np.full((96, 128, 3), 128, np.uint8)).save(img)
    r = subprocess.run(
        [sys.executable, "-m", "summicron.render.convolve", "--image", str(img),
         "--hdr-source", "macos", "--bright-pass-thresh", "auto",
         "--depth-source", "monocular", "--depth-guided", "off",
         "--depth-mapping", "linear", "--downres", "128",
         "--out-dir", str(tmp_path / "o")],
        capture_output=True, text=True, cwd=str(ROOT),
        env=dict(os.environ, PYTHONPATH=str(ROOT), HF_HUB_OFFLINE="1"),
        timeout=900,
    )
    if "monocular depth failed" in r.stderr:
        pytest.skip("DA2 unavailable offline")
    assert r.returncode == 0, r.stderr[-600:]
    assert "could not convert" not in r.stderr
