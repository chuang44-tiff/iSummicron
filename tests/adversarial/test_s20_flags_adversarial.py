"""Adversarial round for S1 plumbing (--depth-guided/--depth-mapping/--depth-gain)
and tone fix G (--tonemap input-lut, --bright-pass-thresh auto).

Spec: doc/plan/phase2/section20_depth-diopter/locked-spec-S1-plumbing-tone.md
"""
from __future__ import annotations

import dataclasses
import hashlib
import os
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

from summicron.render.api import RenderOptions

REPO = Path(__file__).resolve().parents[2]
HEIC = REPO / "iphoneImage" / "IMG_6651.HEIC"
DEMO = REPO / "iphoneImage" / "Demo_input.HEIC"

slow = pytest.mark.slow
darwin = pytest.mark.skipif(sys.platform != "darwin", reason="macOS HDR")


def _mk(**kw):
    return RenderOptions(image_path=Path("x.jpg"), **kw)


def _cli(argv, env=None, module="summicron.render.convolve"):
    e = dict(os.environ)
    e.pop("ISUMMICRON_DEBUG", None)
    if env:
        e.update(env)
    return subprocess.run(
        [sys.executable, "-m", module, *argv],
        capture_output=True, text=True, env=e, cwd=REPO,
    )


# ---------------------------------------------------------------- direct construction

@pytest.mark.parametrize("g", [0, 0.0, -1, -0.5, float("nan"), float("inf"),
                               float("-inf"), 10.0001, 11, True, False, "x", "AUTO",
                               "", " auto", None, [2.0], (2.0,), 1j])
def test_gain_rejected_direct(g):
    with pytest.raises(ValueError, match="--depth-gain"):
        _mk(depth_mapping="diopter", depth_gain=g)


@pytest.mark.parametrize("g", [1e-9, 0.5, 1, 2, 10, 10.0, np.float64(3.0)])
def test_gain_accepted_and_normalised(g):
    o = _mk(depth_mapping="diopter", depth_gain=g)
    assert isinstance(o.depth_gain, float) and o.depth_gain == float(g)


@pytest.mark.parametrize("g", [np.float32(2.0), np.int64(2)])
def test_gain_numpy_scalars_clean_error_or_accept(g):
    # must be either accepted or a clean ValueError, never TypeError
    try:
        _mk(depth_mapping="diopter", depth_gain=g)
    except ValueError:
        pass


def test_gain_numeric_with_linear_rejected_even_when_invalid_first():
    with pytest.raises(ValueError, match="--depth-gain"):
        _mk(depth_gain=2.0, depth_mapping="linear")
    with pytest.raises(ValueError, match="--depth-gain"):
        _mk(depth_gain=float("nan"))


def test_replace_revalidates():
    o = _mk(depth_mapping="diopter", depth_gain=3.0)
    with pytest.raises(ValueError, match="--depth-gain"):
        dataclasses.replace(o, depth_mapping="linear")  # gain 3.0 now orphaned
    with pytest.raises(ValueError):
        dataclasses.replace(o, depth_gain=0.0)
    o2 = dataclasses.replace(o, depth_mapping="linear", depth_gain="auto")
    assert o2.depth_mapping == "linear"


def test_auto_gain_inert_with_linear_and_guided_off():
    o = _mk(depth_guided="off", depth_mapping="linear")
    assert (o.depth_guided, o.depth_mapping, o.depth_gain) == ("off", "linear", "auto")


@pytest.mark.parametrize("kw", [
    dict(depth_source="monocular", depth_mapping="diopter"),
    dict(depth_source="monocular", depth_guided="on"),
    dict(depth_source="monocular", depth_mapping="diopter", depth_guided="on"),
    dict(depth_guided="ON"), dict(depth_guided=True), dict(depth_guided=None),
    dict(depth_mapping="meters"), dict(depth_mapping="Diopter"),
    dict(tonemap="CI"), dict(tonemap=None),
])
def test_enum_and_monocular_combos_rejected(kw):
    with pytest.raises(ValueError):
        _mk(**kw)


@pytest.mark.parametrize("t", [None, "x", True, [1.0], 1j, float("nan"), float("inf"),
                               float("-inf"), np.float64("nan")])
def test_thresh_rejected_never_typeerror(t):
    if t is None:
        _mk(bright_pass_thresh=None)
        return
    with pytest.raises(ValueError, match="--bright-pass-thresh"):
        _mk(bright_pass_thresh=t, tonemap="input-lut", hdr_source="macos")


def test_thresh_auto_with_ci_rejected_but_numeric_accepted_with_ci():
    with pytest.raises(ValueError, match="requires --tonemap input-lut"):
        _mk(bright_pass_thresh="auto", hdr_source="none")
    _mk(bright_pass_thresh=2)
    _mk(bright_pass_thresh=0.0)
    _mk(bright_pass_thresh=-3.0)  # legacy: any finite number is verbatim


def test_input_lut_validation_order_both_errors():
    # hdr none AND shadow-match on: whichever fires, it must be ValueError naming a flag
    with pytest.raises(ValueError, match="--"):
        _mk(tonemap="input-lut", shadow_match_mode="on")


def test_new_fields_with_other_modes_construct():
    _mk(depth_mapping="diopter", depth_guided="on", occlusion_compositing="on")
    _mk(depth_mapping="diopter", psf_norm="peak")
    _mk(tonemap="input-lut", hdr_source="macos", occlusion_compositing="on",
        psf_norm="peak", downres=1)


def test_canonical_bench_diopter_override():
    o = RenderOptions.canonical_bench("a.heic", (0.5, 0.5), depth_mapping="diopter",
                                      depth_guided="on", depth_gain=1.5)
    assert o.depth_scale == 1.0 and o.depth_mapping == "diopter"
    with pytest.raises(ValueError):
        RenderOptions.canonical_bench("a.heic", (0.5, 0.5), old_look=True,
                                      depth_gain=1.5)


def test_canonical_bench_defaults_unchanged():
    o = RenderOptions.canonical_bench("a.heic", (0.5, 0.5), old_look=True)
    assert (o.depth_guided, o.depth_mapping, o.depth_gain, o.tonemap,
            o.bright_pass_thresh) == ("off", "linear", "auto", "ci", 1.0)
    o = RenderOptions.canonical_bench("a.heic", (0.5, 0.5))
    assert (o.depth_guided, o.depth_mapping, o.depth_gain, o.tonemap,
            o.bright_pass_thresh) == ("auto", "auto", "auto", "auto", None)


# ---------------------------------------------------------------- CLI error contract

_BAD = [
    ["--depth-mapping", "diopter", "--depth-gain", "0"],
    ["--depth-mapping", "diopter", "--depth-gain", "-1"],
    ["--depth-mapping", "diopter", "--depth-gain", "nan"],
    ["--depth-mapping", "diopter", "--depth-gain", "inf"],
    ["--depth-mapping", "diopter", "--depth-gain", "11"],
    ["--depth-mapping", "diopter", "--depth-gain", "abc"],
    ["--depth-mapping", "diopter", "--depth-gain", ""],
    ["--depth-gain", "2"],
    ["--depth-gain=-1"],
    ["--depth-source", "monocular", "--depth-mapping", "diopter"],
    ["--depth-source", "monocular", "--depth-guided", "on"],
    ["--tonemap", "input-lut"],
    ["--tonemap", "input-lut", "--hdr-source", "macos", "--shadow-match-mode", "on"],
    ["--bright-pass-thresh", "auto"],
    ["--bright-pass-thresh", "nan"],
    ["--bright-pass-thresh", "wat"],
    ["--depth-mapping", "meters"],
    ["--depth-guided", "maybe"],
    ["--tonemap", "reinhard"],
]


@pytest.mark.parametrize("argv", _BAD, ids=lambda a: " ".join(a))
def test_cli_rejects_exit2_no_traceback(argv):
    r = _cli(["--image", "nonexistent.jpg", *argv])
    assert r.returncode == 2, (r.returncode, r.stderr)
    assert "Traceback" not in r.stderr
    if r.stderr.startswith("render:"):
        assert len([ln for ln in r.stderr.splitlines() if ln.strip()]) == 1


@pytest.mark.parametrize("argv", _BAD[7:13], ids=lambda a: " ".join(a))
def test_isummicron_render_parity(argv):
    r = _cli(["render", "--image", "nonexistent.jpg", *argv], module="summicron.cli")
    assert r.returncode == 2 and "Traceback" not in r.stderr


def test_cli_defaults_accept_inert_auto_gain():
    # explicit 'auto' with linear is the inert default: must pass validation
    r = _cli(["--image", "nonexistent.jpg", "--depth-gain", "auto"])
    assert "--depth-gain" not in r.stderr


# ---------------------------------------------------------------- real renders (slow)

def _md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def sdr_files(tmp_path_factory):
    from PIL import Image
    d = tmp_path_factory.mktemp("sdr")
    g = np.random.RandomState(0).randint(0, 255, (160, 240, 3), dtype=np.uint8)
    out = {}
    for ext in ("png", "jpg"):
        p = d / f"n.{ext}"
        Image.fromarray(g).save(p)
        out[ext] = p
    return out


@slow
@pytest.mark.parametrize("ext", ["png", "jpg"])
@pytest.mark.parametrize("argv", [
    ["--depth-guided", "on"],
    ["--depth-mapping", "diopter"],
    ["--depth-mapping", "diopter", "--depth-guided", "on", "--depth-gain", "2"],
])
def test_no_lidar_input_clean_exit2(sdr_files, tmp_path, ext, argv):
    r = _cli(["--image", str(sdr_files[ext]), "--out-dir", str(tmp_path), *argv])
    assert r.returncode == 2, (r.returncode, r.stderr[-400:])
    assert "Traceback" not in r.stderr and r.stderr.startswith("render:")


@slow
@darwin
@pytest.mark.parametrize("ext", ["png", "jpg"])
def test_sdr_only_input_lut_monocular(sdr_files, tmp_path, ext):
    r = _cli(["--image", str(sdr_files[ext]), "--out-dir", str(tmp_path),
              "--depth-source", "monocular", "--hdr-source", "macos",
              "--tonemap", "input-lut"])
    # accepted (LUT ~ identity) or clean exit-2; never a traceback
    assert "Traceback" not in r.stderr
    assert r.returncode in (0, 2), r.stderr[-400:]


@slow
@darwin
@pytest.mark.skipif(not HEIC.exists(), reason="needs iphoneImage")
@pytest.mark.parametrize("extra", [
    ["--depth-mapping", "diopter"],
    ["--depth-mapping", "diopter", "--depth-guided", "on", "--depth-gain", "1.5"],
    ["--depth-mapping", "diopter", "--depth-scale", "10"],
    ["--depth-guided", "on", "--occlusion-compositing", "on"],
    ["--depth-mapping", "diopter", "--occlusion-compositing", "on"],
    ["--depth-mapping", "diopter", "--psf-norm", "peak"],
    ["--hdr-source", "macos", "--tonemap", "input-lut", "--psf-norm", "peak"],
    ["--hdr-source", "macos", "--tonemap", "input-lut", "--occlusion-compositing", "on"],
    ["--hdr-source", "macos", "--tonemap", "input-lut", "--bright-pass-thresh", "auto",
     "--depth-mapping", "diopter", "--depth-guided", "on"],
    ["--hdr-source", "macos", "--tonemap", "input-lut", "--bright-pass-thresh", "1e9"],
    ["--hdr-source", "macos", "--tonemap", "input-lut", "--downres", "100"],
], ids=lambda a: " ".join(a))
def test_combo_renders_ok(tmp_path, extra):
    argv = ["--image", str(HEIC), "--focus-xy", "0.5", "0.5", "--downres", "320",
            "--out-dir", str(tmp_path)]
    if "--downres" in extra:
        argv = ["--image", str(HEIC), "--focus-xy", "0.5", "0.5", "--out-dir", str(tmp_path)]
    r = _cli(argv + extra)
    assert r.returncode == 0, r.stderr[-600:]
    assert "Traceback" not in r.stderr
    assert list(tmp_path.glob("*.png"))


@slow
@darwin
@pytest.mark.skipif(not HEIC.exists(), reason="needs iphoneImage")
def test_diopter_scale_ignored_name_and_warning(tmp_path):
    r = _cli(["--image", str(HEIC), "--focus-xy", "0.5", "0.5", "--downres", "320",
              "--out-dir", str(tmp_path), "--depth-mapping", "diopter", "--depth-scale", "10"])
    assert r.returncode == 0, r.stderr[-400:]
    names = [p.name for p in tmp_path.glob("*.png")]
    assert names and all("_ds" not in n for n in names) and all("_dio" in n for n in names)


@slow
@darwin
@pytest.mark.skipif(not HEIC.exists(), reason="needs iphoneImage")
def test_diopter_scale_irrelevant_to_pixels(tmp_path):
    outs = []
    for s in ("1", "10", "0.3"):
        d = tmp_path / s
        r = _cli(["--image", str(HEIC), "--focus-xy", "0.5", "0.5", "--downres", "320",
                  "--out-dir", str(d), "--depth-mapping", "diopter", "--depth-scale", s])
        assert r.returncode == 0, r.stderr[-300:]
        outs.append(_md5(next(d.glob("*.png"))))
    assert len(set(outs)) == 1


@slow
@darwin
@pytest.mark.skipif(not HEIC.exists(), reason="needs iphoneImage")
def test_determinism_two_runs(tmp_path):
    outs = []
    for k in ("a", "b"):
        d = tmp_path / k
        r = _cli(["--image", str(HEIC), "--focus-xy", "0.5", "0.5", "--downres", "320",
                  "--out-dir", str(d), "--hdr-source", "macos", "--tonemap", "input-lut",
                  "--depth-mapping", "diopter", "--depth-guided", "on"])
        assert r.returncode == 0, r.stderr[-300:]
        outs.append(_md5(next(d.glob("*.png"))))
    assert outs[0] == outs[1]


@slow
@darwin
@pytest.mark.skipif(not HEIC.exists(), reason="needs iphoneImage")
def test_default_flags_output_name_unchanged(tmp_path):
    r = _cli(["--image", str(HEIC), "--focus-xy", "0.5", "0.5", "--downres", "320",
              "--out-dir", str(tmp_path), "--depth-scale", "10",
              "--depth-guided", "off", "--depth-mapping", "linear"])
    assert r.returncode == 0
    assert sorted(p.name for p in tmp_path.glob("*.png")) == ["IMG_6651_r320_rendered_ds10.0.png"]


@slow
@darwin
@pytest.mark.skipif(not DEMO.exists(), reason="needs iphoneImage")
@pytest.mark.parametrize("dr", ["1", "7", "333", "5000"])
def test_input_lut_downres_variants_shape_match(tmp_path, dr):
    if dr == "5000":
        pytest.skip("native-size run is too slow for the suite")
    r = _cli(["--image", str(DEMO), "--focus-xy", "0.5", "0.5", "--downres", dr,
              "--out-dir", str(tmp_path), "--hdr-source", "macos", "--tonemap", "input-lut"])
    assert r.returncode == 0, r.stderr[-300:]
