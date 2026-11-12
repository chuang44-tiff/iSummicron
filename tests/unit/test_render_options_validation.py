"""CLI-1b C4: RenderOptions.__post_init__ validation + shared _check_platform."""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import pytest

from summicron.render.api import RenderOptions, _check_platform
from summicron.render.convolve import _validate_args, build_parser

IMG = Path("dummy.heic")

# (field overrides, byte-identical historical parser.error message)
CASES = [
    ({"focus_xy": (1.5, 0.5)}, "--focus-xy values must be in [0, 1]"),
    ({"focus_xy": (0.5, -0.1)}, "--focus-xy values must be in [0, 1]"),
    ({"focus_xy": (float("nan"), 0.5)}, "--focus-xy values must be in [0, 1]"),
    ({"focus_target_m": 0.0}, "--focus-target-m must be > 0"),
    ({"focus_target_m": -1.0}, "--focus-target-m must be > 0"),
    ({"focus_target_m": float("nan")}, "--focus-target-m must be > 0"),
    ({"focus_target_m": float("inf")}, "--focus-target-m must be > 0"),
    ({"shadow_band_hi": 1.5}, "--shadow-band-hi must be in [0.0, 1.0]"),
    ({"shadow_match_max_gain": 0.5}, "--shadow-match-max-gain must be in [1.0, 16.0]"),
    ({"shadow_match_strength": 5.0}, "--shadow-match-strength must be in [0.0, 4.0]"),
    ({"slab_batch_size": 0}, "--slab-batch-size must be >= 1"),
    ({"occlusion_compositing": "maybe"}, "--occlusion-compositing must be 'off' or 'on'"),
    (
        {"shadow_match_mode": "on", "hdr_source": "none"},
        "--shadow-match-mode on requires --hdr-source != none "
        "(needs HDR linear buffer + CI preview pass).",
    ),
]


@pytest.mark.parametrize("kw,msg", CASES)
def test_invalid_raises_exact_message(kw, msg):
    with pytest.raises(ValueError) as ei:
        RenderOptions(image_path=IMG, **kw)
    assert str(ei.value) == msg


@pytest.mark.parametrize(
    "kw",
    [
        {"depth_source": "bogus"},
        {"psf_norm": "bogus"},
        {"hdr_source": "bogus"},
        {"shadow_match_mode": "bogus"},
    ],
)
def test_enum_fields_rejected(kw):
    with pytest.raises(ValueError):
        RenderOptions(image_path=IMG, **kw)


@pytest.mark.parametrize("kw,msg", CASES)
def test_replace_revalidates(kw, msg):
    base = RenderOptions(image_path=IMG)
    with pytest.raises(ValueError) as ei:
        dataclasses.replace(base, **kw)
    assert str(ei.value) == msg


def test_defaults_construct():
    RenderOptions(image_path=IMG)


def test_macos_constructs_off_darwin_but_render_image_rejects(monkeypatch):
    monkeypatch.setattr("sys.platform", "linux")
    opts = RenderOptions(image_path=IMG, hdr_source="macos")  # must not raise
    from summicron.render.api import render_image

    with pytest.raises(ValueError, match="requires macOS; got linux"):
        render_image(opts)


def test_check_platform_message(monkeypatch):
    monkeypatch.setattr("sys.platform", "linux")
    with pytest.raises(ValueError) as ei:
        _check_platform("macos")
    assert str(ei.value) == (
        "--hdr-source macos requires macOS; got linux. "
        "Use --hdr-source none on this platform."
    )
    _check_platform("none")  # no raise
    monkeypatch.setattr("sys.platform", "darwin")
    _check_platform("macos")  # no raise


@pytest.mark.parametrize(
    "argv,msg",
    [
        (["--focus-xy", "2", "0.5"], "--focus-xy values must be in [0, 1]"),
        (["--focus-target-m", "-1"], "--focus-target-m must be > 0"),
        (["--shadow-band-hi", "2"], "--shadow-band-hi must be in [0.0, 1.0]"),
    ],
)
def test_validate_args_exit_2(argv, msg, capsys):
    parser = build_parser()
    args = parser.parse_args(["--image", "x.heic", *argv])
    with pytest.raises(SystemExit) as ei:
        _validate_args(args, parser)
    assert ei.value.code == 2
    assert msg in capsys.readouterr().err


def test_validate_args_platform_exit_2(monkeypatch, capsys):
    monkeypatch.setattr(sys, "platform", "linux")
    parser = build_parser()
    args = parser.parse_args(["--image", "x.heic", "--hdr-source", "macos"])
    with pytest.raises(SystemExit) as ei:
        _validate_args(args, parser)
    assert ei.value.code == 2
    assert "requires macOS; got linux" in capsys.readouterr().err


# --- CLI-1b C3/C4 additions -------------------------------------------------


def test_canonical_bench_builds_expected_options():
    import dataclasses

    from summicron.render.api import RenderOptions

    o = RenderOptions.canonical_bench("a.heic", (0.4, 0.45), save=False)
    assert o.focus_xy == (0.4, 0.45)
    assert (o.depth_scale, o.downres, o.hdr_source) == (10.0, 2048, "macos")
    assert o.save is False and o.out_dir is None
    with pytest.raises(ValueError):
        dataclasses.replace(o, shadow_band_hi=2.0)


def test_default_tensor_load_emits_no_schema_warning():
    import warnings

    from summicron.render.convolve import load_tensor
    from summicron.render._artifacts import default_tensor_path

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        load_tensor(default_tensor_path())
