"""S22 S1: --kernel-family / --lens-aperture-mm / --bright-stamp[-gain]."""
from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from summicron.render import _render_args
from summicron.render._resolve import resolve_family
from summicron.render.api import RenderOptions


def _parse(*argv):
    p = argparse.ArgumentParser()
    _render_args.add_render_arguments(p)
    return p.parse_args(["--image", "x.heic", *argv])


def _opts(**kw):
    return RenderOptions(image_path=Path("x.heic"), **kw)


class TestParser:
    def test_defaults(self):
        ns = _parse()
        assert ns.kernel_family == "codev"
        assert ns.lens_aperture_mm is None
        assert ns.bright_stamp == "auto"
        assert ns.bright_stamp_gain is None

    @pytest.mark.parametrize(
        "val,exp",
        [("35/2", 17.5), ("50/2", 25.0), ("apple", 13.5), ("20", 20.0), ("7.25", 7.25)],
    )
    def test_aperture_presets_and_floats(self, val, exp):
        assert _parse("--lens-aperture-mm", val).lens_aperture_mm == exp

    @pytest.mark.parametrize("val", ["abc", "35/3", ""])
    def test_aperture_rejects_garbage(self, val):
        with pytest.raises(SystemExit):
            _parse("--lens-aperture-mm", val)

    def test_nan_inf_parse_then_rejected_by_options(self):
        for v in ("nan", "inf", "-1"):
            ns = _parse("--kernel-family", "synthetic", "--lens-aperture-mm", v)
            with pytest.raises(ValueError, match="--lens-aperture-mm"):
                RenderOptions.from_argparse(ns)

    def test_parser_stays_stdlib_only(self):
        assert "numpy" not in _render_args.__dict__

    def test_from_argparse_roundtrip(self):
        ns = _parse("--kernel-family", "synthetic", "--lens-aperture-mm", "50/2",
                    "--bright-stamp-gain", "4")
        o = RenderOptions.from_argparse(ns)
        assert (o.kernel_family, o.lens_aperture_mm, o.bright_stamp_gain) == (
            "synthetic", 25.0, 4.0)


class TestPostInit:
    def test_defaults_ok(self):
        o = _opts()
        assert (o.kernel_family, o.lens_aperture_mm, o.bright_stamp,
                o.bright_stamp_gain) == ("codev", None, "auto", None)

    def test_synthetic_defaults_ok(self):
        _opts(kernel_family="synthetic")

    def test_aperture_with_codev(self):
        with pytest.raises(ValueError, match="requires --kernel-family synthetic"):
            _opts(lens_aperture_mm=20.0)

    @pytest.mark.parametrize("a,ok", [(2.99, False), (3.0, True), (30.0, True),
                                      (30.01, False), (True, False), ("35/2", False),
                                      (float("nan"), False), (float("inf"), False),
                                      (20, True)])
    def test_aperture_range(self, a, ok):
        if ok:
            o = _opts(kernel_family="synthetic", lens_aperture_mm=a)
            assert isinstance(o.lens_aperture_mm, float)
        else:
            with pytest.raises(ValueError, match="--lens-aperture-mm must be a number"):
                _opts(kernel_family="synthetic", lens_aperture_mm=a)

    def test_tensor_with_synthetic(self):
        with pytest.raises(ValueError, match="--tensor is not allowed"):
            _opts(kernel_family="synthetic", tensor_path=Path("t.npz"))

    def test_peak_norm_with_synthetic(self):
        with pytest.raises(ValueError, match="requires --psf-norm l1"):
            _opts(kernel_family="synthetic", psf_norm="peak")

    def test_linear_mapping_with_synthetic(self):
        with pytest.raises(ValueError, match="requires --depth-mapping diopter"):
            _opts(kernel_family="synthetic", depth_mapping="linear")

    def test_monocular_with_synthetic(self):
        with pytest.raises(ValueError, match="requires --depth-mapping diopter"):
            _opts(kernel_family="synthetic", depth_source="monocular")

    def test_l1_stamp_needs_l1_norm(self):
        with pytest.raises(ValueError, match="--bright-stamp l1 requires --psf-norm l1"):
            _opts(bright_stamp="l1", psf_norm="peak")

    def test_gain_with_peak_stamp(self):
        with pytest.raises(ValueError, match="--bright-stamp-gain requires --bright-stamp l1"):
            _opts(bright_stamp_gain=2.0)
        with pytest.raises(ValueError, match="--bright-stamp-gain requires"):
            _opts(kernel_family="synthetic", bright_stamp="peak", bright_stamp_gain=2.0)

    @pytest.mark.parametrize("g,ok", [(0.0, False), (1e-9, True), (16.0, True),
                                      (16.01, False), (-1.0, False), (True, False),
                                      (float("nan"), False), ("2", False)])
    def test_gain_range(self, g, ok):
        if ok:
            _opts(bright_stamp="l1", bright_stamp_gain=g)
        else:
            with pytest.raises(ValueError, match="--bright-stamp-gain must be a number"):
                _opts(bright_stamp="l1", bright_stamp_gain=g)

    def test_explicit_peak_with_synthetic_allowed(self):
        _opts(kernel_family="synthetic", bright_stamp="peak")

    def test_explicit_l1_with_codev_allowed(self):
        _opts(bright_stamp="l1")

    def test_enum_values(self):
        with pytest.raises(ValueError, match="--kernel-family"):
            _opts(kernel_family="x")
        with pytest.raises(ValueError, match="--bright-stamp"):
            _opts(bright_stamp="x")


class TestResolveFamily:
    def test_codev_default(self):
        r = resolve_family(_opts())
        assert (r.kernel_family, r.lens_aperture_mm, r.bright_stamp,
                r.bright_stamp_gain) == ("codev", None, "peak", None)

    def test_synthetic_default(self):
        r = resolve_family(_opts(kernel_family="synthetic"))
        assert (r.lens_aperture_mm, r.bright_stamp, r.bright_stamp_gain) == (17.5, "l1", 2.0)

    def test_synthetic_explicit(self):
        r = resolve_family(_opts(kernel_family="synthetic", lens_aperture_mm=25.0,
                                 bright_stamp_gain=4.0))
        assert (r.lens_aperture_mm, r.bright_stamp_gain) == (25.0, 4.0)

    def test_synthetic_peak_has_no_gain(self):
        r = resolve_family(_opts(kernel_family="synthetic", bright_stamp="peak"))
        assert (r.bright_stamp, r.bright_stamp_gain) == ("peak", None)

    def test_codev_explicit_l1(self):
        r = resolve_family(_opts(bright_stamp="l1"))
        assert (r.bright_stamp, r.bright_stamp_gain) == ("l1", 2.0)

    def test_small_gain_not_truthiness(self):
        r = resolve_family(_opts(bright_stamp="l1", bright_stamp_gain=1e-9))
        assert r.bright_stamp_gain == 1e-9


def test_resolve_auto_synthetic_requires_lidar():
    from summicron.render._resolve import resolve_auto

    o = _opts(kernel_family="synthetic")
    with pytest.raises(ValueError, match="requires LiDAR depth"):
        resolve_auto(o, has_lidar=False, hdr_headroom=False)
    resolve_auto(o, has_lidar=True, hdr_headroom=False)


def test_canonical_bench_new_fields_at_defaults():
    o = RenderOptions.canonical_bench("x.heic", (0.4, 0.4))
    assert (o.kernel_family, o.lens_aperture_mm, o.bright_stamp,
            o.bright_stamp_gain) == ("codev", None, "auto", None)


def test_isummicron_help_lists_flags():
    from summicron.cli import build_parser

    p = build_parser()
    sub = [a for a in p._actions if isinstance(a, argparse._SubParsersAction)][0]
    h = sub.choices["render"].format_help()
    for f in ("--kernel-family", "--lens-aperture-mm", "--bright-stamp", "--bright-stamp-gain"):
        assert f in h
