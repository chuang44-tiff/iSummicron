"""P1.audit Item 3 — RenderOptions / RenderResult / render_image() contract.

Covers four sub-cases:
  (a) LDR render at downres=256: mode='ldr', uint8 output, no linear.
  (b) HDR macOS render (skipped on non-darwin): mode='hdr', linear populated.
  (c) RenderOptions.from_argparse round-trip from build_parser().
  (d) Defaults round-trip on RenderOptions.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pytest

from summicron.render._artifacts import default_tensor_path

from summicron.render.api import (
    CancelledError,
    RenderDiagnostics,
    RenderOptions,
    RenderResult,
    render_image,
)
from summicron.render.convolve import build_parser

REPO = Path(__file__).resolve().parents[2]
IMG = REPO / "iphoneImage" / "IMG_6915.HEIC"
TENSOR = default_tensor_path()


def _have_canonical_inputs() -> bool:
    return IMG.exists() and TENSOR.exists()


@pytest.mark.skipif(
    not _have_canonical_inputs(), reason="canonical IMG_6915 / tensor missing"
)
def test_render_image_ldr_downres256_returns_populated_result(tmp_path):
    """(a) LDR path: uint8 output ALWAYS populated; output_linear is None
    by default; mode == 'ldr'; elapsed_s > 0; warnings tuple."""
    opts = RenderOptions(
        image_path=IMG,
        tensor_path=TENSOR,
        out_path=tmp_path / "out.jpg",
        depth_source="lidar",
        focus_xy=(0.40, 0.45),
        depth_scale=10.0,
        downres=256,
        hdr_source="none",
    )
    result = render_image(opts)
    assert isinstance(result, RenderResult)
    assert result.mode == "ldr"
    assert result.output_u8.dtype == np.uint8
    assert result.output_u8.ndim == 3
    assert result.output_u8.shape[2] == 3
    # downres limits the long edge; portrait orientation post-R0 means
    # H >= W with long edge = 256.
    assert max(result.output_u8.shape[:2]) == 256
    assert result.output_linear is None
    assert result.elapsed_s > 0
    assert isinstance(result.diagnostics, RenderDiagnostics)
    assert isinstance(result.diagnostics.warnings, tuple)


@pytest.mark.skipif(
    sys.platform != "darwin", reason="HDR macOS path requires Quartz / CoreImage"
)
@pytest.mark.skipif(
    not _have_canonical_inputs(), reason="canonical IMG_6915 / tensor missing"
)
def test_render_image_hdr_populates_linear_buffer(tmp_path):
    """(b) HDR path: output_u8 STILL populated; output_linear populated."""
    opts = RenderOptions(
        image_path=IMG,
        tensor_path=TENSOR,
        out_path=tmp_path / "out.jpg",
        depth_source="lidar",
        focus_xy=(0.40, 0.45),
        depth_scale=10.0,
        downres=256,
        hdr_source="macos",
    )
    result = render_image(opts)
    assert result.mode == "hdr"
    assert result.output_u8.dtype == np.uint8
    assert result.output_linear is not None
    assert result.output_linear.dtype == np.float32


def test_render_options_from_argparse_round_trip():
    """(c) build_parser() output → RenderOptions.from_argparse() round-trip."""
    parser = build_parser()
    args = parser.parse_args([
        "--image", "foo.heic",
        "--tensor", "bar.npz",
        "--depth-source", "lidar",
        "--focus-xy", "0.4", "0.45",
        "--depth-scale", "10",
        "--downres", "2048",
        "--hdr-source", "macos",
        "--shadow-match-mode", "on",
        "--shadow-band-hi", "0.6",
        "--shadow-match-strength", "1.2",
        "--shadow-match-max-gain", "3.0",
    ])
    opts = RenderOptions.from_argparse(args)
    assert opts.image_path == Path("foo.heic")
    assert opts.tensor_path == Path("bar.npz")
    assert opts.depth_source == "lidar"
    assert opts.focus_xy == (0.4, 0.45)
    assert opts.depth_scale == 10.0
    assert opts.downres == 2048
    assert opts.hdr_source == "macos"
    assert opts.shadow_match_mode == "on"
    assert opts.shadow_band_hi == pytest.approx(0.6)
    assert opts.shadow_match_strength == pytest.approx(1.2)
    assert opts.shadow_match_max_gain == pytest.approx(3.0)


def test_render_options_defaults_are_locked_per_spec():
    """(d) Default values match the locked-spec § Item 3 dataclass shape."""
    opts = RenderOptions(image_path=Path("dummy.heic"))
    assert opts.tensor_path is None
    assert opts.depth_source == "auto"  # Run 6 default flip
    assert opts.focus_xy is None
    assert opts.depth_scale == 1.0
    assert opts.downres is None
    assert opts.hdr_source == "auto"  # Rev 3
    assert opts.bright_pass_region_size == 31
    assert opts.bright_pass_floor == pytest.approx(0.10)
    assert opts.bright_pass_norm_pct == pytest.approx(99.0)
    assert opts.shadow_match_mode == "off"
    assert opts.shadow_band_hi == pytest.approx(0.70)
    assert opts.shadow_match_max_gain == pytest.approx(4.0)
    assert opts.shadow_match_strength == pytest.approx(1.0)
    assert opts.return_linear is False


def test_render_options_is_frozen():
    """RenderOptions / RenderResult are frozen dataclasses (immutable)."""
    opts = RenderOptions(image_path=Path("dummy.heic"))
    with pytest.raises((AttributeError, TypeError)):
        opts.lens = "50"  # type: ignore[misc]


def test_cancelled_error_is_an_exception_subclass():
    """CancelledError is a public Exception subclass."""
    assert issubclass(CancelledError, Exception)
    e = CancelledError("test message")
    assert "test message" in str(e)
