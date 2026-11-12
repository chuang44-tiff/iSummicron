"""Unit tests for the isummicron CLI pipeline switch and render CLI flags."""

from __future__ import annotations

import argparse

import numpy as np
import pytest

from summicron.render.convolve import downres_array, save_results


# ---------- Render CLI: --downres flag ----------


def _render_parser() -> argparse.ArgumentParser:
    """Build a minimal parser matching the render CLI's --downres argument."""
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--tensor", required=True)
    p.add_argument("--downres", type=int, default=None)
    return p


def test_render_downres_parsed_as_int():
    ns = _render_parser().parse_args(["--image", "a.jpg", "--tensor", "b.npz", "--downres", "2048"])
    assert ns.downres == 2048


def test_render_downres_default_is_none():
    ns = _render_parser().parse_args(["--image", "a.jpg", "--tensor", "b.npz"])
    assert ns.downres is None


def test_save_results_downres_suffix(tmp_path):
    """save_results with downres=2048 produces filenames containing _r2048."""
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    save_results(img, "IMG_6651", tmp_path, depth_scale=1.0, downres=2048)
    assert (tmp_path / "IMG_6651_r2048_rendered.jpg").exists()
    assert (tmp_path / "IMG_6651_r2048_rendered.png").exists()


def test_save_results_no_downres_suffix(tmp_path):
    """save_results without downres has no _r tag in the filename."""
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    save_results(img, "IMG_6651", tmp_path, depth_scale=1.0, downres=None)
    assert (tmp_path / "IMG_6651_rendered.jpg").exists()
    assert (tmp_path / "IMG_6651_rendered.png").exists()


# ---------- Render CLI: downres_array resize logic ----------


def test_downres_array_shrinks_landscape():
    """100x200 image with target=100 -> long edge becomes 100."""
    img = np.random.rand(100, 200, 3).astype(np.float32)
    out = downres_array(img, 100)
    assert out.shape == (50, 100, 3)


def test_downres_array_shrinks_portrait():
    """200x100 image with target=100 -> long edge becomes 100."""
    img = np.random.rand(200, 100, 3).astype(np.float32)
    out = downres_array(img, 100)
    assert out.shape == (100, 50, 3)


def test_downres_array_skip_already_small():
    """Image already <= target is returned unchanged (no upscale)."""
    img = np.random.rand(50, 80, 3).astype(np.float32)
    out = downres_array(img, 100)
    assert out is img  # exact same object, not a copy


def test_downres_array_skip_exact_match():
    """Image whose long edge exactly matches target is returned unchanged."""
    img = np.random.rand(60, 100, 3).astype(np.float32)
    out = downres_array(img, 100)
    assert out is img


# ---------- Render CLI: --psf-norm flag (P1.7 stamp-mode) ----------


def _render_parser_psf_norm() -> argparse.ArgumentParser:
    """Shim mirroring the render CLI's --psf-norm definition."""
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--tensor", required=True)
    p.add_argument(
        "--psf-norm", choices=["l1", "peak"], default="l1",
    )
    return p


@pytest.mark.parametrize("value", ["l1", "peak"])
def test_render_psf_norm_accepts_valid_values(value):
    ns = _render_parser_psf_norm().parse_args(
        ["--image", "a.jpg", "--tensor", "b.npz", "--psf-norm", value]
    )
    assert ns.psf_norm == value


def test_render_psf_norm_default_is_l1():
    ns = _render_parser_psf_norm().parse_args(
        ["--image", "a.jpg", "--tensor", "b.npz"]
    )
    assert ns.psf_norm == "l1"


def test_render_psf_norm_rejects_invalid_value():
    with pytest.raises(SystemExit):
        _render_parser_psf_norm().parse_args(
            ["--image", "a.jpg", "--tensor", "b.npz", "--psf-norm", "invalid"]
        )
