"""Run 6 default flip: adversarial edge cases around the ``auto`` resolution."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

from summicron.render.api import RenderOptions, render_image
from tests.unit.test_default_flip import _drive
from tests.unit.test_depth_s1_flags import _make_tensor_npz

darwin = pytest.mark.skipif(sys.platform != "darwin", reason="hdr macos")


def _opts(img, ten, tmp_path, **kw):
    return RenderOptions(image_path=img, tensor_path=ten, focus_xy=(0.5, 0.5),
                         out_dir=tmp_path / "o", save=False, **kw)


def test_corrupt_heic_surfaces_the_real_load_error_not_monocular(tmp_path):
    bad = tmp_path / "bad.heic"
    bad.write_bytes(b"not a heic at all")
    ten = _make_tensor_npz(tmp_path / "t.npz")
    with patch("summicron.depth.estimate.load_depth_model") as mono:
        with pytest.raises(Exception) as ei:
            render_image(_opts(bad, ten, tmp_path))
    mono.assert_not_called()
    assert "monocular" not in str(ei.value)


@pytest.mark.parametrize("mode,size", [("RGB", (1, 1)), ("RGBA", (48, 32))])
def test_odd_pngs_fall_back_to_monocular_and_render(tmp_path, mode, size):
    img = tmp_path / "odd.png"
    Image.new(mode, size, (120, 90, 60, 255)[: len(mode)]).save(img)
    ten = _make_tensor_npz(tmp_path / "t.npz")
    with patch("summicron.depth.estimate.load_depth_model", return_value=object()), \
         patch("summicron.depth.estimate.estimate_depth",
               side_effect=lambda im, b: np.full(im.shape[:2], 0.5, np.float32)):
        res = render_image(_opts(img, ten, tmp_path))
    assert res.diagnostics.resolved["depth_source"] == "monocular"
    assert res.output_u8.shape[2] == 3


def test_uppercase_heic_suffix_is_not_swallowed_as_non_heic(tmp_path):
    img = tmp_path / "UP.HEIC"
    Image.new("RGB", (16, 16)).save(img.with_suffix(".png"))
    img.write_bytes(img.with_suffix(".png").read_bytes())
    ten = _make_tensor_npz(tmp_path / "t.npz")
    # real extract_lidar_depth on a PNG body named .HEIC: the HEIF opener error
    # must propagate for HEIC-named inputs (real errors are never hidden).
    with patch("summicron.depth.estimate.load_depth_model") as mono:
        with pytest.raises(Exception):
            render_image(_opts(img, ten, tmp_path))
    mono.assert_not_called()


def test_lidar_with_hdr_none_is_guided_diopter_ci(tmp_path):
    cap = _drive(tmp_path)
    r = cap["res"].diagnostics.resolved
    assert (r["depth_guided"], r["depth_mapping"], r["tonemap"]) == ("on", "diopter", "ci")


@darwin
def test_shadow_match_with_default_tonemap_resolves_ci(tmp_path):
    cap = _drive(tmp_path, hdr_gain=3.0, shadow_match_mode="on",
                 depth_guided="off", depth_mapping="linear")
    assert cap["res"].diagnostics.resolved["tonemap"] == "ci"
    assert "shadow-match" in cap["out"]


def test_explicit_lidar_on_png_keeps_strict_error(tmp_path):
    img = tmp_path / "p.png"
    Image.new("RGB", (16, 16)).save(img)
    ten = _make_tensor_npz(tmp_path / "t.npz")
    with pytest.raises(Exception):
        render_image(_opts(img, ten, tmp_path, depth_source="lidar"))
