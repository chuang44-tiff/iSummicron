"""0.5.0 defaults: ray-traced tensors, physical gain, l1 stamp, fixed knee.

Fast fakes around the pipeline (no real HEIC render) plus a tiny real
``render`` call that exercises the downres (``pix_scale < 0.95``) delta path.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from summicron.psf_model import ray_kernel as rk
from summicron.render import _artifacts, convolve
from summicron.render._resolve import (
    DEFAULT_BRIGHT_STAMP_GAIN,
    resolve_stamp,
)
from summicron.render.api import RenderOptions, render_image


def _opts(**kw):
    return RenderOptions(image_path=Path("x.heic"), **kw)


# ---- stamp resolution -------------------------------------------------------


def test_stamp_defaults_l1_gain_2():
    s = resolve_stamp(_opts())
    assert s.bright_stamp == "l1"
    assert s.bright_stamp_gain == DEFAULT_BRIGHT_STAMP_GAIN == 2.0


def test_stamp_peak_explicit():
    s = resolve_stamp(_opts(bright_stamp="peak"))
    assert s.bright_stamp == "peak" and s.bright_stamp_gain is None


def test_stamp_explicit_gain_and_errors():
    assert resolve_stamp(_opts(bright_stamp_gain=3.5)).bright_stamp_gain == 3.5
    with pytest.raises(ValueError, match="requires --bright-stamp l1"):
        _opts(bright_stamp="peak", bright_stamp_gain=2.0)
    with pytest.raises(ValueError, match="bright-stamp-gain"):
        _opts(bright_stamp_gain=0.0)


# ---- knee -------------------------------------------------------------------


def test_knee_constant_only():
    assert not hasattr(convolve, "_physical_knee_ratio")
    assert not hasattr(convolve, "KNEE_DIOPTER")
    assert not hasattr(convolve, "_SYNTH_FAMILY_NAME")


def test_render_knee_is_layered_constant_and_downres_delta():
    """Real tiny render with the shipped 35 mm tensor at pix_scale << 0.95."""
    from tests.fixtures.defocus import render_with_defocus

    t = convolve.load_tensor(_artifacts.default_tensor_path())
    rng = np.random.default_rng(3)
    img = rng.random((48, 64, 3)).astype(np.float32)
    stats: dict = {}
    out, lin = render_with_defocus(
        img, t, hdr_source="none", bright_thresh=1e9, stats=stats
    )
    assert stats["knee_ratio"] == convolve.D_KNEE_LAYERED
    assert np.isfinite(lin).all() and lin.shape == img.shape


def test_make_delta_float32_exact():
    d = convolve._make_delta(7)
    assert d.dtype == np.float32 and d.shape == (7, 7)
    assert d[3, 3] == 1.0 and d.sum() == 1.0
    ref = np.zeros((7, 7), np.float32)
    ref[3, 3] = 1.0
    assert np.array_equal(d, ref)
    assert rk.make_delta(7).dtype == np.float64  # NOT interchangeable


def test_make_delta_branch_exercised_at_downres():
    """A strict d0 delta at pix_scale < 0.95 maps to an exact centred float32 delta."""
    k = convolve._make_delta(183)
    base = np.zeros((183, 183), np.float32)
    base[91, 91] = 1.0
    assert np.array_equal(k, base)


# ---- artifacts --------------------------------------------------------------


@pytest.mark.parametrize("lens", ["35", "50"])
def test_artifacts_load_and_schema(lens):
    t = convolve.load_tensor(_artifacts.default_tensor_path(lens))
    k = t["kernels"]
    assert k.shape == (12, 11, 183, 183) and k.dtype == np.float32
    assert t["R"] == 12 and t["D"] == 11
    assert t["delta_guard_applied"] is True
    l1 = k.sum(axis=(2, 3), dtype=np.float64)
    assert np.abs(l1 - 1.0).max() < 1e-6
    delta = np.zeros((183, 183), np.float32)
    delta[91, 91] = 1.0
    assert all(np.array_equal(k[r, 0], delta) for r in range(12))
    assert np.array_equal(t["radial_edges"], rk.RADIAL_EDGES)
    assert np.array_equal(t["defocus_ratios"], rk.DEFOCUS_RATIOS)
    assert np.array_equal(t["object_distances_m"], rk.OBJECT_DISTANCES_M)
    with np.load(_artifacts.default_tensor_path(lens), allow_pickle=False) as z:
        assert str(z["schema_version"]) == "1.0"
        assert float(z["focus_distance_m"]) == 0.7


# ---- pipeline wiring (fakes) -----------------------------------------------


def _image(path: Path) -> Path:
    from PIL import Image

    arr = np.random.default_rng(0).integers(0, 256, (64, 64, 3), dtype=np.uint8)
    Image.fromarray(arr, "RGB").save(str(path), "JPEG", quality=80)
    return path


def _run(tmp_path, **kw):
    img = _image(tmp_path / "t.jpg")
    depth = np.linspace(0.7, 4.0, 64 * 64, dtype=np.float32).reshape(64, 64)
    cap: dict = {"loads": [], "msgs": []}
    real_load = convolve.load_tensor

    def spy_load(p):
        cap["loads"].append(Path(p).name)
        return real_load(p)

    def fake_render(image, tensor, **k):
        cap["render"] = k
        h, w = image.shape[:2]
        return np.zeros((h, w, 3), np.uint8), np.zeros((h, w, 3), np.float32)

    def fake_save(u8, stem, out_dir, **k):
        cap["stem"] = stem
        cap["suffix"] = k.get("name_suffix", "")
        return out_dir / "x.jpg"

    opts = RenderOptions(image_path=img, focus_xy=(0.5, 0.5), out_dir=tmp_path / "o",
                         depth_source="lidar", hdr_source="none", **kw)
    with patch("summicron.depth.estimate.extract_lidar_depth",
               side_effect=lambda *a, **k: depth.copy()), \
         patch("summicron.render.convolve.load_tensor", side_effect=spy_load), \
         patch("summicron.render.convolve.render", side_effect=fake_render), \
         patch("summicron.render.convolve.save_results", side_effect=fake_save):
        res = render_image(opts, on_progress=cap["msgs"].append)
    cap["res"] = res
    return cap


def test_default_uses_35_artifact_physical_gain_l1_stamp(tmp_path):
    cap = _run(tmp_path)
    assert cap["loads"] == [_artifacts.LENS_ARTIFACTS["35"]]
    assert cap["render"]["bright_stamp_gain"] == 2.0
    joined = "\n".join(cap["msgs"])
    assert "gain=1.000" in joined and "(physical)" in joined
    assert cap["res"].diagnostics.resolved["lens"] == "35"
    assert cap["res"].diagnostics.resolved["bright_stamp"] == "l1"


def test_lens_50_selects_50_artifact(tmp_path):
    cap = _run(tmp_path, lens="50")
    assert cap["loads"] == [_artifacts.LENS_ARTIFACTS["50"]]
    assert cap["res"].diagnostics.resolved["lens"] == "50"
    assert "_lens50" in cap["suffix"]


def test_explicit_gain_wins(tmp_path):
    cap = _run(tmp_path, depth_gain=1.5)
    assert "gain=1.500" in "\n".join(cap["msgs"])


def test_explicit_peak_stamp(tmp_path):
    cap = _run(tmp_path, bright_stamp="peak")
    assert cap["render"]["bright_stamp_gain"] is None
    assert "_peak" in cap["suffix"]


def test_custom_tensor_reports_custom(tmp_path):
    import shutil

    p = tmp_path / "mine.npz"
    shutil.copy(_artifacts.default_tensor_path("35"), p)
    cap = _run(tmp_path, tensor_path=p)
    assert cap["loads"] == ["mine.npz"]
    assert cap["res"].diagnostics.resolved["lens"] == "custom"
