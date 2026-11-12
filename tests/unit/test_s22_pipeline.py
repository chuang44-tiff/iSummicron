"""S22 S1: pipeline wiring of the synthetic family (fast fakes, no real render)."""
from __future__ import annotations

import io
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from summicron.render import convolve
from summicron.render.api import RenderOptions, render_image


def _tensor_npz(path: Path, *, synth: bool, D: int = 5) -> Path:
    R, K = 3, 11
    kernels = np.zeros((R, D, K, K), dtype=np.float32)
    kernels[:, :, K // 2, K // 2] = 1.0
    extra = {"synth_family": np.array("synthetic_disc_v0")} if synth else {}
    np.savez(
        str(path), kernels=kernels,
        radial_edges=np.linspace(0.0, 1.0, R + 1),
        defocus_ratios=np.linspace(0.0, 1.0, D),
        corpus_ref_long_edge=100, otf_norm_applied=True, delta_guard_applied=True,
        schema_version=np.array("1.0"),
        object_distances_m=np.linspace(0.7, 5.0, D), focus_distance_m=0.7, **extra,
    )
    return path


def _image(path: Path) -> Path:
    from PIL import Image

    arr = np.random.default_rng(0).integers(0, 256, (64, 64, 3), dtype=np.uint8)
    Image.fromarray(arr, "RGB").save(str(path), "JPEG", quality=80)
    return path


def _run(tmp_path, *, on_progress=None, **kw):
    img = _image(tmp_path / "t.jpg")
    import importlib

    # other tests reload/replace the module; resolve it at call time
    convolve = importlib.import_module("summicron.render.convolve")
    synth_npz = _tensor_npz(tmp_path / "s.npz", synth=True)
    codev_npz = _tensor_npz(tmp_path / "c.npz", synth=False)
    depth = np.linspace(0.7, 4.0, 64 * 64, dtype=np.float32).reshape(64, 64)
    cap: dict = {"family_calls": [], "tensor_loads": [], "auto_gain": 0}

    def fake_family(aperture_mm, *, on_progress=None, **k):
        cap["family_calls"].append(aperture_mm)
        return convolve.load_tensor(synth_npz), synth_npz, True

    real_load = convolve.load_tensor

    def spy_load(p):
        cap["tensor_loads"].append(Path(p))
        return real_load(p)

    from summicron.depth import diopter

    real_ag = diopter.auto_gain

    def spy_ag(*a, **k):
        cap["auto_gain"] += 1
        return real_ag(*a, **k)

    def fake_render(image, tensor, **k):
        cap["render"] = k
        cap["tensor"] = tensor
        h, w = image.shape[:2]
        return np.zeros((h, w, 3), np.uint8), np.zeros((h, w, 3), np.float32)

    saved: dict = {}

    def fake_save(u8, stem, out_dir, **k):
        saved.update(k)
        return out_dir / "x.jpg"

    if kw.get("kernel_family", "codev") == "codev":
        kw.setdefault("tensor_path", codev_npz)
    opts = RenderOptions(image_path=img, focus_xy=(0.5, 0.5), out_dir=tmp_path / "o",
                         depth_source="lidar", hdr_source="none", **kw)
    with patch("summicron.depth.estimate.extract_lidar_depth", side_effect=lambda *a, **k: depth.copy()), \
         patch("summicron.render._family.load_family_tensor", side_effect=fake_family), \
         patch("summicron.render.convolve.load_tensor", side_effect=spy_load), \
         patch("summicron.depth.diopter.auto_gain", side_effect=spy_ag), \
         patch("summicron.render.convolve.render", side_effect=fake_render), \
         patch("summicron.render.convolve.save_results", side_effect=fake_save):
        res = render_image(opts, on_progress=on_progress)
    cap.update(saved=saved, res=res)
    return cap


def test_synthetic_uses_family_not_tensor_load(tmp_path):
    cap = _run(tmp_path, kernel_family="synthetic")
    assert cap["family_calls"] == [17.5]
    # the only load_tensor call is the (faked) family builder's own validation load
    assert cap["tensor_loads"] == [tmp_path / "s.npz"]
    assert cap["render"]["bright_stamp_gain"] == 2.0
    assert cap["tensor"]["synth_family"] == "synthetic_disc_v0"


def test_synthetic_gain_is_exactly_one_and_labelled(tmp_path):
    lines: list[str] = []
    cap = _run(tmp_path, kernel_family="synthetic", on_progress=lines.append)
    assert cap["auto_gain"] == 0
    assert any("gain=1.000" in l and "(synthetic: physical)" in l for l in lines)
    assert any(l.startswith("Loading synthetic family: A=17.5 mm (cache hit):") for l in lines)
    assert "_dio1.00" in cap["saved"]["name_suffix"]


def test_synthetic_explicit_gain_honoured(tmp_path):
    lines: list[str] = []
    cap = _run(tmp_path, kernel_family="synthetic", depth_gain=1.5, on_progress=lines.append)
    assert "_dio1.50" in cap["saved"]["name_suffix"]
    assert not any("synthetic: physical" in l for l in lines)


def test_aperture_passed_to_family(tmp_path):
    cap = _run(tmp_path, kernel_family="synthetic", lens_aperture_mm=25.0)
    assert cap["family_calls"] == [25.0]
    assert cap["saved"]["name_suffix"].endswith("_synA25_l1g2")


@pytest.mark.parametrize("kw,suffix", [
    ({}, ""),
    ({"kernel_family": "synthetic"}, "_synA17.5_l1g2"),
    ({"kernel_family": "synthetic", "bright_stamp_gain": 4.0}, "_synA17.5_l1g4"),
    ({"kernel_family": "synthetic", "bright_stamp": "peak"}, "_synA17.5_peak"),
    ({"bright_stamp": "l1"}, "_l1g2"),
])
def test_name_suffix(tmp_path, kw, suffix):
    cap = _run(tmp_path, depth_mapping="diopter", **kw)
    got = cap["saved"]["name_suffix"]
    assert got.endswith(suffix) if suffix else not any(
        t in got for t in ("syn", "l1g", "_peak"))


def test_codev_default_unchanged(tmp_path):
    lines: list[str] = []
    cap = _run(tmp_path, depth_mapping="diopter", on_progress=lines.append)
    assert cap["auto_gain"] == 1
    assert cap["render"]["bright_stamp_gain"] is None
    assert cap["family_calls"] == []
    assert not any("synthetic" in l or "knee=" in l or "l1g" in l for l in lines)
    r = cap["res"].diagnostics.resolved
    assert r["kernel_family"] == "codev" and r["bright_stamp"] == "peak"
    assert "lens_aperture_mm" not in r and "bright_stamp_gain" not in r


def test_resolved_dict_synthetic(tmp_path):
    r = _run(tmp_path, kernel_family="synthetic")["res"].diagnostics.resolved
    assert r["kernel_family"] == "synthetic" and r["bright_stamp"] == "l1"
    assert r["lens_aperture_mm"] == "17.5" and r["bright_stamp_gain"] == "2"


def test_default_stdout_has_no_family_text(tmp_path):
    buf = io.StringIO()
    with redirect_stdout(buf):
        _run(tmp_path, depth_mapping="diopter")
    out = buf.getvalue()
    assert "synthetic" not in out and "knee=" not in out


def test_native_synthetic_passes_memory_guard():
    from summicron.render import convolve as cv

    est = cv._estimate_peak_bytes(4284, 5712, 16, kernel_size=183)
    assert est < cv.MAX_BYTES
    assert cv._estimate_peak_bytes(4284, 5712, 16, kernel_size=183, occlusion=True) < cv.MAX_BYTES


def test_matte_spill_dp():
    from summicron.render import _quality

    dp = np.zeros((10, 10), np.float32)
    dp[:2] = 0.5   # 20 px above 0.41
    dp[:1] = 1.0   # 10 px above 0.9
    matte = np.ones((10, 10), np.float32)
    r = _quality.matte_spill_dp(dp, matte)
    assert r[0.41] == pytest.approx(0.2) and r[0.9] == pytest.approx(0.1)
    half = np.zeros((10, 10), np.float32)
    half[5:] = 1.0  # subject excludes the spill rows
    assert _quality.matte_spill_dp(dp, half)[0.41] == 0.0
    assert np.isnan(_quality.matte_spill_dp(dp, half * 0)[0.41])
    # smaller matte is nearest-resized
    assert _quality.matte_spill_dp(dp, np.ones((5, 5), np.float32))[0.41] == pytest.approx(0.2)
