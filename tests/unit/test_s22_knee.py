"""S22 S1: physical identity-snap knee."""
from __future__ import annotations

import numpy as np
import pytest

from summicron.render import convolve as cv
from summicron.render._artifacts import default_tensor_path
from tests.fixtures.synthetic_tensor import make_synthetic_tensor


@pytest.fixture(scope="module")
def shipped():
    return cv.load_tensor(default_tensor_path())


def test_knee_diopter_literal_is_node1_offset(shipped):
    od = shipped["object_distances_m"]
    assert cv.KNEE_DIOPTER == 1 / 0.7 - 1 / od[1]  # never a rounded 0.407
    assert cv.KNEE_DIOPTER != 0.407


def test_shipped_axis_knee_is_exactly_d_knee_layered(shipped):
    k = cv._physical_knee_ratio(
        shipped["object_distances_m"], shipped["defocus_ratios"],
        shipped["focus_distance_m"],
    )
    assert abs(k - cv.D_KNEE_LAYERED) < 1e-12
    assert k == 0.1


def test_family_axis_knee():
    from summicron.psf_model import synth_family as sf

    od = sf.object_distances_for_axis("diopter16", np.array([0.7, 20.0]), n_d=16)
    ratios = np.linspace(0.0, 1.0, 16)
    k = cv._physical_knee_ratio(od, ratios, 0.7)
    assert 0.2950 <= k <= 0.2954


def test_clamp_when_knee_exceeds_axis():
    od = np.array([0.7, 1.0, 2.0])
    k = cv._physical_knee_ratio(od, np.array([0.0, 0.5, 1.0]), 0.7, knee_diopter=50.0)
    assert k == 1.0


def _tiny(synth=None):
    t = make_synthetic_tensor(R=2, D=3, H=9, W=9)
    t["synth_family"] = synth
    return t


def _render(t):
    img = np.full((16, 16, 3), 0.4, np.float32)
    depth = np.full((16, 16), 0.7, np.float32)
    lines: list[str] = []
    stats: dict = {}
    cv.render(img, t, depth_m=depth, on_progress=lines.append, stats=stats)
    return lines, stats


def test_codev_tensor_without_key_uses_d_knee_and_identical_line():
    t = _tiny()
    t.pop("synth_family")  # hand-built dicts lack the key entirely
    lines, stats = _render(t)
    assert stats["knee_ratio"] == cv.D_KNEE_LAYERED
    assert any(
        l.startswith("  α-blend: D_KNEE_LAYERED=0.1, alpha_mean=") for l in lines
    )


def test_synthetic_tensor_uses_converted_knee():
    t = _tiny("synthetic_disc_v0")
    lines, stats = _render(t)
    exp = cv._physical_knee_ratio(
        t["object_distances_m"], t["defocus_ratios"], t["focus_distance_m"]
    )
    assert stats["knee_ratio"] == pytest.approx(exp)
    assert stats["knee_ratio"] != cv.D_KNEE_LAYERED
    assert any("knee=" in l and "D physical" in l for l in lines)


@pytest.mark.parametrize("tag", ["zoom_control", "something_else"])
def test_other_synth_family_keeps_ratio_knee(tag):
    _, stats = _render(_tiny(tag))
    assert stats["knee_ratio"] == cv.D_KNEE_LAYERED


def test_load_tensor_synth_family_key_none_for_shipped(shipped):
    assert shipped["synth_family"] is None
