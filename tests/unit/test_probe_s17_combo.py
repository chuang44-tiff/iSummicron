"""Composition tests for scripts/probe_s17_combo.py (guided before diopter) on synthetic arrays."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
if not (_SCRIPTS / "probe_s17_combo.py").exists():
    pytest.skip("scripts/probe_s17_combo.py absent (staged release tree)", allow_module_level=True)
sys.path.insert(0, str(_SCRIPTS))
import probe_s17_combo as P  # noqa: E402

OD = np.geomspace(0.7, 20.0, 11)


def _fakes(raw):
    calls = {"lidar": [], "tap": []}

    def lidar(path, target_hw=None, depth_guided="off", depth_matte="off"):
        calls["lidar"].append((depth_guided, depth_matte))
        return raw * 0.5 if depth_guided == "on" else raw.copy()  # "guided" = distinguishable output

    def tap(depth, focus_xy, focus_target_m=0.7, *a, **k):
        calls["tap"].append(np.array(depth, copy=True))
        anchor = float(np.median(depth))
        return depth * 0 + focus_target_m, anchor

    return lidar, tap, calls


@pytest.mark.parametrize("arm,guided,diopter", [("today", 0, 0), ("guided", 1, 0), ("diopter-auto", 0, 1), ("combo", 1, 1)])
def test_arm_flags(arm, guided, diopter):
    assert P.arm_flags(arm) == (bool(guided), bool(diopter))


def test_unknown_arm():
    with pytest.raises(ValueError):
        P.arm_flags("nope")


def test_today_is_passthrough():
    raw = np.linspace(0.8, 9.0, 64, dtype=np.float32).reshape(8, 8)
    lidar, tap, calls = _fakes(raw)
    lw, tw, st = P.build_wrappers("today", lidar, tap, od=OD)
    d = lw("x", target_hw=(8, 8))
    shifted, anchor = tw(d, (0.5, 0.5))
    assert calls["lidar"] == [("off", "off")]
    assert np.array_equal(calls["tap"][0], raw)
    assert np.all(shifted == 0.7) and st["g"] is None


def test_combo_guided_feeds_diopter():
    raw = np.linspace(0.8, 9.0, 64, dtype=np.float32).reshape(8, 8)
    lidar, tap, calls = _fakes(raw)
    lw, tw, st = P.build_wrappers("combo", lidar, tap, od=OD)
    d = lw("x", target_hw=(8, 8))
    mapped, anchor = tw(d, (0.5, 0.5))
    assert calls["lidar"] == [("on", "off")]
    assert np.array_equal(calls["tap"][0], raw * 0.5)  # tap/diopter saw the GUIDED map
    assert st["order"] == ["lidar", "tap"]
    assert anchor == pytest.approx(float(np.median(raw * 0.5)))
    from summicron.depth.diopter import auto_gain, diopter_equivalent_depth

    g = auto_gain(raw * 0.5, anchor, z_focus=0.7, od=OD)
    assert st["g"] == pytest.approx(g)
    assert np.allclose(mapped, diopter_equivalent_depth(raw * 0.5, anchor, gain=g, z_focus=0.7, z_far=OD[-1]))
    assert mapped.min() >= 0.7 - 1e-6 and mapped.max() <= OD[-1] + 1e-4


def test_diopter_only_uses_unguided_and_auto_lo():
    raw = np.full((8, 8), 1.0, np.float32)  # flat scene -> auto_gain returns lo
    lidar, tap, calls = _fakes(raw)
    lw, tw, st = P.build_wrappers("diopter-auto", lidar, tap, od=OD, auto_lo=0.5)
    tw(lw("x"), (0.5, 0.5))
    assert calls["lidar"] == [("off", "off")]
    assert st["g"] == 0.5


def test_guided_arm_keeps_meters():
    raw = np.linspace(0.8, 9.0, 64, dtype=np.float32).reshape(8, 8)
    lidar, tap, calls = _fakes(raw)
    lw, tw, st = P.build_wrappers("guided", lidar, tap, od=OD, state={"depth_scale": 10.0})
    st["depth_scale"] = 10.0
    out, _ = tw(lw("x"), (0.5, 0.5))
    assert calls["lidar"] == [("on", "off")] and st["g"] is None and np.all(out == 0.7)


def test_tags():
    assert P.make_tag("A", "today") == "A_today_off"
    assert P.make_tag("A", "combo", "off", 0.5) == "A_combo_lo0.5_off"
    assert P.make_tag("A", "guided", "off", 0.5) == "A_guided_off"
