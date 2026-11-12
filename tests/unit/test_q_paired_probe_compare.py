"""Q-a fix round: ``q_paired_probe.py compare`` fail-closed behaviour.

FAST; synthetic temp dumps (no renders).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("skimage")

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "q_paired_probe", ROOT / "scripts" / "q_paired_probe.py")
probe = importlib.util.module_from_spec(_spec)
sys.modules["q_paired_probe"] = probe
_spec.loader.exec_module(probe)

N = 200


def _dump(path: Path, out_scale: float = 1.0, alpha_shift: float = 0.0):
    rng = np.random.default_rng(0)
    ref = (0.3 + 0.2 * rng.random((N, N, 3))).astype(np.float32)
    yy, xx = np.mgrid[:N, :N]
    r = np.hypot(yy - N / 2, xx - N / 2)
    alpha = np.clip((r - 20.0) / 60.0 + alpha_shift, 0, 1).astype(np.float32)
    dmap = alpha.copy()
    out = (ref * out_scale).astype(np.float32)
    np.savez(path, out=out, ref=ref, alpha=alpha, dmap=dmap)


def _args(root, base="base", var="var"):
    class A:
        pass
    a = A()
    a.root, a.base, a.var = str(root), base, var
    a.lever, a.expect, a.footprint = "band", "better", "none"
    return a


def _mk(root: Path, base_scenes, var_scenes, **vkw):
    (root / "base").mkdir()
    (root / "var").mkdir()
    for s in base_scenes:
        _dump(root / "base" / f"{s}_2048.npz")
    for s in var_scenes:
        _dump(root / "var" / f"{s}_2048.npz", **vkw)


def test_missing_var_for_gated_scene_fails(tmp_path):
    _mk(tmp_path, ["IMG_6651", "Demo_input"], ["Demo_input"])
    assert probe.cmd_compare(_args(tmp_path)) != 0


def test_all_var_missing_fails(tmp_path):
    _mk(tmp_path, ["IMG_6651", "Demo_input"], [])
    assert probe.cmd_compare(_args(tmp_path)) != 0


def test_mismatched_alpha_fails(tmp_path, capsys):
    _mk(tmp_path, ["IMG_6651", "Demo_input"], ["IMG_6651", "Demo_input"],
        alpha_shift=0.05)
    assert probe.cmd_compare(_args(tmp_path)) != 0
    assert "differs from BASE" in capsys.readouterr().out
