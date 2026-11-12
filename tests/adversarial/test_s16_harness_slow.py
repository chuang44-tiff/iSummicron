"""Q-d section 16 harness self-checks (spec section 7, SLOW): the probe's meters
pass-through must hash to G1_MD5_OLD on the q95 JPEG (and NOT on the PNG), and a
pass-through wrapper that rescales its input must MISMATCH."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from summicron.render._bench import G1_FOCUS_XY, G1_IMAGE, G1_MD5_OLD

ROOT = Path(__file__).resolve().parents[2]
PROBE = ROOT / "scripts" / "probe_s16_diopter.py"
TAG = "IMG_6915_meters_off"

pytestmark = pytest.mark.slow


def _md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def _run(out_dir: Path, *extra: str) -> subprocess.CompletedProcess:
    img = ROOT / G1_IMAGE if not Path(G1_IMAGE).is_absolute() else Path(G1_IMAGE)
    cmd = [sys.executable, str(PROBE), "--image", str(img),
           "--focus-xy", str(G1_FOCUS_XY[0]), str(G1_FOCUS_XY[1]),
           "--mode", "meters", "--arm", "meters", "--occlusion-compositing", "off",
           "--out-dir", str(out_dir), *extra]
    env = {**__import__("os").environ, "PYTHONPATH": str(ROOT)}
    return subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=900)


def test_meters_passthrough_jpg_matches_g1_and_png_does_not(tmp_path):
    r = _run(tmp_path)
    assert r.returncode == 0, r.stdout[-800:] + r.stderr[-800:]
    meta = json.loads((tmp_path / f"{TAG}.meta.json").read_text())
    assert meta["pin"]["status"] == "MATCH"
    assert _md5(tmp_path / f"{TAG}.jpg") == G1_MD5_OLD
    # guard against a future "fix" that hashes the wrong artifact
    assert _md5(tmp_path / f"{TAG}.png") != G1_MD5_OLD


def test_rescaling_passthrough_wrapper_mismatches(tmp_path):
    r = _run(tmp_path, "--rescale-input-div", "10")
    assert r.returncode == 3, r.stdout[-800:] + r.stderr[-800:]
    meta = json.loads((tmp_path / f"{TAG}.meta.json").read_text())
    assert meta["pin"]["status"] == "MISMATCH"
    assert _md5(tmp_path / f"{TAG}.jpg") != G1_MD5_OLD
