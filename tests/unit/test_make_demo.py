"""WP-B (RUNS-PRUNE + R-demo) — demo/make_demo.py.

FAST tests mock ``subprocess.run`` (fake render writes a synthetic JPEG),
``load_image`` (no HEIC decode) and ``_md5`` (per-filename control), and
redirect ``OUT`` / ``ON_PIN_JPG`` into ``tmp_path`` so the real ``runs/``
tree is never touched. One SLOW test runs ``--only g1`` for real.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from summicron.render._bench import CANONICAL_FLAGS, G1_MD5, ON_MD5

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "demo" / "make_demo.py"

pytestmark = pytest.mark.skipif(
    not SCRIPT.is_file(), reason="demo/ not shipped (public release)"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("make_demo", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def md(tmp_path, monkeypatch):
    """make_demo with OUT/ON_PIN_JPG in tmp_path and all I/O mocked."""
    mod = _load_module()
    out = tmp_path / "runs" / "demo"
    monkeypatch.setattr(mod, "OUT", out)
    pin = tmp_path / "pin.jpg"
    Image.fromarray(np.full((32, 24, 3), 100, np.uint8)).save(pin, quality=95)
    monkeypatch.setattr(mod, "ON_PIN_JPG", pin)

    state = {"calls": [], "md5": {}, "loads": []}

    def fake_run(argv, **kw):
        state["calls"].append(list(argv))
        dst = Path(argv[argv.index("--output") + 1])
        dst.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.full((32, 24, 3), 120, np.uint8)).save(dst, quality=95)
        return subprocess.CompletedProcess(argv, 0, "", "")

    def fake_load(path):
        state["loads"].append(str(path))
        return np.full((64, 48, 3), 0.5, np.float32)

    def fake_md5(path):
        name = Path(path).stem
        return state["md5"].get(name, {"g1": G1_MD5, "g1_occ": ON_MD5}.get(name, "f" * 32))

    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    monkeypatch.setattr(mod, "load_image", fake_load)
    monkeypatch.setattr(mod, "_md5", fake_md5)
    monkeypatch.setattr(mod, "_tensor_md5", lambda: "t" * 32)
    mod._state = state
    return mod


def _rows(calls):
    return [Path(c[c.index("--output") + 1]).stem for c in calls]


def test_manifest_parses_and_images_tracked():
    mod = _load_module()
    rows = mod.load_manifest()
    assert [r["name"] for r in rows] == ["g1", "g1_occ", "stuffed", "demo"]
    tracked = set(
        subprocess.run(["git", "ls-files", "-z", "iphoneImage"], cwd=ROOT,
                       capture_output=True, text=True, check=True)
        .stdout.split("\0")
    )
    for r in rows:
        for key in ("image", "ref"):
            if r.get(key):
                assert f"iphoneImage/{r[key]}" in tracked, r[key]
    assert "IMG_4343" not in json.dumps(rows)


def test_argv_canonical_and_ref_path_preserved(md):
    rows = {r["name"]: r for r in md.load_manifest()}
    out = md.OUT / "x.jpg"
    g1 = md.build_argv(rows["g1"], out)
    assert g1 == [sys.executable, "-m", "summicron.render.convolve",
                  "--image", "iphoneImage/IMG_6915.HEIC", *CANONICAL_FLAGS,
                  "--focus-xy", "0.4", "0.45", "--output", str(out)]
    assert "--tensor" not in g1
    occ = md.build_argv(rows["g1_occ"], out)
    assert occ[:-2] == g1[:-2] + ["--occlusion-compositing", "on"]
    assert md.main(["--only", "stuffed"]) == 0
    assert str(ROOT / "iphoneImage" / "IMG_6651 2.HEIC") in md._state["loads"]


def test_g1_runs_first_and_mismatch_aborts(md):
    md._state["md5"]["g1"] = "0" * 32
    assert md.main([]) == 1
    assert _rows(md._state["calls"]) == ["g1"]
    assert not (md.OUT / "manifest.lock.json").exists()


def test_only_stuffed_still_runs_g1_first(md):
    assert md.main(["--only", "stuffed"]) == 0
    assert _rows(md._state["calls"]) == ["g1", "stuffed"]


def test_full_run_order(md):
    assert md.main([]) == 0
    assert _rows(md._state["calls"]) == ["g1", "g1_occ", "stuffed", "demo"]
    for n in ("g1", "g1_occ", "stuffed", "demo"):
        assert (md.OUT / f"{n}_panel.jpg").exists()


def test_g1_occ_mismatch_warns_records_max_abs(md, capsys):
    md._state["md5"]["g1_occ"] = "1" * 32
    assert md.main(["--only", "g1_occ"]) == 0
    assert "WARNING g1_occ" in capsys.readouterr().out
    lock = json.loads((md.OUT / "manifest.lock.json").read_text())
    rec = lock["rows"]["g1_occ"]
    assert rec["pin_match"] is False and rec["md5"] == "1" * 32
    assert isinstance(rec["max_abs_vs_pin"], int) and rec["max_abs_vs_pin"] > 0
    assert lock["rows"]["g1"]["pin_match"] is True
    for key in ("git_sha", "tensor_md5", "mac_ver", "pillow"):
        assert key in lock


def test_panel_shape_and_no_exif(md, tmp_path):
    inp = np.random.default_rng(0).random((120, 90, 3)).astype(np.float32)
    render = np.zeros((60, 45, 3), np.uint8)
    ref = np.ones((240, 180, 3), np.float32)
    panel = md.make_panel(inp, render, ref, (0.5, 0.5), gap=4)
    side = 45 // 2
    assert panel.shape == (60, 45 * 3 + side + 3 * 4, 3)
    assert panel.dtype == np.uint8
    no_ref = md.make_panel(inp, render, None, (0.0, 1.0), gap=4)
    assert no_ref.shape == (60, 45 * 2 + side + 2 * 4, 3)
    p = tmp_path / "panel.jpg"
    md.save_panel(panel, p)
    with Image.open(p) as im:
        assert len(im.getexif()) == 0


def test_outputs_under_runs_demo(md):
    assert md.main([]) == 0
    out = md.OUT.resolve()
    for c in md._state["calls"]:
        assert Path(c[c.index("--output") + 1]).resolve().is_relative_to(out)
    produced = [p for p in md.OUT.parent.rglob("*") if p.is_file()]
    assert produced and all(p.resolve().is_relative_to(out) for p in produced)
    with pytest.raises(ValueError):
        md._out_path("../escape")


def test_unknown_only_row_errors(md):
    with pytest.raises(SystemExit):
        md.main(["--only", "IMG_4343"])
    assert md._state["calls"] == []


@pytest.mark.slow
def test_real_g1_md5(tmp_path, monkeypatch):
    if sys.platform != "darwin":
        pytest.skip("canonical bench requires macOS HDR")
    mod = _load_module()
    monkeypatch.setattr(mod, "OUT", tmp_path / "demo")
    assert mod.main(["--only", "g1"]) == 0
    assert mod._md5(tmp_path / "demo" / "g1.jpg") == G1_MD5
