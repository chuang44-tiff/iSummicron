"""Harness tests for scripts/probe_s15_matte_depth.py + scripts/s15_matte_metrics.py
(section15 Track A, spec section 7 "Harness polarity" and G2b ties)."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
if not ((ROOT / "scripts" / "probe_s15_matte_depth.py").exists()
        and (ROOT / "scripts" / "s15_matte_metrics.py").exists()):
    pytest.skip("scripts/ probes absent (staged release tree)", allow_module_level=True)
sys.path.insert(0, str(ROOT / "scripts"))

import probe_s15_matte_depth as probe  # noqa: E402
import s15_matte_metrics as M  # noqa: E402

IMG = ROOT / "iphoneImage" / "IMG_6915.HEIC"
FOCUS = ("0.40", "0.45")
needs_img = pytest.mark.skipif(not IMG.exists(), reason="IMG_6915.HEIC absent")


def _dry(out_dir: Path, *extra: str) -> None:
    rc = probe.main(["--image", str(IMG), "--focus-xy", *FOCUS, "--dry-run",
                     "--out-dir", str(out_dir), *extra])
    assert rc == 0


@pytest.fixture(scope="module")
def dry_dir(tmp_path_factory):
    if not IMG.exists():
        pytest.skip("IMG_6915.HEIC absent")
    d = tmp_path_factory.mktemp("probe_dry")
    _dry(d, "--arm", "baseline")
    _dry(d, "--arm", "guided")
    _dry(d, "--arm", "matte", "--curve", "linear")
    _dry(d, "--arm", "both", "--curve", "linear")
    _dry(d, "--arm", "both", "--curve", "linear", "--wrong-way")
    _dry(d, "--arm", "hook", "--hook-guided", "on", "--hook-matte", "off")
    _dry(d, "--arm", "hook", "--hook-guided", "off", "--hook-matte", "on")
    _dry(d, "--arm", "hook", "--hook-guided", "on", "--hook-matte", "on")
    return d


def _depth(d: Path, arm: str) -> np.ndarray:
    return np.load(d / f"IMG_6915_{arm}_off.depth.npy")


def test_tags():
    assert probe.make_tag("S", "baseline", "off") == "S_baseline_off"
    assert probe.make_tag("S", "matte", "off", "sqrt") == "S_matte-sqrt_off"
    assert probe.make_tag("S", "both", "on", "linear", True) == "S_both-linear-wrongway_on"
    assert probe.make_tag("S", "hook", "off", hook_g="on", hook_m="off") == "S_hook-gon-moff_off"


@pytest.mark.parametrize("extra", [
    ["--arm", "baseline", "--curve", "sqrt"],
    ["--arm", "guided", "--wrong-way"],
    ["--arm", "matte", "--hook-guided", "on"],
    [],
])
def test_cli_misuse_rejected(extra):
    with pytest.raises(SystemExit):
        probe.main(["--image", "x.heic", "--focus-xy", "0.5", "0.5", "--dry-run", *extra])


@needs_img
def test_hook_ties_g2b(dry_dir):
    assert np.array_equal(_depth(dry_dir, "hook-gon-moff"), _depth(dry_dir, "guided"))
    assert np.array_equal(_depth(dry_dir, "hook-goff-mon"), _depth(dry_dir, "matte-linear"))
    assert np.array_equal(_depth(dry_dir, "hook-gon-mon"), _depth(dry_dir, "both-linear"))


@needs_img
def test_guided_runs_inside_hook(dry_dir):
    assert not np.array_equal(_depth(dry_dir, "hook-gon-mon"), _depth(dry_dir, "hook-goff-mon"))


@needs_img
def test_wrong_way_reaches_leaf_and_changes_same_pixels(dry_dir):
    n, w = _depth(dry_dir, "both-linear"), _depth(dry_dir, "both-linear-wrongway")
    assert not np.array_equal(n, w)
    assert np.array_equal(M._png_mask(dry_dir / "IMG_6915_both-linear_off.changed.png"),
                          M._png_mask(dry_dir / "IMG_6915_both-linear-wrongway_off.changed.png"))


@needs_img
def test_artifacts_and_dtypes(dry_dir):
    t = "IMG_6915_both-linear_off"
    for suf in (".depth.npy", ".base.npy", ".alpha.npy", ".band.png", ".changed.png", ".report.json",
                ".warnings.txt", ".info.json", ".depth.png"):
        assert (dry_dir / f"{t}{suf}").exists(), suf
    assert np.load(dry_dir / f"{t}.depth.npy").dtype == np.float32
    assert np.load(dry_dir / f"{t}.base.npy").dtype == np.float32
    assert np.load(dry_dir / f"{t}.alpha.npy").dtype == np.float16
    # guided arm: no report/band artefacts
    assert not (dry_dir / "IMG_6915_guided_off.report.json").exists()


@needs_img
def test_outside_band_identity_via_metrics(dry_dir):
    run = M.Run(dry_dir, "IMG_6915")
    rep, _lines = M.depth_stage(run)
    assert rep["G2"]["both-linear"]["pass"]
    assert rep["G2b"]["hook-gon-mon==both-linear"]
    assert rep["polarity"]["identity_ok"] and rep["polarity"]["changed_array_equal"]


def test_dry_run_hw_matches_downres():
    if not IMG.exists():
        pytest.skip("IMG_6915.HEIC absent")
    assert probe.render_hw(IMG) == (2048, 1536)


# ---- metric helpers (synthetic) -------------------------------------------
def test_aai_axis_vs_diagonal():
    h = w = 64
    x = np.arange(w)[None, :].repeat(h, 0).astype(float)
    y = np.arange(h)[:, None].repeat(w, 1).astype(float)
    mask = np.ones((h, w), bool)
    gx, gy = M.sobel8(np.where(x > 32, 1.0, 0.0))
    assert M.aai(gx, gy, mask) > 0.99
    gx, gy = M.sobel8(np.where(x + y > 64, 1.0, 0.0))
    assert M.aai(gx, gy, mask) < 0.1


def test_step_geometry_and_identity_confinement():
    d = np.full((64, 64), 1.0, np.float32)
    d[:, 32:] = 4.0
    geom = M.step_geometry(d)
    assert geom[0].any() and geom[2][:, 30].all() and geom[3][:, 0].all()
    r = M.staircase_confinement(d, d.copy(), geom)
    assert r["changed_frac"] == 0.0 and r["p99_far"] == 0.0
