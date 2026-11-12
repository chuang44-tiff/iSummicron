"""Tests for scripts/clean_runs.py (RUNS-PRUNE WP-A). tmp_path fake trees ONLY."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "clean_runs", Path(__file__).resolve().parents[2] / "scripts" / "clean_runs.py")
cr = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(cr)

G1 = b"g1-bytes"
ON = b"on-bytes"
STALE = b"stale-on"
OTF = b"otf-explicit"


def _md5(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()


def _w(root: Path, rel: str, data: bytes = b"x") -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)


def _plan():
    """The real PLAN with expect_md5 values swapped for fake-content md5s."""
    swap = {cr.G1_MD5_PRE_QB: _md5(G1), cr.ON_STALE_MD5: _md5(STALE),
            cr.OTF_EXPLICIT_MD5: _md5(OTF)}
    return tuple(r[:4] + ((swap[r[4]],) if len(r) > 4 else ()) for r in cr.PLAN)


PINS = {
    "vibe_bench/IMG_6915_R2_layered.jpg": _md5(G1),
    "vibe_bench/IMG_6915_G1_baseline_default.jpg": _md5(G1),
    "render/current_check/IMG_6915_occ_on.jpg": _md5(ON),
}

KEEP = [
    "ag_" "dblgauss/a.npz", "vibe_bench/README.md", "vibe_bench/IMG_6915_occ_ON_pin.jpg",
    "cherry_tensor4d.npz", "extract_r2/extract_cutouts.npz", "extract_r3/extract_report.json",
    "eval/bubble_grid_input.npz", "eval/bubble_grid_input.json",
    "eval/identity_tensor_iphone5712.npz", "render/current_check/other.jpg",
]
UNCLASSIFIED = ["render/armedge_visual/a.png", "render/edge_crop/b.png"]


@pytest.fixture
def tree(tmp_path):
    root = tmp_path / "runs"
    for rel in KEEP + UNCLASSIFIED:
        _w(root, rel, rel.encode())
    _w(root, "vibe_bench/IMG_6915_R2_layered.jpg", G1)
    _w(root, "vibe_bench/IMG_6915_G1_baseline_default.jpg", G1)
    _w(root, "render/current_check/IMG_6915_occ_on.jpg", ON)
    _w(root, "extract_r2/tmp/s0.npy")
    _w(root, "extract_r3/tmp/s0.npy")
    _w(root, "vibe_bench/IMG_6915_OFF_after.jpg", G1)
    _w(root, "vibe_bench/IMG_6915_ON2.jpg", STALE)
    _w(root, "vibe_bench/IMG_6915_G1_baseline_explicit.jpg", OTF)
    _w(root, "render/visual_p13/a.png")
    _w(root, "render/visual_p02/a.png")
    _w(root, "render/depth_refine_v2/a.png")
    _w(root, "render/probe_occlusion/a.png")
    _w(root, "eval/usaf_depth_rendered_preP0.npz")
    _w(root, "eval/probe_section9_baselines.json")
    _w(root, "day1_full_r2.json")
    _w(root, "kernels/k.png")
    _w(root, ".DS_Store")
    return root


def _snap(root: Path, skip_archive: bool = True) -> dict:
    out = {}
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root).as_posix()
        if skip_archive and rel.startswith("_archive"):
            continue
        if p.is_symlink():
            out[rel] = ("link", str(p.readlink()))
        elif p.is_file():
            out[rel] = ("file", _md5(p.read_bytes()))
        else:
            out[rel] = ("dir",)
    return out


def _run(root, apply, **kw):
    lines: list[str] = []
    kw.setdefault("plan", _plan())
    kw.setdefault("pins", PINS)
    rc = cr.run(root, apply=apply, out=lines.append, **kw)
    return rc, lines


def test_brace_expand():
    assert cr.expand("a{,2,_r}.jpg") == ["a.jpg", "a2.jpg", "a_r.jpg"]
    assert cr.expand("r{2,3}/{x,y}") == ["r2/x", "r2/y", "r3/x", "r3/y"]


def test_dry_run_changes_nothing(tree):
    before = _snap(tree, skip_archive=False)
    rc, lines = _run(tree, apply=False)
    assert rc == 0
    assert _snap(tree, skip_archive=False) == before
    assert not (tree / "_archive").exists()
    unc = [ln.split("\t")[0] for ln in lines if ln.endswith("\tUNCLASSIFIED")]
    assert unc == ["render/armedge_visual", "render/edge_crop"]


def test_apply_moves_and_deletes_exactly_plan(tree):
    keep_before = {k: v for k, v in _snap(tree).items()
                   if any(k.startswith(x) for x in KEEP + UNCLASSIFIED + list(PINS))}
    rc, _ = _run(tree, apply=True)
    assert rc == 0
    after = _snap(tree)
    for k, v in keep_before.items():
        assert after[k] == v
    for gone in ["extract_r2/tmp", "extract_r3/tmp", "vibe_bench/IMG_6915_OFF_after.jpg",
                 "vibe_bench/IMG_6915_ON2.jpg", "vibe_bench/IMG_6915_G1_baseline_explicit.jpg",
                 "render/visual_p13", "render/visual_p02", "render/depth_refine_v2",
                 "render/probe_occlusion", "eval/usaf_depth_rendered_preP0.npz",
                 "day1_full_r2.json", "kernels"]:
        assert not (tree / gone).exists(), gone
    a = tree / "_archive"
    for moved in ["phase0-1/visual_p02/a.png", "R-highlight-guard/depth_refine_v2/a.png",
                  "R-occlusion-layering/probe_occlusion/a.png",
                  "eval-P0/usaf_depth_rendered_preP0.npz",
                  "eval-P0/probe_section9_baselines.json",
                  "shelved-corpus/day1_full_r2.json", "shelved-corpus/kernels/k.png"]:
        assert (a / moved).is_file(), moved
    assert not (a / "phase0-1/visual_p13").exists()


def test_allowlist_overlap_aborts(tree):
    before = _snap(tree, skip_archive=False)
    plan = _plan() + (("render/current_check", "delete", None, "bad"),)
    rc, lines = _run(tree, apply=True, plan=plan)
    assert rc == 1
    assert any("allowlist overlap" in ln for ln in lines)
    assert _snap(tree, skip_archive=False) == before


def test_symlink_target_aborts_and_outside_sentinel_survives(tree, tmp_path):
    outside = tmp_path / "outside"
    _w(outside, "sentinel.txt", b"keep me")
    (tree / "render" / "visual_p15").symlink_to(outside, target_is_directory=True)
    before = _snap(tree, skip_archive=False)
    rc, lines = _run(tree, apply=True)
    assert rc == 1
    assert any("symlink" in ln for ln in lines)
    assert _snap(tree, skip_archive=False) == before
    assert (outside / "sentinel.txt").read_bytes() == b"keep me"


def test_pin_mismatch_aborts(tree):
    before = _snap(tree, skip_archive=False)
    pins = dict(PINS, **{"render/current_check/IMG_6915_occ_on.jpg": _md5(b"other")})
    rc, lines = _run(tree, apply=True, pins=pins)
    assert rc == 1
    assert any("pin md5 mismatch" in ln for ln in lines)
    assert _snap(tree, skip_archive=False) == before


def test_absent_pin_is_reported_not_fatal(tree):
    (tree / "vibe_bench/IMG_6915_R2_layered.jpg").unlink()
    rc, lines = _run(tree, apply=False)
    assert rc == 0
    assert any("pin absent" in ln for ln in lines)
    assert not any("PREFLIGHT FAIL" in ln for ln in lines)


def test_expect_md5_mismatch_aborts(tree):
    _w(tree, "vibe_bench/IMG_6915_OFF_r2.jpg", b"not g1")
    before = _snap(tree, skip_archive=False)
    rc, lines = _run(tree, apply=True)
    assert rc == 1
    assert any("expect_md5 mismatch: vibe_bench/IMG_6915_OFF_r2.jpg" in ln for ln in lines)
    assert _snap(tree, skip_archive=False) == before


def test_existing_archive_destination_aborts(tree):
    _w(tree, "_archive/phase0-1/visual_p02/old.png")
    before = _snap(tree, skip_archive=False)
    rc, lines = _run(tree, apply=True)
    assert rc == 1
    assert any("archive destination exists" in ln for ln in lines)
    assert _snap(tree, skip_archive=False) == before


def test_jsonl_one_line_per_action(tree):
    rc, lines = _run(tree, apply=True)
    assert rc == 0
    n_targets = sum(1 for ln in lines if ln.count("\t") == 4)
    recs = [json.loads(ln) for ln in
            (tree / "_archive/clean_log.jsonl").read_text().splitlines()]
    assert len(recs) == n_targets == 13
    for r in recs:
        assert set(r) == {"action", "old_path", "new_path", "bytes", "group", "ts"}
        if r["action"] == "archive":
            assert r["old_path"] and r["new_path"].startswith(f"_archive/{r['group']}/")
            assert (tree / r["new_path"]).exists() and not (tree / r["old_path"]).exists()
        else:
            assert r["new_path"] is None


def test_runs_demo_is_allowlisted(tree):
    _w(tree, "demo/g1.jpg", G1)
    _w(tree, "demo/manifest.lock.json", b"{}")
    rc, lines = _run(tree, apply=False)
    assert rc == 0
    assert not any(ln.startswith("demo") for ln in lines)
    rc, _ = _run(tree, apply=True)
    assert rc == 0
    assert (tree / "demo/g1.jpg").read_bytes() == G1
    assert (tree / "demo/manifest.lock.json").is_file()
