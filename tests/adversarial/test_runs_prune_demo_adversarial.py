"""Adversarial probes for RUNS-PRUNE (scripts/clean_runs.py) + R-demo (demo/make_demo.py).

clean_runs: tmp_path fake runs trees ONLY -- the real runs/ is never scanned or
mutated. make_demo: subprocess.run / load_image / _md5 mocked, OUT redirected
into tmp_path. One read-only check of an existing runs/demo/manifest.lock.json.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from summicron.render import _bench
from summicron.render._bench import CANONICAL_FLAGS, G1_MD5, ON_MD5

ROOT = Path(__file__).resolve().parents[2]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cr = _load("clean_runs_adv", ROOT / "scripts" / "clean_runs.py")

# ---------------------------------------------------------------- clean_runs

G1 = b"g1-bytes"
ON = b"on-bytes"
STALE = b"stale-on"
OTF = b"otf-explicit"


def _md5b(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()


def _w(root: Path, rel: str, data: bytes = b"x") -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    return p


def _plan():
    swap = {cr.G1_MD5_PRE_QB: _md5b(G1), cr.ON_STALE_MD5: _md5b(STALE),
            cr.OTF_EXPLICIT_MD5: _md5b(OTF)}
    return tuple(r[:4] + ((swap[r[4]],) if len(r) > 4 else ()) for r in cr.PLAN)


PINS = {
    "vibe_bench/IMG_6915_R2_layered.jpg": _md5b(G1),
    "vibe_bench/IMG_6915_G1_baseline_default.jpg": _md5b(G1),
    "render/current_check/IMG_6915_occ_on.jpg": _md5b(ON),
}


@pytest.fixture
def tree(tmp_path):
    root = tmp_path / "runs"
    for rel in ("ag_" "dblgauss/a.npz", "vibe_bench/README.md", "cherry_tensor4d.npz",
                "eval/bubble_input.npz", "render/armedge_visual/a.png",
                "demo/g1.jpg"):
        _w(root, rel, rel.encode())
    _w(root, "vibe_bench/IMG_6915_R2_layered.jpg", G1)
    _w(root, "vibe_bench/IMG_6915_G1_baseline_default.jpg", G1)
    _w(root, "render/current_check/IMG_6915_occ_on.jpg", ON)
    _w(root, "extract_r2/tmp/s0.npy")
    _w(root, "vibe_bench/IMG_6915_OFF_after.jpg", G1)
    _w(root, "render/visual_p13/a.png")
    _w(root, "render/visual_p02/a.png")
    _w(root, "render/depth_refine/a.png")
    _w(root, "eval/usaf_rendered_preP0.npz")
    _w(root, "kernels/k.png")
    return root


def _snap(base: Path) -> dict:
    out = {}
    for p in sorted(base.rglob("*")):
        rel = p.relative_to(base).as_posix()
        if p.is_symlink():
            out[rel] = ("link", str(p.readlink()))
        elif p.is_file():
            out[rel] = ("file", _md5b(p.read_bytes()))
        else:
            out[rel] = ("dir",)
    return out


def _run(root, apply, **kw):
    lines: list[str] = []
    kw.setdefault("plan", _plan())
    kw.setdefault("pins", PINS)
    rc = cr.run(root, apply=apply, out=lines.append, **kw)
    return rc, lines


def test_dry_run_byte_stable_and_mutation_free(tree, tmp_path):
    before = _snap(tmp_path)
    rc1, l1 = _run(tree, apply=False)
    rc2, l2 = _run(tree, apply=False)
    assert rc1 == rc2 == 0
    assert l1 == l2, "dry-run output not deterministic"
    assert _snap(tmp_path) == before
    assert not (tree / "_archive").exists()


def test_plan_glob_hitting_allowlisted_eval_input_aborts_before_any_mutation(tree, tmp_path):
    # eval/*_rendered_preP0* (archive) also matches the allowlist eval/*_input.npz.
    _w(tree, "eval/x_rendered_preP0_input.npz", b"precious")
    before = _snap(tmp_path)
    rc, lines = _run(tree, apply=True)
    assert rc == 1
    assert any("allowlist overlap" in ln and "x_rendered_preP0_input" in ln for ln in lines)
    assert _snap(tmp_path) == before


@pytest.mark.parametrize("victim", [
    "ag_" "dblgauss", "vibe_bench/IMG_6915_R2_layered.jpg", "render/current_check", "demo",
])
def test_custom_plan_targeting_allowlist_aborts(tree, tmp_path, victim):
    plan = _plan() + ((victim, "delete", None, "hostile row"),)
    before = _snap(tmp_path)
    rc, lines = _run(tree, apply=True, plan=plan)
    assert rc == 1, lines
    assert _snap(tmp_path) == before


def test_nested_symlink_inside_deleted_dir_does_not_follow(tree, tmp_path):
    outside = tmp_path / "outside"
    _w(outside, "sentinel.txt", b"keep me")
    os.symlink(outside, tree / "render" / "visual_p13" / "escape")
    os.symlink(outside, tree / "render" / "visual_p02" / "escape")
    rc, _ = _run(tree, apply=True)
    assert rc == 0
    assert (outside / "sentinel.txt").read_bytes() == b"keep me"
    assert not (tree / "render" / "visual_p13").exists()


def test_top_level_symlink_target_refused(tree, tmp_path):
    outside = tmp_path / "outside_k"
    _w(outside, "sentinel.txt", b"keep")
    os.symlink(outside, tree / "psf_model")  # plan row 10 archives psf_model
    before = _snap(tmp_path)
    rc, lines = _run(tree, apply=True)
    assert rc == 1
    assert any("symlink target refused: psf_model" in ln for ln in lines)
    assert _snap(tmp_path) == before


def test_symlinked_archive_dir_refused(tree, tmp_path):
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    os.symlink(outside, tree / "_archive")
    rc, _ = _run(tree, apply=True)
    escaped = [p for p in outside.rglob("*")]
    assert rc == 1 and escaped == []


def test_archive_group_path_is_file_aborts_before_any_mutation(tree, tmp_path):
    _w(tree, "_archive/R-highlight-guard", b"i am a file")  # blocks depth_refine
    before = _snap(tmp_path)
    try:
        rc, _ = _run(tree, apply=True)
    except OSError:
        rc = "raised"
    assert rc == 1
    assert _snap(tmp_path) == before


def test_symlinked_runs_root_operates_on_real_tree_consistently(tree, tmp_path):
    link = tmp_path / "runs_link"
    os.symlink(tree, link)
    rc, _ = _run(link, apply=True)
    assert rc == 0
    # resolved once: everything lands in the real tree, the link stays a link
    assert link.is_symlink()
    assert (tree / "_archive" / "clean_log.jsonl").is_file()
    assert (tree / "_archive" / "phase0-1" / "visual_p02" / "a.png").is_file()


def test_corrupted_g1_baseline_aborts_wrong_polarity(tree, tmp_path):
    # The pin itself is bad (not a target): guard must NOT trust it.
    _w(tree, "vibe_bench/IMG_6915_R2_layered.jpg", b"corrupted")
    before = _snap(tmp_path)
    rc, lines = _run(tree, apply=True)
    assert rc == 1
    assert any("pin md5 mismatch: vibe_bench/IMG_6915_R2_layered.jpg" in ln for ln in lines)
    assert _snap(tmp_path) == before


def test_absent_pin_reported_but_apply_proceeds(tree):
    (tree / "render/current_check/IMG_6915_occ_on.jpg").unlink()
    rc, lines = _run(tree, apply=True)
    assert rc == 0
    assert any("pin absent (not fatal): render/current_check/IMG_6915_occ_on.jpg" in ln
               for ln in lines)


def test_expect_md5_on_directory_impostor_aborts(tree, tmp_path):
    # A directory named like a vibe_bench duplicate must fail expect_md5, not be rmtree'd.
    _w(tree, "vibe_bench/IMG_6915_OFF_r2.jpg/inner.bin", b"surprise")
    before = _snap(tmp_path)
    rc, lines = _run(tree, apply=True)
    assert rc == 1
    assert any("expect_md5 mismatch: vibe_bench/IMG_6915_OFF_r2.jpg" in ln for ln in lines)
    assert _snap(tmp_path) == before


def test_rerun_apply_is_noop_and_log_complete(tree, tmp_path):
    rc, _ = _run(tree, apply=True)
    assert rc == 0
    log = tree / "_archive" / "clean_log.jsonl"
    recs = [json.loads(x) for x in log.read_text().splitlines()]
    moved = {r["old_path"]: r["new_path"] for r in recs if r["action"] == "archive"}
    assert moved == {
        "render/visual_p02": "_archive/phase0-1/visual_p02",
        "render/depth_refine": "_archive/R-highlight-guard/depth_refine",
        "eval/usaf_rendered_preP0.npz": "_archive/eval-P0/usaf_rendered_preP0.npz",
        "kernels": "_archive/shelved-corpus/kernels",
    }
    for old, new in moved.items():
        assert not (tree / old).exists() and (tree / new).exists()
    deleted = {r["old_path"] for r in recs if r["action"] == "delete"}
    assert deleted == {"extract_r2/tmp", "vibe_bench/IMG_6915_OFF_after.jpg",
                       "render/visual_p13"}
    assert all(r["new_path"] is None for r in recs if r["action"] == "delete")
    logged = {r["old_path"] for r in recs}
    assert "render/armedge_visual" not in logged and "ag_" "dblgauss" not in logged
    # second apply: nothing to do, log unchanged, tree unchanged
    snap = _snap(tmp_path)
    rc2, _ = _run(tree, apply=True)
    assert rc2 == 0
    assert _snap(tmp_path) == snap


def test_unclassified_and_case_variant_never_touched(tree, tmp_path):
    # Case-variant names of planned dirs must not be matched (fnmatchcase).
    _w(tree, "render/Visual_P13/keep.png", b"k")
    _w(tree, "KERNELS_old/keep.png", b"k")
    rc, lines = _run(tree, apply=True)
    if (tree / "render" / "visual_p13").exists():
        pytest.skip("case-insensitive collision: Visual_P13 aliased visual_p13")
    assert rc == 0
    assert (tree / "render/armedge_visual/a.png").read_bytes() == b"render/armedge_visual/a.png"
    assert (tree / "KERNELS_old/keep.png").exists()
    unc = [ln for ln in lines if ln.endswith("UNCLASSIFIED")]
    assert any(ln.startswith("KERNELS_old\t") for ln in unc)


# ---------------------------------------------------------------- make_demo


@pytest.fixture
def md(tmp_path, monkeypatch):
    if not (ROOT / "demo" / "make_demo.py").is_file():
        pytest.skip("demo/ not shipped (public release)")
    mod = _load("make_demo_adv", ROOT / "demo" / "make_demo.py")
    monkeypatch.setattr(mod, "OUT", tmp_path / "runs" / "demo")
    pin = tmp_path / "pin.jpg"
    Image.fromarray(np.full((32, 24, 3), 100, np.uint8)).save(pin, quality=95)
    monkeypatch.setattr(mod, "ON_PIN_JPG", pin)
    state = {"calls": [], "md5": {}, "rc": {}}

    def fake_run(argv, **kw):
        state["calls"].append(list(argv))
        dst = Path(argv[argv.index("--output") + 1])
        rc = state["rc"].get(dst.stem, 0)
        if rc == 0:
            dst.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.full((32, 24, 3), 120, np.uint8)).save(dst, quality=95)
        return subprocess.CompletedProcess(argv, rc, "", "err")

    def fake_md5(path):
        n = Path(path).stem
        return state["md5"].get(n, {"g1": G1_MD5, "g1_occ": ON_MD5}.get(n, "f" * 32))

    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    monkeypatch.setattr(mod, "load_image", lambda p: np.full((64, 48, 3), 0.5, np.float32))
    monkeypatch.setattr(mod, "_md5", fake_md5)
    monkeypatch.setattr(mod, "_tensor_md5", lambda: "t" * 32)
    mod._state = state
    return mod


def _stems(calls):
    return [Path(c[c.index("--output") + 1]).stem for c in calls]


def _write_manifest(tmp_path, body: str) -> Path:
    p = tmp_path / "manifest.toml"
    p.write_text(body)
    return p


G1_ROW = ('[[row]]\nname = "g1"\nimage = "IMG_6915.HEIC"\nfocus_xy = [0.40, 0.45]\n'
          'extra = []\npin = "G1_MD5"\npin_mode = "abort"\n')


def test_g1_mismatch_with_only_aborts_before_other_scene(md):
    md._state["md5"]["g1"] = "0" * 32
    assert md.main(["--only", "demo,stuffed"]) == 1
    assert _stems(md._state["calls"]) == ["g1"]
    assert not (md.OUT / "manifest.lock.json").exists()
    assert not (md.OUT / "g1_panel.jpg").exists()


def test_g1_nonzero_exit_aborts_like_non_darwin(md):
    md._state["rc"]["g1"] = 2  # what convolve's parser.error yields off-darwin
    assert md.main([]) == 1
    assert _stems(md._state["calls"]) == ["g1"]


def test_g1_occ_warn_records_max_abs_and_exits_zero(md, capsys):
    md._state["md5"]["g1_occ"] = "e" * 32
    assert md.main([]) == 0
    assert _stems(md._state["calls"]) == ["g1", "g1_occ", "stuffed", "demo"]
    rec = json.loads((md.OUT / "manifest.lock.json").read_text())["rows"]["g1_occ"]
    assert rec["pin_match"] is False and rec["max_abs_vs_pin"] == 20
    assert "WARNING g1_occ" in capsys.readouterr().out


def test_space_in_image_kept_as_one_argv_element(md):
    row = {"name": "x", "image": "IMG_6651 2.HEIC", "focus_xy": [0.5, 0.5]}
    argv = md.build_argv(row, md.OUT / "x.jpg")
    assert argv[argv.index("--image") + 1] == "iphoneImage/IMG_6651 2.HEIC"


def test_canonical_flags_imported_not_copied(md, monkeypatch):
    assert md.CANONICAL_FLAGS is _bench.CANONICAL_FLAGS
    row = {"name": "g1", "image": "IMG_6915.HEIC", "focus_xy": [0.4, 0.45]}
    argv = md.build_argv(row, md.OUT / "g1.jpg")
    i = argv.index("--image") + 2
    assert tuple(argv[i:i + len(CANONICAL_FLAGS)]) == CANONICAL_FLAGS
    monkeypatch.setattr(md, "CANONICAL_FLAGS", ("--sentinel",))
    assert "--sentinel" in md.build_argv(row, md.OUT / "g1.jpg")


@pytest.mark.parametrize("name", ["../escape", "../../runs/vibe_bench/IMG_6915_R2_layered",
                                  "/tmp/abs"])
def test_hostile_row_name_refused_before_render(md, tmp_path, monkeypatch, name):
    body = G1_ROW + f'[[row]]\nname = "{name}"\nimage = "IMG_6915.HEIC"\nfocus_xy = [0.5, 0.5]\n'
    p = _write_manifest(tmp_path, body)
    orig = md.load_manifest
    monkeypatch.setattr(md, "load_manifest", lambda path=p: orig(p))
    with pytest.raises(ValueError):
        md.main([])
    assert _stems(md._state["calls"]) == ["g1"]  # hostile row never rendered


def test_manifest_requires_g1_first(md, tmp_path):
    body = ('[[row]]\nname = "demo"\nimage = "Demo_input.HEIC"\nfocus_xy = [0.4, 0.39]\n'
            + G1_ROW)
    with pytest.raises(ValueError, match="first manifest row"):
        md.load_manifest(_write_manifest(tmp_path, body))


def test_manifest_g1_without_hard_pin_rejected(md, tmp_path):
    body = G1_ROW.replace('pin_mode = "abort"', 'pin_mode = "warn"')
    with pytest.raises(ValueError):
        md.load_manifest(_write_manifest(tmp_path, body))


def test_bogus_pin_name_rejected_before_any_render(md, tmp_path, monkeypatch):
    body = G1_ROW + ('[[row]]\nname = "bad"\nimage = "IMG_6915.HEIC"\n'
                     'focus_xy = [0.5, 0.5]\npin = "NOPE_MD5"\npin_mode = "warn"\n')
    p = _write_manifest(tmp_path, body)
    monkeypatch.setattr(md, "MANIFEST", p)
    with pytest.raises(ValueError):
        md.load_manifest(p)


def test_unknown_only_ids_exit_before_any_subprocess(md):
    with pytest.raises(SystemExit) as ei:
        md.main(["--only", "stuffed,../g1"])
    assert ei.value.code == 2
    assert md._state["calls"] == []


def test_panel_saved_without_exif(md, tmp_path):
    panel = md.make_panel(np.zeros((10, 8, 3), np.float32), np.zeros((20, 16, 3), np.uint8),
                          None, (1.0, 1.0))
    p = tmp_path / "p.jpg"
    md.save_panel(panel, p)
    with Image.open(p) as im:
        assert len(im.getexif()) == 0 and "exif" not in im.info


def test_existing_real_demo_lock_g1_matches_pin_readonly():
    lock = ROOT / "runs" / "demo" / "manifest.lock.json"
    if not lock.is_file():
        pytest.skip("no runs/demo lock file")
    rows = json.loads(lock.read_text())["rows"]
    assert rows["g1"]["md5"] == G1_MD5 and rows["g1"]["pin_match"] is True
