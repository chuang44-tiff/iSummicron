"""RUNS-PRUNE: dry-run-first cleanup of ``runs/`` (WP-A, locked spec 2026-11-08).

Usage::

    python scripts/clean_runs.py [--runs-root PATH] [--apply]

Without ``--apply`` this is a dry run: it prints the plan, the UNCLASSIFIED
list and byte totals, and changes nothing. ``--apply`` runs every pre-flight
check first and aborts (exit 1) with nothing touched on any failure. Archive is
``os.rename`` into ``runs/_archive/<group>/``. Every action is logged to
``runs/_archive/clean_log.jsonl``. Pin md5s are re-checked post-flight (exit 2
on drift). Stdlib only, plus ``summicron.render._bench`` for the pin md5s.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
from datetime import datetime, timezone
from fnmatch import fnmatchcase
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from summicron.render._bench import G1_MD5, ON_MD5  # noqa: E402

# Pre-Q-b historical renders (the removed ``--kernel-centring legacy`` path);
# literals so the old pin files stay protected.
G1_MD5_PRE_QB = "38c75dd253571ece9c9a0e3223fdb580"  # pre-Q-b historical render
ON_MD5_PRE_QB = "07f5b91bc2a6ff46f1e93ddd96d6fc2e"  # pre-Q-b historical render

# Pre-Run-6 (s20 default flip) pins: still reproducible through explicit flags
# (``CANONICAL_FLAGS_OLD``), kept as literals for the ``_delta.jpg`` files.
G1_MD5_PRE_S20 = "a106801a97a05ee7d5033e2b8d647f11"
ON_MD5_PRE_S20 = "6cebff5403f7f0fdcda109cb8bbff35a"

ON_STALE_MD5 = "2529c3272016390507836bf1098625d0"
OTF_EXPLICIT_MD5 = "9eb07890c9d5e7ac0e24acf28f97cf87"

# Top-level dirs whose immediate children are the unit of action.
CONTAINERS = ("render", "vibe_bench", "eval", "extract_r2", "extract_r3")

ALLOWLIST = (
    "ag_dblgauss",
    "vibe_bench/README.md",
    "vibe_bench/IMG_6915_R2_layered.jpg",
    "vibe_bench/IMG_6915_G1_baseline_default.jpg",
    "vibe_bench/IMG_6915_occ_ON_pin.jpg",
    "vibe_bench/IMG_6915_G1_baseline_delta.jpg",
    "vibe_bench/IMG_6915_occ_ON_pin_delta.jpg",
    "vibe_bench/IMG_6915_G1_baseline_s20.jpg",
    "vibe_bench/IMG_6915_occ_ON_pin_s20.jpg",
    "render/current_check",
    "cherry_tensor4d.npz",
    "extract_r{2,3}/{extract_cutouts.npz,extract_report.json,extract.progress.jsonl}",
    "eval/*_input.{npz,json}",
    "eval/identity_tensor_iphone5712.npz",
    "_archive",
    "demo",  # runs/demo/** — gitignored output of the tracked demo/ (WP-B)
)

# (glob, action, group, reason[, expect_md5])
PLAN = (
    ("extract_r{2,3}/tmp", "delete", None, "scratch .npy shards, zero refs"),
    ("vibe_bench/IMG_6915_OFF_{after,after2,r2,r3}.jpg", "delete", None,
     "byte-identical duplicate of G1 (legacy)", G1_MD5_PRE_QB),
    ("vibe_bench/IMG_6915_ON{,2,_r2,_r2b}.jpg", "delete", None,
     "stale pre-cc880ac intermediate", ON_STALE_MD5),
    ("vibe_bench/IMG_6915_G1_baseline_explicit.jpg", "delete", None,
     "OTF-tensor render, not a pin", OTF_EXPLICIT_MD5),
    ("render/{archive_preP0,visual_p13,visual_p15,visual_10x_depth,visual_log_10x,"
     "visual_log_psf,visual_logpsf}", "delete", None,
     "April era, cited only as delete candidates"),
    ("render/visual_p{02,10,11,12,14}", "archive", "phase0-1", "cited in doc/archive"),
    ("render/{bloom_char,bloom_spike,tone_anchor,tone_sat,headroom_proof,contrast_diag,"
     "contrast_finalize,washout_fanout,sat_recover,brightness_probe,depth_refine,"
     "depth_refine_v2,full_demo,final_renders,headroom_spike,brightstream_spike,joint_spike}",
     "archive", "R-highlight-guard", "negative-result evidence (findings doc)"),
    ("render/probe_occlusion", "archive", "R-occlusion-layering", "cited by section3 docs"),
    ("eval/{*_rendered_preP0*,*_rendered_postP0_*,*_identity_ceiling*,probe_*_baselines.json}",
     "archive", "eval-P0", "P0-era eval outputs"),
    ("{day1_full_r2.*,day1_full_r3.*,corpus_inventory.*,vibe_check_*.png,day2_visual_probe.png,"
     "kernels,psf_model}", "archive", "shelved-corpus", "shelved pipeline outputs"),
)

PINS = {
    "vibe_bench/IMG_6915_R2_layered.jpg": G1_MD5,
    "vibe_bench/IMG_6915_G1_baseline_delta.jpg": G1_MD5_PRE_S20,
    "vibe_bench/IMG_6915_occ_ON_pin_delta.jpg": ON_MD5_PRE_S20,
    "vibe_bench/IMG_6915_G1_baseline_s20.jpg": G1_MD5,
    "vibe_bench/IMG_6915_occ_ON_pin_s20.jpg": ON_MD5,
    "vibe_bench/IMG_6915_G1_baseline_default.jpg": G1_MD5_PRE_QB,
    "vibe_bench/IMG_6915_occ_ON_pin.jpg": ON_MD5_PRE_QB,
    "render/current_check/IMG_6915_occ_on.jpg": ON_MD5_PRE_QB,
}

_BRACE = re.compile(r"\{([^{}]*)\}")


def expand(pattern: str) -> list[str]:
    """Expand shell-style ``{a,b}`` braces (fnmatch does not)."""
    m = _BRACE.search(pattern)
    if not m:
        return [pattern]
    out: list[str] = []
    for alt in m.group(1).split(","):
        out.extend(expand(pattern[: m.start()] + alt + pattern[m.end():]))
    return out


def scan(root: Path) -> list[str]:
    """Relative paths of the action units (immediate children; see CONTAINERS)."""
    items: list[str] = []
    for name in sorted(os.listdir(root)):
        if name == ".DS_Store" or name == "_archive":
            continue
        p = root / name
        if name in CONTAINERS and p.is_dir() and not p.is_symlink():
            items.extend(f"{name}/{c}" for c in sorted(os.listdir(p)) if c != ".DS_Store")
        else:
            items.append(name)
    return items


def size_of(p: Path) -> int:
    st = os.lstat(p)
    if not os.path.isdir(p) or os.path.islink(p):
        return st.st_size
    total = 0
    for dp, dns, fns in os.walk(p, followlinks=False):
        for n in fns + dns:
            total += os.lstat(os.path.join(dp, n)).st_size
    return total


def md5_of(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pin_state(root: Path, pins: dict) -> dict:
    return {rel: (md5_of(root / rel) if (root / rel).is_file() else None) for rel in pins}


def run(runs_root, apply: bool = False, plan=PLAN, allowlist=ALLOWLIST, pins=None,
        out=print) -> int:
    pins = PINS if pins is None else pins
    root = Path(os.path.realpath(runs_root))
    items = scan(root)
    allow_pats = [p for a in allowlist for p in expand(a)]
    allowed = [i for i in items if any(fnmatchcase(i, a) or i.startswith(a + "/") for a in allow_pats)]
    allowed_abs = [Path(os.path.realpath(root / a)) for a in allowed] + [
        Path(os.path.realpath(root / a)) for a in allow_pats if not any(c in a for c in "*?[")]

    targets: list[dict] = []
    claimed: set[str] = set()
    errors: list[str] = []
    for row in plan:
        glob, action, group, reason = row[:4]
        expect = row[4] if len(row) > 4 else None
        pats = expand(glob)
        hits = [i for i in items if i not in claimed and any(fnmatchcase(i, p) for p in pats)]
        out(f"# plan row {glob!r}: {len(hits)} match(es)")
        for rel in hits:
            claimed.add(rel)
            targets.append(dict(rel=rel, action=action, group=group, reason=reason,
                                expect=expect, bytes=size_of(root / rel)))

    # Pre-flight checks (always evaluated; only fatal under --apply).
    archive = root / "_archive"
    if archive.is_symlink():
        errors.append("archive dir is a symlink: _archive")
    elif os.path.lexists(archive):
        if not archive.is_dir():
            errors.append("archive path is not a directory: _archive")
        elif not Path(os.path.realpath(archive)).is_relative_to(root):
            errors.append("archive dir escapes runs root: _archive")
    log_path = archive / "clean_log.jsonl"
    if log_path.is_symlink() or (os.path.lexists(log_path) and not log_path.is_file()):
        errors.append("archive log is not a regular file: _archive/clean_log.jsonl")
    dests: set[Path] = set()
    bad_groups: set[str] = set()
    for t in targets:
        p = root / t["rel"]
        if p.is_symlink():
            errors.append(f"symlink target refused: {t['rel']}")
            continue
        rp = Path(os.path.realpath(p))
        if not rp.is_relative_to(root):
            errors.append(f"target escapes runs root: {t['rel']}")
        for a in allowed_abs:
            if rp.is_relative_to(a) or a.is_relative_to(rp):
                errors.append(f"allowlist overlap: {t['rel']} <-> {a.relative_to(root)}")
        if t["expect"] and (not p.is_file() or md5_of(p) != t["expect"]):
            errors.append(f"expect_md5 mismatch: {t['rel']}")
        if t["action"] == "archive":
            gdir = archive / t["group"]
            d = gdir / p.name
            t["dest"] = d
            if t["group"] not in bad_groups and os.path.lexists(gdir) and (
                    gdir.is_symlink() or not gdir.is_dir()):
                bad_groups.add(t["group"])
                errors.append(f"archive group path is not a real directory: "
                              f"{gdir.relative_to(root)}")
            if os.path.lexists(d) or d in dests:
                errors.append(f"archive destination exists: {d.relative_to(root)}")
            dests.add(d)
    pre = pin_state(root, pins)
    for rel, got in pre.items():
        if got is None:
            out(f"# pin absent (not fatal): {rel}")
        elif got != pins[rel]:
            errors.append(f"pin md5 mismatch: {rel} = {got} (want {pins[rel]})")

    claimed_set = set(claimed)
    unclassified = [i for i in items if i not in claimed_set and i not in allowed]

    for t in targets:
        out(f"{t['rel']}\t{t['bytes']}\t{t['action']}\t{t['group'] or '-'}\t{t['reason']}")
    out("# UNCLASSIFIED (reported, never touched):")
    unc_bytes = 0
    for rel in unclassified:
        b = size_of(root / rel)
        unc_bytes += b
        out(f"{rel}\t{b}\tUNCLASSIFIED")
    del_b = sum(t["bytes"] for t in targets if t["action"] == "delete")
    arc_b = sum(t["bytes"] for t in targets if t["action"] == "archive")
    out(f"# TOTALS delete={del_b} archive={arc_b} unclassified={unc_bytes}")
    for e in errors:
        out(f"# PREFLIGHT FAIL: {e}")

    if not apply:
        out("# dry run: nothing changed")
        return 0
    if errors:
        out("# ABORT: pre-flight failed, nothing touched")
        return 1

    log = log_path
    log.parent.mkdir(parents=True, exist_ok=True)
    ordered = [t for t in targets if t["action"] == "archive"] + [
        t for t in targets if t["action"] == "delete"]
    with open(log, "a") as fh:
        for t in ordered:
            p = root / t["rel"]
            new = None
            if t["action"] == "archive":
                t["dest"].parent.mkdir(parents=True, exist_ok=True)
                os.rename(p, t["dest"])
                new = str(t["dest"].relative_to(root))
            elif p.is_dir():
                shutil.rmtree(p)
            else:
                p.unlink()
            fh.write(json.dumps(dict(action=t["action"], old_path=t["rel"], new_path=new,
                                     bytes=t["bytes"], group=t["group"],
                                     ts=datetime.now(timezone.utc).isoformat())) + "\n")
    post = pin_state(root, pins)
    if post != pre:
        out(f"# POST-FLIGHT FAIL: pin md5s changed {pre} -> {post}")
        return 2
    out(f"# applied {len(ordered)} action(s); log: _archive/clean_log.jsonl")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs-root", default=str(REPO / "runs"))
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args(argv)
    return run(a.runs_root, apply=a.apply)


if __name__ == "__main__":
    sys.exit(main())
