"""Run 9 (section20) adversarial probes.

Covers: ``--hdr-source auto`` ImportError fallback (broken Quartz / partial
pyobjc), explicit ``macos`` one-line error, overlay quickstart vs
CANONICAL_FLAGS, staged-tree skip guards (skip iff resource absent, run when
present) and default-resolution non-regression. Fast; fake renders only.
Bugs are strict xfails.
"""
from __future__ import annotations

import dataclasses
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from summicron.render.api import RenderOptions
from tests.adversarial.test_s20_hdr_auto_adversarial import GM, LD, LUT
from tests.unit.test_hdr_auto import _fakes, _opts, _run

ROOT = Path(__file__).resolve().parents[2]

GUARDED = [
    "tests/unit/test_s16_metrics.py",
    "tests/unit/test_probe_s17_combo.py",
    "tests/unit/test_probe_s15_matte_depth.py",
    "tests/adversarial/test_s20_flip_adversarial.py",
    "tests/adversarial/test_depth_matte_s15_adversarial_r2.py",
    "tests/adversarial/test_depth_diopter_adversarial2.py",
]


def _auto_ps(tmp_path, **kw):
    # Real load_hdr_heic is NOT patched: the genuine import must fail.
    cap, ps = _fakes(**kw)
    return cap, ps + [GM, LD, LUT]


# ---- 1. auto + broken pyobjc -------------------------------------------------

@pytest.mark.parametrize("broken", [("Quartz",), ("Foundation",),
                                    ("Quartz", "Foundation")])
def test_auto_broken_import_falls_back_ldr_with_one_note(tmp_path, broken):
    """find_spec says yes but the import fails (Quartz and/or Foundation)."""
    cap, ps = _auto_ps(tmp_path)
    o = _opts(tmp_path, tonemap="ci")
    with patch.dict(sys.modules, {m: None for m in broken}):
        res, out = _run(o, ps)
    assert res.diagnostics.resolved["hdr_source"] == "none"
    assert res.mode == "ldr"
    assert cap["render"]["hdr_source"] == "none"
    assert out.count("note: hdr_source auto -> none") == 1
    assert "HDR loader import failed" in out


def test_auto_broken_import_default_tonemap_resolves_ci(tmp_path):
    """Default tonemap=auto must degrade to ci (not input-lut) on fallback."""
    cap, ps = _auto_ps(tmp_path)
    with patch.dict(sys.modules, {"Quartz": None}):
        res, _ = _run(_opts(tmp_path), ps)
    assert res.diagnostics.resolved["hdr_source"] == "none"
    assert res.diagnostics.resolved["tonemap"] == "ci"


def test_auto_broken_import_input_lut_explicit_is_clear_valueerror(tmp_path):
    cap, ps = _auto_ps(tmp_path)
    o = _opts(tmp_path, tonemap="input-lut")
    with patch.dict(sys.modules, {"Quartz": None}), \
            pytest.raises(ValueError, match="needs an HDR input"):
        _run(o, ps)


def test_auto_broken_import_shadow_match_on_is_clear_valueerror(tmp_path):
    cap, ps = _auto_ps(tmp_path)
    o = _opts(tmp_path, shadow_match_mode="on")
    with patch.dict(sys.modules, {"Quartz": None}), \
            pytest.raises(ValueError, match="needs an HDR input"):
        _run(o, ps)


def test_auto_non_import_loader_error_is_not_swallowed(tmp_path):
    """Only ImportError may fall back; a RuntimeError from the loader must
    surface (silent LDR fallback would hide real HDR load bugs)."""
    cap, ps = _fakes()
    ps += [GM, LD, LUT,
           patch("summicron.io.hdr_load_macos.load_hdr_heic",
                 side_effect=RuntimeError("boom"))]
    with pytest.raises(RuntimeError, match="boom"):
        _run(_opts(tmp_path, tonemap="ci"), ps)


def test_auto_fallback_does_not_leave_hdr_loaded(tmp_path):
    """After fallback the pipeline must take the LDR load (render sees an
    8-bit-derived image whose max <= 1)."""
    cap, ps = _auto_ps(tmp_path)
    with patch.dict(sys.modules, {"Quartz": None}):
        _run(_opts(tmp_path, tonemap="ci"), ps)
    assert cap["render"]["hdr_source"] == "none"


# ---- 2. explicit macos --------------------------------------------------------

def test_explicit_macos_broken_import_raises_actionable_importerror(tmp_path):
    cap, ps = _auto_ps(tmp_path)
    o = _opts(tmp_path, hdr_source="macos", tonemap="ci")
    with patch.dict(sys.modules, {"Quartz": None}), \
            pytest.raises(ImportError, match=r"hdr-macos"):
        _run(o, ps)


def test_explicit_macos_cli_one_line_exit2(tmp_path, capsys, monkeypatch):
    from summicron.render import convolve
    cap, ps = _auto_ps(tmp_path)
    img = _opts(tmp_path).image_path
    tens = _opts(tmp_path).tensor_path
    monkeypatch.delenv("ISUMMICRON_DEBUG", raising=False)
    argv = ["--image", str(img), "--tensor", str(tens), "--focus-xy", "0.5", "0.5",
            "--hdr-source", "macos", "--tonemap", "ci", "--bright-pass-thresh", "1.0",
            "--out-dir", str(tmp_path / "o")]
    from contextlib import ExitStack
    with ExitStack() as st, patch.dict(sys.modules, {"Quartz": None}):
        for p in ps:
            st.enter_context(p)
        rc = convolve.main(argv)
    err = capsys.readouterr().err
    assert rc == 2
    lines = [ln for ln in err.splitlines() if ln.strip()]
    assert len(lines) == 1 and lines[0].startswith("render: ")
    assert "Traceback" not in err and "hdr-macos" in lines[0]


def test_explicit_macos_cli_debug_env_reraises(tmp_path, monkeypatch):
    from contextlib import ExitStack

    from summicron.render import convolve
    cap, ps = _auto_ps(tmp_path)
    o = _opts(tmp_path)
    monkeypatch.setenv("ISUMMICRON_DEBUG", "1")
    argv = ["--image", str(o.image_path), "--tensor", str(o.tensor_path),
            "--focus-xy", "0.5", "0.5", "--hdr-source", "macos",
            "--tonemap", "ci", "--bright-pass-thresh", "1.0",
            "--out-dir", str(tmp_path / "o")]
    with ExitStack() as st, patch.dict(sys.modules, {"Quartz": None}):
        for p in ps:
            st.enter_context(p)
        with pytest.raises(ImportError):
            convolve.main(argv)


def test_auto_cli_broken_import_exit0(tmp_path, capsys, monkeypatch):
    from contextlib import ExitStack

    from summicron.render import convolve
    cap, ps = _auto_ps(tmp_path)
    o = _opts(tmp_path)
    monkeypatch.delenv("ISUMMICRON_DEBUG", raising=False)
    argv = ["--image", str(o.image_path), "--tensor", str(o.tensor_path),
            "--focus-xy", "0.5", "0.5", "--tonemap", "ci",
            "--bright-pass-thresh", "1.0", "--out-dir", str(tmp_path / "o")]
    with ExitStack() as st, patch.dict(sys.modules, {"Quartz": None}):
        for p in ps:
            st.enter_context(p)
        rc = convolve.main(argv)
    cap_out = capsys.readouterr()
    assert rc == 0
    assert cap_out.out.count("note: hdr_source auto -> none") == 1


# ---- 3. overlay quickstart vs CANONICAL_FLAGS --------------------------------

def _has_git_head() -> bool:
    # A freshly `git init`'d staged tree has .git but no HEAD.
    r = subprocess.run(["git", "rev-parse", "--verify", "-q", "HEAD"],
                       cwd=ROOT, capture_output=True, text=True)
    return r.returncode == 0


_needs_overlay = pytest.mark.skipif(
    not (ROOT / "release" / "public-overlay").is_dir(),
    reason="release/public-overlay absent (staged release tree)")


def _overlay_quickstart_cmd():
    txt = (ROOT / "release" / "public-overlay" / "README.md").read_text()
    m = re.search(r"^isummicron render .*$", txt, re.M)
    assert m, "overlay README has no `isummicron render` line"
    return m.group(0)


@_needs_overlay
def test_overlay_quickstart_has_no_depth_scale_and_parses():
    import argparse

    from summicron.render import _render_args
    cmd = _overlay_quickstart_cmd()
    assert "--depth-scale" not in cmd
    p = argparse.ArgumentParser()
    _render_args.add_render_arguments(p)
    toks = cmd.split()[2:]
    args = p.parse_args(toks)
    assert args.image
    # constructing the options must pass __post_init__ validation
    RenderOptions.from_argparse(args)


@_needs_overlay
def test_overlay_quickstart_flags_subset_of_canonical_and_agree():
    from summicron.render._bench import CANONICAL_FLAGS
    toks = _overlay_quickstart_cmd().split()
    canon = list(CANONICAL_FLAGS)
    for flag in ("--depth-source", "--downres", "--psf-norm"):
        if flag in toks and flag in canon:
            assert toks[toks.index(flag) + 1] == canon[canon.index(flag) + 1], flag
    assert "--depth-scale" not in canon or "--depth-scale" not in toks


@_needs_overlay
def test_overlay_changelog_default_look_mentions_every_flipped_default():
    txt = (ROOT / "release" / "public-overlay" / "CHANGELOG.md").read_text()
    for needle in ("--depth-guided", "--depth-mapping", "--tonemap input-lut",
                   "--hdr-source auto"):
        assert needle in txt, needle


@_needs_overlay
def test_overlay_changelog_old_look_recipe_is_valid_cli():
    """The advertised way back to the previous look must parse and validate."""
    import argparse

    from summicron.render import _render_args
    p = argparse.ArgumentParser()
    _render_args.add_render_arguments(p)
    args = p.parse_args(["--image", "x.jpg", "--depth-guided", "off",
                         "--depth-mapping", "linear", "--tonemap", "ci",
                         "--bright-pass-thresh", "1.0", "--depth-scale", "10"])
    RenderOptions.from_argparse(args)


def test_vibe_bench_readme_in_release_allowlist():
    """runs/vibe_bench/README.md documents the bench and MUST ship."""
    sh = ROOT / "scripts" / "release" / "build_release.sh"
    if not sh.exists():
        pytest.skip("build script absent")
    assert "runs/vibe_bench/README.md" in sh.read_text()


# ---- 4. staged-tree skip guards ----------------------------------------------

_pyr = ["-p", "no:xdist", "-q", "-rs", "-m", "not slow", "-p", "no:cacheprovider"]


def _pytest(cwd: Path, files, extra_env=None):
    env = {**os.environ, "PYTHONPATH": str(cwd)}
    env.update(extra_env or {})
    return subprocess.run(
        [sys.executable, "-m", "pytest", *files, *_pyr],
        cwd=cwd, env=env, capture_output=True, text=True, timeout=900)


def test_guards_do_not_skip_when_resources_present():
    """In the full dev tree no guard may report a staged-release skip."""
    if not ((ROOT / "scripts" / "s16_metrics.py").exists()
            and (ROOT / "iphoneImage").is_dir() and _has_git_head()):
        pytest.skip("not the full dev tree")
    r = _pytest(ROOT, [g for g in GUARDED
                       if g != "tests/adversarial/test_s20_flip_adversarial.py"])
    assert "staged release tree" not in r.stdout, r.stdout[-3000:]
    assert " failed" not in r.stdout.splitlines()[-1], r.stdout[-3000:]


def test_guards_skip_and_never_fail_in_staged_tree(tmp_path):
    """Archive only summicron/ + tests/ (no scripts/, iphoneImage/, demo/,
    .git): every guarded file must skip cleanly (0 failed, 0 errors) and the
    skips must carry the staged-tree reason."""
    if not _has_git_head():
        pytest.skip("not a git checkout")
    stage = tmp_path / "stg"
    stage.mkdir()
    ar = subprocess.run(
        f"git -C {ROOT} archive HEAD summicron tests pyproject.toml | tar -x -C {stage}",
        shell=True, capture_output=True, text=True)
    assert ar.returncode == 0, ar.stderr
    assert not (stage / "scripts").exists() and not (stage / "iphoneImage").exists()
    r = _pytest(stage, GUARDED)
    tail = r.stdout[-3000:] + r.stderr[-1000:]
    assert r.returncode in (0, 5), tail
    assert not re.search(r"\b\d+ (failed|error)", r.stdout), tail
    assert "staged release tree" in r.stdout, tail


def test_guard_partial_staged_tree_runs_when_resource_present(tmp_path):
    """Stage tree + only scripts/s16_metrics.py present: the s16 module must
    NOT skip with the staged-tree reason (guard keyed to the resource, not to
    a blanket flag)."""
    if not (ROOT / "scripts" / "s16_metrics.py").exists():
        pytest.skip("script absent")
    stage = tmp_path / "stg"
    stage.mkdir()
    subprocess.run(
        f"git -C {ROOT} archive HEAD summicron tests pyproject.toml scripts/s16_metrics.py"
        f" | tar -x -C {stage}", shell=True, check=True)
    r = _pytest(stage, ["tests/unit/test_s16_metrics.py"])
    assert "staged release tree" not in r.stdout, r.stdout[-2000:]


# ---- 5. default resolution non-regression ------------------------------------

def test_default_options_resolve_unchanged_on_png(tmp_path):
    cap, ps = _fakes()
    res, _ = _run(_opts(tmp_path), ps)
    r = res.diagnostics.resolved
    assert r["hdr_source"] == "none"
    assert r["tonemap"] == "ci"
    assert r["depth_guided"] in ("on", "off") and r["depth_mapping"] in ("linear", "diopter")


def test_default_with_working_loader_still_prefers_macos(tmp_path):
    hdr = np.full((64, 64, 3), 0.5, np.float32)
    hdr[0, 0] = 2.0
    cap, ps = _fakes(hdr=hdr)
    ps += [GM, LD, LUT]
    res, out = _run(_opts(tmp_path), ps)
    assert res.diagnostics.resolved["hdr_source"] == "macos"
    assert cap["load_hdr"] == 1
    assert "note: hdr_source auto -> none" not in out


def test_explicit_none_never_attempts_import(tmp_path):
    cap, ps = _auto_ps(tmp_path)
    with patch.dict(sys.modules, {"Quartz": None}):
        res, out = _run(_opts(tmp_path, hdr_source="none", tonemap="ci"), ps)
    assert res.diagnostics.resolved["hdr_source"] == "none"
    assert "HDR loader import failed" not in out


def test_render_options_dataclass_defaults_stable():
    o = RenderOptions(image_path=Path("x.png"))
    d = dataclasses.asdict(o)
    for k, v in (("hdr_source", "auto"), ("tonemap", "auto"),
                 ("depth_guided", "auto"), ("depth_mapping", "auto")):
        assert d[k] == v, k
    assert o.depth_scale == 1.0
