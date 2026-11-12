"""0.5.0 purge guard: the shipped surface offers and depends on no legacy PSF source.

A pathlib walk from the repo root (no git), so it also works in a ``git archive`` tree and
in public CI. The guarded tokens are built by concatenation so this file never matches
itself. ``CHANGELOG.md`` is the only exempt file (release history, plus the migration note
that names the removed file).
"""
from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

TOKENS = (
    "code" + "v",
    "code" + " v",
    "code" + "_v",
    "code" + "-v",
    "ag_" + "dblgauss",
    "35" + "f2",
)
NAMES = ("Mand" + "ler", "Le" + "itz")  # §15: patent-only wording outside the NOTICE

SCAN_ROOTS = (
    "summicron", "tests", "pyproject.toml", "scripts/build_psf",
    "scripts/q_paired_probe.py", "scripts/clean_runs.py", "runs/vibe_bench/README.md",
    "release/public-overlay/README.md", "release/public-overlay/NOTICE",
    "release/public-overlay/environment-optiland.yml", "README.md", "NOTICE",
)
NAME_ROOTS = ("summicron", "scripts/build_psf", "README.md", "release/public-overlay/README.md")
SKIP_DIRS = {"__pycache__", ".pytest_cache", ".git"}
BINARY_SUFFIXES = {".npz", ".npy", ".pyc", ".jpg", ".jpeg", ".png", ".heic", ".webp", ".gif", ".ico"}


def _files(rels):
    for rel in rels:
        p = ROOT / rel
        if p.is_file():
            yield p
        elif p.is_dir():
            for f in sorted(p.rglob("*")):
                if f.is_file() and not (set(f.relative_to(ROOT).parts) & SKIP_DIRS):
                    yield f


def _scan(rels, tokens):
    hits = []
    for f in _files(rels):
        if f.name == "CHANGELOG.md":
            continue  # release history (names the removed file in the migration note)
        rel = f.relative_to(ROOT).as_posix()
        low_name = rel.lower()
        hits += [f"{rel} (file name): {t}" for t in tokens if t.lower() in low_name]
        if f.suffix.lower() in BINARY_SUFFIXES:
            continue
        try:
            text = f.read_text(encoding="utf-8").lower()
        except (UnicodeDecodeError, OSError):
            continue
        hits += [f"{rel}: {t}" for t in tokens if t.lower() in text]
    return hits


def test_scan_roots_exist_and_walk_is_nonempty():
    assert (ROOT / "summicron").is_dir() and (ROOT / "tests").is_dir()
    assert sum(1 for _ in _files(("summicron",))) > 20


def test_scanner_polarity_control(tmp_path):
    """The scanner must FIRE on a planted token (wrong-polarity control)."""
    (tmp_path / "x.md").write_text("see " + TOKENS[0].upper() + " output")
    (tmp_path / "CHANGELOG.md").write_text(TOKENS[0])
    # reuse the walker's logic on a foreign root
    global ROOT
    old = ROOT
    ROOT = tmp_path
    try:
        hits = _scan(("x.md", "CHANGELOG.md"), TOKENS)
    finally:
        ROOT = old
    assert hits == [f"x.md: {TOKENS[0]}"], hits


def test_no_legacy_psf_tokens_in_shipped_surface():
    hits = _scan(SCAN_ROOTS, TOKENS)
    assert hits == [], "\n".join(hits)


def test_new_public_text_has_no_legacy_tokens():
    """The top ([0.5.0]) CHANGELOG section and the release notes are new public text: no legacy
    tokens and no patent-holder names. Older sections stay as history (exempt above)."""
    src = ROOT / "release/public-overlay/CHANGELOG.md"
    if not src.is_file():
        src = ROOT / "CHANGELOG.md"  # staged public tree
    cl = src.read_text(encoding="utf-8")
    if "## [0.5.0]" not in cl:
        pytest.skip("no [0.5.0] section in this tree")
    start = cl.index("## [0.5.0]")
    end = cl.find("\n## [", start + 1)
    end = len(cl) if end < 0 else end
    top = cl[start:end].lower()
    texts = [("CHANGELOG [0.5.0]", top)]
    nf = ROOT / "release/notes/0.5.0.md"
    if nf.is_file():  # release notes are not shipped in the public tree
        texts.append(("notes/0.5.0.md", nf.read_text(encoding="utf-8").lower()))
    hits = [f"{name}: {t}" for name, text in texts for t in TOKENS + NAMES if t.lower() in text]
    assert hits == [], "\n".join(hits)


def test_no_patent_holder_names_outside_notice():
    hits = _scan(NAME_ROOTS, NAMES)
    assert hits == [], "\n".join(hits)


@pytest.mark.parametrize("mod", [
    "summicron.render." + "_family",
    "summicron.psf_model." + "synth_family",
    "summicron.psf_model." + "downscale_tensor",
    "summicron.psf_model." + "code" + "v_to_render",
])
def test_removed_modules_not_importable(mod):
    assert importlib.util.find_spec(mod) is None
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(mod)


def test_artifacts_dir_holds_only_the_two_optiland_tensors():
    d = ROOT / "summicron" / "_artifacts"
    npz = sorted(p.name for p in d.glob("*.npz"))
    assert npz == ["psf_lens35_us3006249_fu0.4.npz", "psf_lens50_doublegauss_fu0.4.npz"], npz
    assert not any(p.suffix in (".npy", ".json") for p in d.iterdir())


def test_cli_help_offers_no_legacy_source(capsys):
    from summicron.render import _render_args

    import argparse

    p = argparse.ArgumentParser()
    _render_args.add_render_arguments(p)
    help_text = p.format_help().lower()
    assert ("kernel" + "-family") not in help_text
    assert ("lens" + "-aperture") not in help_text
    for t in TOKENS:
        assert t.lower() not in help_text, t
    lens = next(a for a in p._actions if a.dest == "lens")
    assert list(lens.choices) == ["35", "50"]
