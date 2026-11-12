"""Dev-only release-docs checks (skipped in the public tree, where scripts/release is absent).

Covers the release notes (render + privacy gate), the privacy gate over every overlay .md,
and the consistency of the asset list between the notes renderer, the docs and the publisher.
Spec: doc/plan/phase2/section30_release-build/docs-spec.md (D6).
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
REL = ROOT / "scripts" / "release"
OVERLAY = ROOT / "release" / "public-overlay"
NOTES = ROOT / "release" / "notes"
if not (REL / "render_notes.py").is_file() or not OVERLAY.is_dir():
    pytest.skip("scripts/release or the overlay is not present (public tree)", allow_module_level=True)
sys.path.insert(0, str(REL))
sys.path.insert(0, str(ROOT / "tests" / "unit"))
import privacy_gate as pg  # noqa: E402
import render_notes as rn  # noqa: E402

def _current_version() -> str:
    """The version being prepared: the top entry of the overlay CHANGELOG (never hard-coded)."""
    m = re.search(r"^## \[([0-9][^\]]*)\]", (OVERLAY / "CHANGELOG.md").read_text(), re.M)
    return m.group(1)


CUR = _current_version()

FAKE_SUMS = "\n".join(f"{'ab' * 32}  {n}" for n in rn.assets(CUR)[:-1]) + "\n"
OV_MD = [OVERLAY / "README.md", OVERLAY / "CHANGELOG.md", *sorted((OVERLAY / "docs").glob("*.md"))]
DEV_ONLY_RE = [r"--kernel-centring", r"\bpins?\b", r"\bG1\b", r"\bdeploy\b", r"\bsection ?\d", r"\bround [A-Z0-9]",
               r"\bQ-[a-e]\b", r"\bR-[a-z]"]


def _render(src: Path, version: str = CUR) -> str:
    return rn.render(src.read_text(), version, (OVERLAY / "CHANGELOG.md").read_text(),
                     (REL / "release-notes-disclosure.md").read_text(), FAKE_SUMS)


def test_a_notes_render_and_pass_the_privacy_gate(tmp_path):
    out = _render(NOTES / f"{CUR}.md")
    p = tmp_path / "notes.md"
    p.write_text(out)
    tokens = pg.build_token_patterns(ROOT)
    rep = pg.Report()
    pg.check_commit_msg(rep, p, tokens)
    pg.scan_text(rep, "notes", out, tokens)
    pg.scan_wide(rep, "notes", out, tokens, "P5")
    assert not rep.fails, rep.fails
    src = (NOTES / f"{CUR}.md").read_text()
    for rx in DEV_ONLY_RE:
        assert not re.search(rx, src), rx
    assert len(src.splitlines()) <= 140


def test_a_template_is_complete():
    t = (NOTES / "TEMPLATE.md").read_text()
    for h in rn.HEADINGS:
        assert h in t
    for ph in ("{{VERSION}}", "{{CHANGELOG_SECTION}}", "{{DISCLOSURE}}", "{{SHA256SUMS}}"):
        assert ph in t


@pytest.mark.parametrize("path", OV_MD, ids=lambda p: p.name)
def test_b_overlay_docs_pass_the_gate_scan(path):
    text = path.read_text()
    rep = pg.Report()
    pg.scan_text(rep, path.name, text, pg.build_token_patterns(ROOT))
    assert not rep.fails, rep.fails
    assert not list(pg.DEV_LINK_RE.finditer(text))


def test_c_url_allowlist_is_a_subset_of_the_gate():
    import test_public_docs as tpd

    assert set(tpd.URL_ALLOW_DOCS) <= set(pg.URL_ALLOW)


def test_d_asset_lists_agree():
    import test_public_docs as tpd

    assert sorted(tpd.ASSETS) == sorted(rn.assets(tpd.VER if tpd.STAMPED else "{{VERSION}}"))
    r = subprocess.run([sys.executable, str(REL / "render_notes.py"), "--list-assets", "1.2.3"],
                       capture_output=True, text=True)
    assert r.stdout.split() == rn.assets("1.2.3")
    pub = (REL / "publish_release.sh").read_text()
    assert "render_notes.py\" --list-assets" in pub


def test_e_every_notes_file_has_a_changelog_section():
    cl = (OVERLAY / "CHANGELOG.md").read_text()
    for p in NOTES.glob("*.md"):
        if p.name != "TEMPLATE.md":
            assert f"## [{p.stem}]" in cl, p.name


def test_f_stamp_simulation_leaves_no_placeholder(tmp_path):
    cp = tmp_path / "ov"
    shutil.copytree(OVERLAY, cp)
    for p in [cp / "README.md", *(cp / "docs").glob("*.md")]:
        p.write_text(p.read_text().replace("{{VERSION}}", CUR))
    for p in cp.rglob("*.md"):
        assert not re.search(r"\{\{[A-Z0-9_]+\}\}", p.read_text()), p
