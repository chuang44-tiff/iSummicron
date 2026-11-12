"""Adversarial probes for the release docs, notes and gallery tooling (dev-only).

Each case mutates a COPY (tmp_path) of the overlay or notes, or feeds a refused input, and
asserts that the corresponding check fails. Nothing in the worktree is modified. Skipped in the
public tree, where scripts/release is absent.
"""
from __future__ import annotations

import hashlib
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
import test_public_docs as tpd  # noqa: E402

FAKE_HEX = "ab" * 16  # an md5-shaped run, built at runtime so this file carries no literal one
def _current_version() -> str:
    """The version being prepared: the top entry of the overlay CHANGELOG (never hard-coded)."""
    m = re.search(r"^## \[([0-9][^\]]*)\]", (OVERLAY / "CHANGELOG.md").read_text(), re.M)
    return m.group(1)


CUR = _current_version()

FAKE_SUMS = "\n".join(f"{'cd' * 32}  {n}" for n in rn.assets(CUR)[:-1]) + "\n"


@pytest.fixture
def ov(tmp_path):
    cp = tmp_path / "ov"
    shutil.copytree(OVERLAY, cp)
    return cp


def _append(p: Path, text: str) -> Path:
    p.write_text(p.read_text() + "\n" + text + "\n")
    return p


def _fails(fn, *args) -> bool:
    try:
        fn(*args)
    except (AssertionError, SystemExit, ValueError, TypeError, pytest.fail.Exception):
        return True
    return False


# ------------------------------------------------------------- control: the copy passes
@pytest.mark.parametrize("name", ["README.md", "docs/usage.md", "docs/faq.md", "docs/troubleshooting.md"])
def test_control_unmutated_copy_passes(ov, name):
    p = ov / name
    for fn in (tpd.test_t1_commands, tpd.test_t3_links, tpd.test_t4_privacy_lite):
        fn(p)
    if name != "CHANGELOG.md":
        tpd.test_t2_inline_flags(p)


# ------------------------------------------------------------- doc mutations
def test_removed_flag_in_usage_fails(ov):
    p = _append(ov / "docs" / "usage.md", "Pass `--kernel-centring off` for the old look.")
    assert _fails(tpd.test_t2_inline_flags, p)


def test_removed_flag_in_fenced_command_fails(ov):
    p = _append(ov / "docs" / "usage.md", "```bash\nisummicron render --image a.heic --dof-m 2\n```")
    assert _fails(tpd.test_t1_commands, p)


def test_bad_choice_and_range_in_fenced_command_fail(ov):
    p = _append(ov / "docs" / "faq.md", "```bash\nisummicron render --image a.heic --hdr-source linux\n```")
    assert _fails(tpd.test_t1_commands, p)
    q = _append(ov / "docs" / "usage.md", "```bash\nisummicron render --image a.heic --focus-xy 1.5 0.2\n```")
    assert _fails(tpd.test_t1_commands, q)


def test_dev_doc_link_fails(ov):
    p = _append(ov / "docs" / "usage.md", "See doc/plan/x.md for details.")
    assert _fails(tpd.test_t4_privacy_lite, p)


def test_wrong_version_release_url_fails(ov):
    url = tpd.BASE + "/releases/download/v0.2.9/Demo_input.HEIC"
    p = _append(ov / "README.md", f"Get it at {url} today.")
    assert _fails(tpd.test_t3_links, p)


def test_unknown_asset_url_fails(ov):
    url = tpd.BASE + "/releases/download/v{{VERSION}}/secret.HEIC"
    p = _append(ov / "README.md", f"Get it at {url} today.")
    assert _fails(tpd.test_t3_links, p)


def test_md5_in_doc_fails(ov):
    p = _append(ov / "docs" / "faq.md", f"The render md5 is {FAKE_HEX}.")
    assert _fails(tpd.test_t4_privacy_lite, p)


def test_unknown_command_head_fails(ov):
    p = _append(ov / "docs" / "usage.md", "```bash\nrm -rf x\n```")
    assert _fails(tpd.test_t1_commands, p)


def test_broken_anchor_fails(ov):
    p = _append(ov / "docs" / "faq.md", "See [the install](../README.md#no-such-heading).")
    assert _fails(tpd.test_t3_links, p)


def test_broken_relative_link_fails(ov):
    p = _append(ov / "docs" / "faq.md", "See [the guide](guide-missing.md).")
    assert _fails(tpd.test_t3_links, p)


def test_python_api_unknown_field_fails(ov):
    p = ov / "docs" / "python-api.md"
    p.write_text(p.read_text().replace("downres=2048", "downresx=2048"))
    assert _fails(tpd.test_t1_commands, p)


def test_private_repo_name_fails(ov):
    p = _append(ov / "docs" / "faq.md", "Source: " + "iSummicron" + "-dev.")
    assert _fails(tpd.test_t4_privacy_lite, p)


# ------------------------------------------------------------- notes mutations
def _render(src: str) -> str:
    return rn.render(src, CUR, (OVERLAY / "CHANGELOG.md").read_text(),
                     (REL / "release-notes-disclosure.md").read_text(), FAKE_SUMS)


def test_unknown_placeholder_in_notes_refused():
    src = (NOTES / f"{CUR}.md").read_text().replace("{{VERSION}}", "{{VERSIONS}}", 1)
    with pytest.raises(ValueError):
        _render(src)


def test_md5_in_notes_fails_the_gate(tmp_path):
    src = (NOTES / f"{CUR}.md").read_text().replace("## Highlights", f"Build {FAKE_HEX}.\n\n## Highlights", 1)
    out = _render(src)
    p = tmp_path / "notes.md"
    p.write_text(out)
    rep = pg.Report()
    pg.check_commit_msg(rep, p, pg.build_token_patterns(ROOT))
    assert rep.fails


def test_wrong_version_url_in_notes_refused():
    src = (NOTES / f"{CUR}.md").read_text().replace("v{{VERSION}}/Demo_input.HEIC", "v0.2.9/Demo_input.HEIC", 1)
    with pytest.raises(ValueError):
        _render(src)


def test_missing_heading_in_notes_refused():
    src = (NOTES / f"{CUR}.md").read_text().replace("## Known limitations", "## Limits", 1)
    with pytest.raises(ValueError):
        _render(src)


# ------------------------------------------------------------- generator drift
def test_generator_refuses_an_undescribed_flag(monkeypatch, tmp_path):
    from summicron.render import _render_args

    real = _render_args.add_render_arguments

    def extra(parser):
        real(parser)
        parser.add_argument("--fake-flag", default=None)
        return parser

    monkeypatch.setattr(_render_args, "add_render_arguments", extra)
    gen = tpd._gen()
    with pytest.raises(SystemExit) as e:
        gen.main(["--check", str(OVERLAY / "docs" / "cli-reference.md")])
    assert "fake_flag" in str(e.value.code)


def test_generator_check_detects_hand_edit(tmp_path):
    p = tmp_path / "cli-reference.md"
    p.write_text((OVERLAY / "docs" / "cli-reference.md").read_text().replace("Default: `2048`", "x") + "edited\n")
    assert tpd._gen().main(["--check", str(p)]) == 1


# ------------------------------------------------------------- make_gallery refusals (exit before render)
def _gallery(tmp_path: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-B", str(REL / "make_gallery.py"), *args],
                          capture_output=True, text=True, cwd=tmp_path)


def _assets(tmp_path: Path, listed: bool) -> Path:
    a = tmp_path / "assets"
    a.mkdir()
    heic = a / "Demo_input.HEIC"
    heic.write_bytes(b"not really a heic, unstripped stand-in")
    digest = hashlib.md5(heic.read_bytes()).hexdigest() if listed else "0" * 32
    (a / "MANIFEST.txt").write_text(f"Demo_input.HEIC {digest}\npost-checks (i)-(v): all passed\n")
    return a


def test_gallery_refuses_heic_not_in_manifest(tmp_path):
    if shutil.which("exiftool") is None:
        pytest.skip("exiftool not installed")
    a = _assets(tmp_path, listed=False)
    r = _gallery(tmp_path, "--stage", str(ROOT), "--assets", str(a), "--version", CUR, "--preview")
    assert r.returncode == 2 and "MANIFEST" in r.stderr, r.stderr
    assert not list(a.glob("*.jpg"))


def test_gallery_refuses_iphoneimage_path(tmp_path):
    d = tmp_path / "iphoneImage"
    d.mkdir()
    r = _gallery(tmp_path, "--stage", str(ROOT), "--assets", str(d), "--version", CUR, "--preview")
    assert r.returncode == 2 and "iphoneImage" in r.stderr, r.stderr


def test_gallery_refuses_assets_inside_a_git_tree(tmp_path):
    d = tmp_path / "repo"
    d.mkdir()
    subprocess.run(["git", "init", "-q", str(d)], check=True)
    r = _gallery(tmp_path, "--stage", str(ROOT), "--assets", str(d), "--version", CUR, "--preview")
    assert r.returncode == 2 and "git work tree" in r.stderr, r.stderr


def test_gallery_check_i_sees_injected_metadata(tmp_path):
    """Check (i)'s row filter must report GPS/time/serial tags injected into a clean JPEG."""
    if shutil.which("exiftool") is None:
        pytest.skip("exiftool not installed")
    from PIL import Image

    sys.path.insert(0, str(REL))
    import make_gallery as mg

    f = tmp_path / "g.jpg"
    Image.new("RGB", (8, 8)).save(f, "JPEG", quality=95)
    subprocess.run(["exiftool", "-q", "-overwrite_original", "-all=", str(f)], check=True)
    assert mg.exif_rows(f) == []
    subprocess.run(["exiftool", "-q", "-overwrite_original", "-GPSLatitude=1", "-GPSLatitudeRef=N",
                    "-DateTimeOriginal=2026:01:01 00:00:00", "-SerialNumber=X1", str(f)], check=True)
    rows = " ".join(mg.exif_rows(f))
    assert "GPSLatitude" in rows and "DateTimeOriginal" in rows and "SerialNumber" in rows
