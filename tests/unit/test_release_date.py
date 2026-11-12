"""Release-date stamping: changelog_date.py plus the build script's wiring (no network, no builds)."""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
REL = ROOT / "scripts" / "release"
if not (REL / "changelog_date.py").is_file():
    pytest.skip("scripts/release is not shipped here", allow_module_level=True)
sys.path.insert(0, str(REL))
import changelog_date as cd  # noqa: E402

CHG = """# Changelog

## [0.4.0] — unreleased

- new thing

## [0.3.0] — 2026-11-10

- old thing

[0.4.0]: ../v0.4.0
"""


def test_unreleased_is_stamped():
    out = cd.stamp(CHG, "0.4.0", "2026-11-12")
    assert "## [0.4.0] — 2026-11-12\n" in out and "unreleased" not in out


def test_existing_date_is_replaced():
    out = cd.stamp(CHG.replace("[0.4.0] — unreleased", "[0.4.0] — 2026-11-10"), "0.4.0", "2026-11-12")
    assert "## [0.4.0] — 2026-11-12\n" in out and "2026-11-10\n\n- new" not in out


def test_hyphen_and_bare_headings_are_normalised():
    assert cd.stamp("## [1.0.0] - 2020-01-01\nx\n", "1.0.0", "2026-11-12") == "## [1.0.0] — 2026-11-12\nx\n"
    assert cd.stamp("## [1.0.0]\nx\n", "1.0.0", "2026-11-12") == "## [1.0.0] — 2026-11-12\nx\n"


def test_older_entries_unchanged():
    out = cd.stamp(CHG, "0.4.0", "2026-11-12")
    a, b = CHG.split("\n"), out.split("\n")
    assert len(a) == len(b)
    assert [i for i, (x, y) in enumerate(zip(a, b)) if x != y] == [2]  # only the top heading line
    assert "## [0.3.0] — 2026-11-10" in out


def test_wrong_top_version_refused():
    with pytest.raises(ValueError):
        cd.stamp(CHG, "0.3.0", "2026-11-12")


@pytest.mark.parametrize("bad", ["2026-13-01", "2026-02-30", "26-11-12", "2026/11/12", "today", "2026-11-12 ", "2026-1-2"])
def test_invalid_release_date_refused(bad):
    with pytest.raises(ValueError):
        cd.resolve({"RELEASE_DATE": bad})
    with pytest.raises(ValueError):
        cd.stamp(CHG, "0.4.0", bad)


def test_resolve_default_is_today_and_env_wins():
    import datetime
    assert cd.resolve({}) == datetime.date.today().isoformat()
    assert cd.resolve({"RELEASE_DATE": "2026-11-12"}) == "2026-11-12"


def test_build_script_refuses_invalid_release_date():
    env = dict(os.environ, RELEASE_DATE="2026-99-99", RELEASE_ROOT="/nonexistent-release-root")
    r = subprocess.run(["bash", str(REL / "build_release.sh"), "HEAD", "9.9.9"], cwd=ROOT, env=env,
                       capture_output=True, text=True)
    assert r.returncode != 0 and "RELEASE_DATE" in r.stderr, (r.returncode, r.stderr)


def test_commit_message_and_heading_share_one_source():
    sh = (REL / "build_release.sh").read_text()
    # the build-time message (G-msg) and the --commit check (G-date) both read the one recorded date
    assert "'release: %s (%s)\\n\\n%s\\n\\n%s\\n' \"$VERSION\" \"$(cat \"$W/release-date.txt\")\"" in sh
    assert '"release: $VERSION ($RDATE)"' in sh
    assert "date +%Y-%m-%d" not in sh  # no second clock read in the build
    assert "release-date.txt" in sh and "G-date" in sh
    assert re.search(r'changelog_date\.py" stamp "\$STAGE/CHANGELOG\.md" "\$VERSION" "\$rdate"', sh)


def test_cli_stamp_and_heading_roundtrip(tmp_path):
    p = tmp_path / "CHANGELOG.md"
    p.write_text(CHG, encoding="utf-8")
    run = lambda *a: subprocess.run([sys.executable, str(REL / "changelog_date.py"), *a],  # noqa: E731
                                    capture_output=True, text=True)
    assert run("stamp", str(p), "0.4.0", "2026-11-12").returncode == 0
    r = run("heading", str(p), "0.4.0")
    assert r.stdout.strip() == "2026-11-12"
    assert p.read_text(encoding="utf-8") == cd.stamp(CHG, "0.4.0", "2026-11-12")
    assert run("stamp", str(p), "0.4.0", "nope").returncode == 1


def test_history_gate_flags_changed_or_unreleased_old_entries():
    import importlib.util, pathlib
    spec = importlib.util.spec_from_file_location(
        "changelog_date", pathlib.Path(__file__).resolve().parents[2] / "scripts/release/changelog_date.py")
    cd = importlib.util.module_from_spec(spec); spec.loader.exec_module(cd)
    base = "## [0.3.0] — 2026-11-12\n- a\n\n[0.3.0]: x\n"
    ok = "## [0.4.0] — unreleased\n- b\n\n## [0.3.0] — 2026-11-12\n- a\n\n[0.4.0]: y\n[0.3.0]: x\n"
    assert cd.history_problems(ok, base) == []
    assert cd.history_problems(ok.replace("2026-11-12", "2026-11-10"), base)
    assert cd.history_problems(ok.replace("0.3.0] — 2026-11-12", "0.3.0] — unreleased"), base)
