"""Release version rule: X.Y.Z or X.Y.Z.N (digits only), shared by build, publish, the gates and the notes."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
REL = ROOT / "scripts" / "release"
if not (REL / "release_checks.py").is_file():
    pytest.skip("scripts/release is not shipped here", allow_module_level=True)
sys.path.insert(0, str(REL))
import privacy_gate as pg  # noqa: E402
import release_checks as rc  # noqa: E402
import render_notes as rn  # noqa: E402

GOOD = ["0.5.2", "0.5.2.1", "0.5.2.10", "10.20.30"]
BAD = ["0.5.2.1a", "0.5.2.1.1", "0.5.2-1", "0.5", "0.5.2.", "0.5.2.1\n", "v0.5.2.1", "0.5.2.1.dev0", "0.5.2rc1",
       "0.5.2.-1", "0.5.2.１", "", None]


@pytest.mark.parametrize("v", GOOD)
def test_good_versions_parse(v):
    assert rc.parse_version(v) is not None


@pytest.mark.parametrize("v", BAD)
def test_bad_versions_rejected(v):
    assert rc.parse_version(v) is None


def test_ordering():
    p = rc.parse_version
    assert p("0.5.2") < p("0.5.2.1") < p("0.5.3")
    assert p("0.5.2.1") < p("0.5.2.2") < p("0.5.2.10")
    assert p("0.5.2") == p("0.5.2.0")


def test_lens_layout_rule_uses_first_three_parts():
    assert rc.version_needs_lens_layout("0.5.0") and rc.version_needs_lens_layout("0.5.2.1")
    assert rc.version_needs_lens_layout("0.5.0.3")
    assert not rc.version_needs_lens_layout("0.4.9.9")
    assert not rc.version_needs_lens_layout("0.5.2.1a")


def _layout(v):
    return subprocess.run([sys.executable, str(REL / "release_checks.py"), "layout", "--version", v],
                          capture_output=True, text=True)


def test_layout_cli():
    ok = _layout("0.5.2.1")
    assert ok.returncode == 0 and ok.stdout.strip() == "1"
    assert _layout("0.4.0").stdout.strip() == "0"
    for v in ("0.5.2.1a", "0.5.2.1.1", "0.5.2-1"):
        r = _layout(v)
        assert r.returncode == 2 and "X.Y.Z" in r.stderr


def _publish(v):
    # an unknown option stops right after the version check, before anything touches disk or network
    return subprocess.run(["bash", str(REL / "publish_release.sh"), v, "--bogus"], capture_output=True, text=True)


def test_publish_version_regex():
    for v in ("0.5.2", "0.5.2.1"):
        r = _publish(v)
        assert r.returncode == 2 and "unknown option" in r.stderr, (v, r.stderr)
    for v in ("0.5.2.1a", "0.5.2.1.1", "0.5.2-1", "0.5", "v0.5.2.1"):
        r = _publish(v)
        assert r.returncode == 2 and "bad VERSION" in r.stderr, (v, r.stderr)


def test_notes_assets_for_four_part_version():
    names = rn.assets("0.5.2.1")
    assert "isummicron-0.5.2.1-py3-none-any.whl" in names and "isummicron-0.5.2.1.tar.gz" in names
    assert rn.PAGE in names and len(names) == len(rn.assets("0.5.2"))
    assert rn.extras("0.4.9.1") == []


def _scan(text):
    rep = pg.Report()
    pg.scan_text(rep, "t", text, [])
    return rep.fails


def test_privacy_gate_accepts_four_part_version_but_not_addresses():
    ok = ("## [0.5.2.1] 2026-11-15\nhttps://github.com/chuang44-tiff/iSummicron/compare/v0.5.2...v0.5.2.1\n"
          "isummicron-0.5.2.1-py3-none-any.whl and v0.5.2.1\n")
    assert not [f for f in _scan(ok) if "ipv4" in f]
    # built at run time: a literal address would trip the gate on this very file
    for addr in (".".join(("192", "168", "1", "20")), ".".join(("10", "0", "0", "5")), ".".join(("1", "0", "5", "2")),
                 ".".join(("127", "0", "0", "1")), ".".join(("8", "8", "8", "8"))):
        assert any("ipv4" in f for f in _scan(f"host {addr}\n")), addr


def test_privacy_gate_url_allowlist_has_only_the_exact_mlx_page():
    def urls(t):
        return [f for f in _scan(t) if "url-not-allowlisted" in f]

    base = "https" + "://github.com/ml-explore/"  # split so this file holds no URL for the gate to judge
    assert not urls(f"see {base}mlx) here\n")
    assert urls(f"see {base}mlx-evil here\n")
    assert urls(f"see {base}other here\n")
