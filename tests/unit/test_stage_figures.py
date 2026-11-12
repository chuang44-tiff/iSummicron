"""Dev-only checks for the 0.5.0 figure staging rule (skipped in the public tree, where scripts/release is absent).

stage_figures.sh copies the README/docs figures into the release assets dir, strips every metadata row
with exiftool, and writes FIGURES.txt; publish_release.sh re-verifies it. The asset list comes from
render_notes.py and is gated by version (no figures below 0.5.0).
"""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
REL = ROOT / "scripts" / "release"
if not (REL / "stage_figures.sh").is_file():
    pytest.skip("scripts/release is not present (public tree)", allow_module_level=True)
if sys.platform != "darwin" or shutil.which("exiftool") is None:
    pytest.skip("needs macOS and exiftool", allow_module_level=True)
sys.path.insert(0, str(REL))
import render_notes as rn  # noqa: E402

PIL = pytest.importorskip("PIL.Image")
PAGE = rn.PAGE


def _run(*args, env_extra=None):
    env = dict(os.environ, PUBLISH_PY=sys.executable, **(env_extra or {}))
    return subprocess.run(["bash", str(REL / "stage_figures.sh"), *args], capture_output=True, text=True, env=env)


@pytest.fixture()
def figdir(tmp_path):
    """Synthetic figures carrying a planted EXIF/XMP comment so the strip is observable."""
    d = tmp_path / "figs"
    d.mkdir()
    for i, n in enumerate(rn.FIGURES):
        ext = n.rsplit(".", 1)[1]
        im = PIL.new("RGB", (8, 6), (20 + 8 * i, 200 - 6 * i, 40 + 3 * i))  # distinct bytes per file
        if ext == "jpg":
            im.save(d / n, "JPEG", comment=b"secret-comment")
        elif ext == "png":
            im.save(d / n, "PNG")
        else:
            im.save(d / n, "GIF")
    subprocess.run(["exiftool", "-q", "-overwrite_original", "-Artist=planted-owner", "-Software=planted-tool",
                    str(d / "fig_xmas_35_real_lens.jpg"), str(d / "fig_wedges_8.png")], check=True)
    return d


@pytest.fixture()
def page(tmp_path, monkeypatch):
    """A synthetic page plus a pin file (PAGE_SHA256_FILE) holding its sha256."""
    p = tmp_path / "page.html"
    p.write_text("<!doctype html><title>t</title><p>a synthetic page</p>\n")
    sha = hashlib.sha256(p.read_bytes()).hexdigest()
    pin = tmp_path / "page.sha256"
    pin.write_text(sha + "\n")
    monkeypatch.setenv("PAGE_SHA256_FILE", str(pin))
    return p, sha


def test_asset_list_is_version_gated():
    assert rn.extras("0.4.9") == [] and rn.extras("0.3.0") == []
    old = rn.assets("0.4.9")
    new = rn.assets("0.5.0")
    assert [n for n in new if n not in {*rn.FIGURES, PAGE}] == [n.replace("0.4.9", "0.5.0") for n in old]
    assert set(new) & set(rn.FIGURES) == set(rn.FIGURES) and PAGE in new
    assert new[-1] == "SHA256SUMS" and len(set(new)) == len(new)
    assert rn.extras("{{VERSION}}") == rn.extras("0.5.0")  # the docs' placeholder form lists them too
    out = subprocess.run([sys.executable, str(REL / "render_notes.py"), "--list-figures", "0.4.0"],
                         capture_output=True, text=True)
    assert out.stdout.strip() == ""


def test_stage_strips_and_verifies(tmp_path, figdir, page):
    p, sha = page
    out = tmp_path / "assets"
    r = _run("0.5.0", str(figdir), str(out), "--page", str(p))
    assert r.returncode == 0, r.stderr + r.stdout
    for n in (*rn.FIGURES, PAGE):
        assert (out / n).is_file(), n
    rows = subprocess.run(["exiftool", "-a", "-G1", "-s", str(out / "fig_xmas_35_real_lens.jpg")],
                          capture_output=True, text=True).stdout
    assert "planted" not in rows and "secret-comment" not in rows
    rows = subprocess.run(["exiftool", "-a", "-G1", "-s", str(out / "fig_wedges_8.png")], capture_output=True, text=True).stdout
    assert "planted" not in rows
    txt = (out / "FIGURES.txt").read_text()
    assert txt.startswith("version: 0.5.0\n") and txt.endswith("figure post-checks: all passed\n")
    assert _run("--verify", "0.5.0", str(out)).returncode == 0
    assert (figdir / "fig_xmas_35_real_lens.jpg").read_bytes() != (out / "fig_xmas_35_real_lens.jpg").read_bytes()


def test_verify_refuses_edited_missing_or_dirty(tmp_path, figdir, page):
    p, sha = page
    out = tmp_path / "assets"
    assert _run("0.5.0", str(figdir), str(out), "--page", str(p)).returncode == 0
    # edited after staging
    f = out / "fig_wedges_8.png"
    good = f.read_bytes()
    f.write_bytes(good + b"x")
    r = _run("--verify", "0.5.0", str(out))
    assert r.returncode != 0 and "sha256 not found" in r.stderr
    f.write_bytes(good)
    assert _run("--verify", "0.5.0", str(out)).returncode == 0
    # re-metadata'd but re-hashed file: the metadata rows are still caught
    dirty = out / "fig_xmas_35_real_lens.jpg"
    keep = dirty.read_bytes()
    subprocess.run(["exiftool", "-q", "-overwrite_original", "-Artist=planted-owner", str(dirty)], check=True)
    txt = (out / "FIGURES.txt").read_text().replace(hashlib.sha256(keep).hexdigest(), hashlib.sha256(dirty.read_bytes()).hexdigest())
    (out / "FIGURES.txt").write_text(txt)
    r = _run("--verify", "0.5.0", str(out))
    assert r.returncode != 0 and "metadata rows" in r.stderr
    dirty.write_bytes(keep)
    # missing figure
    (out / "fig_layer_weights.png").unlink()
    r = _run("--verify", "0.5.0", str(out))
    assert r.returncode != 0 and "missing" in r.stderr


def test_page_needs_matching_sha_and_no_sync_markers(tmp_path, figdir, page):
    p, sha = page
    pin = tmp_path / "page.sha256"
    out = tmp_path / "assets"
    pin.write_text("0" * 64 + "\n")  # wrong pin: the page must be refused
    r = _run("0.5.0", str(figdir), str(out), "--page", str(p))
    assert r.returncode != 0 and "does not match" in r.stderr
    assert not (out / PAGE).exists()
    marked = tmp_path / "marked.html"
    marked.write_text(p.read_text() + "<!-- sync-after-0.5.0 -->\n")
    pin.write_text(hashlib.sha256(marked.read_bytes()).hexdigest() + "\n")  # right sha, bad content
    r = _run("0.5.0", str(figdir), str(out), "--page", str(marked))
    assert r.returncode != 0 and "sync-after" in r.stderr
    for tok in ("IMG_", "S24_"):
        bad = tmp_path / "tok.html"
        bad.write_text(p.read_text() + f"<p>{tok}1234</p>\n")
        pin.write_text(hashlib.sha256(bad.read_bytes()).hexdigest() + "\n")
        r = _run("0.5.0", str(figdir), str(out), "--page", str(bad))
        assert r.returncode != 0 and tok in r.stderr
    # no page given: figures stage, the page does not, and verify (hence the publish) refuses
    out2 = tmp_path / "assets2"
    r = _run("0.5.0", str(figdir), str(out2))
    assert r.returncode == 0 and "NOT staged" in r.stderr
    r = _run("--verify", "0.5.0", str(out2))
    assert r.returncode != 0 and PAGE in r.stderr


def test_below_050_stages_nothing(tmp_path, figdir):
    r = _run("0.4.9", str(figdir), str(tmp_path / "a"))
    assert r.returncode != 0 and "0.5.0 and later" in r.stderr
    assert _run("--verify", "0.4.9", str(tmp_path / "a")).returncode == 0


def test_publish_helper_runs_the_verify_step():
    src = (REL / "publish_release.sh").read_text()
    assert 'stage_figures.sh" --verify "$VERSION" "$A"' in src
    assert "--list-figures" in src


def test_pin_file_is_the_reviewed_page_hash_and_verify_rechecks_it(tmp_path, figdir, page):
    real = (REL / "page.sha256").read_text().strip()
    assert real == "705a1c5d178347fa0429d3914b96b3d38b8e8aaff4cb86c8c3206b7059d4e176"
    p, sha = page
    out = tmp_path / "assets"
    assert _run("0.5.0", str(figdir), str(out), "--page", str(p)).returncode == 0
    assert _run("--verify", "0.5.0", str(out)).returncode == 0
    (tmp_path / "page.sha256").write_text("1" * 64 + "\n")  # the pin moved: the staged page no longer matches
    r = _run("--verify", "0.5.0", str(out))
    assert r.returncode != 0 and "page.sha256" in r.stderr
