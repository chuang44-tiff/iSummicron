"""R-release-build — Adversarial round (privacy gate holes, shipped-tree leaks).

Two halves:

* Synthetic trees (always run): probe ``scripts/release/privacy_gate.py`` with
  seeds the coder's self-test does not cover — the md5-pinned npz payload
  skip, structured-dtype / UTF-16 / IPv6 encodings, a forged TENSOR_MD5.
* Built artefacts (skip when the scratch build root is absent): the staged
  public clone, its release commit, the no-push state and the stripped sample
  images.

Secrets are assembled from pieces so this file passes ``.githooks/pre-commit``.
GAP xfails document defence-in-depth holes that are masked by another check
(the gate still exits non-zero); BUG xfails are real defects.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
REL = ROOT / "scripts" / "release"
if not (REL / "privacy_gate.py").is_file():
    pytest.skip("scripts/release not present (public tree)", allow_module_level=True)
sys.path.insert(0, str(REL))
import privacy_gate as pg  # noqa: E402

NPZ_SRC = ROOT / "summicron" / "_artifacts" / "codev_render_v1.npz"
MOUNT = "/Vol" + "umes/Da" + "ta/x"
# Built artefacts live outside every repo: set ISUMMICRON_RELEASE_ROOT to the release
# root (this file ships in tests/, so no local path is hard-coded here).
_RR = os.environ.get("ISUMMICRON_RELEASE_ROOT")
BUILD = Path(_RR) if _RR else None
CLONE = (BUILD / "public-clone") if BUILD else None
# The stripped assets sit in <root>/<12-hex-sha>/assets; take the newest one, no hard-coded SHA.
_asset_dirs = sorted(BUILD.glob("*/assets"), key=lambda p: p.stat().st_mtime) if BUILD and BUILD.is_dir() else []
ASSETS = _asset_dirs[-1] if _asset_dirs else None
_NO_ROOT = "ISUMMICRON_RELEASE_ROOT not set or not a build root (no public-clone)"


@pytest.fixture(scope="module")
def tokens():
    return pg.build_token_patterns(ROOT)


def _md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def _tree(tmp: Path, npz: Path | None = None, md5: str | None = None) -> Path:
    t = tmp / "tree"
    (t / "summicron/render").mkdir(parents=True)
    (t / "summicron/_artifacts").mkdir(parents=True)
    src = npz or NPZ_SRC
    dst = t / pg.NPZ_REL
    shutil.copyfile(src, dst)
    m = md5 or _md5(NPZ_SRC)
    (t / "summicron/render/_bench.py").write_text(f'TENSOR_MD5 = "{m}"\n')
    return t


def _checks(rep) -> set[str]:
    return {ln.split("]")[0].split("[")[1] for ln in rep.fails}


# ── synthetic: npz pin / payload skip ────────────────────────────────────────
def test_shipped_npz_has_a_chance_token_in_payload(tokens):
    """Confirms the deviation's premise: the float payload holds a chance NAS-token hit."""
    raw = NPZ_SRC.read_bytes().decode("latin-1")
    hits = [lbl for lbl, pat in tokens if pat.search(raw)]
    assert hits == ["nas-name"], hits


def test_pinned_npz_control_passes_content_checks(tmp_path, tokens):
    rep, _ = pg.check_tree(_tree(tmp_path), tokens, wheel=False)
    assert not (_checks(rep) & {"P1", "P5", "P6", "P7"}), rep.fails


def test_npz_with_trailing_mount_path_fails(tmp_path, tokens):
    bad = tmp_path / "bad.npz"
    bad.write_bytes(NPZ_SRC.read_bytes() + MOUNT.encode())
    rep, _ = pg.check_tree(_tree(tmp_path, bad), tokens, wheel=False)
    assert "P1" in _checks(rep)  # md5 pin broken -> not pinned -> full scan
    assert any("mount-path" in f for f in rep.fails), rep.fails


def test_rebuilt_npz_with_unicode_key_fails_p6(tmp_path, tokens):
    with np.load(NPZ_SRC, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    d["source"] = np.array(MOUNT)
    bad = tmp_path / "bad.npz"
    np.savez_compressed(bad, **d)
    rep, _ = pg.check_tree(_tree(tmp_path, bad), tokens, wheel=False)
    assert {"P1", "P6"} <= _checks(rep), rep.fails


def test_forged_tensor_md5_with_right_prefix_fails(tmp_path, tokens):
    fake = pg.TENSOR_PREFIX + "0" * 24
    rep, _ = pg.check_tree(_tree(tmp_path, md5=fake), tokens, wheel=False)
    assert "P1" in _checks(rep), rep.fails


def test_new_tensor_md5_is_fail_closed(tmp_path, tokens):
    """A future rebuilt tensor (new md5, new _bench pin) cannot be pinned: prefix gate fails."""
    other = tmp_path / "other.npz"
    with np.load(NPZ_SRC, allow_pickle=False) as z:
        np.savez(other, **{k: z[k] for k in z.files}, extra=np.zeros(1))
    rep, _ = pg.check_tree(_tree(tmp_path, other, md5=_md5(other)), tokens, wheel=False)
    assert "P1" in _checks(rep), rep.fails


def test_structured_dtype_string_in_npz_is_content_scanned(tmp_path, tokens):
    arr = np.array([(MOUNT, 1)], dtype=[("p", "U40"), ("n", "i4")])
    f = tmp_path / "s.npz"
    np.savez(f, meta=arr)
    rep = pg.Report()
    pg.check_npz(rep, f, "s.npz", tokens)
    assert any("mount-path" in x for x in rep.fails), rep.fails


def test_utf16_text_is_scanned(tmp_path, tokens):
    t = _tree(tmp_path)
    (t / "notes.txt").write_bytes(MOUNT.encode("utf-16"))
    rep, _ = pg.check_tree(t, tokens, wheel=False)
    assert any("mount-path" in x for x in rep.fails), rep.fails


def test_tailnet_ipv6_is_caught(tmp_path, tokens):
    t = _tree(tmp_path)
    (t / "notes.md").write_text("host fd7a:" + "115c:a1e0:" + ":12\n")
    rep, _ = pg.check_tree(t, tokens, wheel=False)
    assert "P5" in _checks(rep), rep.fails


def test_uppercase_ext_and_dotfile_rejected(tmp_path, tokens):
    t = _tree(tmp_path)
    (t / "a.PY").write_text("x\n")
    (t / ".env").write_text("x\n")
    rep, _ = pg.check_tree(t, tokens, wheel=False)
    assert sum("P1" in f for f in rep.fails) >= 2, rep.fails


# ── static: build script ─────────────────────────────────────────────────────
def test_release_scripts_never_amend_rebase_tag_or_reset():
    for p in REL.iterdir():
        if p.suffix not in (".sh", ".py"):
            continue
        if p.name == "publish_release.sh":
            continue  # needs `git tag -a` and `push --no-verify`; test_publish_release_adversarial T14d bans the dangerous forms
        txt = p.read_text()
        for bad in ("--amend", "rebase", "git tag", "reset --hard", "push --force", "--no-verify"):
            code = [ln for ln in txt.splitlines() if bad in ln and not ln.lstrip().startswith("#")]
            assert not code, (p.name, bad, code)


# ── overlay docs ─────────────────────────────────────────────────────────────
OVERLAY = ROOT / "release" / "public-overlay"


@pytest.mark.skipif(not OVERLAY.is_dir(), reason="no overlay")
def test_overlay_disclaimer_and_lens_facts():
    disc = ("Summicron and Leica are trademarks of Leica Camera AG. iSummicron is not "
            "affiliated with, sponsored by, or endorsed by Leica Camera AG.")
    for f in ("README.md", "NOTICE"):
        assert disc in (OVERLAY / f).read_text(), f
    rd = (OVERLAY / "README.md").read_text()
    assert "~100 mm f/2" in rd and "183, 183" in rd and "ag_dblgauss" in rd
    for gone in ("--dof-m", "--radial-blend-mode", "--layered-gather", "--kernel-centring",
                 "doc/", "team-dev", ".githooks", "deploy"):
        assert gone not in rd, gone


@pytest.mark.skipif(not OVERLAY.is_dir(), reason="no overlay")
def test_overlay_quickstart_command_matches_canonical_flags():
    """A1: the README quick-start flags equal summicron.render._bench.CANONICAL_FLAGS."""
    from summicron.render._bench import CANONICAL_FLAGS

    rd = (OVERLAY / "README.md").read_text()
    cmd = re.search(r"```bash\n(isummicron render[^\n]*)\n```", rd).group(1)
    toks = cmd.split()
    assert "--focus-xy" in toks and toks[toks.index("--focus-xy") + 1:][:2] == ["0.40", "0.39"]
    i = toks.index(CANONICAL_FLAGS[0])
    assert tuple(toks[i:i + len(CANONICAL_FLAGS)]) == tuple(CANONICAL_FLAGS), cmd


def test_package_docstring_is_honest():
    first = (ROOT / "summicron" / "__init__.py").read_text().splitlines()[0]
    assert "measured" not in first and "Summicron v4" not in first, first


# ── built artefacts (scratch build root) ─────────────────────────────────────
needs_clone = pytest.mark.skipif(CLONE is None or not (CLONE / ".git").is_dir(), reason=_NO_ROOT)


def _git(*a: str) -> str:
    return subprocess.run(["git", "-C", str(CLONE), *a], capture_output=True, text=True,
                          check=True).stdout


@needs_clone
def test_clone_has_no_remote_and_effective_blocking_prepush():
    assert _git("remote").strip() == ""
    hooks = _git("rev-parse", "--git-path", "hooks").strip()
    hp = (CLONE / hooks / "pre-push") if not os.path.isabs(hooks) else Path(hooks, "pre-push")
    assert hp.read_text().strip().endswith("exit 1") and os.access(hp, os.X_OK)


@needs_clone
def test_release_commit_identity_and_message(tokens, tmp_path):
    rng = "HEAD~1..HEAD"
    ids = _git("log", "--format=%ae%n%ce", rng).split()
    assert ids and all(pg.NOREPLY_ADDR_RE.match(e) for e in ids), ids
    msg = tmp_path / "m.txt"
    msg.write_text(_git("log", "-1", "--format=%B"))
    rep = pg.Report()
    pg.check_commit_msg(rep, msg, tokens)
    assert not rep.fails, rep.fails
    assert _git("rev-list", "--count", "HEAD").strip() == "2"  # one new commit atop public


@needs_clone
def test_release_commit_patch_has_no_tokens(tokens):
    patch = subprocess.run(["git", "-C", str(CLONE), "log", "-p", "--binary", "-1",
                            "--", ".", ":!summicron/_artifacts/codev_render_v1.npz"],
                           capture_output=True, check=True).stdout.decode("latin-1")
    hits = sorted({lbl for lbl, pat in tokens if pat.search(patch)})
    assert hits == [], hits
    body = re.sub(r"(?m)^[+ -]", "", patch)  # drop diff markers ("+@pytest" is not an email)
    emails = {m for m in pg.EMAIL_RE.findall(body) if not pg.NOREPLY_ADDR_RE.match(m)}
    assert not emails


@needs_clone
def test_clone_npz_is_pinned_bytes():
    assert _md5(CLONE / pg.NPZ_REL) == _md5(NPZ_SRC)


needs_assets = pytest.mark.skipif(ASSETS is None or shutil.which("exiftool") is None,
                                  reason="ISUMMICRON_RELEASE_ROOT has no <sha>/assets dir, or exiftool is missing")
FORBIDDEN_TAG = re.compile(r"GPS|Region|Face|DateTimeOriginal|CreateDate|ModifyDate|SubSec|"
                           r"OffsetTime|Serial|Owner|LensMake|LensModel|LensInfo|^Make$|^Model$|"
                           r"Software|HostComputer|ContentIdentifier|ImageUniqueID|Location", re.I)


@needs_assets
@pytest.mark.parametrize("name", ["Demo_input.HEIC", "Demo_portraitmode.HEIC"])
def test_stripped_asset_has_no_identifying_tags(name):
    out = subprocess.run(["exiftool", "-j", "-a", "-G1", "-ee", "-u", str(ASSETS / name)],
                         capture_output=True, text=True, check=True).stdout
    rec = json.loads(out)[0]
    bad = [k for k in rec
           if not k.startswith(("System:", "File:", "ExifTool:", "Composite:"))
           and FORBIDDEN_TAG.search(k.split(":", 1)[-1])]
    assert bad == [], bad


# ── --commit refuses without a matching all-gates marker ─────────────────────
def _commit_env(tmp_path):
    """A fake build workspace for the dev HEAD sha: stage + tree.sha, but no marker yet."""
    sha = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True,
                         text=True, check=True).stdout.strip()
    pub = tmp_path / "pub"
    subprocess.run(["git", "init", "-q", str(pub)], check=True)
    subprocess.run(["git", "-C", str(pub), "-c", "user.name=t", "-c", "user.email=t" + "@" + "users.noreply.github.com",
                    "commit", "-q", "--allow-empty", "-m", "x"], check=True)
    root = tmp_path / "rel"
    w = root / sha[:12]
    (w / "stage").mkdir(parents=True)
    (w / "stage" / "f.txt").write_text("x\n")
    (w / "tree.sha").write_text("0" * 40 + "\n")
    env = dict(os.environ, RELEASE_ROOT=str(root), PUBLIC_SOURCE=str(pub))
    env.pop("RELEASE_AUTHOR", None)
    return sha, w, env


def _run_commit(sha, env):
    return subprocess.run(["bash", str(REL / "build_release.sh"), sha, "0.3.0", "--commit"],
                          cwd=ROOT, env=env, capture_output=True, text=True)


def test_commit_refused_without_gates_marker(tmp_path):
    sha, w, env = _commit_env(tmp_path)
    r = _run_commit(sha, env)
    assert r.returncode != 0 and "gates-passed" in r.stderr, (r.returncode, r.stderr)


def test_commit_refused_when_marker_hash_differs_from_stage(tmp_path):
    sha, w, env = _commit_env(tmp_path)
    (w / "gates-passed").write_text("f" * 40 + "\n")
    r = _run_commit(sha, env)
    assert r.returncode != 0 and "marker hash" in r.stderr, (r.returncode, r.stderr)
