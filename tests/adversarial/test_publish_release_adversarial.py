"""R-release-publish: tests for scripts/release/publish_release.sh and its helpers.

Never touches GitHub: the "public" remote is a scratch bare repo reached by a file URL,
and `gh` is a stub on PATH that records its argv and keeps release state on disk.
Spec: doc/plan/phase2/section30_release-build/publish-helper-spec.md
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
REL = ROOT / "scripts" / "release"
PUBLISH = REL / "publish_release.sh"
if not PUBLISH.is_file():
    pytest.skip("scripts/release/publish_release.sh not present (public tree)", allow_module_level=True)
sys.path.insert(0, str(REL))
import changelog_section as cs  # noqa: E402
import render_notes as rn  # noqa: E402

NPZ_SRC = ROOT / "summicron" / "_artifacts" / "codev_render_v1.npz"
V = "9.9.9"
AUTHOR = "T <1+chuang44-tiff@users.noreply.github.com>"
NOREPLY = "1+chuang44-tiff@users.noreply.github.com"
SCHEME_FILE = "file" + ":" + "//"
SCHEME_HTTPS = "ht" + "tps" + ":" + "//"
PERSONAL = "me" + "@" + "example.com"  # built at runtime: P5 scans shipped files for literal emails
PH = "{{" + "VERSION}}"
HOOK = "#!/bin/sh\nexit 1\n"
ASSET_NAMES = rn.assets(V)
GALLERY_JPEGS = ["Demo_input_before.jpg", "Demo_input_after.jpg",
                 "Demo_input_before_edge.jpg", "Demo_input_after_edge.jpg"]
NOTES_SRC = """Lead sentence.

## Highlights

- one

## Gallery

![a](https://github.com/chuang44-tiff/iSummicron/releases/download/v{{VERSION}}/Demo_input_after.jpg)

## Install

pip install the wheel from v{{VERSION}}.

## Try it

run it

## Upgrading from 9.9.8

- nothing

## What changed

{{CHANGELOG_SECTION}}

## Known limitations

1. some

## Credits

- some

## Images and privacy

{{DISCLOSURE}}

## Checksums

{{SHA256SUMS}}
"""
CHANGELOG = """# Changelog

## [9.9.9] - 2026-01-03

### Added

- newest thing

## [9.9.8] - 2026-01-02

- middle thing

## [9.9.7] - 2026-01-01

- oldest thing

[9.9.8]: ./compare-a-b
[9.9.7]: ./tag-v9.9.7
"""

pytestmark = pytest.mark.skipif(sys.platform != "darwin", reason="release tooling uses macOS md5/sed")


def _git(*a, cwd=None, env=None, check=True):
    e = {"PATH": "/usr/bin:/bin", "HOME": str(cwd or "/"), "GIT_CONFIG_GLOBAL": "/dev/null",
         "GIT_CONFIG_SYSTEM": "/dev/null"}
    e.update(env or {})
    return subprocess.run(["git", *a], cwd=cwd, env=e, capture_output=True, text=True, check=check)


def _ident(who=NOREPLY):
    return {"GIT_AUTHOR_NAME": "T", "GIT_AUTHOR_EMAIL": who, "GIT_COMMITTER_NAME": "T",
            "GIT_COMMITTER_EMAIL": who}


STUB_GH = r"""#!/bin/bash
echo "$*" >> "$GH_LOG"
S="$GH_STATE"; mkdir -p "$S/assets"
has() { local x; for x in "$@"; do [ "$x" = "$NEEDLE" ] && return 0; done; return 1; }
case "$1 $2" in
  "auth status") exit 0 ;;
  "release view")
    [ -f "$S/exists" ] || exit 1
    draft=false; [ -f "$S/draft" ] && draft=true
    out=""; for f in "$S"/assets/*; do [ -e "$f" ] || continue
      out="$out{\"name\":\"$(basename "$f")\",\"size\":$(cat "$f")},"; done
    echo "{\"tagName\":\"v9.9.9\",\"isDraft\":$draft,\"assets\":[${out%,}]}" ;;
  "release create")
    NEEDLE=--verify-tag; has "$@" || exit 99
    NEEDLE=--clobber; has "$@" && exit 99
    touch "$S/exists"; NEEDLE=--draft; has "$@" && touch "$S/draft"; exit 0 ;;
  "release upload")
    NEEDLE=--clobber; has "$@" && exit 99
    f="${@: -1}"; wc -c < "$f" | tr -d ' ' > "$S/assets/$(basename "$f")" ;;
  "release edit") NEEDLE=--draft=false; has "$@" && rm -f "$S/draft"; exit 0 ;;
  "pr create") touch "$S/pr" ;;
  "pr view") [ -f "$S/pr" ] || exit 1; echo '{"url":"x","state":"OPEN"}' ;;
  *) exit 98 ;;
esac
"""


class World:
    def __init__(self, tmp: Path):
        self.tmp = tmp
        self.fake = tmp / "fake.git"
        self.root = tmp / "relroot"
        self.clone = self.root / "public-clone"
        self.ws = self.root / "abcdef012345"
        self.gh_log = tmp / "gh.log"
        self.gh_state = tmp / "ghstate"
        self.bin = tmp / "bin"
        self._build()

    def _build(self):
        t = self.tmp
        _git("init", "-q", "--bare", "-b", "main", str(self.fake))
        seed = t / "seed"
        seed.mkdir()
        _git("init", "-q", "-b", "main", cwd=seed)
        (seed / "README.md").write_text("base\n")
        _git("add", "-A", cwd=seed)
        _git("commit", "-q", "-m", "base", cwd=seed, env=_ident())
        _git("push", "-q", str(self.fake), "main", cwd=seed)
        self.root.mkdir()
        _git("clone", "-q", str(self.fake), str(self.clone))
        _git("remote", "remove", "origin", cwd=self.clone)
        _git("config", "push.default", "nothing", cwd=self.clone)
        hp = self.clone / ".git" / "hooks" / "pre-push"
        hp.parent.mkdir(exist_ok=True)
        hp.write_text(HOOK)
        hp.chmod(0o755)
        _git("checkout", "-q", "-b", f"release/{V}", cwd=self.clone)
        (self.clone / "pyproject.toml").write_text(f'[project]\nname = "isummicron"\nversion = "{V}"\n')
        (self.clone / "summicron").mkdir()
        (self.clone / "summicron" / "__init__.py").write_text(f'__version__ = "{V}"\n')
        (self.clone / "CHANGELOG.md").write_text(CHANGELOG)
        _git("add", "-A", cwd=self.clone)
        _git("commit", "-q", "-m", f"release: {V} (2026-01-03)", cwd=self.clone, env=_ident())
        self.tip = self.rev(f"release/{V}")
        self.base = self.rev(f"release/{V}~1")
        tree = _git("rev-parse", f"{self.tip}^{{tree}}", cwd=self.clone).stdout.strip()
        # workspace
        (self.ws / "dist").mkdir(parents=True)
        (self.ws / "assets").mkdir()
        (self.ws / "gates-passed").write_text(tree + "\n")
        (self.ws / "tree.sha").write_text(tree + "\n")
        whl = self.ws / "dist" / ASSET_NAMES[0]
        sd = self.ws / "dist" / ASSET_NAMES[1]
        whl.write_bytes(b"wheel-bytes")
        sd.write_bytes(b"sdist-bytes!!")
        (self.ws / "wheel.md5").write_text(self.md5(whl) + "\n")
        (self.ws / "sdist.md5").write_text(self.md5(sd) + "\n")
        for n, b in (("Demo_input.HEIC", b"heic-one"), ("Demo_portraitmode.HEIC", b"heic-two-xx")):
            (self.ws / "assets" / n).write_bytes(b)
        (self.ws / "assets" / "MANIFEST.txt").write_text(
            "".join(f"{self.md5(self.ws / 'assets' / n)}  {n}\n"
                    for n in ("Demo_input.HEIC", "Demo_portraitmode.HEIC"))
            + "post-checks (i)-(v): all passed\n")
        from PIL import Image
        for n in GALLERY_JPEGS:  # real, metadata-free JPEGs
            Image.new("RGB", (8, 6), (90, 90, 90)).save(self.ws / "assets" / n, "JPEG")
            subprocess.run(["exiftool", "-q", "-overwrite_original", "-all=", str(self.ws / "assets" / n)], check=True)
        (self.ws / "assets" / "GALLERY.txt").write_text(
            f"version: {V}\n" + "".join(f"{n} md5: {self.md5(self.ws / 'assets' / n)}\n" for n in GALLERY_JPEGS)
            + "gallery post-checks (i)-(v): all passed\n")
        (self.ws / "notes-src.md").write_text(NOTES_SRC)
        (self.ws / "notes-src.md5").write_text(self.md5(self.ws / "notes-src.md") + "\n")
        # minimal stage that passes the REAL privacy gate
        st = self.ws / "stage"
        (st / "summicron/render").mkdir(parents=True)
        (st / "summicron/_artifacts").mkdir(parents=True)
        shutil.copyfile(NPZ_SRC, st / "summicron/_artifacts/codev_render_v1.npz")
        md5 = self.md5(NPZ_SRC)
        (st / "summicron/render/_bench.py").write_text(f'TENSOR_MD5 = "{md5}"\n')
        # stub gh + env
        self.bin.mkdir()
        gh = self.bin / "gh"
        gh.write_text(STUB_GH)
        gh.chmod(0o755)
        (self.bin / "exiftool").symlink_to(shutil.which("exiftool"))  # the privacy gate needs it
        self.gh_log.write_text("")
        self.env = {
            "PATH": f"{self.bin}:/usr/bin:/bin:/sbin", "HOME": str(t), "GH_CONFIG_DIR": str(t / "ghcfg"),
            "GH_LOG": str(self.gh_log), "GH_STATE": str(self.gh_state),
            "ISUMMICRON_PUBLISH_TEST": "1", "PUBLISH_PY": sys.executable,
            "PUBLIC_URL": SCHEME_FILE + str(self.fake), "GH_REPO": "test-owner/test-repo",
            "RELEASE_AUTHOR": AUTHOR, "RELEASE_ROOT": str(self.root),
        }

    @staticmethod
    def md5(p: Path) -> str:
        return subprocess.run(["/sbin/md5", "-q", str(p)], capture_output=True, text=True).stdout.strip()

    def rev(self, ref: str) -> str:
        return _git("rev-parse", ref, cwd=self.clone).stdout.strip()

    def run(self, *args, inp: str | None = "", env_drop=(), env_set=None, version=V):
        env = {k: v for k, v in self.env.items() if k not in env_drop}
        env.update(env_set or {})
        kw = {"stdin": subprocess.DEVNULL} if inp is None else {"input": inp}
        return subprocess.run(["bash", str(PUBLISH), version, *args], env=env, capture_output=True,
                              text=True, timeout=300, **kw)

    def refs(self) -> str:
        return _git("for-each-ref", cwd=self.fake).stdout

    def log(self) -> list[str]:
        return [ln for ln in self.gh_log.read_text().splitlines() if ln]

    def mutating(self) -> list[str]:
        return [ln for ln in self.log() if re.match(r"(release (create|upload|edit)|pr create)", ln)]


@pytest.fixture
def w(tmp_path):
    if shutil.which("exiftool") is None:
        pytest.skip("exiftool missing (the privacy gate fails closed without it)")
    return World(tmp_path)


def _clone_pristine(w: World):
    assert _git("remote", cwd=w.clone).stdout.strip() == ""
    assert (w.clone / ".git/hooks/pre-push").read_text() == HOOK


# ---------------------------------------------------------------- T1-T4
def test_t1_dry_run_is_offline_and_touches_nothing(w):
    before = w.refs()
    r = w.run("--dry-run")
    assert r.returncode == 0, r.stderr + r.stdout
    out = r.stdout
    assert "DRY RUN" in out
    assert re.search(r'git .*push --no-verify "?file:/{2}\S+ refs/heads/release/9\.9\.9:refs/heads/main', out)
    assert "gh release create v9.9.9" in out and "--verify-tag" in out
    for n in ASSET_NAMES:
        assert n in out
    assert w.log() == []
    assert w.refs() == before
    assert _git("tag", "-l", cwd=w.clone).stdout.strip() == ""
    assert _git("for-each-ref", "refs/publish", cwd=w.clone).stdout.strip() == ""
    _clone_pristine(w)


def test_t2_dry_run_needs_no_stdin_or_author(w):
    r = w.run("--dry-run", inp=None, env_drop=("RELEASE_AUTHOR",))
    assert r.returncode == 0, r.stderr + r.stdout


def test_t3_confirm_mismatch(w):
    before = w.refs()
    r = w.run(inp="9.9.8\n")
    assert r.returncode == 3, r.stderr + r.stdout
    assert w.refs() == before
    assert all(ln.startswith("auth status") for ln in w.log())
    assert _git("tag", "-l", cwd=w.clone).stdout.strip() == ""


def test_t4_confirm_eof(w):
    before = w.refs()
    r = w.run(inp="")
    assert r.returncode == 3, r.stderr + r.stdout
    assert w.refs() == before
    assert all(ln.startswith("auth status") for ln in w.log())
    assert _git("tag", "-l", cwd=w.clone).stdout.strip() == ""


# ---------------------------------------------------------------- T5-T7
def _check_published(w: World):
    assert _git("rev-parse", "refs/heads/main", cwd=w.fake).stdout.strip() == w.tip
    assert _git("cat-file", "-t", f"refs/tags/v{V}", cwd=w.fake).stdout.strip() == "tag"
    assert _git("rev-parse", f"refs/tags/v{V}^{{commit}}", cwd=w.fake).stdout.strip() == w.tip
    tagger = [ln for ln in _git("cat-file", "-p", f"refs/tags/v{V}", cwd=w.fake).stdout.splitlines()
              if ln.startswith("tagger ")][0]
    assert f"<{NOREPLY}>" in tagger
    assert (w.gh_state / "exists").exists() and not (w.gh_state / "draft").exists()
    assert sorted(p.name for p in (w.gh_state / "assets").iterdir()) == sorted(ASSET_NAMES)
    _clone_pristine(w)


def test_t5_happy_path(w):
    r = w.run(inp=f"{V}\n")
    assert r.returncode == 0, r.stderr + r.stdout
    _check_published(w)
    mut = w.mutating()
    assert mut[0].startswith("release create") and "--draft" in mut[0] and "--verify-tag" in mut[0]
    n = len(ASSET_NAMES)
    assert [m.split()[1] for m in mut[1:1 + n]] == ["upload"] * n
    assert all("--clobber" not in m for m in mut)
    assert mut[1 + n].startswith("release edit") and "--draft=false" in mut[1 + n] and len(mut) == n + 2


def test_t26_fix_a2_locale_independent_sort(w):
    """PS9 sort must not depend on the caller's locale (LANG=en_US.UTF-8, LC_ALL unset)."""
    r = w.run(inp=f"{V}\n", env_drop=("LC_ALL",), env_set={"LANG": "en_US.UTF-8"})
    assert r.returncode == 0, r.stderr + r.stdout
    _check_published(w)


def test_t27_fix_a3_two_line_marker_accepted(w):
    """The rel04 tooling writes tree id + a second line; only the first line is compared."""
    tree = (w.ws / "tree.sha").read_text().strip()
    (w.ws / "gates-passed").write_text(tree + "\nsecond line from the rel04 tooling\n")
    r = w.run("--dry-run")
    assert r.returncode == 0, r.stderr + r.stdout
    (w.ws / "gates-passed").write_text("deadbeef\n" + tree + "\n")  # tree only on line 2: refused
    r = w.run("--dry-run")
    assert r.returncode == 4 and "no all-gates marker" in r.stderr


def test_t6_idempotent_rerun(w):
    assert w.run(inp=f"{V}\n").returncode == 0
    refs, log = w.refs(), w.log()
    r = w.run(inp=f"{V}\n")
    assert r.returncode == 0, r.stderr + r.stdout
    assert r.stdout.count("SKIP") >= 3
    assert len(w.mutating()) == len(ASSET_NAMES) + 2
    assert w.refs() == refs
    assert [ln for ln in w.log()[len(log):] if not ln.startswith(("auth", "release view"))] == []


def test_t7_resume_after_partial(w):
    assert w.run(inp=f"{V}\n").returncode == 0
    # simulate a crash: back to a draft holding 2 of the assets
    (w.gh_state / "draft").touch()
    for n in ASSET_NAMES[2:]:
        (w.gh_state / "assets" / n).unlink()
    n0 = len(w.mutating())
    r = w.run(inp=f"{V}\n")
    assert r.returncode == 0, r.stderr + r.stdout
    new = w.mutating()[n0:]
    assert [m.split()[1] for m in new] == ["upload"] * (len(ASSET_NAMES) - 2) + ["edit"]
    _check_published(w)


# ---------------------------------------------------------------- T8-T13
def test_t8_refuse_to_move_remote_tag(w):
    _git("tag", f"v{V}", w.base, cwd=w.fake)
    before = w.refs()
    r = w.run(inp=f"{V}\n")
    assert r.returncode == 5 and "refusing to move existing tag" in r.stderr, r.stderr
    assert w.refs() == before
    assert not any(ln.startswith("release create") for ln in w.log())


def test_t9_refuse_to_move_local_tag(w):
    _git("tag", f"v{V}", w.base, cwd=w.clone)
    before = w.refs()
    r = w.run(inp=f"{V}\n")
    assert r.returncode == 5 and "refusing to move existing tag" in r.stderr, r.stderr
    assert w.refs() == before


def _mk_tag(w, *args, env=None):
    _git("-c", "tag.gpgsign=false", "tag", *args, cwd=w.clone, env=env)


def test_t22_fix_b2_local_tag_must_be_annotated_noreply(w):
    before = w.refs()
    _mk_tag(w, f"v{V}", str(w.tip))  # lightweight
    for args in (("--dry-run",), ()):
        r = w.run(*args, inp=f"{V}\n")
        assert r.returncode == 5 and "not an annotated tag" in r.stderr and "tag -d" in r.stderr, r.stderr
    _git("tag", "-d", f"v{V}", cwd=w.clone)
    _mk_tag(w, "-a", f"v{V}", "-m", f"iSummicron {V}", str(w.tip), env=_ident(PERSONAL))  # personal tagger
    r = w.run("--dry-run")
    assert r.returncode == 5 and "noreply" in r.stderr, r.stderr
    _git("tag", "-d", f"v{V}", cwd=w.clone)
    _mk_tag(w, "-a", f"v{V}", "-m", "wrong message", str(w.tip),
            env={"GIT_COMMITTER_NAME": "x", "GIT_COMMITTER_EMAIL": NOREPLY})
    r = w.run("--dry-run")
    assert r.returncode == 5 and "message" in r.stderr, r.stderr
    assert w.refs() == before


def test_t23_fix_b1_extra_commit_below_tip_refused(w):
    # personal-identity commit under a correct release commit, remote has only base
    _git("checkout", "-q", "-b", "tmp", w.base, cwd=w.clone)
    (w.clone / "x.txt").write_text("x\n")
    _git("add", "-A", cwd=w.clone)
    _git("commit", "-q", "-m", "hand edit", cwd=w.clone, env=_ident(PERSONAL))
    _git("cherry-pick", w.tip, cwd=w.clone, env={"GIT_COMMITTER_NAME": "x", "GIT_COMMITTER_EMAIL": NOREPLY})
    _git("branch", "-f", f"release/{V}", "HEAD", cwd=w.clone)
    _git("checkout", "-q", f"release/{V}", cwd=w.clone)
    tip = w.rev(f"release/{V}")
    tree = _git("rev-parse", f"{tip}^{{tree}}", cwd=w.clone).stdout.strip()
    (w.ws / "gates-passed").write_text(tree + "\n")
    (w.ws / "tree.sha").write_text(tree + "\n")
    before = w.refs()
    r = w.run(inp=f"{V}\n")
    assert r.returncode == 4 and "not a noreply address; nothing was pushed" in r.stderr, r.stderr
    assert w.refs() == before and not w.mutating()


def test_t24_empty_changelog_section_refused(w):
    # section header present but body empty
    chg = "# Changelog\n\n## [9.9.9] - 2026-01-03\n\n## [0.0.1] - 2025-01-01\n\n- old\n"
    (w.clone / "CHANGELOG.md").write_text(chg)
    _git("add", "-A", cwd=w.clone)
    _git("commit", "-q", "--amend", "--no-edit", cwd=w.clone, env=_ident())
    tip = w.rev(f"release/{V}")
    tree = _git("rev-parse", f"{tip}^{{tree}}", cwd=w.clone).stdout.strip()
    (w.ws / "gates-passed").write_text(tree + "\n")
    (w.ws / "tree.sha").write_text(tree + "\n")
    r = w.run("--dry-run")
    assert r.returncode == 4 and "empty" in r.stderr, r.stderr


def test_t25_test_mode_rejects_canonical_gh_repo(w):
    r = w.run("--dry-run", env_set={"GH_REPO": "chuang44-tiff/iSummicron"})
    assert r.returncode == 4 and "GH_REPO" in r.stderr, r.stderr


def test_t10_missing_marker(w):
    (w.ws / "gates-passed").unlink()
    for args in ((), ("--dry-run",)):
        r = w.run(*args)
        assert r.returncode == 4 and "no all-gates marker" in r.stderr, r.stderr
    assert w.log() == []


def test_t11_marker_mismatch_and_ambiguous(w):
    (w.ws / "gates-passed").write_text("deadbeef\n")
    r = w.run("--dry-run")
    assert r.returncode == 4 and "no all-gates marker" in r.stderr
    tree = (w.ws / "tree.sha").read_text()
    (w.ws / "gates-passed").write_text(tree)
    other = w.root / "0123456789ab"
    other.mkdir()
    (other / "gates-passed").write_text(tree)
    r = w.run("--dry-run")
    assert r.returncode == 4 and "ambiguous" in r.stderr, r.stderr


def test_t12_artefact_tamper(w):
    (w.ws / "dist" / ASSET_NAMES[0]).write_bytes(b"tampered")
    assert w.run("--dry-run").returncode == 4
    (w.tmp / "second").mkdir()
    w2 = World(w.tmp / "second")
    (w2.ws / "assets" / "Demo_input.HEIC").write_bytes(b"different")
    assert w2.run("--dry-run").returncode == 4


def test_t13_diverged_main(w):
    other = w.tmp / "other"
    other.mkdir()
    _git("init", "-q", "-b", "main", cwd=other)
    (other / "x").write_text("x\n")
    _git("add", "-A", cwd=other)
    _git("commit", "-q", "-m", "unrelated", cwd=other, env=_ident())
    _git("push", "-q", "--force", str(w.fake), "main", cwd=other)
    before = w.refs()
    r = w.run(inp=f"{V}\n")
    assert r.returncode == 6, r.stderr + r.stdout
    assert w.refs() == before
    assert not any(ln.startswith("release create") for ln in w.log())


# ---------------------------------------------------------------- T14
def test_t14_guard_rails(w):
    r = w.run("--dry-run", env_set={"PUBLIC_URL": SCHEME_HTTPS + "x.invalid/x.git"})
    assert r.returncode == 4
    r = w.run("--dry-run", env_drop=("ISUMMICRON_PUBLISH_TEST",))
    assert r.returncode == 4
    for bad in ("0.3.0.dev0", "v0.3.0", "1.2"):
        assert w.run("--dry-run", version=bad).returncode == 2
    assert w.run("--bogus").returncode == 2


def test_t14d_static_scan_of_publisher():
    code = [ln for ln in PUBLISH.read_text().splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    banned = ["remote add", "--force", "push -f", "tag -f", "tag -d", "--delete", "--clobber",
              "release delete", "config --unset", "refs/tags/+", "\"+refs"]
    for ln in code:
        for b in banned:
            assert b not in ln, (b, ln)
        assert not ("rm " in ln and "hooks" in ln), ln
        assert not re.search(r"""['" ]\+refs/""", ln), ln
        if re.search(r"(\bgit\b|\bG)\s+(-C\s+\S+\s+)?push\b", ln):
            assert "--no-verify" in ln and "$PUBLIC_URL" in ln, ln


# ---------------------------------------------------------------- T15
def test_t15_pr_mode(w):
    main0 = _git("rev-parse", "refs/heads/main", cwd=w.fake).stdout.strip()
    r = w.run("--pr", inp=f"{V}\n")
    assert r.returncode == 0, r.stderr + r.stdout
    assert _git("rev-parse", f"refs/heads/release/{V}", cwd=w.fake).stdout.strip() == w.tip
    assert _git("rev-parse", "refs/heads/main", cwd=w.fake).stdout.strip() == main0
    assert any("pr create" in ln and "--base main" in ln and f"--head release/{V}" in ln for ln in w.log())
    assert not any(ln.startswith("release") and "create" in ln for ln in w.log())
    assert _git("tag", "-l", cwd=w.fake).stdout.strip() == ""
    r = w.run("--pr", inp=f"{V}\n")
    assert r.returncode == 0
    assert sum("pr create" in ln for ln in w.log()) == 1
    r = w.run(inp=f"{V}\n")
    assert r.returncode == 0, r.stderr + r.stdout
    _check_published(w)


# ---------------------------------------------------------------- T16-T17
def test_t16_changelog_extraction():
    assert cs.section(CHANGELOG, "9.9.9") == "### Added\n\n- newest thing"
    assert cs.section(CHANGELOG, "9.9.8") == "- middle thing"
    assert cs.section(CHANGELOG, "9.9.7") == "- oldest thing"  # link refs excluded
    with pytest.raises(ValueError):
        cs.section(CHANGELOG.replace("[9.9.9]", "[9x9x9]"), "9.9.9")
    with pytest.raises(ValueError):
        cs.section("## [1.0.0]\n\n## [0.9.0]\n- a\n", "1.0.0")
    with pytest.raises(ValueError):
        cs.section(CHANGELOG, "1.2.3")
    ov = ROOT / "release" / "public-overlay" / "CHANGELOG.md"
    if ov.is_file():
        body = cs.section(ov.read_text(), "0.3.0")
        assert body.startswith("### Breaking") and "## [0.2.0]" not in body
        assert not re.search(r"(?m)^\[[^\]]+\]:", body)


def test_t17_notes_and_sums(w):
    assert w.run("--dry-run").returncode == 0
    notes = (w.ws / "publish" / "notes.md").read_text()
    disc = (REL / "release-notes-disclosure.md").read_text()
    sums_txt = (w.ws / "publish" / "SHA256SUMS").read_text()
    expect = rn.render(NOTES_SRC, V, CHANGELOG, disc, sums_txt) + "\n"
    assert notes == expect
    assert cs.section(CHANGELOG, V) in notes and disc.strip("\n") in notes and sums_txt.strip("\n") in notes
    sums = sums_txt.splitlines()
    assert [ln.split()[-1] for ln in sums] == ASSET_NAMES[:-1]
    for ln in sums:  # digests verified against the files themselves
        d, n = ln.split()
        src = w.ws / ("dist" if n.startswith("isummicron") else "assets") / n
        assert subprocess.run(["shasum", "-a", "256", str(src)], capture_output=True,
                              text=True).stdout.split()[0] == d


# ---------------------------------------------------------------- T18-T19
OVERLAY_README = ROOT / "release" / "public-overlay" / "README.md"
needs_overlay = pytest.mark.skipif(not OVERLAY_README.is_file(), reason="no overlay")


@needs_overlay
def test_t18_readme_placeholders():
    rd = OVERLAY_README.read_text()
    base = "https://github.com/chuang44-tiff/iSummicron"
    assert f"{base}/releases/download/v{PH}/isummicron-{PH}-py3-none-any.whl" in rd
    assert f"git+{base}@v{PH}" in rd
    assert f"{base}/releases/download/v{PH}/Demo_input.HEIC" in rd
    assert f"{base}/releases/download/v{PH}/SHA256SUMS" in rd
    assert f"{base}/releases" in rd
    assert not re.search(r"releases/download/v\d", rd) and not re.search(r"isummicron-\d", rd)
    assert not re.search(r"@v\d", rd)


@needs_overlay
@pytest.mark.skipif(shutil.which("sed") is None, reason="no sed")
def test_t19_readme_stamp_simulation(tmp_path):
    cp = tmp_path / "README.md"
    shutil.copyfile(OVERLAY_README, cp)
    subprocess.run(["sed", "-i", "", f"s/{PH}/{V}/g", str(cp)], check=True)
    rd = cp.read_text()
    assert PH not in rd
    assert f"/v{V}/isummicron-{V}-py3-none-any.whl" in rd
    sys.path.insert(0, str(ROOT))
    from summicron.render._bench import CANONICAL_FLAGS

    cmd = re.search(r"```bash\n(isummicron render[^\n]*)\n```", rd).group(1)
    toks = cmd.split()
    assert toks[toks.index("--focus-xy") + 1:][:2] == ["0.40", "0.39"]
    i = toks.index(CANONICAL_FLAGS[0])
    assert tuple(toks[i:i + len(CANONICAL_FLAGS)]) == tuple(CANONICAL_FLAGS)


# ---------------------------------------------------------------- T20
def _nopush_static(release_dir: Path) -> subprocess.CompletedProcess:
    txt = (REL / "build_release.sh").read_text()
    m = re.search(r"(?ms)^nopush_static\(\) \{.*?^\}\n", txt)
    assert m
    script = f'HERE="{release_dir}"\n{m.group(0)}\nnopush_static\n'
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True)


def test_t20_build_static_gate_with_publisher_present(tmp_path):
    copy = tmp_path / "rel"
    shutil.copytree(REL, copy, ignore=shutil.ignore_patterns("__pycache__"))
    assert (copy / "publish_release.sh").is_file()
    r = _nopush_static(copy)
    assert r.returncode == 0, r.stdout + r.stderr
    # another script that invokes the publisher must trip the gate
    (copy / "evil.sh").write_text('bash "$HERE/publish_release.sh" 1.0.0\n')
    r = _nopush_static(copy)
    assert r.returncode != 0
    (copy / "evil.sh").unlink()
    # and a push line in any other file still trips it
    (copy / "evil.sh").write_text("git push origin main\n")
    assert _nopush_static(copy).returncode != 0


# ---------------------------------------------------------------- T21
def _mk_sdist(tmp: Path, stage: Path, *, drop=None, extra=None, mutate=None, version=V, info=None):
    top = f"isummicron-{version}"
    out = tmp / f"{top}.tar.gz"
    pkg = info if info is not None else f"Metadata-Version: 2.4\nName: isummicron\nVersion: {version}\n"
    with tarfile.open(out, "w:gz") as tf:
        def add(name: str, data: bytes):
            ti = tarfile.TarInfo(f"{top}/{name}")
            ti.size = len(data)
            import io
            tf.addfile(ti, io.BytesIO(data))
        for p in sorted(stage.rglob("*")):
            if p.is_file():
                n = p.relative_to(stage).as_posix()
                if n == drop:
                    continue
                add(n, b"changed" if n == mutate else p.read_bytes())
        add("PKG-INFO", pkg.encode())
        if extra:
            add(extra, b"x")
    return out


def _sdist_check(tgz: Path, stage: Path, version=V) -> int:
    return subprocess.run([sys.executable, str(REL / "release_checks.py"), "sdist", "--sdist", str(tgz),
                           "--stage", str(stage), "--version", version, "--dev-repo", str(ROOT)],
                          capture_output=True, text=True).returncode


def test_t21_sdist_check(tmp_path):
    stage = tmp_path / "stage"
    (stage / "summicron").mkdir(parents=True)
    (stage / "pyproject.toml").write_text("x\n")
    (stage / "summicron" / "a.py").write_text("a = 1\n")
    (tmp_path / "ok").mkdir()
    assert _sdist_check(_mk_sdist(tmp_path / "ok", stage), stage) == 0
    for i, kw in enumerate([
        {"drop": "summicron/a.py"}, {"extra": "stray.txt"}, {"mutate": "pyproject.toml"},
        {"info": "Name: isummicron\nVersion: 0.0.1\n"},
        {"info": f"Name: isummicron\nVersion: {V}\nAuthor-email: a <a@b.c>\n"},
    ]):
        d = tmp_path / f"bad{i}"
        d.mkdir()
        assert _sdist_check(_mk_sdist(d, stage, **kw), stage) != 0, kw


# ---------------------------------------------------------------- gallery refusals (docs round)
def test_gallery_preview_marker_refused(w):
    g = w.ws / "assets" / "GALLERY.txt"
    g.write_text(g.read_text().replace(f"version: {V}\n", f"version: {V}\nPREVIEW: not for publish\n"))
    r = w.run("--dry-run")
    assert r.returncode == 4 and "PREVIEW" in r.stderr, r.stderr


def test_gallery_jpeg_with_metadata_refused(w):
    jp = w.ws / "assets" / GALLERY_JPEGS[0]
    subprocess.run(["exiftool", "-q", "-overwrite_original", "-Artist=someone", str(jp)], check=True)
    g = w.ws / "assets" / "GALLERY.txt"  # keep the md5 line honest so only the metadata check can refuse
    g.write_text(f"version: {V}\n" + "".join(
        f"{n} md5: {w.md5(w.ws / 'assets' / n)}\n" for n in GALLERY_JPEGS) + "gallery post-checks (i)-(v): all passed\n")
    r = w.run("--dry-run")
    assert r.returncode == 4 and "metadata" in r.stderr, r.stderr


def test_gallery_missing_or_tampered_refused(w):
    (w.ws / "assets" / GALLERY_JPEGS[1]).write_bytes(b"x")
    r = w.run("--dry-run")
    assert r.returncode == 4 and "GALLERY.txt" in r.stderr, r.stderr
    (w.ws / "assets" / GALLERY_JPEGS[1]).unlink()
    r = w.run("--dry-run")
    assert r.returncode == 4 and "missing" in r.stderr, r.stderr


def test_notes_source_tamper_refused(w):
    (w.ws / "notes-src.md").write_text(NOTES_SRC + "\nextra\n")
    r = w.run("--dry-run")
    assert r.returncode == 4 and "notes-src" in r.stderr, r.stderr


# ── release date: heading, commit message and publisher agree ────────────────
def test_date_publish_heading_commit_mismatch_refused(w):
    chg = (w.clone / "CHANGELOG.md").read_text().replace("## [9.9.9] - 2026-01-03", "## [9.9.9] - 2026-01-04")
    (w.clone / "CHANGELOG.md").write_text(chg)
    _git("add", "-A", cwd=w.clone)
    _git("commit", "-q", "--amend", "-m", f"release: {V} (2026-01-03)", cwd=w.clone, env=_ident())
    r = w.run("--dry-run")
    assert r.returncode != 0 and "heading date" in r.stderr, (r.returncode, r.stderr)


def test_date_publish_later_day_warns_and_continues(w):
    r = w.run("--dry-run")  # the fixture is dated 2026-01-03, long before today
    assert r.returncode == 0, r.stderr + r.stdout
    assert "WARNING" in r.stderr and "2026-01-03" in r.stderr and "RELEASE_DATE=" in r.stderr
    import datetime
    assert datetime.date.today().isoformat() in r.stderr
