"""R-release-build / R-release-0.4 — Adversarial round (privacy gate holes, shipped-tree leaks).

Two halves:

* Synthetic trees (always run): probe ``scripts/release/privacy_gate.py`` with
  seeds the coder's self-test does not cover — the structural npz scan (zip and
  npy slack, which no md5 pin can excuse), JSON / YAML decoded strings, the exact
  patent URL, draft markers, structured-dtype / UTF-16 / IPv6 encodings, a pin
  that does not match the tensor, the build script's own guards.
* Built artefacts (skip when the scratch build root is absent): the staged
  public clone, its release commit, the no-push state and the stripped sample
  images.

Nothing here hard-codes a tensor md5, a tensor file name, a kernel size or a
version: the default tensor is resolved from ``summicron/render/_artifacts.py``
the way the gate does it.

Secrets are assembled from pieces so this file passes ``.githooks/pre-commit``.
GAP xfails document defence-in-depth holes that are masked by another check
(the gate still exits non-zero); BUG xfails are real defects.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
REL = ROOT / "scripts" / "release"
if not (REL / "privacy_gate.py").is_file():
    pytest.skip("scripts/release not present (public tree)", allow_module_level=True)
sys.path.insert(0, str(REL))
import _selftest_seeds as seeds  # noqa: E402
import privacy_gate as pg  # noqa: E402

NPZ_NAME = pg.default_npz_name(ROOT)
NPZ_SRC = ROOT / pg.ARTIFACT_DIR / (NPZ_NAME or "missing.npz")
NPZ_REL = f"{pg.ARTIFACT_DIR}/{NPZ_NAME}"
MOUNT = "/Vol" + "umes/Da" + "ta/x"
PATENT_URL = "https://patentimages.storage.googleapis.com/95/cd/35/d5b49e0c357612/US3006249.pdf"
# Built artefacts live outside every repo: set ISUMMICRON_RELEASE_ROOT to the release
# root (this file ships in tests/, so no local path is hard-coded here).
_RR = os.environ.get("ISUMMICRON_RELEASE_ROOT")
BUILD = Path(_RR) if _RR else None
CLONE = (BUILD / "public-clone") if BUILD else None
# The stripped assets sit in <root>/<12-hex-sha>/assets; take the newest one, no hard-coded SHA.
_asset_dirs = sorted(BUILD.glob("*/assets"), key=lambda p: p.stat().st_mtime) if BUILD and BUILD.is_dir() else []
ASSETS = _asset_dirs[-1] if _asset_dirs else None
_NO_ROOT = "ISUMMICRON_RELEASE_ROOT not set or not a build root (no public-clone)"
OVERLAY = ROOT / "release" / "public-overlay"
GATE_0_3 = "a87c565"   # the 0.3.0 release-tooling commit (its gate must NOT catch the new seeds)


@pytest.fixture(scope="module")
def tokens():
    return pg.build_token_patterns(ROOT)


def _md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def _tree(tmp: Path, npz: Path | None = None, pin: str | None = None, repin: bool = False,
          extra_pins: dict | None = None) -> Path:
    """A minimal stage: the default tensor, ``_artifacts.py`` (name) and ``_bench.py`` (pins)."""
    t = tmp / "tree"
    (t / "summicron/render").mkdir(parents=True)
    (t / "summicron/_artifacts").mkdir(parents=True)
    src = npz or NPZ_SRC
    shutil.copyfile(src, t / NPZ_REL)
    (t / pg.ARTIFACTS_PY).write_text(f'_ARTIFACT_NAME = "{NPZ_NAME}"\n')
    m = pin or (_md5(src) if repin else _md5(NPZ_SRC))
    lines = f'TENSOR_MD5 = "{m}"\n'
    if extra_pins is not None:
        lines += f"EXTRA_ARTIFACT_MD5 = {extra_pins!r}\n"
    (t / pg.BENCH_PY).write_text(lines)
    return t


def _checks(rep) -> set[str]:
    return {ln.split("]")[0].split("[")[1] for ln in rep.fails}


def _run_gate(t: Path, tokens):
    rep, _ = pg.check_tree(t, tokens, wheel=False)
    return rep


def _npz_with_member_slack(dst: Path, member: str, junk: bytes) -> Path:
    seeds.rebuild_npz_with_slack(NPZ_SRC, dst, member, junk)
    return dst


# ── synthetic: pins and the structural npz scan ──────────────────────────────
def test_shipped_npz_has_a_chance_token_in_payload(tokens):
    """Confirms the premise: the float payload holds a chance NAS-token hit (skipped by TYPE now)."""
    raw = NPZ_SRC.read_bytes().decode("latin-1")
    hits = [lbl for lbl, pat in tokens if pat.search(raw)]
    assert hits == ["nas-name"], hits


def test_pinned_npz_control_passes_content_checks(tmp_path, tokens):
    rep = _run_gate(_tree(tmp_path), tokens)
    assert not (_checks(rep) & {"P1", "P5", "P6", "P7"}), rep.fails


def test_npz_with_trailing_mount_path_fails_p6_even_when_repinned(tmp_path, tokens):
    """The defence no longer rests on the md5: appended bytes are zip slack (P6) with the pin UPDATED."""
    bad = tmp_path / "bad.npz"
    bad.write_bytes(NPZ_SRC.read_bytes() + MOUNT.encode())
    rep = _run_gate(_tree(tmp_path, bad, repin=True), tokens)
    assert "P1" not in _checks(rep), rep.fails          # the pin matches
    assert any("[P6]" in f and "zip slack" in f for f in rep.fails), rep.fails
    assert any("mount-path" in f for f in rep.fails), rep.fails


def test_npz_with_npy_slack_fails_p6_even_when_repinned(tmp_path, tokens):
    bad = _npz_with_member_slack(tmp_path / "bad.npz", "kernels.npy", MOUNT.encode())
    rep = _run_gate(_tree(tmp_path, bad, repin=True), tokens)
    assert "P1" not in _checks(rep), rep.fails
    assert any("[P6]" in f and "npy slack" in f for f in rep.fails), rep.fails


def test_npz_with_non_npy_member_fails_p6(tmp_path, tokens):
    bad = tmp_path / "bad.npz"
    with zipfile.ZipFile(NPZ_SRC) as zi, zipfile.ZipFile(bad, "w") as zo:
        for info in zi.infolist():
            zo.writestr(info.filename, zi.read(info))
        zo.writestr("readme.txt", b"hello")
    rep = _run_gate(_tree(tmp_path, bad, repin=True), tokens)
    assert any("non-npy member" in f for f in rep.fails), rep.fails


def test_npz_with_zip_comment_fails_p6(tmp_path, tokens):
    bad = tmp_path / "bad.npz"
    with zipfile.ZipFile(NPZ_SRC) as zi, zipfile.ZipFile(bad, "w") as zo:
        for info in zi.infolist():
            zo.writestr(info.filename, zi.read(info))
        zo.comment = b"note"
    rep = _run_gate(_tree(tmp_path, bad, repin=True), tokens)
    assert any("[P6]" in f and "comment" in f for f in rep.fails), rep.fails


def test_rebuilt_npz_with_unicode_key_fails_p6(tmp_path, tokens):
    with np.load(NPZ_SRC, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    d["source"] = np.array(MOUNT)
    bad = tmp_path / "bad.npz"
    np.savez_compressed(bad, **d)
    rep = _run_gate(_tree(tmp_path, bad), tokens)           # not re-pinned
    assert {"P1", "P6"} <= _checks(rep), rep.fails
    rep = _run_gate(_tree(tmp_path / "again", bad, repin=True), tokens)   # re-pinned: still P6
    assert "P1" not in _checks(rep) and "P6" in _checks(rep), rep.fails


def test_bench_pin_not_matching_npz_fails_p1(tmp_path, tokens):
    rep = _run_gate(_tree(tmp_path, pin="0123456789abcdef0123456789abcdef"), tokens)
    assert "P1" in _checks(rep), rep.fails


def test_new_tensor_with_matching_pin_passes_gate(tmp_path, tokens):
    """A re-saved tensor plus the matching _bench pin is accepted by the gate (no P1, no P6)."""
    other = tmp_path / "other.npz"
    with np.load(NPZ_SRC, allow_pickle=False) as z:
        np.savez(other, **{k: z[k] for k in z.files}, note=np.array("plain text"))
    rep = _run_gate(_tree(tmp_path, other, repin=True), tokens)
    assert not (_checks(rep) & {"P1", "P5", "P6", "P7"}), rep.fails


def test_unpinned_second_npz_fails_p1_and_pin_in_extra_artifact_md5_passes(tmp_path, tokens):
    t = _tree(tmp_path)
    extra = t / pg.ARTIFACT_DIR / "side.npz"
    np.savez(extra, a=np.zeros(4, dtype=np.float32))
    assert "P1" in _checks(_run_gate(t, tokens))
    (t / pg.BENCH_PY).write_text((t / pg.BENCH_PY).read_text()
                                 + f'EXTRA_ARTIFACT_MD5 = {{"side.npz": "{_md5(extra)}"}}\n')
    rep = _run_gate(t, tokens)
    assert not (_checks(rep) & {"P1", "P5", "P6"}), rep.fails


def test_unpinned_copy_of_shipped_npz_fails_only_location_not_content(tmp_path, tokens):
    """The chance NAS token in the float payload must not turn a location error into P5/P6 noise."""
    t = _tree(tmp_path)
    (t / "tests/fixtures").mkdir(parents=True)
    shutil.copyfile(NPZ_SRC, t / "tests/fixtures/copy.npz")
    rep = _run_gate(t, tokens)
    assert "P1" in _checks(rep)
    assert not (_checks(rep) & {"P5", "P6"}), rep.fails


def test_structured_dtype_string_in_npz_is_content_scanned(tmp_path, tokens):
    arr = np.array([(MOUNT, 1)], dtype=[("p", "U40"), ("n", "i4")])
    f = tmp_path / "s.npz"
    np.savez(f, meta=arr)
    rep = pg.Report()
    pg.check_npz(rep, f, "s.npz", tokens)
    assert any("mount-path" in x for x in rep.fails), rep.fails


def test_default_tensor_name_is_resolved_not_hard_coded(tmp_path, tokens):
    """The gate follows ``_ARTIFACT_NAME``: a renamed default tensor passes when pinned."""
    t = _tree(tmp_path)
    (t / NPZ_REL).rename(t / pg.ARTIFACT_DIR / "renamed.npz")
    (t / pg.ARTIFACTS_PY).write_text('_ARTIFACT_NAME = "renamed.npz"\n')
    rep = _run_gate(t, tokens)
    assert not (_checks(rep) & {"P1", "P5", "P6"}), rep.fails


# ── synthetic: json / yaml / url / draft markers ─────────────────────────────
def _lens(t: Path, name: str, body: str | bytes) -> None:
    d = t / "scripts/s24/lenses"
    d.mkdir(parents=True, exist_ok=True)
    (d / name).write_bytes(body if isinstance(body, bytes) else body.encode())


def test_lens_json_control_with_allowlisted_citation_passes(tmp_path, tokens):
    t = _tree(tmp_path)
    _lens(t, "ok_lens.json", json.dumps({"citation": f"US 3,006,249 (Mandler), Example II. {PATENT_URL}"}))
    rep = _run_gate(t, tokens)
    assert not (_checks(rep) & {"P1", "P2", "P5", "P6"}), rep.fails


def test_lens_json_sibling_patent_url_fails_exact_url_rule(tmp_path, tokens):
    t = _tree(tmp_path)
    _lens(t, "bad.json", json.dumps({"c": PATENT_URL.replace("US3006249", "US3006250")}))
    assert "P5" in _checks(_run_gate(t, tokens))


def test_lens_json_escaped_mount_path_is_caught(tmp_path, tokens):
    t = _tree(tmp_path)
    esc = "".join(f"\\u{ord(c):04x}" if c in "/Vu" else c for c in MOUNT)
    _lens(t, "bad.json", '{"note": "' + esc + '"}')
    rep = _run_gate(t, tokens)
    assert any("mount-path" in f for f in rep.fails), rep.fails


def test_json_outside_lens_dir_and_unparsable_json_fail_p1(tmp_path, tokens):
    t = _tree(tmp_path)
    (t / "summicron/x.json").write_text("{}")
    _lens(t, "trunc.json", '{"a": ')
    _lens(t, "nan.json", '{"a": NaN}')
    rep = _run_gate(t, tokens)
    p1 = [f for f in rep.fails if "[P1]" in f]
    assert any("x.json" in f for f in p1) and any("trunc.json" in f for f in p1) and any("nan.json" in f for f in p1), p1


def test_oversize_json_fails_p2(tmp_path, tokens):
    t = _tree(tmp_path)
    _lens(t, "big.json", '{"pad": "' + "abc " * (75 * 1024) + '"}')
    assert "P2" in _checks(_run_gate(t, tokens))


def test_json_bad_file_name_fails_p1(tmp_path, tokens):
    t = _tree(tmp_path)
    _lens(t, "Bad-Name.json", "{}")
    assert "P1" in _checks(_run_gate(t, tokens))


def test_yaml_escaped_mount_path_is_caught(tmp_path, tokens):
    t = _tree(tmp_path)
    (t / "tests").mkdir()
    (t / "tests/seed.yml").write_text('note: "\\x2FVol\\x75mes/Da\\x74a/x"\n')
    rep = _run_gate(t, tokens)
    assert any("mount-path" in f for f in rep.fails), rep.fails


def test_yaml_conda_prefix_is_caught(tmp_path, tokens):
    t = _tree(tmp_path)
    (t / "environment-optiland.yml").write_text("name: optiland\nprefix: /opt/conda/envs/optiland\n")
    rep = _run_gate(t, tokens)
    assert any("conda prefix" in f for f in rep.fails), rep.fails


def test_yaml_without_prefix_passes(tmp_path, tokens):
    t = _tree(tmp_path)
    (t / "environment-optiland.yml").write_text("name: optiland\ndependencies:\n  - numpy\n")
    assert not (_checks(_run_gate(t, tokens)) & {"P1", "P5"})


def test_exact_patent_url_is_allowed_but_its_host_is_not(tmp_path, tokens):
    t = _tree(tmp_path)
    (t / "notes.md").write_text(f"Patent: {PATENT_URL}\n")
    assert "P5" not in _checks(_run_gate(t, tokens))
    (t / "notes.md").write_text("Other: https://" + "patentimages.storage.googleapis.com/00/00/00/abc/other.pdf\n")
    rep = _run_gate(t, tokens)
    assert any("url-not-allowlisted" in f for f in rep.fails), rep.fails


@pytest.mark.parametrize("marker", ["DRAFT: text", "TBD", "a TODO here", "<!-- hidden -->"])
def test_draft_markers_in_shipped_documents_fail_p11(tmp_path, tokens, marker):
    for name in ("README.md", "CHANGELOG.md", "NOTICE"):
        t = _tree(tmp_path / name.replace(".", "_"))
        (t / name).write_text(f"# t\n{marker}\n")
        assert "P11" in _checks(_run_gate(t, tokens)), name


def test_ordinary_prose_does_not_trip_p11(tmp_path, tokens):
    t = _tree(tmp_path)
    (t / "README.md").write_text("A drafted note about todo lists and tbd-free text.\n")
    assert "P11" not in _checks(_run_gate(t, tokens))


def test_runs_directory_forbidden_except_bench_readme(tmp_path, tokens):
    t = _tree(tmp_path)
    (t / "runs/vibe_bench").mkdir(parents=True)
    (t / "runs/vibe_bench/README.md").write_text("# bench\n")
    rep = _run_gate(t, tokens)
    assert not any("[P3]" in f and "runs/" in f for f in rep.fails), rep.fails
    (t / "runs/vibe_bench/other.jpg").write_text("x")
    rep = _run_gate(t, tokens)
    assert any("[P3]" in f and "runs/" in f for f in rep.fails), rep.fails


def test_utf16_text_is_scanned(tmp_path, tokens):
    t = _tree(tmp_path)
    (t / "notes.txt").write_bytes(MOUNT.encode("utf-16"))
    rep = _run_gate(t, tokens)
    assert any("mount-path" in x for x in rep.fails), rep.fails


def test_tailnet_ipv6_is_caught(tmp_path, tokens):
    t = _tree(tmp_path)
    (t / "notes.md").write_text("host fd7a:" + "115c:a1e0:" + ":12\n")
    assert "P5" in _checks(_run_gate(t, tokens))


def test_uppercase_ext_and_dotfile_rejected(tmp_path, tokens):
    t = _tree(tmp_path)
    (t / "a.PY").write_text("x\n")
    (t / ".env").write_text("x\n")
    rep = _run_gate(t, tokens)
    assert sum("P1" in f for f in rep.fails) >= 2, rep.fails


# ── the new self-test seeds prove the new logic (the 0.3.0 gate misses them) ──
@pytest.mark.parametrize("n", [29, 33, 34, 37, 38, 40, 41])
def test_new_seeds_not_caught_by_0_3_gate(tmp_path, n):
    r = subprocess.run(["git", "-C", str(ROOT), "show", f"{GATE_0_3}:scripts/release/privacy_gate.py"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        pytest.skip(f"{GATE_0_3} not in this checkout")
    old = tmp_path / "old_gate.py"
    old.write_text(r.stdout)
    t = _tree(tmp_path / "t")
    (t / "CHANGELOG.md").write_text("# Changelog\n")
    seeds.inject(n, t, ROOT)
    out = subprocess.run([sys.executable, str(old), str(t), "--dev-repo", str(ROOT)],
                         capture_output=True, text=True)
    want = seeds.EXPECT[n].split(",")
    missed = [w for w in want if f"[{w}]" not in out.stdout]
    assert out.returncode == 0 or missed, (n, out.stdout[-400:])


def test_seed_count_matches_expect_table():
    assert seeds.COUNT == len(seeds.EXPECT) == 49


# ── static: build scripts ────────────────────────────────────────────────────
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


def test_release_scripts_hard_code_no_tensor_prefix_name_or_size():
    for p in REL.iterdir():
        if p.suffix not in (".sh", ".py"):
            continue
        txt = p.read_text()
        for bad in ("e6f50154", "8e982d13", "codev_render_v1", "TENSOR_PREFIX", "NPZ_REL"):
            assert bad not in txt, (p.name, bad)
        assert not re.search(r"\b183\b", txt), p.name


def test_fixture_ref_script_static_guards():
    f = REL / "fixture_ref.sh"
    txt = f.read_text()
    # every line that touches a publishing path carries the guard marker
    for ln in txt.splitlines():
        if re.search(r"(^|[^A-Za-z_])(git[^#]*[ \t]push|gh[ \t])", ln):
            assert "# nopush-guard" in ln, ln
    assert "push.default" in txt and "# nopush-guard" in txt
    # no hard-coded absolute location of this machine
    for lit in ("/Vol" + "umes", "/Us" + "ers/", "/priv" + "ate/", "/home/"):
        assert lit not in txt, lit


def test_fixture_ref_refuses_out_inside_a_repo(tmp_path):
    out = ROOT / "runs" / "__fixture_refuse_probe__"   # inside this repo, never created
    r = subprocess.run(["bash", str(REL / "fixture_ref.sh"), "--dev", str(ROOT), "--rel", str(ROOT),
                        "--out", str(out), "--profile", "a"], capture_output=True, text=True)
    assert r.returncode != 0 and "REFUSED" in (r.stderr + r.stdout), (r.returncode, r.stderr)
    assert not out.exists()


def test_fixture_ref_clean_refuses_without_manifest(tmp_path):
    (tmp_path / "x").mkdir()
    r = subprocess.run(["bash", str(REL / "fixture_ref.sh"), "--clean", str(tmp_path / "x")],
                       capture_output=True, text=True)
    assert r.returncode != 0 and (tmp_path / "x").exists()


# ── build_release.sh guards on a tiny scratch repo ───────────────────────────
def _git_c(repo: Path, *a: str) -> str:
    return subprocess.run(["git", "-C", str(repo), "-c", "user.name=t",
                           "-c", "user.email=t" + "@" + "users.noreply.github.com", *a],
                          capture_output=True, text=True, check=True).stdout.strip()


def _mini_repo(tmp: Path, overlay: bool = False) -> tuple[Path, str, dict]:
    repo = tmp / "repo"
    (repo / "scripts/release").mkdir(parents=True)
    for p in REL.iterdir():
        if p.is_file():
            shutil.copy2(p, repo / "scripts/release" / p.name)
    (repo / "summicron").mkdir()
    (repo / "summicron/__init__.py").write_text('__version__ = "0.0.0"\n')
    (repo / "tests").mkdir()
    (repo / "tests/test_x.py").write_text("def test_x():\n    pass\n")
    (repo / "scripts/clean_runs.py").write_text("X = 1\n")
    (repo / "scripts/q_paired_probe.py").write_text("X = 1\n")
    (repo / "scripts/gen_cli_reference.py").write_text("X = 1\n")
    (repo / "runs/vibe_bench").mkdir(parents=True)
    (repo / "runs/vibe_bench/README.md").write_text("# bench\n")
    (repo / "pyproject.toml").write_text('version = "0.0.0"\n')
    if overlay:
        (repo / "release/public-overlay").mkdir(parents=True)
        (repo / "release/public-overlay/README.md").write_text("# x\n")
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    _git_c(repo, "add", "-A")
    _git_c(repo, "commit", "-q", "-m", "x")
    sha = _git_c(repo, "rev-parse", "HEAD")
    pub = tmp / "pub"
    subprocess.run(["git", "init", "-q", str(pub)], check=True)
    _git_c(pub, "commit", "-q", "--allow-empty", "-m", "x")
    env = dict(os.environ, RELEASE_ROOT=str(tmp / "rel"), PUBLIC_SOURCE=str(pub))
    env.pop("RELEASE_AUTHOR", None)
    return repo, sha, env


def _build(repo: Path, sha: str, env: dict, *extra: str, script: Path | None = None):
    return subprocess.run(["bash", str(script or repo / "scripts/release/build_release.sh"), sha, "9.9.9", *extra],
                          cwd=repo, env=env, capture_output=True, text=True)


def test_build_without_overlay_reports_failed_at_b3_and_optional_paths_absent(tmp_path):
    repo, sha, env = _mini_repo(tmp_path)
    r = _build(repo, sha, env)
    assert r.returncode != 0 and "FAILED at B3" in r.stderr, (r.returncode, r.stderr[-500:])
    log = (tmp_path / "rel" / sha[:12] / "logs" / "B2.log").read_text()
    assert "optional scripts/s24: absent" in log and "optional environment-optiland.yml: absent" in log, log


def test_build_with_missing_mandatory_path_reports_failed_at_b2(tmp_path):
    repo, sha, env = _mini_repo(tmp_path)
    _git_c(repo, "rm", "-q", "scripts/q_paired_probe.py")
    _git_c(repo, "commit", "-q", "-m", "rm")
    r = _build(repo, _git_c(repo, "rev-parse", "HEAD"), env)
    assert r.returncode != 0 and "FAILED at B2" in r.stderr, (r.returncode, r.stderr[-500:])


def test_build_with_s24_payload_lists_it_as_present(tmp_path):
    repo, sha, env = _mini_repo(tmp_path)
    (repo / "scripts/s24").mkdir()
    (repo / "scripts/s24/_gates.py").write_text("X = 1\n")
    (repo / "scripts/s24/vibe_sheet.py").write_text("X = 1\n")
    _git_c(repo, "add", "-A")
    _git_c(repo, "commit", "-q", "-m", "s24")
    sha = _git_c(repo, "rev-parse", "HEAD")
    r = _build(repo, sha, env)
    assert "FAILED at B3" in r.stderr
    log = (tmp_path / "rel" / sha[:12] / "logs" / "B2.log").read_text()
    assert "optional scripts/s24: present" in log, log
    assert "s24 ships: scripts/s24/_gates.py" in log and "s24 NOT shipped" in log and "vibe_sheet.py" in log, log
    assert not (tmp_path / "rel" / sha[:12] / "stage" / "scripts/s24/vibe_sheet.py").exists()


def test_build_fails_at_g_tool_when_a_worktree_script_drifted_from_the_ref(tmp_path):
    repo, sha, env = _mini_repo(tmp_path)
    with (repo / "scripts/release/privacy_gate.py").open("a") as f:
        f.write("\n# edited after the ref was cut\n")
    r = _build(repo, sha, env)
    assert r.returncode != 0 and "FAILED at G-tool" in r.stderr, (r.returncode, r.stderr[-500:])


@pytest.mark.slow
def test_dry_run_lists_independent_failures_and_never_writes_the_marker(tmp_path):
    repo, sha, env = _mini_repo(tmp_path)          # no overlay (B3) and no commit-summary (G-msg)
    r = _build(repo, sha, env, "--dry-run")
    w = tmp_path / "rel" / sha[:12]
    verdicts = (w / "dryrun-verdicts.txt").read_text().splitlines()
    assert r.returncode == 1, (r.returncode, r.stderr[-500:])
    assert "B3 FAIL" in verdicts and "G-msg FAIL" in verdicts, verdicts
    assert not (w / "gates-passed").exists()


# ── overlay docs ─────────────────────────────────────────────────────────────
@pytest.mark.skipif(not OVERLAY.is_dir(), reason="no overlay")
def test_overlay_disclaimer_and_lens_facts():
    disc = ("Summicron and Leica are trademarks of Leica Camera AG. iSummicron is not "
            "affiliated with, sponsored by, or endorsed by Leica Camera AG.")
    for f in ("README.md", "NOTICE"):
        assert disc in (OVERLAY / f).read_text(), f
    rd = (OVERLAY / "README.md").read_text()
    # derived from the default tensor, never typed in: shape string + provenance
    with np.load(NPZ_SRC, allow_pickle=False) as z:
        R, D, K = z["kernels"].shape[0], z["kernels"].shape[1], z["kernels"].shape[2]
        cited = "lens_citation" in z.files
    assert f"(R={R} radial bins, D={D} distances, {K}, {K})" in rd
    if cited:
        assert "3,006,249" in rd and "optiland" in rd
    else:
        assert "ag_dblgauss" in rd and "~100 mm f/2" in rd
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
                            "--", ".", ":!summicron/_artifacts/*.npz"],
                           capture_output=True, check=True).stdout.decode("latin-1")
    hits = sorted({lbl for lbl, pat in tokens if pat.search(patch)})
    assert hits == [], hits
    body = re.sub(r"(?m)^[+ -]", "", patch)  # drop diff markers ("+@pytest" is not an email)
    emails = {m for m in pg.EMAIL_RE.findall(body) if not pg.NOREPLY_ADDR_RE.match(m)}
    assert not emails


@needs_clone
def test_clone_npz_is_pinned_bytes():
    assert _md5(CLONE / NPZ_REL) == _md5(NPZ_SRC)


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


# ── --commit refuses without a matching all-gates marker / ack ───────────────
def _overlay_version() -> str:
    m = re.search(r"^## \[([^\]]+)\]", (OVERLAY / "CHANGELOG.md").read_text(), re.M)
    return m.group(1) if m else "0.0.0"


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
    (w / "stage" / "CHANGELOG.md").write_text(f"# Changelog\n\n## [{_overlay_version()}] \u2014 2026-01-01\n")
    (w / "tree.sha").write_text("0" * 40 + "\n")
    env = dict(os.environ, RELEASE_ROOT=str(root), PUBLIC_SOURCE=str(pub))
    env.pop("RELEASE_AUTHOR", None)
    env.pop("RELEASE_ACK_ARTIFACTS", None)
    return sha, w, env


def _run_commit(sha, env):
    return subprocess.run(["bash", str(REL / "build_release.sh"), sha, _overlay_version(), "--commit"],
                          cwd=ROOT, env=env, capture_output=True, text=True)


def _stage_tree_hash(tmp_path: Path, stage: Path) -> str:
    gd = tmp_path / "gd"
    subprocess.run(["git", "init", "-q", str(gd)], check=True)
    env = dict(os.environ, GIT_DIR=str(gd / ".git"), GIT_WORK_TREE=str(stage))
    subprocess.run(["git", "-c", "core.excludesFile=/dev/null", "add", "-A", "--force"], env=env, check=True)
    return subprocess.run(["git", "write-tree"], env=env, capture_output=True, text=True, check=True).stdout.strip()


def test_commit_refused_without_gates_marker(tmp_path):
    sha, w, env = _commit_env(tmp_path)
    r = _run_commit(sha, env)
    assert r.returncode != 0 and "gates-passed" in r.stderr, (r.returncode, r.stderr)


def test_commit_refused_when_marker_hash_differs_from_stage(tmp_path):
    sha, w, env = _commit_env(tmp_path)
    (w / "gates-passed").write_text("f" * 40 + "\n")
    r = _run_commit(sha, env)
    assert r.returncode != 0 and "marker hash" in r.stderr, (r.returncode, r.stderr)


def test_commit_refused_when_commit_message_differs_from_the_approved_one(tmp_path):
    sha, w, env = _commit_env(tmp_path)
    (w / "commit-msg.txt").write_text("release: x\n")
    (w / "gates-passed").write_text(_stage_tree_hash(tmp_path, w / "stage") + "\n" + "0" * 32 + "\n")
    r = _run_commit(sha, env)
    assert r.returncode != 0 and "commit-msg" in r.stderr, (r.returncode, r.stderr)


def _marker(tmp_path, w, changes="", version=None):
    """Write a complete 4-line gates-passed marker for the fake workspace."""
    (w / "commit-msg.txt").write_text("release: x\n")
    (w / "artifact-changes.txt").write_text(changes)
    lines = [_stage_tree_hash(tmp_path, w / "stage"),
             hashlib.md5((w / "commit-msg.txt").read_bytes()).hexdigest(),
             hashlib.md5((w / "artifact-changes.txt").read_bytes()).hexdigest(),
             version or _overlay_version()]
    (w / "gates-passed").write_text("\n".join(lines) + "\n")


def test_commit_refuses_unacked_artifact_change(tmp_path):
    """G-ack: a changed shipped artifact needs RELEASE_ACK_ARTIFACTS to list exactly <path>=<md5>."""
    sha, w, env = _commit_env(tmp_path)
    path = "summicron/_artifacts/new.npz"
    _marker(tmp_path, w, f"{path}={'a' * 32}\n")
    env = dict(env, RELEASE_AUTHOR="t <1+chuang44-tiff@users.noreply.github.com>")
    r = _run_commit(sha, env)
    assert r.returncode != 0 and "not acknowledged" in r.stderr, (r.returncode, r.stderr)
    r = _run_commit(sha, dict(env, RELEASE_ACK_ARTIFACTS=f"{path}={'b' * 32}"))
    assert r.returncode != 0 and "not acknowledged" in r.stderr, (r.returncode, r.stderr)
    r = _run_commit(sha, dict(env, RELEASE_ACK_ARTIFACTS=""))
    assert r.returncode != 0 and "not acknowledged" in r.stderr, (r.returncode, r.stderr)


def test_commit_refuses_emptied_artifact_changes_file(tmp_path):
    """G-ack cannot be bypassed by emptying artifact-changes.txt: its md5 is bound into the marker."""
    sha, w, env = _commit_env(tmp_path)
    _marker(tmp_path, w, f"summicron/_artifacts/new.npz={'a' * 32}\n")
    (w / "artifact-changes.txt").write_text("")
    env = dict(env, RELEASE_AUTHOR="t <1+chuang44-tiff@users.noreply.github.com>")
    r = _run_commit(sha, env)
    assert r.returncode != 0 and "artifact-changes.txt differs" in r.stderr, (r.returncode, r.stderr)


def test_commit_refuses_version_not_bound_to_the_build(tmp_path):
    sha, w, env = _commit_env(tmp_path)
    _marker(tmp_path, w, version="9.9.9")
    env = dict(env, RELEASE_AUTHOR="t <1+chuang44-tiff@users.noreply.github.com>")
    r = _run_commit(sha, env)
    assert r.returncode != 0 and "VERSION" in r.stderr and "9.9.9" in r.stderr, (r.returncode, r.stderr)


@pytest.mark.parametrize("head", ["## [{v}] \u2014 YYYY-MM-DD", "## [{v}] \u2014 unreleased"])
def test_commit_refuses_placeholder_changelog_heading(tmp_path, head):
    sha, w, env = _commit_env(tmp_path)
    (w / "stage" / "CHANGELOG.md").write_text("# Changelog\n\n" + head.format(v=_overlay_version()) + "\n")
    _marker(tmp_path, w)
    env = dict(env, RELEASE_AUTHOR="t <1+chuang44-tiff@users.noreply.github.com>")
    r = _run_commit(sha, env)
    assert r.returncode != 0 and "placeholder" in r.stderr, (r.returncode, r.stderr)


@pytest.mark.parametrize("planted", [
    'subprocess.run(["git", "pu' + 'sh", "x"])',
    'subprocess.run(["g' + 'h", "pr", "list"])',
    'command git remote ' + 'add x y',
    '"$P" --no-' + 'verify',
])
def test_static_nopush_check_catches_real_push_forms(tmp_path, planted):
    """The G-nopush static grep (same regex as build_release.sh) must flag every planted form."""
    sh = (REL / "build_release.sh").read_text()
    func = sh[sh.index("nopush_static() {"):]
    func = func[:func.index("\n}\n") + 3]
    d = tmp_path / "here"
    d.mkdir()
    (d / "planted.py").write_text(planted + "\n")
    r = subprocess.run(["bash", "-c", f'HERE="{d}"\n{func}\nnopush_static'], capture_output=True, text=True)
    assert r.returncode == 1 and "static grep found" in r.stdout, (planted, r.stdout, r.stderr)
    (d / "planted.py").write_text("x = 1\n")
    r = subprocess.run(["bash", "-c", f'HERE="{d}"\n{func}\nnopush_static'], capture_output=True, text=True)
    assert "static grep found" not in r.stdout, r.stdout


# ── fix round 1: npz zip-structure holes, dot-directory GPS, duplicate keys, schemeless share links ──
@pytest.mark.parametrize("n", [43, 44, 45, 46, 47, 48, 49])
def test_fix_round_seeds_are_caught_with_the_right_ids(tmp_path, tokens, n):
    t = _tree(tmp_path)
    (t / "CHANGELOG.md").write_text("# Changelog\n")
    seeds.inject(n, t, ROOT)
    got = _checks(_run_gate(t, tokens))
    for want in seeds.EXPECT[n].split(","):
        assert want in got, (n, got)


def test_unseeded_tree_has_no_fix_round_findings(tmp_path, tokens):
    t = _tree(tmp_path)
    (t / "CHANGELOG.md").write_text("# Changelog\n")
    rep = _run_gate(t, tokens)
    assert not rep.fails, rep.fails


class _Unseekable(io.RawIOBase):
    """Forces the zip writer to emit real data descriptors (flag bit 3)."""

    def __init__(self):
        self.buf = bytearray()

    def writable(self):
        return True

    def write(self, b):
        self.buf += bytes(b)
        return len(b)


def test_legitimate_zip_variants_are_not_flagged(tmp_path, tokens):
    arr = np.arange(12, dtype=np.float32).reshape(3, 4)
    plain, comp, desc = tmp_path / "a.npz", tmp_path / "b.npz", tmp_path / "c.npz"
    np.savez(plain, a=arr)
    np.savez_compressed(comp, a=arr)
    u = _Unseekable()
    with zipfile.ZipFile(u, "w", zipfile.ZIP_DEFLATED) as zf:
        bio = io.BytesIO()
        np.save(bio, arr)
        zf.writestr("a.npy", bio.getvalue())
    desc.write_bytes(bytes(u.buf))
    assert zipfile.ZipFile(desc).infolist()[0].flag_bits & 0x08, "fixture must use a data descriptor"
    for f in (plain, comp, desc):
        rep = pg.Report()
        pg.check_npz(rep, f, f.name, tokens)
        assert not rep.fails, (f.name, rep.fails)


def test_gps_in_dot_directory_is_found(tmp_path, tokens):
    t = _tree(tmp_path)
    seeds.gps_jpeg(t / "plain.bin")
    assert "P4" in _checks(_run_gate(t, tokens))
    (t / "plain.bin").unlink()
    seeds.gps_jpeg(t / ".hidden_dir/img.bin")
    assert "P4" in _checks(_run_gate(t, tokens))


def test_schemeless_claude_ai_links_are_caught(tmp_path, tokens):
    for tail in ("share/abc", "chat/abc", "code/session"):
        t = _tree(tmp_path / tail.replace("/", "_"))
        (t / "notes.md").write_text("see claude" + ".ai/" + tail + "\n")
        assert "P5" in _checks(_run_gate(t, tokens)), tail
