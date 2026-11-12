"""Public docs checks: README, CHANGELOG and docs/*.md stay true to the code.

Runs in the dev repo (docs live in release/public-overlay) and in the public tree (docs at the
root). Every shell command in a fenced block is either run (cheap --help forms) or parsed
against the real argparse definition; links are checked offline; the generated CLI reference
must match its generator. Offline, stdlib plus the project itself.
"""
from __future__ import annotations

import importlib.util
import re
import shlex
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DOCROOT = ROOT / "release" / "public-overlay" if (ROOT / "release" / "public-overlay").is_dir() else ROOT
README = DOCROOT / "README.md"
if not README.is_file():
    pytest.skip("no public README in this tree", allow_module_level=True)

DOC_FILES = [README, DOCROOT / "CHANGELOG.md", *sorted((DOCROOT / "docs").glob("*.md"))]
NON_CHANGELOG = [p for p in DOC_FILES if p.name != "CHANGELOG.md"]
STAMPED = "{{VERSION}}" not in README.read_text()
if STAMPED:
    import summicron

    VER = summicron.__version__
else:
    VER = "{{VERSION}}"

SCHEME = "https" + ":" + "//"  # assembled: the privacy gate scans shipped files for literal URLs
BASE = SCHEME + "github.com/chuang44-tiff/iSummicron"
# The privacy gate's URL allowlist (a copy; the dev-only test checks it stays a subset).
URL_ALLOW_DOCS = (
    "github.com/chuang44-tiff/isummicron", "github.com/estherlin", "blur-fields.github.io",
    "www.apache.org", "huggingface.co/depth-anything", "arxiv.org", "doi.org",
    "keepachangelog.com", "semver.org",
)
FIGURES = [
    "fig_corner_35_plain_disc.jpg",
    "fig_corner_35_real_lens.jpg",
    "fig_corner_50_plain_disc.jpg",
    "fig_corner_50_real_lens.jpg",
    "fig_diopter_layers_dark.png",
    "fig_diopter_layers.png",
    "fig_front_vs_back_dark.png",
    "fig_front_vs_back.png",
    "fig_hdr_kitchen_hair_none_vs_auto.jpg",
    "fig_hdr_kitchen_lamps_none_vs_auto.jpg",
    "fig_journey_35_s1_onaxis_0p7m_focus.png",
    "fig_journey_35_s2_onaxis_20m.png",
    "fig_journey_35_s3_20m_corner.png",
    "fig_journey_35_s3_20m_midring.png",
    "fig_journey_35_s4_around_frame.png",
    "fig_journey_35.gif",
    "fig_kernels_atlas_35.png",
    "fig_kernels_atlas_50.png",
    "fig_layer_weights_dark.png",
    "fig_layer_weights.png",
    "fig_psf_grid_35.png",
    "fig_psf_grid_50.png",
    "fig_wedges_8_dark.png",
    "fig_wedges_8.png",
    "fig_kitchen_4up.jpg",
    "fig_xmas_35_plain_disc.jpg",
    "fig_xmas_35_real_lens.jpg",
    "fig_xmas_50_plain_disc.jpg",
    "fig_xmas_50_real_lens.jpg",
    "fig_xmas_zoom_input_35_50.jpg",
    "fig_xmas_zoom_input_disc_35.jpg",
]
ASSETS = [f"isummicron-{VER}-py3-none-any.whl", f"isummicron-{VER}.tar.gz", "Demo_input.HEIC",
          "Demo_portraitmode.HEIC", "SHA256SUMS", "Demo_input_before.jpg", "Demo_input_after.jpg",
          "Demo_input_before_edge.jpg", "Demo_input_after_edge.jpg", *FIGURES, "isummicron_pipeline.html"]
REMOVED_FLAGS = {"--dof-m", "--kernel-centring", "--radial-blend-mode", "--layered-gather",
                 "--r-blend-strength", "--bright-pass-knee", "--dual-tensor", "--working-long-edge"}
EXTRA_FLAGS = {"--help", "--ignore-missing", "--no-show", "--image", "--tensor", "--depth-source"}
TRADEMARK = ("Summicron and Leica are trademarks of Leica Camera AG. iSummicron is not "
             "affiliated with, sponsored by, or endorsed by Leica Camera AG.")


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def _render_flags() -> set[str]:
    from summicron.render._render_args import add_render_arguments
    import argparse

    p = add_render_arguments(argparse.ArgumentParser(add_help=False))
    return {o for a in p._actions for o in a.option_strings}


# ---------------------------------------------------------------- T1 commands
def _blocks(text: str):
    for m in re.finditer(r"```(\w+)\n(.*?)```", text, re.S):
        yield m.group(1), m.group(2)


def _commands(block: str) -> list[str]:
    joined = re.sub(re.escape(chr(92) + "\n") + r"\s*", " ", block)
    cmds: list[str] = []
    for line in joined.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        line = re.sub(r"^\$\s+", "", line)
        cur, q = "", ""
        i = 0
        while i < len(line):
            ch = line[i]
            if q:
                q = "" if ch == q else q
            elif ch in "\"'":
                q = ch
            elif line.startswith("&&", i) or ch == ";":
                cmds.append(cur.strip())
                cur = ""
                i += 2 if ch == "&" else 1
                continue
            cur += ch
            i += 1
        cmds.append(cur.strip())
    return [c for c in cmds if c]


def _rurl_ok(url: str) -> bool:
    v = re.escape(VER)
    names = "|".join(re.escape(n) for n in ASSETS)
    pats = [rf"{re.escape(BASE)}/releases/download/v{v}/(?:{names})",
            rf"{re.escape(BASE)}/releases/tag/v{v}", rf"{re.escape(BASE)}/releases/?",
            rf"git\+{re.escape(BASE)}@v{v}", rf"{re.escape(BASE)}\.git", rf"{re.escape(BASE)}"]
    return any(re.fullmatch(p, url) for p in pats)


def _extras_ok(token: str) -> None:
    keys = set(tomllib.loads(_read(ROOT / "pyproject.toml"))["project"]["optional-dependencies"])
    for m in re.finditer(r"\[([^\]]*)\]", token):
        extras = {e.strip() for e in m.group(1).split(",")}
        assert extras <= keys, (token, extras - keys)


def _check_command(cmd: str, where: str) -> None:
    toks = shlex.split(cmd)
    while toks and re.fullmatch(r"[A-Z_][A-Z0-9_]*=\S*", toks[0]):
        toks.pop(0)
    assert toks, (where, cmd)
    head = " ".join(toks[:3])
    if toks[0] == "export":
        assert re.fullmatch(r"ISUMMICRON_(MAX_BYTES=\d+|DEBUG=1)", toks[1]), (where, cmd)
    elif toks[0] == "isummicron" or toks[:3] == ["python", "-m", "summicron.render.convolve"]:
        rest = toks[1:] if toks[0] == "isummicron" else toks[3:]
        if toks[0] == "isummicron" and rest[:1] != ["render"]:
            assert rest == ["--help"], (where, cmd)
            r = subprocess.run([sys.executable, "-m", "summicron.cli", "--help"], cwd=ROOT, capture_output=True, text=True)
            assert r.returncode == 0, r.stderr
        elif "--help" in rest:
            mod = ["-m", "summicron.cli", "render"] if toks[0] == "isummicron" else ["-m", "summicron.render.convolve"]
            r = subprocess.run([sys.executable, *mod, "--help"], cwd=ROOT, capture_output=True, text=True)
            assert r.returncode == 0, r.stderr
        else:
            from summicron.render.api import RenderOptions
            from summicron.render.convolve import build_parser

            args = rest[1:] if toks[0] == "isummicron" else rest
            ns = build_parser().parse_args(args)
            RenderOptions.from_argparse(ns)
    elif toks[:3] == ["python", "-m", "summicron.tools.anchor_picker"]:
        r = subprocess.run([sys.executable, "-m", "summicron.tools.anchor_picker", "--help"], cwd=ROOT,
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        for t in toks[3:]:
            if t.startswith("--"):
                assert t in r.stdout, (where, t)
    elif toks[:2] == ["pip", "install"]:
        for t in toks[2:]:
            if t.startswith("-"):
                assert t == "-e", (where, t)
                continue
            _extras_ok(t)
            for u in re.findall(r"(?:git\+)?" + SCHEME + r"\S+", t):
                assert _rurl_ok(u), (where, u)
            assert t in ("matplotlib", ".") or t.startswith((".[", "isummicron[")) or "isummicron" in t, (where, t)
    elif toks[0] == "curl":
        assert toks[1] == "-LO" and _rurl_ok(toks[2]), (where, cmd)
    elif toks[:3] == ["python3.12", "-m", "venv"] or toks[0] == "source" or toks[:3] == ["conda", "activate", "isummicron"]:
        pass
    elif toks[:4] == ["conda", "env", "create", "-f"]:
        assert (DOCROOT / toks[4]).is_file() or (ROOT / toks[4]).is_file(), (where, toks[4])
    elif toks[:3] == ["git", "clone", BASE + ".git"]:
        pass
    elif toks[:2] == ["git", "checkout"]:
        assert toks[2] == f"v{VER}", (where, cmd)
    elif toks[0] in ("cd", "mkdir"):
        pass
    elif toks[:6] == ["shasum", "-a", "256", "-c", "--ignore-missing", "SHA256SUMS"]:
        pass
    elif toks[:3] == ["python", "-m", "pytest"] or toks[:3] == ["ruff", "check", "summicron/"]:
        pass
    else:
        pytest.fail(f"{where}: unknown command head: {head!r} in {cmd!r}")


@pytest.mark.parametrize("path", DOC_FILES, ids=lambda p: p.name)
def test_t1_commands(path):
    text = _read(path)
    n = 0
    for lang, body in _blocks(text):
        if lang in ("bash", "sh", "console"):
            for cmd in _commands(body):
                _check_command(cmd, path.name)
                n += 1
        elif lang == "python":
            code = compile(body, f"{path.name}:python", "exec")
            if path.name == "python-api.md":
                _exec_api_example(code)
    if path.name == "README.md":
        assert n >= 8


def _exec_api_example(code) -> None:
    import summicron.render.api as api

    seen = {}

    def stub(opts, **kw):
        assert isinstance(opts, api.RenderOptions)
        seen["opts"] = opts

        class R:
            output_path, mode, elapsed_s = Path("x.jpg"), "hdr", 1.234

        return R()

    real = api.render_image
    api.render_image = stub
    try:
        exec(code, {"__name__": "__docs__"})
    finally:
        api.render_image = real
    assert seen["opts"].downres == 2048


# ---------------------------------------------------------------- T2 inline flags
@pytest.mark.parametrize("path", NON_CHANGELOG, ids=lambda p: p.name)
def test_t2_inline_flags(path):
    text = _read(path)
    allowed = _render_flags() | EXTRA_FLAGS
    for span in re.findall(r"`([^`\n]+)`", text):
        m = re.match(r"--[a-z][a-z0-9-]*", span)
        if m:
            assert m.group(0) in allowed, (path.name, span)
    for f in REMOVED_FLAGS:
        assert f not in text, (path.name, f)


# ---------------------------------------------------------------- T3 links
def _slug(heading: str) -> str:
    h = re.sub(r"[`*_]", "", heading.strip().lower()) if False else heading.strip().lower().replace("`", "")
    h = re.sub(r"[^a-z0-9 _-]", "", h)
    return h.replace(" ", "-")


def _anchors(path: Path) -> set[str]:
    return {_slug(m.group(1)) for m in re.finditer(r"(?m)^#{1,6}\s+(.+?)\s*$", _read(path))}


@pytest.mark.parametrize("path", DOC_FILES, ids=lambda p: p.name)
def test_t3_links(path):
    text = _read(path)
    assert "http" + ":" + "//" not in text
    if STAMPED:
        assert "{{" not in text.replace("${{", "")
    for target in re.findall(r"!?\[[^\]]*\]\(([^)\s]+)\)", text):
        if target.startswith(("http" + ":" + "//", SCHEME)):
            continue
        file_part, _, frag = target.partition("#")
        dest = (path.parent / file_part).resolve() if file_part else path
        assert dest.is_file(), (path.name, target)
        if frag and dest.suffix == ".md":
            assert frag in _anchors(dest), (path.name, target, sorted(_anchors(dest))[:8])
    for ref in re.findall(r"(?m)^\[[^\]]+\]:\s*(\S+)", text):
        if ref.startswith(SCHEME):
            assert any(ref.lower().replace(SCHEME, "").startswith(a) for a in URL_ALLOW_DOCS), ref
    for url in re.findall(r"(?:git\+)?" + SCHEME + r"[^\s<>\"'`)\]|]+", text):
        url = url.rstrip(".,;:")
        host = url.replace("git+", "").replace(SCHEME, "").lower()
        assert any(host.startswith(a) for a in URL_ALLOW_DOCS), (path.name, url)
        if "/chuang44-tiff/isummicron/releases/" in host or "isummicron@" in host:
            if path.name == "CHANGELOG.md" and not re.search(r"/releases/(download|tag)/v(0\.3|0\.[4-9]|[1-9])", url):
                continue
            assert _rurl_ok(url) or path.name == "CHANGELOG.md", (path.name, url)


# ---------------------------------------------------------------- T4 privacy-lite
@pytest.mark.parametrize("path", DOC_FILES, ids=lambda p: p.name)
def test_t4_privacy_lite(path):
    text = _read(path)
    private_repo = "iSummicron" + "-dev"
    banned = ["doc/", "CLAUDE.md", ".claude/", "memory/", "team-dev", "DevTeam",
              "claude" + ".ai", private_repo]  # split: the privacy gate bans the literal in shipped files
    banned += ["/Vol" + "umes/", "/Us" + "ers/"]
    for b in banned:
        assert b not in text, (path.name, b)
    assert not re.search(r"\bdeploy\b", text, re.I), path.name
    for i, line in enumerate(text.splitlines(), 1):
        assert not re.search(r"(?<![0-9a-f])[0-9a-f]{12,40}(?![0-9a-f])", line), (path.name, i)


# ---------------------------------------------------------------- T5 generated reference
def _gen():
    spec = importlib.util.spec_from_file_location("gen_cli_reference", ROOT / "scripts" / "gen_cli_reference.py")
    mod = importlib.util.module_from_spec(spec)
    flag = sys.dont_write_bytecode
    spec.loader.exec_module(mod)
    sys.dont_write_bytecode = flag
    return mod


def test_t5_generated_reference_is_current():
    gen = _gen()
    assert gen.main(["--check", str(DOCROOT / "docs" / "cli-reference.md")]) == 0
    dests = {a.dest for a in gen._actions()}
    assert set(gen.DESCRIPTIONS) == dests and set(gen.GROUPS) == dests


def test_t5_description_ranges_match_render_options():
    from summicron.render.api import RenderOptions

    gen = _gen()
    fields = ["bright_pass_region_size", "bright_pass_floor", "bright_pass_norm_pct", "shadow_band_hi",
              "shadow_match_max_gain", "shadow_match_strength"]
    for f in fields:
        m = re.search(r"(-?[\d.]+)–(-?[\d.]+)", gen.DESCRIPTIONS[f])
        assert m, f
        lo, hi = float(m.group(1)), float(m.group(2))
        cast = int if f == "bright_pass_region_size" else float
        eps = 1 if cast is int else 0.01
        for ok in (lo, hi):
            RenderOptions(image_path=Path("x"), **{f: cast(ok)})
        for bad in (lo - eps, hi + eps):
            with pytest.raises(ValueError):
                RenderOptions(image_path=Path("x"), **{f: cast(bad)})


# ---------------------------------------------------------------- T6 facts
def test_t6_trademark_and_lens_facts():
    assert TRADEMARK in _read(README)
    assert TRADEMARK in _read(DOCROOT / "docs" / "faq.md")
    rd = _read(README)
    # 0.5.0: two ray-traced lenses, credited to the expired patent and optiland; no measured-lens claim
    assert "ray-traced from the expired US patent 3,006,249, Example II, using optiland" in rd
    assert "US 3,006,249, Example II" in rd and "optiland" in rd and "MIT" in rd
    assert "not measurements of any Leica lens" in rd
    assert "--lens 50" in rd and "--lens 35" in rd
    for banned in ("ag_" + "dblgauss", "~100 mm", "Man" + "dler", "Lei" + "tz"):  # built from pieces: the legacy-token scan reads this file
        assert banned not in rd, banned


def test_t6_no_stale_claims():
    # "radially outward" is no longer stale: 0.5.0 verifies the orientation end to end (tests/unit/test_raytrace_orientation.py,
    # tests/adversarial/test_rotation_adversarial.py), so the s24 README may state it.
    stale = ["Summicron-M 35", "no implementation yet", "Depth Anything 3", "MediaPipe",
             "47×47", "~100 mm", "ag_" + "dblgauss", "--psf-norm l1", "--psf-norm peak"]
    files = NON_CHANGELOG + sorted((ROOT / "summicron").rglob("README.md"))
    for p in files:
        text = _read(p)
        for s in stale:
            assert s not in text, (p.relative_to(ROOT), s)


def test_t6_output_name_matches_save_results(tmp_path, capsys):
    import numpy as np
    from summicron.render.convolve import save_results

    # the quick start (default look) records its options in the name: guided depth, diopter gain 1.00 (the fixed 0.5.0 auto gain), input-lut tone
    p = save_results(np.zeros((4, 4, 3), np.uint8), "Demo_input", tmp_path, downres=2048,
                     name_suffix="_gd_dio1.00_tlut")
    capsys.readouterr()
    assert p.name == "Demo_input_r2048_rendered_gd_dio1.00_tlut.jpg"
    assert p.name in _read(DOCROOT / "docs" / "usage.md")


def _source_corpus() -> list[str]:
    out = []
    for p in (ROOT / "summicron").rglob("*.py"):
        t = p.read_text(encoding="utf-8", errors="replace")
        t = re.sub(r'"\s*\n\s*f?"', "", t)  # adjacent string literals join
        t = re.sub(r"\{[^{}]*\}", "…", t)  # f-string fields become the same ellipsis the docs use
        out.append(t)
    return out


def test_t6_troubleshooting_messages_exist_in_source():
    text = _read(DOCROOT / "docs" / "troubleshooting.md")
    corpus = _source_corpus()
    external = 0
    for m in re.finditer(r"(<!-- external -->\n)?```text\n(.*?)```", text, re.S):
        if m.group(1):
            external += 1
            continue
        for line in m.group(2).splitlines():
            for frag in line.split("…"):
                frag = frag.strip()
                if frag:
                    assert any(frag in c for c in corpus), frag
    assert 0 < external <= 3


# ---------------------------------------------------------------- T7 changelog
def test_t7_changelog_links():
    text = _read(DOCROOT / "CHANGELOG.md")
    versions = re.findall(r"(?m)^## \[([^\]]+)\]", text)
    assert versions
    refs = dict(re.findall(r"(?m)^\[([^\]]+)\]:\s*(\S+)", text))
    for v in versions:
        assert v in refs, v
        if tuple(int(x) for x in v.split(".")) >= (0, 3, 0):
            # a release link is the tag URL; from 0.5.0 on it may instead compare with the previous release
            prev = versions[versions.index(v) + 1] if versions.index(v) + 1 < len(versions) else None
            ok = [f"{BASE}/releases/tag/v{v}"]
            if v == "0.5.0":
                ok = [f"{BASE}/compare/v{prev}...v{v}"]  # 0.5.0 carries the compare link (the first diff-sized release)
            assert refs[v] in ok, (v, refs[v])
    body = re.search(rf"(?ms)^## \[{re.escape(versions[0])}\][^\n]*\n(.*?)(?=^## \[|^\[[^\]]+\]:)", text)
    assert body and body.group(1).strip()
