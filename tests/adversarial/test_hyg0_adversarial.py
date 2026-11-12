"""HYG-0 adversarial probes — try to break the new test/bench hygiene contracts.

Wrong-polarity discipline (Post-mortem 1): every guard introduced or
repointed by HYG-0 is fed a deliberately BAD input and must FAIL, proving
it is not vacuous against the shipped tensor / pinned JPEG.

Targets:
- ``tests/unit/test_convolve_load_tensor.py::test_default_tensor_metadata_flags``
  (item 11 metadata guard)
- ``tests/adversarial/test_delta_guard_p1_9_adversarial.py::test_b4`` (item 10
  delta-content guard, repointed to ``default_tensor_path()``)
- ``_assert_npz_content_equal`` helpers (item 12 byte-md5 -> content compare)
- ``tests/unit/test_back_compat_canonical.py`` SLOW G1 md5 assertion +
  ``_env_diagnostics`` (item 3)
- ``tests/conftest.py`` stray-``summicron`` guard (item 9)
- ``summicron/render/_bench.py`` (item 1), ``anchor_picker`` template (item 2),
  ``runs/vibe_bench/README.md`` primary command (item 5)
- no stale ``*_render_iphone5712.npz`` references (item 10)
"""
from __future__ import annotations

import importlib
import io
import json
import re
import shlex
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from summicron.render._artifacts import default_tensor_path

ROOT = Path(__file__).resolve().parents[2]
SHIPPED = default_tensor_path()
README = ROOT / "runs" / "vibe_bench" / "README.md"

pytestmark = pytest.mark.skipif(
    not SHIPPED.exists(), reason="wheel-shipped tensor missing"
)


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------


def _shipped_arrays() -> dict[str, np.ndarray]:
    with np.load(str(SHIPPED)) as d:
        return {k: d[k].copy() for k in d.files}


def _write_npz(path: Path, arrays: dict[str, np.ndarray]) -> Path:
    np.savez(str(path), **arrays)
    return path


def _import_test_module(dotted: str) -> types.ModuleType:
    # tests/ has no __init__.py; repo root is on sys.path via
    # pyproject `pythonpath = ["."]`, so `tests.*` resolves as a namespace
    # package (same mechanism test_convolve_load_tensor uses for fixtures).
    return importlib.import_module(dotted)


def _cli_args_from_command(cmd_text: str) -> list[str]:
    toks = shlex.split(cmd_text.replace("\\\n", " "))
    i = toks.index("summicron.render.convolve")
    return toks[i + 1:]


def _readme_primary_command() -> str:
    text = README.read_text(encoding="utf-8")
    sec = text.split("## Primary canonical bench", 1)[1]
    m = re.search(r"```bash\n(.*?)```", sec, re.S)
    assert m, "no bash block in README primary section"
    return m.group(1)


# ----------------------------------------------------------------------------
# 1. Metadata guard (item 11) — wrong polarity
# ----------------------------------------------------------------------------


def _bad_metadata(variant: str) -> dict[str, np.ndarray]:
    a = _shipped_arrays()
    if variant == "otf_true":
        a["otf_norm_applied"] = np.bool_(True)
    elif variant == "delta_guard_missing":
        del a["delta_guard_applied"]
    elif variant == "delta_guard_false":
        a["delta_guard_applied"] = np.bool_(False)
    else:  # pragma: no cover
        raise AssertionError(variant)
    return a


@pytest.mark.parametrize(
    "variant", ["otf_true", "delta_guard_missing", "delta_guard_false"]
)
def test_metadata_guard_fires_on_bad_tensor(tmp_path, monkeypatch, variant):
    mod = _import_test_module("tests.unit.test_convolve_load_tensor")
    bad = _write_npz(tmp_path / f"bad_{variant}.npz", _bad_metadata(variant))
    monkeypatch.setattr(mod, "DEFAULT_TENSOR", bad)
    with pytest.raises(AssertionError):
        mod.TestLoadTensorShapeDerivation().test_default_tensor_metadata_flags()


def test_metadata_guard_positive_control(monkeypatch):
    mod = _import_test_module("tests.unit.test_convolve_load_tensor")
    monkeypatch.setattr(mod, "DEFAULT_TENSOR", SHIPPED)
    mod.TestLoadTensorShapeDerivation().test_default_tensor_metadata_flags()


# ----------------------------------------------------------------------------
# 2. test_b4 delta-content guard (item 10) — wrong polarity
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
# 3. Content-compare helpers (item 12) — wrong polarity
# ----------------------------------------------------------------------------



# ----------------------------------------------------------------------------
# 4. G1 SLOW md5 assertion + diagnostics (item 3) — wrong polarity
# ----------------------------------------------------------------------------


def _perturbed_jpeg_bytes(src: Path | None) -> bytes:
    from PIL import Image

    if src is not None and src.exists():
        with Image.open(str(src)) as im:
            arr = np.asarray(im.convert("RGB"), dtype=np.uint8).copy()
    else:
        arr = np.full((64, 48, 3), 128, dtype=np.uint8)
    arr[:32, :32] = 255 - arr[:32, :32]
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def test_g1_md5_assertion_fires_with_diagnostics(tmp_path, monkeypatch):
    """Fake the render subprocess to emit a perturbed JPEG; the SLOW G1 test
    must raise AssertionError whose message carries macOS build, pyobjc,
    tensor md5 and (if baseline present) a nonzero max-abs."""
    mod = _import_test_module("tests.unit.test_back_compat_canonical")
    if not mod.CANONICAL_IMAGE.exists():
        pytest.skip("canonical image missing")
    payload = _perturbed_jpeg_bytes(mod.BASELINE_JPG)
    real_run = subprocess.run
    seen = {}

    def fake_run(cmd, *args, **kwargs):
        if cmd and cmd[0] == "sw_vers":
            return real_run(cmd, *args, **kwargs)
        seen["cmd"] = list(cmd)
        out = Path(cmd[cmd.index("--output") + 1])
        out.write_bytes(payload)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    fake_sub = types.SimpleNamespace(
        run=fake_run, SubprocessError=subprocess.SubprocessError
    )
    monkeypatch.setattr(mod, "subprocess", fake_sub)
    with pytest.raises(AssertionError) as ei:
        mod.test_backcompat_canonical_render_matches_baseline(tmp_path)
    msg = str(ei.value)
    assert "G1 FAIL" in msg and mod.G1_MD5 in msg
    assert re.search(r"macOS: \S+ build \S+", msg), msg
    assert "pyobjc-core:" in msg
    assert f"tensor md5: {mod.TENSOR_MD5}" in msg, msg
    if mod.BASELINE_JPG.exists():
        m = re.search(r"max-abs vs baseline: (\d+) \(differing px: (\d+)\)", msg)
        assert m and int(m.group(1)) > 0 and int(m.group(2)) > 0, msg
    # argv sanity: no --tensor, flags from CANONICAL_FLAGS, default tensor.
    assert "--tensor" not in seen["cmd"]


def test_tensor_md5_pin_fires_on_one_ulp_tensor(tmp_path, monkeypatch):
    """Item 7b wrong polarity: a 1-ulp-perturbed wheel tensor must fail the
    FAST TENSOR_MD5 gate."""
    import summicron.render._artifacts as art

    mod = _import_test_module("tests.unit.test_back_compat_canonical")
    a = _shipped_arrays()
    a["kernels"][3, 4, 91, 91] = np.nextafter(
        a["kernels"][3, 4, 91, 91], np.float32(np.inf)
    )
    bad = _write_npz(tmp_path / "t.npz", a)
    monkeypatch.setattr(art, "default_tensor_path", lambda *a, **k: bad)
    with pytest.raises(AssertionError, match="!= pin"):
        mod.test_default_tensor_md5_matches_pin()


def test_baseline_gate_fires_on_wrong_jpeg_and_skips_with_restore_cmd(
    tmp_path, monkeypatch
):
    mod = _import_test_module("tests.unit.test_back_compat_canonical")
    wrong = tmp_path / "IMG_6915_R2_layered.jpg"
    wrong.write_bytes(_perturbed_jpeg_bytes(None))
    monkeypatch.setattr(mod, "BASELINE_JPG", wrong)
    with pytest.raises(AssertionError, match="G1_MD5"):
        mod.test_backcompat_baseline_exists()
    with pytest.raises(AssertionError):
        mod.test_backcompat_baseline_readable()
    monkeypatch.setattr(mod, "BASELINE_JPG", tmp_path / "absent.jpg")
    for fn in (mod.test_backcompat_baseline_exists,
               mod.test_backcompat_baseline_readable):
        with pytest.raises(pytest.skip.Exception) as ei:
            fn()
        assert mod.RESTORE_CMD in str(ei.value.msg)


def test_g1_md5_assertion_quiet_on_pinned_bytes(tmp_path, monkeypatch):
    """Positive control: fake render emitting the pinned baseline passes."""
    mod = _import_test_module("tests.unit.test_back_compat_canonical")
    if not (mod.CANONICAL_IMAGE.exists() and mod.BASELINE_JPG.exists()):
        pytest.skip("canonical image or baseline missing")
    payload = mod.BASELINE_JPG.read_bytes()

    def fake_run(cmd, *args, **kwargs):
        Path(cmd[cmd.index("--output") + 1]).write_bytes(payload)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(
        mod, "subprocess",
        types.SimpleNamespace(run=fake_run, SubprocessError=subprocess.SubprocessError),
    )
    mod.test_backcompat_canonical_render_matches_baseline(tmp_path)


# ----------------------------------------------------------------------------
# 5. conftest stray-summicron guard (item 9)
# ----------------------------------------------------------------------------


def _root_conftest():
    return _import_test_module("tests.conftest")


@pytest.mark.parametrize(
    "fake_file",
    [
        None,  # namespace fragment (stray site-packages/_artifacts dir)
        "/opt/elsewhere/site-packages/summicron/__init__.py",
        str(ROOT.parent / (ROOT.name + "-evil") / "summicron" / "__init__.py"),
    ],
    ids=["namespace_none", "site_packages", "sibling_prefix_dir"],
)
def test_conftest_guard_exits_rc4(monkeypatch, fake_file):
    cf = _root_conftest()
    fake = types.ModuleType("summicron")
    fake.__file__ = fake_file
    monkeypatch.setitem(sys.modules, "summicron", fake)
    with pytest.raises(pytest.exit.Exception) as ei:
        cf.pytest_configure(None)
    assert ei.value.returncode == 4
    assert "not under repo root" in str(ei.value.msg)


def test_conftest_guard_subprocess_decoy(tmp_path):
    """End-to-end: decoy package first on PYTHONPATH, pyproject pythonpath
    neutralised, run from a tmp cwd -> pytest exits 4 with a clear message.
    Then the same invocation without the decoy passes."""
    lib = tmp_path / "lib" / "summicron"
    lib.mkdir(parents=True)
    (lib / "__init__.py").write_text('__version__ = "decoy"\n')
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    target = str(ROOT / "tests" / "unit" / "test_back_compat_canonical.py")
    base = [sys.executable, "-m", "pytest", target, "-k", "flag_removed",
            "-p", "no:xdist", "-p", "no:cacheprovider", "-q", "-o", "pythonpath="]
    import os

    env = dict(os.environ)
    env["PYTHONPATH"] = str(lib.parent)
    r = subprocess.run(base, cwd=str(cwd), env=env, capture_output=True,
                       text=True, timeout=120)
    assert r.returncode == 4, (r.returncode, r.stdout[-2000:], r.stderr[-2000:])
    assert "not under repo root" in (r.stdout + r.stderr)
    assert "decoy" in (r.stdout + r.stderr) or str(lib) in (r.stdout + r.stderr)

    env.pop("PYTHONPATH", None)
    r2 = subprocess.run(base[:-2], cwd=str(cwd), env=env, capture_output=True,
                        text=True, timeout=120)
    assert r2.returncode == 0, (r2.stdout[-2000:], r2.stderr[-2000:])


def test_conftest_guard_rejects_copy_inside_repo(monkeypatch):
    cf = _root_conftest()
    fake = types.ModuleType("summicron")
    fake.__file__ = str(
        ROOT / ".venv" / "lib" / "python3.11" / "site-packages" / "summicron" / "__init__.py"
    )
    monkeypatch.setitem(sys.modules, "summicron", fake)
    with pytest.raises(pytest.exit.Exception):
        cf.pytest_configure(None)


# ----------------------------------------------------------------------------
# 6. _bench.py contract (item 1)
# ----------------------------------------------------------------------------


def test_bench_import_is_stdlib_only():
    code = (
        "import sys, json; import summicron.render._bench as b; "
        "print(json.dumps(sorted(m for m in ('numpy','scipy','PIL') if m in sys.modules)))"
    )
    r = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT),
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout.strip().splitlines()[-1]) == []
    src = (ROOT / "summicron" / "render" / "_bench.py").read_text(encoding="utf-8")
    assert not re.search(r"^\s*(import|from)\s", src, re.M), "_bench.py has imports"


def test_bench_not_reexported_and_flags_clean():
    import summicron.render as sr
    from summicron.render import _bench

    init_src = Path(sr.__file__).read_text(encoding="utf-8")
    assert "_bench" not in init_src
    for name in ("CANONICAL_FLAGS", "G1_MD5", "ON_MD5", "TENSOR_MD5"):
        assert not hasattr(sr, name)
    flags = _bench.CANONICAL_FLAGS
    assert isinstance(flags, tuple) and all(isinstance(t, str) for t in flags)
    for bad in ("--tensor", "--image", "--focus-xy", "--output", "--out-dir"):
        assert bad not in flags
    for v in (_bench.G1_MD5, _bench.ON_MD5, _bench.TENSOR_MD5):
        assert re.fullmatch(r"[0-9a-f]{32}", v)
    assert _bench.G1_MD5 != _bench.ON_MD5


def test_canonical_flags_accepted_by_real_parser_and_match_readme():
    from summicron.render._bench import CANONICAL_FLAGS, G1_FOCUS_XY, G1_IMAGE
    from summicron.render.convolve import _validate_args, build_parser

    p = build_parser()
    ns = p.parse_args(
        ["--image", G1_IMAGE, *CANONICAL_FLAGS,
         "--focus-xy", *(str(v) for v in G1_FOCUS_XY), "--output", "x.jpg"]
    )
    _validate_args(ns, p)  # cross-flag validators (parser.error -> SystemExit)
    assert ns.tensor is None
    readme_args = _cli_args_from_command(_readme_primary_command())
    i, j = readme_args.index("--image"), readme_args.index("--focus-xy")
    assert readme_args[i + 1] == G1_IMAGE
    assert tuple(readme_args[i + 2:j]) == CANONICAL_FLAGS
    assert [float(v) for v in readme_args[j + 1:j + 3]] == list(G1_FOCUS_XY)
    assert "--tensor" not in readme_args
    assert vars(p.parse_args(readme_args)) | {"output": "x.jpg"} == vars(ns)


def test_readme_pins_table_mirrors_bench():
    from summicron.render import _bench

    text = README.read_text(encoding="utf-8")
    pins = text.split("## PINS", 1)[1].split("\n## ", 1)[0]
    assert f"`{_bench.G1_MD5}` (`G1_MD5`)" in pins
    assert f"`{_bench.ON_MD5}` (`ON_MD5`)" in pins
    assert f"`{_bench.TENSOR_MD5}` (`TENSOR_MD5`)" in pins


# ----------------------------------------------------------------------------
# 7. anchor_picker emitted command (item 2)
# ----------------------------------------------------------------------------


def test_anchor_picker_command_parses_with_real_cli():
    from summicron.render._bench import CANONICAL_FLAGS
    from summicron.render.convolve import _validate_args, build_parser
    from summicron.tools.anchor_picker import emit_canonical_command

    cmd = emit_canonical_command("iphoneImage/IMG_6651.HEIC", 0.123456, 0.9)
    assert "--tensor" not in cmd
    args = _cli_args_from_command(cmd)
    p = build_parser()
    ns = p.parse_args(args)
    _validate_args(ns, p)
    assert ns.tensor is None
    assert ns.focus_xy == pytest.approx([0.1235, 0.9])
    i, j = args.index("--image"), args.index("--focus-xy")
    assert tuple(args[i + 2:j]) == CANONICAL_FLAGS


@pytest.mark.xfail(
    strict=True,
    reason="BUG (minor): emit_canonical_command interpolates {IMAGE} unquoted, "
    "so a path containing a space yields a command whose shlex tokens split "
    "the path (copy-paste into a shell breaks). Fix: shlex.quote(image_path).",
)
def test_anchor_picker_quotes_image_path_with_space():
    from summicron.tools.anchor_picker import emit_canonical_command

    path = "iphoneImage/IMG 6651.HEIC"
    args = _cli_args_from_command(emit_canonical_command(path, 0.5, 0.5))
    assert args[args.index("--image") + 1] == path


# ----------------------------------------------------------------------------
# 8. no stale OTF/iphone5712 tensor references (item 10)
# ----------------------------------------------------------------------------


def test_no_stale_iphone5712_tensor_references():
    stale = "ag_" "dblgauss_f2.0_render_iphone5712"
    me = Path(__file__).resolve()
    hits = []
    for base in (ROOT / "summicron", ROOT / "tests"):
        for f in base.rglob("*.py"):
            if f.resolve() == me:
                continue
            for n, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
                if stale in line:
                    hits.append(f"{f.relative_to(ROOT)}:{n}: {line.strip()}")
    assert not hits, "stale tensor refs:\n" + "\n".join(hits)
