"""SHIP-3 adversarial probes — try to BREAK the shelved-pipeline removal.

Spec: doc/plan/phase2/section11_ship-3/locked-spec-SHIP-3.md.

Every absence check here is paired with a wrong-polarity control proving
the check is capable of failing (Post-mortem 1): a ``find_spec(...) is None``
assertion is worthless if ``find_spec`` would also return None for a
module that IS importable, or if a stale ``__pycache__``-only directory
would silently resurrect the package as an empty namespace package.
"""
from __future__ import annotations

import ast
import importlib
import importlib.util
import py_compile
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
PKG = REPO / "summicron"

REMOVED_PACKAGES = ("harvest", "harvest_r3", "psf_extract")
REMOVED_PSF_MODEL = ("build", "cherry", "cherry_viz", "tensor4d", "viz_tensor4d", "viz")
REMOVED_MODULES = tuple(f"summicron.{p}" for p in REMOVED_PACKAGES) + tuple(
    f"summicron.psf_model.{m}" for m in REMOVED_PSF_MODEL
)
REMOVED_SUBCOMMANDS = ("inventory", "harvest", "extract", "psf-model")


def _is_removed_import(name: str) -> bool:
    """True iff an absolute import name is (or lives under) a removed module."""
    return any(name == m or name.startswith(m + ".") for m in REMOVED_MODULES)


def _imports_in(path: Path) -> list[tuple[int, str]]:
    """All absolute import targets in a file, incl. ``from summicron.psf_model import build``."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    out: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out.extend((node.lineno, a.name) for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            out.append((node.lineno, node.module))
            out.extend((node.lineno, f"{node.module}.{a.name}") for a in node.names)
        elif isinstance(node, ast.Call):
            # importlib.import_module("summicron.harvest") / __import__(...)
            fn = node.func
            fname = getattr(fn, "attr", None) or getattr(fn, "id", None)
            if fname in {"import_module", "__import__"} and node.args:
                a0 = node.args[0]
                if isinstance(a0, ast.Constant) and isinstance(a0.value, str):
                    out.append((node.lineno, a0.value))
    return out


# --------------------------------------------------------------------------
# 1. Absence on disk + importlib, with wrong-polarity decoy controls
# --------------------------------------------------------------------------


@pytest.mark.parametrize("pkg", REMOVED_PACKAGES)
def test_removed_package_dir_absent_on_disk(pkg: str) -> None:
    """No directory at all — not even a ``__pycache__``-only husk (the trap)."""
    d = PKG / pkg
    assert not d.exists(), f"{d} still exists: {sorted(p.name for p in d.iterdir())}"


@pytest.mark.parametrize("name", REMOVED_MODULES)
def test_removed_module_find_spec_is_none(name: str) -> None:
    importlib.invalidate_caches()
    assert importlib.util.find_spec(name) is None, name
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(name)


def test_find_spec_check_detects_pycache_namespace_trap(tmp_path, monkeypatch) -> None:
    """Wrong-polarity control for the two tests above.

    A ``harvest/__pycache__/`` husk left behind after ``git rm`` makes
    ``harvest`` an importable *namespace package*. Prove find_spec is NOT
    vacuously None in that situation, i.e. the absence test above would fail
    if the trap were present — and that it IS None once the husk is gone.
    """
    root = tmp_path / "summicron_fake"
    (root / "harvest" / "__pycache__").mkdir(parents=True)
    (root / "__init__.py").write_text("")
    (root / "render").mkdir()
    (root / "render" / "__init__.py").write_text("")
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    try:
        spec = importlib.util.find_spec("summicron_fake.harvest")
        assert spec is not None, "decoy husk should resolve as a namespace pkg"
        assert spec.origin in (None, "namespace")  # namespace package
        assert importlib.util.find_spec("summicron_fake.render") is not None
        shutil.rmtree(root / "harvest")
        importlib.invalidate_caches()
        sys.modules.pop("summicron_fake.harvest", None)
        assert importlib.util.find_spec("summicron_fake.harvest") is None
    finally:
        for k in [k for k in sys.modules if k.startswith("summicron_fake")]:
            sys.modules.pop(k, None)


def test_stale_psf_model_pycs_cannot_resurrect_modules(tmp_path, monkeypatch) -> None:
    """``summicron/psf_model/__pycache__/build.cpython-*.pyc`` may linger
    (gitignored). PEP 3147 pycs in ``__pycache__`` are never imported without
    source — but a *legacy* sourceless ``build.pyc`` next to ``__init__.py``
    would be. Control: show the legacy placement IS importable in a decoy, so
    the real-tree check below is not vacuous, then assert no legacy-location
    pyc exists for any removed module anywhere under summicron/.
    """
    pkg = tmp_path / "decoy_pm"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    src = tmp_path / "build.py"
    src.write_text("X = 42\n")
    # PEP 3147 location only -> not importable
    (pkg / "__pycache__").mkdir()
    tag = sys.implementation.cache_tag
    py_compile.compile(str(src), cfile=str(pkg / "__pycache__" / f"build.{tag}.pyc"))
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    try:
        assert importlib.util.find_spec("decoy_pm.build") is None
        # legacy sourceless location -> importable (the dangerous case)
        py_compile.compile(str(src), cfile=str(pkg / "build.pyc"))
        importlib.invalidate_caches()
        assert importlib.util.find_spec("decoy_pm.build") is not None
    finally:
        for k in [k for k in sys.modules if k.startswith("decoy_pm")]:
            sys.modules.pop(k, None)

    legacy = [
        p for p in PKG.rglob("*.pyc")
        if p.parent.name != "__pycache__"
    ]
    assert legacy == [], f"sourceless legacy pycs could shadow removed code: {legacy}"


# --------------------------------------------------------------------------
# 2. CLI surface
# --------------------------------------------------------------------------


@pytest.mark.parametrize("sub", REMOVED_SUBCOMMANDS)
def test_removed_subcommand_exits_2(sub: str, capsys) -> None:
    from summicron.cli import main

    with pytest.raises(SystemExit) as exc:
        main([sub, "--help"])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "invalid choice" in err and repr(sub) in err


def test_removed_subcommand_polarity_control(capsys) -> None:
    """Control: a kept subcommand with ``--help`` exits 0, so exit-2 above is
    caused by the subcommand being gone, not by ``--help`` handling."""
    from summicron.cli import main

    with pytest.raises(SystemExit) as exc:
        main(["render", "--help"])
    assert exc.value.code == 0
    with pytest.raises(SystemExit) as exc:
        main(["eval", "--help"])
    assert exc.value.code == 0


def test_main_empty_argv_still_errors_with_usage(capsys) -> None:
    """``main([])`` keeps its pre-SHIP-3 behaviour: required subparser -> exit 2,
    and the usage choices list only render/eval."""
    from summicron.cli import main

    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "{render,eval}" in err
    for sub in REMOVED_SUBCOMMANDS:
        assert sub not in err.replace("isummicron", "")


def test_installed_console_script_rejects_removed_subcommand() -> None:
    """The real entry point (editable install) — not just main()."""
    exe = Path(sys.executable).parent / "isummicron"
    if not exe.exists():
        exe_s = shutil.which("isummicron")
        if exe_s is None:
            pytest.skip("isummicron console script not installed")
        exe = Path(exe_s)
    r = subprocess.run([str(exe), "harvest", "--help"], capture_output=True, text=True,
                       timeout=60, cwd=REPO)
    assert r.returncode == 2, r.stderr
    assert "Traceback" not in r.stderr
    r = subprocess.run([str(exe), "--help"], capture_output=True, text=True,
                       timeout=60, cwd=REPO)
    assert r.returncode == 0, r.stderr
    assert "{render,eval}" in r.stdout


def test_python_m_removed_package_fails_cleanly() -> None:
    r = subprocess.run([sys.executable, "-m", "summicron.harvest"], capture_output=True,
                       text=True, timeout=60, cwd=REPO)
    assert r.returncode != 0
    assert "No module named summicron.harvest" in r.stderr


# --------------------------------------------------------------------------
# 3. Kept psf_model tools
# --------------------------------------------------------------------------


@pytest.mark.parametrize("mod", ["downscale_tensor", "codev_to_render"])
def test_kept_psf_model_tool_help_exits_0(mod: str) -> None:
    r = subprocess.run([sys.executable, "-m", f"summicron.psf_model.{mod}", "--help"],
                       capture_output=True, text=True, timeout=120, cwd=REPO)
    assert r.returncode == 0, r.stderr
    assert "usage" in r.stdout.lower()


@pytest.mark.parametrize("mod", ["downscale_tensor", "codev_to_render", "__init__"])
def test_kept_psf_model_files_import_no_removed_module(mod: str) -> None:
    bad = [(ln, n) for ln, n in _imports_in(PKG / "psf_model" / f"{mod}.py")
           if _is_removed_import(n)]
    assert bad == []


def test_psf_model_package_has_only_kept_modules() -> None:
    py = sorted(p.stem for p in (PKG / "psf_model").glob("*.py"))
    assert py == ["__init__", "codev_to_render", "downscale_tensor"]


# --------------------------------------------------------------------------
# 4. Whole-tree AST scan (with polarity control on the scanner itself)
# --------------------------------------------------------------------------


def test_ast_scanner_polarity_control(tmp_path) -> None:
    f = tmp_path / "probe.py"
    f.write_text(
        "import summicron.harvest.detect\n"
        "from summicron.psf_model import cherry\n"
        "from summicron.psf_extract.extract import x\n"
        "import importlib\nimportlib.import_module('summicron.harvest_r3')\n"
        "from summicron.psf_model import downscale_tensor\n"
        "import summicron.harvesting_ok\n"
    )
    hits = sorted({n for _, n in _imports_in(f) if _is_removed_import(n)})
    assert hits == [
        "summicron.harvest.detect",
        "summicron.harvest_r3",
        "summicron.psf_extract.extract",
        "summicron.psf_extract.extract.x",
        "summicron.psf_model.cherry",
    ]


def test_no_tracked_python_imports_removed_module() -> None:
    roots = [REPO / d for d in ("summicron", "tests", "scripts", "demo")]
    files = [p for r in roots if r.exists() for p in r.rglob("*.py")
             if "__pycache__" not in p.parts]
    assert len(files) > 50  # sanity: the scan actually covers the tree
    me = Path(__file__).resolve()
    bad = []
    for p in files:
        if p.resolve() == me:
            continue
        bad += [f"{p.relative_to(REPO)}:{ln}: {n}" for ln, n in _imports_in(p)
                if _is_removed_import(n)]
    assert bad == [], "\n".join(bad)


def test_no_monkeypatch_string_targets_removed_module() -> None:
    """test_cli.py used to ``monkeypatch.setattr("summicron.harvest.…")`` —
    string targets are invisible to the import scan."""
    needles = [m + "." for m in REMOVED_MODULES] + [f'"{m}"' for m in REMOVED_MODULES]
    me = Path(__file__).resolve()
    bad = []
    for p in (REPO / "tests").rglob("*.py"):
        if p.resolve() == me or "__pycache__" in p.parts:
            continue
        txt = p.read_text(encoding="utf-8")
        bad += [f"{p.relative_to(REPO)}: {n}" for n in needles if n in txt]
    assert bad == [], "\n".join(bad)


# --------------------------------------------------------------------------
# 5. Packaging
# --------------------------------------------------------------------------


def test_pyproject_has_no_shelved_marker_or_removed_deps() -> None:
    import tomllib

    cfg = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    markers = cfg["tool"]["pytest"]["ini_options"].get("markers", [])
    assert not any(m.split(":")[0].strip() == "shelved" for m in markers)
    deps = list(cfg["project"].get("dependencies", []))
    for extra in cfg["project"].get("optional-dependencies", {}).values():
        deps += extra
    low = " ".join(deps).lower()
    assert "astropy" not in low and "photutils" not in low
    scripts = cfg["project"].get("scripts", {})
    assert all("harvest" not in v and "psf_extract" not in v for v in scripts.values())


def test_wheel_contents(tmp_path) -> None:
    r = subprocess.run(
        [sys.executable, "-m", "pip", "wheel", str(REPO), "--no-deps", "-q",
         "-w", str(tmp_path)],
        capture_output=True, text=True, timeout=300,
    )
    wheels = list(tmp_path.glob("*.whl"))
    if r.returncode != 0 or not wheels:
        pytest.skip(f"wheel build unavailable: {r.stderr[-400:]}")
    names = zipfile.ZipFile(wheels[0]).namelist()
    bad = [n for n in names
           if n.startswith(("summicron/harvest", "summicron/psf_extract"))
           or any(n == f"summicron/psf_model/{m}.py" for m in REMOVED_PSF_MODEL)
           or n.endswith(".pyc")]
    assert bad == [], bad
    assert "summicron/_artifacts/codev_render_v1.npz" in names
    assert "summicron/psf_model/downscale_tensor.py" in names
    assert "summicron/psf_model/codev_to_render.py" in names
    assert not any(n.startswith("tests/") for n in names)
