"""S33 Tier C WP1: GPU engine import, isolation and availability tests.

Runs in BOTH envs (it is not importorskip-guarded at module level): the plain
``isummicron`` env proves the package imports and renders without mlx; the
dedicated mlx env makes the sentinel and the Metal-off check mandatory
(G-C0: select them with ``-m gpu``).
"""
from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
RENDER = REPO / "summicron" / "render"
ENGINE = RENDER / "_engine_gpu.py"


def _run(code: str, timeout=300):
    env = dict(os.environ, PYTHONPATH=str(REPO))
    return subprocess.run([sys.executable, "-c", code], cwd=REPO, env=env,
                          capture_output=True, text=True, timeout=timeout)


# --------------------------------------------------------------------------
# static checks
# --------------------------------------------------------------------------

def test_no_top_level_mlx_import_in_engine():
    tree = ast.parse(ENGINE.read_text())

    def mlx_import(node):
        if isinstance(node, ast.Import):
            return any(a.name.split(".")[0] == "mlx" for a in node.names)
        if isinstance(node, ast.ImportFrom):
            return (node.module or "").split(".")[0] == "mlx"
        return False

    # Everything that runs at import time: the module body, descending into
    # compound statements (if/try/with) but not into function or class bodies
    # that only run when called.
    stack = list(tree.body)
    while stack:
        node = stack.pop()
        assert not mlx_import(node), f"top-level mlx import at line {node.lineno}"
        if isinstance(node, (ast.If, ast.Try, ast.With)):
            stack.extend(ast.iter_child_nodes(node))
    # ... and function-level imports are the intended mechanism
    assert re.search(r"^\s+import mlx\.core as mx", ENGINE.read_text(), re.M)


@pytest.mark.parametrize("name", [
    "_tile_schedule.py", "_engine_tiled.py", "convolve.py", "api.py",
    "_resolve.py", "_render_args.py",
])
def test_static_isolation_no_mlx_in_other_modules(name):
    text = (RENDER / name).read_text()
    assert not re.search(r"^\s*(import|from)\s+mlx\b", text, re.M), name
    # the gpu engine is reached only by a function-local import
    assert not re.search(
        r"^(import|from)\s+summicron\.render(\._engine_gpu\b|\s+import\s+.*\b_engine_gpu\b)",
        text, re.M), f"{name}: top-level import of _engine_gpu"


def test_engine_module_uses_only_shared_planner():
    text = ENGINE.read_text()
    assert "from summicron.render import _tile_schedule as ts" in text
    assert "from summicron.render import _engine_tiled as et" in text
    for banned in ("metal_kernel", "float16", "bfloat16", "set_default_device",
                   ".at[", "math_mode"):
        assert banned not in text, banned


# --------------------------------------------------------------------------
# lazy import
# --------------------------------------------------------------------------

def test_lazy_import_reference_and_tiled_do_not_load_gpu_engine_or_mlx():
    code = (
        "import sys\n"
        "import numpy as np\n"
        "import summicron\n"
        "import summicron.render.convolve as cv\n"
        "import argparse\n"
        "from summicron.render._render_args import add_render_arguments\n"
        "add_render_arguments(argparse.ArgumentParser()).format_help()\n"
        "from tests.fixtures.synthetic_tensor import make_synthetic_tensor\n"
        "from tests.fixtures.defocus import render_with_defocus\n"
        "t = make_synthetic_tensor(12, 5, 17, 17)\n"
        "img = np.random.default_rng(0).uniform(0, 1, (40, 48, 3)).astype(np.float32)\n"
        "render_with_defocus(img, t)\n"
        "render_with_defocus(img, t, engine='tiled')\n"
        "bad = [m for m in ('summicron.render._engine_gpu', 'mlx', 'mlx.core')\n"
        "       if m in sys.modules]\n"
        "assert not bad, bad\n"
        "print('LAZY_OK')\n"
    )
    r = _run(code)
    assert r.returncode == 0 and "LAZY_OK" in r.stdout, r.stdout + r.stderr


# --------------------------------------------------------------------------
# availability
# --------------------------------------------------------------------------

def test_module_imports_and_reports_unavailable_without_mlx():
    code = (
        "import sys\n"
        "sys.modules['mlx'] = None\n"
        "sys.modules['mlx.core'] = None\n"
        "import summicron.render._engine_gpu as e\n"
        "assert e.availability() == 'mlx is not importable', e.availability()\n"
        "print('ABSENT_OK')\n"
    )
    r = _run(code)
    assert r.returncode == 0 and "ABSENT_OK" in r.stdout, r.stdout + r.stderr


@pytest.mark.gpu
def test_availability_none_when_usable_and_reason_when_metal_off():
    mx = pytest.importorskip("mlx.core")
    from summicron.render import _engine_gpu as eg

    if not mx.metal.is_available():
        pytest.fail("no Metal device in a gpu test run")
    assert eg.availability() is None
    real = mx.metal.is_available
    mx.metal.is_available = lambda: False
    try:
        assert eg.availability() == "no Metal GPU device"
    finally:
        mx.metal.is_available = real
    assert eg.availability() is None


@pytest.mark.gpu
def test_gpu_env_sentinel_fails_not_skips():
    """With ISUMMICRON_REQUIRE_GPU=1, a missing mlx or Metal is a FAILURE (G-C0).
    Without it this passes trivially (the gpu tests skip)."""
    if os.environ.get("ISUMMICRON_REQUIRE_GPU") != "1":
        return
    try:
        import mlx.core as mx
    except ImportError as e:     # pragma: no cover
        pytest.fail(f"ISUMMICRON_REQUIRE_GPU=1 without importable mlx.core: {e}")
    if not mx.metal.is_available():
        pytest.fail("ISUMMICRON_REQUIRE_GPU=1 without a Metal device")
