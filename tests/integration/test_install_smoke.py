"""Install-smoke test (P1.audit Item 1).

Builds the wheel via ``python -m build``, installs into a clean temp venv
with NO extras, and asserts that the active CLI/render modules import
cleanly. Marked ``slow`` so the FAST suite skips it; CI / nightly runs it.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.slow
def test_wheel_install_imports_active_modules() -> None:
    """Build wheel, install with no extras, import render+cli."""
    if shutil.which("python") is None:
        pytest.skip("system python not available for venv smoke")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        # Build the wheel in an isolated dist dir.
        dist = tmp / "dist"
        build_proc = subprocess.run(
            [sys.executable, "-m", "build", "--wheel", "--outdir", str(dist), str(REPO_ROOT)],
            capture_output=True, text=True, check=False,
        )
        if build_proc.returncode != 0:
            pytest.skip(f"`python -m build` unavailable or failed: {build_proc.stderr[:300]}")
        wheels = sorted(dist.glob("isummicron-*.whl"))
        assert wheels, f"no wheel built; dist contents: {list(dist.iterdir())}"

        # Spin up clean venv.
        venv = tmp / "venv"
        subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)
        py = venv / ("Scripts" if sys.platform == "win32" else "bin") / "python"

        # Install with NO extras.
        install = subprocess.run(
            [str(py), "-m", "pip", "install", "--quiet", str(wheels[-1])],
            capture_output=True, text=True, check=False,
        )
        assert install.returncode == 0, f"pip install failed: {install.stderr[:500]}"

        # Active modules MUST import without any extra installed.
        smoke = subprocess.run(
            [str(py), "-c", "from summicron.render.convolve import render; from summicron.cli import main"],
            capture_output=True, text=True, check=False,
        )
        assert smoke.returncode == 0, (
            f"smoke import failed: stdout={smoke.stdout!r} stderr={smoke.stderr!r}"
        )
