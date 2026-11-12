"""0.5.0: ``--psf-norm`` is removed (tombstone); L1 is the only normalisation."""
import inspect
import os
import pathlib
import subprocess
import sys

import pytest

from summicron.render import convolve as cv
from summicron.render.api import RenderOptions

ROOT = pathlib.Path(__file__).resolve().parents[2]


def _run(argv, tmp_path):
    return subprocess.run(
        argv, capture_output=True, text=True, cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
    )


@pytest.mark.parametrize("value", ["peak", "l1"])
@pytest.mark.parametrize("argv", [
    [sys.executable, "-m", "summicron.render.convolve"],
    [sys.executable, "-m", "summicron.cli", "render"],
])
def test_cli_psf_norm_tombstone_one_line(argv, value, tmp_path):
    img = tmp_path / "a.jpg"
    img.write_bytes(b"x")
    r = _run(argv + ["--image", str(img), "--psf-norm", value], tmp_path)
    assert r.returncode == 2, r.stderr
    lines = r.stderr.strip().splitlines()
    assert len(lines) == 1
    assert lines[0].startswith("render: --psf-norm was removed in 0.5.0")
    assert "Migrating to 0.5.0" in lines[0]
    assert "PSFs are always L1-normalised" in lines[0]
    assert "--lens" not in lines[0]


def test_psf_norm_absent_from_help():
    assert "psf-norm" not in cv.build_parser().format_help()


def test_render_options_rejects_psf_norm(tmp_path):
    with pytest.raises(TypeError):
        RenderOptions(image_path=tmp_path / "a.jpg", psf_norm="l1")  # type: ignore[call-arg]


def test_render_has_no_psf_norm_kwarg():
    assert "psf_norm" not in inspect.signature(cv.render).parameters
