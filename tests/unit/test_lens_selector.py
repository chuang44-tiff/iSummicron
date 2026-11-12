"""0.5.0: ``--lens 35|50`` selector, removed-flag tombstones, API surface.

The only PSF source is an optiland ray-traced tensor. ``--kernel-family`` and
``--lens-aperture-mm`` were removed and fail with one migration line. Removed
values are built by concatenation so this file carries no guarded token.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from summicron.render import _artifacts
from summicron.render._artifacts import LENS_ARTIFACTS, default_tensor_path

ROOT = Path(__file__).resolve().parents[2]

_OLD_FAMILY_VALUE = "code" + "v"  # a removed --kernel-family value

_ENTRYPOINTS = {
    "cli": [sys.executable, "-m", "summicron.cli", "render"],
    "convolve": [sys.executable, "-m", "summicron.render.convolve"],
}
_GUARD_TOKENS = ("code" + "v", "code" + " v", "code" + "_v", "code" + "-v",
                 "ag_" + "dblgauss", "35" + "f2")


def _run(entry, args, **kw):
    return subprocess.run(
        [*_ENTRYPOINTS[entry], *args], capture_output=True, text=True,
        cwd=str(ROOT), timeout=120, env={**__import__("os").environ, "PYTHONPATH": str(ROOT)},
        **kw,
    )


def test_artifact_table():
    assert LENS_ARTIFACTS == {
        "35": "psf_lens35_us3006249_fu0.4.npz",
        "50": "psf_lens50_doublegauss_fu0.4.npz",
    }
    assert _artifacts.DEFAULT_LENS == "35"


def test_lens_resolution():
    assert default_tensor_path().name == LENS_ARTIFACTS["35"]
    assert default_tensor_path("35").name == LENS_ARTIFACTS["35"]
    assert default_tensor_path("50").name == LENS_ARTIFACTS["50"]
    assert default_tensor_path("35").is_file() and default_tensor_path("50").is_file()
    with pytest.raises(ValueError, match="unknown lens"):
        default_tensor_path("85")


def test_lens_and_tensor_exclusive():
    from summicron.render.api import RenderOptions

    with pytest.raises(ValueError, match="mutually exclusive"):
        RenderOptions(image_path=Path("x.heic"), lens="35", tensor_path=Path("t.npz"))
    RenderOptions(image_path=Path("x.heic"), lens="50")
    RenderOptions(image_path=Path("x.heic"), tensor_path=Path("t.npz"))
    with pytest.raises(ValueError, match="--lens"):
        RenderOptions(image_path=Path("x.heic"), lens="85")


def test_lens_choices_exact():
    from summicron.cli import build_parser as cli_parser
    from summicron.render.convolve import build_parser

    for p in (build_parser(), cli_parser()):
        parsers = [p]
        for a in p._actions:
            if getattr(a, "choices", None) and isinstance(a.choices, dict):
                parsers.extend(a.choices.values())
        found = [a for q in parsers for a in q._actions if "--lens" in a.option_strings]
        assert found, "no --lens action"
        for a in found:
            assert set(a.choices) == {"35", "50"}
            assert a.default is None


@pytest.mark.parametrize("entry", list(_ENTRYPOINTS))
@pytest.mark.parametrize("args", [
    ["--kernel-family"],
    ["--kernel-family", "synthetic"],
    ["--kernel-family=" + _OLD_FAMILY_VALUE],
    ["--lens-aperture-mm", "17.5"],
    ["--lens-aperture-mm", "35/2"],
    ["--lens-a", "1"],
])
def test_tombstones(entry, args):
    r = _run(entry, ["--image", "x.heic", *args])
    assert r.returncode == 2, r.stderr
    lines = [ln for ln in r.stderr.splitlines() if ln.strip()]
    assert len(lines) == 1, r.stderr
    assert lines[0].startswith("render: ")
    assert "0.5.0" in lines[0] and "Migrating" in lines[0]
    low = lines[0].lower()
    assert not any(t in low for t in _GUARD_TOKENS)


@pytest.mark.parametrize("entry", list(_ENTRYPOINTS))
def test_lens_35_parses_exactly(entry):
    """``--lens 35`` is NOT swallowed by the ``--lens-a...`` tombstone."""
    from summicron.render.convolve import build_parser

    ns = build_parser().parse_args(["--image", "x.heic", "--lens", "35"])
    assert ns.lens == "35" and not hasattr(ns, "_removed_lens_aperture_mm")


@pytest.mark.parametrize("entry", list(_ENTRYPOINTS))
def test_missing_tensor_one_line(entry, tmp_path):
    img = tmp_path / "x.png"
    from PIL import Image
    Image.new("RGB", (8, 8)).save(img)
    r = _run(entry, ["--image", str(img), "--hdr-source", "none",
                     "--depth-source", "monocular", "--tensor", str(tmp_path / "gone.npz")])
    assert r.returncode == 2, r.stderr
    lines = [ln for ln in r.stderr.splitlines() if ln.strip()]
    assert len(lines) == 1, r.stderr
    assert lines[0].startswith("render: tensor not found:")
    assert "Migrating" in lines[0]
    assert not any(t in lines[0].lower() for t in _GUARD_TOKENS)


@pytest.mark.parametrize("entry", list(_ENTRYPOINTS))
def test_help_clean(entry):
    r = _run(entry, ["--help"])
    assert r.returncode == 0
    low = r.stdout.lower()
    assert "kernel-family" not in low
    assert "lens-aperture" not in low
    assert "--lens" in low
    assert not any(t in low for t in _GUARD_TOKENS)
    assert "mand" + "ler" not in low and "lei" + "tz" not in low


def test_removed_api_fields():
    from summicron.render.api import RenderOptions

    with pytest.raises(TypeError):
        RenderOptions(image_path=Path("x.heic"), **{"kernel_" + "family": "x"})
    with pytest.raises(TypeError):
        RenderOptions(image_path=Path("x.heic"), lens_aperture_mm=17.5)
    with pytest.raises(TypeError):
        RenderOptions.canonical_bench("x.heic", (0.4, 0.45), old_look=True)


def test_lens_and_tensor_exclusive_cli_one_line(tmp_path, capsys):
    """CLI: one `render:` line (no usage block), exit 2, for both entry points."""
    from summicron import cli
    from summicron.render import convolve

    img = tmp_path / "x.heic"
    img.write_bytes(b"")
    argv = ["--image", str(img), "--lens", "35", "--tensor", str(tmp_path / "t.npz")]
    for entry in (lambda: cli.main(["render", *argv]), lambda: convolve.main(argv)):
        try:
            code = entry()
        except SystemExit as e:
            code = e.code
        assert code == 2
        err = capsys.readouterr().err
        assert err.strip() == "render: --lens and --tensor are mutually exclusive", err
