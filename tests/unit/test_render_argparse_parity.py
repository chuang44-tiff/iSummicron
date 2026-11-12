"""CLI-1a C3 — single argparse source parity.

``python -m summicron.render.convolve`` and ``isummicron render`` both build
their render flags via ``summicron.render._render_args.add_render_arguments``.
These tests pin that the two surfaces and ``RenderOptions`` cannot drift.
"""
from __future__ import annotations

import argparse
import dataclasses
import subprocess
import sys

import pytest

from summicron.render._bench import CANONICAL_FLAGS, G1_FOCUS_XY, G1_IMAGE

# argparse dest -> RenderOptions field (mirror of api._ARGPARSE_ALIAS).
_ALIAS = {"image": "image_path", "tensor": "tensor_path", "output": "out_path"}


def _convolve_parser() -> argparse.ArgumentParser:
    from summicron.render.convolve import build_parser

    return build_parser()


def _cli_render_parser() -> argparse.ArgumentParser:
    from summicron.cli import build_parser

    parser = build_parser()
    sub = next(
        a for a in parser._actions if isinstance(a, argparse._SubParsersAction)
    )
    return sub.choices["render"]


def _spec(parser: argparse.ArgumentParser) -> dict:
    return {
        a.dest: (a.default, a.type, a.choices, a.nargs, a.required)
        for a in parser._actions
        if not isinstance(a, argparse._HelpAction)
    }


def _canonical_argv(*extra: str) -> list[str]:
    return [
        "--image", G1_IMAGE,
        *CANONICAL_FLAGS,
        "--focus-xy", *(str(v) for v in G1_FOCUS_XY),
        "--output", "out.jpg",
        *extra,
    ]


def test_convolve_and_cli_render_flags_identical():
    """(i) dest, default, type, choices, nargs, required all match."""
    assert _spec(_convolve_parser()) == _spec(_cli_render_parser())


def test_render_dests_match_render_options_fields():
    """(ii) parser dests == fields(RenderOptions) - {return_linear}, mod alias."""
    from summicron.render.api import RenderOptions

    dests = {_ALIAS.get(d, d) for d in _spec(_convolve_parser())}
    names = {f.name for f in dataclasses.fields(RenderOptions)} - {"return_linear", "save"}
    assert dests == names


@pytest.mark.parametrize("extra", [(), ("--occlusion-compositing", "on")])
def test_canonical_argv_gives_equal_render_options(extra):
    """(iii) canonical bench argv parses to equal RenderOptions via both."""
    from summicron.render.api import RenderOptions

    argv = _canonical_argv(*extra)
    a = RenderOptions.from_argparse(_convolve_parser().parse_args(argv))
    b = RenderOptions.from_argparse(_cli_render_parser().parse_args(argv))
    assert a == b
    assert a.focus_xy == G1_FOCUS_XY
    assert a.occlusion_compositing == ("on" if extra else "off")


@pytest.mark.parametrize("which", ["convolve", "cli"])
def test_dof_m_rejected(which):
    """(iv) --dof-m is gone from both surfaces (CLI-1a C2)."""
    if which == "convolve":
        parse = _convolve_parser().parse_args
        argv = _canonical_argv("--dof-m", "0.1")
    else:
        from summicron.cli import build_parser

        parse = build_parser().parse_args
        argv = ["render", *_canonical_argv("--dof-m", "0.1")]
    with pytest.raises(SystemExit) as exc:
        parse(argv)
    assert exc.value.code == 2


def test_isummicron_render_accepts_occlusion_compositing_on():
    """(v) the flag missing from `isummicron render` pre-CLI-1a now parses."""
    from summicron.cli import build_parser

    ns = build_parser().parse_args(
        ["render", *_canonical_argv("--occlusion-compositing", "on")]
    )
    assert ns.occlusion_compositing == "on"


def test_cli_build_parser_does_not_import_convolve():
    """Import fence: building the cli parser stays free of renderer deps."""
    code = (
        "import sys, summicron.cli; summicron.cli.build_parser(); "
        "sys.exit(1 if 'summicron.render.convolve' in sys.modules else 0)"
    )
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert res.returncode == 0, (
        f"summicron.cli.build_parser() imported summicron.render.convolve\n"
        f"{res.stderr}"
    )


@pytest.mark.parametrize("which", ["convolve", "cli"])
def test_focus_xy_out_of_range_rejected(which):
    """--focus-xy outside [0, 1] is rejected on both entrypoints (exit 2).

    Restores coverage lost with the deleted dof-CLI adversarial file.
    """
    bad = ("--focus-xy", "1.5", "0.5")
    if which == "convolve":
        from summicron.render.convolve import _validate_args

        parser = _convolve_parser()
        args = parser.parse_args(["--image", G1_IMAGE, *bad])
        with pytest.raises(SystemExit) as exc:
            _validate_args(args, parser)
        assert exc.value.code == 2
    else:
        from summicron.cli import main

        try:
            rc = main(["render", "--image", G1_IMAGE, *bad])
        except SystemExit as e:
            rc = e.code
        assert rc == 2
