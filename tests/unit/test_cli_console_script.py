"""CLI console-script surface (P1.audit Item 2 + Item 9 exit propagation).

In-process tests against ``summicron.cli.build_parser()`` and ``main()``;
one slow subprocess test exercises the installed entry point's ``--help``
without invoking the renderer.
"""
from __future__ import annotations

import subprocess
import sys

import pytest


def test_render_help_succeeds() -> None:
    """``isummicron render --help`` exits 0 (no traceback)."""
    from summicron.cli import build_parser

    parser = build_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["render", "--help"])
    assert exc.value.code == 0


def test_eval_help_succeeds() -> None:
    """``isummicron eval --help`` exits 0."""
    from summicron.cli import main

    # P1.audit Item 2 fix-round (F1): main() intercepts eval BEFORE
    # parse_args because argparse.REMAINDER does not capture flag-like
    # tokens before a positional. The intercept routes argv[1:] to
    # eval.__main__.main, whose own argparse handles --help (sys.exit(0)).
    with pytest.raises(SystemExit) as exc:
        main(["eval", "--help"])
    assert exc.value.code == 0


def test_eval_passthrough_natural_form(monkeypatch) -> None:
    """``isummicron eval --input X --rendered Y --output Z`` reaches eval.__main__.main.

    Adversarial F1 (BLOCKING): the spec § Item 2 acceptance criterion
    cites this exact natural form as working. argparse.REMAINDER does
    NOT capture flag-like tokens before a positional, so the previous
    REMAINDER-based plumbing failed at the parent parser. main() now
    intercepts argv[0] == "eval" before parse_args.
    """
    captured: list[list[str] | None] = []

    def fake_eval_main(argv: list[str] | None = None) -> int:
        captured.append(argv)
        return 0

    monkeypatch.setattr("summicron.eval.__main__.main", fake_eval_main)
    from summicron.cli import main

    rc = main([
        "eval",
        "--input", "/p/in.npz",
        "--rendered", "/p/r.npz",
        "--output", "/p/o.json",
    ])
    assert rc == 0
    assert captured == [[
        "--input", "/p/in.npz",
        "--rendered", "/p/r.npz",
        "--output", "/p/o.json",
    ]]


def test_eval_error_returns_2(monkeypatch) -> None:
    """eval.__main__.main raising (ValueError, RuntimeError, FileNotFoundError)
    is caught by the cli.main intercept and returns rc=2."""
    def fake_eval_main(argv: list[str] | None = None) -> int:
        raise FileNotFoundError("missing fixture")

    monkeypatch.setattr("summicron.eval.__main__.main", fake_eval_main)
    from summicron.cli import main

    rc = main(["eval", "--input", "/missing.npz"])
    assert rc == 2


def test_subcommands_are_render_and_eval_only() -> None:
    """SHIP-3: the shelved subcommands are gone; only render/eval remain."""
    import argparse

    from summicron.cli import build_parser

    parser = build_parser()
    actions = [
        a for a in parser._actions if isinstance(a, argparse._SubParsersAction)
    ]
    assert len(actions) == 1
    assert set(actions[0].choices) == {"render", "eval"}


@pytest.mark.slow
def test_console_script_render_help_subprocess() -> None:
    """``python -m summicron.cli render --help`` exits 0 in a subprocess."""
    proc = subprocess.run(
        [sys.executable, "-m", "summicron.cli", "render", "--help"],
        capture_output=True, text=True, check=False,
    )
    assert proc.returncode == 0, (
        f"unexpected exit {proc.returncode}: stderr={proc.stderr!r}"
    )
    assert "render" in proc.stdout.lower()
    # No traceback on the happy path
    assert "Traceback" not in proc.stderr
