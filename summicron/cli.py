"""Command-line entry point for isummicron.

Currently implemented:
    isummicron render   --image INPUT --tensor TENSOR.npz [--output OUT] ...
    isummicron eval     --input INPUT.npz --rendered RENDERED.npz --output REPORT.json
                        [--metrics m1,m2,...]
"""

from __future__ import annotations

import argparse
import sys


def _cmd_render(args: argparse.Namespace) -> int:
    """Render a single image through the v0 pipeline.

    P1.audit Item 2: in-process delegation (no subprocess) to
    :func:`summicron.render.api.render_image`. Lazy imports keep the
    cli.py import-time path lean — renderer-only deps (scipy, PIL,
    pillow_heif) load on first invocation.

    Exit-code map: 0 on success, 2 on expected user-input failure
    (``ValueError``, ``RuntimeError``, ``TensorValidationError``,
    ``FileNotFoundError``). Unhandled exceptions still raise with a
    traceback (matches existing convolve.main behavior).
    """
    # Local imports per cli.py:53-57 lazy-import discipline.
    from summicron.render.api import RenderOptions, render_image
    from summicron.render.convolve import (
        _validate_args,
        run_cli_render,
        build_parser as build_render_parser,
    )

    # Re-validate using the convolve.py argparse validator so users get the
    # same error surface as `python -m summicron.render.convolve`.
    render_parser = build_render_parser()
    try:
        _validate_args(args, render_parser)
    except SystemExit as e:
        # parser.error() raises SystemExit; propagate code as-is.
        return int(e.code) if e.code is not None else 2

    opts = RenderOptions.from_argparse(args)
    return run_cli_render(lambda: render_image(opts))


def _cmd_eval(args: argparse.Namespace) -> int:
    """Pass-through to ``python -m summicron.eval``.

    P1.audit Item 2: ``--`` collects the eval-CLI argv via
    ``argparse.REMAINDER`` so the eval subcommand surface
    (``--input/--rendered/--output/--metrics``) is inherited verbatim
    without flag-translation drift.
    """
    from summicron.eval.__main__ import main as eval_main

    eval_argv = list(getattr(args, "eval_argv", None) or [])
    try:
        return int(eval_main(argv=eval_argv))
    except (ValueError, RuntimeError, FileNotFoundError) as e:
        print(f"eval: {e}", file=sys.stderr)
        return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="isummicron")
    sub = parser.add_subparsers(dest="command", required=True)

    # --- Active v0 subcommands (P1.audit Item 2) ---------------------------
    rnd = sub.add_parser(
        "render",
        help="Render a single image through the v0 pipeline (HEIC/JPG/PNG in, JPG/PNG out)",
    )
    # CLI-1a C3: single argparse source shared with
    # `python -m summicron.render.convolve` (stdlib-only module, so the
    # renderer's numpy/scipy/PIL deps stay lazy in _cmd_render).
    from summicron.render._render_args import add_render_arguments

    add_render_arguments(rnd)
    rnd.set_defaults(func=_cmd_render)

    evl = sub.add_parser(
        "eval",
        help="Score a rendered fixture pair (delegates to `python -m summicron.eval`)",
    )
    evl.add_argument(
        "eval_argv",
        nargs=argparse.REMAINDER,
        help="Pass-through args; see `python -m summicron.eval --help`",
    )
    evl.set_defaults(func=_cmd_eval)

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint. Returns the subcommand exit code so callers
    (console-script, ``python -m summicron.cli``, tests) can propagate
    via ``sys.exit(main())``.

    P1.audit Item 2 fix-round (F1): the eval subcommand intercepts
    BEFORE parse_args because ``argparse.REMAINDER`` does not capture
    flag-like tokens that appear before any positional starts consuming.
    The natural form ``isummicron eval --input X --rendered Y --output Z``
    would otherwise fail at the parent parser with
    "unrecognized arguments: --input". Routing the post-"eval" argv
    directly to ``summicron.eval.__main__.main`` lets eval's own
    argparse handle its full flag surface.
    """
    if argv is None:
        argv = sys.argv[1:]

    if argv and argv[0] == "eval":
        from summicron.eval.__main__ import main as eval_main

        try:
            rc = eval_main(argv=argv[1:])
        except (ValueError, RuntimeError, FileNotFoundError) as e:
            print(f"eval: {e}", file=sys.stderr)
            return 2
        return int(rc) if rc is not None else 0

    parser = build_parser()
    args = parser.parse_args(argv)
    rc = args.func(args)
    return int(rc) if rc is not None else 0


if __name__ == "__main__":
    sys.exit(main())
