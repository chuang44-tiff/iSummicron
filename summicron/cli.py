"""Command-line entry point for isummicron.

Subcommands:
    isummicron render   --image INPUT [--tensor TENSOR.npz] [--output OUT] ...

Returns ``int`` exit codes that the console-script wrapper propagates
via ``sys.exit(main())``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _cmd_render(args: argparse.Namespace) -> int:
    """Render a single image through the v0 pipeline.

    In-process delegation (no subprocess) to
    :func:`summicron.render.api.render_image`. Lazy imports keep the
    cli.py import-time path lean — renderer-only deps (scipy, PIL,
    pillow_heif) load on first invocation.

    Exit-code map: 0 on success, 2 on expected user-input failure
    (``ValueError``, ``RuntimeError``, ``TensorValidationError``,
    ``FileNotFoundError``). Unhandled exceptions still raise with a
    traceback (matches existing convolve.main behavior).
    """
    from summicron.render.api import RenderOptions, render_image
    from summicron.render.convolve import (
        TensorValidationError,
        _validate_args,
        build_parser as build_render_parser,
    )

    # Re-validate using the convolve.py argparse validator so users get the
    # same error surface as `python -m summicron.render.convolve`.
    render_parser = build_render_parser()
    try:
        _validate_args(args, render_parser)
    except SystemExit as e:
        return int(e.code) if e.code is not None else 2

    opts = RenderOptions.from_argparse(args)
    try:
        render_image(opts)
    except (ValueError, RuntimeError, FileNotFoundError, TensorValidationError) as e:
        print(f"render: {e}", file=sys.stderr)
        return 2
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="isummicron")
    sub = parser.add_subparsers(dest="command", required=True)

    rnd = sub.add_parser(
        "render",
        help="Render a single image through the v0 pipeline (HEIC/JPG/PNG in, JPG/PNG out)",
    )
    rnd.add_argument("--image", required=True, help="Input image path (HEIC/JPEG/PNG)")
    rnd.add_argument(
        "--tensor",
        default=None,
        help="4D PSF tensor .npz path. When omitted, defaults to the wheel-shipped "
             "tensor (summicron.render._artifacts.default_tensor_path()).",
    )
    rnd.add_argument(
        "--out-dir",
        default=str(Path(__file__).resolve().parents[1] / "runs" / "render"),
        help="Output directory (used when --output is not set)",
    )
    rnd.add_argument(
        "--output",
        default=None,
        help="Explicit output JPG/PNG path. Overrides --out-dir/auto-naming when set.",
    )
    rnd.add_argument("--depth-source", choices=["lidar", "monocular"], default="lidar")
    rnd.add_argument("--depth-scale", type=float, default=1.0)
    rnd.add_argument("--focus-xy", nargs=2, type=float, default=None, metavar=("X", "Y"))
    rnd.add_argument("--focus-target-m", type=float, default=None)
    rnd.add_argument("--dof-m", type=float, default=0.0)
    rnd.add_argument("--downres", type=int, default=None)
    rnd.add_argument("--psf-norm", choices=["l1", "peak"], default="l1")
    rnd.add_argument("--hdr-source", choices=["none", "macos"], default="none")
    rnd.add_argument("--bright-pass-thresh", type=float, default=1.0)
    rnd.add_argument("--bright-pass-region-size", type=int, default=31)
    rnd.add_argument("--bright-pass-floor", type=float, default=0.10)
    rnd.add_argument("--bright-pass-norm-pct", type=float, default=99.0)
    rnd.add_argument("--shadow-match-mode", choices=["off", "on"], default="off")
    rnd.add_argument("--shadow-band-hi", type=float, default=0.70)
    rnd.add_argument("--shadow-match-max-gain", type=float, default=4.0)
    rnd.add_argument("--shadow-match-strength", type=float, default=1.0)
    rnd.add_argument(
        "--slab-batch-size",
        type=int,
        default=None,
        help="Process depth slabs in batches of N (default 4). Lower = less peak RAM, "
             "no speed cost. Higher than D collapses to all-at-once.",
    )
    rnd.set_defaults(func=_cmd_render)

    return parser


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    parser = build_parser()
    args = parser.parse_args(argv)
    rc = args.func(args)
    return int(rc) if rc is not None else 0


if __name__ == "__main__":
    sys.exit(main())
