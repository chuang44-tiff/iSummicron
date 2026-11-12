"""CLI for generating eval fixtures.

Usage::

    python -m summicron.eval.fixtures --scene bubble_grid \
        --out runs/eval/bubble_grid_input.npz

    python -m summicron.eval.fixtures --scene bubble_random --seed 42 \
        --out runs/eval/bubble_random_seed42_input.npz

    python -m summicron.eval.fixtures --scene usaf_depth \
        --out runs/eval/usaf_depth_input.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path

from summicron.eval.fixtures.bubble import (
    generate_bubble_grid,
    generate_bubble_random,
)
from summicron.eval.fixtures.common import save_fixture, sha256_file
from summicron.eval.fixtures.usaf import generate_usaf_depth

SCENES = ("bubble_grid", "bubble_random", "usaf_depth")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate an eval-harness fixture (.npz + .json)."
    )
    parser.add_argument("--scene", choices=SCENES, required=True)
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Seed for bubble_random (ignored for deterministic scenes).",
    )
    parser.add_argument(
        "--out", type=Path, required=True,
        help="Output .npz path; .json sidecar written alongside.",
    )
    args = parser.parse_args()

    if args.scene == "bubble_grid":
        rgb, depth_m, annotation = generate_bubble_grid()
    elif args.scene == "bubble_random":
        rgb, depth_m, annotation = generate_bubble_random(seed=args.seed)
    elif args.scene == "usaf_depth":
        rgb, depth_m, annotation = generate_usaf_depth()
    else:  # pragma: no cover — argparse enforces.
        raise AssertionError(f"unreachable: scene={args.scene}")

    out = save_fixture(args.out, rgb, depth_m, annotation)
    digest = sha256_file(out)
    print(f"Wrote {out} (sha256={digest}) + sidecar {out.with_suffix('.json')}")


if __name__ == "__main__":
    main()
