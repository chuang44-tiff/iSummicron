"""CLI: ``python -m summicron.eval --input ... --rendered ... --output ...``.

Thin wrapper over ``score()``. Loads input + rendered .npz pairs (each
containing ``rgb``, ``depth_m``, ``annotation`` — the last stored as a
0-d object array of a dict, or as separate JSON sidecar next to the .npz).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .score import DEFAULT_METRICS, score


def _load_fixture(npz_path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    data = np.load(str(npz_path), allow_pickle=True)
    rgb = np.asarray(data["rgb"], dtype=np.float32)
    depth = np.asarray(data["depth_m"], dtype=np.float32)
    # Prefer sidecar .json; fall back to an 'annotation' key stored in the npz.
    sidecar = npz_path.with_suffix(".json")
    if sidecar.exists():
        with sidecar.open("r") as fh:
            annotation = json.load(fh)
    elif "annotation" in data.files:
        ann_arr = data["annotation"]
        if ann_arr.dtype == object:
            annotation = ann_arr.item()
        else:
            annotation = json.loads(str(ann_arr))
    else:
        raise FileNotFoundError(
            f"No annotation sidecar {sidecar} and no 'annotation' key in {npz_path}"
        )
    return rgb, depth, annotation


def main(argv: list[str] | None = None) -> int:
    """Score a rendered fixture against its input + annotation.

    P1.audit Item 9: signature is now ``main(argv) -> int`` so the
    in-process CLI delegation in :mod:`summicron.cli` can propagate
    the exit code.
    """
    parser = argparse.ArgumentParser(
        description="Score a rendered fixture against its input + annotation."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input fixture .npz (rgb/depth_m + sidecar .json)",
    )
    parser.add_argument(
        "--rendered",
        required=True,
        help="Rendered fixture .npz (rgb/depth_m + sidecar .json, annotation carried through)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSON report path",
    )
    parser.add_argument(
        "--metrics",
        default=",".join(DEFAULT_METRICS),
        help="Comma-separated metric IDs (default: %(default)s)",
    )
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    rendered_path = Path(args.rendered)
    output_path = Path(args.output)

    metrics = tuple(m.strip() for m in args.metrics.split(",") if m.strip())

    input_rgb, _input_depth, input_ann = _load_fixture(input_path)
    rendered_rgb, depth_m, annotation = _load_fixture(rendered_path)

    if input_ann.get("scene") != annotation.get("scene"):
        raise ValueError(
            f"Scene mismatch: input={input_ann.get('scene')!r} "
            f"rendered={annotation.get('scene')!r}"
        )

    report = score(
        rendered_rgb=rendered_rgb,
        depth_m=depth_m,
        annotation=annotation,
        input_rgb=input_rgb,
        metrics=metrics,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as fh:
        json.dump(report, fh, indent=2)
    print(f"Wrote eval report: {output_path}")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
