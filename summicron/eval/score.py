"""Top-level eval scoring dispatcher.

Routes by ``annotation["scene"]`` and returns a flat JSON-serializable dict
of numeric metric values plus sibling ``<metric>__flag`` keys.

Scene -> metric defaults:
  bubble_grid   -> M1 + M2 + M3 + M4 + M7
  bubble_random -> M1 + M2 + M3 + M4
  usaf_depth    -> M1 + M5 (+ M6 if requested)

A caller may override via ``metrics=`` tuple. Metrics requested but
inapplicable to the scene are skipped with a ``__flag`` note.
"""
from __future__ import annotations

import datetime as _dt

import numpy as np

from .metrics import (
    energy_ratio,
    m2_roundness,
    m3_sag_tan,
    m4_diameter_depth_monotonicity,
    m5_in_focus_sharpness,
    m6_falloff_curve,
    m7_wrap_probe,
)


DEFAULT_METRICS = ("M1", "M2", "M3", "M4", "M5", "M7")
_BUBBLE_SCENES = {"bubble_grid", "bubble_random"}
_USAF_SCENES = {"usaf_depth"}


def _py(value):
    """Recursively coerce numpy scalars to python floats/ints for JSON."""
    if isinstance(value, dict):
        return {k: _py(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_py(v) for v in value]
    if isinstance(value, tuple):
        return [_py(v) for v in value]
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def score(
    rendered_rgb: np.ndarray,
    depth_m: np.ndarray,
    annotation: dict,
    input_rgb: np.ndarray | None = None,
    metrics: tuple[str, ...] = DEFAULT_METRICS,
) -> dict:
    """Compute eval metrics on a rendered fixture.

    Parameters
    ----------
    rendered_rgb : np.ndarray, (H, W, 3) float32
        Renderer output.
    depth_m : np.ndarray, (H, W) float32
        Depth map from the fixture (for diagnostics; not presently consumed
        by any metric since discs carry per-annotation depth).
    annotation : dict
        Fixture annotation (loaded from sidecar .json).
    input_rgb : np.ndarray | None
        Required for M1 and M5. Raises ValueError when requested metric
        needs it and it is missing.
    metrics : tuple[str, ...]
        Metric IDs to run. Unknown IDs raise ValueError. Defaults to
        ``("M1", "M2", "M3", "M4", "M5", "M7")``.

    Returns
    -------
    dict
        Flat, JSON-serializable report.
    """
    if rendered_rgb.ndim != 3 or rendered_rgb.shape[-1] != 3:
        raise ValueError(f"rendered_rgb must be (H, W, 3), got {rendered_rgb.shape}")
    if depth_m.ndim != 2:
        raise ValueError(f"depth_m must be 2D, got {depth_m.shape}")
    if depth_m.shape != rendered_rgb.shape[:2]:
        raise ValueError(
            f"depth_m shape {depth_m.shape} must match rendered_rgb {rendered_rgb.shape[:2]}"
        )
    if "scene" not in annotation:
        raise ValueError("annotation missing 'scene' key")
    if "discs" not in annotation and annotation["scene"] in _BUBBLE_SCENES:
        raise ValueError("bubble annotation missing 'discs' list")
    if "tiles" not in annotation and annotation["scene"] in _USAF_SCENES:
        raise ValueError("usaf annotation missing 'tiles' list")

    scene = annotation["scene"]
    known = {"M1", "M2", "M3", "M4", "M5", "M6", "M7"}
    unknown = set(metrics) - known
    if unknown:
        raise ValueError(f"unknown metric IDs: {sorted(unknown)}")

    report: dict = {}

    # M1 — energy (both bubble and usaf)
    if "M1" in metrics:
        if input_rgb is None:
            raise ValueError("M1 requires input_rgb")
        ratio, flag = energy_ratio(input_rgb, rendered_rgb)
        report["M1"] = ratio
        report["M1__flag"] = flag

    # M2/M3/M4 — bubble only
    if "M2" in metrics:
        if scene in _BUBBLE_SCENES:
            val, flag = m2_roundness(rendered_rgb, annotation)
            report["M2"] = val
            report["M2__flag"] = flag
        else:
            report["M2"] = float("nan")
            report["M2__flag"] = "scene_not_applicable"

    if "M3" in metrics:
        if scene in _BUBBLE_SCENES:
            report["M3"] = m3_sag_tan(rendered_rgb, annotation)
            report["M3__flag"] = None
        else:
            report["M3"] = []
            report["M3__flag"] = "scene_not_applicable"

    if "M4" in metrics:
        if scene in _BUBBLE_SCENES:
            rhos, col_flags = m4_diameter_depth_monotonicity(
                rendered_rgb, annotation
            )
            report["M4"] = rhos
            for col, fl in col_flags.items():
                report[f"M4__flag_{col}"] = fl
        else:
            report["M4"] = {}
            report["M4__flag"] = "scene_not_applicable"

    # M5/M6 — usaf only
    if "M5" in metrics:
        if scene in _USAF_SCENES:
            if input_rgb is None:
                raise ValueError("M5 requires input_rgb")
            val, flag = m5_in_focus_sharpness(input_rgb, rendered_rgb, annotation)
            report["M5"] = val
            report["M5__flag"] = flag
        else:
            report["M5"] = float("nan")
            report["M5__flag"] = "scene_not_applicable"

    if "M6" in metrics:
        if scene in _USAF_SCENES:
            if input_rgb is None:
                raise ValueError("M6 requires input_rgb")
            report["M6"] = m6_falloff_curve(input_rgb, rendered_rgb, annotation)
            report["M6__flag"] = None
        else:
            report["M6"] = {}
            report["M6__flag"] = "scene_not_applicable"

    # M7 — wrap probe (bubble_grid only)
    if "M7" in metrics:
        if scene in _BUBBLE_SCENES and any(
            d.get("wrap_applied", False) for d in annotation.get("discs", [])
        ):
            probe = m7_wrap_probe(rendered_rgb, annotation)
            flag = probe.pop("flag")
            report["M7"] = probe
            report["M7__flag"] = flag
        else:
            report["M7"] = {
                "finite_ok": False,
                "support_px": 0,
                "diameter_ratio": float("nan"),
            }
            report["M7__flag"] = "scene_has_no_wrap_probe"

    report["_meta"] = {
        "scene": scene,
        "fixture_name": annotation.get("scene"),
        "n_discs": len(annotation.get("discs", [])),
        "n_tiles": len(annotation.get("tiles", [])),
        "frame_hw": list(annotation.get("frame_hw", list(rendered_rgb.shape[:2]))),
        "generated_at": _dt.datetime.now(_dt.timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "metrics_requested": list(metrics),
    }

    return _py(report)
