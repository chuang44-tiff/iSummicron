"""M5 — USAF in-focus sharpness (gradient-energy ratio).

M6 — fall-off curve is [drop-if-tight] per LOCKED SPEC §3.M6; implemented in
minimal form returning a dict of per-tile ratios when requested.
"""
from __future__ import annotations

import numpy as np

from .common import gradient_energy


TILE_INSET_PX = 8
INPUT_GRAD_FLOOR = 1e-6


def _tile_bbox_inset(bbox: tuple[int, int, int, int], inset: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox
    return (x0 + inset, y0 + inset, x1 - inset, y1 - inset)


def _grad_ratio_on_tile(
    input_mono: np.ndarray, rendered_mono: np.ndarray, bbox_xyxy
) -> tuple[float, str | None]:
    x0, y0, x1, y1 = _tile_bbox_inset(tuple(int(v) for v in bbox_xyxy), TILE_INSET_PX)
    in_tile = input_mono[y0:y1, x0:x1]
    rn_tile = rendered_mono[y0:y1, x0:x1]
    if not np.isfinite(rn_tile).all():
        raise ValueError("rendered tile contains NaN/Inf")
    in_e = gradient_energy(in_tile)
    if in_e < INPUT_GRAD_FLOOR:
        return float("nan"), "input_flat"
    rn_e = gradient_energy(rn_tile)
    return rn_e / in_e, None


def m5_in_focus_sharpness(
    input_rgb: np.ndarray, rendered_rgb: np.ndarray, annotation: dict
) -> tuple[float, str | None]:
    """Return gradient-energy ratio on the 0.70m center tile."""
    tiles = annotation.get("tiles", [])
    center = None
    for tile in tiles:
        if abs(float(tile["depth_m"]) - 0.70) < 1e-6:
            center = tile
            break
    if center is None:
        return float("nan"), "no_center_tile"
    return _grad_ratio_on_tile(
        input_rgb[..., 0], rendered_rgb[..., 0], center["bbox_xyxy"]
    )


def m6_falloff_curve(
    input_rgb: np.ndarray, rendered_rgb: np.ndarray, annotation: dict
) -> dict:
    """Per-tile gradient-energy ratio keyed by depth (string). [drop-if-tight]

    Minimal implementation so the metric is invokable behind the metrics filter.
    """
    tiles = annotation.get("tiles", [])
    out: dict[str, float] = {}
    for tile in tiles:
        ratio, _flag = _grad_ratio_on_tile(
            input_rgb[..., 0], rendered_rgb[..., 0], tile["bbox_xyxy"]
        )
        out[f"{float(tile['depth_m']):.3f}"] = ratio
    return out
