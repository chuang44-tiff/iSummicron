"""USAF-depth fixture generator (LOCKED SPEC §2.3).

5-tile cross layout. Each 320x320 tile carries a minimal USAF-style
4-group binary bar chart (bar widths {16,8,4,2} px, horizontal + vertical
per width). Depths are tensor-native: [0.70, 0.98, 1.91, 3.74, 10.23].
0.70m sits at the center tile (mandatory per spec).
"""
from __future__ import annotations

from typing import Any

import numpy as np

from summicron.eval.fixtures.common import (
    FOCUS_DISTANCE_M,
    FRAME_H,
    FRAME_W,
    OPTICAL_CENTER_XY,
    TENSOR_DEPTHS_M,
    assert_depth_range,
    assert_monochrome,
    draw_usaf_tile,
)

USAF_TILE_SIZE: int = 320
USAF_BAR_WIDTHS: tuple[int, ...] = (16, 8, 4, 2)
# (center_xy, depth_m). Order matters: idx 0 is always the center 0.70m tile.
USAF_TILE_TABLE: tuple[tuple[tuple[int, int], float], ...] = (
    ((512, 512),  0.70),
    ((200, 200),  0.98),
    ((824, 200),  1.91),
    ((200, 824),  3.74),
    ((824, 824), 10.23),
)


def generate_usaf_depth() -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Generate the deterministic 5-tile USAF-depth fixture."""
    # Precondition: center tile depth must be 0.70m.
    if USAF_TILE_TABLE[0][1] != FOCUS_DISTANCE_M:
        raise ValueError(
            f"usaf_depth layout infeasible: center tile depth "
            f"{USAF_TILE_TABLE[0][1]} != mandatory focus {FOCUS_DISTANCE_M}"
        )
    # All tile depths (incl. center) must be in the tensor-native set.
    for (cx, cy), depth in USAF_TILE_TABLE:
        if float(depth) not in TENSOR_DEPTHS_M:
            raise ValueError(
                f"usaf_depth tile at ({cx},{cy}) has depth {depth}, "
                f"not in tensor-native set {TENSOR_DEPTHS_M}"
            )

    rgb = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.float32)
    depth_m = np.full((FRAME_H, FRAME_W), FOCUS_DISTANCE_M, dtype=np.float32)

    tiles: list[dict[str, Any]] = []
    for idx, ((cx, cy), depth) in enumerate(USAF_TILE_TABLE):
        half = USAF_TILE_SIZE // 2
        x0 = cx - half
        y0 = cy - half
        x1 = x0 + USAF_TILE_SIZE
        y1 = y0 + USAF_TILE_SIZE
        if x0 < 0 or y0 < 0 or x1 > FRAME_W or y1 > FRAME_H:
            raise ValueError(
                f"usaf_depth tile {idx} bbox [{x0},{y0},{x1},{y1}] "
                f"exceeds frame {FRAME_W}x{FRAME_H}"
            )
        group_bboxes = draw_usaf_tile(
            rgb, x0, y0,
            tile_size=USAF_TILE_SIZE,
            bar_widths=USAF_BAR_WIDTHS,
        )
        depth_m[y0:y1, x0:x1] = float(depth)
        tiles.append({
            "idx": idx,
            "bbox_xyxy": [int(x0), int(y0), int(x1), int(y1)],
            "depth_m": float(depth),
            "bar_widths_px": list(USAF_BAR_WIDTHS),
            "group_bboxes_xyxy": group_bboxes,
        })

    annotation: dict[str, Any] = {
        "scene": "usaf_depth",
        "frame_hw": [FRAME_H, FRAME_W],
        "optical_center_xy": list(OPTICAL_CENTER_XY),
        "focus_distance_m": float(FOCUS_DISTANCE_M),
        "tiles": tiles,
    }
    assert_monochrome(rgb)
    # No 'discs' in annotation; assert_depth_range will reject the
    # wrap-probe 0.40m unless a disc claims it → correct behaviour here.
    assert_depth_range(depth_m, None)
    return rgb, depth_m, annotation
