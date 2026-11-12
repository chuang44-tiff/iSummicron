"""M5 known-answer: identity through trivial pipeline => ratio ~1.0 ± 0.01."""
from __future__ import annotations

import numpy as np

from summicron.eval.metrics import m5_in_focus_sharpness
from tests.unit.eval.conftest import mono_rgb


def _usaf_tile_mono(h: int = 1024, w: int = 1024) -> np.ndarray:
    """Draw a minimal bar pattern at center tile bbox (352, 352, 672, 672)."""
    mono = np.zeros((h, w), dtype=np.float32)
    # simple vertical bars, 8 px wide, 8 px gap, within the tile
    x0, y0, x1, y1 = 352, 352, 672, 672
    for x in range(x0 + 16, x1 - 16, 16):
        mono[y0 + 16 : y1 - 16, x : x + 8] = 1.0
    # simple horizontal bars
    for y in range(y0 + 16, y1 - 16, 32):
        mono[y : y + 4, x0 + 16 : x1 - 16] = 1.0
    return mono


def test_m5_identity_through_trivial_pipeline():
    mono = _usaf_tile_mono()
    rgb = mono_rgb(mono)
    ann = {
        "scene": "usaf_depth",
        "frame_hw": [1024, 1024],
        "optical_center_xy": [512, 512],
        "focus_distance_m": 0.70,
        "tiles": [
            {
                "idx": 0,
                "bbox_xyxy": [352, 352, 672, 672],
                "depth_m": 0.70,
                "bar_widths_px": [16, 8, 4, 2],
                "group_bboxes_xyxy": [],
            },
            {
                "idx": 1,
                "bbox_xyxy": [40, 40, 360, 360],
                "depth_m": 0.98,
                "bar_widths_px": [16, 8, 4, 2],
                "group_bboxes_xyxy": [],
            },
        ],
    }
    val, flag = m5_in_focus_sharpness(rgb, rgb, ann)
    assert flag is None
    assert abs(val - 1.0) < 0.01
