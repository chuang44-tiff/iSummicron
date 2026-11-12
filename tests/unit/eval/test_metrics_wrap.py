"""M7 known-answer: equal blobs at disc #11 and disc #3 => diameter_ratio ~1.0.

Per LOCKED SPEC item 17.
"""
from __future__ import annotations

import numpy as np

from summicron.eval.metrics import m7_wrap_probe
from tests.unit.eval.conftest import (
    bubble_grid_annotation_stub,
    mono_rgb,
    rasterize_disc,
    standard_grid_discs,
)


def test_m7_equal_blobs_ratio_one():
    h = w = 1024
    mono = np.zeros((h, w), dtype=np.float32)
    # radius 20 = diameter 40 at both disc #11 (720, 512) and disc #3 (800, 512)
    mono += rasterize_disc(h, w, 720, 512, radius=20.0)
    mono += rasterize_disc(h, w, 800, 512, radius=20.0)
    mono = np.clip(mono, 0.0, 1.0)
    rgb = mono_rgb(mono)
    ann = bubble_grid_annotation_stub(standard_grid_discs())
    result = m7_wrap_probe(rgb, ann)
    assert result["finite_ok"] is True
    assert result["support_px"] >= 20
    assert abs(result["diameter_ratio"] - 1.0) < 0.05
    assert result["flag"] is None
