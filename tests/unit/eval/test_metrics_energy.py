"""M1 known-answer: energy(x, x) == 1.0 ± 1e-6."""
from __future__ import annotations

import numpy as np
import pytest

from summicron.eval.metrics import energy_ratio


def test_m1_identity(std_bubble_grid_fixture):
    rgb, _depth, _ann = std_bubble_grid_fixture
    ratio, flag = energy_ratio(rgb, rgb)
    assert flag is None
    assert abs(ratio - 1.0) < 1e-6


def test_m1_scaled():
    rgb = np.ones((64, 64, 3), dtype=np.float32) * 0.5
    # make sure sum > floor (64*64*0.5 = 2048)
    ratio, flag = energy_ratio(rgb, rgb * 2.0)
    assert flag is None
    assert abs(ratio - 2.0) < 1e-6


def test_m1_too_dark_flag():
    rgb = np.zeros((64, 64, 3), dtype=np.float32)
    ratio, flag = energy_ratio(rgb, rgb)
    assert np.isnan(ratio)
    assert flag == "input_too_dark"
