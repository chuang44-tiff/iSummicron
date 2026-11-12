"""Round-trip identity check for the EXIF orientation helper.

The helper maps display->storage. We verify by hand-rolling each inverse map
(storage->display) and checking composition is identity on a 100-point grid.
"""
from __future__ import annotations

import numpy as np
import pytest

from summicron.tools._orientation import display_to_storage_norm

_INVERSES = {
    1: lambda x, y: (x, y),
    2: lambda x, y: (1.0 - x, y),
    3: lambda x, y: (1.0 - x, 1.0 - y),
    4: lambda x, y: (x, 1.0 - y),
    5: lambda x, y: (y, x),
    6: lambda x, y: (y, 1.0 - x),       # forward: x_s=1-v, y_s=u
    7: lambda x, y: (1.0 - y, 1.0 - x),  # forward: x_s=1-v, y_s=1-u
    8: lambda x, y: (1.0 - y, x),        # forward: x_s=v, y_s=1-u
}


@pytest.mark.parametrize("orientation", list(range(1, 9)))
def test_orientation_8way_roundtrip(orientation):
    rng = np.random.default_rng(seed=42 + orientation)
    pts = rng.uniform(0.0, 1.0, size=(100, 2))
    w, h = 1000, 800
    inv = _INVERSES[orientation]
    for u, v in pts:
        x_s, y_s = display_to_storage_norm(u * w, v * h, w, h, orientation)
        u_back, v_back = inv(x_s, y_s)
        assert pytest.approx(u, abs=1e-6) == u_back
        assert pytest.approx(v, abs=1e-6) == v_back


def test_orientation_invalid_raises():
    with pytest.raises(ValueError):
        display_to_storage_norm(10.0, 20.0, 100, 100, 0)
    with pytest.raises(ValueError):
        display_to_storage_norm(10.0, 20.0, 100, 100, 9)
