"""P1.9 adversarial: older .npz without delta_guard_applied must warn + default to False."""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np

from summicron.render._artifacts import default_tensor_path
from summicron.render.convolve import load_tensor


def test_delta_guard_schema_legacy_npz(tmp_path: Path):
    data = dict(np.load(str(default_tensor_path("35")), allow_pickle=False))
    assert bool(data["delta_guard_applied"]) is True  # shipped artifact carries the guard
    data.pop("delta_guard_applied", None)
    legacy = tmp_path / "legacy.npz"
    np.savez(str(legacy), **data)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = load_tensor(legacy)
    assert result["delta_guard_applied"] is False
    assert any("delta_guard_applied=False" in str(w.message) for w in caught)
