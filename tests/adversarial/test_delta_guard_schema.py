"""P1.9 adversarial: older .npz without delta_guard_applied must warn + default to False."""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest

from summicron.psf_model.downscale_tensor import downscale_tensor
from summicron.render.convolve import load_tensor

REAL_CODEV = Path(__file__).resolve().parents[2] / "runs" / "ag_dblgauss" / "ag_dblgauss_f2.0_render.npz"


@pytest.mark.skipif(not REAL_CODEV.exists(), reason="real CodeV tensor not present")
def test_delta_guard_schema_legacy_npz(tmp_path: Path):
    regen = tmp_path / "regen.npz"
    downscale_tensor(REAL_CODEV, regen, target_long_edge=5712)
    data = dict(np.load(str(regen), allow_pickle=True))
    data.pop("delta_guard_applied", None)
    legacy = tmp_path / "legacy.npz"
    np.savez(str(legacy), **data)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = load_tensor(legacy)
    assert result["delta_guard_applied"] is False
    assert any("delta_guard_applied=False" in str(w.message) for w in caught)
