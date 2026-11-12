"""Diopter depth->defocus mapping (Q-d, leaf module; numpy only).

Maps unscaled metric LiDAR depth to an *equivalent object distance* on the
tensor's distance axis, so that a pixel with diopter offset ``dp`` from the
focus anchor is rendered like a real object that has the same diopter offset
from the simulated focus distance.  See
``doc/plan/phase2/section20_depth-diopter/locked-spec-Q-d.md`` section 2.2.

No imports from ``summicron.render``: the meters path must stay isolated.
"""
from __future__ import annotations

import numpy as np


def _diopter_offset(depth_m, anchor_m: float, eps_m: float) -> np.ndarray:
    """``|1/z - 1/anchor|`` in float64; non-finite depth -> NaN."""
    d = np.asarray(depth_m, dtype=np.float64)
    inv = 1.0 / np.maximum(d, eps_m)
    inv_a = 1.0 / max(float(anchor_m), eps_m)
    dp = np.abs(inv - inv_a)
    return np.where(np.isfinite(d), dp, np.nan)


def diopter_equivalent_depth(
    depth_m,
    anchor_m: float,
    *,
    gain: float,
    z_focus: float,
    z_far: float,
    eps_m: float = 1e-3,
) -> np.ndarray:
    """Equivalent object distance in ``[z_focus, z_far]`` (float32, same shape).

    ``dp = |1/z - 1/anchor|``; ``e = min(gain * dp, 1/z_focus - 1/z_far)``;
    ``z_t = 1 / max(1/z_focus - e, 1/z_far)``.  Non-finite input pixels map to
    ``z_focus``.  The input is never mutated.
    """
    dp = _diopter_offset(depth_m, anchor_m, eps_m)
    cap = 1.0 / z_focus - 1.0 / z_far
    e = np.minimum(gain * np.nan_to_num(dp, nan=0.0, posinf=np.inf), cap)
    z_t = 1.0 / np.maximum(1.0 / z_focus - e, 1.0 / z_far)
    return z_t.astype(np.float32)


def auto_gain(
    depth_m,
    anchor_m: float,
    *,
    z_focus: float,
    od,
    lo: float = 1.0,
    hi: float = 2.0,
    eps_m: float = 1e-3,
) -> float:
    """Gain putting the 98th-percentile diopter offset at tensor node ``D-3``.

    Flat scene (p98 ~ 0) or no finite pixels returns ``lo``.
    """
    od = np.asarray(od, dtype=np.float64)
    dp_target = 1.0 / z_focus - 1.0 / float(od[-3])
    dp = _diopter_offset(depth_m, anchor_m, eps_m)
    dp = dp[np.isfinite(dp)]
    if dp.size == 0:
        return float(lo)
    p98 = float(np.percentile(dp, 98))
    if p98 < 1e-6:
        return float(lo)
    return float(np.clip(dp_target / p98, lo, hi))
