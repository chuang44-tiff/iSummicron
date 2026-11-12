"""M1 — Energy conservation.

ratio = rendered.sum() / input.sum() on the R-channel (monochrome fixtures
enforce R=G=B). LOCKED SPEC §3.M1.
"""
from __future__ import annotations

import numpy as np

from .common import assert_monochrome


# Absolute floor below which the input is considered degenerately dark
# (1024x1024 fixtures contain >>1.0 summed intensity).
INPUT_SUM_FLOOR = 1.0


def energy_ratio(
    input_rgb: np.ndarray, rendered_rgb: np.ndarray
) -> tuple[float, str | None]:
    """Return (ratio, flag). ratio is NaN on guard trip."""
    assert_monochrome(input_rgb, "input_rgb")
    assert_monochrome(rendered_rgb, "rendered_rgb")
    if input_rgb.shape != rendered_rgb.shape:
        raise ValueError(
            f"shape mismatch: input {input_rgb.shape} vs rendered {rendered_rgb.shape}"
        )

    in_sum = float(input_rgb[..., 0].sum())
    if in_sum < INPUT_SUM_FLOOR:
        return float("nan"), "input_too_dark"

    ratio = float(rendered_rgb[..., 0].sum()) / in_sum
    return ratio, None
