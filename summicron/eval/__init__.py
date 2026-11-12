"""Eval harness: PSF-QC synthetic-fixture scoring (Section 9 / P0.3).

Public API — ``score()`` dispatcher, three fixture generators, and the
seven metric functions re-exported from ``summicron.eval.metrics``.
"""
from __future__ import annotations

from .fixtures import (
    generate_bubble_grid,
    generate_bubble_random,
    generate_usaf_depth,
)
from .metrics import (
    assert_monochrome,
    energy_ratio,
    fit_ellipse_guarded,
    gradient_energy,
    m2_roundness,
    m3_sag_tan,
    m4_diameter_depth_monotonicity,
    m5_in_focus_sharpness,
    m6_falloff_curve,
    m7_wrap_probe,
    project_covariance_radial,
)
from .score import score

__all__ = [
    "score",
    "generate_bubble_grid",
    "generate_bubble_random",
    "generate_usaf_depth",
    "energy_ratio",
    "m2_roundness",
    "m3_sag_tan",
    "m4_diameter_depth_monotonicity",
    "m5_in_focus_sharpness",
    "m6_falloff_curve",
    "m7_wrap_probe",
    "fit_ellipse_guarded",
    "gradient_energy",
    "project_covariance_radial",
    "assert_monochrome",
]
