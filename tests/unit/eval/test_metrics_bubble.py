"""Known-answer tests for M2, M3, M4 (LOCKED SPEC §4 items 2-5)."""
from __future__ import annotations

import numpy as np
import pytest

from summicron.eval.metrics import (
    m2_roundness,
    m3_sag_tan,
    m4_diameter_depth_monotonicity,
)
from tests.unit.eval.conftest import (
    bubble_grid_annotation_stub,
    mono_rgb,
    rasterize_disc,
    rasterize_ellipse,
)


def test_m2_perfect_circle():
    """Rasterized perfect circle, diameter=40 px => roundness ~1.0 ± 0.02."""
    mono = rasterize_disc(1024, 1024, 512, 512, radius=20.0)
    rgb = mono_rgb(mono)
    ann = bubble_grid_annotation_stub(
        [
            {
                "idx": 0,
                "center_xy_px": [512, 512],
                "r_frac": 0.0,
                "depth_m": 0.70,
                "wrap_applied": False,
            }
        ]
    )
    val, flag = m2_roundness(rgb, ann)
    assert flag is None
    assert abs(val - 1.0) < 0.03  # allow a little for rasterization


def test_m2_two_to_one_ellipse():
    """Rasterized 2:1 ellipse => roundness ~0.5 ± 0.05."""
    mono = rasterize_ellipse(1024, 1024, 512, 512, a=20.0, b=10.0, theta_rad=0.0)
    rgb = mono_rgb(mono)
    ann = bubble_grid_annotation_stub(
        [
            {
                "idx": 0,
                "center_xy_px": [512, 512],
                "r_frac": 0.0,
                "depth_m": 0.70,
                "wrap_applied": False,
            }
        ]
    )
    val, flag = m2_roundness(rgb, ann)
    assert flag is None
    assert abs(val - 0.5) < 0.05


def test_m3_sag_tan_1p5_at_corner():
    """Synthetic elongated blob at a known corner with sag:tan = 1.5:1.

    P0.2 regression witness. Place at corner NE (904, 120) so
    radial direction from optical center (512, 512) is ~(+1, -1)/sqrt(2).
    Build an ellipse whose major axis aligns with that radial direction
    (sagittal) with semi-major = 1.5 * semi-minor; then |sag_tan| ≈ 1.5.
    """
    # radial direction from (512, 512) to (904, 120)
    dx = 904 - 512
    dy = 120 - 512
    theta = np.arctan2(dy, dx)  # radians
    a = 15.0  # semi-major aligned with radial
    b = 10.0  # sag/tan = 1.5
    mono = rasterize_ellipse(1024, 1024, 904, 120, a=a, b=b, theta_rad=theta)
    rgb = mono_rgb(mono)
    ann = bubble_grid_annotation_stub(
        [
            {
                "idx": 8,
                "center_xy_px": [904, 120],
                "r_frac": 0.77,
                "depth_m": 7.32,
                "wrap_applied": False,
            }
        ]
    )
    results = m3_sag_tan(rgb, ann)
    assert len(results) == 1
    r = results[0]
    assert r["flag"] is None, f"unexpected flag: {r['flag']}"
    assert abs(r["sag_tan"] - 1.5) < 0.15, f"sag_tan={r['sag_tan']}"
    assert abs(r["theta_deg"]) < 5.0, f"theta_deg={r['theta_deg']}"


def test_m3_wrap_applied_excluded():
    """wrap_applied=True discs are filtered out."""
    mono = rasterize_disc(1024, 1024, 720, 512, radius=10.0)
    rgb = mono_rgb(mono)
    ann = bubble_grid_annotation_stub(
        [
            {
                "idx": 11,
                "center_xy_px": [720, 512],
                "r_frac": 0.29,
                "depth_m": 0.40,
                "wrap_applied": True,
            }
        ]
    )
    results = m3_sag_tan(rgb, ann)
    assert results == []


def test_m4_monotonic_synthetic_diameters():
    """Synthetic diameters [5,6,7,8,9] px at center-column-like radii => ρ=1.0.

    We rasterize 5 concentric discs with strictly-increasing radii into 5
    fake center-column discs at distinct (x, y) and distinct depths.
    """
    h = w = 1024
    mono = np.zeros((h, w), dtype=np.float32)
    # place 5 discs along a vertical line with distinct radii
    centers = [(512, 100), (512, 300), (512, 500), (512, 700), (512, 900)]
    radii = [5, 6, 7, 8, 9]
    depths = [0.70, 0.98, 1.37, 1.91, 2.68]  # monotonic |d - 0.70|
    for (cx, cy), r in zip(centers, radii):
        mono += rasterize_disc(h, w, cx, cy, radius=float(r))
    mono = np.clip(mono, 0.0, 1.0)
    rgb = mono_rgb(mono)
    discs = []
    for i, ((cx, cy), r, d) in enumerate(zip(centers, radii, depths)):
        discs.append(
            {
                "idx": i,
                "center_xy_px": [cx, cy],
                "r_frac": 0.25,  # all center-column
                "depth_m": d,
                "source_diameter_px": 2 * r,
                "wrap_applied": False,
            }
        )
    ann = bubble_grid_annotation_stub(discs)
    rhos, flags = m4_diameter_depth_monotonicity(rgb, ann)
    assert flags["center"] is None, f"center flag: {flags['center']}"
    assert abs(rhos["center"] - 1.0) < 1e-6, f"ρ_center={rhos['center']}"
