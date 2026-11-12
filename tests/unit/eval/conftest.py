"""Shared synthetic builders for eval unit tests.

No dependency on Coder A's fixture generators — tests here construct
minimal annotations + arrays inline so they are order-independent.
"""
from __future__ import annotations

import numpy as np
import pytest


def rasterize_disc(
    h: int, w: int, cx: float, cy: float, radius: float
) -> np.ndarray:
    """Binary 0/1 float32 disc."""
    ys, xs = np.mgrid[0:h, 0:w]
    mask = ((xs - cx) ** 2 + (ys - cy) ** 2) <= radius ** 2
    return mask.astype(np.float32)


def rasterize_ellipse(
    h: int, w: int, cx: float, cy: float, a: float, b: float, theta_rad: float = 0.0
) -> np.ndarray:
    """Binary 0/1 float32 ellipse with semi-axes (a, b) and rotation theta."""
    ys, xs = np.mgrid[0:h, 0:w]
    dx = xs - cx
    dy = ys - cy
    c = np.cos(theta_rad)
    s = np.sin(theta_rad)
    xr = c * dx + s * dy
    yr = -s * dx + c * dy
    mask = (xr / a) ** 2 + (yr / b) ** 2 <= 1.0
    return mask.astype(np.float32)


def mono_rgb(mono: np.ndarray) -> np.ndarray:
    """Stack a 2D array into (H, W, 3) with R=G=B."""
    return np.stack([mono, mono, mono], axis=-1).astype(np.float32)


def bubble_grid_annotation_stub(discs: list[dict]) -> dict:
    """Minimal bubble-grid-shaped annotation for score() routing."""
    return {
        "scene": "bubble_grid",
        "frame_hw": [1024, 1024],
        "optical_center_xy": [512, 512],
        "focus_distance_m": 0.70,
        "discs": discs,
    }


def standard_grid_discs() -> list[dict]:
    """The 12-disc table from LOCKED SPEC §2.1."""
    return [
        {"idx": 0, "center_xy_px": [512, 512], "r_frac": 0.00, "depth_m": 0.70, "source_diameter_px": 16, "wrap_applied": False},
        {"idx": 1, "center_xy_px": [512, 332], "r_frac": 0.25, "depth_m": 1.91, "source_diameter_px": 16, "wrap_applied": False},
        {"idx": 2, "center_xy_px": [512, 692], "r_frac": 0.25, "depth_m": 5.23, "source_diameter_px": 16, "wrap_applied": False},
        {"idx": 3, "center_xy_px": [800, 512], "r_frac": 0.40, "depth_m": 0.98, "source_diameter_px": 16, "wrap_applied": False},
        {"idx": 4, "center_xy_px": [224, 512], "r_frac": 0.40, "depth_m": 3.74, "source_diameter_px": 16, "wrap_applied": False},
        {"idx": 5, "center_xy_px": [740, 284], "r_frac": 0.45, "depth_m": 1.37, "source_diameter_px": 16, "wrap_applied": False},
        {"idx": 6, "center_xy_px": [284, 740], "r_frac": 0.45, "depth_m": 2.68, "source_diameter_px": 16, "wrap_applied": False},
        {"idx": 7, "center_xy_px": [120, 120], "r_frac": 0.77, "depth_m": 0.70, "source_diameter_px": 16, "wrap_applied": False},
        {"idx": 8, "center_xy_px": [904, 120], "r_frac": 0.77, "depth_m": 7.32, "source_diameter_px": 16, "wrap_applied": False},
        {"idx": 9, "center_xy_px": [904, 904], "r_frac": 0.77, "depth_m": 10.23, "source_diameter_px": 16, "wrap_applied": False},
        {"idx": 10, "center_xy_px": [120, 904], "r_frac": 0.77, "depth_m": 3.74, "source_diameter_px": 16, "wrap_applied": False},
        {"idx": 11, "center_xy_px": [720, 512], "r_frac": 0.29, "depth_m": 0.40, "source_diameter_px": 16, "wrap_applied": True},
    ]


def render_discs(
    discs: list[dict], h: int = 1024, w: int = 1024, radius: float = 8.0
) -> np.ndarray:
    """Rasterize all discs at uniform radius into a mono 2D array."""
    out = np.zeros((h, w), dtype=np.float32)
    for d in discs:
        cx, cy = d["center_xy_px"]
        out += rasterize_disc(h, w, cx, cy, radius)
    return np.clip(out, 0.0, 1.0)


@pytest.fixture
def std_bubble_grid_fixture():
    """Build a canonical bubble_grid input: all 12 discs at uniform radius."""
    discs = standard_grid_discs()
    mono = render_discs(discs)
    rgb = mono_rgb(mono)
    depth = np.full((1024, 1024), 0.70, dtype=np.float32)
    annotation = bubble_grid_annotation_stub(discs)
    return rgb, depth, annotation
