"""Bubble-chart fixture generators.

Two scenes per LOCKED SPEC §2.1 + §2.2:

- ``generate_bubble_grid()`` — 12 deterministic discs (11 PSF-QC + 1
  0.40m wrap probe at index 11).
- ``generate_bubble_random(seed=42, n=12)`` — 1 rejection-sampled scene.

All discs are 16 px binary (no AA). Each disc stamps a 40x40 depth-square
around its center onto the depth map. Background depth = 0.70m.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from summicron.eval.fixtures.common import (
    FOCUS_DISTANCE_M,
    FRAME_H,
    FRAME_W,
    OPTICAL_CENTER_XY,
    TENSOR_DEPTHS_M,
    WRAP_PROBE_DEPTH_M,
    assert_depth_range,
    assert_monochrome,
    draw_disc,
)

SOURCE_DIAMETER_PX: int = 16
DEPTH_SQUARE_PX: int = 40
BUBBLE_GRID_TABLE: tuple[tuple[int, int, int, float, float, bool], ...] = (
    # (idx, x, y, r_frac, depth_m, wrap_applied)  — LOCKED SPEC §2.1
    (0,  512, 512, 0.00,  0.70, False),
    (1,  512, 332, 0.25,  1.91, False),
    (2,  512, 692, 0.25,  5.23, False),
    (3,  800, 512, 0.40,  0.98, False),
    (4,  224, 512, 0.40,  3.74, False),
    (5,  740, 284, 0.45,  1.37, False),
    (6,  284, 740, 0.45,  2.68, False),
    (7,  120, 120, 0.77,  0.70, False),
    (8,  904, 120, 0.77,  7.32, False),
    (9,  904, 904, 0.77, 10.23, False),
    (10, 120, 904, 0.77,  3.74, False),
    (11, 720, 512, 0.29,  0.40, True),
)


def _stamp_depth_square(
    depth_m: np.ndarray, cx: int, cy: int, value: float, size: int = DEPTH_SQUARE_PX
) -> None:
    """Write a size-by-size constant square of ``value`` centered on (cx, cy)."""
    half = size // 2
    y0 = max(0, cy - half)
    y1 = min(depth_m.shape[0], cy + half)
    x0 = max(0, cx - half)
    x1 = min(depth_m.shape[1], cx + half)
    depth_m[y0:y1, x0:x1] = value


def _build_scene(
    discs: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray]:
    """Shared rasterization for grid + random bubble scenes.

    Depth stamping uses disc list order, so a disc listed later
    overwrites an overlapping square (not expected with spec margins).
    """
    rgb = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.float32)
    depth_m = np.full((FRAME_H, FRAME_W), FOCUS_DISTANCE_M, dtype=np.float32)
    for d in discs:
        cx, cy = d["center_xy_px"]
        draw_disc(rgb, cx, cy, d["source_diameter_px"])
        _stamp_depth_square(depth_m, cx, cy, float(d["depth_m"]))
    return rgb, depth_m


def generate_bubble_grid() -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Generate the deterministic 12-disc bubble_grid fixture.

    Returns ``(rgb, depth_m, annotation)``. See LOCKED SPEC §2.1.
    """
    discs: list[dict[str, Any]] = []
    for idx, x, y, r_frac, depth, wrap in BUBBLE_GRID_TABLE:
        discs.append({
            "idx": int(idx),
            "center_xy_px": [int(x), int(y)],
            "r_frac": float(r_frac),
            "depth_m": float(depth),
            "source_diameter_px": int(SOURCE_DIAMETER_PX),
            "wrap_applied": bool(wrap),
        })
    rgb, depth_m = _build_scene(discs)

    annotation: dict[str, Any] = {
        "scene": "bubble_grid",
        "frame_hw": [FRAME_H, FRAME_W],
        "optical_center_xy": list(OPTICAL_CENTER_XY),
        "focus_distance_m": float(FOCUS_DISTANCE_M),
        "discs": discs,
    }

    # Monochrome + depth-range post-conditions (fail loud if we broke it).
    assert_monochrome(rgb)
    assert_depth_range(depth_m, discs)
    return rgb, depth_m, annotation


def generate_bubble_random(
    seed: int = 42, n: int = 12
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Generate one seeded rejection-sampled bubble scene per LOCKED SPEC §2.2.

    - ``n`` discs, ``np.random.default_rng(seed)``.
    - (x, y) uniform in [64, 960]; depth sampled from ``TENSOR_DEPTHS_M``.
    - Reject if any pair of centers is within 96 px (>= 6x source_diameter).
    - Retry cap: 500 rejections per disc → raise.
    - No wrap probe (deterministic-only per LOCKED SPEC).
    """
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, bool):
        raise TypeError(f"seed must be a non-negative int, got {type(seed).__name__}")
    if int(seed) < 0:
        raise TypeError(f"seed must be non-negative, got {seed}")
    if not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError(f"n must be a positive int, got {n!r}")

    rng = np.random.default_rng(int(seed))
    min_pair_dist_px = 96
    coord_lo, coord_hi = 64, 960
    discs: list[dict[str, Any]] = []
    max_rejections = 500
    rejections = 0

    # Optical-center distance for r_frac.
    ocx, ocy = OPTICAL_CENTER_XY
    r_max = float(np.hypot(FRAME_W - ocx, FRAME_H - ocy))

    while len(discs) < n:
        if rejections > max_rejections:
            raise RuntimeError(
                f"bubble_random: exceeded {max_rejections} rejections after "
                f"placing {len(discs)}/{n} discs; tighten layout or raise cap."
            )
        x = int(rng.integers(coord_lo, coord_hi + 1))
        y = int(rng.integers(coord_lo, coord_hi + 1))
        depth = float(TENSOR_DEPTHS_M[int(rng.integers(0, len(TENSOR_DEPTHS_M)))])
        ok = True
        for d in discs:
            dx = d["center_xy_px"][0] - x
            dy = d["center_xy_px"][1] - y
            if (dx * dx + dy * dy) < (min_pair_dist_px * min_pair_dist_px):
                ok = False
                break
        if not ok:
            rejections += 1
            continue
        r_frac = float(np.hypot(x - ocx, y - ocy) / r_max)
        discs.append({
            "idx": len(discs),
            "center_xy_px": [x, y],
            "r_frac": r_frac,
            "depth_m": depth,
            "source_diameter_px": int(SOURCE_DIAMETER_PX),
            "wrap_applied": False,
        })

    rgb, depth_m = _build_scene(discs)
    annotation: dict[str, Any] = {
        "scene": "bubble_random",
        "frame_hw": [FRAME_H, FRAME_W],
        "optical_center_xy": list(OPTICAL_CENTER_XY),
        "focus_distance_m": float(FOCUS_DISTANCE_M),
        "seed": int(seed),
        "discs": discs,
    }
    assert_monochrome(rgb)
    assert_depth_range(depth_m, discs)
    # Defensive: the wrap-probe depth (0.40m) must not appear in random.
    if WRAP_PROBE_DEPTH_M in {float(d["depth_m"]) for d in discs}:
        raise RuntimeError(
            "bubble_random emitted a 0.40m disc — wrap probe is grid-only."
        )
    return rgb, depth_m, annotation
