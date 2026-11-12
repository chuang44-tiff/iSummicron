"""Shared synthesis helpers for eval fixtures.

Annotation contract (flat dict, <=5 top-level keys per fixture family,
per LOCKED SPEC Section 9 / P0.3 §7):

- ``scene``: str, one of {"bubble_grid", "bubble_random", "usaf_depth"}.
- ``frame_hw``: [int, int] — (H, W), always [1024, 1024] in v0.
- ``optical_center_xy``: [int, int] — always [512, 512] in v0.
- ``focus_distance_m``: float — load-bearing for wrap-probe diagnostic;
  always 0.70 in v0 (matches the shipped tensors).
- ``discs`` (bubble_*) OR ``tiles`` (usaf_depth): per-element ground
  truth list. ``seed`` is appended for ``bubble_random``.

Disc entry schema: ``{idx:int, center_xy_px:[int,int], r_frac:float,
depth_m:float, source_diameter_px:int, wrap_applied:bool}``.

Tile entry schema: ``{idx:int, bbox_xyxy:[int,int,int,int],
depth_m:float, bar_widths_px:[int,...], group_bboxes_xyxy:[[int,...]]}``.

All fixture arrays:
- ``rgb``: float32, shape (H, W, 3), values in [0.0, 1.0], R=G=B.
- ``depth_m``: float32, shape (H, W), values >= 0 in meters.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

# Tensor-native in-scope depths (from
# summicron/psf_model/ray_kernel.py → OBJECT_DISTANCES_M, restricted
# to the 9 values the LOCKED SPEC §1 uses for PSF-QC discs).
TENSOR_DEPTHS_M: tuple[float, ...] = (
    0.70, 0.98, 1.37, 1.91, 2.68, 3.74, 5.23, 7.32, 10.23,
)

# The single wrap-probe depth permitted outside the tensor-native set
# when the corresponding disc carries ``wrap_applied=true``.
WRAP_PROBE_DEPTH_M: float = 0.40

# v0 focus distance (the shipped tensors match this). Load-bearing: consumers
# use this to compute the wrapped effective depth for disc #11.
FOCUS_DISTANCE_M: float = 0.70

FRAME_H: int = 1024
FRAME_W: int = 1024
OPTICAL_CENTER_XY: tuple[int, int] = (512, 512)


def draw_disc(img: np.ndarray, cx: int, cy: int, diameter: int) -> None:
    """Rasterize a binary disc (value=1.0 on all 3 channels, no AA).

    Mutates ``img`` in place. No-op if the disc lands fully off-frame;
    partial clipping at edges is allowed but LOCKED SPEC §2.1 discs are
    all well inside the frame.
    """
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"expected (H,W,3) image, got shape {img.shape}")
    if diameter < 1:
        raise ValueError(f"diameter must be >= 1, got {diameter}")
    h, w = img.shape[:2]
    yy, xx = np.ogrid[:h, :w]
    r2 = (diameter / 2.0) ** 2
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r2
    img[mask] = 1.0  # writes all 3 channels (R=G=B=1)


def draw_usaf_tile(
    img: np.ndarray,
    x0: int,
    y0: int,
    tile_size: int = 320,
    bar_widths: tuple[int, ...] = (16, 8, 4, 2),
) -> list[list[int]]:
    """Rasterize a minimal USAF target inside a ``tile_size``x``tile_size``
    tile anchored at (x0, y0) (top-left).

    For each bar-width ``b``, two 5b-tall group rows (horizontal then
    vertical, 3 bars each with ``b`` spacing) are stacked vertically.
    Total vertical stack = 2 * sum(5*b) px. Default widths {16,8,4,2}
    yield 300 px stacks, fitting inside a 320 px tile with a 10 px
    top-margin. Horizontal-bar x-extent is ``tile_size - 8`` (centered,
    4 px side margin). Bars are binary 1.0 on background 0.0, R=G=B.
    Returns the list of group bboxes (``[x0,y0,x1,y1]``).
    """
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"expected (H,W,3) image, got shape {img.shape}")
    if len(bar_widths) < 1:
        raise ValueError("bar_widths must have at least one entry")
    # Each group row is exactly 5b tall (3 bars + 2 gaps of thickness b).
    group_heights = [5 * int(b) for b in bar_widths]
    total_rows = 2 * sum(group_heights)  # horizontal + vertical per width
    top_margin = 10
    if total_rows + top_margin > tile_size:
        raise ValueError(
            f"tile layout infeasible: {total_rows + top_margin} px needed "
            f"for bar-group stack + margin, tile is {tile_size} px"
        )
    group_bboxes: list[list[int]] = []
    y_cursor = y0 + top_margin
    strip_x0 = x0 + 4
    strip_x1 = x0 + tile_size - 4
    for b in bar_widths:
        gh = 5 * b  # per-group height for this bar-width
        # --- Horizontal bars: 3 rows of b-thick bars, separated by b. ---
        for i in range(3):
            y_bar = y_cursor + i * 2 * b
            img[y_bar:y_bar + b, strip_x0:strip_x1, :] = 1.0
        group_bboxes.append([strip_x0, y_cursor, strip_x1, y_cursor + gh])
        y_cursor += gh

        # --- Vertical bars: 3 columns of b-thick bars, separated by b. ---
        # Center the 5b-wide bar block horizontally inside the tile.
        v_x0 = x0 + (tile_size - gh) // 2
        for i in range(3):
            x_bar = v_x0 + i * 2 * b
            img[y_cursor:y_cursor + gh, x_bar:x_bar + b, :] = 1.0
        group_bboxes.append([v_x0, y_cursor, v_x0 + gh, y_cursor + gh])
        y_cursor += gh
    return group_bboxes


def assert_monochrome(rgb: np.ndarray) -> None:
    """Hard-fail if R, G, B channels are not bit-exact identical."""
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise AssertionError(f"not (H,W,3): shape={rgb.shape}")
    if not (np.array_equal(rgb[..., 0], rgb[..., 1])
            and np.array_equal(rgb[..., 1], rgb[..., 2])):
        raise AssertionError(
            "monochrome contract violated: R,G,B channels differ"
        )


def assert_depth_range(depth_m: np.ndarray, discs: list[dict] | None) -> None:
    """Validate every unique non-background depth in ``depth_m``.

    Background 0.70m is always allowed. Every other unique depth must be
    in ``TENSOR_DEPTHS_M`` OR equal to ``WRAP_PROBE_DEPTH_M`` iff at
    least one disc in ``discs`` has ``wrap_applied=true`` at that depth.
    """
    allowed = set(float(d) for d in TENSOR_DEPTHS_M)
    wrap_allowed = False
    if discs is not None:
        for d in discs:
            if bool(d.get("wrap_applied", False)) and (
                float(d["depth_m"]) == WRAP_PROBE_DEPTH_M
            ):
                wrap_allowed = True
                break
    if wrap_allowed:
        allowed.add(WRAP_PROBE_DEPTH_M)
    uniq = np.unique(depth_m).astype(float).tolist()
    for v in uniq:
        # tolerate tiny float noise from float32 round-trip
        if not any(abs(v - a) < 1e-4 for a in allowed):
            raise ValueError(
                f"depth value {v} not in allowed set {sorted(allowed)}; "
                "PSF-QC discs must use tensor-native depths, only the "
                "wrap-probe may use 0.40m (and only with wrap_applied=true)"
            )


def save_fixture(
    out_path: str | Path,
    rgb: np.ndarray,
    depth_m: np.ndarray,
    annotation: dict[str, Any],
) -> Path:
    """Write the ``(rgb, depth_m, annotation)`` triple to disk.

    Produces ``<out>.npz`` with keys ``rgb`` and ``depth_m`` plus a
    sidecar ``<out>.json`` with the annotation. Returns the .npz path.
    """
    out_path = Path(out_path)
    if out_path.suffix != ".npz":
        raise ValueError(f"out_path must end in .npz, got {out_path}")
    # Contract assertions before I/O: fail loud at the boundary.
    if rgb.dtype != np.float32:
        raise ValueError(f"rgb dtype must be float32, got {rgb.dtype}")
    if depth_m.dtype != np.float32:
        raise ValueError(f"depth_m dtype must be float32, got {depth_m.dtype}")
    if rgb.shape[:2] != depth_m.shape:
        raise ValueError(
            f"rgb spatial shape {rgb.shape[:2]} != depth_m shape {depth_m.shape}"
        )
    if float(rgb.min()) < 0.0 or float(rgb.max()) > 1.0:
        raise ValueError(
            f"rgb out of [0,1]: min={rgb.min()}, max={rgb.max()}"
        )
    assert_monochrome(rgb)
    discs = annotation.get("discs")
    assert_depth_range(depth_m, discs)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(out_path), rgb=rgb, depth_m=depth_m)
    sidecar = out_path.with_suffix(".json")
    with open(sidecar, "w") as f:
        json.dump(annotation, f, indent=2, sort_keys=True)
    return out_path


def sha256_file(path: str | Path) -> str:
    """SHA256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
