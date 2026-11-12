"""EXIF Orientation 1-8 transform helper (display click -> storage normalized).

Utility for future callers (non-HEIC JPEG/PNG with missing EXIF auto-apply).
NOT on the picker's hot path: the picker uses the identity contract because
``summicron.render.convolve.load_image`` returns the array as pillow_heif
decoded it (auto-EXIF for HEIC) and the picker mirrors that bit-for-bit.

Orientations: 1 identity, 2 mirror H, 3 rot 180, 4 mirror V, 5 transpose,
6 rot 90 CW, 7 transverse, 8 rot 270 CW (= 90 CCW).
"""
from __future__ import annotations


def display_to_storage_norm(x_display: float, y_display: float,
                            w_display: int, h_display: int,
                            orientation: int) -> tuple[float, float]:
    """Map a display click (col, row) -> storage (x_norm, y_norm) in [0, 1]."""
    if orientation not in range(1, 9):
        raise ValueError(f"orientation must be in 1..8, got {orientation}")
    u = x_display / w_display
    v = y_display / h_display
    if orientation == 1:
        return u, v
    if orientation == 2:
        return 1.0 - u, v
    if orientation == 3:
        return 1.0 - u, 1.0 - v
    if orientation == 4:
        return u, 1.0 - v
    if orientation == 5:
        return v, u
    if orientation == 6:
        return 1.0 - v, u
    if orientation == 7:
        return 1.0 - v, 1.0 - u
    # orientation == 8
    return v, 1.0 - u
