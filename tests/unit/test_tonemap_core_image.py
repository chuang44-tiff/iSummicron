"""Unit tests for macOS Core Image tonemap (`tonemap_core_image`).

Test #1 runs everywhere (monkey-patches platform). Tests #2-#4 require the
macOS 15+ Core Image stack with `CIToneMapHeadroom`.
"""
from __future__ import annotations

import sys

import numpy as np
import pytest

from summicron.io.hdr_tonemap_macos import tonemap_core_image

MACOS_ONLY = pytest.mark.skipif(
    sys.platform != "darwin", reason="Core Image is macOS-only"
)


def test_non_macos_raises(monkeypatch):
    """Platform gate: non-darwin must raise NotImplementedError."""
    monkeypatch.setattr(sys, "platform", "linux")
    rgb = np.zeros((4, 4, 3), dtype=np.float32)
    with pytest.raises(NotImplementedError, match="macOS 15"):
        tonemap_core_image(rgb)


@MACOS_ONLY
def test_basic_shape_and_dtype():
    """(8, 8, 3) random float32 in [0, 2] → (8, 8, 3) uint8."""
    rng = np.random.default_rng(0)
    rgb = rng.uniform(0.0, 2.0, size=(8, 8, 3)).astype(np.float32)
    out = tonemap_core_image(rgb)
    assert out.shape == (8, 8, 3)
    assert out.dtype == np.uint8


@MACOS_ONLY
def test_saturates_above_one_no_nan():
    """Input with max=6.5 → output has no NaN and stays in [0, 255]."""
    rng = np.random.default_rng(1)
    rgb = rng.uniform(0.0, 6.5, size=(16, 16, 3)).astype(np.float32)
    # Ensure we actually hit 6.5 as max.
    rgb[0, 0, 0] = 6.5
    out = tonemap_core_image(rgb)
    assert not np.isnan(out).any()
    assert out.max() <= 255
    assert out.min() >= 0


@MACOS_ONLY
def test_ldr_roundtrip_near_noop():
    """Uniform 0.5 linear input → uniform ~188 ± 2 uint8 sRGB.

    `linear_to_srgb(0.5) * 255 ≈ 188`. Below the headroom threshold the filter
    should not apply meaningful compression.
    """
    rgb = np.full((16, 16, 3), 0.5, dtype=np.float32)
    out = tonemap_core_image(rgb)
    assert abs(float(out.mean()) - 188.0) <= 2.0, (
        f"LDR 0.5 uniform tonemapped to mean {out.mean()}, expected ~188 ± 2"
    )
