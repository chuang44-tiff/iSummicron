"""Unit tests for macOS HDR HEIC loader.

Test #1 runs everywhere (monkey-patches platform). Tests #2 and #3 require a
real macOS Core Image stack + the canonical vibe-bench HEIC.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from summicron.io.hdr_load_macos import load_hdr_heic

REPO_ROOT = Path(__file__).resolve().parents[2]
HDR_FIXTURE = REPO_ROOT / "iphoneImage" / "IMG_6651.HEIC"
HDR_FIXTURE_6915 = REPO_ROOT / "iphoneImage" / "IMG_6915.HEIC"


def test_load_hdr_heic_non_macos_raises(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    with pytest.raises(NotImplementedError):
        load_hdr_heic("/tmp/does-not-matter.heic")


@pytest.mark.skipif(sys.platform != "darwin", reason="Core Image is macOS-only")
@pytest.mark.skipif(not HDR_FIXTURE.exists(), reason=f"missing {HDR_FIXTURE}")
def test_load_hdr_heic_returns_float32_3ch():
    result = load_hdr_heic(HDR_FIXTURE)
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float32
    assert result.ndim == 3
    assert result.shape[-1] == 3


@pytest.mark.skipif(sys.platform != "darwin", reason="Core Image is macOS-only")
@pytest.mark.parametrize(
    "fixture",
    [
        pytest.param(HDR_FIXTURE, id="IMG_6651"),
        pytest.param(HDR_FIXTURE_6915, id="IMG_6915"),
    ],
)
def test_load_hdr_heic_matches_load_image_shape(fixture):
    """R0 I7: HDR loader shape must match `load_image` (PIL+pillow_heif) shape
    for both canonical-bench HEICs. Shape-only comparison — HDR is extended-
    linear float32; LDR is sRGB-gamma [0,1]. Values do NOT match even when
    orientation is correct; only the (H, W, 3) tuple is compared.
    """
    if not fixture.exists():
        pytest.skip(f"missing {fixture}")
    from summicron.render.convolve import load_image

    hdr = load_hdr_heic(fixture)
    ldr = load_image(fixture)
    assert hdr.shape == ldr.shape, (
        f"shape mismatch on {fixture.name}: HDR {hdr.shape} vs LDR {ldr.shape}"
    )


@pytest.mark.skipif(sys.platform != "darwin", reason="Core Image is macOS-only")
@pytest.mark.skipif(not HDR_FIXTURE.exists(), reason=f"missing {HDR_FIXTURE}")
def test_load_hdr_heic_max_exceeds_one():
    result = load_hdr_heic(HDR_FIXTURE)
    # Real-data probe: expandToHDR must recover HDR content. If this fails on
    # IMG_6651, the spec premise is invalid (see P1.8 locked spec test #3).
    assert result.max() > 1.001, (
        f"HDR max {result.max():.4f} did not exceed 1.0; "
        "expandToHDR did not recover HDR content on IMG_6651."
    )
    assert np.isfinite(result).all()
