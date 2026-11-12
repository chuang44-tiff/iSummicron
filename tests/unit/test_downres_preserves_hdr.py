"""Regression tests for downres_array HDR headroom preservation.

P1.8.1: the prior LDR-only implementation clipped HDR to uint8 before Lanczos
resize, destroying values > 1.0. These tests lock in the HDR-aware branch and
a smoke test that the LDR path is unaffected.
"""
import numpy as np

from summicron.render.convolve import downres_array


def test_downres_array_preserves_hdr_headroom():
    """downres_array must not clip HDR values > 1.0 during resize."""
    rng = np.random.default_rng(0)
    hdr = rng.uniform(0.0, 1.0, size=(512, 768, 3)).astype(np.float32)
    hdr[100:120, 100:120, :] = 6.5  # bright HDR patch
    out = downres_array(hdr, target_long_edge=384)
    assert out.shape[:2] == (256, 384), out.shape
    assert out.max() > 4.0, f"HDR headroom lost: max={out.max():.4f}"
    assert out.dtype == np.float32


def test_downres_array_ldr_path_unchanged():
    """LDR path must stay unchanged (smoke test bit-identical behavior)."""
    rng = np.random.default_rng(0)
    ldr = rng.uniform(0.0, 1.0, size=(512, 768, 3)).astype(np.float32)
    out = downres_array(ldr, target_long_edge=384)
    assert out.shape[:2] == (256, 384)
    assert out.dtype == np.float32
    assert out.min() >= 0.0 and out.max() <= 1.0
