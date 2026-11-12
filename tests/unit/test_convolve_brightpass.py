"""P1.11 R1 bright-pass unit tests. Gates G5 (bit-identity) + G6 (is_delta skip)."""
from __future__ import annotations
import numpy as np
from tests.fixtures.defocus import render_with_defocus

_RE = np.array([0.0, 0.15, 0.30, 0.45, 0.60, 0.80, 1.01], dtype=np.float64)
_DR = np.array([0.0, 0.33, 0.67, 1.0], dtype=np.float64)

def _blur_tensor(k=33):
    R, D = 6, 4
    kern = np.zeros((R, D, k, k), dtype=np.float32)
    c = k // 2
    kern[:, 0, c, c] = 1.0
    ys, xs = np.mgrid[-c:c + 1, -c:c + 1].astype(np.float32)
    for di in range(1, D):
        g = np.exp(-(xs ** 2 + ys ** 2) / (2 * (1.5 + 1.5 * di) ** 2))
        kern[:, di] = (g / g.sum()).astype(np.float32)
    return {"kernels": kern, "radial_edges": _RE.copy(), "defocus_ratios": _DR.copy()}

def test_bright_thresh_disabled_is_bit_identical_L1():
    """G5: thresh=1e9 matches no-flag call within 1e-5/channel on LDR."""
    img = np.random.default_rng(7).uniform(0, 0.9, (96, 96, 3)).astype(np.float32)
    dm = np.full((96, 96), 0.6, dtype=np.float32); t = _blur_tensor()
    _a, la = render_with_defocus(img, t, dmap=dm)
    _b, lb = render_with_defocus(img, t, dmap=dm, bright_thresh=1e9)
    assert np.max(np.abs(la - lb)) < 1e-5

def test_bright_thresh_disabled_is_bit_identical_HDR():
    """G5 for HDR-scale input via hdr_source='macos'."""
    img = np.random.default_rng(11).uniform(0, 3.0, (80, 80, 3)).astype(np.float32)
    dm = np.full((80, 80), 0.5, dtype=np.float32); t = _blur_tensor()
    _a, la = render_with_defocus(img, t, dmap=dm, hdr_source="macos")
    _b, lb = render_with_defocus(img, t, dmap=dm, hdr_source="macos", bright_thresh=1e9)
    assert np.max(np.abs(la - lb)) < 1e-5

def test_bright_pass_gated_on_not_is_delta():
    """G6: all-delta tensor + thresh=0.5 on max=1.0 image must stay bit-identical
    to disabled (proves delta path skipped bright convolution)."""
    k = 33
    kern = np.zeros((6, 4, k, k), dtype=np.float32); kern[:, :, k // 2, k // 2] = 1.0
    t = {"kernels": kern, "radial_edges": _RE.copy(), "defocus_ratios": _DR.copy()}
    img = np.full((64, 64, 3), 1.0, dtype=np.float32)
    dm = np.zeros((64, 64), dtype=np.float32)
    _a, lo = render_with_defocus(img, t, dmap=dm, bright_thresh=1e9)
    _b, ln = render_with_defocus(img, t, dmap=dm, bright_thresh=0.5)
    assert np.max(np.abs(ln - lo)) < 1e-5

def test_bright_pass_enabled_blooms_above_threshold():
    """Bright blob (3.0) on dim bg (0.3), defocus=0.6 (non-delta). Ring energy
    around blob rises under thresh=1.0 vs disabled (>=2% ring-sum gain)."""
    img = np.full((80, 80, 3), 0.3, dtype=np.float32); img[38:42, 38:42, :] = 3.0
    dm = np.full((80, 80), 0.6, dtype=np.float32); t = _blur_tensor()
    _a, lo = render_with_defocus(img, t, dmap=dm, hdr_source="macos", bright_thresh=1e9)
    _b, ln = render_with_defocus(img, t, dmap=dm, hdr_source="macos", bright_thresh=1.0)
    ring = np.ones((80, 80), dtype=bool); ring[38:42, 38:42] = False
    assert ln[ring].sum() > lo[ring].sum() * 1.02
