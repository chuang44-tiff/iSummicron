"""P1.11 R1-PATCH soft-knee bright-pass adversarial tests (T1-T6).

Gates the TL-revised spec: K=0.4 default + CLI flag. See
doc/plan/phase1/section11/locked-spec-R1-PATCH-softknee.md §6/§11.
"""
from __future__ import annotations

import numpy as np
import pytest

from summicron.render.convolve import render
from tests.fixtures.defocus import render_with_defocus

_RE = np.array([0.0, 0.15, 0.30, 0.45, 0.60, 0.80, 1.01], dtype=np.float64)
_DR = np.array([0.0, 0.33, 0.67, 1.0], dtype=np.float64)


def _blur_tensor(k: int = 33) -> dict:
    """Tensor with delta at d=0 and expanding gaussians at d>0."""
    kern = np.zeros((6, 4, k, k), dtype=np.float32)
    c = k // 2
    kern[:, 0, c, c] = 1.0
    ys, xs = np.mgrid[-c:c + 1, -c:c + 1].astype(np.float32)
    for di in range(1, 4):
        g = np.exp(-(xs ** 2 + ys ** 2) / (2 * (1.5 + 1.5 * di) ** 2))
        kern[:, di] = (g / g.sum()).astype(np.float32)
    return {
        "kernels": kern,
        "radial_edges": _RE.copy(),
        "defocus_ratios": _DR.copy(),
    }


# P1.10.2 Cut 3: test_soft_knee_uniform_canopy_no_patchy_amplification DELETED
# (compared bright_knee=0.4 vs bright_knee=0.0; both kwargs removed).


def test_soft_knee_midtone_preserved_far_from_thresh():
    """T2/G2: flat lum=0.5 on thresh=1.0 (far below T-K=0.6). bright_weight=0
    exactly, center crop within 5% of L1-expected 0.5."""
    img = np.full((64, 64, 3), 0.5, dtype=np.float32)
    dm = np.full((64, 64), 0.6, dtype=np.float32)
    t = _blur_tensor()
    _a, out = render_with_defocus(img, t, dmap=dm, hdr_source="macos", bright_thresh=1.0)
    center = out[20:44, 20:44, :]
    assert np.all(np.abs(center - 0.5) / 0.5 < 0.05), (
        f"midtone contamination: min={center.min():.4f}, max={center.max():.4f}"
    )


def test_soft_knee_preserves_string_light_bloom():
    """T3/G3: bright blob lum=3.0 (>> T+K=1.4) still hits bright_weight=1
    exactly and blooms like R1. Ring-sum ≥ 2% gain vs disabled."""
    img = np.full((80, 80, 3), 0.3, dtype=np.float32)
    img[38:42, 38:42, :] = 3.0
    dm = np.full((80, 80), 0.6, dtype=np.float32)
    t = _blur_tensor()
    _a, lo = render_with_defocus(img, t, dmap=dm, hdr_source="macos", bright_thresh=1e9)
    _b, ln = render_with_defocus(img, t, dmap=dm, hdr_source="macos", bright_thresh=1.0)
    ring = np.ones((80, 80), dtype=bool); ring[38:42, 38:42] = False
    assert ln[ring].sum() > lo[ring].sum() * 1.02


def test_soft_knee_backward_compat_thresh_1e9():
    """T4/G4: at thresh=1e9, K=0.4 still gives bright_weight=0 everywhere
    (realistic lum ≪ 1e9-0.4). Bit-identical to default-thresh no-op call
    within 1e-5 per channel."""
    img = np.random.default_rng(7).uniform(0, 3.0, (96, 96, 3)).astype(np.float32)
    dm = np.full((96, 96), 0.6, dtype=np.float32)
    t = _blur_tensor()
    _a, la = render_with_defocus(img, t, dmap=dm, hdr_source="macos")
    _b, lb = render_with_defocus(img, t, dmap=dm, hdr_source="macos", bright_thresh=1e9)
    assert np.max(np.abs(la - lb)) < 1e-5


# P1.10.2 Cut 3: test_bright_pass_knee_default_is_0_20 DELETED
# (kwarg removed; default is now the module constant _BRIGHT_KNEE = 0.20).
# test_bright_pass_knee_zero_collapses_to_hard_boolean DELETED
# (K=0 / K=0.01 kwargs removed).
