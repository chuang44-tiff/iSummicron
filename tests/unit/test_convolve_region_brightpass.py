"""P1.10 region-aware bright-pass — unit tests T1-T7.

Spec: ``doc/plan/phase1/section16/locked-spec-P1_10.md`` § 7.

T1: kill-switch bit-identity (region_size=0 → bare smoothstep).
T2: flat-bright suppression (top-hat suppresses uniform bright).
T3: point-light preservation (top-hat preserves true highlights).
T4: floor=1.0 collapses to lum-only path (no region gating).
T5: floor=0.0 collapses to pure top-hat-norm.
T6: render() validators reject out-of-range kwargs.
T7: all-zero input is finite (no NaN, no division-by-zero blow-up).
"""
from __future__ import annotations

import numpy as np
import pytest

from summicron.render.convolve import (
    _compute_region_aware_bright_weight,
)
from tests.fixtures.defocus import render_with_defocus



_RE = np.array([0.0, 0.15, 0.30, 0.45, 0.60, 0.80, 1.01], dtype=np.float64)
_DR = np.array([0.0, 0.33, 0.67, 1.0], dtype=np.float64)


def _blur_tensor(k: int = 33) -> dict:
    kern = np.zeros((6, 4, k, k), dtype=np.float32)
    c = k // 2
    kern[:, 0, c, c] = 1.0
    ys, xs = np.mgrid[-c:c + 1, -c:c + 1].astype(np.float32)
    for di in range(1, 4):
        g = np.exp(-(xs ** 2 + ys ** 2) / (2 * (1.5 + 1.5 * di) ** 2))
        kern[:, di] = (g / g.sum()).astype(np.float32)
    return {"kernels": kern, "radial_edges": _RE.copy(), "defocus_ratios": _DR.copy()}


def _inline_smoothstep(lum: np.ndarray, T: float, K: float) -> np.ndarray:
    """Reference: the inline smoothstep formula that lived at L880 pre-P1.10."""
    bw = np.clip((lum - (T - K)) / (2.0 * K + 1e-12), 0.0, 1.0)
    bw = bw * bw * (3.0 - 2.0 * bw)
    return bw.astype(np.float32)


# T1 — kill-switch bit-identity (covers default + negative region_size)
@pytest.mark.parametrize("rs", [0, -1, -5])
def test_T1_killswitch_bit_identical_to_inline_smoothstep(rs):
    """region_size <= 0 returns bit-identical output to the inline smoothstep."""
    rng = np.random.default_rng(7)
    lum = rng.uniform(0.0, 2.0, (128, 128)).astype(np.float32)
    expected = _inline_smoothstep(lum, 1.0, 0.20)
    actual = _compute_region_aware_bright_weight(
        lum, 1.0, region_size=rs, region_floor=0.10, region_norm_pct=99.0,
    )
    assert np.array_equal(expected, actual)


# T2 — flat-bright suppression
def test_T2_flat_bright_image_strongly_suppressed():
    """Flat-bright (lum=2.0, 256×256) at floor=0.0: top-hat = 0 → bw_region ≈ 0."""
    lum = np.full((256, 256), 2.0, dtype=np.float32)
    bw_lum = _compute_region_aware_bright_weight(lum, 1.0, 0, 0.10, 99.0)
    bw_region = _compute_region_aware_bright_weight(lum, 1.0, 31, 0.0, 99.0)
    assert bw_lum.mean() > 0.5, "smoothstep at lum=2.0 should saturate"
    assert bw_region.mean() < 0.02 * bw_lum.mean(), (
        f"flat-bright must suppress: bw_region.mean={bw_region.mean():.6f} "
        f"vs bw_lum.mean={bw_lum.mean():.6f}"
    )


# T3 — point-light preservation
def test_T3_point_light_preserved():
    """Tiny bright spot: top-hat ≈ L → bw_region ≥ 0.95 · bw_lum at the center."""
    lum = np.zeros((256, 256), dtype=np.float32)
    lum[126:130, 126:130] = 10.0
    bw_lum = _compute_region_aware_bright_weight(lum, 1.0, 0, 0.10, 99.0)
    bw_region = _compute_region_aware_bright_weight(lum, 1.0, 31, 0.0, 99.0)
    ratio = bw_region[128, 128] / max(float(bw_lum[128, 128]), 1e-9)
    assert ratio >= 0.95, f"point-light center must survive top-hat, ratio={ratio:.4f}"


# T4 — floor=1.0 collapses to lum-only
def test_T4_floor_one_equals_killswitch():
    """floor=1.0 → th_clipped = 1.0 → bw_region = bw_lum."""
    rng = np.random.default_rng(13)
    lum = rng.uniform(0.0, 2.5, (96, 96)).astype(np.float32)
    bw_killswitch = _compute_region_aware_bright_weight(lum, 1.0, 0, 0.10, 99.0)
    bw_floor_one = _compute_region_aware_bright_weight(lum, 1.0, 31, 1.0, 99.0)
    np.testing.assert_allclose(bw_killswitch, bw_floor_one, atol=1e-6)


# T5 — floor=0.0 collapses to pure top-hat-norm
def test_T5_floor_zero_collapses_to_pure_tophat_norm():
    """floor=0.0: flat-bg pixels go to ~0 (top-hat = 0); bright dot stays > 0.5."""
    lum = np.full((128, 128), 1.5, dtype=np.float32)
    lum[62:66, 62:66] = 8.0
    bw = _compute_region_aware_bright_weight(lum, 1.0, 15, 0.0, 99.0)
    assert bw[10, 10] < 1e-3, f"flat region must collapse to ~0, got {bw[10, 10]:.6f}"
    assert bw[64, 64] > 0.5, f"bright-dot pixel must be > 0.5, got {bw[64, 64]:.6f}"


# T6 — validators
@pytest.mark.parametrize(
    "kw,bad,err_substr",
    [
        ({"bright_pass_region_size": -1}, "bright_pass_region_size", "[0, 255]"),
        ({"bright_pass_region_size": 256}, "bright_pass_region_size", "[0, 255]"),
        ({"bright_pass_region_size": 1.5}, "bright_pass_region_size", "[0, 255]"),
        ({"bright_pass_floor": 1.5}, "bright_pass_floor", "[0.0, 1.0]"),
        ({"bright_pass_floor": -0.01}, "bright_pass_floor", "[0.0, 1.0]"),
        ({"bright_pass_floor": float("nan")}, "bright_pass_floor", "[0.0, 1.0]"),
        ({"bright_pass_norm_pct": 0.0}, "bright_pass_norm_pct", "[50.0, 100.0]"),
        ({"bright_pass_norm_pct": 49.9}, "bright_pass_norm_pct", "[50.0, 100.0]"),
        ({"bright_pass_norm_pct": 100.5}, "bright_pass_norm_pct", "[50.0, 100.0]"),
        ({"bright_pass_norm_pct": float("inf")}, "bright_pass_norm_pct", "[50.0, 100.0]"),
    ],
)
def test_T6_validators_reject_bad_kwargs(kw, bad, err_substr):
    """All three new kwargs are range-validated by render() with a ValueError."""
    img = np.zeros((32, 32, 3), dtype=np.float32)
    dm = np.zeros((32, 32), dtype=np.float32)
    with pytest.raises(ValueError) as excinfo:
        render_with_defocus(img, _blur_tensor(k=9), dmap=dm, **kw)
    msg = str(excinfo.value)
    assert bad in msg and err_substr in msg, f"unexpected error: {msg!r}"


# T7 — all-zero input is finite
def test_T7_all_zero_input_returns_finite_zeros():
    """Zero lum: bw_lum=0, top-hat=0 → bw_region = 0 (finite, no div-by-zero)."""
    lum = np.zeros((64, 64), dtype=np.float32)
    bw = _compute_region_aware_bright_weight(lum, 1.0, 31, 0.10, 99.0)
    assert np.all(np.isfinite(bw)), "must not produce NaN/Inf on all-zero input"
    assert np.all(bw == 0.0), f"all-zero input must yield all-zero weight, max={bw.max()}"


# Bonus — default render kwargs equal explicit-default path (default-drift guard)
def test_default_render_kwargs_match_no_kwargs_path():
    img = np.random.default_rng(3).uniform(0, 0.9, (64, 64, 3)).astype(np.float32)
    dm = np.full((64, 64), 0.5, dtype=np.float32)
    t = _blur_tensor()
    _a, la = render_with_defocus(img, t, dmap=dm)
    _b, lb = render_with_defocus(
        img, t, dmap=dm,
        bright_pass_region_size=0, bright_pass_floor=0.10, bright_pass_norm_pct=99.0,
    )
    np.testing.assert_array_equal(la, lb)
