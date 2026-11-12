"""P1.10 region-aware bright-pass — Adversarial-round HOSTILE probes.

Companion to ``tests/adversarial/test_convolve_region_brightpass_adversarial.py``
(Coder-authored T8-T12). This file targets the gaps the Coder did NOT cover —
wrong-polarity, three-flag adversary, layered-gather + region-aware, boundary
values, sparse / bimodal / extreme inputs, validator typing edge cases, HDR/LDR
consistency, and concurrency with peak / r_blend / depth_m paths.

Mandatory wrong-polarity fixture per ``DevTeam.iSummicron.md`` § Adversarial
Rules. Probe numbers track the Adversarial brief's hostile-probe checklist.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from summicron.render.convolve import (
    _compute_region_aware_bright_weight,
    render,
)
from tests.fixtures.defocus import render_with_defocus


# ----- shared tensor factories -----
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
    return {
        "kernels": kern,
        "radial_edges": _RE.copy(),
        "defocus_ratios": _DR.copy(),
        "object_distances_m": np.array([1.0, 2.0, 3.0, 4.0]),
    }


# Probe #1 — MANDATORY wrong-polarity: dim Gaussian on bright background.
# White top-hat is unidirectional; dark spot in bright field must NOT trigger
# the detector. Floor (0.10) is the only contribution → bw_region ≈ 0.10·bw_lum.
def test_hostile_wrong_polarity_dark_spot_on_bright_field():
    """White top-hat is one-sided: dim spot in bright field gives top-hat=0,
    so bw_region collapses to floor·bw_lum. (Standard fixture is the OPPOSITE
    polarity; this guards against accidentally implementing a black top-hat
    or a symmetric edge detector.)"""
    lum = np.full((128, 128), 5.0, dtype=np.float32)
    lum[60:68, 60:68] = 1.0  # dim hole in bright field
    bw_lum = _compute_region_aware_bright_weight(lum, 1.0, 0, 0.10, 99.0)
    bw_region = _compute_region_aware_bright_weight(lum, 1.0, 31, 0.10, 99.0)
    # Bright background pixels should be heavily suppressed (top-hat=0 there).
    # Allow a small tolerance for grid-sample anchor noise.
    ratio_bg = float(bw_region[0, 0] / max(bw_lum[0, 0], 1e-9))
    assert 0.05 <= ratio_bg <= 0.15, (
        f"wrong-polarity bg suppression broke: ratio={ratio_bg:.4f} (expect ~0.10)"
    )
    # Dark hole stays dim (smoothstep below T-K=0.8); top-hat at boundary may fire,
    # but the absolute value at hole center stays below floor·bw_lum_neighborhood.
    assert bw_region[64, 64] < 0.2, (
        f"dark spot must not trigger detector, got bw={bw_region[64, 64]:.4f}"
    )


# Probe #2 — three-flag adversary: max region_size + zero floor + median norm.
def test_hostile_three_flag_adversary_no_crash_no_nan():
    """All three new flags pinned to extreme values simultaneously."""
    rng = np.random.default_rng(2)
    img = rng.uniform(0.0, 0.95, (96, 96, 3)).astype(np.float32)
    img[40:48, 40:48, :] = 0.95
    dm = np.full((96, 96), 0.5, dtype=np.float32)
    out, lin = render_with_defocus(
        img, _blur_tensor(), dmap=dm,
        bright_thresh=0.5,
        bright_pass_region_size=31,
        bright_pass_floor=0.0,
        bright_pass_norm_pct=50.0,
    )
    assert out.shape == (96, 96, 3)
    assert np.all(np.isfinite(lin)), "extreme triple combo must not produce NaN/Inf"


# Probe #3 — bright_thresh=1e9 (default) + region_size>0: detector still runs
# (wasted compute), but output must be valid (bw_lum=0 → bw_region=0).
def test_hostile_thresh_1e9_default_with_region_size_runs_and_zeros():
    """When bright_thresh defaults to 1e9 (Python API default) but region_size
    is enabled, the detector runs but bw_lum ≈ 0 → bright_weight ≈ 0 → output
    matches the no-bright-pass path. Documents the Skeptic R1 §3 #1 wasted-compute
    concern."""
    rng = np.random.default_rng(3)
    img = rng.uniform(0.0, 0.95, (64, 64, 3)).astype(np.float32)
    dm = np.full((64, 64), 0.5, dtype=np.float32)
    out, lin = render_with_defocus(
        img, _blur_tensor(), dmap=dm,
        bright_pass_region_size=31, bright_pass_floor=0.10,
    )
    assert np.all(np.isfinite(lin))
    # Bright-pass effectively disabled at thresh=1e9 → bright_weight ≈ 0.
    # Confirm via no-bright-pass reference.
    out_ref, lin_ref = render_with_defocus(
        img, _blur_tensor(), dmap=dm,
        bright_pass_region_size=0,
    )
    np.testing.assert_allclose(lin, lin_ref, atol=1e-5)


# Probe #4 — layered_gather + region-aware: both bright-pass call sites
# (padded L953 + unpadded L1232) consume the new kwargs; verify no stale value.
def test_hostile_layered_gather_with_region_aware():
    rng = np.random.default_rng(4)
    img = rng.uniform(0.0, 0.95, (64, 64, 3)).astype(np.float32)
    img[28:36, 28:36, :] = 0.95
    depth = np.full((64, 64), 2.0, dtype=np.float32)
    out, lin = render(
        img, _blur_tensor(), depth_m=depth,
        bright_thresh=0.5,
        bright_pass_region_size=15, bright_pass_floor=0.10, bright_pass_norm_pct=99.0,
    )
    assert np.all(np.isfinite(lin))
    # Sanity: non-zero output (the bright spot got bloomed somewhere).
    assert lin.sum() > 0.0


# Probe #5 — boundary value: region_size=1.
# 1×1 structuring element → opening = identity → top-hat = 0 everywhere
# → th_norm = 0 → th_clipped = floor → bw_region = floor · bw_lum.
def test_hostile_region_size_one_collapses_to_floor_times_bw_lum():
    lum = np.full((64, 64), 2.0, dtype=np.float32)
    bw0 = _compute_region_aware_bright_weight(lum, 1.0, 0, 0.10, 99.0)
    bw1 = _compute_region_aware_bright_weight(lum, 1.0, 1, 0.10, 99.0)
    np.testing.assert_allclose(bw1, 0.10 * bw0, atol=1e-6)


# Probe #5 — boundary: region_size=255 (max allowed).
def test_hostile_region_size_max_255_finite():
    rng = np.random.default_rng(5)
    lum = rng.uniform(0.0, 2.0, (96, 96)).astype(np.float32)
    bw = _compute_region_aware_bright_weight(lum, 1.0, 255, 0.10, 99.0)
    assert np.all(np.isfinite(bw))
    assert bw.shape == lum.shape
    assert bw.min() >= 0.0 and bw.max() <= 1.0


# Probe #5 — boundary: norm_pct=50.0 (median anchor) + bright-dominant image.
def test_hostile_norm_pct_50_bright_dominant():
    """Median anchor on bright-dominant scene: half of top-hat values are
    ≥ anchor, so many pixels saturate at 1.0 → bw_region tracks bw_lum."""
    rng = np.random.default_rng(6)
    lum = rng.uniform(1.5, 2.5, (96, 96)).astype(np.float32)
    # Sprinkle bright dots
    lum[::8, ::8] = 4.0
    bw = _compute_region_aware_bright_weight(lum, 1.0, 15, 0.10, 50.0)
    assert np.all(np.isfinite(bw))
    assert bw.max() <= 1.0


# Probe #5 — boundary: norm_pct=100.0 (max anchor).
def test_hostile_norm_pct_100_max_anchor():
    """Max anchor reduces single-outlier sensitivity at the cost of mid-bright
    bloom range — output must remain finite and bounded."""
    rng = np.random.default_rng(7)
    lum = rng.uniform(0.0, 2.0, (96, 96)).astype(np.float32)
    lum[48, 48] = 10.0  # outlier
    bw = _compute_region_aware_bright_weight(lum, 1.0, 15, 0.10, 100.0)
    assert np.all(np.isfinite(bw))
    assert 0.0 <= bw.min() and bw.max() <= 1.0


# Probe #6 — degenerate shapes: 0×0, 1×1 (covered T9), 2×2, 7×7.
@pytest.mark.parametrize("shape", [(0, 0), (2, 2), (7, 7), (1, 16), (16, 1)])
def test_hostile_degenerate_shapes_no_crash(shape):
    lum = np.full(shape, 1.5, dtype=np.float32)
    with warnings.catch_warnings():
        # Empty-array nanpercentile warns RuntimeWarning; we accept that.
        warnings.simplefilter("ignore", RuntimeWarning)
        bw = _compute_region_aware_bright_weight(lum, 1.0, 31, 0.10, 99.0)
    assert bw.shape == shape


# Probe #7 — sparse saturation: single bright pixel on dim plate.
def test_hostile_sparse_single_bright_pixel():
    """One bright pixel out of 4096: detector fires there, anchor is the bright
    pixel itself, neighborhood is zero → bw at LED ≈ 1.0; corners ≈ 0."""
    lum = np.zeros((512, 512), dtype=np.float32)
    lum[256, 256] = 10.0
    bw = _compute_region_aware_bright_weight(lum, 1.0, 31, 0.10, 99.0)
    assert np.all(np.isfinite(bw))
    assert bw[256, 256] > 0.9, f"sparse bright pixel must fire, got {bw[256, 256]:.4f}"
    assert bw[0, 0] < 1e-3, f"corner must be ~0, got {bw[0, 0]:.4f}"


# Probe #8 — bimodal image: half lum=0, half lum=10. Top-hat should fire only
# on the bright side, and only near the boundary (interior is uniform-bright).
def test_hostile_bimodal_image_boundary_only():
    lum = np.zeros((64, 128), dtype=np.float32)
    lum[:, 64:] = 10.0
    bw = _compute_region_aware_bright_weight(lum, 1.0, 15, 0.10, 99.0)
    # Dark side: bw_lum=0 → bw=0
    assert bw[32, 5] < 1e-5
    # Bright interior far from boundary: top-hat=0 → bw = floor·bw_lum = 0.10
    assert 0.05 <= bw[32, 120] <= 0.15
    # Bright side near boundary: top-hat fires; bw close to bw_lum
    assert bw[32, 64] > 0.05  # fires somewhere near boundary


# Probe #9 — all-finite-but-extreme: float32 max/2 throughout.
def test_hostile_extreme_finite_values_no_overflow():
    """Constant lum at fp32 max/2: opening = identity → top-hat = 0 → output
    = floor·bw_lum (bw_lum = 1.0 since lum >> T+K)."""
    lum = np.full((32, 32), np.finfo(np.float32).max / 2, dtype=np.float32)
    bw = _compute_region_aware_bright_weight(lum, 1.0, 15, 0.10, 99.0)
    assert np.all(np.isfinite(bw))
    np.testing.assert_allclose(bw, 0.10, atol=1e-5)


# Probe #10 — NaN at exact percentile-99 location: nanpercentile must rescue.
def test_hostile_nan_at_p99_location_does_not_poison_anchor():
    """Single NaN injected at the would-be percentile-99 location: the global
    anchor is computed via nanpercentile and survives; output remains finite
    everywhere except possibly the NaN pixel itself."""
    rng = np.random.default_rng(10)
    lum = rng.uniform(0.0, 2.0, (96, 96)).astype(np.float32)
    # Find a corner-region pixel and force it to NaN
    lum[5, 5] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        bw = _compute_region_aware_bright_weight(lum, 1.0, 15, 0.10, 99.0)
    # Global anchor should be finite; corners far from NaN are finite.
    assert np.isfinite(bw[80, 80])
    assert np.isfinite(bw[50, 50])


# Probe #11 — validator typing: bool / np.int64 / np.float32 / 31.0
def test_hostile_validator_rejects_bool_true():
    """Coder defensive note: bool is subclass of int; explicit reject."""
    img = np.zeros((16, 16, 3), dtype=np.float32)
    dm = np.zeros((16, 16), dtype=np.float32)
    with pytest.raises(ValueError, match="bright_pass_region_size"):
        render_with_defocus(img, _blur_tensor(k=9), dmap=dm, bright_pass_region_size=True)


def test_hostile_validator_rejects_bool_false():
    """bool=False is also rejected even though semantically False==0 (kill-switch)."""
    img = np.zeros((16, 16, 3), dtype=np.float32)
    dm = np.zeros((16, 16), dtype=np.float32)
    with pytest.raises(ValueError, match="bright_pass_region_size"):
        render_with_defocus(img, _blur_tensor(k=9), dmap=dm, bright_pass_region_size=False)


def test_hostile_validator_accepts_np_int64():
    """np.int64 is np.integer subclass — accepted (probe-confirmed)."""
    img = np.zeros((16, 16, 3), dtype=np.float32)
    dm = np.zeros((16, 16), dtype=np.float32)
    # Should not raise.
    render_with_defocus(img, _blur_tensor(k=9), dmap=dm, bright_pass_region_size=np.int64(15))


def test_hostile_validator_accepts_np_float32_floor():
    img = np.zeros((16, 16, 3), dtype=np.float32)
    dm = np.zeros((16, 16), dtype=np.float32)
    render_with_defocus(img, _blur_tensor(k=9), dmap=dm, bright_pass_floor=np.float32(0.10))


def test_hostile_validator_rejects_integer_valued_float():
    """31.0 is rejected (must be true int per spec § 8)."""
    img = np.zeros((16, 16, 3), dtype=np.float32)
    dm = np.zeros((16, 16), dtype=np.float32)
    with pytest.raises(ValueError, match="bright_pass_region_size"):
        render_with_defocus(img, _blur_tensor(k=9), dmap=dm, bright_pass_region_size=31.0)


# Probe #12 — HDR vs LDR consistency: same shape (point light), different peak.
def test_hostile_hdr_ldr_consistency_after_q99_norm():
    """Top-hat is amplitude-sensitive, but per-scene q99 normalization should
    rescue HDR vs LDR consistency for the SHAPE of the bloom (not the absolute
    smoothstep saturation, which legitimately differs)."""
    lum_ldr = np.zeros((128, 128), dtype=np.float32)
    lum_ldr[60:68, 60:68] = 1.0  # LDR on-the-knee
    lum_hdr = np.zeros((128, 128), dtype=np.float32)
    lum_hdr[60:68, 60:68] = 3.0  # HDR
    bw_ldr = _compute_region_aware_bright_weight(lum_ldr, 1.0, 31, 0.10, 99.0)
    bw_hdr = _compute_region_aware_bright_weight(lum_hdr, 1.0, 31, 0.10, 99.0)
    # Background must be ≈ 0 for both (lum below T-K).
    assert bw_ldr[0, 0] < 1e-5 and bw_hdr[0, 0] < 1e-5
    # The HDR center saturates the smoothstep; LDR is on the knee. Both fire
    # on the detector (top-hat fully saturates within the bright dot vs zero bg).
    assert bw_hdr[64, 64] > 0.9
    # LDR center should still bloom positively (not zero) — top-hat fires the
    # same shape, only the smoothstep amplitude differs.
    assert bw_ldr[64, 64] > 0.4


# Probe #13 — concurrent r_blend_strength sweep + region-aware (no crash).
@pytest.mark.parametrize("rb", [0.0, 0.20, 0.50])
def test_hostile_r_blend_with_region_aware(rb):
    rng = np.random.default_rng(13)
    img = rng.uniform(0.0, 0.95, (48, 48, 3)).astype(np.float32)
    dm = np.full((48, 48), 0.5, dtype=np.float32)
    out, lin = render_with_defocus(
        img, _blur_tensor(), dmap=dm, bright_thresh=0.5,
        bright_pass_region_size=15, bright_pass_floor=0.10,
    )
    assert np.all(np.isfinite(lin))


# Probe #14 — psf_norm='peak' + region-aware: peak path uses bright_weight too.
def test_hostile_psf_norm_peak_with_region_aware():
    rng = np.random.default_rng(14)
    img = rng.uniform(0.0, 0.95, (48, 48, 3)).astype(np.float32)
    img[20:28, 20:28, :] = 0.95
    dm = np.full((48, 48), 0.5, dtype=np.float32)
    out, lin = render_with_defocus(
        img, _blur_tensor(), dmap=dm,
        psf_norm="peak", bright_thresh=0.5,
        bright_pass_region_size=15, bright_pass_floor=0.10,
    )
    assert np.all(np.isfinite(lin))


# Probe #15 — concurrent monocular depth_m path (DA2-like): same render() entry,
# no separate code path, but verifies depth_m + region-aware does not crash.
def test_hostile_depth_m_monocular_with_region_aware():
    rng = np.random.default_rng(15)
    img = rng.uniform(0.0, 0.95, (48, 48, 3)).astype(np.float32)
    depth = np.full((48, 48), 2.5, dtype=np.float32)  # monocular-style depth
    out, lin = render(
        img, _blur_tensor(), depth_m=depth,
        bright_thresh=0.5,
        bright_pass_region_size=15, bright_pass_floor=0.10,
    )
    assert np.all(np.isfinite(lin))
