"""R-native-mem honesty tests for the structural memory estimator (spec
``doc/plan/phase2/section17_native-memory/locked-spec-mem.md`` §3.9c / §3.9d).

``tracemalloc`` counts exactly the numpy bytes a render allocates, so the
estimator's numpy-side phase terms (``max(prep, gather, over)``) can be checked
against a measured peak deterministically (no allocator-lag noise). The OS-level
check (estimate >= measured ``phys_footprint``) is the G5 probe gate, not here.
"""
from __future__ import annotations

import tracemalloc
import warnings

import numpy as np
import pytest

import summicron.render.convolve as cv
from summicron.render.convolve import (
    MAX_BYTES,
    PLATFORM_BASE,
    _build_image_fft_state,
    _estimate_peak_breakdown,
    _estimate_peak_bytes,
    _fft_geometry,
)

_KSIZE = 31  # synthetic tensor: corpus_ref_long_edge absent => scaled_ksize == kH
_NR = 12


def _tensor(d: int, k: int = _KSIZE, n_r: int = _NR) -> dict:
    od = (
        np.array([0.7, 0.98, 1.37, 1.91, 2.68, 3.74, 5.23, 7.32, 10.23, 14.30, 20.0])
        if d <= 11 else np.geomspace(0.7, 20.0, d)
    )[:d].astype(np.float64)
    kern = np.zeros((n_r, d, k, k), np.float32)
    c = k // 2
    kern[:, 0, c, c] = 1.0
    ys, xs = np.mgrid[-c:c + 1, -c:c + 1].astype(np.float32)
    for di in range(1, d):
        g = np.exp(-(xs ** 2 + ys ** 2) / (2 * (1.0 + 0.5 * di) ** 2))
        kern[:, di] = g / g.sum()
    return {
        "kernels": kern,
        "radial_edges": np.sqrt(np.linspace(0.0, 1.0, n_r + 1)),
        "defocus_ratios": np.linspace(0.0, 1.0, d),
        "object_distances_m": od,
        "focus_distance_m": 0.7,
        "corpus_ref_long_edge": None,
        "otf_norm_applied": False,
        "delta_guard_applied": True,
    }


def _scene(h: int, w: int, d: int):
    """Image + a depth RAMP across all distances plus a STEP (activates every
    slab and the regime-b OVER)."""
    rng = np.random.default_rng(0)
    img = rng.uniform(0.0, 0.8, (h, w, 3)).astype(np.float32)
    od = _tensor(d)["object_distances_m"]
    ramp = np.geomspace(od[0], od[-1], w, dtype=np.float32)
    depth = np.broadcast_to(ramp[None, :], (h, w)).copy()
    depth[h // 3: 2 * h // 3, w // 3: 2 * w // 3] = np.float32(od[min(2, d - 1)])
    return img, depth


def _measure(h, w, d, occlusion, batch) -> tuple[int, dict]:
    """tracemalloc peak delta inside render() and the estimator breakdown.

    ``hdr_source="macos"`` keeps ``img_lin`` the caller's array (as in the
    production HDR path), so the delta holds render()'s own allocations."""
    img, depth = _scene(h, w, d)
    tensor = _tensor(d)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tracemalloc.start()
        try:
            tracemalloc.reset_peak()
            base = tracemalloc.get_traced_memory()[0]
            cv.render(
                img, tensor, depth_m=depth, hdr_source="macos",
                occlusion_compositing="on" if occlusion else "off",
                slab_batch_size=batch,
            )
            delta = tracemalloc.get_traced_memory()[1] - base
        finally:
            tracemalloc.stop()
    est = _estimate_peak_breakdown(
        h, w, d, occlusion=occlusion, kernel_size=_KSIZE, n_radial=_NR,
        slab_batch_size=batch,
    )
    return delta, est


def _check(h, w, d, occlusion, batch, slack):
    delta, est = _measure(h, w, d, occlusion, batch)
    phase = max(est["prep"], est["gather"], est["over"])
    assert phase >= 0.98 * delta - slack, (
        f"UNDER-estimate {h}x{w} D={d} b={batch} occ={occlusion}: "
        f"phase {phase} < 0.98 x measured {delta} - {slack}"
    )
    assert phase <= 1.25 * delta, (
        f"BLOATED estimate {h}x{w} D={d} b={batch} occ={occlusion}: "
        f"phase {phase} > 1.25 x measured {delta}"
    )


# Always-on (FAST): one OFF and one ON combo, both orientations.
@pytest.mark.parametrize(
    "h, w, occlusion", [(256, 192, False), (192, 256, True)],
)
def test_estimate_honest_vs_tracemalloc_fast(h, w, occlusion):
    _check(h, w, 11, occlusion, 1, 512 * 1024)


_SLOW_COMBOS = [
    (h, w, occ, d, b)
    for (h, w) in [(256, 192), (192, 256)]
    for occ in (False, True)
    for (d, b) in [(11, 1), (11, 11), (3, 1)]
]


@pytest.mark.slow
@pytest.mark.parametrize("h, w, occlusion, d, batch", _SLOW_COMBOS)
def test_estimate_honest_vs_tracemalloc_grid(h, w, occlusion, d, batch):
    _check(h, w, d, occlusion, batch, 512 * 1024)


@pytest.mark.slow
@pytest.mark.parametrize("occlusion", [False, True])
def test_estimate_honest_vs_tracemalloc_512x384(occlusion):
    _check(512, 384, 11, occlusion, 1, 1024 * 1024)


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------

def _e(h, w, d=11, **kw):
    kw.setdefault("kernel_size", 65)
    return _estimate_peak_bytes(h, w, d, **kw)


def test_monotone_in_pixels_d_kernel_and_nradial():
    assert _e(256, 256) < _e(512, 512) < _e(1024, 1024)
    assert _e(512, 512, 1) <= _e(512, 512, 11) <= _e(512, 512, 22)
    assert _e(512, 512, kernel_size=31) < _e(512, 512, kernel_size=65) < _e(512, 512, kernel_size=183)
    assert _e(512, 512, n_radial=6) < _e(512, 512, n_radial=12) < _e(512, 512, n_radial=24)
    for occ in (False, True):
        assert _e(512, 512, occlusion=occ) <= _e(512, 512, 11, occlusion=occ)


def test_on_at_least_off():
    for h, w, d in [(256, 192, 3), (512, 512, 11), (4284, 5712, 11)]:
        assert _e(h, w, d, occlusion=True) >= _e(h, w, d, occlusion=False)


def test_batch_4_exceeds_batch_1_for_d_ge_4():
    for d in (4, 11, 22):
        assert _e(512, 512, d, slab_batch_size=4) > _e(512, 512, d, slab_batch_size=1)
    # batch is clamped to D: no extra retention beyond D slabs
    assert _e(512, 512, 3, slab_batch_size=3) == _e(512, 512, 3, slab_batch_size=11)


def test_breakdown_shape_and_total_rule():
    bd = _estimate_peak_breakdown(1024, 768, 11, occlusion=True, kernel_size=65)
    assert set(bd) == {"pipeline", "prep", "gather", "over", "base", "total"}
    assert bd["base"] == PLATFORM_BASE
    assert bd["total"] == bd["pipeline"] + max(bd["prep"], bd["gather"], bd["over"]) + bd["base"]
    off = _estimate_peak_breakdown(1024, 768, 11, occlusion=False, kernel_size=65)
    assert off["over"] == 0 and off["gather"] == bd["gather"]


@pytest.mark.parametrize("h, w, k", [(37, 53, 7), (256, 192, 31), (100, 100, 65)])
def test_fft_geometry_matches_build_image_fft_state(h, w, k):
    pad = k // 2
    geo = _fft_geometry(h, w, k)
    padded = np.zeros((h + 2 * pad, w + 2 * pad, 3), np.float32)
    st = _build_image_fft_state(padded, padded, padded, k)
    assert st["fshape"] == geo["fshape"]
    assert st["full_shape"] == geo["full_shape"]
    assert geo["padded_shape"] == padded.shape[:2]
    assert st["img_F"].shape == (3, *geo["spec_shape"])


# ---------------------------------------------------------------------------
# 9c: CI determinism - the FAST-suite shapes never approach the cap
# ---------------------------------------------------------------------------

def test_largest_fast_suite_shape_is_far_below_cap(monkeypatch):
    """1024x1024, D=22, k=183, ON bounds every render in the FAST suite (the
    Coder verified no FAST test renders larger). The numpy-model part of the
    estimate (everything but the fixed PLATFORM_BASE) must stay within 10% of
    the cap, so the CI path can never trip the guard."""
    est = _estimate_peak_bytes(1024, 1024, 22, occlusion=True, kernel_size=183, n_radial=12)
    assert est - PLATFORM_BASE <= 0.10 * MAX_BYTES, (
        f"estimate {est} - base {PLATFORM_BASE} > 10% of cap {MAX_BYTES}"
    )
    # The autouse conftest fixture clears the env override.
    import os
    assert "ISUMMICRON_MAX_BYTES" not in os.environ
