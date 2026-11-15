"""S33 Tier B WP2, gate G-B6: configurations that need no engine switch.

Each row is judged by calling the tiled gather directly against the reference
gather (Tier B ceilings, full frame and field-edge annulus):

* ``--downres`` giving K = 17, 33, 67, 131 (FAST with a forced ``_tile_n``; the
  production-N versions are SLOW);
* ``slab_batch_size`` 1 / 3 / D: the tiled output is ``np.array_equal``;
* linear mapping (quantised staircase) and monocular (smooth, noisy) depth;
* a small lens-50 frame;
* a custom tensor with K > 255 is "not available".
"""
from __future__ import annotations

import numpy as np
import pytest

from summicron.render import _tile_schedule as ts
from tests.fixtures import tolerance as tol
from tests.fixtures.synthetic_tensor import make_synthetic_tensor
from tests.fixtures.truth import assert_stress
from tests.unit.test_s33_tiled_core import (  # noqa: F401  (shared helpers)
    _gather_kwargs, _run_ref, _run_tiled, _tensor, _total,
)
from tests.unit.test_s33_tiled_field_character import (
    AMPS, build_chart, chart_coverage, judge,
)

# K -> forced FAST N (T = N - K + 1 >= 24) and a frame holding >= 3x3 tiles. The
# frames keep the light density at or below about 1 light per 3500 px: denser
# HDR charts sit on the float32 round-off floor (the dense-highlight family, R-B2)
# and would judge noise, not geometry.
FAST_K = {17: (40, (200, 190)), 33: (64, (220, 200)), 67: (96, (240, 220)),
          131: (192, (190, 200))}


def _chart_case(h, w, K, *, tile_n, n_lights, amp_name="hdr16_l1g2", lens="35",
                depth_fn=None, tensor=None, amp_gain=None):
    amp, gain = amp_gain if amp_gain is not None else AMPS[amp_name]
    t = dict(tensor if tensor is not None else _tensor(lens))
    img, depth, lights = build_chart(h, w, amp, n_lights=n_lights, tensor=t)
    if depth_fn is not None:
        depth = depth_fn(h, w, np.asarray(t["object_distances_m"], np.float64))
    kw = _gather_kwargs(img, depth, K=K, thresh=0.5, bright_stamp_gain=gain,
                        lens=lens, tensor=t)
    ref5, _ = _run_ref(kw)
    got5, _ = _run_tiled(kw, **({"_tile_n": tile_n} if tile_n else {}))
    assert float(ref5[1].max()) > 0.0 or K <= 17     # bright image used
    return ref5, got5, lights, kw


# --------------------------------------------------------------------------
# --downres: K = 17 / 33 / 67 / 131
# --------------------------------------------------------------------------

@pytest.mark.parametrize("K", [17, 33, 67])
def test_downres_k_forced_tiles(K):
    N, (h, w) = FAST_K[K]
    _, tiles = ts.plan_tiles(h, w, K, N)
    assert len({t[0] for t in tiles}) >= 3 and len({t[2] for t in tiles}) >= 3
    # HDR 16x at L1 gain 2 (amp_name default) is the standard chart: hard Tier B.
    ref5, got5, lights, _ = _chart_case(h, w, K, tile_n=N, n_lights=6 if K < 67 else 12)
    res = tol.evaluate(_total(ref5), _total(got5), "B")
    assert res["ok"], f"K={K}: {tol.summary(res)}"
    for i in (2, 3, 4):
        assert np.array_equal(ref5[i], got5[i])
    print(f"downres K={K} N={N} {h}x{w}: {tol.summary(res)}")


def test_downres_k131_forced_tiles():
    """K=131 with _tile_n=192 on a 190x200 frame (a K=131 production frame is
    4096-class; this one checks geometry at a different K than 67 / 183)."""
    N, (h, w) = FAST_K[131]
    ref5, got5, lights, _ = _chart_case(h, w, 131, tile_n=N, n_lights=12)
    res = tol.evaluate(_total(ref5), _total(got5), "B")
    assert res["ok"], f"K=131: {tol.summary(res)}"
    for i in (2, 3, 4):
        assert np.array_equal(ref5[i], got5[i])
    print(f"downres K=131 N={N} {h}x{w}: {tol.summary(res)}")


@pytest.mark.slow
@pytest.mark.parametrize("K,hw", [(17, (640, 700)), (33, (640, 700)),
                                  (67, (700, 640)), (131, (820, 780))])
def test_downres_k_production_n(K, hw):
    """Production N (324 for K <= 99, 512 above) on frames holding >= 2x2 tiles."""
    ref5, got5, lights, _ = _chart_case(*hw, K, tile_n=None, n_lights=16)
    N = ts.tile_n_for_ksize(K)
    _, tiles = ts.plan_tiles(*hw, K, N)
    assert len(tiles) >= 4
    judge(ref5, got5, lights, f"downres K={K} N={N} {hw}")


# --------------------------------------------------------------------------
# slab_batch_size invariance
# --------------------------------------------------------------------------

def test_slab_batch_size_invariance_tiled_array_equal():
    img, depth, _ = build_chart(240, 220, 16.0, n_lights=12, tensor=_tensor("35"))
    outs = {}
    for sbs in (1, 3, None, 11):          # None = the module default; 11 = D
        kw = _gather_kwargs(img, depth, K=31, thresh=0.5, bright_stamp_gain=2.0)
        kw["slab_batch_size"] = sbs
        outs[sbs], _ = _run_tiled(kw, _tile_n=64)
    base = outs[None]
    for sbs in (1, 3, 11):
        for i in range(5):
            assert np.array_equal(base[i], outs[sbs][i]), (sbs, i)
            assert base[i].tobytes() == outs[sbs][i].tobytes(), (sbs, i)
    # and the batch size does not move the reference either (same Tier B bound)
    kw = _gather_kwargs(img, depth, K=31, thresh=0.5, bright_stamp_gain=2.0)
    kw["slab_batch_size"] = 3
    ref5, _ = _run_ref(kw)
    res = tol.evaluate(_total(ref5), _total(base), "B")
    assert res["ok"], tol.summary(res)


def test_slab_batch_size_validation_matches_reference():
    img, depth, _ = build_chart(60, 60, 16.0, n_lights=12, tensor=_tensor("35"))
    for bad in (0, -2):
        kw = _gather_kwargs(img, depth, K=17, thresh=0.5, bright_stamp_gain=2.0)
        kw["slab_batch_size"] = bad
        with pytest.raises(ValueError) as e_ref:
            _run_ref(kw)
        with pytest.raises(ValueError) as e_til:
            _run_tiled(kw, _tile_n=40)
        assert str(e_ref.value) == str(e_til.value)


# --------------------------------------------------------------------------
# linear mapping / monocular depth
# --------------------------------------------------------------------------

def _staircase(h, w, od):
    """LiDAR-like: a few quantised depth steps in a linear ramp (--depth-mapping
    linear gives metres through a scale; the tiled engine only sees depth_m)."""
    ramp = np.linspace(od[0], od[-1], w, dtype=np.float64)
    ramp = np.round(ramp / 1.25) * 1.25 + 0.3
    yy = np.linspace(0.0, 2.0, h)[:, None]
    return np.clip(ramp[None, :] + yy, od[0], od[-1]).astype(np.float32)


def _monocular(h, w, od, seed=17):
    """Smooth non-monotone relative depth plus noise, geometric over all slabs."""
    rng = np.random.default_rng(seed)
    f = rng.normal(size=(h // 8 + 2, w // 8 + 2))
    f = np.kron(f, np.ones((8, 8)))[:h, :w]
    f = (f - f.min()) / (np.ptp(f) + 1e-9)
    f = np.clip(f + rng.normal(0, 0.02, (h, w)), 0.0, 1.0)
    return np.exp(np.log(od[0]) + f * (np.log(od[-1]) - np.log(od[0]))).astype(np.float32)


@pytest.mark.parametrize("name,fn", [("linear-staircase", _staircase),
                                     ("monocular", _monocular)])
@pytest.mark.parametrize("hw", [(260, 220), (220, 260)])
def test_depth_field_families(name, fn, hw):
    ref5, got5, lights, _ = _chart_case(*hw, 67, tile_n=96, n_lights=16, depth_fn=fn)
    judge(ref5, got5, lights, f"{name} {hw}")


# --------------------------------------------------------------------------
# lens 50
# --------------------------------------------------------------------------

@pytest.mark.parametrize("hw", [(360, 320), (320, 360)])
def test_lens50_small_frame_hdr16(hw):
    amp, gain = AMPS["hdr16_l1g2"]
    t = dict(_tensor("50"))
    img, depth, lights = build_chart(*hw, amp, n_lights=16, tensor=t)
    ris, azs = chart_coverage(lights)
    assert ris == set(range(12)) and len(azs) >= 16
    kw = _gather_kwargs(img, depth, K=67, thresh=0.5, bright_stamp_gain=gain, lens="50")
    ref5, _ = _run_ref(kw)
    got5, _ = _run_tiled(kw, _tile_n=128)
    assert float(ref5[1].max()) > 0.01
    judge(ref5, got5, lights, f"lens50 {hw} hdr16_l1g2")


@pytest.mark.parametrize("hw", [(900, 700), (700, 900)])
def test_lens50_ldr_peak_production_n(hw):
    """Lens 50's peak stamp concentrates far more energy per light than lens
    35's. On a frame that is not dense (production N=324, 900 x 700, 16 lights)
    the full hard Tier B (p99.9 included) holds, full frame and annulus."""
    amp, gain = AMPS["ldr_peak"]
    t = dict(_tensor("50"))
    img, depth, lights = build_chart(*hw, amp, n_lights=16, tensor=t)
    ris, azs = chart_coverage(lights)
    assert ris == set(range(12)) and len(azs) >= 16
    kw = _gather_kwargs(img, depth, K=67, thresh=0.5, bright_stamp_gain=gain, lens="50")
    ref5, _ = _run_ref(kw)
    got5, _ = _run_tiled(kw)
    _, tiles = ts.plan_tiles(*hw, 67, ts.tile_n_for_ksize(67))
    assert len(tiles) >= 4
    assert float(ref5[1].max()) > 0.01
    judge(ref5, got5, lights, f"lens50 peak production N {hw}")


@pytest.mark.parametrize("hw", [(360, 320), (320, 360)])
def test_lens50_ldr_peak_dense_small_frame_stress_ratio(hw):
    """The dense small-frame case (16 lights on a 360 x 320 frame, N=128) sits
    on the float32 floor for both engines (raw Tier B p99.9 ~1.2e-6): stress
    family (R-B2 revised), gated by the ratio vs a float64 truth."""
    amp, gain = AMPS["ldr_peak"]
    t = dict(_tensor("50"))
    img, depth, lights = build_chart(*hw, amp, n_lights=16, tensor=t)
    ris, azs = chart_coverage(lights)
    assert ris == set(range(12)) and len(azs) >= 16
    kw = _gather_kwargs(img, depth, K=67, thresh=0.5, bright_stamp_gain=gain, lens="50")
    ref5, _ = _run_ref(kw)
    got5, _ = _run_tiled(kw, _tile_n=128)
    assert float(ref5[1].max()) > 0.01
    assert_stress(kw, ref5, got5, f"lens50 ldr_peak dense {hw}")


# --------------------------------------------------------------------------
# K > 255
# --------------------------------------------------------------------------

def test_custom_tensor_k_over_255_is_not_available():
    t = make_synthetic_tensor(12, 5, 257, 257)
    h, w = 40, 40
    img = np.full((h, w, 3), 0.3, np.float32)
    depth = np.full((h, w), float(t["object_distances_m"][2]), np.float32)
    kw = _gather_kwargs(img, depth, K=257, tensor=t)
    assert kw["scaled_ksize"] == 257
    with pytest.raises(ValueError, match="not available"):
        _run_tiled(kw)
