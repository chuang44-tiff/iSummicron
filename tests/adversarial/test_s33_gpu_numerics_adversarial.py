"""S33 Tier C adversarial (numerics lens): try to break the MLX gpu engine's
tolerance contract against the in-tree reference gather.

Self-contained: no ``runs/``, no goldens, no IMG_6915, no other checkout. The
oracle is ``convolve._layered_gather_diffuse`` (the reference) run in-test,
judged by the test-local copy of the frozen metric
(``tests/fixtures/tolerance.py``, Tier C ceilings, full frame AND field-edge
annulus) or, for the ruled STRESS family only, by the p99.9 ratio against a
float64 truth (``tests/fixtures/truth.py``) at the standing gpu ceiling 7
(user ruling, 2026-11-14, float32 spectra ruling). Ceilings are never loosened.

Classification (user ruling): the hard Tier C ceilings apply to scenes and
charts at L1 gain <= 4 with lights at the goldens' linear range (<= 6.8x).
Dense 16x HDR fields, 50x+ lights and the 1e3 light are stress (ratio only);
1e5 is out of scope (report only).

Batch configurations: MLX ``irfft2`` is not batch-invariant, so different
``_pairs`` / ``_tile_chunk`` / ``_host_acc`` configurations are each judged
against the reference by Tier C, never bit-equal to each other. Bit-equality is
asserted only where the contract promises it: weight_sum / delta_weight_sum /
acc_alpha vs the reference, the same configuration run twice, the CPU planner
thread count, and the global default device.

The engine-unavailable tests carry no ``gpu`` marker and run in both envs; the
MLX tests are marked ``gpu`` and skip where mlx/Metal is absent, EXCEPT in the
dedicated mlx env, where they run (and fail) instead of skipping.
"""
from __future__ import annotations

import sys
import warnings
from functools import lru_cache

import numpy as np
import pytest

import summicron.render.convolve as cv
from summicron.render import _engine_gpu as eg
from summicron.render import _engine_tiled as et
from summicron.render._artifacts import default_tensor_path
from tests.fixtures import tolerance as tol
from tests.fixtures.truth import format_ratio, stress_ratio

_GPU_REASON = eg.availability()
_IN_GPU_ENV = sys.prefix.rstrip("/").endswith("isummicron-gpu")

needs_gpu = [
    pytest.mark.gpu,
    pytest.mark.skipif(_GPU_REASON is not None and not _IN_GPU_ENV,
                       reason=f"gpu engine unavailable: {_GPU_REASON}"),
]



def _gpu_test(fn):
    for m in needs_gpu:
        fn = m(fn)
    return fn


GPU_STRESS_RATIO = 7.0     # user ruling 2026-11-14 (float32 spectra); Tier B is 1.5
_C = tol.TIERS["C"]


# --------------------------------------------------------------------------
# builders (self-contained copies of what render() hands to the gather)
# --------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _tensor(lens: str) -> dict:
    return cv.load_tensor(default_tensor_path(lens))


def _od():
    return np.asarray(_tensor("35")["object_distances_m"])


def _geometry(h, w, radial_edges):
    ys = np.arange(h, dtype=np.float32) - np.float32((h - 1) / 2.0)
    xs = np.arange(w, dtype=np.float32) - np.float32((w - 1) / 2.0)
    max_radial = max(float(np.hypot((h - 1) / 2.0, (w - 1) / 2.0)), 1e-6)
    rf = (np.hypot(ys[:, None], xs[None, :]) / np.float32(max_radial)).astype(np.float32)
    theta = np.arctan2(ys[:, None], xs[None, :])
    wedge = ((theta + np.pi) / (2 * np.pi) * cv.N_ANG).astype(np.int32) % cv.N_ANG
    rc = 0.5 * (radial_edges[:-1] + radial_edges[1:])
    return rf, wedge, rc


def _kw(img, depth, *, lens="35", K=17, thresh=1.0, gain=2.0, tensor=None,
        slab_w=None):
    t = tensor if tensor is not None else _tensor(lens)
    kernels = t["kernels"]
    kH, kW = kernels.shape[2], kernels.shape[3]
    H, W = img.shape[:2]
    pix_scale = 1.0 if K == kH else K / kH
    pad = K // 2
    rf, wedge, rc = _geometry(H, W, np.asarray(t["radial_edges"]))
    ip = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
    lum = 0.2126 * ip[..., 0] + 0.7152 * ip[..., 1] + 0.0722 * ip[..., 2]
    bright = ip * (lum > thresh).astype(np.float32)[..., None]
    if slab_w is None:
        slab_w = cv.compute_slab_weights(depth, t["object_distances_m"])
    return dict(
        depth_m=depth, slab_w=slab_w, img_lin=img,
        img_padded=ip, img_diffuse_padded=ip - bright, img_bright_padded=bright,
        kernels=kernels, defocus_ratios=t["defocus_ratios"],
        radial_frac_map=rf, r_centers=rc, wedge_idx=wedge,
        kH=kH, kW=kW, pad=pad, pix_scale=pix_scale, scaled_ksize=K, H=H, W=W,
        slab_batch_size=None, cancel=None, bright_stamp_gain=gain,
    )


def _ref(kw):
    k = dict(kw)
    k["fft_state"] = cv._build_image_fft_state(
        kw["img_padded"], kw["img_diffuse_padded"], kw["img_bright_padded"],
        kw["scaled_ksize"])
    k["on_progress"] = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return cv._layered_gather_diffuse(**k)


def _gpu(kw, **extra):
    k = dict(kw)
    k["fft_state"] = None
    k["on_progress"] = None
    k.update(extra)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return eg.gpu_layered_gather(**k)


def _total(t5):
    return (t5[0] + t5[1]).astype(np.float32)


def _exact_parts(ref5, got5, what):
    for i, name in ((2, "weight_sum"), (3, "delta_weight_sum"), (4, "acc_alpha")):
        assert np.array_equal(ref5[i], got5[i]), f"{what}: {name} differs"


def _assert_tier_c(ref5, got5, what):
    assert np.isfinite(got5[0]).all() and np.isfinite(got5[1]).all(), what
    res = tol.evaluate(_total(ref5), _total(got5), "C")
    assert res["ok"], f"{what}: {tol.summary(res)}"
    eb = np.abs(got5[1].astype(np.float64) - ref5[1]) / np.maximum(ref5[1], 1.0)
    assert float(eb.max()) <= _C["max_e"], f"{what} (bright) max e {float(eb.max()):.3g}"
    _exact_parts(ref5, got5, what)


def _assert_gpu_stress(kw, ref5, got5, what):
    _exact_parts(ref5, got5, what)
    assert np.isfinite(got5[0]).all() and np.isfinite(got5[1]).all(), what
    tc = tol.summary(tol.evaluate(_total(ref5), _total(got5), "C"))
    res = stress_ratio(kw, ref5, got5)
    print(f"[gpu-stress] {what}\n  raw Tier C: {tc}\n  vs truth: {format_ratio(res)}")
    for reg, v in res.items():
        assert v["ratio"][1] <= GPU_STRESS_RATIO, (
            f"{what}: {reg} p99.9 ratio {v['ratio'][1]:.3g} > {GPU_STRESS_RATIO}: "
            f"{format_ratio(res)}")


def _ramp_depth(h, w, patches=True):
    od = _od()
    ramp = np.geomspace(od[0], od[-1], w, dtype=np.float32)
    depth = np.broadcast_to(ramp[None, :], (h, w)).copy()
    if patches and h >= 8 and w >= 8:
        depth[h // 4: h // 2, w // 5: w // 2] = np.float32(od[3])
        depth[h // 2: 3 * h // 4, w // 3: 4 * w // 5] = np.float32(od[7])
    return depth


def _noise(h, w, seed, lo=0.0, hi=0.6):
    return np.random.default_rng(seed).uniform(lo, hi, (h, w, 3)).astype(np.float32)


# --------------------------------------------------------------------------
# 1. tile seams
# --------------------------------------------------------------------------

@_gpu_test
@pytest.mark.parametrize("hw", [(100, 90), (90, 100)])
@pytest.mark.parametrize("n", [40, 41])
def test_point_lights_on_tile_corners_edges_and_reflect_border(hw, n):
    """K=17, N=40 -> T=24 (N=41 -> T=25): 6x single-pixel lights exactly on
    tile corners (all four sides of the seam), on tile edges and on the frame
    border. 6x <= the goldens' linear max: a hard Tier C scene."""
    H, W = hw
    K = 17
    T = n - K + 1
    img = np.full((H, W, 3), 0.02, np.float32)
    pts = [(T - 1, T - 1), (T, T), (T - 1, T), (T, T - 1),
           (2 * T - 1, 2 * T), (2 * T, 2 * T - 1), (0, 0), (0, T), (T, 0),
           (H - 1, W - 1), (H - 1, T), (3 * T, W - 1), (min(H - 1, 4 * T), 3 * T)]
    for y, x in pts:
        img[y, x] = 6.0
    kw = _kw(img, _ramp_depth(H, W), K=K, thresh=0.5)
    ref5 = _ref(kw)
    got5 = _gpu(kw, _tile_n=n)
    assert float(ref5[1].max()) > 0.1          # the bright pass is exercised
    _assert_tier_c(ref5, got5, f"seams {hw} N={n}")


@_gpu_test
def test_single_pixel_slab_support_on_tile_corners_and_wedge_boundaries():
    """A one-pixel depth outlier (its own slab) on tile corners, tile edges,
    the frame corners and the optical centre / wedge boundaries."""
    H, W = 100, 90
    od = _od()
    img = _noise(H, W, 2, 0.0, 0.5)
    img[48:51, 30:33] = 3.0
    depth = np.full((H, W), od[5], np.float32)
    cy, cx = (H - 1) // 2, (W - 1) // 2
    for y, x in ((23, 23), (24, 24), (23, 47), (48, 24), (0, 0), (H - 1, W - 1),
                 (cy, cx), (cy + 10, cx), (cy, cx + 10), (cy - 10, cx - 10)):
        depth[y, x] = od[9]
    kw = _kw(img, depth, K=17)
    ref5 = _ref(kw)
    for n in (40, 29):
        _assert_tier_c(ref5, _gpu(kw, _tile_n=n), f"one-pixel support N={n}")


# --------------------------------------------------------------------------
# 2. kernels larger than a tile, frames smaller than a tile
# --------------------------------------------------------------------------

@_gpu_test
@pytest.mark.parametrize("K,n,hw", [(17, 18, (30, 26)), (17, 19, (26, 30)),
                                    (67, 72, (44, 38))])
def test_kernel_larger_than_output_tile(K, n, hw):
    """T = N - K + 1 is 1, 2 or 6 px: the halo dwarfs the tile; hundreds of
    tiles per chunk sequence, every wave split across many ranks."""
    H, W = hw
    img = _noise(H, W, 5)
    img[H // 2 - 1:H // 2 + 2, W // 3:W // 3 + 3] = 5.0
    img[2, W - 3] = 6.0
    kw = _kw(img, _ramp_depth(H, W), K=K)
    _assert_tier_c(_ref(kw), _gpu(kw, _tile_n=n), f"K={K} N={n} {hw}")


@_gpu_test
@pytest.mark.parametrize("hw", [(40, 3), (3, 40), (1, 1), (2, 5), (5, 2), (9, 13)])
@pytest.mark.parametrize("K", [17, 67, 183])
def test_frames_smaller_than_one_tile_and_narrower_than_the_kernel(hw, K):
    """One partial tile; the reflect pad exceeds the frame (pad 33 / 91 > 3 px).
    K=183 is the native kernel at its production N=512."""
    H, W = hw
    img = _noise(H, W, 7, 0.0, 0.5)
    img[0, 0] = 3.0
    kw = _kw(img, np.full(hw, _od()[6], np.float32), K=K)
    ref5 = _ref(kw)
    got5 = _gpu(kw)
    e = np.abs(_total(got5).astype(np.float64) - _total(ref5)) / np.maximum(_total(ref5), 1.0)
    assert np.isfinite(_total(got5)).all()
    assert float(e.max()) <= _C["max_e"], float(e.max())
    _exact_parts(ref5, got5, f"tiny {hw} K={K}")
    if tol.annulus_mask(H, W).any() and H * W >= 1000:
        _assert_tier_c(ref5, got5, f"tiny {hw} K={K}")


# --------------------------------------------------------------------------
# 3. lenses, orientations, tile size
# --------------------------------------------------------------------------

def _lights_scene(lens, hw, amp, gain, K=17):
    H, W = hw
    rng = np.random.default_rng(11)
    img = _noise(H, W, 11, 0.0, 0.8)
    for _ in range(10):
        y, x = int(rng.integers(1, H - 4)), int(rng.integers(1, W - 4))
        img[y:y + 3, x:x + 3] = amp
    return _kw(img, _ramp_depth(H, W), lens=lens, K=K, gain=gain, thresh=0.5)


@_gpu_test
@pytest.mark.parametrize("lens", ["35", "50"])
@pytest.mark.parametrize("hw", [(130, 110), (110, 130)])
@pytest.mark.parametrize("amp,gain", [(1.0, None), (6.8, 2.0), (6.8, 4.0)])
def test_both_lenses_both_orientations_non_multiple_tiles(lens, hw, amp, gain):
    """LDR peak stamp, and HDR at the goldens' linear maximum (6.8) with L1
    gain 2 and 4 (the edge of the hard range)."""
    kw = _lights_scene(lens, hw, amp, gain)
    _assert_tier_c(_ref(kw), _gpu(kw, _tile_n=43), f"lens {lens} {hw} amp {amp} g {gain}")


@_gpu_test
@pytest.mark.parametrize("lens", ["35", "50"])
@pytest.mark.parametrize("hw", [(130, 110), (110, 130)])
def test_hdr16_dense_light_field_stress_ratio(lens, hw):
    """Stress family: ten 3x3 16x lights at L1 gain 2. Ratio only (<= 7)."""
    kw = _lights_scene(lens, hw, 16.0, 2.0)
    _assert_gpu_stress(kw, _ref(kw), _gpu(kw, _tile_n=43), f"hdr16 lens {lens} {hw}")


@_gpu_test
def test_tile_size_changes_stay_within_tolerance_of_the_reference():
    """T = 2, 8, 23 and the production N: every tiling inside Tier C."""
    H, W = 60, 52
    img = _noise(H, W, 13)
    img[20:23, 30:33] = 6.0
    kw = _kw(img, _ramp_depth(H, W), K=17)
    ref5 = _ref(kw)
    for n in (18, 24, 39, None):
        _assert_tier_c(ref5, _gpu(kw, **({} if n is None else {"_tile_n": n})), f"N={n}")


# --------------------------------------------------------------------------
# 4. batch / chunk / accumulator / thread / device configurations
# --------------------------------------------------------------------------

def _mixed_scene():
    """Delta units (an in-focus d0 band), rotated wedge units, bright tiles
    and dark tiles together: ranks of mixed kind inside every wave."""
    H, W = 90, 76
    od = _od()
    img = _noise(H, W, 21, 0.0, 0.7)
    for y, x in ((10, 10), (40, 60), (70, 20), (85, 70), (5, 70)):
        img[y:y + 3, x:x + 3] = 5.0
    depth = _ramp_depth(H, W)
    depth[:, :12] = np.float32(od[0])
    depth[60:75, 40:60] = np.float32(od[0])
    return _kw(img, depth, K=17, thresh=0.5)


@_gpu_test
@pytest.mark.parametrize("pairs,chunk,host_acc", [
    (1, 1, False), (1, 1, True), (3, 2, False), (5, 7, True),
    (2, 64, False), (1000, 1000, True), (64, 16, False),
])
def test_batch_chunk_and_accumulator_configs_each_within_tier_c(pairs, chunk, host_acc):
    """Every configuration is judged against the reference (MLX irfft2 is not
    batch-invariant, so configurations are not compared bit for bit). pairs=1
    splits every rank into its own wave; chunk=1 makes one tile per chunk."""
    kw = _mixed_scene()
    ref5 = _ref(kw)
    got5 = _gpu(kw, _tile_n=40, _pairs=pairs, _tile_chunk=chunk, _host_acc=host_acc)
    _assert_tier_c(ref5, got5, f"pairs={pairs} chunk={chunk} host={host_acc}")
    again = _gpu(kw, _tile_n=40, _pairs=pairs, _tile_chunk=chunk, _host_acc=host_acc)
    for i in range(5):
        assert np.array_equal(got5[i], again[i]), f"not deterministic per config ({i})"


@_gpu_test
def test_host_vs_gpu_accumulators_are_bit_equal_for_a_fixed_wave_config():
    """Spec 2.4: host and GPU output accumulators are one float32 add of the
    same addend -> np.array_equal."""
    kw = _mixed_scene()
    a = _gpu(kw, _tile_n=40, _pairs=7, _tile_chunk=3, _host_acc=False)
    b = _gpu(kw, _tile_n=40, _pairs=7, _tile_chunk=3, _host_acc=True)
    for i in range(5):
        assert np.array_equal(a[i], b[i]), i


@_gpu_test
def test_planner_threads_and_slab_batch_size_are_bit_neutral():
    kw = _mixed_scene()
    base = _gpu(kw, _tile_n=40)
    D = kw["slab_w"].shape[0]
    for extra in ({"threads": 1}, {"threads": 8}, {"slab_batch_size": 1},
                  {"slab_batch_size": D}):
        k = dict(kw)
        k.update({key: v for key, v in extra.items() if key == "slab_batch_size"})
        got = _gpu(k, _tile_n=40, **{key: v for key, v in extra.items() if key == "threads"})
        for i in range(5):
            assert np.array_equal(base[i], got[i]), (extra, i)


@_gpu_test
def test_global_default_device_cpu_does_not_move_or_change_the_engine():
    """A caller that set MLX's default device to the CPU: the engine runs on
    its explicit GPU stream, so the output is bit-equal and the caller's
    default device is left as it was."""
    import mlx.core as mx

    kw = _mixed_scene()
    base = _gpu(kw, _tile_n=40)
    prev = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        got = _gpu(kw, _tile_n=40)
        assert mx.default_device() == mx.cpu
        assert eg._LAST_RUN.get("device") == "gpu"
    finally:
        mx.set_default_device(prev)
    for i in range(5):
        assert np.array_equal(base[i], got[i]), i


# --------------------------------------------------------------------------
# 5. extreme, negative, flat and black inputs
# --------------------------------------------------------------------------

@_gpu_test
def test_extreme_highlight_1e3_stress_ratio():
    """A 1000x light: stress family, p99.9 ratio vs float64 truth <= 7."""
    H, W = 90, 80
    img = _noise(H, W, 1, 0.0, 0.5)
    img[44:47, 38:41] = 1e3
    kw = _kw(img, _ramp_depth(H, W, patches=False), K=17)
    _assert_gpu_stress(kw, _ref(kw), _gpu(kw, _tile_n=40), "extreme amp 1000")


@_gpu_test
@pytest.mark.parametrize("amp", [1e5, 1e30])
def test_out_of_range_highlights_stay_finite_and_keep_exact_invariants(amp):
    """Out of the ruled range: finite output, exact weight planes, no NaN/Inf
    spread from the float32 FFT (1e30 * a 324^2 sum stays < float32 max).
    The ratio is printed only."""
    H, W = 90, 80
    img = _noise(H, W, 1, 0.0, 0.5)
    img[44:47, 38:41] = amp
    kw = _kw(img, _ramp_depth(H, W, patches=False), K=17)
    ref5 = _ref(kw)
    got5 = _gpu(kw, _tile_n=40)
    assert np.isfinite(ref5[0]).all() == np.isfinite(got5[0]).all()
    assert np.isfinite(got5[0]).all() and np.isfinite(got5[1]).all()
    _exact_parts(ref5, got5, f"amp {amp}")
    print(f"amp {amp} (report only): {format_ratio(stress_ratio(kw, ref5, got5))}")


@_gpu_test
def test_negative_extended_range_values():
    """Negative linear values, also inside the bright image."""
    H, W = 90, 80
    img = _noise(H, W, 4, -0.2, 0.8)
    img[30:33, 50:53] = np.array([5.0, -0.5, 4.0], np.float32)
    img[70:73, 10:13] = np.array([-0.3, 6.0, 2.0], np.float32)
    kw = _kw(img, _ramp_depth(H, W), K=17)
    assert float(kw["img_bright_padded"].min()) < 0.0      # non-vacuous
    _assert_tier_c(_ref(kw), _gpu(kw, _tile_n=40), "negative values")


@_gpu_test
@pytest.mark.parametrize("gain", [2.0, None])
def test_all_black_frame_is_exactly_zero(gain):
    H, W = 70, 60
    img = np.zeros((H, W, 3), np.float32)
    kw = _kw(img, _ramp_depth(H, W), K=17, gain=gain)
    ref5 = _ref(kw)
    got5 = _gpu(kw, _tile_n=40)
    assert not np.any(_total(ref5))
    assert not np.any(_total(got5))
    _exact_parts(ref5, got5, "black")


@_gpu_test
@pytest.mark.parametrize("val", [1.0, 5.0])
@pytest.mark.parametrize("gain", [2.0, None])
def test_flat_white_and_flat_hdr_frames(val, gain):
    """Flat frames: every tile bright (5.0) or at the threshold edge (1.0)."""
    H, W = 70, 60
    img = np.full((H, W, 3), val, np.float32)
    kw = _kw(img, _ramp_depth(H, W), K=17, gain=gain)
    got5 = _gpu(kw, _tile_n=40)
    _assert_tier_c(_ref(kw), got5, f"flat {val} gain {gain}")


# --------------------------------------------------------------------------
# 6. d0 delta slab, empty / sub-threshold slabs, whole-tile holes
# --------------------------------------------------------------------------

@_gpu_test
@pytest.mark.parametrize("K", [17, 67])
def test_d0_delta_slab_only_is_identity(K):
    H, W = 80, 70
    img = _noise(H, W, 9)
    img[30:33, 30:33] = 4.0
    kw = _kw(img, np.full((H, W), _od()[0], np.float32), K=K)
    ref5 = _ref(kw)
    got5 = _gpu(kw, _tile_n=K + 30)
    _assert_tier_c(ref5, got5, f"d0 K={K}")
    e = np.abs(_total(got5).astype(np.float64) - img) / np.maximum(img, 1.0)
    assert float(e.max()) <= _C["max_e"]


@_gpu_test
def test_empty_and_sub_threshold_slabs():
    H, W = 80, 70
    D = _tensor("35")["kernels"].shape[1]
    img = _noise(H, W, 6)
    img[40:43, 20:23] = 5.0
    sw = np.zeros((D, H, W), np.float32)
    sw[3] = 0.005
    sw[4, :20, :20] = np.float32(cv._EMPTY_SKIP_THRESH)
    sw[6] = 1.0 - sw[3] - sw[4]
    kw = _kw(img, _ramp_depth(H, W), K=17, slab_w=sw)
    ref5 = _ref(kw)
    for n in (40, None):
        _assert_tier_c(ref5, _gpu(kw, **({} if n is None else {"_tile_n": n})),
                       f"empty slabs N={n}")


@_gpu_test
@pytest.mark.parametrize("n", [40, 30])
def test_whole_tile_holes_with_zero_weight_in_every_slab(n):
    """Holes (acc_alpha < 0.02) covering WHOLE tiles whose slab weight is
    exactly 0 in every slab while those slabs are active elsewhere: the gpu
    engine skips such tiles (``tile_has_sw``); donors / fallback must still
    match the reference. Plus a sub-0.02 sliver and a one-pixel hole."""
    H, W = 90, 80
    D = _tensor("35")["kernels"].shape[1]
    img = _noise(H, W, 8, 0.0, 0.7)
    img[50:53, 50:53] = 5.0
    img[5:8, 5:8] = 5.0                       # a light inside the hole
    sw = np.zeros((D, H, W), np.float32)
    sw[2] = 0.6
    sw[8] = 0.4
    sw[:, 0:50, 0:50] = 0.0                   # >= 2 whole tiles at T <= 24
    sw[8, 60:70, 0:30] = 0.015                # alpha 0.015 < 0.02 (a sliver)
    sw[2, 60:70, 0:30] = 0.0
    sw[:, 80, 75] = 0.0                       # one-pixel hole
    kw = _kw(img, _ramp_depth(H, W), K=17, slab_w=sw, thresh=0.5)
    from summicron.render import _tile_schedule as ts

    assert ts.predict_holes(sw, cv._EMPTY_SKIP_THRESH)     # non-vacuous
    ref5 = _ref(kw)
    _assert_tier_c(ref5, _gpu(kw, _tile_n=n), f"whole-tile holes N={n}")


# --------------------------------------------------------------------------
# 7. polarity: convolution, not correlation, on asymmetric kernels
# --------------------------------------------------------------------------

def _asym_tensor(k, delta):
    kk = np.zeros((k, k), np.float32)
    if delta:
        kk[k // 2 - 3, k // 2 + 2] = 0.7
        kk[k // 2 - 3, k // 2 + 3:k - 1] = 0.3 / (k - 1 - (k // 2 + 3))
    else:
        kk[2:6, 1:4] = 1.0
        kk[k // 2, k // 2] = 0.3
        kk[k - 3, k - 8:k - 1] = 0.7
    kk /= kk.sum()
    t = _tensor("35")
    kern = np.broadcast_to(kk, (12, 11, k, k)).copy()
    return dict(kernels=kern, radial_edges=t["radial_edges"],
                object_distances_m=t["object_distances_m"],
                defocus_ratios=t["defocus_ratios"])


@_gpu_test
@pytest.mark.parametrize("k", [21, 20])
@pytest.mark.parametrize("delta", [False, True])
def test_asymmetric_kernel_polarity_odd_and_even_k(k, delta):
    """Odd and even K, rotated-wedge and is_delta branches. A flipped
    (correlation) kernel is far outside Tier C, so this is not vacuous."""
    H, W = 90, 80
    img = np.full((H, W, 3), 0.05, np.float32)
    for y, x in ((20, 20), (45, 40), (70, 60), (5, 75), (85, 3)):
        img[y, x] = 4.0
    tens = _asym_tensor(k, delta)
    kb = cv._build_basis_kernel(0, 5, True, tens["defocus_ratios"], tens["kernels"],
                                k, k, 1.0, k)
    assert bool(kb.max() > 0.5 * kb.sum()) == delta
    kw = _kw(img, _ramp_depth(H, W), K=k, tensor=tens)
    ref5 = _ref(kw)
    for n in (k + 19, None):
        _assert_tier_c(ref5, _gpu(kw, **({} if n is None else {"_tile_n": n})),
                       f"asym k={k} delta={delta} N={n}")
    flipped = dict(tens)
    flipped["kernels"] = tens["kernels"][:, :, ::-1, ::-1].copy()
    got_flip = _gpu(_kw(img, _ramp_depth(H, W), K=k, tensor=flipped), _tile_n=k + 19)
    res = tol.evaluate(_total(ref5), _total(got_flip), "C")
    assert not res["ok"] and res["full"]["max_e"][0] > 1e-2


# --------------------------------------------------------------------------
# 8. through render(): both lenses, portrait and landscape
# --------------------------------------------------------------------------

@_gpu_test
@pytest.mark.parametrize("lens", ["35", "50"])
@pytest.mark.parametrize("hw", [(150, 118), (118, 150)])
def test_render_level_gpu_vs_reference_linear_tier_c(lens, hw):
    """render() with a real tensor at K ~ 23 (corpus_ref_long_edge 1200): the
    returned linear output of --engine gpu is within Tier C of the reference,
    and the stdout lines are identical."""
    t = dict(_tensor(lens))
    t["corpus_ref_long_edge"] = 1200
    od = np.asarray(t["object_distances_m"])
    H, W = hw
    rng = np.random.default_rng(31)
    img = rng.uniform(0.0, 0.6, (H, W, 3)).astype(np.float32)
    for _ in range(8):
        y, x = int(rng.integers(2, H - 4)), int(rng.integers(2, W - 4))
        img[y:y + 3, x:x + 3] = rng.uniform(2.0, 6.0)
    img[0:3, 0:3] = 5.0
    img[H - 3:H, W - 3:W] = 5.0
    ramp = np.geomspace(od[0], od[-1], W, dtype=np.float32)
    depth = np.broadcast_to(ramp[None], (H, W)).copy()

    def run(engine):
        lines = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = cv.render(img, t, depth_m=depth, hdr_source="macos", bright_thresh=1.0,
                            bright_stamp_gain=2.0, on_progress=lines.append, engine=engine)
        return out, lines

    (u8r, linr), lr = run("reference")
    (u8g, ling), lg = run("gpu")
    res = tol.evaluate(linr, ling, "C")
    assert res["ok"], f"render lens {lens} {hw}: {tol.summary(res)}"
    assert int(np.abs(u8r.astype(np.int16) - u8g).max()) <= 1
    strip = [ln for ln in lr if "bright_energy_ratio=" not in ln]
    assert strip == [ln for ln in lg if "bright_energy_ratio=" not in ln]


# --------------------------------------------------------------------------
# 9. the switch: unavailable engine, mid-render failure, no fallback
# --------------------------------------------------------------------------

def _spy_gathers(monkeypatch):
    calls = {"ref_gather": 0, "apply": 0, "tiled": 0, "gpu": 0}

    def wrap(mod, name, key):
        orig = getattr(mod, name)

        def w(*a, **k):
            calls[key] += 1
            return orig(*a, **k)
        monkeypatch.setattr(mod, name, w)

    wrap(cv, "_layered_gather_diffuse", "ref_gather")
    wrap(cv, "_apply_kernel", "apply")
    wrap(et, "tiled_layered_gather", "tiled")
    wrap(eg, "gpu_layered_gather", "gpu")
    return calls


def _synthetic(k=17):
    from tests.fixtures.synthetic_tensor import make_synthetic_tensor

    return make_synthetic_tensor(12, 11, k, k, corpus_ref_long_edge=40)


def _small_scene(t):
    img = _noise(40, 40, 3, 0.0, 0.4)
    img[10:13, 20:23] = 0.95                 # sRGB input (hdr_source none)
    depth = np.full((40, 40), float(np.asarray(t["object_distances_m"])[6]), np.float32)
    return img, depth


@pytest.mark.parametrize("occ", ["off", "on"])
def test_mlx_absent_is_one_cli_line_and_runs_no_engine(monkeypatch, capsys, occ):
    """mlx not importable: strict refusal before the ON rule, through the CLI
    contract (one 'render:' line, exit 2); no gather of any engine runs."""
    monkeypatch.setitem(sys.modules, "mlx", None)
    monkeypatch.setitem(sys.modules, "mlx.core", None)
    monkeypatch.delenv("ISUMMICRON_DEBUG", raising=False)
    calls = _spy_gathers(monkeypatch)
    t = _synthetic()
    img, depth = _small_scene(t)
    rc = cv.run_cli_render(lambda: cv.render(img, t, depth_m=depth, engine="gpu",
                                             occlusion_compositing=occ))
    err = [ln for ln in capsys.readouterr().err.splitlines() if ln.strip()]
    assert rc == 2
    assert len(err) == 1 and err[0].startswith("render: ") and "not available" in err[0], err
    assert "mlx is not importable" in err[0]
    assert calls == {"ref_gather": 0, "apply": 0, "tiled": 0, "gpu": 0}


@_gpu_test
def test_metal_probe_raising_is_not_available(monkeypatch):
    import mlx.core as mx

    def boom():
        raise RuntimeError("metal init failed")
    monkeypatch.setattr(mx.metal, "is_available", boom)
    assert eg.availability() == "no Metal GPU device"
    calls = _spy_gathers(monkeypatch)
    t = _synthetic()
    img, depth = _small_scene(t)
    with pytest.raises(ValueError, match="engine 'gpu' is not available: no Metal GPU device"):
        cv.render(img, t, depth_m=depth, engine="gpu")
    assert calls == {"ref_gather": 0, "apply": 0, "tiled": 0, "gpu": 0}


@_gpu_test
def test_mid_render_mlx_runtime_error_propagates_without_fallback(monkeypatch, capsys):
    """Spec 3.2: a mid-render MLX RuntimeError is not 'unavailable'; it leaves
    as one 'render:' line, exit 2, and no CPU engine silently takes over."""
    calls = _spy_gathers(monkeypatch)

    def boom(*a, **k):
        raise RuntimeError("[metal::malloc] simulated allocation failure")
    monkeypatch.setattr(eg, "_run_waves", boom)
    monkeypatch.delenv("ISUMMICRON_DEBUG", raising=False)
    t = _synthetic()
    img, depth = _small_scene(t)
    with pytest.raises(RuntimeError, match="simulated allocation failure"):
        cv.render(img, t, depth_m=depth, engine="gpu", bright_stamp_gain=2.0,
                  bright_thresh=0.6)
    rc = cv.run_cli_render(lambda: cv.render(img, t, depth_m=depth, engine="gpu",
                                             bright_stamp_gain=2.0, bright_thresh=0.6))
    err = [ln for ln in capsys.readouterr().err.splitlines() if ln.strip()]
    assert rc == 2 and len(err) == 1 and err[0].startswith("render: "), err
    assert calls["ref_gather"] == 0 and calls["apply"] == 0 and calls["tiled"] == 0
    assert calls["gpu"] == 2


@_gpu_test
def test_gpu_on_path_is_reference_bit_identical_with_one_note():
    """--engine gpu + ON: reference engine, bit-identical, exactly one note."""
    from summicron.render._resolve import ENGINE_ON_NOTE_GPU

    t = _synthetic()
    img, depth = _small_scene(t)
    depth[:, :20] = float(np.asarray(t["object_distances_m"])[2])

    def run(engine):
        lines = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = cv.render(img, t, depth_m=depth, engine=engine, bright_stamp_gain=2.0,
                            bright_thresh=0.6,
                            occlusion_compositing="on", on_progress=lines.append)
        return out, lines

    (u8r, linr), lr = run("reference")
    (u8g, ling), lg = run("gpu")
    assert np.array_equal(linr, ling) and np.array_equal(u8r, u8g)
    assert [ln for ln in lg if ln not in lr] == [ENGINE_ON_NOTE_GPU]


# --------------------------------------------------------------------------
# 10. the float32 kernel spectra ruling (user ruling 2026-11-14)
# --------------------------------------------------------------------------

@_gpu_test
def test_gpu_kernel_spectra_are_not_the_cpu_float64_path(monkeypatch):
    calls = {"n": 0}
    orig = et._kernel_spectra

    def spy(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)
    monkeypatch.setattr(et, "_kernel_spectra", spy)
    kw = _mixed_scene()
    _gpu(kw, _tile_n=40)
    assert calls["n"] == 0, f"CPU float64 _kernel_spectra called {calls['n']}x"


# --------------------------------------------------------------------------
# 11. two gathers in one process (app use: a new render while one runs)
# --------------------------------------------------------------------------

def _paused_gather(kw, started, release, out, key):
    def prog(msg):
        if msg.startswith("  Slab 1/"):
            started.set()
            assert release.wait(60)
    k = dict(kw)
    k["fft_state"] = None
    k["on_progress"] = prog
    k["emit_log"] = False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out[key] = eg.gpu_layered_gather(_tile_n=40, **k)


@_gpu_test
def test_overlapping_gathers_are_bit_equal_to_serial_runs():
    """Two gathers interleaved on two threads (A paused mid-render while B
    runs): each output is bit-equal to its own serial run (same config)."""
    import threading

    kw_a = _mixed_scene()
    kw_b = _lights_scene("50", (110, 130), 6.8, 2.0)
    ser_a = _gpu(kw_a, _tile_n=40)
    ser_b = _gpu(kw_b, _tile_n=40)
    out = {}
    sa, ra, sb, rb = (threading.Event() for _ in range(4))
    ta = threading.Thread(target=_paused_gather, args=(kw_a, sa, ra, out, "a"))
    tb = threading.Thread(target=_paused_gather, args=(kw_b, sb, rb, out, "b"))
    ta.start()
    assert sa.wait(60)
    tb.start()
    assert sb.wait(60)
    ra.set()
    ta.join(120)
    rb.set()
    tb.join(120)
    for key, ser in (("a", ser_a), ("b", ser_b)):
        for i in range(5):
            assert np.array_equal(ser[i], out[key][i]), (key, i)


@_gpu_test
def test_overlapping_gathers_restore_the_callers_cache_limit():
    import threading

    import mlx.core as mx

    orig = mx.set_cache_limit(1 << 33)          # a caller's own limit (8 GiB)
    try:
        kw_a = _mixed_scene()
        kw_b = _lights_scene("50", (110, 130), 6.8, 2.0)
        out = {}
        sa, ra, sb, rb = (threading.Event() for _ in range(4))
        ta = threading.Thread(target=_paused_gather, args=(kw_a, sa, ra, out, "a"))
        tb = threading.Thread(target=_paused_gather, args=(kw_b, sb, rb, out, "b"))
        ta.start()
        assert sa.wait(60)
        tb.start()
        assert sb.wait(60)
        ra.set()                                  # A restores first ...
        ta.join(120)
        rb.set()                                  # ... then B restores A's limit
        tb.join(120)
        assert set(out) == {"a", "b"}
        after = mx.set_cache_limit(orig)
        assert after == (1 << 33), f"cache limit left at {after}, caller had {1 << 33}"
    finally:
        mx.set_cache_limit(orig)
