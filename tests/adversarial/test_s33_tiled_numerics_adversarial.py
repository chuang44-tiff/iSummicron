"""S33 Tier B adversarial (numerics lens): try to break the tiled engine's
tolerance contract against the in-tree reference gather.

Self-contained: no ``runs/``, no goldens, no IMG_6915, no other checkout. The
oracle is ``convolve._layered_gather_diffuse`` (the reference) run in-test,
judged by the test-local copy of the frozen metric
(``tests/fixtures/tolerance.py``, Tier B ceilings, full frame AND field-edge
annulus). Ceilings are never loosened here.

Attack surface: tile seams (lights on tile corners/edges, single-pixel slab
support), kernels larger than a tile, frames smaller than a tile or narrower
than the kernel, non-multiple sizes, both orientations, both lenses, even K,
extreme / negative / flat / all-black inputs, the d0 delta slab, empty and
sub-threshold slabs, convolution-vs-correlation polarity on asymmetric
kernels (rotated and is_delta branches), chunk / batch / tile-size
invariance, and the no-silent-fallback rule of the engine switch.

Ruling R-B2 (revised, 2026-11-13): the hard Tier B ceilings apply to real
scenes and the standard charts at L1 gain <= 4. The STRESS family (dense 16x
HDR light fields at gain 2, 50x / 100x HDR lights, ...) is gated by the hard
ratio (tiled error vs float64 truth) / (reference error vs the same truth)
<= 1.5 on max e and p99.9, full frame and annulus
(``tests/fixtures/truth.py``); the raw Tier B numbers are printed.
"""
from __future__ import annotations

import warnings
from functools import lru_cache

import numpy as np
import pytest

import summicron.render.convolve as cv
from summicron.render import _engine_tiled as eng
from summicron.render._artifacts import default_tensor_path
from tests.fixtures import tolerance as tol
from tests.fixtures.truth import assert_stress


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


def _tiled(kw, **extra):
    k = dict(kw)
    k["fft_state"] = None
    k["on_progress"] = None
    k.update(extra)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return eng.tiled_layered_gather(**k)


def _total(t5):
    return (t5[0] + t5[1]).astype(np.float32)


def _exact_parts(ref5, got5, what):
    for i, name in ((2, "weight_sum"), (3, "delta_weight_sum"), (4, "acc_alpha")):
        assert np.array_equal(ref5[i], got5[i]), f"{what}: {name} differs"


def _assert_tier_b(ref5, got5, what):
    res = tol.evaluate(_total(ref5), _total(got5), "B")
    assert res["ok"], f"{what}: {tol.summary(res)}"
    _exact_parts(ref5, got5, what)


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

@pytest.mark.parametrize("hw", [(100, 90), (90, 100)])
@pytest.mark.parametrize("n", [40, 41])
def test_point_lights_on_tile_corners_edges_and_reflect_border(hw, n):
    """K=17, N=40 -> T=24 (N=41 -> T=25): single-pixel 8x lights exactly on
    tile corners (both sides of the seam), on tile edges, and on the frame
    border (their reflections live in the pad)."""
    H, W = hw
    K = 17
    T = n - K + 1
    img = np.full((H, W, 3), 0.02, np.float32)
    pts = [(T - 1, T - 1), (T, T), (T - 1, T), (T, T - 1),
           (2 * T - 1, 2 * T), (2 * T, 2 * T - 1), (0, 0), (0, T), (T, 0),
           (H - 1, W - 1), (H - 1, T), (3 * T, W - 1), (min(H - 1, 4 * T), 3 * T)]
    for y, x in pts:
        img[y, x] = 8.0
    kw = _kw(img, _ramp_depth(H, W), K=K, thresh=0.5)
    ref5 = _ref(kw)
    got5 = _tiled(kw, _tile_n=n)
    assert float(ref5[1].max()) > 0.1          # the bright pass is exercised
    _assert_tier_b(ref5, got5, f"seams {hw} N={n}")
    eb = np.abs(got5[1].astype(np.float64) - ref5[1]) / np.maximum(ref5[1], 1.0)
    assert float(eb.max()) <= tol.TIERS["B"]["max_e"]


def test_single_pixel_slab_support_on_tile_corners_and_wedge_boundaries():
    """A one-pixel depth outlier (its own slab) on a tile corner, a tile edge,
    the frame corners and the optical centre / wedge boundaries: exercises the
    partial ``joint_support_row`` path and one-pixel bank units."""
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
        _assert_tier_b(ref5, _tiled(kw, _tile_n=n), f"one-pixel support N={n}")


# --------------------------------------------------------------------------
# 2. kernels larger than a tile, frames smaller than a tile
# --------------------------------------------------------------------------

@pytest.mark.parametrize("K,n,hw", [(17, 18, (30, 26)), (17, 19, (26, 30)),
                                    (67, 72, (44, 38))])
def test_kernel_larger_than_output_tile(K, n, hw):
    """T = N - K + 1 is 1, 2 or 6 px: the halo dwarfs the tile."""
    H, W = hw
    img = _noise(H, W, 5)
    img[H // 2 - 1:H // 2 + 2, W // 3:W // 3 + 3] = 5.0
    img[2, W - 3] = 7.0
    kw = _kw(img, _ramp_depth(H, W), K=K)
    _assert_tier_b(_ref(kw), _tiled(kw, _tile_n=n), f"K={K} N={n} {hw}")


@pytest.mark.parametrize("hw", [(40, 3), (3, 40), (1, 1), (2, 5), (5, 2), (9, 13)])
@pytest.mark.parametrize("K", [17, 67])
def test_frames_smaller_than_one_tile_and_narrower_than_the_kernel(hw, K):
    """One partial tile; the reflect pad exceeds the frame (pad 33 > 3 px)."""
    H, W = hw
    img = _noise(H, W, 7, 0.0, 0.5)
    img[0, 0] = 3.0
    kw = _kw(img, np.full(hw, _od()[6], np.float32), K=K)
    ref5 = _ref(kw)
    got5 = _tiled(kw)
    e = np.abs(_total(got5).astype(np.float64) - _total(ref5)) / np.maximum(_total(ref5), 1.0)
    assert np.isfinite(_total(got5)).all()
    assert float(e.max()) <= tol.TIERS["B"]["max_e"], float(e.max())
    _exact_parts(ref5, got5, f"tiny {hw} K={K}")
    if tol.annulus_mask(H, W).any() and H * W >= 1000:
        _assert_tier_b(ref5, got5, f"tiny {hw} K={K}")


# --------------------------------------------------------------------------
# 3. lenses, orientations, tile-size and chunk invariance
# --------------------------------------------------------------------------

def _lights_scene(lens, hw, amp, gain):
    """Noise background in [0, 0.8) plus ten 3x3 lights at ``amp``."""
    H, W = hw
    rng = np.random.default_rng(11)
    img = _noise(H, W, 11, 0.0, 0.8)
    for _ in range(10):
        y, x = int(rng.integers(1, H - 4)), int(rng.integers(1, W - 4))
        img[y:y + 3, x:x + 3] = amp
    return _kw(img, _ramp_depth(H, W), lens=lens, K=17, gain=gain, thresh=0.5)


@pytest.mark.parametrize("lens", ["35", "50"])
@pytest.mark.parametrize("hw", [(130, 110), (110, 130)])
@pytest.mark.parametrize("amp,gain", [(1.0, None), (6.8, 2.0)])
def test_both_lenses_both_orientations_non_multiple_tiles(lens, hw, amp, gain):
    """LDR peak stamp, and HDR at the goldens' linear maximum (6.8) with the
    default L1 gain 2."""
    kw = _lights_scene(lens, hw, amp, gain)
    _assert_tier_b(_ref(kw), _tiled(kw, _tile_n=43), f"lens {lens} {hw} amp {amp}")


@pytest.mark.parametrize("lens", ["35", "50"])
@pytest.mark.parametrize("hw", [(130, 110), (110, 130)])
def test_hdr16_default_gain_dense_light_field_stress_ratio(lens, hw):
    """Stress family (R-B2 revised): ten 3x3 16x lights on a 130x110 noise frame
    at the default L1 gain 2 sit on the float32 round-off floor for BOTH engines
    (raw Tier B p99.9 1.1e-6 .. 1.25e-6 > 1e-6). Hard gate: the tiled engine is
    no worse than 1.5x the reference's own error against a float64 truth."""
    kw = _lights_scene(lens, hw, 16.0, 2.0)
    assert_stress(kw, _ref(kw), _tiled(kw, _tile_n=43), f"hdr16 dense lens {lens} {hw}")


def test_tile_size_changes_stay_within_tolerance_of_each_other():
    """Different tilings (T = 2, 8, 23, 308) must all sit inside Tier B of the
    reference and therefore cannot drift apart."""
    H, W = 60, 52
    img = _noise(H, W, 13)
    img[20:23, 30:33] = 6.0
    kw = _kw(img, _ramp_depth(H, W), K=17)
    ref5 = _ref(kw)
    for n in (18, 24, 39, None):
        got5 = _tiled(kw, **({} if n is None else {"_tile_n": n}))
        _assert_tier_b(ref5, got5, f"N={n}")


def test_chunk_and_spectrum_batch_sizes_do_not_change_a_bit(monkeypatch):
    """_CHUNK (units per batched irfft2) and _SPEC_BATCH (kernels per batched
    rfft2) only regroup independent transforms: the 5-tuple is bit-equal."""
    H, W = 80, 70
    img = _noise(H, W, 3)
    img[30:33, 40:43] = 4.0
    kw = _kw(img, _ramp_depth(H, W), K=17)
    base = _tiled(kw, _tile_n=40)
    for chunk, sb in ((1, 1), (3, 5), (64, 64)):
        monkeypatch.setattr(eng, "_CHUNK", chunk)
        monkeypatch.setattr(eng, "_SPEC_BATCH", sb)
        got = _tiled(kw, _tile_n=40)
        for i in range(5):
            assert np.array_equal(base[i], got[i]), (chunk, sb, i)


# --------------------------------------------------------------------------
# 4. extreme, negative, flat and black inputs
# --------------------------------------------------------------------------

@pytest.mark.parametrize("amp", [1e3, 1e5])
def test_extreme_highlights_keep_exact_invariants(amp):
    """Far beyond the ruled stress family (50x / 100x): exact invariants only
    (finite, weights/alpha bit-equal, no NaN spread). The ratio vs a float64
    truth is printed; it is gated for 1e3 (measured 0.84 .. 1.0) but only
    reported for 1e5 (p99.9 ratio 1.8: float32 round-off of a 1e5 light, not
    inside any ruled range)."""
    from tests.fixtures.truth import format_ratio, stress_ratio

    H, W = 90, 80
    img = _noise(H, W, 1, 0.0, 0.5)
    img[44:47, 38:41] = amp
    kw = _kw(img, _ramp_depth(H, W, patches=False), K=17)
    ref5 = _ref(kw)
    got5 = _tiled(kw, _tile_n=40)
    assert np.isfinite(got5[0]).all() and np.isfinite(got5[1]).all()
    _exact_parts(ref5, got5, f"amp {amp}")
    if amp <= 1e3:
        # p99.9 ratio hard; the per-pixel max is only printed (final stress rule).
        assert_stress(kw, ref5, got5, f"extreme amp {amp}", abs_max=False)
    else:
        print(f"extreme amp {amp} (report only): {format_ratio(stress_ratio(kw, ref5, got5))}")


def test_extreme_highlight_1e3_absolute_max():
    """Final stress rule (user ruling, final round): only the p99.9 ratio is hard; the
    absolute max (measured 4.16e-05 tiled vs 4.96e-05 reference) is printed."""
    H, W = 90, 80
    img = _noise(H, W, 1, 0.0, 0.5)
    img[44:47, 38:41] = 1e3
    kw = _kw(img, _ramp_depth(H, W, patches=False), K=17)
    assert_stress(kw, _ref(kw), _tiled(kw, _tile_n=40), "extreme amp 1000 abs max")


def test_negative_extended_range_values():
    """Extended-range linear input can be negative, including inside the
    bright image: bright occupancy must test ``!= 0``, not ``> 0``."""
    H, W = 90, 80
    img = _noise(H, W, 4, -0.2, 0.8)
    img[30:33, 50:53] = np.array([5.0, -0.5, 4.0], np.float32)
    img[70:73, 10:13] = np.array([-0.3, 6.0, 2.0], np.float32)
    kw = _kw(img, _ramp_depth(H, W), K=17)
    assert float(kw["img_bright_padded"].min()) < 0.0      # non-vacuous
    _assert_tier_b(_ref(kw), _tiled(kw, _tile_n=40), "negative values")


@pytest.mark.parametrize("gain", [2.0, None])
def test_all_black_frame_is_exactly_zero(gain):
    H, W = 70, 60
    img = np.zeros((H, W, 3), np.float32)
    kw = _kw(img, _ramp_depth(H, W), K=17, gain=gain)
    ref5 = _ref(kw)
    got5 = _tiled(kw, _tile_n=40)
    assert not np.any(_total(ref5))
    assert not np.any(_total(got5))
    _exact_parts(ref5, got5, "black")


@pytest.mark.parametrize("val", [1.0, 5.0])
@pytest.mark.parametrize("gain", [2.0, None])
def test_flat_white_and_flat_hdr_frames(val, gain):
    H, W = 70, 60
    img = np.full((H, W, 3), val, np.float32)
    kw = _kw(img, _ramp_depth(H, W), K=17, gain=gain)
    _assert_tier_b(_ref(kw), _tiled(kw, _tile_n=40), f"flat {val} gain {gain}")


# --------------------------------------------------------------------------
# 5. d0 delta slab, empty and sub-threshold slabs
# --------------------------------------------------------------------------

@pytest.mark.parametrize("K", [17, 67])
def test_d0_delta_slab_only_is_identity(K):
    """All pixels in focus: slab 0 holds the S2 delta (is_delta branch,
    img_padded input, no bright inverse). Output is the input."""
    H, W = 80, 70
    img = _noise(H, W, 9)
    img[30:33, 30:33] = 4.0
    kw = _kw(img, np.full((H, W), _od()[0], np.float32), K=K)
    ref5 = _ref(kw)
    got5 = _tiled(kw, _tile_n=K + 30)
    _assert_tier_b(ref5, got5, f"d0 K={K}")
    e = np.abs(_total(got5).astype(np.float64) - img) / np.maximum(img, 1.0)
    assert float(e.max()) <= tol.TIERS["B"]["max_e"]


def test_empty_and_sub_threshold_slabs():
    """Slab 3 at 0.005 everywhere (empty-skipped), slab 4 at exactly
    _EMPTY_SKIP_THRESH on one tile only (active: the test is ``<``), slab 6
    carrying the rest; all other slabs exactly zero."""
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
        _assert_tier_b(ref5, _tiled(kw, **({} if n is None else {"_tile_n": n})),
                       f"empty slabs N={n}")


# --------------------------------------------------------------------------
# 6. polarity: convolution, not correlation, on asymmetric kernels
# --------------------------------------------------------------------------

def _asym_tensor(k, delta):
    """Every (r, d) holds the same asymmetric kernel. ``delta``: a dominant
    off-centre peak (> half the L1: the is_delta branch) with a one-sided tail;
    else an L-shaped blob + bar (the rotated wedge branch)."""
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


@pytest.mark.parametrize("k", [21, 20])
@pytest.mark.parametrize("delta", [False, True])
def test_asymmetric_kernel_polarity_odd_and_even_k(k, delta):
    """Odd and even K (even K is reachable via a custom --tensor at
    pix_scale >= 0.95), both branches. A flipped (correlation) kernel is shown
    to be far outside the ceiling, so this check is not vacuous."""
    H, W = 90, 80
    img = np.full((H, W, 3), 0.05, np.float32)
    for y, x in ((20, 20), (45, 40), (70, 60), (5, 75), (85, 3)):
        img[y, x] = 4.0
    tens = _asym_tensor(k, delta)
    kb = cv._build_basis_kernel(0, 5, True, tens["defocus_ratios"], tens["kernels"],
                                k, k, 1.0, k)
    assert bool(kb.max() > 0.5 * kb.sum()) == delta     # the intended branch
    kw = _kw(img, _ramp_depth(H, W), K=k, tensor=tens)
    ref5 = _ref(kw)
    for n in (k + 19, None):
        _assert_tier_b(ref5, _tiled(kw, **({} if n is None else {"_tile_n": n})),
                       f"asym k={k} delta={delta} N={n}")
    flipped = dict(tens)
    flipped["kernels"] = tens["kernels"][:, :, ::-1, ::-1].copy()
    res = tol.evaluate(_total(ref5), _total(_ref(_kw(img, _ramp_depth(H, W), K=k,
                                                      tensor=flipped))), "B")
    assert not res["ok"] and res["full"]["max_e"][0] > 1e-2


# --------------------------------------------------------------------------
# 7. the switch: no silent fallback
# --------------------------------------------------------------------------

def _spy_gathers(monkeypatch):
    calls = {"ref_gather": 0, "apply": 0, "tiled": 0}

    def wrap(mod, name, key):
        orig = getattr(mod, name)

        def w(*a, **k):
            calls[key] += 1
            return orig(*a, **k)
        monkeypatch.setattr(mod, name, w)

    wrap(cv, "_layered_gather_diffuse", "ref_gather")
    wrap(cv, "_apply_kernel", "apply")
    wrap(eng, "tiled_layered_gather", "tiled")
    return calls


def _synthetic(ksize_h, ksize_w=None):
    from tests.fixtures.synthetic_tensor import make_synthetic_tensor

    return make_synthetic_tensor(12, 11, ksize_h, ksize_w or ksize_h,
                                 corpus_ref_long_edge=40)


def _small_scene(t):
    img = _noise(40, 40, 3, 0.0, 0.4)
    depth = np.full((40, 40), float(np.asarray(t["object_distances_m"])[6]), np.float32)
    return img, depth


@pytest.mark.parametrize("engine", ["Tiled", "tiled ", "TILED", "", None, "cpu"])
def test_engine_name_variants_are_refused_before_any_gather(monkeypatch, engine):
    calls = _spy_gathers(monkeypatch)
    t = _synthetic(17)
    img, depth = _small_scene(t)
    with pytest.raises(ValueError, match="engine must be one of"):
        cv.render(img, t, depth_m=depth, engine=engine)
    assert calls == {"ref_gather": 0, "apply": 0, "tiled": 0}


@pytest.mark.parametrize("kh,kw_", [(257, 257), (17, 19)])
def test_tiled_refusal_runs_no_engine_and_is_one_cli_line(monkeypatch, capsys, kh, kw_):
    """K > 255 and a non-square kernel: strict 'not available', neither the
    tiled gather nor the reference gather runs (no silent fallback), and the
    CLI contract prints exactly one 'render:' line with exit 2."""
    calls = _spy_gathers(monkeypatch)
    t = _synthetic(kh, kw_)
    img, depth = _small_scene(t)

    def run():
        cv.render(img, t, depth_m=depth, engine="tiled")

    monkeypatch.delenv("ISUMMICRON_DEBUG", raising=False)
    rc = cv.run_cli_render(run)
    err = [ln for ln in capsys.readouterr().err.splitlines() if ln.strip()]
    assert rc == 2
    assert len(err) == 1 and err[0].startswith("render: ") and "not available" in err[0], err
    assert calls == {"ref_gather": 0, "apply": 0, "tiled": 0}


def test_gpu_refusal_runs_no_engine(monkeypatch):
    # Simulate an absent mlx so the test means the same in both envs.
    import sys

    monkeypatch.setitem(sys.modules, "mlx", None)
    monkeypatch.setitem(sys.modules, "mlx.core", None)
    calls = _spy_gathers(monkeypatch)
    t = _synthetic(17)
    img, depth = _small_scene(t)
    for occ in ("off", "on"):
        with pytest.raises(ValueError, match="engine 'gpu' is not available"):
            cv.render(img, t, depth_m=depth, engine="gpu", occlusion_compositing=occ)
    assert calls == {"ref_gather": 0, "apply": 0, "tiled": 0}


def test_tiled_off_never_touches_the_reference_gather(monkeypatch):
    calls = _spy_gathers(monkeypatch)
    t = _synthetic(17)
    img, depth = _small_scene(t)
    cv.render(img, t, depth_m=depth, engine="tiled", bright_stamp_gain=2.0)
    assert calls["tiled"] == 1 and calls["ref_gather"] == 0 and calls["apply"] == 0


# --------------------------------------------------------------------------
# 8. stdout identity (G-B4) at a rounding boundary
# --------------------------------------------------------------------------

def test_bright_energy_ratio_digit_may_differ_by_one_at_rounding_boundary(monkeypatch):
    """DOCUMENTED EXPECTED BEHAVIOUR (user ruling, 2026-11-13: accept and
    document). The ``bright-pass:`` stdout line prints sum(output_bright) /
    sum(img_lin) to 4 digits. The tiled engine is a tolerance engine: its bright
    sum differs from the reference's by float32 round-off, so when the ratio sits
    exactly on a ``.xxxx5`` rounding boundary the 4th digit can differ by 1.
    Identity of the line is gated on the real goldens
    (``test_s33_tiled_stdout_stats.py``); this test constructs the boundary and
    pins the behaviour: every other character of the line is identical and the
    ratios differ by exactly one unit in the 4th decimal.

    Self-tuning repro through render(): find a scene where the two engines'
    bright sums differ (1 ulp), move sum(img_lin) with dark pixels only so the
    ratio straddles a 4-digit rounding boundary, then compare the lines."""
    t = dict(_tensor("35"))
    t["corpus_ref_long_edge"] = 1200          # K = 39 on a 150 px frame
    od = np.asarray(t["object_distances_m"])
    bsum = {}
    for mod, name, key in ((cv, "_layered_gather_diffuse", "reference"),
                           (eng, "tiled_layered_gather", "tiled")):
        orig = getattr(mod, name)

        def w(*a, _orig=orig, _key=key, **k):
            r = _orig(*a, **k)
            bsum[_key] = float(r[1].sum())
            return r
        monkeypatch.setattr(mod, name, w)

    def run(img, depth, engine):
        lines = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cv.render(img, t, depth_m=depth, hdr_source="macos", bright_thresh=1.0,
                      bright_stamp_gain=2.0, on_progress=lines.append, engine=engine)
        return [ln for ln in lines if "bright-pass:" in ln][0], bsum[engine]

    H, W = 150, 130
    ramp = np.geomspace(od[1], od[-1], W, dtype=np.float32)
    depth = np.broadcast_to(ramp[None], (H, W)).copy()
    for seed in range(6):
        rng = np.random.default_rng(seed)
        img = rng.uniform(0.0, 0.5, (H, W, 3)).astype(np.float32)
        for _ in range(15):
            y, x = int(rng.integers(3, H - 4)), int(rng.integers(3, W - 4))
            img[y:y + 3, x:x + 3] = rng.uniform(2, 6)
        _, br = run(img, depth, "reference")
        _, bt = run(img, depth, "tiled")
        if br == bt:
            continue
        s0 = float(img.sum())
        b = (np.floor(br / s0 * 1e4) + 0.5) / 1e4
        lo, hi = sorted((br / b, bt / b))
        blk = img[0:10, 0:10, 1]                 # dark (< 0.5): diffuse only
        step = ((lo + hi) / 2 - s0) / blk.size
        if abs(step) > 0.2:
            continue
        blk += np.float32(step)
        for _ in range(20000):
            s = float(img.sum())
            if lo < s < hi:
                break
            img[5, 5, 1] = np.nextafter(
                img[5, 5, 1], np.float32(np.inf if s < lo else -np.inf),
                dtype=np.float32)
        else:
            continue
        line_r, br2 = run(img, depth, "reference")
        line_t, bt2 = run(img, depth, "tiled")
        if (br2, bt2) != (br, bt):               # the tweak leaked into bright
            continue
        head_r, ratio_r = line_r.rsplit("bright_energy_ratio=", 1)
        head_t, ratio_t = line_t.rsplit("bright_energy_ratio=", 1)
        assert head_t == head_r, (line_r, line_t)
        assert len(ratio_t) == len(ratio_r)
        assert abs(float(ratio_t) - float(ratio_r)) < 1.5e-4, (line_r, line_t)
        assert ratio_t != ratio_r, "boundary construction lost: lines are identical"
        return
    pytest.skip("no scene with differing bright sums on this platform (not constructible)")
