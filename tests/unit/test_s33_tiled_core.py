"""S33 Tier B WP1: tiled engine core (not wired into render()).

Needs no ``runs/`` goldens and no IMG_6915: the oracle is the in-tree reference
gather (``convolve._layered_gather_diffuse``) and a float64 ``fftconvolve``.
Gates covered: G-B3 (bank identity, FAST part), G-B3b (geometric halo oracle),
G-B5 always-run oracle, G-B7 (threads, crops, holes and tie, cancel, import,
trace).
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import threading
import warnings
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest
from scipy.signal import fftconvolve

import summicron.render.convolve as cv
from summicron.render import _engine_tiled as eng
from summicron.render import _tile_schedule as ts
from summicron.render._artifacts import default_tensor_path
from summicron.render.api import CancelledError
from tests.fixtures import tolerance as tol

REPO = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------
# scene builders
# --------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _tensor(lens: str) -> dict:
    return cv.load_tensor(default_tensor_path(lens))


def _geometry(h, w, radial_edges):
    ys = np.arange(h, dtype=np.float32) - np.float32((h - 1) / 2.0)
    xs = np.arange(w, dtype=np.float32) - np.float32((w - 1) / 2.0)
    max_radial = float(np.hypot((h - 1) / 2.0, (w - 1) / 2.0))
    rf = (np.hypot(ys[:, None], xs[None, :]) / np.float32(max_radial)).astype(np.float32)
    theta = np.arctan2(ys[:, None], xs[None, :])
    wedge = ((theta + np.pi) / (2 * np.pi) * cv.N_ANG).astype(np.int32) % cv.N_ANG
    rc = 0.5 * (radial_edges[:-1] + radial_edges[1:])
    return rf, wedge, rc


def _gather_kwargs(img, depth, *, lens="35", K=67, thresh=1.0, slab_w=None,
                   tensor=None, bright_stamp_gain=None):
    """The kwargs ``render()`` hands to ``_layered_gather_diffuse``."""
    t = tensor if tensor is not None else _tensor(lens)
    kernels = t["kernels"]
    kH, kW = kernels.shape[2], kernels.shape[3]
    H, W = img.shape[:2]
    pix_scale = 1.0 if K == kH else K / kH
    pad = K // 2
    rf, wedge, rc = _geometry(H, W, t["radial_edges"])
    img_padded = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
    lum = (0.2126 * img_padded[..., 0] + 0.7152 * img_padded[..., 1]
           + 0.0722 * img_padded[..., 2])
    mask = (lum > thresh).astype(np.float32)[..., None]
    bright = img_padded * mask
    diffuse = img_padded - bright
    if slab_w is None:
        slab_w = cv.compute_slab_weights(depth, t["object_distances_m"])
    return dict(
        depth_m=depth, slab_w=slab_w, img_lin=img,
        img_padded=img_padded, img_diffuse_padded=diffuse, img_bright_padded=bright,
        kernels=kernels, defocus_ratios=t["defocus_ratios"],
        radial_frac_map=rf, r_centers=rc, wedge_idx=wedge,
        kH=kH, kW=kW, pad=pad, pix_scale=pix_scale, scaled_ksize=K, H=H, W=W,
        slab_batch_size=None, cancel=None,
        bright_stamp_gain=bright_stamp_gain,
    )


def _run_ref(kw):
    """Reference gather. Returns (5-tuple, progress lines)."""
    lines: list[str] = []
    k = dict(kw)
    k["fft_state"] = cv._build_image_fft_state(
        kw["img_padded"], kw["img_diffuse_padded"], kw["img_bright_padded"],
        kw["scaled_ksize"])
    k["on_progress"] = lines.append
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return cv._layered_gather_diffuse(**k), lines


def _run_tiled(kw, **extra):
    lines: list[str] = []
    k = dict(kw)
    k["fft_state"] = None
    k["on_progress"] = lines.append
    k.update(extra)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return eng.tiled_layered_gather(**k), lines


def _depth(h, w, od, patches=True):
    ramp = np.geomspace(od[0], od[-1], w, dtype=np.float32)
    depth = np.broadcast_to(ramp[None, :], (h, w)).copy()
    if patches:
        depth[h // 4: h // 2, w // 5: w // 2] = np.float32(od[3])
        depth[h // 2: 3 * h // 4, w // 3: 4 * w // 5] = np.float32(od[7])
    return depth


def _scene(h, w, seed=0, n_hot=8, lens="35", K=67, **kw):
    rng = np.random.default_rng(seed)
    img = rng.uniform(0.0, 0.8, (h, w, 3)).astype(np.float32)
    for _ in range(n_hot):
        y = int(rng.integers(2, h - 3))
        x = int(rng.integers(2, w - 3))
        img[y:y + 3, x:x + 3] = rng.uniform(2.0, 4.0)
    od = _tensor(lens)["object_distances_m"]
    kw.setdefault("bright_stamp_gain", 2.0)
    return _gather_kwargs(img, _depth(h, w, od), lens=lens, K=K, **kw)


def _total(t5):
    return (t5[0] + t5[1]).astype(np.float32)


def _assert_tier_b(ref5, got5, what=""):
    res = tol.evaluate(_total(ref5), _total(got5), "B")
    assert res["ok"], f"{what}: {tol.summary(res)}"
    res_d = tol.evaluate(ref5[0], got5[0], "B")
    assert res_d["ok"], f"{what} (diffuse): {tol.summary(res_d)}"
    # the bright image alone is sparse (an annulus can hold ~0 energy), so only
    # the pointwise ceiling applies to it; Tier B proper is the total above.
    eb = np.abs(got5[1].astype(np.float64) - ref5[1]) / np.maximum(ref5[1], 1.0)
    assert float(eb.max()) <= tol.TIERS["B"]["max_e"], f"{what} (bright)"
    for i, name in ((2, "weight_sum"), (3, "delta_weight_sum"), (4, "acc_alpha")):
        assert np.array_equal(ref5[i], got5[i]), f"{what}: {name} differs"


# --------------------------------------------------------------------------
# geometry and table
# --------------------------------------------------------------------------

def test_tile_n_table():
    for K, N in ((3, 324), (17, 324), (67, 324), (99, 324),
                 (100, 512), (131, 512), (183, 512), (255, 512)):
        assert ts.tile_n_for_ksize(K) == N, K
    with pytest.raises(ValueError, match="not available"):
        ts.tile_n_for_ksize(256)


@pytest.mark.parametrize("H,W,K,N", [(905, 1210, 67, 324), (1210, 905, 67, 324),
                                     (100, 115, 67, 96), (700, 760, 183, 512)])
def test_plan_partitions_the_frame(H, W, K, N):
    T, tiles = ts.plan_tiles(H, W, K, N)
    assert T == N - K + 1
    cover = np.zeros((H, W), np.int32)
    for (y0, y1, x0, x1) in tiles:
        cover[y0:y1, x0:x1] += 1
        assert y1 - y0 <= T and x1 - x0 <= T
    assert (cover == 1).all()
    assert tiles == sorted(tiles, key=lambda t: (t[0], t[2]))   # row-major


def test_private_tile_n_must_exceed_k():
    kw = _scene(40, 40, K=17)
    with pytest.raises(ValueError, match="_tile_n"):
        _run_tiled(kw, _tile_n=17)


def test_k_over_255_is_not_available():
    kw = _scene(40, 40, K=17)
    kw["scaled_ksize"] = 256
    kw["pad"] = 128
    with pytest.raises(ValueError, match="not available"):
        _run_tiled(kw)


def test_pad_mismatch_is_not_available():
    kw = _scene(40, 40, K=17)
    kw["pad"] = 5
    with pytest.raises(ValueError, match="not available"):
        _run_tiled(kw)


# --------------------------------------------------------------------------
# G-B3b geometric halo oracle (golden-free, full-support kernels)
# --------------------------------------------------------------------------

def _halo_case(K, N, H, W, seed=1):
    rng = np.random.default_rng(seed)
    pad = K // 2
    img = rng.uniform(0.0, 6.8, (H, W, 3)).astype(np.float32)
    ip = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
    ker = rng.uniform(0.2, 1.0, (K, K)).astype(np.float32)
    ker /= ker.sum()
    got = eng.tile_convolve_full(ip, ker, H, W, N=N, threads=2)
    truth = np.empty((H, W, 3), np.float64)
    for c in range(3):
        full = fftconvolve(ip[:, :, c].astype(np.float64), ker.astype(np.float64),
                           mode="full")
        truth[:, :, c] = full[K - 1:K - 1 + H, K - 1:K - 1 + W]
    e = np.abs(got.astype(np.float64) - truth) / np.maximum(truth, 1.0)
    return e, truth, got


@pytest.mark.parametrize("K,N,H,W", [
    (67, 96, 100, 115), (67, 96, 115, 100),      # >= 3x3 tiles, both orientations
    (66, 96, 110, 100), (66, 96, 100, 110),      # even K
    (41, 64, 70, 90), (41, 64, 90, 70),
    (67, None, 600, 700), (67, None, 700, 600),  # production N=324, 3x3 tiles
])
def test_halo_oracle(K, N, H, W):
    N_eff = N or ts.tile_n_for_ksize(K)
    T, tiles = ts.plan_tiles(H, W, K, N_eff)
    assert len({t[0] for t in tiles}) >= 3 and len({t[2] for t in tiles}) >= 3
    assert H % T and W % T            # not multiples of T
    e, truth, got = _halo_case(K, N, H, W)
    assert float(e.max()) <= 2e-6, float(e.max())
    # discriminating power: a one-pixel slip is orders of magnitude worse
    slip = np.abs(got[1:].astype(np.float64) - truth[:-1]) / np.maximum(truth[:-1], 1.0)
    assert float(slip.max()) > 1e-3


def test_halo_oracle_k183():
    e, _, _ = _halo_case(183, None, 700, 760)
    assert float(e.max()) <= 2e-6, float(e.max())


def test_halo_band_impulse():
    """One bright impulse that lies only in a tile's halo band (outside its
    output rectangle) must still spread into that tile's output."""
    K, Nn = 67, 96                       # T = 30
    h, w = 100, 115
    od = _tensor("35")["object_distances_m"]
    img = np.full((h, w, 3), 0.05, dtype=np.float32)
    iy, ix = 63, 45                      # tile rows [30,60) and cols [30,60): halo
    img[iy, ix] = 5.0
    depth = np.full((h, w), od[4], dtype=np.float32)
    kw = _gather_kwargs(img, depth, K=K)
    ref5, _ = _run_ref(kw)
    got5, _ = _run_tiled(kw, _tile_n=Nn)
    tile_out = (slice(30, 60), slice(30, 60))
    assert float(ref5[1][tile_out].max()) > 1e-4      # energy really reaches the tile
    _assert_tier_b(ref5, got5, "halo-band impulse")
    # the impulse is not inside any output rectangle of the tile it is spread into
    assert not (30 <= iy < 60 and 30 <= ix < 60)


# --------------------------------------------------------------------------
# oracle against the reference gather (always runs in FAST)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("hw", [(200, 170), (170, 200)])
def test_gather_matches_reference(hw):
    kw = _scene(*hw, seed=4)
    ref5, ref_lines = _run_ref(kw)
    got5, got_lines = _run_tiled(kw, _tile_n=96)
    _, tiles = ts.plan_tiles(hw[0], hw[1], 67, 96)
    assert len(tiles) >= 9
    _assert_tier_b(ref5, got5, f"gather {hw}")
    assert ref_lines == got_lines
    # the oracle is not vacuous: highlights spread into the bright image
    assert float(ref5[1].max()) > 0.1


def _chart(h, w, amp, rf_list=(0.3, 0.65, 0.95), n_az=6):
    img = np.full((h, w, 3), 0.02, np.float32)
    od = _tensor("35")["object_distances_m"]
    depth = np.full((h, w), od[5], np.float32)
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    maxr = float(np.hypot(cy, cx))
    centers = []
    k = 0
    for rfv in rf_list:
        for ia in range(n_az):
            th = (ia + (0.5 if (k % 2) else 0.0)) * 2 * np.pi / n_az + 0.07 * k
            y = int(np.clip(round(cy + rfv * maxr * np.sin(th)), 6, h - 7))
            x = int(np.clip(round(cx + rfv * maxr * np.cos(th)), 6, w - 7))
            img[y - 1:y + 2, x - 1:x + 2] = amp
            dv = od[1 + k % 10] if k % 3 else np.sqrt(od[1 + k % 9] * od[2 + k % 9])
            depth[y - 5:y + 6, x - 5:x + 6] = np.float32(dv)
            centers.append((y, x))
            k += 1
    return img, depth, centers


@pytest.mark.parametrize("hw", [(260, 220), (220, 260)])
@pytest.mark.parametrize("amp,gain", [(1.0, None), (16.0, 2.0)])
def test_chart_matches_reference_field_character(hw, amp, gain):
    """Always-run FAST variant of the G-B5 chart: LDR 1.0 peak stamp and HDR
    16x L1 gain 2, 3x3 lights on 6 azimuths at 3 radii (rf 0.3 to 0.95), depths
    cycling over the slab nodes plus between-node values, K=67 with >= 3x3
    non-multiple tiles, reference computed in-test. The light density is the
    spec's realistic range: a 48-light peak chart on this small frame is the
    dense-highlight stress family (float32 round-off, R-B2): it is gated by the
    stress ratio (``test_s33_tiled_stress_ratio.py``), not by the hard ceilings."""
    img, depth, centers = _chart(*hw, amp)
    kw = _gather_kwargs(img, depth, K=67, thresh=0.5, bright_stamp_gain=gain)
    ref5, _ = _run_ref(kw)
    got5, _ = _run_tiled(kw, _tile_n=96)
    _assert_tier_b(ref5, got5, f"chart {hw} amp={amp} gain={gain}")
    local = tol.local_max_e(_total(ref5), _total(got5), centers)
    print(f"chart {hw} amp={amp} gain={gain}: local max e per light "
          f"max={max(local):.3g} min={min(local):.3g}")
    assert max(local) <= tol.TIERS["B"]["max_e"]


def test_second_geometry_k183_n512():
    """K=183 / N=512 (pix_scale 1.0: no zoom) on a small frame, 2x2 tiles via
    _tile_n so the frame stays small; the reference is computed in-test."""
    t = dict(_tensor("35"))
    t.pop("corpus_ref_long_edge", None)
    h, w = 120, 100
    rng = np.random.default_rng(9)
    img = rng.uniform(0.0, 0.8, (h, w, 3)).astype(np.float32)
    img[30:33, 70:73] = 4.0
    img[90:93, 20:23] = 3.0
    depth = _depth(h, w, t["object_distances_m"])
    kw = _gather_kwargs(img, depth, K=183, tensor=t, bright_stamp_gain=2.0)
    ref5, _ = _run_ref(kw)
    got5, _ = _run_tiled(kw, _tile_n=256)      # T = 74: 2x2 tiles
    _assert_tier_b(ref5, got5, "K=183")


# --------------------------------------------------------------------------
# G-B3 kernel bank identity
# --------------------------------------------------------------------------

def _sha(a):
    return hashlib.sha256(np.ascontiguousarray(a, dtype=np.float32).tobytes()).hexdigest()


def _capture_reference_bank(kw):
    seen: list[tuple] = []
    if cv._STAGED_FFT:
        real = cv._kernel_rfft2_staged

        def spy(k, fshape):
            seen.append((_sha(k), bool(k.max() > 0.5 * k.sum())))
            return real(k, fshape)

        cv._kernel_rfft2_staged = spy
        try:
            _run_ref(kw)
        finally:
            cv._kernel_rfft2_staged = real
    else:
        real = cv.sp_fft.rfft2

        def spy(x, *a, **k):
            if getattr(x, "ndim", 0) == 2 and x.shape == (kw["scaled_ksize"],) * 2:
                seen.append((_sha(x), bool(x.max() > 0.5 * x.sum())))
            return real(x, *a, **k)

        cv.sp_fft.rfft2 = spy
        try:
            # the image-side spectra are built (outside) before the spy would
            # see 2-D K x K arrays; they are H_pad x W_pad, so never match.
            _run_ref(kw)
        finally:
            cv.sp_fft.rfft2 = real
    # The reference convolves an is_delta unit once per channel, so it
    # transforms the same kernel three times in a row; collapse those.
    hashes, run = [], 0
    for h, is_delta in seen:
        if is_delta:
            run += 1
            if run % 3 != 1:
                continue
        else:
            run = 0
        hashes.append(h)
    return hashes


def _capture_tiled_bank(kw, **extra):
    seen: list[tuple] = []
    real = eng._kernel_spectra

    def spy(arrays, n, labels=None, workers=1):
        K = kw["scaled_ksize"]
        for a, lab in zip(arrays, labels):
            assert a.shape == (K, K)
            seen.append((lab, _sha(a)))
        return real(arrays, n, labels, workers)

    eng._kernel_spectra = spy
    try:
        _run_tiled(kw, **extra)
    finally:
        eng._kernel_spectra = real
    return seen


@pytest.mark.parametrize("lens,K,n_tile,hw", [
    ("35", 67, 96, (130, 110)),
    ("50", 67, 96, (110, 130)),
    ("35", 17, 40, (80, 70)),
])
def test_kernel_bank_identity(lens, K, n_tile, hw):
    kw = _scene(*hw, seed=11, lens=lens, K=K)
    ref = _capture_reference_bank(kw)
    got = _capture_tiled_bank(kw, _tile_n=n_tile)
    assert len(ref) > 100
    assert [h for _, h in got] == ref          # same keys' hashes, same order
    ris = {lab[1] for lab, _ in got}
    assert ris == set(range(12)), ris
    # 8 wedges for every non-delta ri in every slab where the unit is banked
    wedges: dict = {}
    for (d, ri, a, kind), _ in got:
        if a != "delta" and kind == "diffuse":
            wedges.setdefault((d, ri), set()).add(a)
    assert max(len(v) for v in wedges.values()) >= 5, wedges
    assert all(v <= set(range(8)) for v in wedges.values())


def test_kernel_spectra_places_kxk_block_at_origin():
    import scipy.fft as F

    rng = np.random.default_rng(3)
    ks = [rng.random((17, 17)).astype(np.float32) for _ in range(20)]
    got = eng._kernel_spectra(ks, 40)
    for k, g in zip(ks, got):
        # the bank's spectra are float64 transforms rounded once to complex64
        want = F.rfft2(k.astype(np.float64), s=(40, 40)).astype(np.complex64)
        assert np.array_equal(want, g)


def test_rotation_sign_on_asymmetric_kernel():
    """The banked rotation is rot_deg = -(theta_a*180/pi + 90), not mirrored."""
    K, nr, D = 33, 12, 3
    rng = np.random.default_rng(5)
    kern = np.zeros((nr, D, K, K), np.float32)
    ys, xs = np.mgrid[0:K, 0:K].astype(np.float32)
    blob = np.exp(-(((ys - 10) / 2.0) ** 2 + ((xs - 22) / 4.5) ** 2) / 2.0)
    blob += 0.3 * np.exp(-(((ys - 22) / 2.5) ** 2 + ((xs - 8) / 2.0) ** 2) / 2.0)
    for r in range(nr):
        for d in range(1, D):
            kern[r, d] = blob / blob.sum() * (1 + 0.01 * rng.random())
        kern[r, 0, K // 2, K // 2] = 1.0
    ones = np.ones(nr, np.float32)
    flags = np.ones((nr, cv.N_ANG + 1), bool)
    units = ts.slab_bank(
        1, nr=nr, kernels=kern, defocus_ratios=np.linspace(0, 1, D),
        kH=K, kW=K, pix_scale=1.0, scaled_ksize=K, bright_stamp_gain=2.0,
        gmax_ri=ones, gmax_ria=np.ones((nr, cv.N_ANG), np.float32),
        bank_flags=flags)
    assert len(units) == nr * cv.N_ANG
    u = next(x for x in units if x.ri == 3 and x.a == 5)
    base = kern[3, 1].astype(np.float32)
    theta = (5 + 0.5) * (2 * np.pi / cv.N_ANG) - np.pi
    want = cv.ndimage_rotate(base, -(theta * 180.0 / np.pi + 90.0), reshape=False, order=1)
    want = np.maximum(want, 0.0)
    want /= want.sum() + 1e-12
    mirrored = cv.ndimage_rotate(base, +(theta * 180.0 / np.pi + 90.0), reshape=False, order=1)
    mirrored = np.maximum(mirrored, 0.0)
    mirrored /= mirrored.sum() + 1e-12
    assert u.kernel.tobytes() == want.tobytes()
    assert u.kernel_bright.tobytes() == (want * np.float32(2.0)).tobytes()
    assert not np.allclose(u.kernel, mirrored, atol=1e-4)


# --------------------------------------------------------------------------
# G-B7 determinism and exactness of parts
# --------------------------------------------------------------------------

def test_thread_count_invariance_bit_level():
    kw = _scene(130, 110, seed=6, K=31)
    base = None
    for threads in (1, 2, 4, min(8, os.cpu_count() or 1)):
        for _rep in range(3):
            got5, _ = _run_tiled(kw, _tile_n=64, threads=threads)
            assert len(ts.plan_tiles(130, 110, 31, 64)[1]) >= 9
            if base is None:
                base = got5
                continue
            for i in range(5):
                assert np.array_equal(base[i], got5[i]), (threads, i)
                assert base[i].tobytes() == got5[i].tobytes(), (threads, i)


@pytest.mark.parametrize("shape", [(2048, 1536), (1536, 2048)])
def test_tile_weights_tobytes_equal_full_frame(shape):
    t = _tensor("35")
    rf, wedge, rc = _geometry(*shape, t["radial_edges"])
    full = cv._radial_basis_weights(rf, rc)
    K = 67
    N = ts.tile_n_for_ksize(K)
    _, tiles = ts.plan_tiles(*shape, K, N)
    for tile in tiles:
        y0, y1, x0, x1 = tile
        basis, wt = ts.tile_weights(rf, wedge, rc, tile)
        assert basis.tobytes() == np.ascontiguousarray(full[:, y0:y1, x0:x1]).tobytes()
    # w_ang on a sample of units/tiles equals the reference expression on
    # the full plane
    for tile in (tiles[0], tiles[len(tiles) // 2], tiles[-1]):
        y0, y1, x0, x1 = tile
        basis, wt = ts.tile_weights(rf, wedge, rc, tile)
        for ri, a in ((0, 0), (3, 5), (7, 2), (11, 7)):
            full_w = (full[ri] * (wedge == a).astype(np.float32)).astype(np.float32)
            got = ts.wedge_weight(basis[ri], wt, a)
            assert got.tobytes() == np.ascontiguousarray(full_w[y0:y1, x0:x1]).tobytes()


@pytest.mark.slow
def test_tile_weights_native_like_shape():
    shape = (4032, 3024)
    t = _tensor("35")
    rf, wedge, rc = _geometry(*shape, t["radial_edges"])
    full = cv._radial_basis_weights(rf, rc)
    _, tiles = ts.plan_tiles(*shape, 183, ts.tile_n_for_ksize(183))
    for tile in tiles:
        y0, y1, x0, x1 = tile
        basis, _ = ts.tile_weights(rf, wedge, rc, tile)
        assert basis.tobytes() == np.ascontiguousarray(full[:, y0:y1, x0:x1]).tobytes()


def _holes_scene():
    h, w = 110, 120
    kw = _scene(h, w, seed=8, K=31)
    sw = kw["slab_w"].copy()
    sw[:, 10:20, 10:20] = 0.0                   # A: nobody covers (fallback)
    sw[:, 40:50, 40:50] = 0.0
    sw[3, 40:50, 40:50] = 0.015                 # B: single valid donor, d=3
    sw[:, 70:80, 60:70] = 0.0
    sw[6, 70:80, 60:70] = 0.012                 # C: donor d=6 (other side)
    sw[:, 90:100, 90:100] = 0.0
    sw[2, 90:100, 90:100] = 0.004               # D: only sub-threshold weight
    kw["slab_w"] = np.ascontiguousarray(sw)
    return kw


def test_holes_donors_fallback_and_acc_alpha():
    kw = _holes_scene()
    ref5, ref_lines = _run_ref(kw)
    got5, got_lines = _run_tiled(kw, _tile_n=64)
    assert "holes=" in ref_lines[-1] and ref_lines == got_lines
    assert np.array_equal(ref5[4], got5[4])
    n_holes = int((kw["slab_w"].sum(axis=0) < 0.02).sum())
    assert n_holes > 0
    # no-donor fallback hit: block A is exactly img_lin, bright 0
    A = (slice(10, 20), slice(10, 20))
    assert np.array_equal(got5[0][A], kw["img_lin"][A])
    assert not got5[1][A].any()
    # donors used: block B is not the fallback
    B = (slice(40, 50), slice(40, 50))
    assert not np.array_equal(got5[0][B], kw["img_lin"][B])
    _assert_tier_b(ref5, got5, "holes")


def test_holes_tie_lowest_d_wins():
    """Two valid donors at equal |d - argmax_d|: strict '<' keeps the lower d
    (slabs are visited in ascending d)."""
    h, w = 3, 3
    HUGE = np.int32(110)
    argmax = np.full((h, w), 3, np.int32)
    best_dist = np.full((h, w), HUGE, np.int32)
    best_d = np.zeros((h, w, 3), np.float32)
    best_b = np.zeros((h, w, 3), np.float32)
    sw = np.full((h, w), 0.05, np.float32)
    for d, val in ((2, 1.0), (4, 2.0)):
        od_ = np.full((h, w, 3), val, np.float32)
        ob_ = np.full((h, w, 3), val * 10, np.float32)
        best_dist, best_d, best_b = eng._update_best_donor(
            best_dist, best_d, best_b, d, sw, argmax, od_, ob_, HUGE)
    assert (best_dist == 1).all()
    assert (best_d == 1.0).all() and (best_b == 10.0).all()
    # an invalid slab never donates; a nearer valid one displaces a farther one
    inval = np.full((h, w), 0.005, np.float32)
    b0 = best_dist.copy()
    best_dist, best_d, _ = eng._update_best_donor(
        best_dist, best_d, best_b, 3, inval, argmax,
        np.full((h, w, 3), 9.0, np.float32), np.zeros((h, w, 3), np.float32), HUGE)
    assert np.array_equal(best_dist, b0) and (best_d == 1.0).all()
    best_dist, best_d, _ = eng._update_best_donor(
        best_dist, best_d, best_b, 3, sw, argmax,
        np.full((h, w, 3), 9.0, np.float32), np.zeros((h, w, 3), np.float32), HUGE)
    assert (best_dist == 0).all() and (best_d == 9.0).all()


def test_cancel_preset_raises_before_any_fft():
    kw = _scene(60, 60, seed=2, K=17)
    ev = threading.Event()
    ev.set()
    calls = {"spec": 0, "job": 0}
    real_s, real_j = eng._kernel_spectra, eng._tile_job

    def s(*a, **k):
        calls["spec"] += 1
        return real_s(*a, **k)

    def j(*a, **k):
        calls["job"] += 1
        return real_j(*a, **k)

    eng._kernel_spectra, eng._tile_job = s, j
    try:
        with pytest.raises(CancelledError, match=r"^render cancelled at slab 0/11$"):
            _run_tiled(dict(kw, cancel=ev), _tile_n=40)
    finally:
        eng._kernel_spectra, eng._tile_job = real_s, real_j
    assert calls == {"spec": 0, "job": 0}


def test_cancel_mid_slab_uses_tiled_text_and_shuts_pool_down():
    kw = _scene(60, 60, seed=2, K=17)
    ev = threading.Event()
    real = eng._kernel_spectra

    def spy(*a, **k):
        out = real(*a, **k)
        ev.set()
        return out

    before = {t.ident for t in threading.enumerate()}
    eng._kernel_spectra = spy
    try:
        with pytest.raises(CancelledError,
                           match=r"^render cancelled inside slab \d+ \(tiled engine\)$"):
            _run_tiled(dict(kw, cancel=ev), _tile_n=40, threads=4)
    finally:
        eng._kernel_spectra = real
    leftover = [t for t in threading.enumerate()
                if t.ident not in before and t.name.startswith("ThreadPoolExecutor")]
    assert not leftover


def test_import_isolation_subprocess():
    code = (
        "import sys\n"
        "import numpy as np\n"
        "import summicron.render.convolve as cv\n"
        "from tests.fixtures.synthetic_tensor import make_synthetic_tensor\n"
        "from tests.fixtures.defocus import render_with_defocus\n"
        "t = make_synthetic_tensor(12, 5, 17, 17)\n"
        "img = np.random.default_rng(0).uniform(0, 1, (40, 48, 3)).astype(np.float32)\n"
        "render_with_defocus(img, t)\n"
        "bad = [m for m in ('summicron.render._engine_tiled',\n"
        "                   'summicron.render._tile_schedule', 'mlx') if m in sys.modules]\n"
        "assert not bad, bad\n"
        "print('ISOLATED')\n"
    )
    env = dict(os.environ, PYTHONPATH=str(REPO))
    r = subprocess.run([sys.executable, "-c", code], cwd=REPO, env=env,
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0 and "ISOLATED" in r.stdout, r.stdout + r.stderr


def test_trace_tiled_engine_ran_reference_apply_kernel_did_not():
    kw = _scene(60, 60, seed=2, K=17)
    calls = {"spec": 0, "job": 0, "apply": 0}
    real_s, real_j, real_a = eng._kernel_spectra, eng._tile_job, cv._apply_kernel

    def s(*a, **k):
        calls["spec"] += 1
        return real_s(*a, **k)

    def j(*a, **k):
        calls["job"] += 1
        return real_j(*a, **k)

    def ap(*a, **k):
        calls["apply"] += 1
        return real_a(*a, **k)

    eng._kernel_spectra, eng._tile_job, cv._apply_kernel = s, j, ap
    try:
        _run_tiled(kw, _tile_n=40)
    finally:
        eng._kernel_spectra, eng._tile_job, cv._apply_kernel = real_s, real_j, real_a
    assert calls["spec"] > 0 and calls["job"] > 0 and calls["apply"] == 0


def test_max_active_slabs_warning_matches_reference():
    kw = _scene(60, 60, seed=2, K=17)

    def run(fn):
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            fn(dict(kw, active_slab_cap=2))
        return [(w.category, str(w.message)) for w in rec
                if issubclass(w.category, RuntimeWarning)
                and "_MAX_ACTIVE_SLABS" in str(w.message)]

    def ref(k):
        k = dict(k)
        k["fft_state"] = cv._build_image_fft_state(
            k["img_padded"], k["img_diffuse_padded"], k["img_bright_padded"], 17)
        k["on_progress"] = None
        k["emit_log"] = False
        cv._layered_gather_diffuse(**k)

    def til(k):
        k = dict(k, fft_state=None, on_progress=None, emit_log=False)
        eng.tiled_layered_gather(**k, _tile_n=40)

    a, b = run(ref), run(til)
    assert len(a) == len(b) == 1 and a == b


# --------------------------------------------------------------------------
# metric copy
# --------------------------------------------------------------------------

def test_tolerance_fixture_matches_frozen_gate():
    gate_dir = REPO / "scripts" / "s33"
    if not (gate_dir / "gate_tolerance.py").exists():
        pytest.skip("scripts/s33 absent")
    sys.path.insert(0, str(gate_dir))
    try:
        import gate_tolerance as GT
    finally:
        sys.path.remove(str(gate_dir))
    assert GT.TIERS == tol.TIERS
    rng = np.random.default_rng(0)
    ref = rng.uniform(0, 6, (60, 80, 3)).astype(np.float32)
    test = (ref * (1 + rng.normal(0, 2e-7, ref.shape))).astype(np.float32)
    r8, t8 = tol.to_u8(ref), tol.to_u8(test)
    mask = tol.annulus_mask(60, 80)
    assert np.array_equal(mask, GT.annulus_mask(60, 80))
    for m in (None, mask):
        assert tol.region_metrics(ref, test, r8, t8, m) == GT.region_metrics(
            ref, test, r8, t8, m)
    a = tol.evaluate(ref, test, "B", r8, t8)
    b = GT.evaluate(ref, test, r8, t8, "B")
    assert a == b


def test_tolerance_fixture_rejects_a_miss():
    ref = np.full((40, 40, 3), 2.0, np.float32)
    bad = ref.copy()
    bad[5, 5, 0] += 1e-4
    assert not tol.evaluate(ref, bad, "B")["ok"]
    assert tol.evaluate(ref, ref.copy(), "B")["ok"]


def test_image_side_ffts_are_float32_complex64():
    """Spec amendment (section 12): only the KERNEL spectra may be float64
    transforms; the image-side FFT path stays float32 -> complex64."""
    from summicron.render import _engine_tiled as et
    from summicron.render import _tile_schedule as ts

    h, w, K, N = 40, 36, 7, 32
    img = np.random.default_rng(0).uniform(0, 1, (h + 2 * K, w + 2 * K, 3)).astype(np.float32)
    tile = (0, min(N - K + 1, h), 0, min(N - K + 1, w))
    S = et._tile_spec(img, tile, K, N)
    assert S.dtype == np.complex64
    ker = np.full((K, K), 1.0 / (K * K), np.float32)
    ks = et._kernel_spectra([ker], N)
    assert ks.dtype == np.complex64
    conv = et._convolve_chunk([S], [ks[0]], K, tile[1] - tile[0], tile[3] - tile[2], N)
    assert conv.dtype == np.float32
