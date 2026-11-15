"""S33 Tier C WP1: MLX engine core (not wired into render()).

Gates covered (locked-spec-S33-gpu.md): G-C3 (bank identity, FAST part), G-C3b
(geometric halo oracle, per-unit parity, seams and slivers, FFT-side spy), and
the G-C7 parts that need only the gather (compact basis, closed-form weight
sums, take guard, holes, cancel, memory hygiene, ``_LAST_RUN``, device,
execution proof, determinism, knob neutrality).

Everything is gated by the in-tree reference gather (and a float64
``fftconvolve``); no ``runs/``, no HEIC, no IMG_6915. No GPU output bytes or md5
are stored anywhere: determinism is asserted in-session only (G-C0).
"""
from __future__ import annotations

import gc
import hashlib
import os
import subprocess
import sys
import threading
import warnings
from pathlib import Path

import numpy as np
import pytest
from scipy.signal import fftconvolve

mx = pytest.importorskip("mlx.core")

import summicron.render.convolve as cv  # noqa: E402
from summicron.render import _engine_gpu as eg  # noqa: E402
from summicron.render import _engine_tiled as et  # noqa: E402
from summicron.render import _tile_schedule as ts  # noqa: E402
from summicron.render.api import CancelledError  # noqa: E402
from tests.fixtures import tolerance as tol  # noqa: E402
from tests.fixtures.synthetic_tensor import make_synthetic_tensor  # noqa: E402
from tests.unit.test_s33_tiled_core import (  # noqa: E402
    _capture_reference_bank, _chart, _depth, _gather_kwargs, _geometry,
    _holes_scene, _run_ref, _scene, _sha, _tensor, _total,
)

pytestmark = pytest.mark.gpu

REPO = Path(__file__).resolve().parents[2]
_TOL_C = tol.TIERS["C"]


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _run_gpu(kw, **extra):
    lines: list[str] = []
    k = dict(kw)
    k["fft_state"] = None
    k["on_progress"] = lines.append
    k.update(extra)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return eg.gpu_layered_gather(**k), lines


def _assert_tier_c(ref5, got5, what=""):
    res = tol.evaluate(_total(ref5), _total(got5), "C")
    assert res["ok"], f"{what}: {tol.summary(res)}"
    res_d = tol.evaluate(ref5[0], got5[0], "C")
    assert res_d["ok"], f"{what} (diffuse): {tol.summary(res_d)}"
    eb = np.abs(got5[1].astype(np.float64) - ref5[1]) / np.maximum(ref5[1], 1.0)
    assert float(eb.max()) <= _TOL_C["max_e"], f"{what} (bright)"
    for i, name in ((2, "weight_sum"), (3, "delta_weight_sum"), (4, "acc_alpha")):
        assert np.array_equal(ref5[i], got5[i]), f"{what}: {name} differs"


def _same5(a, b, what=""):
    for i in range(5):
        assert a[i].tobytes() == b[i].tobytes(), f"{what}: element {i} differs"


def _imp_metrics(ref, got, centers, K):
    """Per-impulse local metric (R-C12): e_imp, relative L1, centroid shift and
    tail sign, on the impulse's K x K box."""
    h, w = ref.shape[:2]
    half = K // 2
    out = []
    for (y, x) in centers:
        ys, ye = max(0, y - half), min(h, y + half + 1)
        xs, xe = max(0, x - half), min(w, x + half + 1)
        r = ref[ys:ye, xs:xe].astype(np.float64)
        g = got[ys:ye, xs:xe].astype(np.float64)
        e_imp = float(np.abs(g - r).max() / max(float(np.abs(r).max()), 1e-30))
        l1 = abs(float(g.sum()) / float(r.sum()) - 1.0) if r.sum() else 0.0
        yy, xx = np.mgrid[ys:ye, xs:xe]
        rl, gl = r.sum(axis=2), g.sum(axis=2)
        cr = np.array([(yy * rl).sum() / rl.sum() - y, (xx * rl).sum() / rl.sum() - x])
        cg = np.array([(yy * gl).sum() / gl.sum() - y, (xx * gl).sum() / gl.sum() - x])
        shift = float(np.abs(cg - cr).max())
        sign_ok = all(np.sign(a) == np.sign(b) for a, b in zip(cr, cg) if abs(a) > 0.05)
        out.append((e_imp, l1, shift, sign_ok))
    return out


def _assert_impulses(ref5, got5, centers, K, what):
    ms = _imp_metrics(_total(ref5), _total(got5), centers, K)
    worst = (max(m[0] for m in ms), max(m[1] for m in ms), max(m[2] for m in ms))
    print(f"{what}: per-impulse worst e_imp={worst[0]:.3g} L1={worst[1]:.3g} "
          f"centroid={worst[2]:.3g}")
    assert worst[0] <= 1e-4, what
    assert worst[1] <= 1e-5, what
    assert worst[2] <= 0.01, what
    assert all(m[3] for m in ms), what


def _impulse_scene(h, w, K, positions, amp=4.0, d_node=4, lens="35"):
    img = np.full((h, w, 3), 0.03, np.float32)
    od = _tensor(lens)["object_distances_m"]
    depth = np.full((h, w), od[d_node], np.float32)
    for (y, x) in positions:
        img[y - 1:y + 2, x - 1:x + 2] = amp
    return _gather_kwargs(img, depth, lens=lens, K=K, thresh=0.5,
                          bright_stamp_gain=2.0)


# --------------------------------------------------------------------------
# environment
# --------------------------------------------------------------------------

def test_mlx_pin_and_device():
    assert mx.__version__ == "0.32.4"
    assert mx.metal.is_available()
    assert eg.availability() is None
    assert mx.default_device() == mx.gpu


# --------------------------------------------------------------------------
# compact weights (exact by construction, gated by byte equality)
# --------------------------------------------------------------------------

def _special_rf(h, w, rc, seed=0):
    """A radial_frac map that hits nodes exactly (t = 0), sits below rc[0] and
    above rc[-1], and elsewhere is arbitrary."""
    rng = np.random.default_rng(seed)
    rf = rng.uniform(0.0, 1.05, (h, w)).astype(np.float32)
    flat = rf.reshape(-1)
    n = flat.size
    pick = rng.choice(n, size=3 * len(rc) + 20, replace=False)
    nodes = np.concatenate([rc.astype(np.float32), rc.astype(np.float32),
                            np.nextafter(rc.astype(np.float32), np.float32(2)),
                            np.array([0.0, 1e-6, rc[0] * 0.5, rc[-1] + 0.01, 1.2,
                                      rc[0], rc[-1], rc[3], rc[3], 0.0, 1.0,
                                      0.5, 0.25, 0.75, 0.333, 0.9, 0.99, 1.0, 0.1, 0.2],
                                     np.float32)])
    flat[pick[:len(nodes)]] = nodes
    return rf


@pytest.mark.parametrize("shape,K", [((700, 560), 67), ((560, 700), 67),
                                     ((120, 90), 17), ((90, 120), 17)])
def test_compact_basis_tobytes_equals_radial_basis_weights(shape, K):
    t = _tensor("35")
    _, wedge, rc = _geometry(*shape, t["radial_edges"])
    rf = _special_rf(*shape, rc)
    N = ts.tile_n_for_ksize(K)
    T, tiles = ts.plan_tiles(*shape, K, N)
    ri_lo, w_lo, w_hi, wd = eg._compact_basis(rf, rc, tiles, wedge_idx=wedge, T=T)
    full = cv._radial_basis_weights(rf, rc)
    nr = len(rc)
    for ti, (y0, y1, x0, x1) in enumerate(tiles):
        th, tw = y1 - y0, x1 - x0
        rec = np.zeros((nr, th, tw), np.float32)
        yy, xx = np.indices((th, tw))
        r = ri_lo[ti, :th, :tw].astype(np.int64)
        rec[r, yy, xx] += w_lo[ti, :th, :tw]
        rec[r + 1, yy, xx] += w_hi[ti, :th, :tw]
        assert rec.tobytes() == np.ascontiguousarray(full[:, y0:y1, x0:x1]).tobytes()
        assert np.array_equal(wd[ti, :th, :tw], wedge[y0:y1, x0:x1])
        # out-of-frame pixels of partial tiles carry zero weight
        assert not w_lo[ti, th:, :].any() and not w_hi[ti, :, tw:].any()
    # the clamps and the node hits really occurred
    assert (rf < rc[0]).any() and (rf > rc[-1]).any()
    assert np.isin(rf, rc.astype(np.float32)).any()
    assert int(ri_lo.max()) == nr - 2


def test_compact_basis_needs_two_bins():
    rf = np.zeros((10, 10), np.float32)
    with pytest.raises(ValueError, match="not available"):
        eg._compact_basis(rf, np.array([0.5], np.float32), [(0, 10, 0, 10)])


def test_pair_weights_tobytes_equal_wedge_weight():
    shape, K = (700, 560), 67
    t = _tensor("35")
    _, wedge, rc = _geometry(*shape, t["radial_edges"])
    rf = _special_rf(*shape, rc, seed=3)
    T, tiles = ts.plan_tiles(*shape, K, 324)
    ri_lo, w_lo, w_hi, wd = eg._compact_basis(rf, rc, tiles, wedge_idx=wedge, T=T)
    g_ri, g_lo, g_hi, g_wd = (mx.array(x) for x in (ri_lo, w_lo, w_hi, wd))
    pick = [(0, 0, 0), (0, 5, 3), (2, 7, 2), (4, 11, 7), (5, 0, -1), (5, 3, -1),
            (len(tiles) - 1, 11, 6), (len(tiles) - 1, 11, -1), (3, 8, 4), (1, 1, 1)]
    for ti, ri, a in pick:
        tid = mx.array(np.array([ti], np.int32))
        w = np.array(eg._pair_weights(g_ri, g_lo, g_hi, g_wd, tid,
                                      mx.array(np.array([ri], np.int32)),
                                      mx.array(np.array([a], np.int32))))[0]
        y0, y1, x0, x1 = tiles[ti]
        basis, wt = ts.tile_weights(rf, wedge, rc, tiles[ti])
        want = (basis[ri].astype(np.float32) if a < 0
                else ts.wedge_weight(basis[ri], wt, a))
        assert w[:y1 - y0, :x1 - x0].tobytes() == want.tobytes(), (ti, ri, a)


def test_take_source_guard_rejects_duplicates():
    src = eg._build_take_src([2, 0, 3], 4, 3)
    assert src.tolist() == [1, 3, 0, 2]       # tile 1 has no pair -> zero row (3)
    with pytest.raises(RuntimeError, match="duplicate"):
        eg._build_take_src([0, 1, 1], 4, 3)


# --------------------------------------------------------------------------
# G-C3 kernel bank identity at the point of consumption
# --------------------------------------------------------------------------

def _capture_gpu_bank(kw, **extra):
    """Capture the K x K float32 kernels handed to the GPU spectra helper."""
    seen, returned = [], []
    real_spec = eg._gpu_kernel_spectra

    def spy(arrays, n, labels=None):
        K = kw["scaled_ksize"]
        for a, lab in zip(arrays, labels):
            assert a.shape == (K, K) and a.dtype == np.float32
            seen.append((lab, _sha(a)))
        out = real_spec(arrays, n, labels)
        returned.append((n, [np.array(a, copy=True) for a in arrays], out))
        return out

    eg._gpu_kernel_spectra = spy
    try:
        _run_gpu(kw, **extra)
    finally:
        eg._gpu_kernel_spectra = real_spec
    return seen, returned


@pytest.mark.parametrize("lens,K,n_tile,hw", [
    ("35", 67, 96, (130, 110)),
    ("50", 67, 96, (110, 130)),
    ("35", 17, 40, (80, 70)),
])
def test_kernel_bank_identity_and_uploaded_spectra(lens, K, n_tile, hw):
    kw = _scene(*hw, seed=11, lens=lens, K=K)
    ref = _capture_reference_bank(kw)
    got, returned = _capture_gpu_bank(kw, _tile_n=n_tile)
    assert len(ref) > 100
    assert [h for _, h in got] == ref
    assert {lab[1] for lab, _ in got} == set(range(12))
    # the complex64 spectra the engine holds on the GPU are the float32 GPU
    # transforms of the zero-padded bank kernels (an independent recompute is
    # tobytes-equal)
    assert len(returned) > 0
    for n, arrays, dev in returned:
        buf = np.zeros((len(arrays), n, n), np.float32)
        for j, k in enumerate(arrays):
            buf[j, :k.shape[0], :k.shape[1]] = k
        assert not buf[:, kw["scaled_ksize"]:, :].any()      # rest is exactly 0
        with mx.stream(mx.gpu):
            want = mx.fft.rfft2(mx.array(buf))
            mx.eval(want)
        assert dev.dtype == mx.complex64
        assert np.array(dev).tobytes() == np.array(want).tobytes()


def test_kernel_spectra_are_float32_transforms_computed_on_gpu():
    rng = np.random.default_rng(0)
    ks = [rng.uniform(0, 1, (17, 17)).astype(np.float32) for _ in range(3)]
    spec = eg._gpu_kernel_spectra(ks, 40)
    assert isinstance(spec, mx.array) and spec.dtype == mx.complex64
    assert tuple(spec.shape) == (3, 40, 21)
    buf = np.zeros((3, 40, 40), np.float32)
    for i, k in enumerate(ks):
        buf[i, :17, :17] = k
    with mx.stream(mx.gpu):
        want = mx.fft.rfft2(mx.array(buf))
        mx.eval(want)
    assert np.array(spec).tobytes() == np.array(want).tobytes()
    # agrees with a float64 transform to float32 round-off
    assert np.allclose(np.array(spec), np.fft.rfft2(buf.astype(np.float64)),
                       rtol=1e-4, atol=1e-4)


# --------------------------------------------------------------------------
# gather against the reference (always runs)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("hw", [(200, 170), (170, 200)])
def test_gather_matches_reference(hw):
    kw = _scene(*hw, seed=4)
    ref5, ref_lines = _run_ref(kw)
    got5, got_lines = _run_gpu(kw, _tile_n=96)
    _, tiles = ts.plan_tiles(hw[0], hw[1], 67, 96)
    assert len(tiles) >= 9
    _assert_tier_c(ref5, got5, f"gather {hw}")
    assert ref_lines == got_lines                       # stdout and progress
    assert float(ref5[1].max()) > 0.1
    assert got5[0].dtype == np.float32 and got5[0].shape == ref5[0].shape


@pytest.mark.parametrize("hw,K,tile_n", [((48, 40), 17, 40), ((40, 48), 17, 40),
                                         ((23, 31), 9, 24), ((90, 70), 31, 64)])
def test_weight_sums_and_acc_alpha_equal_reference_small_frames(hw, K, tile_n):
    """Tiny frames: the full-frame 0.01 skips fire, is_delta differs per
    (d, ri); weight sums are the closed form of the sequential sum."""
    kw = _scene(*hw, seed=17, K=K)
    ref5, ref_lines = _run_ref(kw)
    got5, got_lines = _run_gpu(kw, _tile_n=tile_n)
    for i in (2, 3, 4):
        assert np.array_equal(ref5[i], got5[i]), i
    assert ref_lines == got_lines
    _assert_tier_c(ref5, got5, f"small {hw} K={K}")


def test_gather_no_bright_pixels():
    kw = _scene(120, 100, seed=2, K=31, n_hot=0)
    kw["img_bright_padded"] = np.zeros_like(kw["img_bright_padded"])
    kw["img_diffuse_padded"] = kw["img_padded"].copy()
    ref5, _ = _run_ref(kw)
    got5, _ = _run_gpu(kw, _tile_n=64)
    assert not got5[1].any()
    _assert_tier_c(ref5, got5, "no bright")


def test_slab_batch_size_does_not_change_output():
    kw = _scene(100, 90, seed=5, K=31)
    base, _ = _run_gpu(kw, _tile_n=64)
    for b in (1, 3, 11):
        got, _ = _run_gpu(dict(kw, slab_batch_size=b), _tile_n=64)
        _same5(base, got, f"slab_batch_size={b}")


def test_slab_batch_size_validation_matches_reference_text():
    kw = _scene(60, 60, seed=2, K=17)
    with pytest.raises(ValueError, match="slab_batch_size must be >= 1"):
        _run_gpu(dict(kw, slab_batch_size=0), _tile_n=40)
    with pytest.raises(ValueError, match="exceeds hard upper bound"):
        _run_gpu(dict(kw, slab_batch_size=1000), _tile_n=40)


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

    def gpu(k):
        eg.gpu_layered_gather(**dict(k, fft_state=None, on_progress=None,
                                     emit_log=False, _tile_n=40))

    a, b = run(ref), run(gpu)
    assert a and a == b


def test_unsupported_configurations_are_not_available():
    kw = _scene(40, 40, seed=1, K=17)
    with pytest.raises(ValueError, match="not available"):
        _run_gpu(dict(kw, scaled_ksize=256, pad=128))
    with pytest.raises(ValueError, match="not available"):
        _run_gpu(dict(kw, pad=5), _tile_n=40)
    with pytest.raises(ValueError, match="_tile_n"):
        _run_gpu(kw, _tile_n=17)
    with pytest.raises(ValueError, match="not available"):
        _run_gpu(dict(kw, r_centers=np.array([0.5], np.float32)), _tile_n=40)


# --------------------------------------------------------------------------
# G-C3b geometric oracles
# --------------------------------------------------------------------------

def _halo_case(K, N, H, W, seed=1):
    rng = np.random.default_rng(seed)
    pad = K // 2
    img = rng.uniform(0.0, 6.8, (H, W, 3)).astype(np.float32)
    ip = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
    ker = rng.uniform(0.2, 1.0, (K, K)).astype(np.float32)
    ker /= ker.sum()
    got = eg.gpu_tile_convolve_full(ip, ker, H, W, N=N)
    truth = np.empty((H, W, 3), np.float64)
    for c in range(3):
        full = fftconvolve(ip[:, :, c].astype(np.float64), ker.astype(np.float64),
                           mode="full")
        truth[:, :, c] = full[K - 1:K - 1 + H, K - 1:K - 1 + W]
    e = np.abs(got.astype(np.float64) - truth) / np.maximum(truth, 1.0)
    return e, truth, got


@pytest.mark.parametrize("K,N,H,W", [
    (67, 96, 100, 115), (67, 96, 115, 100),
    (66, 96, 110, 100), (66, 96, 100, 110),
    (41, 128, 270, 300), (41, 128, 300, 270),
    (67, None, 600, 700), (67, None, 700, 600),
])
def test_halo_oracle(K, N, H, W):
    N_eff = N or ts.tile_n_for_ksize(K)
    T, tiles = ts.plan_tiles(H, W, K, N_eff)
    assert len({t[0] for t in tiles}) >= 3 and len({t[2] for t in tiles}) >= 3
    assert H % T and W % T
    e, truth, got = _halo_case(K, N, H, W)
    print(f"halo K={K} N={N_eff} {H}x{W}: max e = {float(e.max()):.3g}")
    assert float(e.max()) <= 2e-6, float(e.max())
    slip = np.abs(got[1:].astype(np.float64) - truth[:-1]) / np.maximum(truth[:-1], 1.0)
    assert float(slip.max()) > 1e-3


@pytest.mark.slow
def test_halo_oracle_k183_n512():
    e, _, _ = _halo_case(183, None, 700, 760)
    print(f"halo K=183 N=512 700x760: max e = {float(e.max()):.3g}")
    assert float(e.max()) <= 2e-6, float(e.max())


def _unit_kernels(tensor, K, d, picks):
    """Real bank units (rotated, renormalised) of slab ``d``."""
    nr = int(tensor["kernels"].shape[0])
    ones = np.ones(nr, np.float32)
    flags = np.ones((nr, cv.N_ANG + 1), bool)
    kH, kW = tensor["kernels"].shape[2:]
    units = ts.slab_bank(
        d, nr=nr, kernels=tensor["kernels"], defocus_ratios=tensor["defocus_ratios"],
        kH=kH, kW=kW, pix_scale=1.0 if K == kH else K / kH, scaled_ksize=K,
        bright_stamp_gain=None, gmax_ri=ones,
        gmax_ria=np.ones((nr, cv.N_ANG), np.float32), bank_flags=flags)
    out = []
    for ri, a in picks:
        u = next((x for x in units if x.ri == ri and (x.a == a or x.a is None)), None)
        assert u is not None and u.kernel is not None, (ri, a)
        out.append((ri, a, u.kernel))
    return out


def _unit_parity(tensor, K, hw, d=4, seed=7):
    rng = np.random.default_rng(seed)
    H, W = hw
    pad = K // 2
    img = rng.uniform(0.0, 1.0, (H, W, 3)).astype(np.float32)
    img[H // 3:H // 3 + 3, W // 4:W // 4 + 3] = 30.0       # a bright feature
    ip = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
    worst = 0.0
    picks = [(0, 1), (0, 5), (6, 1), (6, 5), (11, 1), (11, 5)]
    for ri, a, ker in _unit_kernels(tensor, K, d, picks):
        gpu = eg.gpu_tile_convolve_full(ip, ker, H, W)
        cpu_tile = et.tile_convolve_full(ip, ker, H, W, threads=2)
        truth = np.empty((H, W, 3), np.float64)
        for c in range(3):
            full = fftconvolve(ip[:, :, c].astype(np.float64), ker.astype(np.float64),
                               mode="full")
            truth[:, :, c] = full[K - 1:K - 1 + H, K - 1:K - 1 + W]
        e_u = float(np.abs(gpu.astype(np.float64) - cpu_tile).max() / np.abs(cpu_tile).max())
        e_f = float(np.abs(gpu.astype(np.float64) - truth).max() / np.abs(truth).max())
        print(f"e_u K={K} ri={ri} a={a}: vs scipy tile {e_u:.3g}, vs full-frame f64 {e_f:.3g}")
        worst = max(worst, e_u, e_f)
    return worst


@pytest.mark.parametrize("lens,K", [("35", 67), ("50", 67)])
def test_per_unit_parity_k67(lens, K):
    assert _unit_parity(_tensor(lens), K, (330, 300)) <= 1e-6


@pytest.mark.slow
@pytest.mark.parametrize("lens", ["35", "50"])
def test_per_unit_parity_k183_native_kernels(lens):
    t = dict(_tensor(lens))
    t.pop("corpus_ref_long_edge", None)
    assert _unit_parity(t, 183, (420, 400)) <= 1e-6


@pytest.mark.slow
def test_per_unit_parity_k255_custom_tensor():
    t = make_synthetic_tensor(12, 5, 255, 255)
    t["object_distances_m"] = np.geomspace(0.7, 20.0, 5)
    assert _unit_parity(t, 255, (400, 400), d=2) <= 1e-6


def _seam_positions(T, K, H, W):
    half = K // 2
    pos = []
    for k in (1, 2):
        for dy in (-half, 0, half):
            y = k * T + dy
            if 4 <= y < H - 4:
                pos.append((y, min(W - 5, max(4, T + 3))))
        for dx in (-half, 0, half):
            x = k * T + dx
            if 4 <= x < W - 4:
                pos.append((min(H - 5, max(4, T + 5)), x))
    pos.append((H - 3, W - 3))                  # last partial row and column
    return pos


@pytest.mark.parametrize("K,tile_n,hw", [
    (67, 96, (100, 115)),          # T=30: 4x4 tiles, seams every 30 px
    (67, 96, (91, 91)),            # H mod T = 1 slivers (T=30 -> 91 = 3*30 + 1)
    (31, 64, (100, 98)),           # T=34; 98 mod 34 = 30
    (31, 64, (103, 69)),           # H mod T = 1 (103 = 3*34 + 1), W mod T = 1
])
def test_seam_impulses_and_slivers_through_the_gather(K, tile_n, hw):
    h, w = hw
    T = tile_n - K + 1
    pos = _seam_positions(T, K, h, w)
    kw = _impulse_scene(h, w, K, pos)
    ref5, _ = _run_ref(kw)
    got5, _ = _run_gpu(kw, _tile_n=tile_n)
    if h % T == 1 or w % T == 1:
        assert (h % T == 1) or (w % T == 1)
    _assert_tier_c(ref5, got5, f"seams K={K} {hw}")
    _assert_impulses(ref5, got5, pos, K, f"seams K={K} {hw}")


@pytest.mark.slow
@pytest.mark.parametrize("hw", [(520, 700), (700, 520)])
def test_seam_impulses_production_n324(hw):
    h, w = hw
    K = 67
    T = 324 - K + 1
    pos = _seam_positions(T, K, h, w)
    kw = _impulse_scene(h, w, K, pos)
    ref5, _ = _run_ref(kw)
    got5, _ = _run_gpu(kw)
    _assert_tier_c(ref5, got5, f"seams N=324 {hw}")
    _assert_impulses(ref5, got5, pos, K, f"seams N=324 {hw}")


def test_halo_band_impulse():
    K, Nn = 67, 96
    h, w = 100, 115
    od = _tensor("35")["object_distances_m"]
    img = np.full((h, w, 3), 0.05, dtype=np.float32)
    iy, ix = 63, 45
    img[iy, ix] = 5.0
    depth = np.full((h, w), od[4], dtype=np.float32)
    kw = _gather_kwargs(img, depth, K=K)
    ref5, _ = _run_ref(kw)
    got5, _ = _run_gpu(kw, _tile_n=Nn)
    assert float(ref5[1][30:60, 30:60].max()) > 1e-4
    _assert_tier_c(ref5, got5, "halo-band impulse")


def test_fft_sides_are_production_n_never_above_4096():
    sides = set()
    real_r, real_i = mx.fft.rfft2, mx.fft.irfft2

    def rs(a, *args, **k):
        sides.add(tuple(a.shape[-2:]))
        return real_r(a, *args, **k)

    def is_(a, *args, **k):
        s = k.get("s")
        sides.add(tuple(s) if s is not None else tuple(a.shape[-2:]))
        return real_i(a, *args, **k)

    mx.fft.rfft2, mx.fft.irfft2 = rs, is_
    try:
        kw = _scene(120, 100, seed=3, K=67)
        _run_gpu(kw)                                     # production N = 324
    finally:
        mx.fft.rfft2, mx.fft.irfft2 = real_r, real_i
    assert sides == {(324, 324)}, sides


@pytest.mark.slow
def test_k183_production_n512_small_frame():
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
    sides = set()
    real_i = mx.fft.irfft2

    def is_(a, *args, **k):
        sides.add(tuple(k["s"]))
        return real_i(a, *args, **k)

    mx.fft.irfft2 = is_
    try:
        got5, _ = _run_gpu(kw)                           # production N = 512, host accumulators
    finally:
        mx.fft.irfft2 = real_i
    assert sides == {(512, 512)}
    _assert_tier_c(ref5, got5, "K=183 N=512")


def test_chart_matches_reference_field_character():
    img, depth, centers = _chart(260, 220, 1.0)
    kw = _gather_kwargs(img, depth, K=67, thresh=0.5, bright_stamp_gain=None)
    ref5, _ = _run_ref(kw)
    got5, _ = _run_gpu(kw, _tile_n=96)
    _assert_tier_c(ref5, got5, "chart")
    _assert_impulses(ref5, got5, centers, 67, "chart")


# --------------------------------------------------------------------------
# G-C7: holes, determinism, knobs, device, hygiene, cancel, trace
# --------------------------------------------------------------------------

def test_holes_donors_fallback_and_acc_alpha():
    kw = _holes_scene()
    ref5, ref_lines = _run_ref(kw)
    spy = {"n": 0}
    real = et._update_best_donor

    def counting(*a, **k):
        spy["n"] += 1
        return real(*a, **k)

    et._update_best_donor = counting
    try:
        got5, got_lines = _run_gpu(kw, _tile_n=64)
    finally:
        et._update_best_donor = real
    assert spy["n"] > 0
    assert "holes=" in ref_lines[-1] and ref_lines == got_lines
    assert np.array_equal(ref5[4], got5[4])
    assert int((kw["slab_w"].sum(axis=0) < 0.02).sum()) > 0
    A = (slice(10, 20), slice(10, 20))
    assert np.array_equal(got5[0][A], kw["img_lin"][A])          # no-donor fallback
    assert not got5[1][A].any()
    B = (slice(40, 50), slice(40, 50))
    assert not np.array_equal(got5[0][B], kw["img_lin"][B])      # donor used
    _assert_tier_c(ref5, got5, "holes")


def test_hole_prediction_mismatch_raises_internal_error():
    kw = _holes_scene()
    real = ts.predict_holes
    ts.predict_holes = lambda *a, **k: False
    try:
        with pytest.raises(RuntimeError, match="internal error"):
            _run_gpu(kw, _tile_n=64)
    finally:
        ts.predict_holes = real


def test_determinism_three_repeats_small():
    kw = _scene(130, 110, seed=6, K=31)
    base = None
    for _ in range(3):
        got5, _ = _run_gpu(kw, _tile_n=64)
        assert len(ts.plan_tiles(130, 110, 31, 64)[1]) >= 9
        if base is None:
            base = got5
        else:
            _same5(base, got5, "repeat")


@pytest.mark.slow
@pytest.mark.parametrize("K,hw,tensor_noscale", [(67, (620, 700), False),
                                                 (183, (700, 760), True)])
def test_determinism_production_n(K, hw, tensor_noscale):
    t = dict(_tensor("35"))
    if tensor_noscale:
        t.pop("corpus_ref_long_edge", None)
    rng = np.random.default_rng(1)
    h, w = hw
    img = rng.uniform(0.0, 0.8, (h, w, 3)).astype(np.float32)
    for _ in range(10):
        y, x = int(rng.integers(5, h - 5)), int(rng.integers(5, w - 5))
        img[y - 1:y + 2, x - 1:x + 2] = 3.0
    kw = _gather_kwargs(img, _depth(h, w, t["object_distances_m"]), K=K, tensor=t,
                        bright_stamp_gain=2.0)
    N = ts.tile_n_for_ksize(K)
    T, tiles = ts.plan_tiles(h, w, K, N)
    assert len({a[0] for a in tiles}) >= 3 and len({a[2] for a in tiles}) >= 3
    base = None
    for _ in range(3):
        got5, _ = _run_gpu(kw)
        if base is None:
            base = got5
        else:
            _same5(base, got5, f"K={K} repeat")


def test_host_vs_gpu_accumulators_and_cache_limits_are_output_neutral():
    kw = _scene(110, 100, seed=7, K=31)
    base, _ = _run_gpu(kw, _tile_n=64, _host_acc=False)
    for extra in (dict(_host_acc=True), dict(_cache_limit=0),
                  dict(_cache_limit=1 << 28), dict(_cache_limit=1 << 40)):
        got5, _ = _run_gpu(kw, _tile_n=64, **extra)
        assert all(np.array_equal(base[i], got5[i]) for i in range(5)), extra


def test_profile_split_is_output_neutral():
    kw = _scene(110, 100, seed=8, K=31)
    base, _ = _run_gpu(kw, _tile_n=64)
    eg._PROFILE_SPLIT = True
    try:
        got5, _ = _run_gpu(kw, _tile_n=64)
    finally:
        eg._PROFILE_SPLIT = False
    _same5(base, got5, "profile split")
    assert "waves_conv" in eg._TIMES and "waves_add" in eg._TIMES


def test_last_run_record_and_default_device_unchanged():
    kw = _scene(100, 90, seed=9, K=31)
    dev = mx.default_device()
    eg._LAST_RUN.clear()
    _run_gpu(kw, _tile_n=64)
    assert mx.default_device() == dev
    assert eg._LAST_RUN["device"] == "gpu"
    assert eg._LAST_RUN["waves"] > 0 and eg._LAST_RUN["peak_bytes"] > 0
    assert eg._LAST_RUN["cache_limit"] == 1 << 30


def test_execution_proof_cpu_executors_patched_to_raise():
    kw = _scene(100, 90, seed=9, K=31)
    ref5, _ = _run_ref(kw)

    def boom(*a, **k):
        raise AssertionError("a CPU executor ran inside the gpu engine")

    saved = (et.tiled_layered_gather, et._convolve_chunk, et.tile_convolve_full,
             et._tile_job, cv._apply_kernel)
    et.tiled_layered_gather = et._convolve_chunk = et.tile_convolve_full = boom
    et._tile_job = cv._apply_kernel = boom
    eg._LAST_RUN.clear()
    try:
        got5, _ = _run_gpu(kw, _tile_n=64)
    finally:
        (et.tiled_layered_gather, et._convolve_chunk, et.tile_convolve_full,
         et._tile_job, cv._apply_kernel) = saved
    assert eg._LAST_RUN["device"] == "gpu" and eg._LAST_RUN["waves"] > 0
    _assert_tier_c(ref5, got5, "execution proof")


def test_trace_gpu_engine_ran_reference_apply_kernel_did_not():
    kw = _scene(60, 60, seed=2, K=17)
    calls = {"spec": 0, "waves": 0, "apply": 0}
    real_s, real_w, real_a = eg._gpu_kernel_spectra, eg._run_waves, cv._apply_kernel

    def s(*a, **k):
        calls["spec"] += 1
        return real_s(*a, **k)

    def w(*a, **k):
        calls["waves"] += 1
        return real_w(*a, **k)

    def ap(*a, **k):
        calls["apply"] += 1
        return real_a(*a, **k)

    eg._gpu_kernel_spectra, eg._run_waves, cv._apply_kernel = s, w, ap
    try:
        _run_gpu(kw, _tile_n=40)
    finally:
        eg._gpu_kernel_spectra, eg._run_waves, cv._apply_kernel = real_s, real_w, real_a
    assert calls["spec"] > 0 and calls["waves"] > 0 and calls["apply"] == 0


def test_memory_hygiene_three_renders():
    kw = _scene(110, 100, seed=7, K=31)
    mx.clear_cache()
    limit0 = mx.set_cache_limit(1 << 33)
    mx.set_cache_limit(limit0)
    after = []
    for _ in range(3):
        entry = mx.get_active_memory()
        _run_gpu(kw, _tile_n=64)
        assert mx.get_cache_memory() == 0
        assert mx.get_active_memory() <= entry + (64 << 20)
        after.append(mx.get_active_memory())
        assert mx.set_cache_limit(limit0) == limit0          # restored to the pre-render value
    assert after[2] <= after[0] + (64 << 20)


def test_cancel_preset_raises_before_any_mlx_call_or_spectrum():
    kw = _scene(60, 60, seed=2, K=17)
    ev = threading.Event()
    ev.set()
    calls = {"spec": 0, "waves": 0, "win": 0}
    real_s, real_w, real_win = eg._gpu_kernel_spectra, eg._run_waves, eg._windows

    def s(*a, **k):
        calls["spec"] += 1
        return real_s(*a, **k)

    def w(*a, **k):
        calls["waves"] += 1
        return real_w(*a, **k)

    def win(*a, **k):
        calls["win"] += 1
        return real_win(*a, **k)

    limit0 = mx.set_cache_limit(1 << 33)
    eg._gpu_kernel_spectra, eg._run_waves, eg._windows = s, w, win
    try:
        with pytest.raises(CancelledError, match=r"^render cancelled at slab 0/11$"):
            _run_gpu(dict(kw, cancel=ev), _tile_n=40)
    finally:
        eg._gpu_kernel_spectra, eg._run_waves, eg._windows = real_s, real_w, real_win
    assert mx.set_cache_limit(limit0) == 1 << 33       # never touched
    assert calls == {"spec": 0, "waves": 0, "win": 0}


def test_cancel_mid_slab_uses_gpu_text_pool_down_cache_restored():
    kw = _scene(60, 60, seed=2, K=17)
    ev = threading.Event()
    real = eg._gpu_kernel_spectra

    def spy(*a, **k):
        out = real(*a, **k)
        ev.set()
        return out

    before = {t.ident for t in threading.enumerate()}
    limit0 = mx.set_cache_limit(1 << 33)
    entry = mx.get_active_memory()
    eg._gpu_kernel_spectra = spy
    try:
        with pytest.raises(CancelledError,
                           match=r"^render cancelled inside slab \d+ \(gpu engine\)$") as ei:
            _run_gpu(dict(kw, cancel=ev), _tile_n=40, threads=4)
    finally:
        eg._gpu_kernel_spectra = real
    assert mx.set_cache_limit(limit0) == 1 << 33        # restored by the finally
    del ei                                               # the traceback owns the frame's arrays
    gc.collect()
    assert mx.get_active_memory() <= entry + (64 << 20)
    leftover = [t for t in threading.enumerate()
                if t.ident not in before and t.name.startswith("ThreadPoolExecutor")]
    assert not leftover


def test_leak_gpu_render_then_reference_equals_fresh_process_reference():
    code = (
        "import hashlib, sys, warnings\n"
        "import numpy as np\n"
        "from tests.unit.test_s33_tiled_core import _scene, _run_ref\n"
        "kw = _scene(60, 60, seed=2, K=17)\n"
        "r, _ = _run_ref(kw)\n"
        "h = hashlib.sha256(b''.join(a.tobytes() for a in r)).hexdigest()\n"
        "print('H', h)\n"
    )
    env = dict(os.environ, PYTHONPATH=str(REPO))
    p = subprocess.run([sys.executable, "-c", code], cwd=REPO, env=env,
                       capture_output=True, text=True, timeout=300)
    assert p.returncode == 0, p.stderr
    fresh = p.stdout.split("H ")[1].strip()
    kw = _scene(60, 60, seed=2, K=17)
    _run_gpu(kw, _tile_n=40)                              # GPU first, same process
    r, _ = _run_ref(kw)
    assert hashlib.sha256(b"".join(a.tobytes() for a in r)).hexdigest() == fresh
