"""S33 WP5: fixed-cost speedups are byte-exact.

Each test compares the shipped fast form with an embedded copy of the code it
replaced (the pre-WP5 loop / mask / serial form), on synthetic data that
exercises the edge cases (empty wedges, partial support, ties, Inf, tiny bins).
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.ndimage import uniform_filter, zoom

import summicron.render.convolve as cv
from summicron.depth import guided
from summicron.io import tone_lut
from summicron.render import _tile_schedule as ts

# ---------------------------------------------------------------- planner


def _old_stats_row(rf_map, wedge_idx, r_centers, tile, bright_padded, K):
    basis, wt = ts.tile_weights(rf_map, wedge_idx, r_centers, tile)
    nr = basis.shape[0]
    bmax = np.empty(nr, np.float32)
    bpos = np.empty(nr, bool)
    wmax = np.zeros((nr, cv.N_ANG), np.float32)
    wpos = np.zeros((nr, cv.N_ANG), bool)
    for ri in range(nr):
        w = basis[ri].astype(np.float32)
        bmax[ri] = w.max()
        bpos[ri] = bool((w > 0).any())
        for a in range(cv.N_ANG):
            w_ang = ts.wedge_weight(basis[ri], wt, a)
            wmax[ri, a] = w_ang.max()
            wpos[ri, a] = bool((w_ang > 0).any())
    py0, py1, px0, px1 = ts.input_window(tile, K)
    has_b = bool((bright_padded[py0:py1, px0:px1] != 0).any())
    return bmax, bpos, wmax, wpos, has_b


def _old_joint_row(stats, ti, sw_tile, rf_map, wedge_idx, r_centers, tile):
    nr = stats.bpos.shape[1]
    flags = np.zeros((nr, cv.N_ANG + 1), bool)
    sw_nz = sw_tile != 0
    if not sw_nz.any():
        return flags
    if sw_nz.all():
        flags[:, : cv.N_ANG] = stats.wpos[ti]
        flags[:, cv.N_ANG] = stats.bpos[ti]
        return flags
    basis, wt = ts.tile_weights(rf_map, wedge_idx, r_centers, tile)
    for ri in range(nr):
        if stats.bpos[ti, ri]:
            flags[ri, cv.N_ANG] = bool(((basis[ri] > 0) & sw_nz).any())
        for a in range(cv.N_ANG):
            if stats.wpos[ti, ri, a]:
                w_ang = ts.wedge_weight(basis[ri], wt, a)
                flags[ri, a] = bool(((w_ang > 0) & sw_nz).any())
    return flags


def _scene(h, w, nr, seed, *, few_wedges=False):
    rng = np.random.default_rng(seed)
    ys = np.arange(h, dtype=np.float32) - (h - 1) / 2
    xs = np.arange(w, dtype=np.float32) - (w - 1) / 2
    rf = (np.hypot(ys[:, None], xs[None, :]) / np.hypot(h / 2, w / 2)).astype(np.float32)
    theta = np.arctan2(ys[:, None], xs[None, :])
    wedge = ((theta + np.pi) / (2 * np.pi) * cv.N_ANG).astype(np.int32) % cv.N_ANG
    if few_wedges:
        wedge = np.where(wedge > 2, 1, wedge).astype(np.int32)
    rc = np.sort(rng.random(nr)).astype(np.float64) * 0.9 + 0.05
    return rf, wedge, rc


@pytest.mark.parametrize("nr,few", [(12, False), (12, True), (3, False), (2, False), (1, False)])
def test_planner_rows_equal_loop(nr, few):
    h, w, K = 97, 131, 9
    rf, wedge, rc = _scene(h, w, nr, 3, few_wedges=few)
    rng = np.random.default_rng(4)
    bright = np.zeros((h + K - 1, w + K - 1, 3), np.float32)
    bright[rng.integers(0, h, 6), rng.integers(0, w, 6)] = -1.5  # negatives count
    for T in (32, 50):
        _, tiles = ts.plan_tiles(h, w, K, T)
        rows_new, rows_old = [], []
        for t in tiles:
            n = ts.tile_stats_row(rf, wedge, rc, t, bright, K)
            o = _old_stats_row(rf, wedge, rc, t, bright, K)
            for x, y in zip(n, o):
                assert np.asarray(x).dtype == np.asarray(y).dtype
                assert np.array_equal(x, y)
            rows_new.append(n)
            rows_old.append(o)
        st = ts.assemble_stats(rows_new)
        for ti, t in enumerate(tiles):
            y0, y1, x0, x1 = t
            for sw in (
                np.zeros((y1 - y0, x1 - x0), np.float32),
                np.ones((y1 - y0, x1 - x0), np.float32),
                (np.random.default_rng(ti).random((y1 - y0, x1 - x0)) > 0.9).astype(np.float32),
                np.eye(y1 - y0, x1 - x0, dtype=np.float32),
            ):
                a = ts.joint_support_row(st, ti, sw, rf, wedge, rc, t)
                b = _old_joint_row(st, ti, sw, rf, wedge, rc, t)
                assert a.dtype == b.dtype and np.array_equal(a, b)


@pytest.mark.parametrize("nr", [2, 5, 12])
def test_compact_planes_equal_shared_basis(nr):
    rf, wedge, rc = _scene(61, 73, nr, 11)
    rf[0, :5] = 0.0          # below the first centre
    rf[-1, :5] = 2.0         # beyond the last centre
    rf[10, 10] = np.float32(rc[1])  # exactly on a node
    basis = cv._radial_basis_weights(rf, rc)
    ri_lo, lo, hi = ts._tile_compact(rf, rc)
    want = np.zeros_like(basis)
    yy, xx = np.indices(rf.shape)
    want[ri_lo, yy, xx] += lo
    want[ri_lo + 1, yy, xx] += hi
    assert np.array_equal(basis, want)


def test_planner_fallback_on_out_of_range_wedge():
    rf, wedge, rc = _scene(40, 40, 12, 5)
    wedge = wedge.copy()
    wedge[3, 3] = cv.N_ANG + 2     # outside 0..N_ANG-1: full-basis path
    K = 7
    bright = np.zeros((40 + K - 1, 40 + K - 1, 3), np.float32)
    tile = (0, 40, 0, 40)
    for x, y in zip(ts.tile_stats_row(rf, wedge, rc, tile, bright, K),
                    _old_stats_row(rf, wedge, rc, tile, bright, K)):
        assert np.array_equal(x, y)


# ------------------------------------------------------------- tone fit


def _old_fit_lut(hdr, sdr):
    L = tone_lut.LUMA_W
    hmax = float(hdr.max())
    hy = hdr.astype(np.float32) @ L
    sy = sdr.astype(np.float32) @ L
    edges = np.geomspace(1e-3, hmax * 1.01, 65)
    idx = np.digitize(hy, edges)
    xs, ys = [0.0], [0.0]
    for i in range(1, len(edges)):
        s = (idx == i) & (sy > 0.02)
        if s.sum() > 200:
            xs.append(float(np.sqrt(edges[i - 1] * edges[i])))
            ys.append(float(np.median(sy[s])))
    return np.array(xs), np.maximum.accumulate(np.array(ys))


def _old_fit_sat(hdr, sdr):
    L = tone_lut.LUMA_W
    hmax = float(hdr.max())
    hy = hdr.astype(np.float32) @ L
    sy = sdr.astype(np.float32) @ L
    edges = np.geomspace(1e-3, hmax * 1.01, 65)
    idx = np.digitize(hy, edges)
    kx, kk = [], []
    for i in range(1, len(edges)):
        s = (idx == i) & (sy > 0.02)
        if s.sum() <= 200:
            continue
        r = hdr[s].astype(np.float64) / np.maximum(hy[s], 1e-6).astype(np.float64)[:, None] - 1.0
        rp = sdr[s].astype(np.float64) / sy[s].astype(np.float64)[:, None] - 1.0
        den = float((r * r).sum())
        if den <= 1e-9:
            continue
        kx.append(float(np.sqrt(edges[i - 1] * edges[i])))
        kk.append(float(np.clip((r * rp).sum() / den, 0.0, 1.2)))
    return np.array(kx), np.array(kk)


@pytest.mark.parametrize("hm", [3.0, 40.0, 1.5])
def test_tone_fit_equals_mask_form(hm):
    rng = np.random.default_rng(int(hm * 10))
    h = (rng.random((180, 220, 3)) ** 3 * hm).astype(np.float32)
    s = np.clip(h * 0.6 + rng.random(h.shape) * 0.1, 0, 1).astype(np.float32)
    xs, ys, kx, kk = tone_lut.fit_input_tone(h, s)
    for got, want in ((xs, _old_fit_lut(h, s)[0]), (ys, _old_fit_lut(h, s)[1]),
                      (kx, _old_fit_sat(h, s)[0]), (kk, _old_fit_sat(h, s)[1])):
        assert got.dtype == want.dtype and np.array_equal(got, want)
    # the public single fits agree with the tone wrapper (no shared state)
    assert np.array_equal(tone_lut.fit_input_lut(h, s)[1], ys)
    assert np.array_equal(tone_lut.fit_input_sat(h, s)[1], kk)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_tone_fit_identity_paths_unchanged():
    h = np.full((30, 30, 3), 0.5, np.float32)       # no headroom -> identity sat
    s = h.copy()
    out = tone_lut.fit_input_tone(h, s)
    assert np.array_equal(out[2], np.array([1e-3, 1.0])) and np.array_equal(out[3], [1.0, 1.0])
    bad = h.copy()
    bad[0, 0, 0] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        tone_lut.fit_input_tone(bad, s)


# ------------------------------------------------------------- downres


def test_downres_hdr_threaded_equals_3d_zoom():
    rng = np.random.default_rng(7)
    im = (rng.random((300, 450, 3)) * 4.0).astype(np.float32)
    got = cv.downres_array(im, 200)
    h, w = im.shape[:2]
    nh, nw = int(round(h * 200 / 450)), 200
    want = zoom(im, (nh / h, nw / w, 1.0), order=1).astype(np.float32)
    assert got.dtype == want.dtype and got.shape == want.shape and np.array_equal(got, want)


def test_downres_hdr_nonfinite_takes_original_path():
    rng = np.random.default_rng(8)
    im = (rng.random((120, 160, 3)) * 4.0).astype(np.float32)
    im[5, 5, 1] = np.inf
    got = cv.downres_array(im, 80)
    h, w = im.shape[:2]
    want = zoom(im, (60 / h, 80 / w, 1.0), order=1).astype(np.float32)
    assert np.array_equal(got, want, equal_nan=True)


# --------------------------------------------------------- guided filter


def _old_guided_filter(guide, src, r, eps):
    box = lambda x: uniform_filter(x, size=2 * r + 1, mode="nearest")  # noqa: E731
    I = np.asarray(guide, dtype=np.float64)  # noqa: E741
    p = np.asarray(src, dtype=np.float64)
    mean_I, mean_p = box(I), box(p)
    cov_Ip = box(I * p) - mean_I * mean_p
    var_I = box(I * I) - mean_I * mean_I
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    return box(a) * I + box(b)


@pytest.mark.parametrize("r,eps", [(4, 1e-4), (1, 1e-3), (16, 1e-3)])
def test_guided_filter_threaded_equals_serial(r, eps):
    rng = np.random.default_rng(r)
    g = rng.random((150, 170)).astype(np.float32)
    d = (1.0 / (0.5 + rng.random((150, 170)))).astype(np.float32)
    got = guided.guided_filter(g, d, r, eps)
    assert got.dtype == np.float64 and np.array_equal(got, _old_guided_filter(g, d, r, eps))
