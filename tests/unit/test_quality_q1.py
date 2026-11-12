"""Q-a R-eval-gates: synthetic tests for ``summicron.render._quality``.

FAST, synthetic only (no real renders).
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy import ndimage

pytest.importorskip("skimage")

from summicron.render import _quality as Q  # noqa: E402

N = 160


def _scene(seed: int = 0):
    """ref: piecewise-constant blobs + mild noise (sharp edges); alpha: radial
    ramp with a wide transition band; base: ref blurred outside the snap."""
    rng = np.random.default_rng(seed)
    f = ndimage.gaussian_filter(rng.standard_normal((N, N)), 3.0)
    f = f / f.std()
    ref1 = 0.25 + 0.15 * (f > 0) + 0.15 * (f > 1.0) + 0.01 * rng.standard_normal((N, N))
    ref = np.repeat(ref1[..., None], 3, axis=2).astype(np.float32)
    ref[..., 0] *= 1.1
    ref[..., 2] *= 0.9
    yy, xx = np.mgrid[:N, :N]
    r = np.hypot(yy - N / 2, xx - N / 2)
    alpha = np.clip((r - 15.0) / (60.0 - 15.0), 0.0, 1.0)
    alpha = (1.0 - alpha).astype(np.float32)  # 1 centre -> 0 outside
    blurred = np.stack(
        [ndimage.gaussian_filter(ref[..., c], 2.0) for c in range(3)], axis=2
    )
    base = (alpha[..., None] * ref + (1 - alpha[..., None]) * blurred).astype(np.float32)
    return ref, alpha, base


def _win():
    return Q.focus_window((N, N), (0.5, 0.5), side_2048=2048)


def _blur(img, sigma, chans=(0, 1, 2)):
    out = img.copy()
    for c in chans:
        out[..., c] = ndimage.gaussian_filter(img[..., c], sigma)
    return out


def _apply_band(base, new, alpha):
    band = (alpha > Q.ALPHA_LO) & (alpha < Q.ALPHA_HI)
    return np.where(band[..., None], new, base).astype(np.float32)


def test_a_identity():
    ref, alpha, _ = _scene()
    m = Q.window_metrics(ref, ref, alpha, _win())
    assert m["ger_band"] == pytest.approx(1.0)
    assert m["ssim_band"] == pytest.approx(1.0)
    assert m["ovfrac"] == 0.0
    assert m["n_px"] > Q.MIN_PX


def test_b_band_blur_is_worse():
    ref, alpha, base = _scene()
    var = _apply_band(base, _blur(base, 1.0), alpha)
    mb = Q.window_metrics(base, ref, alpha, _win())
    mv = Q.window_metrics(var, ref, alpha, _win(), mask_out=base)
    p = Q.paired(mb, mv, "band")
    assert p["d_ger"] < 0 and p["d_ssim"] < 0
    assert not p["gates"]["Q1-B"]
    assert not Q.paired(mb, mv, "nonband")["gates"]["Q1-N"]
    # Identity passes the non-band lever.
    assert Q.paired(mb, mb, "nonband")["gates"]["Q1-N"]


def test_c_band_sharpen_toward_ref_is_better():
    ref, alpha, base = _scene()
    var = _apply_band(base, base + 0.5 * (ref - base), alpha)
    mb = Q.window_metrics(base, ref, alpha, _win())
    mv = Q.window_metrics(var, ref, alpha, _win(), mask_out=base)
    p = Q.paired(mb, mv, "band")
    assert p["d_ger"] > 0 and p["d_ssim"] > 0
    assert p["gates"]["Q1-B"]


def test_d_ringing_vs_blur_ovfrac():
    ref, alpha, _ = _scene()
    base = ref.copy()  # identical outside; isolate the lever
    usm = _apply_band(
        base, base + 1.0 * (base - _blur(base, 1.5)), alpha
    )
    blur = _apply_band(base, _blur(base, 1.0), alpha)
    mb = Q.window_metrics(base, ref, alpha, _win())
    d_usm = Q.paired(mb, Q.window_metrics(usm, ref, alpha, _win(), mask_out=base), "band")
    d_blur = Q.paired(mb, Q.window_metrics(blur, ref, alpha, _win(), mask_out=base), "band")
    assert d_usm["d_ovfrac"] > Q.GATE_OV
    assert not d_usm["gates"]["OV"]
    assert d_blur["d_ovfrac"] <= Q.GATE_OV
    # Sanity ceiling: USM pushes absolute band GER above 1.02.
    assert d_usm["ger_var"] > Q.GATE_CEIL
    assert not d_usm["gates"]["CEIL"]


def test_e_bloom_exclusion():
    rng = np.random.default_rng(1)
    ref = (0.3 + 0.01 * rng.standard_normal((64, 64, 3))).astype(np.float32)
    ref[30:34, 30:34] = 5.0  # small highlight
    out = ref.copy()
    yy, xx = np.mgrid[:64, :64]
    rr = np.hypot(yy - 31.5, xx - 31.5)
    out[rr <= 8.0] = 5.0  # bloomed disc
    out[(rr > 8.0) & (rr <= 10.5)] = 0.9  # sub-1 halo, only reachable via dilation
    alpha = np.full((64, 64), 0.5, dtype=np.float32)
    inner = np.zeros((64, 64), dtype=bool)
    inner[4:-4, 4:-4] = True

    def ov(dil):
        m = Q.band_mask(alpha, Q.luma(ref), Q.luma(out), bloom_dilate=dil, border=0) & inner
        return Q.q1_metrics(out, ref, alpha, m)["ovfrac"]

    assert ov(0) > 0.0
    assert ov(3) == 0.0


def test_f_float32_alpha_trap():
    d = np.full((40, 40), 0.5, dtype=np.float32)  # dmap >= knee
    alpha = (np.cos(0.5 * np.pi * np.minimum(d / 0.10, 1.0)) ** 2).astype(np.float32)
    assert (alpha > 0).all()  # the trap: strictly positive everywhere
    y = np.zeros((40, 40))
    assert not Q.band_mask(alpha, y, y, border=0).any()


def test_g_single_channel_blur_fires_channel_gate():
    ref, alpha, base = _scene()
    var = _apply_band(base, _blur(base, 1.0, chans=(0,)), alpha)
    mb = Q.window_metrics(base, ref, alpha, _win())
    mv = Q.window_metrics(var, ref, alpha, _win(), mask_out=base)
    p = Q.paired(mb, mv, "band")
    d_r = mv["ger_rgb"][0] - mb["ger_rgb"][0]
    assert p["ch_flag"] and "CH" not in p["gates"]
    assert abs(p["d_ger"]) < abs(d_r)


def test_h_global_weakening_fires_bg_gate():
    ref, alpha, base = _scene()
    bgm = (alpha <= Q.ALPHA_LO)[..., None]
    var = np.where(bgm, base + 0.5 * (ref - base), base).astype(np.float32)
    mb = Q.window_metrics(base, ref, alpha, _win())
    mv = Q.window_metrics(var, ref, alpha, _win(), mask_out=base)
    assert mv["bg_ger"] > mb["bg_ger"]
    p = Q.paired(mb, mv, "nonband")
    assert not p["gates"]["BG"]


def test_i_focus_window_xy_orientation():
    shape = (100, 300, 3)
    ys, xs = Q.focus_window(shape, (0.8, 0.2))
    assert abs((xs.start + xs.stop) / 2 - 240) <= 1
    assert abs((ys.start + ys.stop) / 2 - 20) <= 1
    assert (xs.stop - xs.start) == (ys.stop - ys.start)


@pytest.mark.parametrize("xy", [(0.0, 0.0), (1.0, 1.0), (0.0, 1.0), (1.0, 0.0)])
def test_j_focus_window_clamped(xy):
    h, w = 150, 220
    ys, xs = Q.focus_window((h, w, 3), xy, side_2048=2048)
    assert 0 <= ys.start < ys.stop <= h
    assert 0 <= xs.start < xs.stop <= w
    ys, xs = Q.focus_window((h, w, 3), xy, side_2048=128)
    assert 0 <= ys.start < ys.stop <= h
    assert 0 <= xs.start < xs.stop <= w
    assert (xs.stop - xs.start) % 2 == 0


def test_k_small_mask_returns_none():
    ref, alpha, base = _scene()
    mask = np.zeros((N, N), dtype=bool)
    mask[:5, :5] = True
    m = Q.q1_metrics(base, ref, alpha, mask)
    assert m["n_px"] == 25
    assert m["ger_band"] is None and m["ssim_band"] is None
    assert m["ovfrac"] is None and m["ger_rgb"] is None
    assert Q.paired(m, m, "band") == {"valid": False}


def test_l_paired_lever_routing():
    base = {"ger_band": 0.8, "ssim_band": 0.9, "ovfrac": 0.0,
            "ger_rgb": (0.8, 0.8, 0.8), "bg_ger": 0.5}
    var = {"ger_band": 0.9, "ssim_band": 0.95, "ovfrac": 0.001,
           "ger_rgb": (0.9, 0.9, 0.9), "bg_ger": 0.5}
    pb = Q.paired(base, var, "band")
    pn = Q.paired(base, var, "nonband")
    assert "Q1-B" in pb["gates"] and "Q1-N" not in pb["gates"]
    assert "Q1-N" in pn["gates"] and "Q1-B" not in pn["gates"]
    assert pb["gates"]["Q1-B"] and not pn["gates"]["Q1-N"]
    with pytest.raises(ValueError):
        Q.paired(base, var, "bogus")
