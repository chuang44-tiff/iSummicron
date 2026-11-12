"""S1 tone fix G: summicron.io.tone_lut (numpy only)."""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from summicron.io.tone_lut import (
    LUMA_W,
    apply_input_lut,
    auto_bright_threshold,
    fit_input_lut,
)


def _gray(y):
    y = np.asarray(y, np.float32)
    return np.repeat(y[..., None], 3, axis=-1)


def _ident_tol(xs, ys, hmax):
    r = (hmax * 1.01 / 1e-3) ** (1 / 64)
    tol = np.sqrt(r) - 1
    return tol


@pytest.mark.parametrize("kind", ["lognormal", "uniform", "ramp"])
def test_identity_when_hdr_equals_sdr(kind):
    rng = np.random.default_rng(0)
    n = 400 * 300
    if kind == "lognormal":
        v = np.clip(rng.lognormal(-1.5, 0.8, n), 0, 1)
    elif kind == "uniform":
        v = rng.uniform(0, 1, n)
    else:
        v = np.linspace(0, 1, n)
    img = _gray(v.reshape(300, 400))
    xs, ys = fit_input_lut(img, img)
    assert xs[0] == 0.0 and ys[0] == 0.0
    tol = _ident_tol(xs, ys, float(img.max()))
    probe = xs[1:]
    assert np.all(np.abs(ys[1:] - probe) <= tol * probe + 1e-9)


def test_monotone_and_anchor_random():
    rng = np.random.default_rng(1)
    hdr = rng.uniform(0, 8, (200, 300, 3)).astype(np.float32)
    sdr = np.clip(hdr / 8 + rng.normal(0, 0.05, hdr.shape), 0, 1).astype(np.float32)
    xs, ys = fit_input_lut(hdr, sdr)
    assert (xs[0], ys[0]) == (0.0, 0.0)
    assert np.all(np.diff(ys) >= 0) and np.all(np.diff(xs) > 0)


def test_deterministic():
    rng = np.random.default_rng(2)
    hdr = rng.uniform(0, 4, (150, 200, 3)).astype(np.float32)
    sdr = np.clip(hdr / 4, 0, 1)
    a = fit_input_lut(hdr, sdr)
    b = fit_input_lut(hdr, sdr)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_shape_mismatch_and_nonfinite():
    a = np.ones((10, 10, 3), np.float32)
    with pytest.raises(ValueError):
        fit_input_lut(a, np.ones((10, 11, 3), np.float32))
    bad = a.copy()
    bad[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        fit_input_lut(bad, a)
    with pytest.raises(ValueError):
        fit_input_lut(a, np.where(a > 0, np.inf, a))


@pytest.mark.parametrize("shape,val", [((50, 50, 3), 0.0), ((50, 50, 3), 5e-4), ((1, 1, 3), 0.7)])
def test_degenerate_identity_with_warning(shape, val):
    img = np.full(shape, val, np.float32)
    with pytest.warns(RuntimeWarning):
        xs, ys = fit_input_lut(img, img)
    assert list(xs) == [0.0, 1.0] and list(ys) == [0.0, 1.0]


def test_all_white_single_bin_identity_warning():
    img = np.ones((100, 100, 3), np.float32)
    with pytest.warns(RuntimeWarning):
        xs, ys = fit_input_lut(img, img)
    assert list(xs) == [0.0, 1.0]


def test_hdr_max_32_range():
    rng = np.random.default_rng(3)
    hdr = np.exp(rng.uniform(np.log(0.01), np.log(32), (300, 300))).astype(np.float32)
    sdr = np.clip(hdr / 4, 0, 1)
    xs, ys = fit_input_lut(_gray(hdr), _gray(sdr))
    assert xs[-1] < 32 * 1.01 and np.all(np.diff(ys) >= 0)


def test_apply_saturates_and_u8():
    xs = np.array([0.0, 1.0, 2.0])
    ys = np.array([0.0, 0.5, 0.8])
    img = _gray(np.array([[0.0, 2.0, 10.0, 100.0]]))
    out = apply_input_lut(img, xs, ys)
    assert out.dtype == np.uint8 and out.shape == (1, 4, 3)
    assert out[0, 2, 0] == out[0, 3, 0] == out[0, 1, 0]  # saturates above last knot
    assert out[0, 0, 0] == 0


def test_apply_hue_preserved():
    xs = np.array([0.0, 1.0, 4.0])
    ys = np.array([0.0, 0.3, 0.6])
    rgb = np.array([[[0.6, 0.3, 0.15]]], np.float32)  # in-gamut after LUT
    out = apply_input_lut(rgb, xs, ys)
    from summicron.io.tone_lut import _srgb_encode  # noqa: F401
    lin = ((out.astype(np.float32) / 255 + 0.055) / 1.055) ** 2.4
    ratio = lin[0, 0] / lin[0, 0, 1]
    assert ratio[0] == pytest.approx(2.0, rel=0.05)
    assert ratio[2] == pytest.approx(0.5, rel=0.05)


def test_apply_overbright_keeps_hue_via_maxc_divide():
    xs = np.array([0.0, 1.0])
    ys = np.array([0.0, 1.0])
    rgb = np.array([[[3.0, 1.0, 0.2]]], np.float32)
    out = apply_input_lut(rgb, xs, ys)
    assert out[0, 0, 0] >= 254 and out[0, 0, 1] < 254


def test_apply_handles_nan_inf_negative():
    xs = np.array([0.0, 1.0])
    ys = np.array([0.0, 1.0])
    rgb = np.array([[[np.nan, 0.5, 0.5], [np.inf, 1, 1], [-1, -1, -1]]], np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = apply_input_lut(rgb, xs, ys)
    assert out.dtype == np.uint8
    assert tuple(out[0, 2]) == (0, 0, 0)


# ---- threshold rule ----

def test_auto_threshold_known_crossing():
    xs = np.array([0.0, 1.0, 2.0, 3.0])
    ys = np.array([0.0, 0.5, 0.9, 1.0])
    thr, h = auto_bright_threshold(xs, ys)
    assert h == pytest.approx(2.0) and thr == pytest.approx(2.2)
    assert h == pytest.approx(float(np.interp(0.9, ys, xs)))


def test_auto_threshold_floor_one():
    xs = np.array([0.0, 0.5, 0.8])
    ys = np.array([0.0, 0.5, 0.95])
    thr, h = auto_bright_threshold(xs, ys)
    assert thr == 1.0 and h < 0.8


def test_auto_threshold_never_reaches_09():
    xs = np.array([0.0, 1.0, 4.0])
    ys = np.array([0.0, 0.3, 0.6])
    thr, h = auto_bright_threshold(xs, ys)
    assert h == 4.0 and thr == pytest.approx(4.4)


def test_auto_threshold_plateau_uses_first_index():
    xs = np.array([0.0, 1.0, 2.0, 3.0])
    ys = np.array([0.0, 0.9, 0.9, 0.9])
    thr, h = auto_bright_threshold(xs, ys)
    assert h == pytest.approx(1.0)


def test_identity_lut_threshold_in_range():
    rng = np.random.default_rng(4)
    img = _gray(rng.uniform(0, 1, (300, 400)))
    xs, ys = fit_input_lut(img, img)
    thr, _ = auto_bright_threshold(xs, ys)
    assert 1.0 <= thr <= 1.12


# ---- rev 3: per-luma-bin saturation scale (C3) -----------------------------------

from summicron.io.tone_lut import fit_input_sat, fit_input_tone  # noqa: E402


def _warm_pair(k_true=0.85, n=(300, 400), seed=3):
    """HDR (luma 2*y, y<=0.6) with warm chroma; SDR luma y, chroma scaled by k_true."""
    rng = np.random.default_rng(seed)
    y = np.exp(rng.uniform(np.log(0.02), np.log(0.6), n)).astype(np.float64)
    r = np.stack([
        1.0 + rng.uniform(0.1, 0.5, n),
        1.0 + rng.uniform(-0.1, 0.1, n),
        1.0 - rng.uniform(0.1, 0.5, n),
    ], -1)
    r = r / (r @ LUMA_W.astype(np.float64))[..., None]  # unit luma
    hdr = (2.0 * y[..., None] * r).astype(np.float32)
    rp = 1.0 + k_true * (r - 1.0)
    sdr = (y[..., None] * rp).astype(np.float32)
    return hdr, sdr, r


def test_sat_fit_recovers_k():
    hdr, sdr, _ = _warm_pair(0.85)
    kx, kk = fit_input_sat(hdr, sdr)
    assert len(kx) >= 2 and len(kx) == len(kk)
    assert np.all(np.diff(kx) > 0)
    mid = (kx > 0.05) & (kx < 1.1)
    assert mid.sum() >= 3
    assert np.all(np.abs(kk[mid] - 0.85) < 0.02)


def test_sat_identity_when_hdr_equals_sdr():
    hdr, _, _ = _warm_pair()
    hdr = np.minimum(hdr, 1.0)
    # HDR with headroom needed for the fit; add one bright gray pixel row
    hdr[0, :, :] = 2.0
    kx, kk = fit_input_sat(hdr, hdr)
    assert np.allclose(kk, 1.0, atol=1e-6)


def test_sat_no_headroom_is_identity_and_silent():
    hdr, _, _ = _warm_pair()
    hdr = np.minimum(hdr, 1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        kx, kk = fit_input_sat(hdr, hdr)
    assert np.all(kk == 1.0) and len(kx) >= 2 and np.all(kx > 0)


def test_sat_gray_scene_identity():
    img = _gray(np.exp(np.random.default_rng(0).uniform(-4, 1.5, (200, 300))))
    kx, kk = fit_input_sat(img, np.minimum(img, 1.0))
    assert np.all(kk == 1.0)


def test_sat_shape_and_nonfinite_rejected():
    a = np.ones((4, 4, 3), np.float32)
    with pytest.raises(ValueError):
        fit_input_sat(a, np.ones((4, 5, 3), np.float32))
    b = a.copy()
    b[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        fit_input_sat(b, a)


def test_sat_k_clamped():
    hdr, _, r = _warm_pair()
    y = hdr @ LUMA_W
    sdr = ((y / 2)[..., None] * (1.0 + 3.0 * (r - 1.0))).astype(np.float32)
    sdr = np.maximum(sdr, 0)
    _, kk = fit_input_sat(hdr, sdr)
    assert kk.max() <= 1.2 + 1e-9 and kk.min() >= 0.0


def test_apply_sat_none_bit_identical_and_identity_sat_matches():
    rng = np.random.default_rng(5)
    a = rng.uniform(0, 3, (32, 48, 3)).astype(np.float32)
    xs, ys = np.array([0, 1.0, 4.0]), np.array([0, 0.5, 0.9])
    base = apply_input_lut(a, xs, ys)
    assert np.array_equal(base, apply_input_lut(a, xs, ys, None))
    one = (np.array([1e-3, 1.0]), np.array([1.0, 1.0]))
    assert np.abs(apply_input_lut(a, xs, ys, one).astype(int) - base.astype(int)).max() <= 1


def test_apply_sat_reduces_chroma_error_on_warm_pair():
    hdr, sdr, _ = _warm_pair(0.85)
    xs, ys, kx, kk = fit_input_tone(hdr, sdr)

    def disp_ratio(u8):
        m = u8.astype(np.float64).mean((0, 1))
        return m[0] / m[2]

    ref = _srgb(sdr)
    err0 = abs(disp_ratio(apply_input_lut(hdr, xs, ys)) / disp_ratio(ref) - 1)
    err1 = abs(disp_ratio(apply_input_lut(hdr, xs, ys, (kx, kk))) / disp_ratio(ref) - 1)
    assert err1 < err0 * 0.5
    assert err1 < 0.015


def _dec(u8):
    v = u8.astype(np.float32) / 255.0
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4).astype(np.float32)


def _srgb(lin):
    from summicron.io.tone_lut import _srgb_encode
    return np.clip(_srgb_encode(np.clip(lin, 0, 1)) * 255.0, 0, 255).astype(np.uint8)


def test_apply_sat_keeps_luma_gray_and_finite():
    xs, ys = np.array([0, 1.0, 4.0]), np.array([0, 0.5, 0.9])
    sat = (np.array([1e-3, 1.0, 4.0]), np.array([0.7, 0.85, 0.8]))
    rng = np.random.default_rng(6)
    a = rng.uniform(0.05, 0.4, (40, 40, 3)).astype(np.float32)
    o0 = _dec(apply_input_lut(a, xs, ys))
    o1 = _dec(apply_input_lut(a, xs, ys, sat))
    assert np.isfinite(o1).all()
    d = np.abs(o1 @ LUMA_W - o0 @ LUMA_W)  # linear luma, in-gamut pixels
    assert np.percentile(d, 90) < 0.02
    g = _gray(np.linspace(0, 3, 50 * 50).reshape(50, 50))
    og = apply_input_lut(g, xs, ys, sat)
    assert np.array_equal(og[..., 0], og[..., 1]) and np.array_equal(og[..., 1], og[..., 2])
    nasty = np.array([[[0, 0, 0], [np.nan, 1, 1], [np.inf, 0, 0], [-1, 2, 3], [1e9, 1e9, 1e9]]], np.float32)
    out = apply_input_lut(nasty, xs, ys, sat)
    assert out.dtype == np.uint8 and out.shape == (1, 5, 3)


def test_apply_sat_monotone_in_luma_for_fixed_chroma():
    xs, ys = np.array([0, 1.0, 4.0]), np.array([0, 0.5, 0.9])
    sat = (np.array([1e-3, 1.0, 4.0]), np.array([0.7, 0.85, 0.8]))
    t = np.linspace(0.01, 4.0, 400, dtype=np.float32)
    px = (t[:, None] * np.array([1.4, 1.0, 0.6], np.float32))[None]
    out = apply_input_lut(px, xs, ys, sat)[0].astype(int)
    assert np.all(np.diff(out @ np.array([2126, 7152, 722])) >= -20)
