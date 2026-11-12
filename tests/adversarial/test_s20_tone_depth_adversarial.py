"""Adversarial: tone LUT numerics/polarity and diopter mapping polarity (S1)."""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from summicron.depth.diopter import auto_gain, diopter_equivalent_depth
from summicron.io.tone_lut import (
    apply_input_lut, auto_bright_threshold, fit_input_lut,
)

OD = np.geomspace(0.7, 20.0, 11)


def _lin(u8):
    from summicron.render.convolve import srgb_to_linear
    return srgb_to_linear(u8.astype(np.float32) / 255.0)


def _pair(gain, n=40000, seed=0):
    rs = np.random.RandomState(seed)
    sdr = _lin(rs.randint(0, 256, (200, n // 200, 3)).astype(np.uint8))
    return (sdr * gain).astype(np.float32), sdr


# ---- LUT polarity / monotonicity ----------------------------------------------------

@pytest.mark.parametrize("gain", [1.0, 2.0, 3.0, 8.0])
def test_lut_direction_hdr_to_sdr(gain):
    hdr, sdr = _pair(gain)
    xs, ys = fit_input_lut(hdr, sdr)
    assert xs[0] == 0 and ys[0] == 0
    assert np.all(np.diff(xs) > 0) and np.all(np.diff(ys) >= 0)
    # HDR value 3x brighter than SDR: LUT must map hdr luma H -> H/gain (inverse), so
    # for gain>1 the LUT lies BELOW identity (wrong polarity would give >= identity).
    mid = xs[len(xs) // 2]
    got = np.interp(mid, xs, ys)
    assert got == pytest.approx(mid / gain, rel=0.25)
    if gain > 1.5:
        assert got < mid


def test_apply_inverts_exposure():
    hdr, sdr = _pair(3.0)
    xs, ys = fit_input_lut(hdr, sdr)
    out = apply_input_lut(hdr, xs, ys).astype(np.float32) / 255
    ref = np.where(sdr <= 0.0031308, 12.92 * sdr, 1.055 * np.maximum(sdr, 0) ** (1 / 2.4) - 0.055)
    assert np.mean(np.abs(out - ref)) < 0.03


def test_apply_monotone_on_gray_ramp_including_zero():
    hdr, sdr = _pair(4.0)
    xs, ys = fit_input_lut(hdr, sdr)
    ramp = np.geomspace(1e-9, 50, 4000, dtype=np.float32)
    img = np.stack([ramp] * 3, -1)[None]
    out = apply_input_lut(np.concatenate([np.zeros((1, 1, 3), np.float32), img], 1), xs, ys)
    g = out[0, :, 1].astype(int)
    assert g[0] == 0
    assert np.all(np.diff(g) >= 0)
    assert g[-1] <= 255


def test_apply_edge_values_finite_and_dtype():
    xs, ys = np.array([0, 1.0]), np.array([0, 1.0])
    bad = np.array([[[np.nan, np.nan, np.nan], [np.inf, 0, 0], [-np.inf, -1, -5],
                     [1e30, 1e30, 1e30], [0, 0, 0], [1e-30, 0, 0]]], np.float32)
    out = apply_input_lut(bad, xs, ys)
    assert out.dtype == np.uint8 and out.shape == bad.shape
    assert tuple(out[0, 0]) == (0, 0, 0) and tuple(out[0, 2]) == (0, 0, 0)
    assert tuple(out[0, 4]) == (0, 0, 0)


def test_apply_hue_preserved_and_saturated_red():
    xs, ys = np.array([0, 1.0]), np.array([0, 1.0])
    px = np.array([[[0.4, 0.2, 0.1], [1e3, 0, 0]]], np.float32)
    out = apply_input_lut(px, xs, ys)
    assert out[0, 1, 0] >= 254 and out[0, 1, 1] == 0 and out[0, 1, 2] == 0
    assert out[0, 0, 0] > out[0, 0, 1] > out[0, 0, 2]


@pytest.mark.xfail(strict=True, reason=(
    "BUG-2: apply_input_lut float32 luma sum of white (0.2126+0.7152+0.0722) is 0.99999994, "
    "so with an identity LUT (the degenerate/no-headroom fallback xs=[0,1]) any saturated "
    "pixel encodes to 254 instead of 255 after the truncating u8 cast (off-by-one at clip)."))
def test_identity_lut_saturated_white_is_255():
    xs, ys = np.array([0, 1.0]), np.array([0, 1.0])
    out = apply_input_lut(np.full((1, 1, 3), 2.0, np.float32), xs, ys)
    assert tuple(out[0, 0]) == (255, 255, 255)


def test_apply_does_not_mutate_input_and_no_warnings():
    xs, ys = np.array([0, 1.0]), np.array([0, 1.0])
    a = np.random.RandomState(1).rand(8, 8, 3).astype(np.float32) * 5
    b = a.copy()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        apply_input_lut(a, xs, ys)
    assert np.array_equal(a, b)


def test_apply_float64_and_noncontiguous():
    xs, ys = np.array([0, 1.0]), np.array([0, 1.0])
    a = np.random.RandomState(1).rand(8, 16, 3)
    o1 = apply_input_lut(a, xs, ys)
    o2 = apply_input_lut(np.asfortranarray(a.astype(np.float32)), xs, ys)
    assert np.abs(o1.astype(int) - o2.astype(int)).max() <= 1


# ---- fit guards ---------------------------------------------------------------------

def test_fit_deterministic_and_pure():
    hdr, sdr = _pair(2.5)
    h0, s0 = hdr.copy(), sdr.copy()
    a, b = fit_input_lut(hdr, sdr), fit_input_lut(hdr, sdr)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    assert np.array_equal(hdr, h0) and np.array_equal(sdr, s0)


@pytest.mark.parametrize("val", [0.0, 1e-4, 1e-3])
def test_fit_black_identity_with_warning(val):
    z = np.full((50, 50, 3), val, np.float32)
    with pytest.warns(RuntimeWarning):
        xs, ys = fit_input_lut(z, z)
    assert list(xs) == [0, 1] and list(ys) == [0, 1]


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("which", [0, 1])
def test_fit_nonfinite_valueerror(bad, which):
    hdr, sdr = _pair(2.0, n=4000)
    arrs = [hdr, sdr]
    arrs[which] = arrs[which].copy()
    arrs[which][0, 0, 0] = bad
    with pytest.raises(ValueError):
        fit_input_lut(*arrs)


def test_fit_shape_mismatch_and_tiny():
    with pytest.raises(ValueError):
        fit_input_lut(np.ones((4, 4, 3), np.float32), np.ones((4, 5, 3), np.float32))
    with pytest.warns(RuntimeWarning):
        xs, ys = fit_input_lut(np.ones((1, 1, 3), np.float32), np.ones((1, 1, 3), np.float32))
    assert list(xs) == [0, 1]


def test_fit_empty_array():
    e = np.zeros((0, 4, 3), np.float32)
    try:
        with pytest.warns(RuntimeWarning):
            fit_input_lut(e, e)
    except ValueError:
        pass


def test_fit_all_white_single_bin_and_huge_range():
    w = np.ones((100, 100, 3), np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        xs, ys = fit_input_lut(w, w)
    assert np.all(np.diff(ys) >= 0)
    rs = np.random.RandomState(3)
    hdr = np.exp(rs.uniform(np.log(1e-4), np.log(32), (300, 300, 3))).astype(np.float32)
    sdr = np.clip(hdr ** (1 / 3), 0, 1).astype(np.float32)
    xs, ys = fit_input_lut(hdr, sdr)
    assert np.all(np.diff(ys) >= 0) and np.isfinite(xs).all() and np.isfinite(ys).all()


def test_fit_negative_hdr_values_ok():
    hdr, sdr = _pair(2.0, n=20000)
    hdr = hdr.copy()
    hdr[0, :50] = -0.5
    xs, ys = fit_input_lut(hdr, sdr)
    assert np.all(np.diff(ys) >= 0)


def test_fit_float64_inputs():
    hdr, sdr = _pair(2.0, n=20000)
    a = fit_input_lut(hdr, sdr)
    b = fit_input_lut(hdr.astype(np.float64), sdr.astype(np.float64))
    assert np.allclose(a[0], b[0], rtol=1e-4) and np.allclose(a[1], b[1], atol=1e-3)


# ---- auto threshold ------------------------------------------------------------------

def test_thr_identity_is_one_and_never_below_one():
    xs = np.array([0, 0.5, 1.0]); ys = xs.copy()
    thr, h = auto_bright_threshold(xs, ys)
    assert thr == 1.0 and h == pytest.approx(0.9)


def test_thr_polarity_larger_gain_larger_thr():
    t = []
    for g in (1.0, 2.0, 4.0):
        hdr, sdr = _pair(g)
        t.append(auto_bright_threshold(*fit_input_lut(hdr, sdr))[0])
    assert t[0] <= t[1] <= t[2] and t[2] > 2.0


def test_thr_never_reaches_09_disables():
    xs = np.array([0, 1.0, 5.0]); ys = np.array([0, 0.3, 0.5])
    thr, h = auto_bright_threshold(xs, ys)
    assert h == 5.0 and thr == pytest.approx(5.5)


def test_thr_plateau_and_flat_zero():
    xs = np.array([0, 1.0, 2.0, 3.0, 4.0]); ys = np.array([0, 0.95, 0.95, 0.95, 0.95])
    thr, h = auto_bright_threshold(xs, ys)
    assert np.isfinite(thr) and 0 < h <= 1.0
    thr, h = auto_bright_threshold(np.array([0, 1.0]), np.array([0.0, 0.0]))
    assert np.isfinite(thr)


# ---- diopter polarity ----------------------------------------------------------------

def test_diopter_gain_polarity_monotone():
    z = np.array([0.7, 1.0, 2.0, 5.0, 12.0], np.float32)
    prev = None
    for g in (0.5, 1.0, 2.0, 4.0):
        out = diopter_equivalent_depth(z, 0.7, gain=g, z_focus=0.7, z_far=20.0)
        if prev is not None:
            assert np.all(out >= prev - 1e-6)
        prev = out
    assert prev[0] == pytest.approx(0.7, rel=1e-5)  # the anchor stays in focus
    assert np.all(np.diff(prev) >= -1e-6)  # farther -> never sharper


def test_diopter_symmetric_nearer_than_anchor_is_defocused():
    out = diopter_equivalent_depth(np.array([0.5, 2.0], np.float32), 2.0, gain=1.0,
                                   z_focus=0.7, z_far=20.0)
    assert out[1] == pytest.approx(0.7, rel=1e-5) and out[0] > 0.7


def test_diopter_nonfinite_and_zero_depth():
    z = np.array([np.nan, np.inf, -np.inf, 0.0, -1.0], np.float32)
    out = diopter_equivalent_depth(z, 2.0, gain=1.0, z_focus=0.7, z_far=20.0)
    assert np.isfinite(out).all() and out.min() >= 0.7 - 1e-6 and out.max() <= 20.0 + 1e-4


def test_auto_gain_degenerate():
    for d in (np.full((10, 10), 2.0, np.float32), np.full((10, 10), np.nan, np.float32),
              np.full((10, 10), np.inf, np.float32)):
        g = auto_gain(d, 2.0, z_focus=0.7, od=OD, lo=1.0)
        assert g == 1.0 or (np.isfinite(g) and 1.0 <= g <= 2.0)


def test_auto_gain_bounded():
    rs = np.random.RandomState(0)
    for _ in range(20):
        d = rs.uniform(0.3, 30, (50, 50)).astype(np.float32)
        g = auto_gain(d, float(rs.uniform(0.5, 5)), z_focus=0.7, od=OD, lo=1.0)
        assert 1.0 <= g <= 2.0


# ---- focus_target_m outside the tensor range under diopter (pipeline) -----------------

@pytest.mark.xfail(strict=True, reason=(
    "BUG-1: --depth-mapping diopter with --focus-target-m > od[-1] (e.g. 25 > 20 m) makes "
    "cap=1/z_focus-1/z_far negative, so diopter_equivalent_depth maps EVERY pixel, "
    "including the in-focus anchor, to z_far (whole frame max-blurred); no validation/warning."))
def test_diopter_focus_target_beyond_far_keeps_anchor_in_focus():
    out = diopter_equivalent_depth(np.array([25.0, 30.0], np.float32), 25.0, gain=1.0,
                                   z_focus=25.0, z_far=20.0)
    # the anchor pixel (dp = 0) must be sharper than a farther pixel, not slam to z_far
    assert out[0] < out[1]
