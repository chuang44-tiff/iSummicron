"""S21 adversarial: WB chroma correction (tone_lut sat) + diopter far-focus guard."""
from __future__ import annotations

import warnings
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from summicron.io.tone_lut import (
    LUMA_W,
    apply_input_lut,
    fit_input_lut,
    fit_input_sat,
    fit_input_tone,
)
from summicron.render.api import RenderOptions, render_image

_IDENT = (np.array([0.0, 1.0]), np.array([0.0, 1.0]))


def _synth(sat_hdr=1.0, k_true=0.85, hmax=6.0, shape=(240, 320), seed=0):
    """HDR image with chroma, SDR = luma-LUT + desaturated by k_true (linear)."""
    rng = np.random.default_rng(seed)
    y = np.exp(rng.uniform(np.log(0.01), np.log(hmax), shape)).astype(np.float32)
    ratio = (1.0 + sat_hdr * rng.uniform(-0.3, 0.3, shape + (3,))).astype(np.float32)
    ratio = np.maximum(ratio, 0.05)
    ratio /= (ratio @ LUMA_W)[..., None]
    hdr = y[..., None] * ratio
    ys = np.clip(y / hmax * 1.0, 0, 1).astype(np.float32)
    sdr = ys[..., None] * np.maximum(1 + k_true * (ratio - 1), 0)
    sdr = np.clip(sdr, 0, 1).astype(np.float32)
    return hdr, sdr


def _chroma_err(u8, ref_lin):
    """Mean abs unit-luma chroma error (sRGB-decoded u8 vs reference linear)."""
    c = u8.astype(np.float32) / 255.0
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    yo = np.maximum(lin @ LUMA_W, 1e-4)
    yr = np.maximum(ref_lin @ LUMA_W, 1e-4)
    m = yr > 0.05
    return float(np.abs(lin / yo[..., None] - ref_lin / yr[..., None])[m].mean())


# ---- neutral / degenerate inputs -------------------------------------------

def test_grey_stays_neutral_with_sat():
    g = np.repeat(np.linspace(0.0, 4.0, 5000, dtype=np.float32).reshape(50, 100, 1), 3, -1)
    sdr = np.clip(g / 4, 0, 1)
    xs, ys, kx, kk = fit_input_tone(g, sdr)
    out = apply_input_lut(g, xs, ys, (kx, kk))
    assert np.array_equal(out[..., 0], out[..., 1]) and np.array_equal(out[..., 1], out[..., 2])


def test_grey_no_chroma_identity_sat_no_warning():
    g = np.repeat(np.random.default_rng(0).uniform(0, 4, (60, 80, 1)).astype(np.float32), 3, -1)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        kx, kk = fit_input_sat(g, np.clip(g / 4, 0, 1))
    assert len(kx) >= 2 and np.all(np.isfinite(kk))


def test_hdr_equals_sdr_identity_sat_and_matches_luma_only():
    hdr, _ = _synth(hmax=1.0)
    hdr = np.clip(hdr, 0, 1)
    kx, kk = fit_input_sat(hdr, hdr)
    assert np.all(kk == 1.0)
    xs, ys = fit_input_lut(hdr, hdr)
    a = apply_input_lut(hdr, xs, ys, (kx, kk)).astype(int)
    b = apply_input_lut(hdr, xs, ys, None).astype(int)
    assert np.abs(a - b).max() <= 2


def test_near_black_finite():
    z = np.zeros((40, 40, 3), np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        xs, ys, kx, kk = fit_input_tone(z, z)
    out = apply_input_lut(z, xs, ys, (kx, kk))
    assert out.dtype == np.uint8 and out.max() == 0
    t = np.full((40, 40, 3), 1e-5, np.float32)
    out = apply_input_lut(t, xs, ys, (kx, kk))
    assert out.dtype == np.uint8


@pytest.mark.parametrize("ch", [0, 1, 2])
def test_saturated_single_channel_highlight_finite_and_bounded(ch):
    hdr, sdr = _synth()
    xs, ys, kx, kk = fit_input_tone(hdr, sdr)
    img = np.zeros((20, 20, 3), np.float32)
    img[..., ch] = 50.0
    out = apply_input_lut(img, xs, ys, (kx, kk))
    assert out.dtype == np.uint8 and out.shape == (20, 20, 3)
    assert out[..., ch].min() > 0
    others = [c for c in range(3) if c != ch]
    assert out[..., others].max() <= out[..., ch].min()


def test_nan_inf_input_to_apply_finite_and_deterministic():
    hdr, sdr = _synth()
    xs, ys, kx, kk = fit_input_tone(hdr, sdr)
    img = np.ones((8, 8, 3), np.float32)
    img[0, 0] = np.nan
    img[1, 1] = np.inf
    img[2, 2] = -np.inf
    img[3, 3] = -5.0
    img[4, 4, 1] = np.inf
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        a = apply_input_lut(img, xs, ys, (kx, kk))
        b = apply_input_lut(img, xs, ys, (kx, kk))
    assert np.array_equal(a, b)
    assert tuple(a[0, 0]) == (0, 0, 0) and tuple(a[3, 3]) == (0, 0, 0)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_fit_sat_rejects_nonfinite(bad):
    hdr, sdr = _synth()
    h2 = hdr.copy()
    h2[0, 0, 0] = bad
    with pytest.raises(ValueError):
        fit_input_sat(h2, sdr)
    s2 = sdr.copy()
    s2[0, 0, 0] = bad
    with pytest.raises(ValueError):
        fit_input_sat(hdr, s2)


def test_fit_sat_shape_mismatch():
    hdr, sdr = _synth()
    with pytest.raises(ValueError):
        fit_input_sat(hdr, sdr[:-1])


def test_no_gain_map_sdr_equals_hdr_over_one_is_identity():
    # "no gain map": HDR decode == base, max <= 1.001
    hdr, _ = _synth(hmax=0.99)
    kx, kk = fit_input_sat(hdr, hdr)
    assert np.all(kk == 1.0)


def test_k_clamped_to_range():
    hdr, _ = _synth()
    # SDR much MORE saturated than HDR -> raw k > 1.2 -> clamp
    sdr_hi = np.clip(
        (hdr @ LUMA_W)[..., None] / 6.0
        * np.maximum(1 + 3.0 * (hdr / np.maximum(hdr @ LUMA_W, 1e-6)[..., None] - 1), 0),
        0, 1).astype(np.float32)
    _, kk = fit_input_sat(hdr, sdr_hi)
    assert kk.max() <= 1.2 + 1e-9 and kk.min() >= 0.0
    # SDR greyscale -> k ~ 0
    g = np.repeat(((hdr @ LUMA_W) / 6.0).clip(0, 1)[..., None], 3, -1).astype(np.float32)
    _, kk0 = fit_input_sat(hdr, g)
    assert kk0.min() >= 0.0 and kk0.max() < 0.1


def test_fit_sat_deterministic():
    hdr, sdr = _synth()
    a = fit_input_sat(hdr, sdr)
    b = fit_input_sat(hdr, sdr)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


# ---- wrong polarity --------------------------------------------------------

def test_correction_reduces_chroma_error_and_inverted_makes_it_worse():
    hdr, sdr = _synth(k_true=0.80)
    xs, ys, kx, kk = fit_input_tone(hdr, sdr)
    assert 0.6 < float(np.median(kk)) < 0.95  # fit recovers the desaturation
    err_none = _chroma_err(apply_input_lut(hdr, xs, ys, None), sdr)
    err_good = _chroma_err(apply_input_lut(hdr, xs, ys, (kx, kk)), sdr)
    kk_inv = np.clip(2.0 - kk, 0.0, 2.0)  # reflected about 1: pushes the WRONG way
    err_inv = _chroma_err(apply_input_lut(hdr, xs, ys, (kx, kk_inv)), sdr)
    assert err_good < err_none < err_inv, (err_good, err_none, err_inv)


def test_correction_polarity_of_effect_on_saturation():
    hdr, sdr = _synth(k_true=0.7)
    xs, ys, kx, kk = fit_input_tone(hdr, sdr)

    def spread(u8):
        c = u8.astype(np.float32)
        return float((c.max(-1) - c.min(-1)).mean())

    assert spread(apply_input_lut(hdr, xs, ys, (kx, kk))) < spread(apply_input_lut(hdr, xs, ys, None))


# ---- apply robustness ------------------------------------------------------

@pytest.mark.xfail(strict=True, reason="BUG(low): apply_input_lut np.log(kx) emits divide-by-zero RuntimeWarning for kx containing 0 (fit never emits 0; public-API misuse only)")
def test_apply_sat_zero_in_kx_does_not_nan():
    hdr, sdr = _synth()
    xs, ys, _, _ = fit_input_tone(hdr, sdr)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = apply_input_lut(hdr[:10, :10], xs, ys, (np.array([0.0, 1.0]), np.array([0.8, 0.8])))
    assert out.dtype == np.uint8


def test_apply_sat_k_zero_gives_grey():
    hdr, sdr = _synth()
    xs, ys, _, _ = fit_input_tone(hdr, sdr)
    out = apply_input_lut(hdr, xs, ys, (np.array([1e-3, 10.0]), np.array([0.0, 0.0])))
    assert np.array_equal(out[..., 0], out[..., 1]) and np.array_equal(out[..., 1], out[..., 2])


def test_apply_sat_does_not_mutate_input():
    hdr, sdr = _synth()
    xs, ys, kx, kk = fit_input_tone(hdr, sdr)
    h0 = hdr.copy()
    apply_input_lut(hdr, xs, ys, (kx, kk))
    assert np.array_equal(h0, hdr)


# ---- diopter far-focus guard ----------------------------------------------

def _pipeline(tmp_path, **kw):
    from PIL import Image

    img = tmp_path / "t.jpg"
    Image.fromarray(np.random.default_rng(0).integers(0, 256, (64, 64, 3), dtype=np.uint8)).save(img, quality=80)
    D, K = 4, 11
    ker = np.zeros((3, D, K, K), np.float32)
    ker[:, :, K // 2, K // 2] = 1.0
    ten = tmp_path / "t.npz"
    np.savez(str(ten), kernels=ker, radial_edges=np.linspace(0, 1, 4),
             defocus_ratios=np.linspace(0, 1, D), corpus_ref_long_edge=100,
             otf_norm_applied=True, object_distances_m=np.linspace(0.7, 5.0, D),
             focus_distance_m=0.7)
    depth = np.linspace(0.7, 4.0, 64 * 64, dtype=np.float32).reshape(64, 64)

    def fake_render(image, tensor, **k):
        h, w = image.shape[:2]
        return np.zeros((h, w, 3), np.uint8), np.zeros((h, w, 3), np.float32)

    opts = RenderOptions(image_path=img, tensor_path=ten, focus_xy=(0.5, 0.5),
                         out_dir=tmp_path / "o", **kw)
    with patch("summicron.depth.estimate.extract_lidar_depth", side_effect=lambda *a, **k: depth.copy()), \
         patch("summicron.render.convolve.render", side_effect=fake_render), \
         patch("summicron.render.convolve.save_results", side_effect=lambda u8, s, o, **k: o / "x.jpg"):
        return render_image(opts)


def test_guard_boundary_exact_far_end_raises(tmp_path):
    with pytest.raises(ValueError, match="focus-target-m"):
        _pipeline(tmp_path, depth_mapping="diopter", focus_target_m=5.0)


def test_guard_float32_roundtrip_of_far_end(tmp_path):
    # np.float32(5.0) == 5.0 exactly; a float32 far end like 4.3 would not be:
    with pytest.raises(ValueError, match="focus-target-m"):
        _pipeline(tmp_path, depth_mapping="diopter", focus_target_m=float(np.float32(5.0)))


@pytest.mark.parametrize("ft", [np.nextafter(5.0, 0.0), 4.999, 4.5, 0.7])
def test_guard_just_below_far_end_does_not_raise_guard(tmp_path, ft):
    try:
        _pipeline(tmp_path, depth_mapping="diopter", focus_target_m=float(ft))
    except ValueError as e:  # any other ValueError is fine to flag, the guard message is not
        assert "focus-target-m" not in str(e) or "must be <" not in str(e), e


def test_guard_diopter_with_depth_gain_explicit_also_guarded(tmp_path):
    with pytest.raises(ValueError, match="far end"):
        _pipeline(tmp_path, depth_mapping="diopter", depth_gain=2.0, focus_target_m=9.0)


def test_guard_not_applied_for_linear(tmp_path):
    _pipeline(tmp_path, depth_mapping="linear", focus_target_m=9.0)


def test_guard_cli_one_line_exit2(tmp_path, capsys):
    from summicron.render.convolve import run_cli_render

    assert run_cli_render(lambda: _pipeline(tmp_path, depth_mapping="diopter", focus_target_m=9.0)) == 2
    err = capsys.readouterr().err.strip().splitlines()
    assert len(err) == 1 and err[0].startswith("render: ") and "Traceback" not in err[0]


def test_guard_inf_focus_target_rejected_by_options(tmp_path):
    for bad in (float("inf"), float("nan"), 0.0, -1.0):
        with pytest.raises(ValueError):
            RenderOptions(image_path=Path("a.jpg"), focus_target_m=bad, depth_mapping="diopter")
