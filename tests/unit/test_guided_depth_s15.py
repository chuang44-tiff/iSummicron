"""s15 guided-filter depth refinement: synthetic checks."""
import numpy as np
import pytest

from summicron.depth.guided import guided_filter, guided_upsample_depth


def _step_case(n=64, blur_px=3):
    """Sharp depth step at col n//2 (2 m | 10 m) and a matching guide edge;
    the 'bilinear' input is the step smeared by a ramp."""
    x = np.arange(n)
    truth_d = np.where(x < n // 2, 2.0, 10.0)[None, :].repeat(n, 0).astype(np.float32)
    disp = 1.0 / truth_d
    ramp = np.clip((x - (n // 2 - blur_px)) / (2.0 * blur_px), 0, 1)
    soft_disp = (disp[:, :1] * (1 - ramp) + disp[:, -1:] * ramp).astype(np.float32)
    soft_d = (1.0 / soft_disp).astype(np.float32)
    guide = np.where(x < n // 2, 0.2, 0.8)[None, :].repeat(n, 0).astype(np.float32)
    return truth_d, soft_d, guide


def test_step_edge_sharper_than_bilinear():
    truth, soft, guide = _step_case()
    out = guided_upsample_depth(soft, guide, r=6, eps=1e-3)
    band = slice(29, 35)  # the smeared ramp, around the true edge at col 32
    err_soft = np.abs(1 / soft - 1 / truth)[:, band].mean()
    err_out = np.abs(1 / out - 1 / truth)[:, band].mean()
    assert err_out < 0.7 * err_soft
    # Edge steepness: largest adjacent-pixel disparity jump must grow a lot.
    assert np.abs(np.diff(1 / out, axis=1)).max() > 2 * np.abs(np.diff(1 / soft, axis=1)).max()


def test_flat_guide_is_near_identity():
    rng = np.random.default_rng(0)
    d = (2.0 + rng.random((48, 40))).astype(np.float32)
    d = guided_upsample_depth(d, np.full((48, 40), 0.5, np.float32), r=4, eps=1e-3)
    # flat guide -> a=0 -> output is box-smoothed input; compare to smoothed
    from scipy.ndimage import uniform_filter
    disp0 = 1 / np.clip(
        (2.0 + np.random.default_rng(0).random((48, 40))).astype(np.float32), 0.1, 1e3)
    sm = uniform_filter(uniform_filter(disp0.astype(np.float64), 9, mode="nearest"),
                        9, mode="nearest")
    assert np.abs(1 / d - sm).max() < 1e-4
    # smooth ramp input is preserved closely
    ramp = np.linspace(2, 4, 40, dtype=np.float32)[None, :].repeat(48, 0)
    out = guided_upsample_depth(ramp, np.full((48, 40), 0.5, np.float32), r=3, eps=1e-3)
    assert np.abs(out - ramp)[:, 8:-8].max() < 0.05


def test_shape_dtype_finite_and_clip():
    d = np.full((20, 30), 5.0, np.float32)
    d[:, :5] = 0.05   # below clip floor
    d[:, -5:] = 5e3   # above clip ceiling
    guide = np.random.default_rng(1).random((20, 30, 3)).astype(np.float32)
    out = guided_upsample_depth(d, guide, r=2, eps=1e-2)
    assert out.shape == d.shape and out.dtype == np.float32
    assert np.isfinite(out).all()
    assert out.min() >= 0.1 and out.max() <= 1e3


def test_bad_args():
    with pytest.raises(ValueError):
        guided_upsample_depth(np.ones((4, 4), np.float32), np.ones((5, 4), np.float32))
    with pytest.raises(ValueError):
        guided_filter(np.ones((4, 4)), np.ones((4, 4)), 0, 1e-3)
    with pytest.raises(ValueError):
        guided_filter(np.ones((4, 4)), np.ones((4, 4)), 1, 0.0)
