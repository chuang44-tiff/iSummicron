"""S22 S1: bright-pass stamp (`_bright_stamp`, `bright_stamp_gain` threading)."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.ndimage import rotate as ndimage_rotate
from scipy.signal import fftconvolve

from summicron.render import convolve as cv
from tests.fixtures.synthetic_tensor import make_synthetic_tensor


def _kernel(seed=0, n=15):
    rng = np.random.default_rng(seed)
    k = rng.random((n, n)).astype(np.float32) + 0.05
    return k / k.sum()


class TestBrightStampHelper:
    def test_none_is_verbatim_peak_divide(self):
        k = _kernel()
        out = cv._bright_stamp(k, None)
        exp = k / max(float(k.max()), 1e-10)
        assert out.dtype == exp.dtype == np.float32
        assert np.array_equal(out, exp)

    def test_gain_is_float32_multiply(self):
        k = _kernel(1)
        out = cv._bright_stamp(k, 2.0)
        assert out.dtype == np.float32
        assert np.array_equal(out, k * np.float32(2.0))

    def test_gain_two_is_exact_doubling(self):
        k = _kernel(2)
        assert np.array_equal(cv._bright_stamp(k, 2.0), k + k)


def _setup(n=15, H=24, W=24, seed=3):
    pad = n // 2
    rng = np.random.default_rng(seed)
    img = rng.random((H + 2 * pad, W + 2 * pad, 3)).astype(np.float32)
    outs = lambda: (  # noqa: E731
        np.zeros((H, W, 3), np.float32), np.zeros((H, W, 3), np.float32),
        np.zeros((H, W), np.float32), np.zeros((H, W), np.float32),
    )
    w = np.ones((H, W), np.float32)
    wedge = np.zeros((H, W), np.int32)  # every pixel in wedge 0
    return pad, img, outs, w, wedge, H, W


def _run(kernel, gain, hoisted, seed=3):
    pad, img, outs, w, wedge, H, W = _setup(kernel.shape[0], seed=seed)
    out, outb, ws, dws = outs()
    fft_state = (
        cv._build_image_fft_state(img, img, img, kernel.shape[0]) if hoisted else None
    )
    kw = {} if gain is None else {"bright_stamp_gain": gain}
    cv._apply_kernel(
        kernel, w, img, img, img, wedge, pad, H, W, "l1",
        out, outb, ws, dws, fft_state=fft_state, **kw,
    )
    return out, outb, img, pad, H, W


def _expected_peak_bright(kernel, img, pad, H, W):
    rot = ndimage_rotate(kernel, -((0.5) * (2 * np.pi / cv.N_ANG) - np.pi) * 180.0 / np.pi - 90.0,
                         reshape=False, order=1)
    rot = np.maximum(rot, 0.0)
    rot /= rot.sum() + 1e-12
    rb = rot / max(float(rot.max()), 1e-10)
    return np.stack(
        [fftconvolve(img[:, :, c], rb, mode="same").astype(np.float32)[pad:pad + H, pad:pad + W]
         for c in range(3)], axis=-1,
    )


class TestApplyKernelStamp:
    def test_peak_path_legacy_branch_bit_equal_to_inline_reference(self):
        k = _kernel(4)
        out, outb, img, pad, H, W = _run(k, None, hoisted=False)
        exp = _expected_peak_bright(k, img, pad, H, W)
        assert np.array_equal(outb, exp)

    @pytest.mark.parametrize("hoisted", [False, True])
    def test_energy_law_l1_stamp(self, hoisted):
        k = _kernel(5)
        out, outb, *_ = _run(k, 2.0, hoisted)
        np.testing.assert_allclose(outb, 2.0 * out, rtol=1e-4, atol=1e-6)

    @pytest.mark.parametrize("hoisted", [False, True])
    def test_default_none_equals_explicit_none(self, hoisted):
        k = _kernel(6)
        a = _run(k, None, hoisted)[1]
        pad, img, outs, w, wedge, H, W = _setup(k.shape[0])
        out, outb, ws, dws = outs()
        fs = cv._build_image_fft_state(img, img, img, k.shape[0]) if hoisted else None
        cv._apply_kernel(k, w, img, img, img, wedge, pad, H, W, "l1",
                         out, outb, ws, dws, fft_state=fs, bright_stamp_gain=None)
        assert np.array_equal(a, outb)


class TestRenderGainValidation:
    def _args(self):
        t = make_synthetic_tensor(R=2, D=3, H=9, W=9)
        img = np.full((16, 16, 3), 0.4, np.float32)
        depth = np.full((16, 16), 0.7, np.float32)
        return img, t, depth

    def test_peak_with_gain_raises(self):
        img, t, depth = self._args()
        with pytest.raises(ValueError, match="bright_stamp_gain"):
            cv.render(img, t, depth_m=depth, psf_norm="peak", bright_stamp_gain=2.0)

    @pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf"), True, "2"])
    def test_bad_gain_raises(self, bad):
        img, t, depth = self._args()
        with pytest.raises(ValueError, match="bright_stamp_gain"):
            cv.render(img, t, depth_m=depth, bright_stamp_gain=bad)

    def test_gain_accepted(self):
        img, t, depth = self._args()
        cv.render(img, t, depth_m=depth, bright_stamp_gain=2.0, bright_thresh=0.3)

    def test_gain_forwarded_to_gather_and_composite(self, monkeypatch):
        import importlib

        cv = importlib.import_module("summicron.render.convolve")  # reload-proof
        img, t, depth = self._args()
        seen = {}
        real = cv._layered_gather_diffuse

        def spy(**kw):
            seen["g"] = kw["bright_stamp_gain"]
            return real(**kw)

        monkeypatch.setattr(cv, "_layered_gather_diffuse", spy)
        cv.render(img, t, depth_m=depth, bright_stamp_gain=3.0)
        assert seen["g"] == 3.0
        seen.clear()
        cv.render(img, t, depth_m=depth, occlusion_compositing="on",
                  bright_stamp_gain=3.0)
        assert seen["g"] == 3.0
