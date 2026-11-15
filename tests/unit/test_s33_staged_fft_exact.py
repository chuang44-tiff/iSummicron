"""S33 WP2: staged FFTs are byte-equal to rfft2/irfft2 on the guarded build.

Oracle: ``scipy.fft.rfft2`` / ``irfft2`` in-tree. Needs no runs/ goldens.
"""
from __future__ import annotations

import platform
import sys

import numpy as np
import pytest
import scipy
import scipy.fft as sp_fft

import summicron.render.convolve as cv

_GUARD = (
    scipy.__version__ == "1.17.1"
    and platform.machine() == "arm64"
    and sys.platform == "darwin"
)
_SKIP = pytest.mark.skipif(
    not _GUARD, reason=f"staged FFT proof covers scipy 1.17.1 darwin arm64 only (have {scipy.__version__})"
)

# (H, W) of the padded frame at each downres, both orientations, odd/odd and even/odd.
_SHAPES = [(512, 384), (384, 512), (1024, 768), (768, 1024), (1536, 1152),
           (1152, 1536), (2048, 1536), (1536, 2048), (513, 385), (512, 385)]
_NATIVE = [(6144, 4800), (4800, 6144)]


def test_guard_state_matches_condition():
    assert cv._STAGED_FFT == _GUARD


def _fshape(h, w, k):
    return (sp_fft.next_fast_len(h + k - 1, real=True),
            sp_fft.next_fast_len(w + k - 1, real=True))


def _check(h, w, k, workers, seed=0):
    rng = np.random.default_rng(seed)
    fshape = _fshape(h, w, k)
    img = rng.random((h, w), dtype=np.float32)
    ker = rng.random((k, k), dtype=np.float32)
    ker /= ker.sum()
    with sp_fft.set_workers(workers):
        img_F = sp_fft.rfft2(img, s=fshape)
        kref = sp_fft.rfft2(ker, s=fshape)
        kst = cv._kernel_rfft2_staged(ker, fshape)
        assert kref.tobytes() == kst.tobytes()
        X = img_F * kref
        full = sp_fft.irfft2(X, s=fshape)
        base = (k - 1) // 2
        # windows covering every residue mod 4, offsets not multiples of 4
        for n in (h, h - 1, h - 2, h - 3, 37, 38, 39, 40):
            r0 = base + 3 if n != h else base
            r1 = min(r0 + n, fshape[0])
            got = cv._irfft2_rows(X, fshape, r0, r1)
            assert full[r0:r1].tobytes() == got.tobytes(), (h, w, k, workers, r0, r1)


@_SKIP
@pytest.mark.parametrize("workers", [1, -1])
@pytest.mark.parametrize("k", [67, 183])
@pytest.mark.parametrize("shape", _SHAPES)
def test_staged_bytes_equal(shape, k, workers):
    _check(shape[0], shape[1], k, workers)


@_SKIP
@pytest.mark.slow
@pytest.mark.parametrize("workers", [1, -1])
@pytest.mark.parametrize("k", [67, 183])
@pytest.mark.parametrize("shape", _NATIVE)
def test_staged_bytes_equal_native(shape, k, workers):
    _check(shape[0], shape[1], k, workers)


@_SKIP
def test_hoisted_matches_reference_branch():
    rng = np.random.default_rng(3)
    h, w, k = 130, 101, 67
    fshape = _fshape(h, w, k)
    img = rng.random((h, w), dtype=np.float32)
    ker = rng.random((k, k), dtype=np.float32)
    img_F = sp_fft.rfft2(img, s=fshape)
    sy = slice((k - 1) // 2, (k - 1) // 2 + h)
    sx = slice((k - 1) // 2, (k - 1) // 2 + w)
    got = cv._fftconvolve_hoisted(img_F, ker, fshape, sy, sx)
    ref = sp_fft.irfft2(img_F * sp_fft.rfft2(ker, s=fshape), s=fshape)[sy, sx].astype(np.float32)
    assert got.tobytes() == ref.tobytes()


@_SKIP
@pytest.mark.parametrize("support_skip", [False, True])
def test_gather_5tuple_staged_equals_reference(monkeypatch, support_skip):
    """The whole gather (hoisted branch incl. bbox rows) is byte-equal with
    the staged FFTs on vs off."""
    from test_s33_exact_support_bbox import _assert_bytes_equal, _capture, _run

    kw = _capture()
    monkeypatch.setattr(cv, "_STAGED_FFT", False)
    ref = _run(kw, slab_support_skip=support_skip)
    monkeypatch.setattr(cv, "_STAGED_FFT", True)
    got = _run(kw, slab_support_skip=support_skip)
    _assert_bytes_equal(ref, got)
