"""S33 WP1: slab-support skip + bbox mask/accumulate in the gather.

Oracle: the in-tree reference selected by ``slab_support_skip=False`` (which
reproduces the unchanged arithmetic). The 5-tuple of ``_layered_gather_diffuse``
must be ``tobytes``-equal. Needs no runs/ goldens and no IMG_6915.
"""
from __future__ import annotations

import threading
import warnings

import numpy as np
import pytest

import summicron.render.convolve as cv

_K = 31
_D = 11
_H, _W = 96, 80


def _tensor(d: int = _D, k: int = _K, n_r: int = 12) -> dict:
    od = np.array(
        [0.7, 0.98, 1.37, 1.91, 2.68, 3.74, 5.23, 7.32, 10.23, 14.30, 20.0]
    )[:d].astype(np.float64)
    kern = np.zeros((n_r, d, k, k), np.float32)
    c = k // 2
    kern[:, 0, c, c] = 1.0
    ys, xs = np.mgrid[-c:c + 1, -c:c + 1].astype(np.float32)
    for di in range(1, d):
        g = np.exp(-(xs ** 2 + ys ** 2) / (2 * (1.0 + 0.5 * di) ** 2))
        kern[:, di] = g / g.sum()
    return {
        "kernels": kern,
        "radial_edges": np.sqrt(np.linspace(0.0, 1.0, n_r + 1)),
        "defocus_ratios": np.linspace(0.0, 1.0, d),
        "object_distances_m": od,
        "focus_distance_m": 0.7,
        "corpus_ref_long_edge": None,
        "otf_norm_applied": False,
        "delta_guard_applied": True,
    }


def _capture(img=None, depth=None):
    """Run render once with a spy and return the gather kwargs."""
    rng = np.random.default_rng(3)
    if img is None:
        img = rng.uniform(0.0, 0.8, (_H, _W, 3)).astype(np.float32)
    t = _tensor()
    od = t["object_distances_m"]
    if depth is None:
        ramp = np.geomspace(od[0], od[-1], _W, dtype=np.float32)
        depth = np.broadcast_to(ramp[None, :], (_H, _W)).copy()
        depth[_H // 3: 2 * _H // 3, _W // 3: 2 * _W // 3] = np.float32(od[2])
    seen = {}
    real = cv._layered_gather_diffuse

    def spy(**kw):
        seen["kw"] = kw
        return real(**kw)

    orig = cv._layered_gather_diffuse
    cv._layered_gather_diffuse = spy
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cv.render(
                img, t, depth_m=depth, bright_thresh=0.5, hdr_source="macos"
            )
    finally:
        cv._layered_gather_diffuse = orig
    return seen["kw"]


def _run(kw, **over):
    k = dict(kw)
    k.update(over)
    k["on_progress"] = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return cv._layered_gather_diffuse(**k)


def _assert_bytes_equal(ref, got):
    assert len(ref) == len(got) == 5
    for i, (a, b) in enumerate(zip(ref, got)):
        a, b = np.asarray(a), np.asarray(b)
        assert a.dtype == b.dtype and a.shape == b.shape, i
        assert a.tobytes() == b.tobytes(), f"tuple element {i} differs"


class _Counter:
    def __init__(self, monkeypatch):
        self.none = 0
        self.cut = 0
        self.full = 0
        real = cv._support_bbox

        def spy(wt, support):
            bb = real(wt, support)
            if bb is None:
                self.none += 1
            else:
                ys, xs = np.nonzero(wt > 0)
                full = (ys.min(), ys.max() + 1, xs.min(), xs.max() + 1)
                if tuple(int(v) for v in full) == bb:
                    self.full += 1
                else:
                    self.cut += 1
            return bb

        monkeypatch.setattr(cv, "_support_bbox", spy)


@pytest.fixture(scope="module")
def kw_ramp():
    return _capture()


def test_a_depth_ramp_with_step(kw_ramp, monkeypatch):
    kw = dict(kw_ramp)
    ref = _run(kw, slab_support_skip=False)
    c = _Counter(monkeypatch)
    got = _run(kw, slab_support_skip=True)
    assert c.none >= 1, "no support-skipped unit"
    assert c.cut >= 1, "no unit whose bbox edge cuts the wedge"
    _assert_bytes_equal(ref, got)
    # default argument == skip on
    _assert_bytes_equal(got, _run(kw))


def test_b_highlight_on_support_edge(kw_ramp):
    rng = np.random.default_rng(5)
    img = rng.uniform(0.0, 0.3, (_H, _W, 3)).astype(np.float32)
    img[40, 39] = 20.0  # highlight right on the support edge
    img[41, 41] = 15.0
    kw = _capture(img=img)
    D = kw["slab_w"].shape[0]
    sw = np.zeros((D, _H, _W), np.float32)
    # slab 4 support: columns < 40, with a tiny (denormal-adjacent) weight at
    # the edge column so sw goes from tiny to exactly 0 within kernel reach.
    sw[4, :, :39] = 0.5
    sw[4, :, 39] = np.float32(1e-30)
    sw[5, :, 40:] = 0.5
    sw[5, :, 39] = np.float32(0.5)
    sw[6, :, :] = 0.0
    sw[6, 20:30, 38:42] = 0.25
    kw["slab_w"] = sw
    ref = _run(kw, slab_support_skip=False)
    got = _run(kw, slab_support_skip=True)
    _assert_bytes_equal(ref, got)
    assert float(np.abs(ref[0]).sum()) > 0


def test_c_non_pou_with_holes_and_empty_slabs(kw_ramp):
    kw = dict(kw_ramp)
    rng = np.random.default_rng(11)
    D = kw["slab_w"].shape[0]
    sw = np.zeros((D, _H, _W), np.float32)
    for d in range(D):
        if d in (1, 5):  # empty-skipped slabs
            continue
        m = rng.uniform(size=(_H, _W)) < 0.25
        sw[d][m] = rng.uniform(0.05, 0.3, m.sum()).astype(np.float32)
    # a hole block no slab covers
    sw[:, 10:30, 10:30] = 0.0
    kw["slab_w"] = sw
    ref = _run(kw, slab_support_skip=False)
    got = _run(kw, slab_support_skip=True)
    _assert_bytes_equal(ref, got)
    # confirm there really were holes before fill
    assert float(sw.sum(axis=0)[10:30, 10:30].max()) == 0.0


@pytest.mark.parametrize("batch", [1, 3, _D])
def test_d_slab_batch_sizes(kw_ramp, batch):
    kw = dict(kw_ramp)
    ref = _run(kw, slab_support_skip=False, slab_batch_size=batch)
    got = _run(kw, slab_support_skip=True, slab_batch_size=batch)
    _assert_bytes_equal(ref, got)


def test_e_preset_cancel_raises_before_any_fft(kw_ramp, monkeypatch):
    from summicron.render.api import CancelledError

    def boom(*a, **k):
        raise AssertionError("FFT executed after a pre-set cancel")

    monkeypatch.setattr(cv.sp_fft, "rfft2", boom)
    monkeypatch.setattr(cv.sp_fft, "irfft2", boom)
    ev = threading.Event()
    ev.set()
    D = kw_ramp["slab_w"].shape[0]
    with pytest.raises(CancelledError) as ei:
        _run(dict(kw_ramp), cancel=ev)
    assert str(ei.value) == f"render cancelled at slab 0/{D}"


def test_legacy_fft_state_none_ignores_support(kw_ramp):
    kw = dict(kw_ramp)
    kw["fft_state"] = None
    ref = _run(kw, slab_support_skip=False)
    got = _run(kw, slab_support_skip=True)
    _assert_bytes_equal(ref, got)
