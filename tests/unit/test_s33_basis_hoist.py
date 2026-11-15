"""S33 WP3: radial basis as per-(ri, wedge) crops, built once per gather.

Oracle: the in-tree reference (``basis_hoist=False`` / ``basis_crops=None``,
the per-slab dense build). Needs no runs/ goldens and no IMG_6915.
"""
from __future__ import annotations

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


def _geometry(h, w, n_r=12):
    ys = np.arange(h, dtype=np.float32) - (h - 1) / 2.0
    xs = np.arange(w, dtype=np.float32) - (w - 1) / 2.0
    rad = np.hypot(ys[:, None], xs[None, :]).astype(np.float32)
    rf = (rad / rad.max()).astype(np.float32)
    theta = np.arctan2(ys[:, None], xs[None, :])
    wedge = ((theta + np.pi) / (2 * np.pi) * cv.N_ANG).astype(np.int32) % cv.N_ANG
    edges = np.concatenate(
        [[0.0, 0.15076], np.sqrt(np.arange(1, n_r) / 11.0)]
    )[: n_r + 1]
    rc = 0.5 * (edges[:-1] + edges[1:])
    return rf, wedge, rc


@pytest.mark.parametrize("shape", [(96, 80), (80, 96), (203, 151), (151, 203)])
def test_crops_equal_dense_regions(shape):
    rf, wedge, rc = _geometry(*shape)
    dense = cv._radial_basis_weights(rf, rc)
    crops = cv._build_basis_crops(rf, rc, wedge)
    assert len(crops) == len(rc)
    total = 0
    for ri, lst in enumerate(crops):
        assert dense[ri].min() >= 0.0
        union = np.zeros(shape, bool)
        for (y0, y1, x0, x1, crop) in lst:
            assert crop.dtype == np.float32
            assert crop.tobytes() == dense[ri, y0:y1, x0:x1].tobytes()
            union[y0:y1, x0:x1] = True
            total += crop.nbytes
        # basis is zero outside the union of the crops
        assert not np.any(dense[ri][~union] != 0)
        # pasting reproduces the dense plane byte for byte
        w = np.zeros(shape, np.float32)
        for (y0, y1, x0, x1, crop) in lst:
            w[y0:y1, x0:x1] = crop
        assert w.tobytes() == dense[ri].astype(np.float32).tobytes()
        # the per-ri skip uses the crop max: identical to the dense max
        cmax = max((c.max() for *_, c in lst), default=np.float32(0.0))
        assert (cmax < 0.01) == (dense[ri].max() < 0.01)
        assert cmax == dense[ri].max()
    print(f"crop planes {shape}: {total / (shape[0] * shape[1] * 4):.2f}")


def test_single_wedge_geometry():
    rf, _, rc = _geometry(64, 64)
    wedge = np.zeros((64, 64), np.int32)
    dense = cv._radial_basis_weights(rf, rc)
    crops = cv._build_basis_crops(rf, rc, wedge)
    for ri, lst in enumerate(crops):
        w = np.zeros((64, 64), np.float32)
        for (y0, y1, x0, x1, crop) in lst:
            w[y0:y1, x0:x1] = crop
        assert w.tobytes() == dense[ri].tobytes()


@pytest.mark.parametrize("transpose", [False, True])
def test_gather_5tuple_with_and_without_crops(transpose):
    img = None
    if transpose:
        rng = np.random.default_rng(4)
        img = rng.uniform(0.0, 0.8, (_W, _H, 3)).astype(np.float32)
    kw = _capture(img=img) if not transpose else _capture(
        img=img,
        depth=np.broadcast_to(
            np.geomspace(0.7, 20.0, _H, dtype=np.float32)[None, :], (_W, _H)
        ).copy(),
    )
    ref = _run(kw, basis_hoist=False)
    got = _run(kw, basis_hoist=True)
    _assert_bytes_equal(ref, got)
    _assert_bytes_equal(got, _run(kw))  # default is hoisted
