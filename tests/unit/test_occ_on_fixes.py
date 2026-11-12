"""R-occ-on-fixes: +/-Inf depth with occlusion ON, and peak+ON fallback."""
from __future__ import annotations

import subprocess
import sys
import warnings

import numpy as np
import pytest

from summicron.render import convolve as cv
from summicron.render._layered_composite import compute_edge_selector
from tests.unit.test_over_scatter_fused import _step_scene, _tensor


def _depth():
    _, dmap = _step_scene()
    ods = np.geomspace(0.7, 20.0, 11)
    return np.interp(dmap, np.linspace(0, 1, 11), ods).astype(np.float32)


def _render(depth, **kw):
    img, _ = _step_scene()
    res = cv.render(img, _tensor(), depth_m=depth, **kw)
    return res[0] if isinstance(res, tuple) else res


def _with_inf():
    d = _depth()
    d[3, 5] = np.inf
    d[10, 30] = -np.inf
    return d


def test_inf_depth_on_renders_finite_and_matches_clamped():
    d = _with_inf()
    ref = _depth()
    fin = d[np.isfinite(d)]
    clamped = d.copy()
    clamped[np.isposinf(d)] = fin.max()
    clamped[np.isneginf(d)] = fin.min()
    out = _render(d, occlusion_compositing="on")
    assert np.isfinite(out.astype(np.float32)).all()
    # render() maps +/-Inf depth to the tensor's far/near slot (np.interp
    # clamp) while `clamped` uses the map's finite max/min, so only pixels
    # near the two Inf sites may differ; the selector is pinned exactly in
    # the next test.
    other = _render(clamped, occlusion_compositing="on")
    diff = np.abs(out.astype(np.int32) - other.astype(np.int32)).max(axis=-1)
    assert (diff > 8).sum() <= 12
    assert ref.shape == d.shape


def test_selector_inf_equals_clamped_and_finite_noop():
    d = _with_inf()
    fin = d[np.isfinite(d)]
    c = d.copy()
    c[np.isposinf(d)] = fin.max()
    c[np.isneginf(d)] = fin.min()
    dm = np.zeros(d.shape, np.float32)
    assert np.array_equal(compute_edge_selector(d, dm, 8),
                          compute_edge_selector(c, dm, 8))
    f = _depth()
    assert np.isfinite(compute_edge_selector(f, dm, 8)).all()


def test_nan_depth_on_still_fails_loudly():
    d = _depth()
    d[4, 4] = np.nan
    with pytest.raises(RuntimeError, match="Gate FAIL: NaN"):
        _render(d, occlusion_compositing="on")


def test_finite_depth_selector_only_bit_identical_to_head():
    # HEAD selector (no Inf branch) reproduced inline.
    from scipy import ndimage
    from summicron.render import _layered_composite as lc
    d = _depth()
    dm = np.full(d.shape, 0.3, np.float32)
    gx = ndimage.sobel(d, axis=1, mode="reflect")
    gy = ndimage.sobel(d, axis=0, mode="reflect")
    g = np.hypot(gx, gy).astype(np.float32)
    ref_d = float(np.median(np.abs(d)))
    denom = np.maximum(lc._SELECTOR_K * lc._coc_px(dm, 8), np.float32(1e-6))
    expect = lc._smoothstep((g / ref_d) / denom)
    assert np.array_equal(compute_edge_selector(d, dm, 8), expect)


def test_peak_on_equals_peak_off_and_warns():
    d = _depth()
    off = _render(d, psf_norm="peak", occlusion_compositing="off")
    msgs: list[str] = []
    with pytest.warns(RuntimeWarning, match="peak"):
        on = _render(d, psf_norm="peak", occlusion_compositing="on",
                     on_progress=msgs.append)
    assert np.array_equal(on, off)
    assert any("peak" in m for m in msgs)


def test_l1_on_unaffected_no_warning():
    d = _depth()
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        on = _render(d, psf_norm="l1", occlusion_compositing="on")
    off = _render(d, psf_norm="l1", occlusion_compositing="off")
    assert np.isfinite(on).all() and on.shape == off.shape
    assert not np.array_equal(on, off)  # ON path really ran
