"""R-occ-on-fixes adversarial: Inf/NaN depth edge cases under ON, peak+ON fallback.

Spec: doc/plan/phase2/section18_occ-on-fixes/locked-spec-occ-on-fixes.md
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from summicron.render import _layered_composite as lc
from summicron.render import convolve as cv
from summicron.render._layered_composite import compute_edge_selector
from tests.unit.test_over_scatter_fused import _step_scene, _tensor

_NAN_MSG = "Gate FAIL: NaN in layered output"  # HEAD 394c260 message, ON and OFF


def _depth():
    _, dmap = _step_scene()
    ods = np.geomspace(0.7, 20.0, 11)
    return np.interp(dmap, np.linspace(0, 1, 11), ods).astype(np.float32)


def _render(depth, **kw):
    img, _ = _step_scene()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # Center SSIM noise
        return cv.render(img, _tensor(), depth_m=depth, **kw)


def _eq(a, b):
    a = a if isinstance(a, tuple) else (a,)
    b = b if isinstance(b, tuple) else (b,)
    return len(a) == len(b) and all(np.array_equal(x, y) for x, y in zip(a, b))


# ---- NaN keeps failing loudly, with HEAD's exact message -------------------

def _nan():
    d = _depth()
    d[4, 4] = np.nan
    return d


def _nan_inf():
    d = _depth()
    d[4, 4] = np.nan
    d[3, 5] = np.inf
    d[10, 30] = -np.inf
    return d


@pytest.mark.parametrize("mk", [_nan, _nan_inf, lambda: np.full_like(_depth(), np.nan)],
                         ids=["nan", "nan+inf", "all-nan"])
@pytest.mark.parametrize("mode", ["on", "off"])
def test_nan_depth_raises_head_message(mk, mode):
    with pytest.raises(RuntimeError, match=_NAN_MSG):
        _render(mk(), occlusion_compositing=mode)


# ---- Inf variants ----------------------------------------------------------

def test_neg_inf_only_on_finite():
    d = _depth()
    d[10, 30] = -np.inf
    out = _render(d, occlusion_compositing="on")
    assert all(np.isfinite(np.asarray(x, np.float64)).all() for x in out)


@pytest.mark.parametrize("kind", ["all-inf", "single-finite", "all-neg-inf"])
def test_degenerate_inf_maps_on_equals_off(kind):
    d = _depth()
    if kind == "all-inf":
        d[:] = np.inf
    elif kind == "single-finite":
        d[:] = np.inf
        d[0, 0] = 1.0
    else:
        d[:] = -np.inf
    on = _render(d, occlusion_compositing="on")
    off = _render(d, occlusion_compositing="off")
    assert _eq(on, off)


def test_selector_does_not_mutate_caller_and_handles_float64():
    d = _depth().astype(np.float64)
    d[3, 5] = np.inf
    d[10, 30] = -np.inf
    keep = d.copy()
    s = compute_edge_selector(d, np.zeros(d.shape, np.float32), 8)
    assert np.array_equal(d, keep, equal_nan=True)
    assert s.dtype == np.float32 and np.isfinite(s).all()


def test_selector_nan_still_propagates():
    d = _depth()
    d[4, 4] = np.nan
    d[3, 5] = np.inf
    s = compute_edge_selector(d, np.zeros(d.shape, np.float32), 8)
    assert np.isnan(s).any()  # NaN not silenced by the Inf branch


def test_inf_depth_off_never_reaches_sanitiser(monkeypatch):
    calls = []
    real = lc.compute_edge_selector
    monkeypatch.setattr(lc, "compute_edge_selector",
                        lambda *a, **k: calls.append(1) or real(*a, **k))
    d = _depth()
    d[3, 5] = np.inf
    _render(d, occlusion_compositing="off")
    assert calls == []


# ---- peak + ON fallback ----------------------------------------------------

def test_peak_on_is_off_path_exactly(monkeypatch):
    d = _depth()
    off_msgs: list[str] = []
    off_stats: dict = {}
    off = _render(d, psf_norm="peak", occlusion_compositing="off",
                  on_progress=off_msgs.append, stats=off_stats)

    def boom(**_):
        raise AssertionError("ON compositor ran under peak")
    monkeypatch.setattr(lc, "_layered_composite", boom)
    on_msgs: list[str] = []
    on_stats: dict = {}
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        on = _render(d, psf_norm="peak", occlusion_compositing="on",
                     on_progress=on_msgs.append, stats=on_stats)
    rw = [x for x in w if issubclass(x.category, RuntimeWarning)]
    assert len(rw) == 1 and "falling back to off" in str(rw[0].message)
    assert rw[0].filename.endswith("test_occ_on_fixes_adversarial.py")  # stacklevel=2
    assert _eq(on, off)
    assert on_msgs[1:] == off_msgs and "falling back" in on_msgs[0]
    assert not any("occlusion-compositing: ON" in m for m in on_msgs)
    assert on_stats.keys() == off_stats.keys()
    for k in on_stats:
        assert np.array_equal(np.asarray(on_stats[k]), np.asarray(off_stats[k]))


def test_peak_on_still_validates_mode_first():
    with pytest.raises(ValueError, match="occlusion_compositing"):
        _render(_depth(), psf_norm="peak", occlusion_compositing="ON")


def test_peak_on_render_options_not_rejected():
    from summicron.render.api import RenderOptions
    RenderOptions(image_path="x.heic", psf_norm="peak", occlusion_compositing="on")
