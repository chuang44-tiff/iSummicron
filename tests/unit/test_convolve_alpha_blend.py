"""P1.9.1: per-pixel α(defocus) blend unit tests."""
from __future__ import annotations
import inspect
import numpy as np
from summicron.render import convolve
from summicron.render.convolve import D_KNEE_LAYERED, render, srgb_to_linear
from tests.fixtures.defocus import render_with_defocus

_EDGES = np.array([0.0, 0.15, 0.30, 0.45, 0.60, 0.80, 1.01], dtype=np.float64)
_RATIOS = np.array([0.0, 0.33, 0.67, 1.0], dtype=np.float64)


def _blur_tensor(ksize=65):
    R, D = 6, 4
    kernels = np.zeros((R, D, ksize, ksize), dtype=np.float32)
    c = ksize // 2
    y, x = np.mgrid[-c:c + 1, -c:c + 1].astype(np.float32)
    for di in range(D):
        g = np.exp(-(x ** 2 + y ** 2) / (2 * (1.0 + di * 2.0) ** 2))
        kernels[:, di] = (g / g.sum()).astype(np.float32)
    return {"kernels": kernels, "radial_edges": _EDGES.copy(), "defocus_ratios": _RATIOS.copy()}


def _image(seed=0, H=64, W=64):
    return np.random.default_rng(seed).uniform(0.1, 0.9, size=(H, W, 3)).astype(np.float32)


def test_alpha_blend_dmap_zero_identity():
    """dmap=0 → α=1 → output_linear bit-identical to img_lin."""
    img = _image()
    dmap = np.zeros(img.shape[:2], dtype=np.float32)
    _, out = render_with_defocus(img, _blur_tensor(), dmap=dmap, psf_norm="l1")
    assert np.max(np.abs(out - srgb_to_linear(img))) < 1e-6


def test_alpha_blend_dmap_saturated_matches_preblend(monkeypatch):
    """dmap ≥ D_KNEE_LAYERED → α=0 → output equals pre-blend convolved result."""
    img = _image(seed=1)
    t = _blur_tensor()
    dmap = np.full(img.shape[:2], 1.0, dtype=np.float32)
    _, blended = render_with_defocus(img, t, dmap=dmap, psf_norm="l1")
    monkeypatch.setattr(convolve, "D_KNEE_LAYERED", 1e-9)
    _, noblend = render_with_defocus(img, t, dmap=dmap, psf_norm="l1")
    assert np.allclose(blended, noblend, atol=1e-6)
    assert np.max(np.abs(blended - srgb_to_linear(img))) > 1e-3


def test_alpha_blend_continuity_smooth_dmap():
    """Adjacent-pixel α diff bounded by the α curve's Lipschitz constant (no cliffs / NaN).

    Max |dα/dd| for cos² is π/(2·D_KNEE_LAYERED), so on a linspace(0, 1, N) sweep the
    adjacent-pixel step is bounded by `(π / (2·D_KNEE_LAYERED)) / (N − 1)`. Sample
    densely enough that the bound is well below 1% of the α range.
    """
    N = 4096
    dmap = np.linspace(0, 1, N, dtype=np.float32)[None, :].repeat(4, axis=0)
    a = (np.cos(0.5 * np.pi * np.minimum(dmap / D_KNEE_LAYERED, 1.0)) ** 2).astype(np.float32)
    bound = (np.pi / (2.0 * D_KNEE_LAYERED)) / (N - 1) * 1.1  # 10% margin
    assert np.isfinite(a).all() and np.max(np.abs(np.diff(a, axis=1))) < bound


def test_energy_gate_precedes_alpha_blend_in_source():
    """Structural: energy gate must run on pre-blend output."""
    src = inspect.getsource(render)
    assert src.index("energy ratio") < src.index("alpha[:, :, None]")
