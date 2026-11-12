"""Q-e adversarial: fused OVER scatter vs the pre-Q-e two-pass, END TO END.

``tests/unit/test_over_scatter_fused.py`` pins ``_over_scatter_slab`` in
isolation. These probes close the orchestration gaps it does not reach:
the basis_w hoist, the 2-D matte pad, the per-slab ``s_reach`` gate, the
no-silhouette fast path, ``slab_batch_size``, the bright pass, ``psf_norm``
peak, kernel zoom (pix_scale < 0.95), odd/tiny frames, both depth polarities
and an all-zero image -- all through ``render(..., occlusion_compositing="on")``.

Reference: the fused scatter is swapped (monkeypatch) for a shim that runs the
OLD per-slab body -- the colour pass and the 3-channel-replicated matte pass --
through the LIVE ``convolve._apply_kernel(fft_state=None)`` branch (fftconvolve),
with the old discarded bright/wsum outputs. Outputs and progress must be
bit-identical. If CLI-1c deletes ``_apply_kernel``'s ``fft_state=None`` branch
these tests skip; the frozen oracle in the unit file then remains the guard.
"""
from __future__ import annotations

import inspect
import threading

import numpy as np
import pytest

import summicron.render._layered_composite as lc
from summicron.render import convolve as cv
from summicron.render.api import CancelledError
from tests.fixtures.defocus import render_with_defocus

_OD = np.array([0.7, 0.98, 1.37, 1.91, 2.68, 3.74, 5.23, 7.32, 10.23, 14.30, 20.0])

_HAS_LEGACY = "fft_state" in inspect.signature(cv._apply_kernel).parameters
pytestmark = pytest.mark.skipif(
    not _HAS_LEGACY, reason="convolve._apply_kernel fft_state=None branch removed"
)


def _tensor(corpus_ref=None, k=33):
    R, D = 6, 11
    kern = np.zeros((R, D, k, k), np.float32)
    c = k // 2
    kern[:, 0, c, c] = 1.0
    ys, xs = np.mgrid[-c:c + 1, -c:c + 1].astype(np.float32)
    for di in range(1, D):
        for ri in range(R):
            g = np.exp(-(xs ** 2 / (2 * (1.5 + 1.3 * di) ** 2)
                         + ys ** 2 / (2 * (1.0 + 0.9 * di + 0.2 * ri) ** 2)))
            kern[ri, di] = g / g.sum()
    return dict(
        kernels=kern, radial_edges=np.array([0, .15, .3, .45, .6, .8, 1.01]),
        defocus_ratios=np.linspace(0, 1, D), object_distances_m=_OD.copy(),
        focus_distance_m=0.7, corpus_ref_long_edge=corpus_ref,
        otf_norm_applied=False, delta_guard_applied=True,
    )


def _scene(H, W, kind, seed=1):
    rng = np.random.default_rng(seed)
    img = rng.uniform(0.05, 0.9, (H, W, 3)).astype(np.float32)
    d = np.zeros((H, W), np.float32)
    if kind == "near_left":          # near subject in front of far background
        d[:, W // 2:] = 0.9
    elif kind == "far_left":         # wrong-polarity mirror
        d[:, :W // 2] = 0.9
    elif kind == "disc":             # near disc on a far plane
        yy, xx = np.mgrid[:H, :W]
        d[:] = 0.85
        d[(yy - H / 2) ** 2 + (xx - W / 2) ** 2 < (min(H, W) / 4) ** 2] = 0.0
    elif kind == "three":
        d[:, W // 3:] = 0.4
        d[:, 2 * W // 3:] = 1.0
    elif kind == "smooth":
        d[:] = np.linspace(0, 1, W)[None, :]
    elif kind == "flat":             # no silhouette -> fast path
        d[:] = 0.5
    return img, d


def _old_two_pass_shim(
    *, d_idx, premult_pad, matte_pad, basis_w, wedge_idx, kernels,
    defocus_ratios, kH, kW, pad, pix_scale, scaled_ksize, psf_norm, H, W,
    fft_state, conv_c, conv_a, cancel,
):
    """Pre-Q-e OVER body via the LIVE fftconvolve branch (two passes)."""
    matte3 = np.repeat(matte_pad[..., None], 3, axis=2)  # == pad(repeat(H_d))
    conv_a3 = np.zeros((H, W, 3), np.float32)
    for img_pad, out in ((premult_pad, conv_c), (matte3, conv_a3)):
        ws = np.zeros((H, W), np.float32)
        dws = np.zeros((H, W), np.float32)
        for ri in range(basis_w.shape[0]):
            if cancel is not None and cancel.is_set():
                raise CancelledError("shim cancelled")
            w = basis_w[ri].astype(np.float32)
            if w.max() < 0.01:
                continue
            kb = cv._build_basis_kernel(
                ri, d_idx, True, defocus_ratios, kernels,
                kH, kW, psf_norm, pix_scale, scaled_ksize,
            )
            cv._apply_kernel(
                kb, w, img_pad, img_pad, img_pad, wedge_idx, pad, H, W, psf_norm,
                out, np.zeros((H, W, 3), np.float32), ws, dws, fft_state=None,
            )
    conv_a[...] = conv_a3[:, :, 0]


def _render_on(img, d, tensor, **kw):
    msgs: list[str] = []
    out = render_with_defocus(
        img, tensor, dmap=d, occlusion_compositing="on",
        on_progress=msgs.append, **kw,
    )
    return out, msgs


_CASES = [
    # (H, W, kind, psf_norm, bright_thresh, slab_batch_size, zoom)
    (61, 47, "near_left", "l1", 1e9, None, False),
    (61, 47, "far_left", "l1", 1e9, None, False),
    (47, 61, "disc", "l1", 1e9, 1, False),
    (61, 47, "three", "l1", 0.5, 3, False),
    (96, 128, "disc", "l1", 0.5, None, True),
    (96, 128, "near_left", "l1", 0.5, 1, True),
    (61, 47, "smooth", "l1", 1e9, None, True),
    (61, 47, "flat", "l1", 1e9, None, False),
    (5, 3, "near_left", "l1", 1e9, None, False),
]


@pytest.mark.parametrize("H,W,kind,psf_norm,bt,sbs,zoom", _CASES)
def test_render_on_bit_identical_to_old_two_pass(
    monkeypatch, H, W, kind, psf_norm, bt, sbs, zoom
):
    img, d = _scene(H, W, kind)
    if bt < 1e8:
        img[H // 4:H // 4 + 3, W // 4:W // 4 + 3] = 1.0  # bright-pass pixels
    t = _tensor(2.0 * max(H, W) if zoom else None)
    kw = dict(psf_norm=psf_norm, bright_thresh=bt, slab_batch_size=sbs)
    new, new_msgs = _render_on(img, d, t, **kw)
    monkeypatch.setattr(lc, "_over_scatter_slab", _old_two_pass_shim)
    old, old_msgs = _render_on(img, d, t, **kw)
    assert new_msgs == old_msgs
    for a, b in zip(new, old):
        assert np.array_equal(a, b)
    if kind == "flat":
        assert not any("OVER slab" in m for m in new_msgs)


def test_all_zero_image_bit_identical(monkeypatch):
    img, d = _scene(61, 47, "disc")
    img[:] = 0.0
    t = _tensor()
    new, _ = _render_on(img, d, t)
    monkeypatch.setattr(lc, "_over_scatter_slab", _old_two_pass_shim)
    old, _ = _render_on(img, d, t)
    for a, b in zip(new, old):
        assert np.array_equal(a, b)
        assert np.isfinite(a).all()


def test_production_buffers_are_float32_and_shapes_match(monkeypatch):
    """The fused multiply writes into a complex64 ``buf``; a float64 premult
    would be silently downcast (fftconvolve would have stayed complex128).
    Pin that production only ever feeds float32 2-D matte / 3-ch premult."""
    seen = []
    real = lc._over_scatter_slab

    def spy(**kw):
        seen.append((kw["premult_pad"].dtype, kw["matte_pad"].dtype,
                     kw["matte_pad"].ndim, kw["basis_w"].dtype,
                     kw["conv_a"].shape))
        return real(**kw)

    monkeypatch.setattr(lc, "_over_scatter_slab", spy)
    img, d = _scene(61, 47, "disc")
    _render_on(img, d, _tensor())
    assert seen
    for p, m, nd, bw, ca in seen:
        assert p == np.float32 and m == np.float32 and bw == np.float32
        assert nd == 2 and ca == (61, 47)


def test_geometry_guard_rejects_mismatched_fft_state():
    H, W, pad = 21, 17, 4
    prem = np.zeros((H + 2 * pad, W + 2 * pad, 3), np.float32)
    matte = np.zeros((H + 2 * pad + 1, W + 2 * pad), np.float32)  # off by one
    fs = cv._build_image_fft_state(prem, prem, prem, 2 * pad + 1)
    with pytest.raises(AssertionError, match="geometry mismatch"):
        lc._over_scatter_slab(
            d_idx=1, premult_pad=prem, matte_pad=matte,
            basis_w=np.zeros((2, H, W), np.float32),
            wedge_idx=np.zeros((H, W), np.int32), kernels=None,
            defocus_ratios=None, kH=9, kW=9, pad=pad, pix_scale=1.0,
            scaled_ksize=2 * pad + 1, psf_norm="l1", H=H, W=W, fft_state=fs,
            conv_c=np.zeros((H, W, 3), np.float32),
            conv_a=np.zeros((H, W), np.float32), cancel=None,
        )


def test_cancel_on_second_over_slab_matches_old(monkeypatch):
    """Cancel set at the 2nd OVER progress line: both paths raise, and the
    progress transcript up to the raise is identical."""
    img, d = _scene(61, 47, "three")
    t = _tensor()

    def go():
        ev = threading.Event()
        msgs: list[str] = []

        def prog(m):
            msgs.append(m)
            if sum("[occlusion] OVER slab" in x for x in msgs) == 2:
                ev.set()

        with pytest.raises(CancelledError):
            render_with_defocus(img, t, dmap=d, occlusion_compositing="on",
                                on_progress=prog, cancel=ev)
        return msgs

    new = go()
    monkeypatch.setattr(lc, "_over_scatter_slab", _old_two_pass_shim)
    old = go()
    assert new == old
    assert sum("OVER slab" in m for m in new) == 2
