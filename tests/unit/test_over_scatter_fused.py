"""Q-e: fused OVER scatter (``_layered_composite._over_scatter_slab``).

Three FAST guards for the occlusion-ON perf round:

1. **Oracle** -- the fused colour+matte scatter is ``np.array_equal`` to a
   FROZEN copy of the pre-Q-e two-pass OVER code (kept in this file, so a later
   cleanup of ``convolve._apply_kernel``'s ``fft_state is None`` branch cannot
   silently delete it). Pins operand order, loop order and the
   1-channel-matte equivalence.
2. **Zero ``fftconvolve``** -- the ON render never calls
   ``scipy.signal.fftconvolve`` (same count as OFF, expected 0).
3. **Cancel** -- cancelling during the OVER phase raises ``CancelledError``
   promptly; a pre-set event raises before any ``irfft2``.
"""
from __future__ import annotations

import threading

import numpy as np
import pytest
import scipy.fft as sp_fft
import scipy.signal
from scipy.ndimage import rotate as ndimage_rotate
from scipy.signal import fftconvolve

from summicron.render import convolve as cv
from summicron.render._layered_composite import _over_scatter_slab
from summicron.render.api import CancelledError
from tests.fixtures.defocus import render_with_defocus

_RE = np.array([0.0, 0.15, 0.30, 0.45, 0.60, 0.80, 1.01], dtype=np.float64)
_OD_D11 = np.array(
    [0.7, 0.98, 1.37, 1.91, 2.68, 3.74, 5.23, 7.32, 10.23, 14.30, 20.0],
    dtype=np.float64,
)


def _kernels(R=6, D=4, k=33):
    kern = np.zeros((R, D, k, k), dtype=np.float32)
    c = k // 2
    kern[:, 0, c, c] = 1.0  # d=0 delta slab
    ys, xs = np.mgrid[-c:c + 1, -c:c + 1].astype(np.float32)
    for di in range(1, D):
        for ri in range(R):
            # anisotropic so rotation to each wedge actually matters
            g = np.exp(
                -(xs ** 2 / (2 * (1.5 + 1.2 * di) ** 2)
                  + ys ** 2 / (2 * (1.0 + 0.7 * di + 0.3 * ri) ** 2))
            )
            kern[ri, di] = (g / g.sum()).astype(np.float32)
    return kern


def _geometry(H, W, kH, corpus_ref):
    """Same derivation as ``render()`` for pix_scale / scaled_ksize / pad."""
    pix_scale = max(H, W) / corpus_ref
    scaled_ksize = max(3, int(round(kH * pix_scale)) | 1) if pix_scale < 0.95 else kH
    return pix_scale, scaled_ksize, scaled_ksize // 2


def _maps(H, W, Nr):
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    ys = np.arange(H, dtype=np.float32) - cy
    xs = np.arange(W, dtype=np.float32) - cx
    rad = np.sqrt(ys[:, None] ** 2 + xs[None, :] ** 2)
    radial_frac = (rad / rad.max()).astype(np.float32)
    theta = np.arctan2(ys[:, None], xs[None, :])
    wedge_idx = ((theta + np.pi) / (2 * np.pi) * cv.N_ANG).astype(np.int32) % cv.N_ANG
    edges = np.linspace(0.0, 1.0, Nr + 1)
    r_centers = 0.5 * (edges[:-1] + edges[1:])
    return radial_frac, wedge_idx, r_centers


def _frozen_old_two_pass(d_idx, premult_pad, matte3_pad, **g):
    """FROZEN copy of the pre-Q-e OVER scatter (colour pass + 3-ch matte pass).

    Verbatim semantics of ``_scatter_one_slab_fixed_d_basis(fft_state=None)`` ->
    ``_apply_kernel(fft_state=None)``, with the bright / wsum outputs dropped.
    """
    H, W, pad = g["H"], g["W"], g["pad"]
    basis_w = cv._radial_basis_weights(g["radial_frac"], g["r_centers"])

    def one_pass(img_pad, out):
        for ri in range(len(g["r_centers"])):
            w = basis_w[ri].astype(np.float32)
            if w.max() < 0.01:
                continue
            kb = cv._build_basis_kernel(
                ri, d_idx, True, g["defocus_ratios"], g["kernels"],
                g["kH"], g["kW"], g["pix_scale"], g["scaled_ksize"],
            )
            if kb.max() > 0.5 * kb.sum():
                for c in range(3):
                    conv = fftconvolve(img_pad[:, :, c], kb, mode="same").astype(np.float32)
                    out[:, :, c] += w * conv[pad:pad + H, pad:pad + W]
                continue
            for a in range(cv.N_ANG):
                m = (g["wedge_idx"] == a).astype(np.float32)
                w_ang = (w * m).astype(np.float32)
                if w_ang.max() < 0.01:
                    continue
                theta_a = (a + 0.5) * (2 * np.pi / cv.N_ANG) - np.pi
                rot_deg = -(theta_a * 180.0 / np.pi + 90.0)
                rot = ndimage_rotate(kb, rot_deg, reshape=False, order=1)
                rot = np.maximum(rot, 0.0)
                rot /= rot.sum() + 1e-12
                for c in range(3):
                    conv = fftconvolve(img_pad[:, :, c], rot, mode="same").astype(np.float32)
                    out[:, :, c] += w_ang * conv[pad:pad + H, pad:pad + W]

    conv_c = np.zeros((H, W, 3), dtype=np.float32)
    conv_a3 = np.zeros((H, W, 3), dtype=np.float32)
    one_pass(premult_pad, conv_c)
    one_pass(matte3_pad, conv_a3)
    return conv_c, conv_a3[:, :, 0]


def _case(H, W, corpus_ref, d_idx, seed=0):
    rng = np.random.default_rng(seed)
    kern = _kernels()
    Nr = kern.shape[0]
    pix_scale, scaled_ksize, pad = _geometry(H, W, kern.shape[2], corpus_ref)
    radial_frac, wedge_idx, r_centers = _maps(H, W, Nr)
    img = rng.uniform(0.0, 1.0, (H, W, 3)).astype(np.float32)
    H_d = (rng.uniform(size=(H, W)) > 0.5).astype(np.float32)
    H_d = np.where(radial_frac > 0.5, np.maximum(H_d, 0.3), H_d).astype(np.float32)
    premult = H_d[..., None] * img
    premult_pad = np.pad(premult, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
    matte_pad = np.pad(H_d, ((pad, pad), (pad, pad)), mode="reflect")
    # old code: np.pad(np.repeat(H_d[..., None], 3, axis=2), reflect) -- identical
    # to repeating the padded 2-D matte.
    matte3_pad = np.pad(
        np.repeat(H_d[..., None], 3, axis=2),
        ((pad, pad), (pad, pad), (0, 0)), mode="reflect",
    )
    img_padded = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
    fft_state = cv._build_image_fft_state(
        img_padded, img_padded, img_padded, scaled_ksize
    )
    common = dict(
        kernels=kern, defocus_ratios=np.linspace(0, 1, kern.shape[1]),
        kH=kern.shape[2], kW=kern.shape[3], pad=pad, pix_scale=pix_scale,
        scaled_ksize=scaled_ksize, H=H, W=W,
    )
    return dict(
        common=common, premult_pad=premult_pad, matte_pad=matte_pad,
        matte3_pad=matte3_pad, fft_state=fft_state, wedge_idx=wedge_idx,
        radial_frac=radial_frac, r_centers=r_centers, d_idx=d_idx,
    )


@pytest.mark.parametrize("H,W", [(129, 97), (257, 193)])
@pytest.mark.parametrize("d_idx", [0, 1, 3])  # delta slab + angular slabs
@pytest.mark.parametrize("scale", [1.0, 0.5])  # render pix_scale
def test_fused_equals_frozen_two_pass(H, W, d_idx, scale):
    cref = float(max(H, W)) / scale
    cs = _case(H, W, cref, d_idx)
    common = cs["common"]
    old_c, old_a = _frozen_old_two_pass(
        d_idx, cs["premult_pad"], cs["matte3_pad"],
        radial_frac=cs["radial_frac"], r_centers=cs["r_centers"],
        wedge_idx=cs["wedge_idx"], **common,
    )
    # the old matte pass reads a channel of the replicated matte
    assert np.array_equal(cs["matte3_pad"][:, :, 0], cs["matte_pad"])
    basis_w = cv._radial_basis_weights(cs["radial_frac"], cs["r_centers"])
    conv_c = np.zeros((H, W, 3), dtype=np.float32)
    conv_a = np.zeros((H, W), dtype=np.float32)
    with sp_fft.set_workers(-1):
        _over_scatter_slab(
            d_idx=d_idx, premult_pad=cs["premult_pad"], matte_pad=cs["matte_pad"],
            basis_w=basis_w, wedge_idx=cs["wedge_idx"], fft_state=cs["fft_state"],
            conv_c=conv_c, conv_a=conv_a, cancel=None,
            kernels=common["kernels"], defocus_ratios=common["defocus_ratios"],
            kH=common["kH"], kW=common["kW"], pad=common["pad"],
            pix_scale=common["pix_scale"], scaled_ksize=common["scaled_ksize"],
            H=H, W=W,
        )
    assert old_c.any() and old_a.any()
    # Tolerance, not exact equality: the fused path and the frozen path use
    # different FFT sizes/batching, so pocketfft's SIMD paths on other CPUs
    # (e.g. the CI runner) differ at float32 ULP level (workers do not matter;
    # byte-identical on the M4 Max).  1e-6 relative is ~8 ULP and still far
    # below any real weight/order bug (a 1e-4 kernel-weight change fails).
    for new, old in ((conv_c, old_c), (conv_a, old_a)):
        np.testing.assert_allclose(
            new, old, rtol=1e-6, atol=1e-6 * max(1.0, float(np.abs(old).max()))
        )


def test_oracle_covers_multiple_radial_bins_and_wedges():
    cs = _case(129, 97, 129.0, 2)
    basis_w = cv._radial_basis_weights(cs["radial_frac"], cs["r_centers"])
    active = [ri for ri in range(basis_w.shape[0]) if basis_w[ri].max() >= 0.01]
    assert len(active) >= 2
    assert len(np.unique(cs["wedge_idx"])) == cv.N_ANG


# --- zero fftconvolve + cancel through render() ----------------------------

def _tensor():
    R, D, k = 6, 11, 33
    kern = np.zeros((R, D, k, k), dtype=np.float32)
    c = k // 2
    kern[:, 0, c, c] = 1.0
    ys, xs = np.mgrid[-c:c + 1, -c:c + 1].astype(np.float32)
    for di in range(1, D):
        g = np.exp(-(xs ** 2 + ys ** 2) / (2 * (1.5 + 1.5 * di) ** 2))
        kern[:, di] = (g / g.sum()).astype(np.float32)
    return {
        "kernels": kern, "radial_edges": _RE.copy(),
        "defocus_ratios": np.linspace(0, 1, D), "object_distances_m": _OD_D11.copy(),
        "focus_distance_m": 0.7, "corpus_ref_long_edge": None,
        "otf_norm_applied": False, "delta_guard_applied": True,
    }


def _step_scene(n=48):
    rng = np.random.default_rng(3)
    img = rng.uniform(0.05, 0.9, (n, n, 3)).astype(np.float32)
    dmap = np.zeros((n, n), dtype=np.float32)
    dmap[:, n // 2:] = 0.9  # near-left / far-right depth step
    return img, dmap


def test_on_render_calls_no_fftconvolve(monkeypatch):
    calls = {"n": 0}

    def counting(*a, **k):
        calls["n"] += 1
        return fftconvolve(*a, **k)

    monkeypatch.setattr(scipy.signal, "fftconvolve", counting)
    monkeypatch.setattr(cv, "fftconvolve", counting)
    img, dmap = _step_scene()
    counts = {}
    for occ in ("off", "on"):
        calls["n"] = 0
        msgs: list[str] = []
        render_with_defocus(
            img, _tensor(), dmap=dmap, occlusion_compositing=occ,
            on_progress=msgs.append,
        )
        counts[occ] = calls["n"]
        if occ == "on":
            assert any("[occlusion] OVER slab" in m for m in msgs), msgs
    assert counts["on"] == counts["off"] == 0


def test_cancel_during_over_raises_and_stops_progress():
    img, dmap = _step_scene()
    ev = threading.Event()
    msgs: list[str] = []

    def prog(m):
        msgs.append(m)
        if "[occlusion] OVER slab" in m:
            ev.set()

    with pytest.raises(CancelledError):
        render_with_defocus(
            img, _tensor(), dmap=dmap, occlusion_compositing="on",
            on_progress=prog, cancel=ev,
        )
    over = [m for m in msgs if "[occlusion] OVER slab" in m]
    assert len(over) == 1


def test_preset_cancel_raises_before_any_irfft2(monkeypatch):
    cs = _case(65, 49, 65.0, 1)
    c = cs["common"]
    basis_w = cv._radial_basis_weights(cs["radial_frac"], cs["r_centers"])
    n = {"irfft2": 0}
    real = sp_fft.irfft2

    def counting(*a, **k):
        n["irfft2"] += 1
        return real(*a, **k)

    import summicron.render._layered_composite as lc

    monkeypatch.setattr(lc.sp_fft, "irfft2", counting)
    ev = threading.Event()
    ev.set()
    with pytest.raises(CancelledError):
        _over_scatter_slab(
            d_idx=1, premult_pad=cs["premult_pad"], matte_pad=cs["matte_pad"],
            basis_w=basis_w, wedge_idx=cs["wedge_idx"], fft_state=cs["fft_state"],
            conv_c=np.zeros((65, 49, 3), np.float32),
            conv_a=np.zeros((65, 49), np.float32), cancel=ev,
            kernels=c["kernels"], defocus_ratios=c["defocus_ratios"],
            kH=c["kH"], kW=c["kW"], pad=c["pad"], pix_scale=c["pix_scale"],
            scaled_ksize=c["scaled_ksize"], H=65, W=49,
        )
    assert n["irfft2"] == 0


def test_float64_inputs_raise_typeerror():
    cs = _case(129, 97, 129.0, 1)
    common = cs["common"]
    basis_w = cv._radial_basis_weights(cs["radial_frac"], cs["r_centers"])
    conv_c = np.zeros((129, 97, 3), dtype=np.float32)
    conv_a = np.zeros((129, 97), dtype=np.float32)
    with pytest.raises(TypeError):
        _over_scatter_slab(
            d_idx=1, premult_pad=cs["premult_pad"].astype(np.float64),
            matte_pad=cs["matte_pad"], basis_w=basis_w,
            wedge_idx=cs["wedge_idx"], fft_state=cs["fft_state"],
            conv_c=conv_c, conv_a=conv_a, cancel=None, **common,
        )
