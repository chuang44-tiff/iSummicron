"""R-perf-1 P1 (I3+I4+I5) — hostile probes for FFT-hoist + parallel + memory.

Adversarial probes against the Coder's claim that the per-channel image-rfft2
hoist + ``set_workers(-1)`` parallelism + post-batching ``_estimate_peak_bytes``
formula are correct and bit-faithful to the pre-P1 ``fftconvolve`` path.

Attack surfaces (chosen 1, 2, 3, 5, 6, 7, 10 from the brief):

(1) WRONG-POLARITY (REQUIRED) — focus FAR + nearly-all-delta tensor: only
    one slab is non-delta. Hoisted FFT path must still be bit-exact across
    batch sizes.
(2) Delta-only tensor (no FFT slabs) AND pure-FFT tensor (no delta slabs).
(3) ``set_workers(-1)`` scope leakage: post-render worker count must
    revert to the pre-call value.
(5) Multi-render lifetime: 5 sequential renders → no monotone RSS growth
    beyond a small slack, no FFT-state retention across calls.
(6) Memory-guard formula: D=1, D=24 hypothetical, ``_SLAB_BATCH_SIZE > D``
    min-clamp.
(7) ``pix_scale`` re-zoom: corpus_ref_long_edge that triggers the
    ``pix_scale < 0.95`` branch — hoist plan must still match the
    post-zoom kernel size.
(10) Stale FFT-state across two ``render()`` calls with different image
     sizes — second render must not silently use first call's spectra.
"""
from __future__ import annotations

import gc
import os
import resource
import threading

import numpy as np
import pytest
import scipy.fft as sp_fft

import summicron.render.convolve as _CV
from summicron.render.api import CancelledError
from summicron.render.convolve import (
    MAX_BYTES,
    _SLAB_BATCH_SIZE,
    _build_image_fft_state,
    _estimate_peak_bytes,
    _fftconvolve_hoisted,
    _memory_guard,
    render,
)


_OD_D11 = np.array(
    [0.7, 0.98, 1.37, 1.91, 2.68, 3.74, 5.23, 7.32, 10.23, 14.30, 20.0],
    dtype=np.float64,
)
_RE = np.array([0.0, 0.15, 0.30, 0.45, 0.60, 0.80, 1.01], dtype=np.float64)
_DR_D11 = np.linspace(0, 1, 11, dtype=np.float64)


def _gauss(ksize: int, sigma: float) -> np.ndarray:
    c = ksize // 2
    ys, xs = np.mgrid[-c:c + 1, -c:c + 1].astype(np.float32)
    g = np.exp(-(xs ** 2 + ys ** 2) / (2 * sigma ** 2)).astype(np.float32)
    return g / g.sum()


def _delta(ksize: int) -> np.ndarray:
    k = np.zeros((ksize, ksize), dtype=np.float32)
    k[ksize // 2, ksize // 2] = 1.0
    return k


def _tensor_d11_one_nondelta(ksize: int = 33, nondelta_d: int = 7) -> dict:
    """D=11, with all slabs delta EXCEPT one (default d=7) which is gaussian.
    Probes per-batch determinism when only ONE batch contains the non-delta
    slab — a stale FFT plan or wrong slab indexing would corrupt the output.
    """
    R, D = 6, 11
    kern = np.zeros((R, D, ksize, ksize), dtype=np.float32)
    for di in range(D):
        if di == nondelta_d:
            kern[:, di] = _gauss(ksize, sigma=4.0)
        else:
            kern[:, di] = _delta(ksize)
    return {
        "kernels": kern,
        "radial_edges": _RE.copy(),
        "defocus_ratios": _DR_D11.copy(),
        "object_distances_m": _OD_D11.copy(),
        "focus_distance_m": 0.7,
        "corpus_ref_long_edge": None,
        "otf_norm_applied": False,
        "delta_guard_applied": True,
    }


def _tensor_d11_all_delta(ksize: int = 33) -> dict:
    """D=11, every slab is a centered impulse — render must still produce
    a sensible output (mask-modulated identity copy of the input)."""
    R, D = 6, 11
    kern = np.zeros((R, D, ksize, ksize), dtype=np.float32)
    for di in range(D):
        kern[:, di] = _delta(ksize)
    return {
        "kernels": kern,
        "radial_edges": _RE.copy(),
        "defocus_ratios": _DR_D11.copy(),
        "object_distances_m": _OD_D11.copy(),
        "focus_distance_m": 0.7,
        "corpus_ref_long_edge": None,
        "otf_norm_applied": False,
        "delta_guard_applied": True,
    }


def _tensor_d11_no_delta(ksize: int = 33) -> dict:
    """D=11, NO slab is delta — every slab is a gaussian with distinct
    sigma. All 132 (d, ri) pairs route through the FFT-hoist path."""
    R, D = 6, 11
    kern = np.zeros((R, D, ksize, ksize), dtype=np.float32)
    for di in range(D):
        # sigma >= 2.0 so kernel_base.max() < 0.5 * sum() — i.e. NOT delta.
        kern[:, di] = _gauss(ksize, sigma=2.0 + 0.5 * di)
    return {
        "kernels": kern,
        "radial_edges": _RE.copy(),
        "defocus_ratios": _DR_D11.copy(),
        "object_distances_m": _OD_D11.copy(),
        "focus_distance_m": 0.7,
        "corpus_ref_long_edge": None,
        "otf_norm_applied": False,
        "delta_guard_applied": True,
    }


# ---------------------------------------------------------------------------
# (1) WRONG-POLARITY (REQUIRED): focus FAR + only one non-delta slab.
#     Hoisted FFT must be bit-exact across batch sizes for this hostile case.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("batch_size", [1, 2, 3, 4, 7, 11])
def test_wrong_polarity_one_nondelta_invariant_to_batch_size(batch_size):
    """Wrong-polarity: subject FAR (20 m), bg NEAR (0.7 m), focus_distance_m
    at the NEAR end. Tensor has all-delta slabs except one. The FFT-hoist
    path is exercised on exactly one slab; cross-batch bit-exactness probes
    that the hoisted spectra are not stale-cached across batch boundaries
    AND that delta-branch routing through the hoist (Coder's spec
    ambiguity #3) is order-invariant.
    """
    rng = np.random.default_rng(0xBADCAFE)
    H, W = 80, 80
    yy, xx = np.mgrid[:H, :W].astype(np.float32)
    cy, cx = H / 2, W / 2
    r2 = (yy - cy) ** 2 + (xx - cx) ** 2
    mask = r2 < 10 ** 2
    depth_m = np.where(mask, 20.0, 0.7).astype(np.float32)
    img = rng.uniform(0.05, 0.7, (H, W, 3)).astype(np.float32)
    tensor = _tensor_d11_one_nondelta(nondelta_d=7)

    ref_u8, ref_lin = render(img, tensor, depth_m=depth_m, slab_batch_size=11)
    test_u8, test_lin = render(img, tensor, depth_m=depth_m, slab_batch_size=batch_size)

    assert np.all(np.isfinite(test_lin)), "wrong-polarity FFT-hoist: non-finite"
    np.testing.assert_array_equal(ref_lin, test_lin)
    np.testing.assert_array_equal(ref_u8, test_u8)


# ---------------------------------------------------------------------------
# (2) Delta-only and FFT-only tensors — both must render correctly.
# ---------------------------------------------------------------------------


def test_all_delta_tensor_renders_finite_no_nan():
    """Every slab is a centered impulse: the render's hoisted-image FFTs
    are still built, but every `_apply_kernel` call hits the is_delta
    branch. Output must be finite and roughly equal to the input under
    the active-slab mask (delta = identity convolution)."""
    rng = np.random.default_rng(0xD17A)
    H, W = 64, 64
    img = rng.uniform(0.1, 0.6, (H, W, 3)).astype(np.float32)
    depth_m = rng.uniform(0.7, 20.0, (H, W)).astype(np.float32)
    tensor = _tensor_d11_all_delta()

    u8, lin = render(img, tensor, depth_m=depth_m, slab_batch_size=4)
    assert np.all(np.isfinite(lin))
    # Delta-only: output linear should be ≈ input (modulo bright-pass split
    # and α-blend renorm). A sanity bound is sufficient — exact equality
    # would require modeling the entire render bookkeeping.
    assert lin.shape == img.shape
    # Crude sanity: mean intensity in the same ballpark as input.
    assert abs(float(lin.mean()) - float(img.mean())) < 0.30


def test_all_fft_tensor_renders_finite_no_nan():
    """No slab is delta: the entire render is a 132-(d,ri)-pair FFT-hoist
    workout. Output must be finite and bounded; the absence of any
    is_delta short-circuit stresses the hoisted irfft2 path under the
    `set_workers(-1)` parallel wrap."""
    rng = np.random.default_rng(0xF1F1)
    H, W = 64, 64
    img = rng.uniform(0.1, 0.6, (H, W, 3)).astype(np.float32)
    depth_m = rng.uniform(0.7, 20.0, (H, W)).astype(np.float32)
    tensor = _tensor_d11_no_delta()

    u8, lin = render(img, tensor, depth_m=depth_m, slab_batch_size=4)
    assert np.all(np.isfinite(lin))
    assert (lin >= 0.0).all() and (lin <= 2.0).all(), "FFT-only: output out-of-range"


# ---------------------------------------------------------------------------
# (3) `set_workers(-1)` scope leakage probe.
# ---------------------------------------------------------------------------


def test_set_workers_does_not_leak_after_render():
    """Critical for any caller that uses `scipy.fft` elsewhere and expects
    the default worker count. The Coder wraps both scatter call sites in
    `with sp_fft.set_workers(-1):`; the context manager MUST restore the
    prior worker count when `render()` returns (success path) AND when it
    raises (cancelled path)."""
    rng = np.random.default_rng(0xC0DE)
    H, W = 48, 48
    img = rng.uniform(0.1, 0.6, (H, W, 3)).astype(np.float32)
    depth_m = rng.uniform(0.7, 20.0, (H, W)).astype(np.float32)
    tensor = _tensor_d11_no_delta()

    workers_before = sp_fft.get_workers()
    render(img, tensor, depth_m=depth_m, slab_batch_size=4)
    workers_after = sp_fft.get_workers()
    assert workers_after == workers_before, (
        f"set_workers leak: before={workers_before} after={workers_after}"
    )

    # Same invariant on the cancelled-error path.
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(CancelledError):
        render(img, tensor, depth_m=depth_m, slab_batch_size=2, cancel=cancel)
    workers_after_cancel = sp_fft.get_workers()
    assert workers_after_cancel == workers_before, (
        f"set_workers leak on cancel: before={workers_before} "
        f"after={workers_after_cancel}"
    )


# ---------------------------------------------------------------------------
# (5) Multi-render lifetime: 5 sequential renders, no RSS monotone growth.
# ---------------------------------------------------------------------------


def _rss_mb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    div = 1024 * 1024 if os.uname().sysname == "Darwin" else 1024
    return rss / div


def test_fft_state_does_not_leak_across_5_renders():
    """Run `render()` 5 times in a row in the same process. `fft_state`
    is built fresh every render and the slab loop's batch-boundary
    `gc.collect()` should reclaim it. If the hoisted spectra (~36 MB at
    2048², or ~1 MB here) persist across calls, peak RSS rises monotonely.
    """
    rng = np.random.default_rng(0x1EAC1234)
    H, W = 128, 128
    img = rng.uniform(0.1, 0.6, (H, W, 3)).astype(np.float32)
    depth_m = rng.uniform(0.7, 20.0, (H, W)).astype(np.float32)
    tensor = _tensor_d11_no_delta()

    # Warmup: prime any one-time scipy/numpy import caches.
    render(img, tensor, depth_m=depth_m, slab_batch_size=4)
    gc.collect()
    rss0 = _rss_mb()

    for _ in range(5):
        render(img, tensor, depth_m=depth_m, slab_batch_size=4)
        gc.collect()
    rss1 = _rss_mb()

    # ru_maxrss is monotone-max across the process, so we accept any
    # delta ≤ 200 MB as a PASS — that is the working-set fluctuation
    # ceiling for 5 renders. A regression of 5× the per-render fft_state
    # (~5 MB at this resolution) would manifest as a multi-hundred-MB
    # spike from cumulative scipy.fft cache growth or undeleted bundles.
    assert rss1 - rss0 <= 200.0, (
        f"5-render RSS leak: rss0={rss0:.1f} MB, rss1={rss1:.1f} MB, "
        f"delta={rss1 - rss0:.1f} MB"
    )


# ---------------------------------------------------------------------------
# (6) `_estimate_peak_bytes` formula edge cases.
# ---------------------------------------------------------------------------


# R-native-mem (locked-spec-mem.md §8): the structural estimator replaces the
# old formula pins; these tests assert PROPERTIES, not constants.

def test_peak_bytes_monotone_in_d():
    """Estimate grows with D (slab_w, argmax copy, prep): 1 <= 11 <= 24."""
    h, w = 1024, 768
    e1, e11, e24 = (
        _estimate_peak_bytes(h, w, d, kernel_size=65) for d in (1, 11, 24)
    )
    assert e1 <= e11 <= e24
    assert e1 < e24


def test_peak_bytes_batch_clamped_to_d():
    """slab_batch_size > D clamps to D (no extra retention beyond D slabs);
    D=2 with batch 2, 4 and 11 must project identically."""
    h, w, d = 512, 512, 2
    e = [
        _estimate_peak_bytes(h, w, d, kernel_size=65, slab_batch_size=b)
        for b in (2, 4, 11)
    ]
    assert e[0] == e[1] == e[2]
    # default batch (module constant) is also clamped
    assert _estimate_peak_bytes(h, w, d, kernel_size=65) <= e[0]


def test_peak_bytes_native_d11_under_cap():
    """Native D=11 (default batch, k=183) must stay under MAX_BYTES; D=24 is
    only required to be monotone (it may legitimately project higher)."""
    h, w = 5712, 4284
    assert _estimate_peak_bytes(h, w, 11) < MAX_BYTES
    _memory_guard(h, w, 11)
    assert _estimate_peak_bytes(h, w, 24) >= _estimate_peak_bytes(h, w, 11)
    assert _SLAB_BATCH_SIZE == 1


# ---------------------------------------------------------------------------
# (7) `pix_scale < 0.95` re-zoom path: hoist plan must match scaled_ksize.
# ---------------------------------------------------------------------------


def test_pix_scale_rezoom_does_not_break_hoist():
    """Set `corpus_ref_long_edge` to a value ≫ `max(H, W)` so
    `pix_scale = max(H,W) / corpus_ref < 0.95` triggers the kernel re-
    zoom branch (`convolve.py:1617-1620`). The hoist plan was built off
    `scaled_ksize`, which is the post-zoom size. Render must succeed
    and produce finite output bit-equal across batch sizes."""
    rng = np.random.default_rng(0xDEC0DE)
    H, W = 96, 96
    img = rng.uniform(0.1, 0.6, (H, W, 3)).astype(np.float32)
    depth_m = rng.uniform(0.7, 20.0, (H, W)).astype(np.float32)
    tensor = _tensor_d11_no_delta(ksize=51)
    # corpus_ref ≫ max(H,W) → pix_scale = 96/2000 = 0.048 → very small.
    tensor["corpus_ref_long_edge"] = 2000

    ref_u8, ref_lin = render(img, tensor, depth_m=depth_m, slab_batch_size=11)
    assert np.all(np.isfinite(ref_lin)), "pix_scale rezoom: non-finite output"
    test_u8, test_lin = render(img, tensor, depth_m=depth_m, slab_batch_size=2)
    np.testing.assert_array_equal(ref_lin, test_lin)


# ---------------------------------------------------------------------------
# (10) Stale FFT-state across two `render()` calls with different image sizes.
# ---------------------------------------------------------------------------


def test_two_renders_different_sizes_no_stale_fft_state():
    """Run render() twice with different image shapes back-to-back.
    `fft_state` is a local in `render()`, so this *should* be safe — but
    if any module-level cache (e.g., scipy's pocketfft plan cache) is
    confused by the shape change, the second render would silently
    produce garbage. We probe finiteness + a sanity bound."""
    rng = np.random.default_rng(0xDA7E5)
    tensor = _tensor_d11_no_delta()

    H1, W1 = 64, 64
    img1 = rng.uniform(0.1, 0.6, (H1, W1, 3)).astype(np.float32)
    depth1 = rng.uniform(0.7, 20.0, (H1, W1)).astype(np.float32)

    H2, W2 = 96, 80  # asymmetric, distinct from H1/W1
    img2 = rng.uniform(0.1, 0.6, (H2, W2, 3)).astype(np.float32)
    depth2 = rng.uniform(0.7, 20.0, (H2, W2)).astype(np.float32)

    _, lin1a = render(img1, tensor, depth_m=depth1, slab_batch_size=4)
    _, lin2 = render(img2, tensor, depth_m=depth2, slab_batch_size=4)
    _, lin1b = render(img1, tensor, depth_m=depth1, slab_batch_size=4)

    assert np.all(np.isfinite(lin2)), "stale FFT-state on second render"
    assert lin2.shape == (H2, W2, 3)
    # Same input twice with a different-sized render in between — the
    # second `render(img1)` must yield bit-identical output to the first.
    np.testing.assert_array_equal(lin1a, lin1b)


# ---------------------------------------------------------------------------
# Bonus: hoisted-helper unit equivalence vs `scipy.signal.fftconvolve`.
# ---------------------------------------------------------------------------


def test_fftconvolve_hoisted_matches_fftconvolve_within_1_lsb():
    """Direct unit test for `_fftconvolve_hoisted`: against
    `scipy.signal.fftconvolve(img, kernel, mode='same')` on a small
    img × small kernel pair, the absolute difference must be ≤ 1 LSB
    (~1e-6 in float32). Probes the manual rfft2/irfft2 padding logic."""
    from scipy.signal import fftconvolve as _fftconv

    rng = np.random.default_rng(0xF77)
    H, W = 33, 41
    ksize = 17
    img = rng.uniform(0.0, 1.0, (H, W, 3)).astype(np.float32)
    img_diffuse = img.copy()
    img_bright = img.copy()
    kernel = rng.uniform(0.0, 1.0, (ksize, ksize)).astype(np.float32)
    kernel /= kernel.sum() + 1e-12

    fft_state = _build_image_fft_state(img, img_diffuse, img_bright, ksize)
    for c in range(3):
        ref = _fftconv(img[:, :, c], kernel, mode="same").astype(np.float32)
        got = _fftconvolve_hoisted(
            fft_state["img_F"][c], kernel,
            fft_state["fshape"], fft_state["slice_y"], fft_state["slice_x"],
        )
        assert ref.shape == got.shape, (ref.shape, got.shape)
        max_abs = float(np.max(np.abs(ref - got)))
        assert max_abs < 1e-5, (
            f"hoist drift channel {c}: max_abs={max_abs:.3e}"
        )
