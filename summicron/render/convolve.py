"""Bin-and-convolve renderer: apply 4D PSF tensor to an input image.

Pipeline: load -> linearize -> defocus map -> bin-and-convolve -> finalize -> save.
No OpenCV. Float32 throughout. Bilinear kernel interpolation across radial/defocus axes.
"""
from __future__ import annotations

import argparse
import gc
import logging
import math
import os
import sys
import threading
import warnings
from pathlib import Path
from typing import Callable

import numpy as np
import scipy.fft as sp_fft
from scipy.ndimage import rotate as ndimage_rotate
from scipy.ndimage import zoom as ndimage_zoom
from scipy.signal import fftconvolve


# ---------------------------------------------------------------------------
# Typed errors (P1.audit Item 4)
# ---------------------------------------------------------------------------


class TensorValidationError(ValueError):
    """Raised by :func:`load_tensor` when a render tensor fails schema /
    metadata validation. ``ValueError`` subclass so existing
    ``except ValueError`` blocks in CLI shims keep catching it; the
    distinct type lets the GUI surface a tensor-specific error message.
    """


# P1.audit Item 4: schema_version compatibility. Bumped when the on-disk
# tensor layout changes incompatibly. Older tensors without a
# `schema_version` key load with a one-time warning to preserve back-compat
# for the canonical CodeV tensor (no key today).
_SUPPORTED_TENSOR_SCHEMAS = ("1.0",)

# P1.audit Item 3: progress callback type alias. Bound at module level so
# the slab-loop checkpoints can typecheck without re-importing the API.
_OnProgress = Callable[[str], None] | None


def _emit(on_progress: _OnProgress, msg: str) -> None:
    """Route a per-stage status string to the caller's progress sink, or
    fall back to ``print(msg)`` when no sink was supplied.

    ``on_progress=None`` is the CLI default; in that mode this helper is
    a thin wrapper over the existing ``print()`` calls so stdout stays
    byte-identical to the pre-P1.audit-Item-3 CLI.
    """
    if on_progress is None:
        print(msg)
    else:
        on_progress(msg)

_logger = logging.getLogger(__name__)

# P1.12 R2: depth-layered gather module state. Layered gather is the SOLE
# rendering path (CLI-1a: legacy fallback + vestigial kwargs deleted).
_MAX_ACTIVE_SLABS = 8
_EMPTY_SKIP_THRESH = 0.01

# R-perf-1 I1: slab-batching driver. The depth-layered gather loop processes
# slabs in serial batches of `_SLAB_BATCH_SIZE` to bound RAM. Per-batch
# working set (per_slab_alpha + per_slab_outputs_d/b) is freed at the batch
# boundary; running running argmin buffers carry the holes-fill state across
# batches. R-native-mem item 1 (doc/plan/phase2/section17_native-memory/
# locked-spec-mem.md): default 4 -> 1. Batching was introduced for RAM, not
# speed (section1_perf-research/probe-round-1.md:68); at the gather peak every
# retained slab beyond the first holds 7 planes (alpha + two (H,W,3) outputs),
# about 2 GB at native with batch 4. The output is bit-identical for every
# batch size; N > 1 only costs memory. User-tunable via `--slab-batch-size N`
# (CLI) or `slab_batch_size=` kwarg on render().
_SLAB_BATCH_SIZE = 1

# P1.10.2 Cut 3: bright-pass smoothstep knee half-width. Was a CLI flag
# (--bright-pass-knee) and render() kwarg (bright_knee); demoted to a
# module constant after `project_bright_knee_midtone_crush.md` SUPERSEDED
# established that K is empirically inert at the canonical T=1e9 sentinel.
_BRIGHT_KNEE = 0.20

# Alpha-blend identity-snap knee for the layered-gather path: pixels with
# defocus ratio below this snap to the in-focus (identity) kernel.
D_KNEE_LAYERED = 0.10



# ---------------------------------------------------------------------------
# sRGB transfer functions (IEC 61966-2-1)
# ---------------------------------------------------------------------------

def downres_array(image: np.ndarray, target_long_edge: int) -> np.ndarray:
    """Downsample *image* so its long edge equals *target_long_edge*.

    Never upscales: if the image is already <= target, it is returned unchanged.
    LDR path uses PIL Lanczos (bit-identical to pre-fix for LDR input).
    HDR path (``image.max() > 1.001``) uses ``scipy.ndimage.zoom`` bilinear so
    values above 1.0 survive the resize (the PIL path clips to uint8).
    Accepts and returns float32 HxWxC arrays.
    """
    h, w = image.shape[:2]
    long_edge = max(h, w)
    if long_edge <= target_long_edge:
        return image
    scale = target_long_edge / long_edge
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    if image.max() > 1.001:
        # HDR path: preserve float32 headroom via scipy bilinear zoom.
        zoom_factors = (new_h / h, new_w / w, 1.0)
        return ndimage_zoom(image, zoom_factors, order=1).astype(np.float32)
    # LDR path: unchanged (bit-identical to pre-fix for LDR input).
    from PIL import Image as _PILImage
    pil = _PILImage.fromarray(
        np.clip(image * 255.0, 0, 255).astype(np.uint8), mode="RGB"
    ).resize((new_w, new_h), _PILImage.LANCZOS)
    return np.asarray(pil, dtype=np.float32) / 255.0


def srgb_to_linear(srgb: np.ndarray) -> np.ndarray:
    """Invert the sRGB transfer function.  Input & output in [0, 1]."""
    srgb = np.asarray(srgb, dtype=np.float32)
    return np.where(
        srgb <= 0.04045,
        srgb / 12.92,
        np.power((srgb + 0.055) / 1.055, 2.4),
    ).astype(np.float32)


def linear_to_srgb(linear: np.ndarray) -> np.ndarray:
    """Apply sRGB gamma.  Input & output in [0, 1]."""
    linear = np.asarray(linear, dtype=np.float32)
    return np.where(
        linear <= 0.0031308,
        12.92 * linear,
        1.055 * np.power(np.maximum(linear, 0.0), 1.0 / 2.4) - 0.055,
    ).astype(np.float32)


# ---------------------------------------------------------------------------
# P1.13 K11 — LDR-anchored per-channel band-gated shadow match
# ---------------------------------------------------------------------------

def _ldr_shadow_match_gain(
    render_linear: np.ndarray,
    ldr_primary_linear: np.ndarray,
    shadow_band_hi: float = 0.30,
    smoothstep_width: float = 0.10,
    max_gain: float = 4.0,
    strength: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """K11: per-channel band-gated ratio gain to lift render shadows toward
    LDR-primary per-channel shadow means. Fixes brightness AND color temp /
    chromaticity in a single step.

    Identity by construction:
    - render_chan_mean ~ ldr_chan_mean for every channel + strength=1.0 ->
      gain_c=1.0 (no-op).
    - shape mismatch / < 64 valid shadow pixels -> gain=1.0 (degeneracy).
    - raw_c * strength < 1.0 -> clamp to 1.0 (one-sided lift).
    - raw_c * strength > max_gain -> clamp to max_gain.

    Shadow band defined on LDR-primary LUMA (single mask, shared across
    channels) so band edges don't introduce chromatic halo.

    Args:
        render_linear: (H, W, 3) linear-sRGB; HDR or LDR.
        ldr_primary_linear: (H, W, 3) linear-sRGB in [0, 1].
        shadow_band_hi: LDR-luma cutoff. Default 0.30.
        smoothstep_width: smoothstep transition width. Default 0.10.
        max_gain: per-channel upper clamp. Default 4.0.
        strength: multiplier on raw ratio (default 1.0). >1 overshoots,
            <1 undershoots. Multiplied BEFORE clamp.

    Returns:
        (gain_per_pixel, gains_rgb): gain_per_pixel is (H, W, 3) float32
        with values in [1.0, max_gain]; gains_rgb is (3,) float32 - the
        post-strength, post-clamp per-channel gains for diagnostics.
    """
    # F1: ndim/channel guard — helper is now a public-ish API consumed by
    # tests; reject 2D grayscale or non-RGB inputs gracefully.
    if render_linear.ndim != 3 or render_linear.shape[-1] != 3:
        H, W = render_linear.shape[:2] if render_linear.ndim >= 2 else (0, 0)
        return (
            np.ones((H, W, 3), dtype=np.float32),
            np.ones(3, dtype=np.float32),
        )
    H, W, _ = render_linear.shape
    if ldr_primary_linear.shape != render_linear.shape:
        return (
            np.ones((H, W, 3), dtype=np.float32),
            np.ones(3, dtype=np.float32),
        )

    ldr_luma = (
        0.2126 * ldr_primary_linear[..., 0]
        + 0.7152 * ldr_primary_linear[..., 1]
        + 0.0722 * ldr_primary_linear[..., 2]
    )
    shadow_mask_bool = ldr_luma <= shadow_band_hi
    n_shadow = int(shadow_mask_bool.sum())
    if n_shadow < 64:
        return (
            np.ones((H, W, 3), dtype=np.float32),
            np.ones(3, dtype=np.float32),
        )

    # Smoothstep mask: 1 deep in shadow, 0 above shadow_band_hi.
    lo = max(0.0, shadow_band_hi - smoothstep_width)
    hi = min(1.0, shadow_band_hi)
    t = np.clip((ldr_luma - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    band_mask = 1.0 - t * t * (3.0 - 2.0 * t)

    gains_rgb = np.zeros(3, dtype=np.float32)
    gain_per_pixel = np.ones((H, W, 3), dtype=np.float32)
    for c in range(3):
        ldr_c_mean = float(np.clip(
            ldr_primary_linear[..., c][shadow_mask_bool].mean(), 1e-6, 1.0
        ))
        rnd_c_mean = float(np.clip(
            render_linear[..., c][shadow_mask_bool].mean(), 1e-6, None
        ))
        # F2: NaN/Inf guard — defense-in-depth (caller's render() already
        # asserts no-NaN, but the helper shouldn't propagate garbage if
        # that ever weakens).
        if not np.isfinite(rnd_c_mean) or rnd_c_mean <= 1e-5:
            gains_rgb[c] = 1.0
            continue
        raw = ldr_c_mean / rnd_c_mean
        boosted = raw * strength
        clamped = float(np.clip(boosted, 1.0, max_gain))
        gains_rgb[c] = clamped
        gain_per_pixel[..., c] = 1.0 + (clamped - 1.0) * band_mask

    return gain_per_pixel, gains_rgb


# ---------------------------------------------------------------------------
# Image loading (HEIC via pillow-heif, JPEG/PNG via Pillow)
# ---------------------------------------------------------------------------

def load_image(path: str | Path) -> np.ndarray:
    """Load an image as float32 RGB in [0, 1] (sRGB gamma, not linearized)."""
    from PIL import Image

    try:
        import pillow_heif
        pillow_heif.register_heif_opener()
    except ImportError:
        pass  # Pillow alone handles JPEG/PNG

    with Image.open(str(path)) as im:
        im = im.convert("RGB")
        arr = np.asarray(im, dtype=np.uint8)
    return arr.astype(np.float32) / 255.0


# ---------------------------------------------------------------------------
# Memory guard (P1.audit Item 6; R-native-mem items 6-7)
# ---------------------------------------------------------------------------
#
# R-native-mem (doc/plan/phase2/section17_native-memory/locked-spec-mem.md):
# the guard is now an HONEST structural plane-sum estimate (never below the
# measured phys_footprint on the checked scene x size x mode x batch grid) and
# a fixed 12 GiB cap. The pixel-only ``MAX_PIXELS`` envelope stays a fast-path
# early gate (``_pixel_guard``, called at the same call site as before so error
# order and "no allocation before rejection" are unchanged); the byte budget
# (``_memory_guard``) runs just after ``scaled_ksize`` is known, before
# ``np.pad`` and every large allocation.
#
# The on-entry "Memory budget" line goes ONLY to ``on_progress`` so CLI stdout
# stays byte-identical (canonical-bench G1 pins).

MAX_PIXELS = 36_000_000  # 6000 x 6000 — fast-path early gate
MAX_BYTES = 12 * 1024**3  # 12 GiB fixed cap (a deterministic constant; CI-safe)
_GIB = 1024**3

# Process baseline outside the numpy plane model: interpreter + numpy/scipy/
# PIL/objc import footprint + allocator slack. The ONLY free constant of the
# estimator (no fitted per-pixel coefficient). Measured render-entry footprint:
# 0.85 GB at 1024 (Researcher B) and 1.06 GB at native (Researcher A).
PLATFORM_BASE = 1 * _GIB
# Small-array allowance (ys_r/xs_r vectors, per-bin kernels, kernel spectra
# bookkeeping): measured < 0.4 MB at 2048; a flat 1 MiB, NOT per pixel.
_SMALL_ALLOC = 1 << 20


class MemoryBudgetError(RuntimeError):
    """Raised when the estimated peak exceeds the byte budget.

    ``suggested_downres`` is the largest long edge (multiple of 64, >= 256)
    whose estimate fits the budget at the same aspect ratio and parameters
    (``kernel_size`` held fixed — conservative), or ``None`` if none fits.
    """

    def __init__(self, message: str, *, estimated_bytes: int,
                 budget_bytes: int, suggested_downres: int | None) -> None:
        super().__init__(message)
        self.estimated_bytes = int(estimated_bytes)
        self.budget_bytes = int(budget_bytes)
        self.suggested_downres = suggested_downres


def _budget_bytes() -> int:
    """Byte budget, read per call: ``ISUMMICRON_MAX_BYTES`` or ``MAX_BYTES``."""
    raw = os.environ.get("ISUMMICRON_MAX_BYTES")
    if raw is None:
        return int(MAX_BYTES)
    try:
        val = int(raw)
    except ValueError:
        val = 0
    if val <= 0:
        raise ValueError(
            f"ISUMMICRON_MAX_BYTES must be an integer > 0 (bytes); got {raw!r}"
        )
    return val


def _fft_geometry(h: int, w: int, kernel_size: int, *, padded: bool = False) -> dict:
    """Shared FFT/padding geometry for the estimator and the FFT hoist.

    ``h, w`` are the unpadded image dims (or, with ``padded=True``, the
    already-padded buffer dims). Returns ``pad``, ``padded_shape``,
    ``full_shape`` (linear-conv shape), ``fshape`` (next-fast-len FFT shape;
    what scipy's ``fftconvolve`` picks) and ``spec_shape`` (rfft2 output).
    """
    k = int(kernel_size)
    pad = k // 2
    if padded:
        hp, wp = int(h), int(w)
    else:
        hp, wp = int(h) + 2 * pad, int(w) + 2 * pad
    full = (hp + k - 1, wp + k - 1)
    fshape = (
        sp_fft.next_fast_len(full[0], real=True),
        sp_fft.next_fast_len(full[1], real=True),
    )
    return {
        "pad": pad,
        "padded_shape": (hp, wp),
        "full_shape": full,
        "fshape": fshape,
        "spec_shape": (fshape[0], fshape[1] // 2 + 1),
    }


def _estimate_peak_breakdown(
    h: int, w: int, d: int, *,
    occlusion: bool = False,
    kernel_size: int = 183,
    n_radial: int = 12,
    slab_batch_size: int | None = None,
) -> dict:
    """Structural plane-sum projection of the render's peak working set.

    Units: ``P`` = one unpadded float32 plane (``H*W*4`` B), ``Pp`` = one
    padded plane, ``S`` = one complex64 rfft2 spectrum (``fshape0 *
    (fshape1//2+1) * 8`` B; also bounds one real ``fshape`` plane). Every term
    cites the ``convolve.py`` / ``_layered_composite.py`` line family it models
    and counts only buffers live at that line. WORST CASE: all ``d`` slabs and
    all ``n_radial`` bins active. No fitted coefficient; the one free constant
    is ``PLATFORM_BASE``.

    Returns a dict with ``pipeline``, ``prep``, ``gather``, ``over`` (0 when
    ``occlusion`` is False), ``base`` and ``total``::

        total = pipeline + max(prep, gather, over) + base

    ``pipeline`` holds the planes that exist outside the phases (``img_lin``
    3P, ``depth_m`` P, ``dmap`` P). Each phase is the max over its sequential
    sub-phases (a transient counts only alongside what is live at that line).

    Outside the model (see the module docs): the native HEIC load (the HDR
    payload before ``render()``), monocular depth (torch model resident) and
    the Core Image tonemap leak after ``render()``.
    """
    h, w, d = int(h), int(w), int(d)
    nr = int(n_radial)
    geo = _fft_geometry(h, w, kernel_size)
    hp, wp = geo["padded_shape"]
    f0, f1 = geo["spec_shape"]
    P = h * w * 4
    Pp = hp * wp * 4
    S = f0 * f1 * 8
    batch = max(1, min(int(slab_batch_size or _SLAB_BATCH_SIZE), d))

    # Caller-held planes: img_lin (3P) and depth_m (P). ``dmap`` (P, built by
    # np.interp inside render(), :1603) lives through every phase, so it is
    # added to each phase below (keeps ``tracemalloc delta <= max(phases)``
    # exact; the total is the same 5P as counting it here).
    pipeline = 3 * P + P

    # --- prep (render() :1686-1760) -------------------------------------
    # Live through prep: radial_frac_map P (:1641), wedge_idx int32 P (:1670),
    # img_padded 3Pp (:1686).
    r_pre = 2 * P + 3 * Pp
    # bright split (:1702-1727): lum Pp, region-aware bright weight (~5 Pp
    # temporaries inside), bright_mask Pp, img_bright_padded + img_diffuse_padded.
    prep_bright = r_pre + 6 * Pp + 5 * Pp
    # fft_build (:1734, _build_image_fft_state): 3 padded buffers done; img_F
    # and img_diffuse_F spectra (3S each) done while img_bright_F builds
    # (3S list + 3S np.stack) + 1S pocketfft zero-pad/scratch.
    r_pad = 2 * P + 9 * Pp
    prep_fft = r_pad + 6 * S + 6 * S + S
    # compute_slab_weights (:802-863): spectra built; slab_w (D,H,W) + per-slab
    # temporaries (~6P) in the loop, or slab_w + quotient + wsum at the end
    # (the rebinding frees the input before ``astype``).
    prep_slab_w = r_pad + 9 * S + max(d * P + 6 * P, 2 * d * P + 4 * P)
    prep = max(prep_bright, prep_fft, prep_slab_w) + P + _SMALL_ALLOC  # + dmap

    # --- gather (_layered_gather_diffuse :1216-1405; OFF and ON regime-a) ---
    # Resident: radial P, wedge P, 9 padded planes, 9 spectra, slab_w (D,H,W).
    r_g = 2 * P + 9 * Pp + 9 * S + d * P
    accum = 9 * P  # output 3P, output_bright 3P, weight_sum, delta_weight_sum, acc_alpha (:1248-1254)
    # entry: np.argmax(slab_w, axis=0) copies the (D,H,W) array and returns
    # int64 (2P) before the int32 cast (P) (:1266).
    gather_entry = r_g + accum + d * P + 3 * P
    # steady state through the loop: argmax_d_full P + best_d_dist P +
    # best_d_color_d/b 6P (:1266-1271).
    r_loop = r_g + accum + 8 * P
    # per active slab: out_d 3P + out_b_d 3P + wsum_d P + delta_wsum_d P (:1313-1316)
    # + alpha_d P; stale loop locals of the previous slab stay bound until
    # rebound (wsum_d_safe P, dist_d_for_this P, dist_masked_d P, validity_d +
    # better P/2: ~3.5P, :1338/:1365-1369); the other (batch-1) slabs of the
    # batch are retained at 7P each (:1352-1355).
    slab_cur = 9 * P + 7 * P // 2 + (batch - 1) * 7 * P
    # _scatter_one_slab_fixed_d_basis (:1170-1213): dense basis_w (Nr,H,W)
    # (:1194) lives for the whole scatter.
    # (i) its build (:773-793): peak transient ~12P beyond basis_w itself
    #     (measured by tracemalloc: ri_lo/ri_hi, rc_lo/rc_hi/span/t, yy/xx,
    #     fancy-index copies).
    basis_build = nr * P + 12 * P
    # (ii) per-(ri,a) FFT step (_apply_kernel :1128-1166): w P, angular_mask P,
    #     w_ang P, two kernel spectra 2S, product S + irfft2 out S + 1S pocketfft
    #     internals (:1155-1166), crop astype P + w_ang*crop P.
    fft_step = nr * P + 5 * P + 5 * S
    gather_slab = r_loop + slab_cur + max(basis_build, fft_step)
    # batch-boundary streaming holes-fill (:1357-1378): all batch slabs held
    # (7P each) + per-slab int32 temporaries (~5P) + np.where colour copies 3P.
    gather_fill = r_loop + 9 * P + batch * 7 * P + 9 * P
    # render() tail after the gather (:1826-1921): accumulators 9P, output =
    # output + output_bright 3P, alpha P, blend products 3P+3P+3P,
    # output_linear copy 3P, (slab_w and, for OFF, the spectra still bound).
    tail_res = r_pad + d * P + (0 if occlusion else 9 * S)
    gather_tail = tail_res + accum + 3 * P + P + 9 * P + 3 * P
    gather = max(gather_entry, gather_slab, gather_fill, gather_tail) + P + _SMALL_ALLOC  # + dmap

    # --- over (ON, regime-b; _layered_composite :297-577) -----------------
    over = 0
    if occlusion:
        # Resident: radial P, wedge P, 9 padded planes, slab_w; the 9
        # spectra are DROPPED (S5, :381); gather returns 9P of accumulators.
        r_o = 2 * P + 9 * Pp + d * P + accum
        # entry: s_full P (+ ~9P selector temporaries, :106-143), s_active
        # P/4, np.argmax(slab_w) copy D*P + int64 2P (:397).
        over_entry = r_o + P + P // 4 + max(9 * P, d * P + 3 * P)
        # held across the OVER loop: s_full P, argmax_d P, coc_px P,
        # _interior_cover P, edge/boundary_band/s_reach 3 x P/4, acc_c 3P,
        # acc_a P (:397-452).
        hold = 9 * P
        # dense OVER basis_w (:459) + its build transients.
        over_basis = r_o + hold + nr * P + 12 * P
        # per-slab (:461-547): H_d P (+2P where/maximum temporaries),
        # premult 3P, premult_pad 3Pp (+ the previous slab's 3Pp until
        # rebound), matte_pad Pp, conv_c 3P, conv_a P (:474-519); fused OVER
        # scatter (_over_scatter_slab :238-255): prem_F 3S + matte_F S + buf S
        # + ker_F S + irfft2 out S + 1S pocketfft (:238-251), w P, w_ang P,
        # crop astype P + product P (:264-279, :252).
        # The previous slab's conv_c 3P / conv_a P / one_minus_acc P stay bound
        # (layer_premult_c, layer_alpha, one_minus_acc) until rebound: +5P.
        stale_o = 5 * P
        over_setup = (
            r_o + hold + nr * P + stale_o + 3 * P + 3 * P + 6 * Pp + Pp + 4 * P
        )
        over_fft = (
            r_o + hold + nr * P + stale_o + P + 3 * P + 3 * Pp + Pp + 3 * P + P
            + 8 * S + 4 * P
        )
        # OVER fold temporaries (:545-547): 3P product + 3P new acc_c, 2P acc_a.
        over_fold = (
            r_o + hold + nr * P + stale_o + 3 * P + 3 * Pp + Pp + 4 * P + 8 * P
        )
        over_loop = max(over_setup, over_fft, over_fold)
        # tail (:566-575): out_over 3P (+ divide/clip/astype temporaries ~4P),
        # s*out_over 3P, (1-s)*off 3P + P, sum 3P, astype 3P.
        over_tail = r_o + hold + 13 * P + 3 * P
        over = max(over_entry, over_basis, over_loop, over_tail) + P + _SMALL_ALLOC  # + dmap

    return {
        "pipeline": pipeline,
        "prep": prep,
        "gather": gather,
        "over": over,
        "base": PLATFORM_BASE,
        "total": pipeline + max(prep, gather, over) + PLATFORM_BASE,
    }


def _estimate_peak_bytes(
    h: int, w: int, d: int, *,
    occlusion: bool = False,
    kernel_size: int = 183,
    n_radial: int = 12,
    slab_batch_size: int | None = None,
) -> int:
    """Projected peak bytes (``_estimate_peak_breakdown(...)["total"]``).

    See :func:`_estimate_peak_breakdown` for the model and what it excludes.
    """
    return int(_estimate_peak_breakdown(
        h, w, d, occlusion=occlusion, kernel_size=kernel_size,
        n_radial=n_radial, slab_batch_size=slab_batch_size,
    )["total"])


def _pixel_guard(h: int, w: int, d: int | None = None, **kw) -> None:
    """Fast-path early gate: reject > ``MAX_PIXELS`` before any allocation.

    When ``d`` (and the ``_estimate_peak_bytes`` keywords) are given, the
    ``--downres`` suggestion also fits the byte budget, so following it
    passes both guards. Without them the suggestion is pixel-only.
    """
    if h * w > MAX_PIXELS:
        sug = _pixel_downres(h, w)
        if d is not None:
            mem = _suggest_downres(h, w, d, _budget_bytes(), **kw)
            if mem is not None:
                sug = min(sug, mem)
        raise RuntimeError(
            f"Image too large: {h}x{w} = {h * w} pixels > {MAX_PIXELS}; "
            f"pass --downres {sug} to reduce."
        )


def _pixel_downres(h: int, w: int) -> int:
    """Largest long edge (multiple of 64, >= 64) whose ``downres_array``
    output (aspect-preserving, rounded) is within ``MAX_PIXELS``."""
    long_edge = max(h, w)
    le = max(64, int((long_edge * (MAX_PIXELS / (h * w)) ** 0.5) // 64) * 64)
    while le > 64:
        sc = le / long_edge
        if max(1, round(h * sc)) * max(1, round(w * sc)) <= MAX_PIXELS:
            break
        le -= 64
    return le


def _suggest_downres(h: int, w: int, d: int, budget: int, **kw) -> int | None:
    """Largest long edge (multiple of 64, >= 256) that fits ``budget``."""
    long_edge = max(h, w)
    scale_h, scale_w = h / long_edge, w / long_edge

    def fits(le: int) -> bool:
        hh = max(1, int(round(scale_h * le)))
        ww = max(1, int(round(scale_w * le)))
        return _estimate_peak_bytes(hh, ww, d, **kw) <= budget

    lo, hi = 256 // 64, max(256 // 64, (long_edge - 1) // 64)
    if not fits(lo * 64):
        return None
    while lo < hi:  # largest m in [lo, hi] with fits(m*64) (monotone)
        mid = (lo + hi + 1) // 2
        if fits(mid * 64):
            lo = mid
        else:
            hi = mid - 1
    return lo * 64


def _memory_guard(
    h: int, w: int, d: int, *,
    on_progress: _OnProgress = None,
    occlusion: bool = False,
    kernel_size: int = 183,
    n_radial: int = 12,
    slab_batch_size: int | None = None,
) -> None:
    """Reject renders projected to exceed the byte budget.

    Runs ``_pixel_guard`` first (so direct 3-positional callers keep their
    behaviour), then compares :func:`_estimate_peak_bytes` with
    :func:`_budget_bytes`. Raises :class:`MemoryBudgetError` (a
    ``RuntimeError``; the CLI exits 2) with a ``--downres`` suggestion.
    """
    kw = dict(occlusion=occlusion, kernel_size=kernel_size,
              n_radial=n_radial, slab_batch_size=slab_batch_size)
    _pixel_guard(h, w, d, **kw)
    budget = _budget_bytes()
    peak_bytes = _estimate_peak_bytes(h, w, d, **kw)
    # G1 bit-id: only emit the budget diagnostic when a caller registered
    # a progress sink (CLI path keeps canonical-bench stdout byte-identical).
    if on_progress is not None:
        on_progress(
            f"  Memory budget: H={h} W={w} D={d} → "
            f"~{peak_bytes / _GIB:.2f} GiB peak; cap {budget / _GIB:.1f} GiB."
        )
    if peak_bytes > budget:
        sug = _suggest_downres(h, w, d, budget, **kw)
        if sug is not None:
            hint = f"pass --downres {sug}"
        elif budget < PLATFORM_BASE:
            hint = ("budget below the fixed platform overhead; "
                    "raise ISUMMICRON_MAX_BYTES")
        else:
            hint = "pass --downres N (smaller than 256)"
        raise MemoryBudgetError(
            f"Render would exceed {budget / _GIB:.1f} GiB budget "
            f"(estimated {peak_bytes / _GIB:.2f} GiB at "
            f"H={h} W={w} D={d}); {hint}"
            f"{'' if sug is None and budget < PLATFORM_BASE else ' to reduce'}.",
            estimated_bytes=peak_bytes, budget_bytes=budget,
            suggested_downres=sug,
        )


# ---------------------------------------------------------------------------
# P1.10: region-aware bright-pass weight (morphological top-hat × smoothstep)
# ---------------------------------------------------------------------------

def _compute_region_aware_bright_weight(
    lum: np.ndarray,
    bright_thresh: float,
    region_size: int,
    region_floor: float,
    region_norm_pct: float,
) -> np.ndarray:
    """Region-aware bright-pass weight.

    Multiplies the existing luminance smoothstep by a morphological-white-top-
    hat detector (``L − opening(L, region_size)``), normalized by its
    ``region_norm_pct``-th percentile and floor-blended for safety on large
    uniform highlights. Returns ``(H, W)`` float32 in ``[0, 1]``.

    ``region_size <= 0`` is the kill-switch — returns the smoothstep alone,
    bit-identical to the pre-P1.10 inline path. See
    ``doc/plan/phase1/section16/locked-spec-P1_10.md`` § 4.

    P1.10.2 Cut 3: bright-pass-knee parameter dropped; reads module constant
    ``_BRIGHT_KNEE = 0.20`` directly.
    """
    K = _BRIGHT_KNEE
    bw_lum = np.clip(
        (lum - (bright_thresh - K)) / (2.0 * K + 1e-12), 0.0, 1.0
    )
    bw_lum = bw_lum * bw_lum * (3.0 - 2.0 * bw_lum)  # smoothstep — UNCHANGED
    if region_size <= 0:
        return bw_lum.astype(np.float32)             # KILL-SWITCH
    from scipy.ndimage import grey_opening
    opened = grey_opening(lum, size=region_size)
    th = np.maximum(lum - opened, 0.0)
    # T8: use nanpercentile so a stray NaN/Inf in `lum` does not poison the
    # global anchor (spec § 7 NaN/Inf safety). Falls back to 1e-6 floor if
    # the entire input is non-finite.
    p = float(np.nanpercentile(th, region_norm_pct))
    th_anchor = max(p if np.isfinite(p) else 0.0, 1e-6)
    th_norm = np.clip(th / th_anchor, 0.0, 1.0)
    th_clipped = region_floor + (1.0 - region_floor) * th_norm
    return (bw_lum * th_clipped).astype(np.float32)


# ---------------------------------------------------------------------------
# Load 4D tensor
# ---------------------------------------------------------------------------

def _is_default_tensor(path) -> bool:
    """True iff ``path`` resolves to the wheel-shipped canonical tensor.

    Never raises: a missing artifact / unresolvable path is simply "not the
    default" (so the schema_version warning still fires for foreign files).
    """
    try:
        from summicron.render._artifacts import default_tensor_path

        return Path(path).resolve() == default_tensor_path().resolve()
    except (OSError, ValueError, FileNotFoundError, TypeError):
        return False


def load_tensor(path: str | Path) -> dict:
    """Load .npz with keys: kernels(R,D,H,W), radial_edges(R+1), defocus_ratios(D).

    P1.9.3: ``R`` and ``D`` are shape-derived from ``kernels.shape[:2]``
    and exposed in the return dict as ``"R"``/``"D"`` int keys. The
    ``radial_edges`` and ``object_distances_m`` lengths are validated
    against the derived ``(R, D)``; mismatch raises ``ValueError``.

    P1.audit Item 4 hardening:
    - ``allow_pickle=False`` (rejects pickled-object payloads from
      untrusted GUI/user sources).
    - Required-keys check up-front with a typed
      :class:`TensorValidationError`.
    - Optional ``schema_version`` key: missing ⇒ warn (preserves back-compat
      for the existing in-tree CodeV tensor); unknown value ⇒ raise.
    - Square + odd kernel check (basis path requires odd ``kH``).
    - Finite + monotonic ``radial_edges`` and ``defocus_ratios``.
    - Positive-finite ``object_distances_m`` + ``corpus_ref_long_edge``.
    - Focus-distance compare uses ``math.isclose(rel_tol=1e-6)`` rather
      than strict ``==`` (regenerator-precision robustness).
    """
    # P1.audit Item 4: reject pickled object arrays from untrusted sources.
    data = np.load(str(path), allow_pickle=False)

    # Required keys gate (P1.audit Item 4). Surface ALL missing keys at once
    # so the user sees a complete rebuild list in a single error.
    # NOTE: object_distances_m, focus_distance_m, corpus_ref_long_edge are
    # OPTIONAL on read with sensible defaults — many in-tree test fixtures
    # and intermediate (pre-downscale) tensors omit them, and the existing
    # code paths have established back-compat semantics (see
    # tests/adversarial/test_downscale_tensor_adversarial.py
    # ::test_load_tensor_back_compat_missing_corpus_ref). When PRESENT,
    # they are still validated for finite/monotonic/positive constraints
    # below.
    required = ("kernels", "radial_edges", "defocus_ratios")
    missing = [k for k in required if k not in data.files]
    if missing:
        raise TensorValidationError(
            f"load_tensor({path}): missing required keys {missing}; "
            f"required={list(required)}"
        )

    # schema_version: optional with one-time warning if missing; error on
    # unknown value. Preserves back-compat for the in-tree CodeV tensor
    # (which has no schema_version key today).
    if "schema_version" in data.files:
        sv = data["schema_version"]
        sv_str = (
            sv.item().decode() if isinstance(sv.item(), bytes) else str(sv.item())
        )
        if sv_str not in _SUPPORTED_TENSOR_SCHEMAS:
            raise TensorValidationError(
                f"load_tensor({path}): unsupported schema_version={sv_str!r}; "
                f"supported={list(_SUPPORTED_TENSOR_SCHEMAS)}"
            )
    elif not _is_default_tensor(path):
        # CLI-1b: the wheel-shipped canonical tensor is known-legacy; only
        # foreign tensors warrant the notice.
        warnings.warn(
            f"load_tensor({path}): tensor has no `schema_version` key; "
            f"assuming legacy 1.0 layout. Future tensors should declare "
            f"schema_version='1.0'.",
            stacklevel=2,
        )

    kernels = data["kernels"].astype(np.float32)
    if kernels.ndim != 4:
        raise TensorValidationError(
            f"kernels must be 4D (R, D, H, W); got shape {kernels.shape}"
        )
    R, D = int(kernels.shape[0]), int(kernels.shape[1])
    kH, kW = int(kernels.shape[2]), int(kernels.shape[3])

    # P1.9.4 L11 — kernel-array validation hardening (4 P1.9.3 xfails flip).
    # Zero-dim gate: clear ValueError before any indexing; otherwise the
    # downstream _bilinear_weights raises a cryptic IndexError("axis 0 with
    # size 0").
    if R < 1 or D < 1:
        raise TensorValidationError(
            f"kernels has zero-size R or D axis: shape={kernels.shape}"
        )
    # P1.audit Item 4: square check is a hard schema requirement (every
    # downstream path indexes kernels[r, d] as a square spatial slab).
    if kH != kW:
        raise TensorValidationError(
            f"kernels must be square; got (kH, kW)=({kH}, {kW})"
        )
    # Odd-sided check is a render-path constraint (basis path centres
    # padding); raw pre-downscale intermediate tensors are commonly
    # even-sized (e.g. 2048x2048 from CodeV) and load_tensor is also
    # called on those by tests + downscale tooling. Warn-only at load
    # time; the basis-kernel builder will error with a localised message
    # if it actually receives an even-sided tensor at render time.
    if kH % 2 == 0:
        import warnings as _w
        _w.warn(
            f"load_tensor({path}): kernels are even-sided ({kH}x{kW}); "
            "render basis path requires odd-sided. Likely an intermediate "
            "tensor; downscale + delta-guard install before render.",
            stacklevel=2,
        )
    # Non-finite gate: NaN/Inf would propagate silently to render() and emerge
    # as garbled output much later. Report first non-finite multi-index for
    # debug.
    if not np.all(np.isfinite(kernels)):
        bad = tuple(int(x) for x in np.argwhere(~np.isfinite(kernels))[0])
        raise TensorValidationError(
            f"kernels contains non-finite values (NaN/Inf); first at index {bad}"
        )
    # Negative-value gate: PSFs are physically non-negative. Tolerance is
    # -1e-6 (NOT strict < 0) per locked spec L11 — cubic-zoom + np.clip(0)
    # legitimately leaves ULP noise at ~1e-7 in valid post-interp tensors.
    if np.any(kernels < -1e-6):
        raise TensorValidationError(
            f"kernels contains negative values; PSFs must be non-negative. "
            f"min value = {float(kernels.min())}"
        )

    radial_edges = data["radial_edges"].astype(np.float64)
    if len(radial_edges) != R + 1:
        raise TensorValidationError(
            f"radial_edges length {len(radial_edges)} != R+1 = {R + 1} "
            f"(R derived from kernels.shape[0]={R})"
        )
    if not np.all(np.isfinite(radial_edges)):
        raise TensorValidationError("radial_edges contains non-finite values")
    if not np.all(np.diff(radial_edges) > 0):
        raise TensorValidationError("radial_edges must be strictly increasing")

    defocus_ratios = data["defocus_ratios"].astype(np.float64)
    if not np.all(np.isfinite(defocus_ratios)):
        raise TensorValidationError("defocus_ratios contains non-finite values")
    if len(defocus_ratios) >= 2 and not np.all(np.diff(defocus_ratios) > 0):
        raise TensorValidationError(
            "defocus_ratios must be strictly increasing"
        )

    # P1.audit Item 4 (NARROWED): corpus_ref_long_edge is OPTIONAL on read
    # with the established 5976 default (back-compat with the legacy CodeV
    # tensor schema). When PRESENT it is still validated for int + positive.
    if "corpus_ref_long_edge" in data.files:
        corpus_ref_raw = data["corpus_ref_long_edge"]
        try:
            corpus_ref = int(corpus_ref_raw)
        except (TypeError, ValueError) as e:
            raise TensorValidationError(
                f"corpus_ref_long_edge must be an integer; got {corpus_ref_raw!r}"
            ) from e
        if corpus_ref <= 0:
            raise TensorValidationError(
                f"corpus_ref_long_edge must be positive; got {corpus_ref}"
            )
    else:
        corpus_ref = 5976  # Established default for legacy tensors w/o key.

    # P1.10.2 Cut 2: OTF-norm path removed in P1.9; the otf_norm_applied flag
    # is no longer read here. delta_guard_applied is the only safeguard.
    delta_guard_applied = (
        bool(data["delta_guard_applied"]) if "delta_guard_applied" in data.files else False
    )
    if not delta_guard_applied:
        warnings.warn(
            f"load_tensor: {path} has delta_guard_applied=False; rendering "
            "through a raw tensor without the S2 delta guard will smear "
            "center-in-focus detail. Regenerate via "
            "summicron.psf_model.codev_to_render + downscale_tensor.",
            stacklevel=2,
        )

    # P1.audit Item 4 (NARROWED): object_distances_m / focus_distance_m
    # are OPTIONAL on read with sensible defaults (None). When PRESENT
    # they are still validated for shape + finite + positive + monotonic +
    # focus-consistency. Tests + intermediate pre-downscale tensors
    # commonly omit these (back-compat preserved).
    object_distances_m = None
    focus_distance_m = None
    if "object_distances_m" in data.files:
        object_distances_m = data["object_distances_m"].astype(np.float64)
        if len(object_distances_m) != D:
            raise TensorValidationError(
                f"object_distances_m length {len(object_distances_m)} != D = {D} "
                f"(D derived from kernels.shape[1]={D})"
            )
        if not np.all(np.isfinite(object_distances_m)):
            raise TensorValidationError(
                "object_distances_m contains non-finite values"
            )
        if not np.all(object_distances_m > 0):
            raise TensorValidationError(
                "object_distances_m must be strictly positive (meters)"
            )
        if not np.all(np.diff(object_distances_m) > 0):
            raise TensorValidationError(
                "object_distances_m must be monotonically increasing"
            )

    if "focus_distance_m" in data.files:
        focus_distance_m = float(data["focus_distance_m"])
        if not (np.isfinite(focus_distance_m) and focus_distance_m > 0):
            raise TensorValidationError(
                f"focus_distance_m must be positive finite; got {focus_distance_m}"
            )
        # P1.audit Item 4: was `assert object_distances_m[0] == focus_distance_m`;
        # math.isclose tolerates ~ULP regenerator drift (a strict `==` over
        # float-cast metadata trips on bit-rotted but semantically-equal tensors).
        # Only checked when BOTH metadata pieces are present.
        if object_distances_m is not None and not math.isclose(
            float(object_distances_m[0]), focus_distance_m, rel_tol=1e-6
        ):
            raise TensorValidationError(
                f"object_distances_m[0]={object_distances_m[0]} != "
                f"focus_distance_m={focus_distance_m} (rel_tol=1e-6)"
            )

    return {
        "kernels": kernels,
        "radial_edges": radial_edges,
        "defocus_ratios": defocus_ratios,
        "corpus_ref_long_edge": corpus_ref,
        "delta_guard_applied": delta_guard_applied,
        "object_distances_m": object_distances_m,
        "focus_distance_m": focus_distance_m,
        "R": R,
        "D": D,
    }


# ---------------------------------------------------------------------------
# Bilinear kernel interpolation (S1, S2)
# ---------------------------------------------------------------------------

def _bilinear_weights(
    radial_frac: float,
    defocus_val: float,
    radial_edges: np.ndarray,
    defocus_ratios: np.ndarray,
) -> tuple[list[tuple[int, int]], list[float]]:
    """Bilinear interpolation across radial bins and defocus ratios.

    Returns 4 (r_idx, d_idx) tuples and 4 corresponding weights that sum to 1.
    Indices are clamped to [0, N-2] on both axes so the 4 corners always exist.

    Post-OTF-normalization (v2 spec §6), ``kernels[0, 0]`` is an exact
    single-pixel delta by construction and the four-corner blend is
    numerically well-behaved with the delta corner participating. The
    former S2 delta-kernel guard (nearest-non-delta fallback when
    ``d_idx == 0``) has been removed.
    """
    # Radial axis: use bin centers
    r_centers = 0.5 * (radial_edges[:-1] + radial_edges[1:])
    Nr = len(r_centers)
    Nd = len(defocus_ratios)

    # Locate lower-left cell on the radial axis
    r_lo = int(np.clip(np.searchsorted(r_centers, radial_frac) - 1, 0, Nr - 2))
    r_hi = r_lo + 1
    r_span = float(r_centers[r_hi] - r_centers[r_lo])
    if r_span < 1e-12:
        r_t = 0.0
    else:
        r_t = float(np.clip((radial_frac - r_centers[r_lo]) / r_span, 0.0, 1.0))

    # Locate lower-left cell on the defocus axis
    d_lo = int(np.clip(np.searchsorted(defocus_ratios, defocus_val) - 1, 0, Nd - 2))
    d_hi = d_lo + 1
    d_span = float(defocus_ratios[d_hi] - defocus_ratios[d_lo])
    if d_span < 1e-12:
        d_t = 0.0
    else:
        d_t = float(np.clip((defocus_val - defocus_ratios[d_lo]) / d_span, 0.0, 1.0))

    # 4 corners + bilinear weights
    indices = [
        (r_lo, d_lo),
        (r_lo, d_hi),
        (r_hi, d_lo),
        (r_hi, d_hi),
    ]
    weights = [
        (1.0 - r_t) * (1.0 - d_t),
        (1.0 - r_t) * d_t,
        r_t * (1.0 - d_t),
        r_t * d_t,
    ]

    return indices, weights


# ---------------------------------------------------------------------------
# P1.10.1: per-pixel radial-basis weight tensor
# ---------------------------------------------------------------------------

def _radial_basis_weights(
    radial_frac_map: np.ndarray,
    r_centers: np.ndarray,
) -> np.ndarray:
    """Per-pixel piecewise-linear partition-of-unity over Nr radial-bin basis indices.

    For every pixel ``(y, x)``, compute weights ``basis_w[ri, y, x]`` such that
    ``basis_w`` linearly interpolates ``radial_frac_map[y, x]`` against the
    ``r_centers`` axis. AT MOST 2 of the ``Nr`` planes are non-zero per pixel
    (the bracket of ``radial_frac`` in ``r_centers``); exactly 1 plane is 1.0
    when ``radial_frac`` lands exactly on a node. ``Σ_ri basis_w[ri, y, x] = 1``
    by construction (PoU within float32 round-off, ≤ 1e-6).

    Boundary clamp: at ``radial_frac < r_centers[0]``, full weight on
    ``ri=0``; at ``radial_frac > r_centers[-1]``, full weight on ``ri=Nr-1``.

    This is the vectorized r-axis half of ``_bilinear_weights:343-349``.
    Returns shape ``(Nr, H, W)`` float32 — **scalar collapse is a critical
    bug** (see locked-spec-P1_10_1.md § S8.3); without per-pixel dependence
    the basis blend reduces to legacy slab structure with NO ring fix.

    Parameters
    ----------
    radial_frac_map : (H, W) float32
        Per-pixel radial fraction in [0, 1+ε].
    r_centers : (Nr,) float64 / float32
        Radial bin centers, monotonically increasing.

    Returns
    -------
    basis_w : (Nr, H, W) float32
        PoU weights summing to 1 along axis 0.
    """
    rc = np.asarray(r_centers, dtype=np.float32)
    Nr = int(rc.shape[0])
    rf = np.asarray(radial_frac_map, dtype=np.float32)
    H, W = rf.shape
    if Nr <= 0:
        raise ValueError("r_centers must have length >= 1")
    if Nr == 1:
        # Degenerate: single basis bin → weight 1 everywhere.
        return np.ones((1, H, W), dtype=np.float32)

    # Locate lower bracket: ri_lo[y,x] in [0, Nr-2] such that
    # rc[ri_lo] <= rf < rc[ri_lo+1] (clamped).
    ri_lo = np.clip(
        np.searchsorted(rc, rf, side="right") - 1, 0, Nr - 2
    ).astype(np.int32)
    ri_hi = ri_lo + 1
    rc_lo = rc[ri_lo]
    rc_hi = rc[ri_hi]
    span = rc_hi - rc_lo
    span_safe = np.where(span < 1e-12, np.float32(1.0), span)
    t = np.clip((rf - rc_lo) / span_safe, 0.0, 1.0).astype(np.float32)
    # Boundary clamp: where rf < rc[0] all weight on ri=0; where rf > rc[-1]
    # all weight on ri=Nr-1. searchsorted handles this naturally:
    # rf < rc[0] → ri_lo=0, t clamped to 0 → 100% on ri=0.
    # rf >= rc[-1] → ri_lo=Nr-2, t clamped to 1 → 100% on ri=Nr-1.

    basis_w = np.zeros((Nr, H, W), dtype=np.float32)
    # Scatter (1-t) onto ri_lo plane and t onto ri_hi plane.
    yy, xx = np.indices((H, W), dtype=np.int32)
    basis_w[ri_lo, yy, xx] += (1.0 - t).astype(np.float32)
    basis_w[ri_hi, yy, xx] += t

    return basis_w


# ---------------------------------------------------------------------------
# P1.12 R2: depth-layered gather helpers
# ---------------------------------------------------------------------------

def compute_slab_weights(
    depth_m: np.ndarray, distance_grid: np.ndarray
) -> np.ndarray:
    """Per-pixel smoothstep partition-of-unity over D distance bins.

    For D bins at `distance_grid[d]` (monotonically increasing meters), return
    a (D, H, W) float32 tensor `slab_w` such that `Σ_d slab_w[d](x) = 1` at
    every pixel. Uses cubic smoothstep `3t² - 2t³` with support ±2·bin around
    each bin center. Endpoints clamp: slab 0 gets weight 1 for `depth < d_0`,
    slab D-1 gets weight 1 for `depth > d_{D-1}`. Finally row-normalized to
    enforce partition-of-unity exactly (handles numerical drift + edge clamp).

    See doc/plan/phase1/section12/locked-spec-R2-layered.md § "Slab weights".
    """
    depth_m = np.asarray(depth_m, dtype=np.float32)
    grid = np.asarray(distance_grid, dtype=np.float32)
    D = grid.shape[0]
    H, W = depth_m.shape

    # Build per-slab triangle-of-support [d_{d-1}, d_d, d_{d+1}, d_{d+2}] with
    # width ±2·bin (measured in grid units — effectively the distance between
    # neighboring bin centers). The smoothstep rises on [d_{d-1}, d_d] and
    # falls on [d_{d+1}, d_{d+2}] so neighbor bins overlap ~50%.
    slab_w = np.zeros((D, H, W), dtype=np.float32)
    for d in range(D):
        # Left foot: d_{d-1} (clamped to d_0 if d==0).
        left_foot = grid[d - 1] if d > 0 else grid[0]
        left_peak = grid[d]
        # Right peak: d_{d+1}; right foot: d_{d+2}.
        right_peak = grid[d + 1] if d + 1 < D else grid[-1]
        right_foot = grid[d + 2] if d + 2 < D else grid[-1]

        w_slab = np.zeros((H, W), dtype=np.float32)
        # Rising edge: smoothstep on [left_foot, left_peak]
        if left_peak > left_foot:
            mask_rise = (depth_m >= left_foot) & (depth_m < left_peak)
            t = (depth_m - left_foot) / (left_peak - left_foot)
            t = np.clip(t, 0.0, 1.0).astype(np.float32)
            w_slab = np.where(mask_rise, t * t * (3.0 - 2.0 * t), w_slab)
        # Flat peak: [left_peak, right_peak]
        mask_peak = (depth_m >= left_peak) & (depth_m <= right_peak)
        w_slab = np.where(mask_peak, np.float32(1.0), w_slab)
        # Falling edge: smoothstep on [right_peak, right_foot]
        if right_foot > right_peak:
            mask_fall = (depth_m > right_peak) & (depth_m <= right_foot)
            t = (depth_m - right_peak) / (right_foot - right_peak)
            t = np.clip(t, 0.0, 1.0).astype(np.float32)
            w_slab = np.where(mask_fall, (1.0 - t) * (1.0 - t) * (3.0 - 2.0 * (1.0 - t)), w_slab)

        # Endpoint clamp: first/last bins fully cover out-of-range extrapolations.
        if d == 0:
            w_slab = np.where(depth_m < grid[0], np.float32(1.0), w_slab)
        if d == D - 1:
            w_slab = np.where(depth_m > grid[-1], np.float32(1.0), w_slab)

        slab_w[d] = w_slab

    # Row-normalize so Σ_d slab_w[d](x) = 1 exactly (G5 gate).
    wsum = slab_w.sum(axis=0)
    wsum = np.maximum(wsum, 1e-12)
    # R-native-mem item 2 (in-place normalise) was REVERTED by its own keep rule:
    # the 2048 tracemalloc ablation measured an in-call peak drop of 9P
    # (113 MB), not the >= 2*D*P - 1 MiB (276 MB) the spec requires, because
    # rebinding ``slab_w`` already frees the input before the ``astype`` copy.
    # See doc/plan/phase2/section17_native-memory/coder-mem-report.md.
    slab_w = slab_w / wsum[None, :, :]
    return slab_w.astype(np.float32)


N_ANG = 8


def _build_basis_kernel(
    ri, d_center_or_idx, use_idx, defocus_ratios, kernels,
    kH, kW, psf_norm, pix_scale, scaled_ksize,
):
    """Build the (kH, kW) raw-tensor kernel for a single radial-basis index ri.

    Input is a fixed `ri` (basis index into the raw R=12 tensor), output is
    one kernel; the per-pixel blend happens later inside the basis scatter
    functions (`_scatter_one_slab_fixed_d_basis`)
    via `_radial_basis_weights[ri, y, x]`.

    D-axis stance: D-axis remains 2-tap bilinear at the same scalar
    `d_center` as legacy. Per-pixel D-axis blend is OUT OF SCOPE for P1.10.1
    (preserves P1.9.4 R-cubic / D-linear interp scheme; memory
    `project_codev_axes_facts.md`). Layered path pins one `d_idx` per
    slab via `use_idx=True`.

    Tail (pix_scale zoom + L1/peak renorm) preserves per-kernel L1/peak
    semantics. P1.10.4 C3: slab counterpart deleted; basis is the only
    kernel-build path.
    """
    if use_idx:
        # Layered path: pin a single D-index, no D-axis blend.
        kernel_base = kernels[int(ri), int(d_center_or_idx)].astype(np.float32)
    else:
        # Legacy path: 2-tap D-bilinear at scalar d_center.
        Nd = len(defocus_ratios)
        d_lo = int(np.clip(
            np.searchsorted(defocus_ratios, float(d_center_or_idx)) - 1, 0, Nd - 2
        ))
        d_hi = d_lo + 1
        d_span = float(defocus_ratios[d_hi] - defocus_ratios[d_lo])
        d_t = 0.0 if d_span < 1e-12 else float(
            np.clip((float(d_center_or_idx) - defocus_ratios[d_lo]) / d_span, 0.0, 1.0)
        )
        kernel_base = (
            (1.0 - d_t) * kernels[int(ri), d_lo]
            + d_t * kernels[int(ri), d_hi]
        ).astype(np.float32)

    # --- Tail: zoom + L1/peak renorm. ---
    if psf_norm == "peak":
        kernel_base = kernel_base / max(float(kernel_base.max()), 1e-10)

    # Q-b R-infocus-delta: the S2 d0 delta survives cubic ndimage_zoom as a
    # sub-pixel-offset blob (zoom grid is not centre-aligned). A STRICT delta
    # (exactly one nonzero, at (kH//2, kW//2)) maps to an
    # exact centred delta at the scaled size. Off-centre deltas and blended
    # (d_t > 0) kernels fall through to the ordinary zoom.
    # d>=1 centring → B4 / R-highlight-guard (cubic misfires as is_delta on
    # 52-64/132 slices; area resample loses ~70% bright energy; Q-b R1/R2).
    if pix_scale < 0.95:
        nz = np.argwhere(kernel_base != 0)
        if (
            len(nz) == 1
            and int(nz[0][0]) == kH // 2
            and int(nz[0][1]) == kW // 2
            and np.isfinite(kernel_base[kH // 2, kW // 2])
            and kernel_base[kH // 2, kW // 2] > 0
        ):
            from summicron.psf_model.downscale_tensor import _make_delta

            return _make_delta(scaled_ksize)

    if pix_scale < 0.95:
        pre_zoom_l1 = float(kernel_base.sum())
        kernel_base = ndimage_zoom(kernel_base, pix_scale, order=3).astype(np.float32)
        zh, zw = kernel_base.shape
        if zh != scaled_ksize or zw != scaled_ksize:
            tmp = np.zeros((scaled_ksize, scaled_ksize), dtype=np.float32)
            cy, cx = zh // 2, zw // 2
            sy = max(0, cy - scaled_ksize // 2)
            sx = max(0, cx - scaled_ksize // 2)
            ey = min(zh, sy + scaled_ksize)
            ex = min(zw, sx + scaled_ksize)
            dy = (scaled_ksize - (ey - sy)) // 2
            dx = (scaled_ksize - (ex - sx)) // 2
            tmp[dy:dy + (ey - sy), dx:dx + (ex - sx)] = kernel_base[sy:ey, sx:ex]
            kernel_base = tmp
        kernel_base = np.maximum(kernel_base, 0.0)
        ksum = kernel_base.sum()
        if ksum > 1e-10:
            if psf_norm == "l1":
                kernel_base /= ksum
            else:
                kernel_base /= max(float(kernel_base.max()), 1e-10)
        else:
            kernel_base = np.zeros((scaled_ksize, scaled_ksize), dtype=np.float32)
            kernel_base[scaled_ksize // 2, scaled_ksize // 2] = pre_zoom_l1

    return kernel_base


def _build_image_fft_state(
    img_padded: np.ndarray,
    img_diffuse_padded: np.ndarray,
    img_bright_padded: np.ndarray,
    kernel_size: int,
) -> dict:
    """R-perf-1 I3: hoist per-channel image rfft2 to once-per-render.

    The slab/legacy scatter loops call ``fftconvolve(img, kernel,
    mode="same")`` ~ ``D × Nr × (1 + 2 × N_ANG)`` times per render. The
    image side of every one of those FFTs is identical — only the
    kernel changes. Pre-computing ``rfft2(img)`` once per channel
    (3 channels × 3 buffers = 9 complex64 spectra ≈ 36 MB at 2048²)
    saves ~2/3 of the FFT work, since each call now needs only the
    kernel FFT + multiply + inverse.

    The returned bundle stores:

    * ``fshape``: the next-fast-len FFT shape (matches scipy's internal
      ``fftconvolve`` choice so per-call output is bit-identical to
      ``fftconvolve(..., mode="same")`` modulo float-reduction order).
    * ``full_shape``: linear-conv shape (img + kernel - 1).
    * ``slice_y, slice_x``: ``mode="same"`` crop slices.
    * ``img_F, img_diffuse_F, img_bright_F``: per-channel rfft2
      spectra of shape ``(3, fshape[0], fshape[1] // 2 + 1)``.

    Skeptic concern #3 carve-out: the ``is_delta`` branch inside
    ``_apply_kernel`` does NOT use the hoisted image FFTs (it short-
    circuits to a mask-modulated identity copy via the dedicated
    ``img_F`` only). Measured ``is_delta`` fraction of the legacy
    resampled kernels on the canonical bench is 8/132 ``(d, ri)`` pairs
    at downres 512 — but those are d=1 slices, NOT the d0 delta (at 2048
    the legacy d0 delta is smeared off-centre and is not ``is_delta``;
    see Q-b R-infocus-delta). The 1.7× hoist
    multiplier holds.
    """
    H_pad, W_pad = img_padded.shape[:2]
    geo = _fft_geometry(H_pad, W_pad, kernel_size, padded=True)
    full_h, full_w = geo["full_shape"]
    fshape = geo["fshape"]
    # mode="same" crop: scipy's _apply_conv_mode picks centered slices
    # `(full_h - H_pad) // 2 : (full_h - H_pad) // 2 + H_pad`. Since
    # full_h = H_pad + k - 1, that resolves to `(k-1)//2 : (k-1)//2+H_pad`.
    sy = (int(kernel_size) - 1) // 2
    sx = (int(kernel_size) - 1) // 2
    slice_y = slice(sy, sy + H_pad)
    slice_x = slice(sx, sx + W_pad)

    def _per_channel_F(arr: np.ndarray) -> np.ndarray:
        # arr: (H_pad, W_pad, 3) float32. Produce (3, fshape[0], fshape[1]//2+1)
        # complex64. scipy's rfftn promotes float32→complex64 by default.
        return np.stack(
            [sp_fft.rfft2(arr[:, :, c], s=fshape) for c in range(3)],
            axis=0,
        )

    return {
        "fshape": fshape,
        "full_shape": (full_h, full_w),
        "slice_y": slice_y,
        "slice_x": slice_x,
        "img_F": _per_channel_F(img_padded),
        "img_diffuse_F": _per_channel_F(img_diffuse_padded),
        "img_bright_F": _per_channel_F(img_bright_padded),
    }


def _fftconvolve_hoisted(
    img_F_c: np.ndarray, kernel: np.ndarray, fshape, slice_y, slice_x
) -> np.ndarray:
    """``fftconvolve(img_padded[:,:,c], kernel, mode="same")`` with the
    image-side FFT pre-computed.

    Returns a ``(H_pad, W_pad)`` float32 array equal to
    ``fftconvolve(img_padded[:,:,c], kernel, mode="same")`` modulo
    float-reduction-order (≤1 LSB drift on the canonical-bench JPG;
    the ≤1 uint8 contract from
    ``tests/unit/test_back_compat_canonical.py`` is the gate).
    """
    ker_F = sp_fft.rfft2(kernel, s=fshape)
    full = sp_fft.irfft2(img_F_c * ker_F, s=fshape)
    return full[slice_y, slice_x].astype(np.float32, copy=False)


def _apply_kernel(
    kernel_base, w, img_padded, img_diffuse_padded, img_bright_padded,
    wedge_idx, pad, H, W, psf_norm,
    output, output_bright, weight_sum, delta_weight_sum,
    *,
    fft_state: dict | None = None,
):
    """Apply kernel_base via the is_delta/angular-scatter body.

    P1.10.4 C3: now exclusively serves the basis scatter paths
    (`_scatter_one_slab_fixed_d_basis`).
    Renamed in C3a from `_apply_slab_kernel` to `_apply_kernel` (slab path
    fully removed; the "slab" qualifier is no longer meaningful).

    R-perf-1 I3: when ``fft_state`` is supplied, the per-channel
    convolutions route through ``_fftconvolve_hoisted`` (pre-computed
    image rfft2s) instead of ``scipy.signal.fftconvolve``. ``fft_state
    is None`` falls back to the original ``fftconvolve`` path bit-for-
    bit (preserves any direct caller that does not have a hoisted
    bundle handy — currently only synthetic adversarial harnesses).
    """
    is_delta = kernel_base.max() > 0.5 * kernel_base.sum()
    if fft_state is None:
        # Legacy path retained for direct unit-test entry points and any
        # caller that does not pre-compute the FFT bundle.
        if is_delta:
            if w.max() < 0.01:
                return
            for c in range(3):
                conv = fftconvolve(img_padded[:, :, c], kernel_base, mode="same").astype(np.float32)
                output[:, :, c] += w * conv[pad:pad + H, pad:pad + W]
            weight_sum += w
            delta_weight_sum += w
        else:
            for a in range(N_ANG):
                angular_mask = (wedge_idx == a).astype(np.float32)
                w_ang = (w * angular_mask).astype(np.float32)
                if w_ang.max() < 0.01:
                    continue
                theta_a = (a + 0.5) * (2 * np.pi / N_ANG) - np.pi
                rot_deg = -(theta_a * 180.0 / np.pi + 90.0)
                rotated = ndimage_rotate(kernel_base, rot_deg, reshape=False, order=1)
                rotated = np.maximum(rotated, 0.0)
                if psf_norm == "l1":
                    rotated /= rotated.sum() + 1e-12
                else:
                    rotated /= max(float(rotated.max()), 1e-10)
                rotated_bright = rotated / max(float(rotated.max()), 1e-10)
                for c in range(3):
                    conv_d = fftconvolve(
                        img_diffuse_padded[:, :, c], rotated, mode="same"
                    ).astype(np.float32)
                    output[:, :, c] += w_ang * conv_d[pad:pad + H, pad:pad + W]
                    conv_b = fftconvolve(
                        img_bright_padded[:, :, c], rotated_bright, mode="same"
                    ).astype(np.float32)
                    output_bright[:, :, c] += w_ang * conv_b[pad:pad + H, pad:pad + W]
                weight_sum += w_ang
        return

    # R-perf-1 I3 hoisted path.
    fshape = fft_state["fshape"]
    sy = fft_state["slice_y"]
    sx = fft_state["slice_x"]
    img_F = fft_state["img_F"]
    img_diffuse_F = fft_state["img_diffuse_F"]
    img_bright_F = fft_state["img_bright_F"]
    if is_delta:
        if w.max() < 0.01:
            return
        # is_delta carve-out (Skeptic #3): hoist applies — we still reuse
        # the pre-computed `img_F` per channel, so even the 6 % delta
        # branch benefits from one fewer FFT per channel.
        for c in range(3):
            conv = _fftconvolve_hoisted(img_F[c], kernel_base, fshape, sy, sx)
            output[:, :, c] += w * conv[pad:pad + H, pad:pad + W]
        weight_sum += w
        delta_weight_sum += w
    else:
        for a in range(N_ANG):
            angular_mask = (wedge_idx == a).astype(np.float32)
            w_ang = (w * angular_mask).astype(np.float32)
            if w_ang.max() < 0.01:
                continue
            theta_a = (a + 0.5) * (2 * np.pi / N_ANG) - np.pi
            rot_deg = -(theta_a * 180.0 / np.pi + 90.0)
            rotated = ndimage_rotate(kernel_base, rot_deg, reshape=False, order=1)
            rotated = np.maximum(rotated, 0.0)
            if psf_norm == "l1":
                rotated /= rotated.sum() + 1e-12
            else:
                rotated /= max(float(rotated.max()), 1e-10)
            rotated_bright = rotated / max(float(rotated.max()), 1e-10)
            # Pre-FFT the (potentially-different) rotated_bright kernel
            # only when the post-rotate normalization actually diverges
            # — which it does iff `psf_norm == "l1"` AND the L1 sum and
            # the peak-renorm denominator disagree. For the canonical
            # bench `psf_norm == "l1"`, so they ALWAYS diverge; we keep
            # two separate kernel rfft2s. For the `psf_norm == "peak"`
            # branch they are bit-equal — short-circuit one rfft2.
            ker_diffuse_F = sp_fft.rfft2(rotated, s=fshape)
            if psf_norm == "l1":
                ker_bright_F = sp_fft.rfft2(rotated_bright, s=fshape)
            else:
                ker_bright_F = ker_diffuse_F  # rotated == rotated_bright
            for c in range(3):
                conv_d_full = sp_fft.irfft2(
                    img_diffuse_F[c] * ker_diffuse_F, s=fshape
                )
                output[:, :, c] += w_ang * conv_d_full[sy, sx][
                    pad:pad + H, pad:pad + W
                ].astype(np.float32, copy=False)
                conv_b_full = sp_fft.irfft2(
                    img_bright_F[c] * ker_bright_F, s=fshape
                )
                output_bright[:, :, c] += w_ang * conv_b_full[sy, sx][
                    pad:pad + H, pad:pad + W
                ].astype(np.float32, copy=False)
            weight_sum += w_ang


def _scatter_one_slab_fixed_d_basis(
    d_idx, img_diffuse_pad, img_bright_pad, img_full_pad,
    kernels, defocus_ratios,
    radial_frac_map, r_centers, wedge_idx,
    kH, kW, pad, pix_scale, scaled_ksize, psf_norm,
    H, W,
    output, output_bright, weight_sum, delta_weight_sum,
    cancel: threading.Event | None = None,
    *,
    fft_state: dict | None = None,
):
    """P1.10.1 per-pixel radial-basis scatter — layered (fixed-D) path.

    Iterates raw-tensor radial-basis indices `ri ∈ [0, Nr)` with per-pixel
    weights `basis_w[ri, y, x]`. P1.10.4 C3: slab counterpart deleted; this
    is the sole layered fixed-D scatter path.

    P1.audit Item 3: ``cancel`` is an optional cooperative cancellation
    event polled at each radial-bin iteration. When set, raises
    :class:`summicron.render.api.CancelledError`.
    """
    from summicron.render.api import CancelledError

    Nr = int(len(r_centers))
    basis_w = _radial_basis_weights(radial_frac_map, r_centers)  # (Nr, H, W)
    for ri in range(Nr):
        if cancel is not None and cancel.is_set():
            raise CancelledError(
                f"render cancelled inside slab {d_idx}, "
                f"radial bin {ri}/{Nr}"
            )
        w = basis_w[ri].astype(np.float32)
        if w.max() < 0.01:
            continue
        kernel_base = _build_basis_kernel(
            ri, d_idx, True, defocus_ratios, kernels,
            kH, kW, psf_norm, pix_scale, scaled_ksize,
        )
        _apply_kernel(
            kernel_base, w, img_full_pad, img_diffuse_pad, img_bright_pad,
            wedge_idx, pad, H, W, psf_norm,
            output, output_bright, weight_sum, delta_weight_sum,
            fft_state=fft_state,
        )


def _layered_gather_diffuse(
    *,
    depth_m, slab_w, img_lin,
    img_padded, img_diffuse_padded, img_bright_padded,
    kernels, defocus_ratios, radial_frac_map, r_centers, wedge_idx,
    kH, kW, pad, pix_scale, scaled_ksize, psf_norm, H, W,
    fft_state, slab_batch_size, on_progress, cancel,
    emit_log=True,
):
    """The P1.12 depth-layered-gather diffuse assembly — the LIVE OFF path.

    R-occlusion-layering round-2 fix: extracted VERBATIM from the OFF
    ``else:`` (layered) block inside :func:`render` so that BOTH the OFF
    path AND ``_layered_composite`` regime-a call the *same code* (not a copy).
    The probe-round-1 post-mortem caught the round-1 bug where
    ``_layered_composite`` reimplemented only the bare ``output += out_d*mix``
    PoU sum and silently omitted the slab-batching driver, the streaming
    holes-fill (``best_d_color``/``argmin``), and the per-batch GC — so the ON
    path re-rendered the entire diffuse image differently from OFF even where
    the selector ``s≈0``. Reusing this helper makes regime-a bit-faithful to
    OFF by construction.

    This is a PURE EXTRACTION: identical float ops, identical iteration order,
    identical accumulators. The G1 pin md5 is preserved.

    Returns ``(output, output_bright, weight_sum, delta_weight_sum,
    acc_alpha)``. ``emit_log`` gates the two ``_emit`` diagnostics so the
    OFF path keeps its exact stdout (G1) while the ON regime-a call stays
    quiet (the ON branch prints its own occlusion log instead).
    """
    D_gate = slab_w.shape[0]

    output = np.zeros_like(img_lin)
    output_bright = np.zeros_like(img_lin)
    weight_sum = np.zeros((H, W), dtype=np.float32)
    delta_weight_sum = np.zeros((H, W), dtype=np.float32)

    # Porter-Duff over accumulator state.
    acc_alpha = np.zeros((H, W), dtype=np.float32)

    # R-perf-1 I1+I2: slab-batching driver + streaming holes-fill.
    if slab_batch_size is None:
        batch_size = _SLAB_BATCH_SIZE
    else:
        batch_size = int(slab_batch_size)
    if batch_size < 1:
        raise ValueError(
            f"slab_batch_size must be >= 1, got {batch_size!r}"
        )

    argmax_d_full = np.argmax(slab_w, axis=0).astype(np.int32)
    HUGE = np.int32(D_gate * 10)

    best_d_dist = np.full((H, W), HUGE, dtype=np.int32)
    best_d_color_d = np.zeros((H, W, 3), dtype=np.float32)
    best_d_color_b = np.zeros((H, W, 3), dtype=np.float32)

    effective_cap = max(_MAX_ACTIVE_SLABS, D_gate // 2)
    n_batches = (D_gate + batch_size - 1) // batch_size
    if batch_size * n_batches > max(effective_cap * 4, D_gate * 4):
        raise ValueError(
            f"slab_batch_size * n_batches = {batch_size * n_batches} "
            f"exceeds hard upper bound on (D_gate={D_gate}, "
            f"effective_cap={effective_cap}); pick smaller "
            f"--slab-batch-size."
        )
    active_count = 0
    _warned_cap = False
    from summicron.render.api import CancelledError

    for batch_start in range(0, D_gate, batch_size):
        batch_end = min(batch_start + batch_size, D_gate)
        batch_alpha: list[np.ndarray | None] = [None] * (batch_end - batch_start)
        batch_out_d: list[np.ndarray | None] = [None] * (batch_end - batch_start)
        batch_out_b: list[np.ndarray | None] = [None] * (batch_end - batch_start)

        for d in range(batch_start, batch_end):
            if cancel is not None and cancel.is_set():
                raise CancelledError(
                    f"render cancelled at slab {d}/{D_gate}"
                )
            if on_progress is not None:
                on_progress(f"  Slab {d + 1}/{D_gate}")
            sw_d = slab_w[d]  # (H, W)
            if float(sw_d.max()) < _EMPTY_SKIP_THRESH:
                continue  # empty-skip
            active_count += 1
            if active_count > effective_cap and not _warned_cap:
                warnings.warn(
                    f"layered_gather: exceeded _MAX_ACTIVE_SLABS={effective_cap} "
                    f"(D={D_gate}, active_count={active_count}); continuing — "
                    f"possible pathological depth spread.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                _warned_cap = True

            out_d = np.zeros((H, W, 3), dtype=np.float32)
            out_b_d = np.zeros((H, W, 3), dtype=np.float32)
            wsum_d = np.zeros((H, W), dtype=np.float32)
            delta_wsum_d = np.zeros((H, W), dtype=np.float32)

            with sp_fft.set_workers(-1):
                _scatter_one_slab_fixed_d_basis(
                    d_idx=d,
                    img_diffuse_pad=img_diffuse_padded,
                    img_bright_pad=img_bright_padded,
                    img_full_pad=img_padded,
                    kernels=kernels, defocus_ratios=defocus_ratios,
                    radial_frac_map=radial_frac_map,
                    r_centers=r_centers, wedge_idx=wedge_idx,
                    kH=kH, kW=kW, pad=pad, pix_scale=pix_scale,
                    scaled_ksize=scaled_ksize, psf_norm=psf_norm,
                    H=H, W=W,
                    output=out_d, output_bright=out_b_d,
                    weight_sum=wsum_d, delta_weight_sum=delta_wsum_d,
                    cancel=cancel,
                    fft_state=fft_state,
                )

            # Per-slab normalize by spatial-slab weight_sum (radial partition).
            if psf_norm == "l1":
                wsum_d_safe = np.maximum(wsum_d, 1e-8)
                for c in range(3):
                    out_d[:, :, c] /= wsum_d_safe
                    out_b_d[:, :, c] /= wsum_d_safe

            # Gather weight: slab_w[d] at the output.
            alpha_d = sw_d.astype(np.float32)
            mix = alpha_d[..., None]
            output += out_d * mix
            output_bright += out_b_d * mix
            weight_sum += wsum_d
            delta_weight_sum += delta_wsum_d
            acc_alpha += alpha_d  # partition-of-unity => sums to 1 on covered pixels

            local = d - batch_start
            batch_alpha[local] = alpha_d
            batch_out_d[local] = out_d
            batch_out_b[local] = out_b_d

        # I2 streaming-holes-fill: update running argmin from this batch.
        for local in range(batch_end - batch_start):
            d_global = batch_start + local
            alpha_d = batch_alpha[local]
            if alpha_d is None:
                continue  # empty-skipped slab; never a valid donor.
            sw_d = slab_w[d_global]
            validity_d = sw_d > _EMPTY_SKIP_THRESH  # (H, W)
            dist_d_for_this = np.abs(
                np.int32(d_global) - argmax_d_full
            ).astype(np.int32)
            dist_masked_d = np.where(validity_d, dist_d_for_this, HUGE)
            better = dist_masked_d < best_d_dist
            if better.any():
                best_d_dist = np.where(better, dist_masked_d, best_d_dist)
                better_3 = better[..., None]
                best_d_color_d = np.where(
                    better_3, batch_out_d[local], best_d_color_d
                )
                best_d_color_b = np.where(
                    better_3, batch_out_b[local], best_d_color_b
                )

        del batch_alpha, batch_out_d, batch_out_b
        gc.collect()

    if emit_log:
        _emit(
            on_progress,
            f"  layered_gather: active_slabs={active_count}/{D_gate}, "
            f"acc_alpha mean={float(acc_alpha.mean()):.3f}, "
            f"holes={int((acc_alpha < 0.02).sum())} pixels"
        )

    # Holes-fill (I6).
    holes = acc_alpha < 0.02
    if holes.any():
        has_valid = best_d_dist < HUGE  # (H, W)
        fallback = img_lin
        valid_and_holes = (holes & has_valid)[..., None]
        output = np.where(valid_and_holes, best_d_color_d, output)
        output_bright = np.where(valid_and_holes, best_d_color_b, output_bright)
        noone_and_holes = (holes & ~has_valid)[..., None]
        output = np.where(noone_and_holes, fallback, output)
        output_bright = np.where(noone_and_holes, 0.0, output_bright)
        acc_alpha = np.where(holes & has_valid, np.float32(1.0), acc_alpha)
        acc_alpha = np.where(holes & ~has_valid, np.float32(1.0), acc_alpha)

    return output, output_bright, weight_sum, delta_weight_sum, acc_alpha


def render(
    image_srgb: np.ndarray,
    tensor: dict,
    *,
    depth_m: np.ndarray,
    psf_norm: str = "l1",
    hdr_source: str = "none",
    bright_thresh: float = 1e9,
    bright_pass_region_size: int = 31,
    bright_pass_floor: float = 0.10,
    bright_pass_norm_pct: float = 99.0,
    occlusion_compositing: str = "off",
    slab_batch_size: int | None = None,
    on_progress: _OnProgress = None,
    cancel: threading.Event | None = None,
    stats: dict | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Bin-and-convolve render pipeline.

    ``stats`` (CLI-1b): opt-in diagnostics sink. When a dict is passed it is
    filled with ``bright_pixels_pct`` (float), ``alpha_mean`` (float) and
    ``acc_alpha`` (the full per-pixel ``(H, W)`` accumulated-alpha map, on
    both the occlusion-ON and OFF paths). Q-a adds (references, no copies):
    ``alpha`` ((H, W) identity-snap alpha), ``dmap`` ((H, W) defocus map) and
    ``img_lin`` ((H, W, 3) linear reference input); ``defocus_ratios`` is
    surfaced by ``_run_render_pipeline`` via its private ``_eval_sink``, not
    by this ``stats`` dict. The return arity stays 2.

    Parameters
    ----------
    image_srgb : (H, W, 3) float32 in [0, 1] sRGB gamma
    tensor : dict from load_tensor()
    depth_m : (H, W) float32 in meters, distance-indexed depth map (required).
        Requires tensor with object_distances_m metadata.
    psf_norm : {"l1", "peak"}, default "l1"
        Render-time PSF normalization mode. "l1" (default) keeps kernels L1-
        normalized and averages via a post-convolution weight-sum divide —
        bit-identical to the pre-P1.7 render path and energy-conserving for
        HDR inputs. "peak" peak-normalizes kernels (peak = 1) and drops the
        post-conv renorm, turning the PSF into a stamp: bright pixels bloom
        by ``sum(kernel)``; overflow clips to white at finalize. See
        doc/plan/phase1/section7/locked-spec-stamp-mode.md.
    hdr_source : {"none", "macos"}, default "none"
        Input is LDR sRGB when "none" (existing path, bit-identical to P1.7).
        Otherwise the input is already extended-linear HDR float32: skip the
        `max <= 1.001` guard (replaced with `isfinite` check) and the
        `srgb_to_linear` round-trip, and return linear float32 WITHOUT
        clipping / sRGB gamma / uint8 cast. Caller (main()) tonemaps and
        finalizes post-render. See doc/plan/phase1/section8/locked-spec-
        hdr-access.md.

    Returns
    -------
    output_srgb_u8 : (H, W, 3) uint8 sRGB — LDR path only
    output_linear : (H, W, 3) float32 linear

    When ``hdr_source != "none"`` the first tuple element is the SAME linear
    float32 array as ``output_linear`` (no uint8 cast inside render()); the
    caller is responsible for tonemap + gamma + cast.

    P1.9.1: a per-pixel α(defocus) blend runs after the convolve accumulator
    and peak-mode auto-exposure divide. With ``α = cos²(π/2 · min(dmap/D_KNEE_LAYERED,
    1))`` (module constant ``D_KNEE_LAYERED = 0.10``), in-focus pixels (``dmap=0``)
    return the linearized input bit-exactly while out-of-focus pixels
    (``dmap ≥ D_KNEE_LAYERED``) see the raw convolved output; intermediate defocus
    cross-fades smoothly with C¹ continuity at both endpoints. The blend
    happens once in linear space so LDR and HDR paths share the same math.

    Sentinel idiom note (``bright_thresh=1e9``)
    -------------------------------------------
    ``bright_thresh`` defaults to ``1e9`` — a deliberate large-finite
    sentinel that disables the bright-pass split (no pixels exceed it on
    LDR sRGB). It is NOT a magic-number leak; the value is the documented
    P1.10.4 back-compat killswitch ("``bright_thresh=1e9`` keeps pre-P1.11
    callers and synthetic fixtures bit-identical"). 13 testsites currently
    pass this sentinel explicitly; refactoring to a typed sentinel
    (``BRIGHT_THRESH_DISABLED = ...`` constant or ``bright_pass: bool``
    flag) is deferred.

    """
    if stats is not None and not isinstance(stats, dict):
        raise TypeError(
            f"stats must be a dict or None; got {type(stats).__name__}"
        )
    # P1.7: validate psf_norm mode up-front.
    if psf_norm not in ("l1", "peak"):
        raise ValueError(
            f"psf_norm must be 'l1' or 'peak'; got {psf_norm!r}"
        )

    # CLI-1a A3.1b: kernel-L1 guard on the raw tensor (replaces the deleted
    # split-energy gate's only unique catch). Zero slices are exempt.
    if psf_norm == "l1":
        ks = np.asarray(tensor["kernels"]).sum(axis=(-2, -1))
        bad = (~np.isfinite(ks)) | ((ks != 0) & (np.abs(ks - 1.0) > 0.05))
        if bad.any():
            _bv = ks[bad]
            _fin = _bv[np.isfinite(_bv)]
            _rep = float(_fin[np.argmax(np.abs(_fin - 1.0))]) if _fin.size else float(_bv[0])
            _nf = "" if _fin.size == _bv.size else " (non-finite NaN/Inf kernel sum present)"
            raise RuntimeError(
                f"Gate FAIL: kernel energy ratio {_rep:.4f}{_nf} "
                f"(expected L1-normalized kernels)"
            )

    # P1.11 R1: negative bright_thresh is coerced to 0 (bright path then covers
    # the full image). Rejecting non-finite outright — NaN/Inf makes no sense.
    if not np.isfinite(bright_thresh):
        raise ValueError(
            f"bright_thresh must be finite, got {bright_thresh!r}"
        )
    bright_thresh = float(max(bright_thresh, 0.0))

    # P1.8 bug 3: explicit hdr_source validation (Python None != "none" string
    # used to silently drop into the HDR branch).
    if hdr_source not in ("none", "macos"):
        raise ValueError(
            f"hdr_source must be 'none' or 'macos', got {hdr_source!r}"
        )

    # R-occlusion-layering: validate occlusion_compositing mode. "off"
    # (default) is the bit-identical kill-switch; "on" dispatches to the
    # parallel _layered_composite() helper (premultiplied-alpha layered OVER).
    if occlusion_compositing not in ("off", "on"):
        raise ValueError(
            f"occlusion_compositing must be 'off' or 'on', "
            f"got {occlusion_compositing!r}"
        )

    # R-occ-on-fixes F4: peak-normalised kernels give matte alpha > 1, which
    # the layered OVER compositor cannot handle -> fall back to the OFF path.
    if occlusion_compositing == "on" and psf_norm == "peak":
        _msg = (
            "--psf-norm peak is incompatible with --occlusion-compositing on "
            "(peak-normalised matte alpha > 1); falling back to off."
        )
        warnings.warn(_msg, RuntimeWarning, stacklevel=2)
        if on_progress is not None:  # CLI (None) gets the warning only
            on_progress("  " + _msg)
        occlusion_compositing = "off"

    # P1.10: validate region-aware bright-pass kwargs.
    if (
        not isinstance(bright_pass_region_size, (int, np.integer))
        or bool(isinstance(bright_pass_region_size, bool))
        or bright_pass_region_size < 0
        or bright_pass_region_size > 255
    ):
        raise ValueError(
            f"bright_pass_region_size must be int in [0, 255], "
            f"got {bright_pass_region_size!r}"
        )
    if not np.isfinite(bright_pass_floor) or not (
        0.0 <= float(bright_pass_floor) <= 1.0
    ):
        raise ValueError(
            f"bright_pass_floor must be in [0.0, 1.0], got {bright_pass_floor!r}"
        )
    if not np.isfinite(bright_pass_norm_pct) or not (
        50.0 <= float(bright_pass_norm_pct) <= 100.0
    ):
        raise ValueError(
            f"bright_pass_norm_pct must be in [50.0, 100.0], "
            f"got {bright_pass_norm_pct!r}"
        )

    # Bug 3: validate / coerce input dtype
    if image_srgb.dtype == np.uint8:
        image_srgb = image_srgb.astype(np.float32) / 255.0
    elif np.issubdtype(image_srgb.dtype, np.floating):
        if hdr_source == "none":
            if image_srgb.max() > 1.0 + 1e-3:
                raise ValueError(
                    f"Float input max {image_srgb.max():.4f} > 1.0; "
                    "expected [0, 1] sRGB"
                )
        else:
            # HDR branch: extended-linear float32 may exceed 1.0 by design.
            # Only reject NaN/Inf which cannot come from a valid HDR decode.
            if not np.isfinite(image_srgb).all():
                raise ValueError(
                    "HDR input contains non-finite values "
                    "(NaN/Inf); expected finite extended-linear float32"
                )
    else:
        raise ValueError(f"Unsupported dtype {image_srgb.dtype}; expected float32 or uint8")

    H, W = image_srgb.shape[:2]

    # Step 2: linearize (skip if input is already extended-linear HDR)
    if hdr_source == "none":
        img_lin = srgb_to_linear(image_srgb)
    else:
        img_lin = image_srgb.astype(np.float32, copy=False)

    # depth_m -> dmap via entry-point interp
    object_distances_m = tensor.get("object_distances_m")
    if object_distances_m is None:
        raise ValueError(
            "depth_m requires tensor with object_distances_m metadata"
        )
    defocus_ratios_axis = tensor["defocus_ratios"]
    depth_m = np.asarray(depth_m)
    if depth_m.shape != (H, W):
        raise ValueError(
            f"depth_m shape {depth_m.shape} != image (H,W) = ({H}, {W})"
        )
    dmap = np.interp(
        depth_m, object_distances_m, defocus_ratios_axis
    ).astype(np.float32)

    kernels = tensor["kernels"]         # (R, D, kH, kW)
    radial_edges = tensor["radial_edges"]
    defocus_ratios = tensor["defocus_ratios"]

    kH, kW = kernels.shape[2], kernels.shape[3]

    # P1.audit Item 6 / R-native-mem item 7: pixel fast-path gate here (same
    # call site as before: error order unchanged, nothing large allocated yet).
    # The byte-budget guard runs below, once ``scaled_ksize`` is known and
    # before ``np.pad`` and every large allocation.
    # Suggestion also fits the byte budget (guard defaults for kernel size).
    _pixel_guard(
        H, W, int(kernels.shape[1]),
        occlusion=(occlusion_compositing == "on"),
        n_radial=int(kernels.shape[0]),
        slab_batch_size=slab_batch_size,
    )

    # ------------------------------------------------------------------
    # P0.2: radial/depth decoupling. Build a per-pixel spatial
    # radial_frac_map (fraction of half-diagonal). Partition pixels into
    # N_r = len(r_centers) - 1 = 11 radial slabs, and run one kernel
    # blend per (defocus_bin, radial_slab) pair. Replaces the broken
    # `radial_frac = np.sqrt(d_center)` selection which produced
    # position-independent kernels for every in-focus pixel. See
    # doc/plan/phase0/section10/locked-spec-radial-decouple.md.
    # ------------------------------------------------------------------
    cy_opt = (H - 1) / 2.0
    cx_opt = (W - 1) / 2.0
    max_radial = float(np.hypot(cy_opt, cx_opt))
    if max_radial < 1e-6:
        # Degenerate (e.g., 1x1 image): no spatial radial variation.
        radial_frac_map = np.zeros((H, W), dtype=np.float32)
    else:
        ys_r = np.arange(H, dtype=np.float32) - np.float32(cy_opt)
        xs_r = np.arange(W, dtype=np.float32) - np.float32(cx_opt)
        radial_frac_map = (
            np.hypot(ys_r[:, None], xs_r[None, :]) / np.float32(max_radial)
        ).astype(np.float32)
    # P1.audit Item 5: assert→raise for input/shape gates → ValueError.
    if radial_frac_map.min() < 0.0:
        raise ValueError("radial_frac_map has negative values")
    if radial_frac_map.max() > 1.001:
        raise ValueError(
            f"radial_frac_map max {radial_frac_map.max()} > 1.001"
        )

    # ------------------------------------------------------------------
    # P1.0: Angular wedge assignment for per-pixel kernel rotation.
    # Precompute an integer wedge index (0..N_ANG-1) for every pixel.
    # Uses the same ys_r, xs_r displacement arrays from radial_frac_map.
    # ------------------------------------------------------------------
    if max_radial < 1e-6:
        wedge_idx = np.zeros((H, W), dtype=np.int32)
    else:
        theta_map = np.arctan2(ys_r[:, None], xs_r[None, :])  # [-pi, pi]
        wedge_idx = ((theta_map + np.pi) / (2 * np.pi) * N_ANG).astype(np.int32) % N_ANG
        del theta_map  # R-native-mem item 5: never read again (~1 plane)

    r_centers = 0.5 * (radial_edges[:-1] + radial_edges[1:])
    # Radial kernel blend uses `_radial_basis_weights(radial_frac_map, r_centers)`
    # directly (per-pixel partition of unity); no spatial slab POU is built.

    # Pixel-scale: shrink kernels when rendering images smaller than corpus ref.
    # Only activate when corpus_ref_long_edge is explicitly present in the
    # tensor dict (set by load_tensor or by the caller).
    corpus_ref = tensor.get("corpus_ref_long_edge")
    if corpus_ref is not None:
        if not (np.isfinite(corpus_ref) and corpus_ref > 0):
            raise ValueError(
                f"corpus_ref_long_edge must be finite and > 0; got {corpus_ref!r}"
            )
        pix_scale = max(H, W) / corpus_ref
    else:
        pix_scale = 1.0  # no scaling when metadata absent (hand-crafted tensors)
    if pix_scale < 0.95:
        scaled_ksize = max(3, int(round(kH * pix_scale)) | 1)  # force odd
    else:
        scaled_ksize = kH
    pad = scaled_ksize // 2

    # R-native-mem item 7: byte-budget guard (D is known; the "Memory budget"
    # line is routed through ``on_progress`` and silenced on the CLI path so
    # canonical-bench stdout, and therefore the G1 md5 pins, are unchanged).
    _memory_guard(
        H, W, int(kernels.shape[1]),
        on_progress=on_progress,
        occlusion=(occlusion_compositing == "on"),
        kernel_size=scaled_ksize,
        n_radial=int(kernels.shape[0]),
        slab_batch_size=slab_batch_size,
    )

    # Mirror-pad image
    img_padded = np.pad(img_lin, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")

    # P1.11 R1 fix-round 2: gate on luminance, not per-channel. Per-channel
    # thresholding splits channels independently → chromatic halos on warm
    # highlights (orange bloom on IMG_6915 umbrella canopy, hue shift on
    # skin highlights). Luminance gate + full-RGB split preserves chromaticity.
    # Pattern matches Frostbite SIGGRAPH 2014 / Nuke ZDefocus bloom.
    #
    # P1.11 R1-PATCH: soft-knee smoothstep over [T-K, T+K] replaces the hard
    # boolean. Hard boolean produced patchy umbrella-canopy output where
    # lum ≈ T ± natural variation caused adjacent pixels to flip between
    # peak-splat and L1 diffuse kernels (kernel-sum ratio ≈ 100×). Soft-knee
    # gives a C¹-continuous blend in the band and is bit-identical outside it.
    # K (smoothstep half-width) is now the module constant `_BRIGHT_KNEE = 0.20`
    # post-P1.10.2 Cut 3. See doc/plan/phase1/section11/locked-spec-R1-PATCH-softknee.md
    # and the SUPERSEDED memory `project_bright_knee_midtone_crush.md`.
    lum_padded = (
        0.2126 * img_padded[..., 0]
        + 0.7152 * img_padded[..., 1]
        + 0.0722 * img_padded[..., 2]
    )  # (H_pad, W_pad)
    # P1.10: region-aware bright-pass; bright_pass_region_size=0 is the
    # kill-switch and returns the bare smoothstep (bit-identical to pre-P1.10).
    bright_weight = _compute_region_aware_bright_weight(
        lum_padded, bright_thresh,
        bright_pass_region_size, bright_pass_floor, bright_pass_norm_pct,
    )
    bright_mask = bright_weight.astype(np.float32)[..., None]  # (H_pad, W_pad, 1)
    img_bright_padded = img_padded * bright_mask
    img_diffuse_padded = img_padded - img_bright_padded
    # R-native-mem item 5: dead after the bright/diffuse split (~3 padded planes).
    del lum_padded, bright_weight, bright_mask

    # R-perf-1 I3: hoist per-channel rfft2 of all 3 padded image buffers
    # to once-per-render. The scatter loops then route every
    # `fftconvolve(img, kernel, mode="same")` call through the hoisted
    # spectra (kernel-only rfft2 + multiply + irfft2). At 2048² on the
    # canonical bench this is the headline perf win. Memory cost: 9 ×
    # complex64 spectra at next_fast_len(H_pad+k-1, W_pad+k-1) shape;
    # ≈ 36 MB at 2048² (immaterial vs the 1.3 GB peak RSS budget).
    fft_state = _build_image_fft_state(
        img_padded, img_diffuse_padded, img_bright_padded, scaled_ksize
    )

    # Layered gather is the SOLE rendering path (CLI-1a). R-native-mem item 5:
    # the four accumulators (output, output_bright, weight_sum,
    # delta_weight_sum) are allocated only by the helper that returns them;
    # the old pre-dispatch zeros were dead (both branches rebind all four from
    # the helper's tuple). weight_sum / delta_weight_sum track the is_delta
    # branch weight so the diffuse energy gate has the right denominator.
    if occlusion_compositing == "on":
        # R-occlusion-layering ON branch — single dispatch to the parallel
        # _layered_composite() helper. The OFF path (the slab loop below) stays
        # byte-untouched, so G1 (canonical-bench md5) holds by code-path
        # isolation. round-2 fix: _layered_composite REUSES the extracted
        # _layered_gather_diffuse() for regime-a (literally the OFF code), so
        # where the selector s≈0 the ON diffuse == OFF diffuse to float
        # tolerance; only the regime-b premultiplied-OVER overlay is new. It
        # returns the s-blended diffuse + the OFF-verbatim bright buffer; the
        # caller recombines them at the unchanged bright-add below.
        from summicron.render._layered_composite import _layered_composite

        object_distances_m = tensor.get("object_distances_m")
        slab_w = compute_slab_weights(depth_m, object_distances_m)
        (
            output, output_bright, weight_sum, delta_weight_sum, acc_alpha
        ) = _layered_composite(
            depth_m=depth_m,
            slab_w=slab_w,
            object_distances_m=object_distances_m,
            defocus_map=dmap,
            img_lin=img_lin,
            img_padded=img_padded,
            img_diffuse_padded=img_diffuse_padded,
            img_bright_padded=img_bright_padded,
            kernels=kernels,
            defocus_ratios=defocus_ratios,
            radial_frac_map=radial_frac_map,
            r_centers=r_centers,
            wedge_idx=wedge_idx,
            kH=kH, kW=kW, pad=pad, pix_scale=pix_scale,
            scaled_ksize=scaled_ksize, psf_norm=psf_norm,
            H=H, W=W,
            fft_state=fft_state,
            slab_batch_size=slab_batch_size,
            cancel=cancel,
            on_progress=on_progress,
        )
        if stats is not None:
            stats["acc_alpha"] = acc_alpha
        _emit(
            on_progress,
            f"  occlusion-compositing: ON (premultiplied-alpha layered OVER), "
            f"D={slab_w.shape[0]}",
        )
    else:
        # Depth-layered gather (the LIVE OFF diffuse path). R-occlusion-layering
        # round-2: the entire P1.12 layered_gather assembly (slab-batching +
        # streaming holes-fill + per-batch GC) now lives in the extracted
        # _layered_gather_diffuse() helper so _layered_composite regime-a can
        # call the SAME code (probe-round-1 fix; see the helper docstring).
        # PURE EXTRACTION (pre-Q-b pin md5 38c75dd2... was the old pin; current is G1_MD5 in _bench.py).
        object_distances_m = tensor.get("object_distances_m")
        slab_w = compute_slab_weights(depth_m, object_distances_m)
        (
            output, output_bright, weight_sum, delta_weight_sum, acc_alpha
        ) = _layered_gather_diffuse(
            depth_m=depth_m, slab_w=slab_w, img_lin=img_lin,
            img_padded=img_padded, img_diffuse_padded=img_diffuse_padded,
            img_bright_padded=img_bright_padded,
            kernels=kernels, defocus_ratios=defocus_ratios,
            radial_frac_map=radial_frac_map, r_centers=r_centers,
            wedge_idx=wedge_idx,
            kH=kH, kW=kW, pad=pad, pix_scale=pix_scale,
            scaled_ksize=scaled_ksize, psf_norm=psf_norm, H=H, W=W,
            fft_state=fft_state, slab_batch_size=slab_batch_size,
            on_progress=on_progress, cancel=cancel,
            emit_log=True,
        )
        if stats is not None:
            stats["acc_alpha"] = acc_alpha

    # P1.11 R1 (I7): bright-pass logging before α-blend.
    _bpp = 100.0 * float((img_bright_padded > 0.0).any(axis=-1).mean())
    if stats is not None:
        stats["bright_pixels_pct"] = _bpp
    _inp = float(img_lin.sum())
    _ber = float(output_bright.sum()) / _inp if _inp > 1e-6 else 0.0
    _emit(
        on_progress,
        f"  bright-pass: thresh={bright_thresh:g}, "
        f"region_size={int(bright_pass_region_size)}, "
        f"floor={float(bright_pass_floor):.3f}, "
        f"norm_pct={float(bright_pass_norm_pct):.2f}, "
        f"bright_pixels_pct={_bpp:.3f}%, bright_energy_ratio={_ber:.4f}",
    )

    # Combine diffuse + bright before gate checks.
    # P1.audit Item 5: assert→raise for output-quality gates → RuntimeError.
    output = output + output_bright
    if np.any(np.isnan(output)):
        raise RuntimeError("Gate FAIL: NaN in output")
    if np.any(np.isinf(output)):
        raise RuntimeError("Gate FAIL: Inf in output")

    input_sum = float(img_lin.sum())
    if input_sum < 1e-6:
        # Trivially conserved: zero in, ~zero out
        pass
    else:
        # P1.12 R2: layered gather uses per-slab weighted sum composite. Use
        # the linear input (img_lin) as the denominator for energy
        # conservation. L1 kernels + partition-of-unity slab_w preserve
        # Σ energy under the composite.
        lin_sum = float(img_lin.sum())
        energy_ratio = float(output.sum()) / max(lin_sum, 1e-6)
        if psf_norm == "l1":
            if not (0.85 <= energy_ratio <= 100.0):
                raise RuntimeError(
                    f"Gate FAIL (layered): total energy ratio "
                    f"{energy_ratio:.4f} outside [0.85, 100.0] "
                    f"(linear/linear)"
                )
        else:
            if not (energy_ratio <= 500.0):
                raise RuntimeError(
                    f"Gate FAIL: stamp bloom runaway, energy ratio "
                    f"{energy_ratio:.4f} > 500.0"
                )

    # Center SSIM check (warning only)
    if min(H, W) < 7:
        # Skip SSIM: skimage requires images >= 7x7 for default win_size
        pass
    else:
        try:
            from skimage.metrics import structural_similarity
            ch, cw = min(256, H), min(256, W)
            y0, x0 = (H - ch) // 2, (W - cw) // 2
            center_in = img_lin[y0:y0 + ch, x0:x0 + cw]
            center_out = output[y0:y0 + ch, x0:x0 + cw]
            ssim = structural_similarity(center_in, center_out, channel_axis=2, data_range=1.0)
            if ssim < 0.80:
                warnings.warn(f"Center SSIM {ssim:.3f} < 0.80")
            else:
                _emit(on_progress, f"  Center SSIM: {ssim:.3f} (PASS)")
        except ImportError:
            warnings.warn("skimage not available; skipping center SSIM check")

    # P1.7 fix-round: auto-exposure normalize in peak (stamp) mode only.
    # Bright regions in stamp mode legitimately bloom past 1.0 on LDR input
    # (kernel-sum may be ~150 on wide blur). Naive clip crushes mid-tones to
    # white. Divide the whole linear image by its own max so the brightest
    # output pixel lands at 1.0 and mid-tones are preserved proportionally.
    # Gated on ``omax > 1.0`` so dim / in-focus scenes (max already < 1) are
    # untouched — preserves test_in_focus_peak_preservation semantics.
    # L1 mode is intentionally untouched (bit-identity guard).
    if psf_norm == "peak":
        omax = float(output.max())
        if omax > 1.0:
            output = output / omax

    # P1.12 R2 / TL REVISION 1: α-blend as post-layered IDENTITY SNAP.
    # Load-bearing for in-focus sharpness (G12 G_FACE_FALLOFF): at dmap≈0
    # the smoothstep POU spreads weight across ~4 non-identity bins; this
    # snap restores sharpness at the focal plane while preserving the
    # layered composite elsewhere. `output` here = diffuse + bright.
    alpha = (np.cos(0.5 * np.pi * np.minimum(dmap / D_KNEE_LAYERED, 1.0)) ** 2).astype(np.float32)
    output = alpha[:, :, None] * img_lin + (1.0 - alpha[:, :, None]) * output
    if np.any(np.isnan(output)):
        raise RuntimeError("Gate FAIL: NaN in layered output")
    if stats is not None:
        stats["alpha_mean"] = float(alpha.mean())
        # Q-a: references only (no copy, no arithmetic) for the paired
        # quality gates; popped by _run_render_pipeline.
        stats["alpha"] = alpha
        stats["dmap"] = dmap
        stats["img_lin"] = img_lin
    _emit(on_progress, f"  α-blend: D_KNEE_LAYERED={D_KNEE_LAYERED}, alpha_mean={float(alpha.mean()):.3f} (layered identity snap)")

    output_linear = output.copy()

    # Step 7: finalize. HDR path skips clip/gamma/uint8 — tonemap happens in
    # main() AFTER render() returns (load-bearing per locked spec §3; see
    # doc/plan/phase1/section8/locked-spec-hdr-access.md).
    if hdr_source != "none":
        return output_linear, output_linear

    output = np.clip(output, 0.0, 1.0)
    output_srgb = linear_to_srgb(output)
    output_u8 = np.clip(output_srgb * 255.0, 0, 255).astype(np.uint8)

    return output_u8, output_linear


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_results(output_u8: np.ndarray, input_stem: str, out_dir: Path,
                  depth_scale: float = 1.0,
                  downres: int | None = None) -> Path:
    """Save rendered image as JPEG (q95) and PNG; returns the JPEG path."""
    from PIL import Image

    out_dir.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(output_u8, mode="RGB")

    downres_tag = f"_r{downres}" if downres is not None else ""
    suffix = f"_ds{depth_scale}" if depth_scale != 1.0 else ""
    jpg_path = out_dir / f"{input_stem}{downres_tag}_rendered{suffix}.jpg"
    png_path = out_dir / f"{input_stem}{downres_tag}_rendered{suffix}.png"

    img.save(str(jpg_path), "JPEG", quality=95)
    img.save(str(png_path), "PNG")
    print(f"  Saved: {jpg_path}")
    print(f"  Saved: {png_path}")
    return jpg_path


# ---------------------------------------------------------------------------
# CLI — argparse builder + thin shim + lifted render pipeline (P1.audit Item 3)
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Construct the convolve.py CLI parser.

    Lifted out of ``main()`` in P1.audit Item 3 so the GUI / tests can
    introspect flags without exec-side effects. CLI-1a C3: every flag is
    installed by the stdlib-only ``_render_args.add_render_arguments``,
    shared with the ``isummicron render`` subparser.
    """
    from summicron.render._render_args import add_render_arguments

    parser = argparse.ArgumentParser(
        description="Bin-and-convolve renderer: apply 4D PSF tensor to an image"
    )
    add_render_arguments(parser)
    return parser


def _validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Argparse-time validation lifted from the original ``main()`` body.

    Calls ``parser.error()`` (exit 2) on any invalid flag so the CLI
    surface stays unchanged. The invariants themselves live in
    ``RenderOptions.__post_init__`` (and ``api._check_platform`` for the
    macOS preflight); this function only converts their ``ValueError`` into
    an argparse error. GUI/in-process callers construct ``RenderOptions``
    directly and get the same ``ValueError``.
    """
    from summicron.render.api import RenderOptions, _check_platform

    try:
        _check_platform(getattr(args, "hdr_source", "none"))
        RenderOptions.from_argparse(args)
    except ValueError as e:
        parser.error(str(e))


def _run_render_pipeline(
    opts,
    *,
    on_progress: _OnProgress = None,
    cancel: threading.Event | None = None,
    _eval_sink: dict | None = None,
):
    """Execute the v0 render pipeline against a :class:`RenderOptions`.

    P1.audit Item 3: lifted from the original ``main()`` body as a pure
    refactor. ``on_progress=None`` (CLI default) routes status emits
    through ``print()`` so stdout is byte-identical to the pre-refactor
    path; the current bench pin is G1_MD5 (_bench.py; ``38c75dd2…`` was the pre-Q-b pin, no longer reproducible). Returns a
    provisional :class:`~summicron.render.api.RenderResult` whose
    ``elapsed_s`` is 0.0 — the public :func:`render_image` wrapper in
    ``api.py`` overwrites that with the real wall time.

    Cooperative cancellation via ``cancel`` is forwarded into
    :func:`render` (slab-loop checkpoints).
    """
    # Local import keeps the dataclass module out of the hot CLI path.
    from summicron.render.api import RenderDiagnostics, RenderResult

    image_path = Path(opts.image_path)
    tensor_path = Path(opts.tensor_path) if opts.tensor_path is not None else None
    if tensor_path is None:
        # P1.audit Item 7: fall back to the wheel-shipped canonical tensor.
        from summicron.render._artifacts import default_tensor_path
        tensor_path = default_tensor_path()
    out_dir = Path(opts.out_dir) if opts.out_dir is not None else (
        Path.cwd() / "runs" / "render"
    )
    depth_scale = float(opts.depth_scale)
    warnings_collected: list[str] = []

    _emit(on_progress, f"Loading image: {image_path}")
    if opts.hdr_source == "macos":
        from summicron.io.hdr_load_macos import load_hdr_heic
        image_srgb = load_hdr_heic(image_path)
    else:
        image_srgb = load_image(image_path)

    # P1.13 K11 LDR-ref capture (see original main() comment).
    ldr_ref_linear = None
    if opts.shadow_match_mode == "on":
        if opts.hdr_source == "macos":
            ldr_srgb = load_image(image_path)
            ldr_ref_linear = srgb_to_linear(ldr_srgb)
        else:
            ldr_ref_linear = srgb_to_linear(image_srgb)

    if opts.downres is not None:
        orig_h, orig_w = image_srgb.shape[:2]
        image_srgb = downres_array(image_srgb, opts.downres)
        new_h, new_w = image_srgb.shape[:2]
        if (new_h, new_w) != (orig_h, orig_w):
            _emit(on_progress, f"Downres: {orig_h}x{orig_w} -> {new_h}x{new_w}")
        if ldr_ref_linear is not None and ldr_ref_linear.shape[:2] != (
            new_h,
            new_w,
        ):
            ldr_ref_linear = downres_array(ldr_ref_linear, opts.downres)

    _emit(on_progress, f"Loading tensor: {tensor_path}")
    tensor = load_tensor(tensor_path)

    focus_target_m = opts.focus_target_m
    if focus_target_m is None:
        focus_target_m = (
            tensor["focus_distance_m"]
            if tensor["focus_distance_m"] is not None
            else 0.7
        )
    elif (
        tensor["focus_distance_m"] is not None
        and focus_target_m != tensor["focus_distance_m"]
    ):
        msg = (
            f"Warning: --focus-target-m={focus_target_m} != tensor "
            f"focus_distance_m={tensor['focus_distance_m']}; "
            "subject may not be sharp"
        )
        _emit(on_progress, msg)
        warnings_collected.append(msg)

    from summicron.depth.estimate import (
        extract_lidar_depth,
        apply_tap_focus_shift,
        load_depth_model,
        estimate_depth,
    )
    H, W = image_srgb.shape[:2]
    if opts.depth_source == "lidar":
        _emit(on_progress, f"Extracting LiDAR depth (target={H}x{W})...")
        depth_m = extract_lidar_depth(image_path, target_hw=(H, W))
        if depth_m is None:
            raise ValueError(
                "--depth-source=lidar but image has no embedded depth aux."
            )
    else:
        _emit(on_progress, f"Running Depth Anything V2 (target={H}x{W})...")
        bundle = load_depth_model(device="mps")
        depth_01 = estimate_depth(image_srgb, bundle)
        FAR_M = 1.0
        depth_m = (depth_01 * FAR_M).astype(np.float32)
    focus_xy = tuple(opts.focus_xy) if opts.focus_xy is not None else None
    depth_m_scaled = depth_m * depth_scale
    shifted_m, anchor_m = apply_tap_focus_shift(
        depth_m_scaled, focus_xy, focus_target_m
    )
    _emit(
        on_progress,
        f"  raw_anchor_m={anchor_m/depth_scale:.4f} "
        f"scaled_anchor_m={anchor_m:.4f} "
        f"focus_target_m={focus_target_m:.4f}",
    )
    depth_m_arr = shifted_m

    _emit(
        on_progress,
        f"Rendering (depth_scale={depth_scale}, psf_norm={opts.psf_norm}, "
        f"hdr_source={opts.hdr_source})...",
    )
    render_stats: dict = {}
    render_out, output_linear = render(
        image_srgb,
        tensor,
        depth_m=depth_m_arr,
        psf_norm=opts.psf_norm,
        hdr_source=opts.hdr_source,
        bright_thresh=opts.bright_pass_thresh,
        bright_pass_region_size=opts.bright_pass_region_size,
        bright_pass_floor=opts.bright_pass_floor,
        bright_pass_norm_pct=opts.bright_pass_norm_pct,
        occlusion_compositing=getattr(opts, "occlusion_compositing", "off"),
        slab_batch_size=getattr(opts, "slab_batch_size", None),
        on_progress=on_progress,
        cancel=cancel,
        stats=render_stats,
    )

    # Q-a: focus-fidelity metrics on the pre-shadow-match linear output.
    # private: Q-a probe/slow-test hook (_eval_sink)
    _q_alpha = render_stats.pop("alpha", None)
    _q_dmap = render_stats.pop("dmap", None)
    _q_ref = render_stats.pop("img_lin", None)
    focus_fidelity = None
    if focus_xy is not None and _q_alpha is not None and _q_ref is not None:
        try:
            from summicron.render import _quality
            focus_fidelity = _quality.focus_fidelity(
                output_linear, _q_ref, _q_alpha, focus_xy
            )
        except Exception as exc:  # diagnostics must never fail a render
            warnings.warn(
                f"Q1 focus-fidelity computation failed ({exc!r}); "
                "focus_fidelity=None",
                RuntimeWarning,
                stacklevel=2,
            )
            focus_fidelity = None
    if _eval_sink is not None:
        _eval_sink["alpha"] = _q_alpha
        _eval_sink["dmap"] = _q_dmap
        _eval_sink["ref_lin"] = _q_ref
        _eval_sink["defocus_ratios"] = np.asarray(
            tensor["defocus_ratios"], dtype=np.float64
        )
    del _q_alpha, _q_dmap, _q_ref

    shadow_gains_rgb = None
    if opts.shadow_match_mode == "off":
        if opts.hdr_source == "none":
            output_u8 = render_out
        else:
            from summicron.io.hdr_tonemap_macos import tonemap_core_image
            output_u8 = tonemap_core_image(output_linear)
    else:
        if ldr_ref_linear is None:
            raise RuntimeError(
                "--shadow-match-mode on requires LDR reference; internal: "
                "ldr_ref_linear was not captured."
            )
        from summicron.io.hdr_tonemap_macos import tonemap_core_image
        preview_u8 = tonemap_core_image(output_linear)
        preview_linear = srgb_to_linear(
            preview_u8.astype(np.float32) / 255.0
        )
        gain_map, gains_rgb = _ldr_shadow_match_gain(
            preview_linear,
            ldr_ref_linear,
            shadow_band_hi=opts.shadow_band_hi,
            max_gain=opts.shadow_match_max_gain,
            strength=opts.shadow_match_strength,
        )
        lifted_linear = output_linear * gain_map
        _emit(
            on_progress,
            f"  K11 shadow-match: gains_rgb = ("
            f"{float(gains_rgb[0]):.3f}, "
            f"{float(gains_rgb[1]):.3f}, "
            f"{float(gains_rgb[2]):.3f})"
            f" (strength={opts.shadow_match_strength}; "
            f"band_hi={opts.shadow_band_hi})",
        )
        output_u8 = tonemap_core_image(lifted_linear)
        shadow_gains_rgb = (
            float(gains_rgb[0]),
            float(gains_rgb[1]),
            float(gains_rgb[2]),
        )

    output_path: Path | None = None
    if not getattr(opts, "save", True):
        pass  # API-only: skip every disk write
    elif opts.out_path is not None:
        out_path = Path(opts.out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        from PIL import Image as _PILImage
        if out_path.suffix.lower() == ".png":
            _PILImage.fromarray(output_u8, mode="RGB").save(str(out_path), "PNG")
        else:
            _PILImage.fromarray(output_u8, mode="RGB").save(
                str(out_path), "JPEG", quality=95
            )
        _emit(on_progress, f"  Saved: {out_path}")
        output_path = out_path
    else:
        output_path = save_results(
            output_u8, image_path.stem, out_dir,
            depth_scale=depth_scale, downres=opts.downres,
        )

    _emit(on_progress, "Done.")

    mode = "hdr" if opts.hdr_source != "none" else "ldr"
    # ``output_linear`` semantics:
    # - HDR path: always populated (render() returns linear float32).
    # - LDR path: ``render_out`` and ``output_linear`` are the uint8 sRGB
    #   array and the linear float32 buffer respectively. Surface the
    #   linear buffer only when the caller asked for it.
    out_lin = output_linear if (mode == "hdr" or opts.return_linear) else None
    diagnostics = RenderDiagnostics(
        center_ssim=(
            focus_fidelity["ssim_band"] if focus_fidelity is not None else None
        ),
        focus_fidelity=focus_fidelity,
        bright_pixels_pct=float(render_stats.get("bright_pixels_pct", 0.0)),
        alpha_mean=float(render_stats.get("alpha_mean", 0.0)),
        shadow_gains_rgb=shadow_gains_rgb,
        warnings=tuple(warnings_collected),
    )
    return RenderResult(
        output_u8=output_u8,
        output_linear=out_lin,
        mode=mode,  # type: ignore[arg-type]
        elapsed_s=0.0,  # api.render_image() overwrites with real wall time
        diagnostics=diagnostics,
        output_path=output_path,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint shim. Parses ``argv``, validates, runs the pipeline.

    P1.audit Items 3 + 9: signature is now ``main(argv) -> int`` so
    ``sys.exit(main())`` propagates the exit code. Successful renders
    return 0; ``parser.error()`` exits 2 directly.

    The pipeline body lives in :func:`_run_render_pipeline` as a pure
    refactor (driven by :class:`~summicron.render.api.RenderOptions`).
    GUIs and tests should use :func:`summicron.render.api.render_image`
    directly rather than synthesizing argv.
    """
    from summicron.render.api import RenderOptions

    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_args(args, parser)
    opts = RenderOptions.from_argparse(args)
    return run_cli_render(lambda: _run_render_pipeline(opts))


def run_cli_render(run) -> int:
    """Shared CLI error contract (``isummicron render`` and ``main``).

    Validates ``ISUMMICRON_MAX_BYTES`` *before* ``run`` loads any image, then
    maps expected failures to one ``render: <msg>`` stderr line + exit 2.
    ``KeyboardInterrupt`` and other exceptions propagate, as do
    ``NotImplementedError`` / ``RecursionError`` (programming errors), and
    everything when ``ISUMMICRON_DEBUG`` is set to a non-empty value other
    than ``0`` (full traceback).
    """
    try:
        _budget_bytes()
        run()
    except (NotImplementedError, RecursionError):
        raise
    except (ValueError, RuntimeError, FileNotFoundError,
            TensorValidationError) as e:
        if os.environ.get("ISUMMICRON_DEBUG", "") not in ("", "0"):
            raise
        print(f"render: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
