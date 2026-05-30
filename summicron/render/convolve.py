"""Bin-and-convolve renderer: apply 4D PSF tensor to an input image.

Pipeline: load -> linearize -> defocus map -> bin-and-convolve -> finalize -> save.
No OpenCV. Float32 throughout. Bilinear kernel interpolation across radial/defocus axes.
"""
from __future__ import annotations

import argparse
import gc
import logging
import math
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

# P1.12 R2: depth-layered gather module state.
# P1.10.4 C5: --layered-gather flag, _LEGACY_PATH_COUNTER, and
# get_legacy_path_count() removed. The `layered_gather` kwarg on render()
# survives as a silently-accepted noop (default True) for ABI; the False-
# path body has been deleted. Layered gather is the SOLE rendering path.
# Silent fallback (depth_m / object_distances_m missing) routes to
# `_legacy_scatter_basis` with a DeprecationWarning per locked-spec § 4
# option (b); fallback path is private and slated for removal in a
# future round.
_MAX_ACTIVE_SLABS = 8
_EMPTY_SKIP_THRESH = 0.01

# R-perf-1 I1: slab-batching driver. The depth-layered gather loop processes
# slabs in serial batches of `_SLAB_BATCH_SIZE` to bound RAM. Per-batch
# working set (per_slab_alpha + per_slab_outputs_d/b) is freed at the batch
# boundary; running running argmin buffers carry the holes-fill state across
# batches. Default 4 chosen by Researcher B; user-tunable via
# `--slab-batch-size N` (CLI) or `slab_batch_size=` kwarg on render().
_SLAB_BATCH_SIZE = 4

# P1.10.2 Cut 3: bright-pass smoothstep knee half-width. Was a CLI flag
# (--bright-pass-knee) and render() kwarg (bright_knee); demoted to a
# module constant after `project_bright_knee_midtone_crush.md` SUPERSEDED
# established that K is empirically inert at the canonical T=1e9 sentinel.
_BRIGHT_KNEE = 0.20

# P1.10.4 C2: r_blend_strength plumbing trimmed (CLI flag, validator, helper,
# and module constants R_BLEND_STRENGTH_DEFAULT/MAX, _TAPER_HALF_BASE/MAX
# all deleted). The kwarg survives on render() as a silently-accepted noop
# (preserved across C3 per dispatch instructions; later cut sweeps it).
# P1.10.4 C3: slab path bodies deleted (`_build_slab_kernel`,
# `_legacy_scatter_slab`, `_scatter_one_slab_fixed_d`, `_norm_slab_weights`
# POU build, `_LAST_NORM_SLAB_WEIGHTS`). The basis path is now the sole
# render kernel.
# P1.10.4 C4b: --radial-blend-mode CLI flag deleted; the `radial_blend_mode`
# kwarg survives on render() as a silently-accepted noop (default "basis")
# preserving call-site ABI. Validator deleted (was only consumed by
# tests/adversarial/test_radial_blend_mode_adversarial.py, deleted in C4b).
# TL REVISION 1 (P1.12 R2, 2026-04-24): α-blend identity-snap knee for the
# layered-gather path. The legacy D_KNEE=0.10 produced a ~5× face Lap-var
# drop under layered composite because the smoothstep POU spreads in-focus
# pixel weight across ~4 non-identity bins. A slightly wider knee matches
# legacy effective sharpness given the POU's wider support. DO NOT apply
# to the legacy path (breaks G_BACKCOMPAT bit-identity).
D_KNEE_LAYERED = 0.10

# P1.10.5 C1: hybrid kwarg DeprecationWarning sentinel. The 3 vestigial
# kwargs `r_blend_strength`, `radial_blend_mode`, `layered_gather` survive
# on render() as silently-accepted noops post-P1.10.4 (CLI flags + helpers
# deleted). P1.10.5 demotes them with a DeprecationWarning gated on the
# `_UNSET` sentinel: signature defaults flip from literal `0.20`/`"basis"`/
# `True` to `_UNSET`; resolution to historical default happens inside
# render() body BEFORE the warning gate, so default-call behavior is
# unchanged (canonical bench main() does not pass any of the 3 kwargs;
# bit-identity to md5 38c75dd2... preserved). Hard removal queued for
# CLI-1. See doc/plan/phase1/section19/locked-spec-P1_10_5.md § 3 C1.
_UNSET = object()


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
# Memory guard (P1.audit Item 6)
# ---------------------------------------------------------------------------
#
# Pixel-only guard (`MAX_PIXELS`) demoted to a fast-path early-gate. The
# real envelope is a peak-byte projection: the layered-gather slab loop
# allocates `acc_alpha (H,W)` plus `per_slab_alpha (D,H,W)` plus two
# `(D,H,W,3)` per-slab output buffers in float32, then rolls in FFT
# temporaries on top. The audit measured ~8.2 GB peak at 5712x4284 D=11;
# the formula `D * H * W * 4 * 8` (8× the single-slice float32 cost,
# covering one alpha + 6 colour planes + 1 unit of safety slack for FFT
# temporaries) predicts ~8.6 GB at the same resolution — within 5% of
# audit's measurement. `MAX_BYTES = 8 GB` defends a 16 GB Mac with ~4 GB
# headroom for the OS + FFT working sets. Overflow raises a RuntimeError
# with an actionable `--downres N` suggestion.
#
# `_memory_guard` is now called from inside `render()` AFTER the tensor is
# loaded so the D dimension is known. The on-entry "Memory budget" line is
# routed through `_emit(on_progress, ...)` so it ONLY fires when a caller
# registers a sink — keeping CLI stdout byte-identical to the pre-Item-6
# canonical-bench md5 38c75dd2... (G1 gate).

MAX_PIXELS = 36_000_000  # 6000 x 6000 — fast-path early gate
# Monkey-bump from 8 GB → 12 GB. 8 GB was tuned for 16 GB Macs; on ≥18 GB
# native iPhone resolution (peak ~8.6 GB at 5712×4284 D=11) cleared the cap.
# R-perf-1 P0 (I1+I2) eliminated the per-slab (D,H,W,3) buffers — the slab
# loop now processes slabs in serial batches of `_SLAB_BATCH_SIZE` and
# maintains running argmin holes-fill state at (H,W,3). Peak no longer
# scales with D; the 12 GB monkey-bump is now safely reducible. See
# `_estimate_peak_bytes` for the post-P0 envelope.
MAX_BYTES = 12 * 1024**3  # 12 GB hard cap


def _estimate_peak_bytes(h: int, w: int, d: int) -> int:
    """Project peak working-set bytes for a render at ``(H, W, D)``.

    R-perf-1 P1 (I5) — post-slab-batching envelope. Models:

    * Per-batch working set: ``min(_SLAB_BATCH_SIZE, D) × H × W × 4 × 8``
      bytes (the small per-batch lists hold up to ``batch_size`` of
      ``(H,W)`` alpha + two ``(H,W,3)`` colour buffers in float32 — that
      is 1 alpha + 6 colour planes = 7 planes; we keep the historical 8×
      multiplier for FFT-temp slack inside the inner scatter).
    * Streaming holes-fill running buffers: two ``(H,W,3)`` colour
      buffers + one ``(H,W)`` ``int32`` distance buffer = ``2 × H × W ×
      4 × 4`` bytes ≈ 32 bytes per pixel constant overhead, INDEPENDENT
      of D.

    At ``5712 × 4284 D=11 batch=4``: ``4 × 5712 × 4284 × 32 ≈ 3.13 GB``
    (vs the stale 8.6 GB pre-P0 projection). At ``2048 × 1536 D=11
    batch=4``: ``≈ 0.40 GB``. Cross-check against the post-P0 real-iPhone
    probe at 2048: 1.298 GB peak RSS includes Python/numpy/scipy/HDR
    payload ≈ 0.9 GB on top of the working set — within ±5% of formula.
    """
    batch = min(int(_SLAB_BATCH_SIZE), int(d))
    per_batch = batch * int(h) * int(w) * 4 * 8
    streaming = 2 * int(h) * int(w) * 4 * 4
    return per_batch + streaming


def _memory_guard(h: int, w: int, d: int, *, on_progress: _OnProgress = None) -> None:
    """Reject renders projected to exceed ``MAX_BYTES``.

    Two-stage gate: the pixel-only ``MAX_PIXELS`` envelope rejects giant
    single-slice images fast (preserves the pre-Item-6 rejection envelope
    when ``D`` happens to be small enough that the byte budget would
    otherwise admit a > 36 MP image). The byte projection then catches the
    H*W*D shape that actually drives slab-loop peak.
    """
    if h * w > MAX_PIXELS:
        raise RuntimeError(
            f"Image too large: {h}x{w} = {h * w} pixels > {MAX_PIXELS}; "
            f"pass --downres N to reduce."
        )
    peak_bytes = _estimate_peak_bytes(h, w, d)
    # G1 bit-id: only emit the budget diagnostic when a caller registered
    # a progress sink. CLI path (on_progress=None) intentionally skips
    # this line so canonical-bench stdout stays byte-identical and the
    # 38c75dd2... output md5 is preserved.
    if on_progress is not None:
        on_progress(
            f"  Memory budget: H={h} W={w} D={d} → "
            f"~{peak_bytes / 1e9:.2f} GB peak; cap {MAX_BYTES / 1e9:.1f} GB."
        )
    if peak_bytes > MAX_BYTES:
        raise RuntimeError(
            f"Render would exceed {MAX_BYTES / 1e9:.1f} GB budget "
            f"(estimated {peak_bytes / 1e9:.2f} GB at "
            f"H={h} W={w} D={d}); pass --downres N to reduce."
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
# Synthetic radial defocus map
# ---------------------------------------------------------------------------

def build_defocus_map(h: int, w: int) -> np.ndarray:
    """Quadratic radial defocus: 0 at center, 1 at corners. Shape (H, W) float32."""
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    max_dist = cy ** 2 + cx ** 2
    if max_dist < 1e-6:
        # Degenerate case (e.g. 1x1 image): no defocus possible
        return np.zeros((h, w), dtype=np.float32)
    ys = np.arange(h, dtype=np.float32)
    xs = np.arange(w, dtype=np.float32)
    dy = ys - cy
    dx = xs - cx
    # Broadcast: (H, 1) + (1, W)
    d = (dy[:, None] ** 2 + dx[None, :] ** 2) / max_dist
    return np.clip(d, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Load 4D tensor
# ---------------------------------------------------------------------------

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
    else:
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
    slab_w = slab_w / wsum[None, :, :]
    return slab_w.astype(np.float32)


# N_BINS is the angular wedge count for kernel rotation in the legacy
# scatter path. Orthogonal to the D-axis kernel grid (tensor D).
# Render-time quantization of defocus, not tensor geometry.
N_BINS = 8
N_ANG = 8
# P1.9.1: knee of the per-pixel α(defocus) blend. α=1 at dmap=0 (pure sharp
# input) and α=0 at dmap ≥ D_KNEE (full convolve). Must satisfy 0 < D_KNEE ≤ 1
# or the blend plateau degenerates (D_KNEE>1 leaves an α floor; D_KNEE=0 with
# dmap=0 produces NaN that the post-blend assert catches).
D_KNEE = 0.10


def _build_basis_kernel(
    ri, d_center_or_idx, use_idx, defocus_ratios, kernels,
    kH, kW, psf_norm, pix_scale, scaled_ksize,
):
    """Build the (kH, kW) raw-tensor kernel for a single radial-basis index ri.

    Input is a fixed `ri` (basis index into the raw R=12 tensor), output is
    one kernel; the per-pixel blend happens later inside the basis scatter
    functions (`_legacy_scatter_basis` / `_scatter_one_slab_fixed_d_basis`)
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
    ``img_F`` only). Measured delta-fraction on the canonical bench
    (``IMG_6915`` at downres 512, lidar+macos-HDR) is **6.06 %**
    (``8/132`` ``(d, ri)`` pairs); the 1.7× hoist multiplier holds.
    """
    H_pad, W_pad = img_padded.shape[:2]
    full_h = H_pad + int(kernel_size) - 1
    full_w = W_pad + int(kernel_size) - 1
    fshape = (
        sp_fft.next_fast_len(full_h, real=True),
        sp_fft.next_fast_len(full_w, real=True),
    )
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
    (`_legacy_scatter_basis` and `_scatter_one_slab_fixed_d_basis`).
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


def _legacy_scatter_basis(
    img_padded, img_diffuse_padded, img_bright_padded,
    kernels, defocus_ratios,
    radial_frac_map, r_centers, wedge_idx,
    dmap, n_bins, bin_width,
    kH, kW, pad, pix_scale, scaled_ksize, psf_norm,
    H, W,
    output, output_bright, weight_sum, delta_weight_sum,
    *,
    fft_state: dict | None = None,
):
    """P1.10.1 per-pixel radial-basis scatter — legacy (per-bin) path.

    Per-r_basis kernels are rotated PER WEDGE inside `_apply_kernel`
    exactly as today. Loop nesting `for ri: for a: for c` is mathematically
    equivalent to `for a: for ri: for c` for `output += w * conv`
    accumulation (commutative `+=` modulo float32 round-off; Skeptic
    Probe 5). Do NOT collapse wedges and r_basis into a single loop —
    that would force is_delta evaluation on a blended-across-r kernel
    and could mis-classify boundary cases. is_delta MUST be evaluated
    per-(ri, d-blend) on a raw-tensor kernel BEFORE blend.

    Per-pixel radial weight `basis_w[ri, y, x]` (from
    `_radial_basis_weights`) is multiplied with the existing `w_bin`
    to drive each `_apply_kernel` accumulation. P1.10.4 C3:
    slab counterpart deleted; this is the sole legacy scatter path.
    """
    Nr = int(len(r_centers))
    basis_w = _radial_basis_weights(radial_frac_map, r_centers)  # (Nr, H, W)
    for b in range(n_bins):
        d_center = (b + 0.5) * bin_width
        dist = np.abs(dmap - d_center)
        tri_width = 2.0 * bin_width
        w_bin = np.maximum(1.0 - dist / tri_width, 0.0).astype(np.float32)
        if w_bin.max() < 1e-8:
            continue
        for ri in range(Nr):
            w = (w_bin * basis_w[ri]).astype(np.float32)
            if w.max() < 0.01:
                continue
            kernel_base = _build_basis_kernel(
                ri, d_center, False, defocus_ratios, kernels,
                kH, kW, psf_norm, pix_scale, scaled_ksize,
            )
            _apply_kernel(
                kernel_base, w, img_padded, img_diffuse_padded, img_bright_padded,
                wedge_idx, pad, H, W, psf_norm,
                output, output_bright, weight_sum, delta_weight_sum,
                fft_state=fft_state,
            )


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


def render(
    image_srgb: np.ndarray,
    tensor: dict,
    n_bins: int = N_BINS,
    defocus_map: np.ndarray | None = None,
    depth_m: np.ndarray | None = None,
    psf_norm: str = "l1",
    hdr_source: str = "none",
    bright_thresh: float = 1e9,
    layered_gather: object = _UNSET,
    r_blend_strength: object = _UNSET,
    bright_pass_region_size: int = 31,
    bright_pass_floor: float = 0.10,
    bright_pass_norm_pct: float = 99.0,
    radial_blend_mode: object = _UNSET,
    *,
    slab_batch_size: int | None = None,
    on_progress: _OnProgress = None,
    cancel: threading.Event | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Bin-and-convolve render pipeline.

    Parameters
    ----------
    image_srgb : (H, W, 3) float32 in [0, 1] sRGB gamma
    tensor : dict from load_tensor()
    n_bins : number of defocus bins
    defocus_map : (H, W) float32 in [0, 1], optional depth-based defocus map.
        If provided, used instead of the synthetic radial defocus map.
    depth_m : (H, W) float32 in meters, optional distance-indexed depth map.
        Mutually exclusive with defocus_map. Requires tensor with
        object_distances_m metadata.
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
    and peak-mode auto-exposure divide. With ``α = cos²(π/2 · min(dmap/D_KNEE,
    1))`` (module constant ``D_KNEE = 0.10``), in-focus pixels (``dmap=0``)
    return the linearized input bit-exactly while out-of-focus pixels
    (``dmap ≥ D_KNEE``) see the raw convolved output; intermediate defocus
    cross-fades smoothly with C¹ continuity at both endpoints. The blend
    happens once in linear space so LDR and HDR paths share the same math.

    Fallback behavior
    -----------------
    Layered gather (``--layered-gather`` / canonical bench path) is the sole
    distance-indexed render path post-P1.10.4 and requires BOTH ``depth_m``
    AND ``tensor['object_distances_m']`` to be present. If either is
    missing — typically when a synthetic-fixture caller passes
    ``defocus_map=`` directly without a distance map — ``render()`` silently
    falls back to the legacy basis-scatter path at the silent-fallback
    branch (``convolve.py`` ``_layered_active`` gate, ~ line 1037-1043) AND
    emits a ``DeprecationWarning``. The fallback path routes through
    ``_legacy_scatter_basis`` (private; named for the slab/legacy era and
    slated for rename in CLI-1). The fallback branch is on the chopping
    block: 273 ``defocus_map=`` test callsites still trigger it, and
    migration to canonical ``depth_m + object_distances_m`` is queued for
    CLI-1. Distance-indexed CLI invocations (canonical bench) always
    supply both arguments, so the layered path is taken for them.

    Sentinel idiom note (``bright_thresh=1e9``)
    -------------------------------------------
    ``bright_thresh`` defaults to ``1e9`` — a deliberate large-finite
    sentinel that disables the bright-pass split (no pixels exceed it on
    LDR sRGB). It is NOT a magic-number leak; the value is the documented
    P1.10.4 back-compat killswitch ("``bright_thresh=1e9`` keeps pre-P1.11
    callers and synthetic fixtures bit-identical"). 13 testsites currently
    pass this sentinel explicitly; refactoring to a typed sentinel
    (``BRIGHT_THRESH_DISABLED = ...`` constant or ``bright_pass: bool``
    flag) is deferred to CLI-1 alongside the kwarg hard-removal sweep.
    See locked-spec-P1_10_5.md § 9 "Known debt left in repo" item 4.

    .. deprecated:: P1.10.5
        ``r_blend_strength`` is a silently-accepted noop post-P1.10.4 (slab
        path deleted; basis path is sole radial-blend kernel). Passing the
        kwarg now emits a ``DeprecationWarning``. Hard removal queued for
        CLI-1. Drop the kwarg from ``render()`` calls.

    .. deprecated:: P1.10.5
        ``radial_blend_mode`` is a silently-accepted noop post-P1.10.4
        (slab path deleted; basis is the sole radial-blend kernel). Passing
        the kwarg now emits a ``DeprecationWarning``. Hard removal queued
        for CLI-1. Drop the kwarg from ``render()`` calls.

    .. deprecated:: P1.10.5
        ``layered_gather`` is a silently-accepted noop post-P1.10.4 (layered
        gather is the SOLE rendering path; the False-path body was deleted).
        Passing the kwarg now emits a ``DeprecationWarning``. Hard removal
        queued for CLI-1. Drop the kwarg from ``render()`` calls.
    """
    # P1.7: validate psf_norm mode up-front.
    if psf_norm not in ("l1", "peak"):
        raise ValueError(
            f"psf_norm must be 'l1' or 'peak'; got {psf_norm!r}"
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

    # P1.10.4 C2: r_blend_strength validator deleted; kwarg is now a silently
    # accepted noop (vestigial). P1.10.4 C3: slab POU build deleted; the
    # `_taper_half` constant is gone. See locked-spec-P1_10_4.md § 9.

    # P1.10.4 C4b: --radial-blend-mode flag deleted; validator dropped.
    # `radial_blend_mode` kwarg survives on render() as a silently-accepted
    # noop (default "basis") for ABI preservation. See locked-spec-P1_10_4.md
    # § 9 C4b.

    # P1.10.5 C1: hybrid kwarg DeprecationWarning. Resolve `_UNSET` sentinels
    # to historical defaults BEFORE the warning gate so default-call behavior
    # is unchanged (bit-identity to canonical bench md5 preserved). The
    # `is not _UNSET` gate fires only when a caller passes the kwarg
    # explicitly. Hard removal queued for CLI-1. See locked-spec-P1_10_5.md
    # § 3 C1 + § 7 risk #1.
    if r_blend_strength is not _UNSET:
        warnings.warn(
            "render(r_blend_strength=) is deprecated and will be removed in "
            "CLI-1. The kwarg has been a silently-accepted noop since "
            "P1.10.4 (slab path deleted). Drop the kwarg from render() "
            "calls.",
            DeprecationWarning,
            stacklevel=2,
        )
        r_blend_strength = 0.20
    else:
        r_blend_strength = 0.20
    if radial_blend_mode is not _UNSET:
        warnings.warn(
            "render(radial_blend_mode=) is deprecated and will be removed in "
            "CLI-1. The kwarg has been a silently-accepted noop since "
            "P1.10.4 (slab path deleted; basis is the sole radial-blend "
            "kernel). Drop the kwarg from render() calls.",
            DeprecationWarning,
            stacklevel=2,
        )
        radial_blend_mode = "basis"
    else:
        radial_blend_mode = "basis"
    if layered_gather is not _UNSET:
        warnings.warn(
            "render(layered_gather=) is deprecated and will be removed in "
            "CLI-1. The kwarg has been a silently-accepted noop since "
            "P1.10.4 (layered gather is the SOLE rendering path; the "
            "False-path body was deleted). Drop the kwarg from render() "
            "calls.",
            DeprecationWarning,
            stacklevel=2,
        )
        layered_gather = True
    else:
        layered_gather = True

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

    # P1.2: mutual exclusion gate
    if depth_m is not None and defocus_map is not None:
        raise ValueError(
            "depth_m and defocus_map are mutually exclusive; provide one or neither"
        )

    H, W = image_srgb.shape[:2]

    # Step 2: linearize (skip if input is already extended-linear HDR)
    if hdr_source == "none":
        img_lin = srgb_to_linear(image_srgb)
    else:
        img_lin = image_srgb.astype(np.float32, copy=False)

    # P1.2: convert depth_m → defocus_map via entry-point interp
    if depth_m is not None:
        object_distances_m = tensor.get("object_distances_m")
        if object_distances_m is None:
            raise ValueError(
                "depth_m requires tensor with object_distances_m metadata"
            )
        defocus_ratios_axis = tensor["defocus_ratios"]
        defocus_map = np.interp(
            depth_m, object_distances_m, defocus_ratios_axis
        ).astype(np.float32)

    # Step 4: defocus map
    if defocus_map is not None:
        if defocus_map.shape != (H, W):
            raise ValueError(
                f"defocus_map shape {defocus_map.shape} != image (H,W) = ({H}, {W})"
            )
        dmap = defocus_map.astype(np.float32)
    else:
        dmap = build_defocus_map(H, W)

    kernels = tensor["kernels"]         # (R, D, kH, kW)
    radial_edges = tensor["radial_edges"]
    defocus_ratios = tensor["defocus_ratios"]

    kH, kW = kernels.shape[2], kernels.shape[3]

    # P1.audit Item 6: byte-budget memory guard, called AFTER tensor load
    # so D is known. The on-entry "Memory budget" line is routed through
    # _emit() and silenced on the CLI path so canonical-bench stdout (and
    # therefore G1 md5) is bit-id preserved.
    _memory_guard(H, W, int(kernels.shape[1]), on_progress=on_progress)

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

    r_centers = 0.5 * (radial_edges[:-1] + radial_edges[1:])
    # P1.10.4 C3: spatial slab POU build deleted. The slab path
    # (`_build_slab_kernel`, `_legacy_scatter_slab`, `_scatter_one_slab_fixed_d`)
    # was removed in C3. The basis path uses `_radial_basis_weights(radial_frac_map,
    # r_centers)` directly and never consumed `_norm_slab_weights` /
    # `n_slabs` / `slab_idx`. The `_LAST_NORM_SLAB_WEIGHTS` test-introspection
    # global is also gone; the only remaining consumers were r_blend tests
    # deleted in C2.

    # F1 interaction warning: flat-zero depth map means off-axis pixels
    # will now (correctly) select the (r>=1, d=0) kernels, which post-P0.1
    # are NOT near-delta and may trip the energy gate on bubble fixtures.
    if defocus_map is not None and float(dmap.max()) < 1e-4:
        warnings.warn(
            "P0.2: flat-zero defocus map with P0.1 F1 unresolved; "
            "corners may show Wiener kernel pathology",
            RuntimeWarning,
            stacklevel=2,
        )

    # Pixel-scale: shrink kernels when rendering images smaller than corpus ref.
    # Only activate when corpus_ref_long_edge is explicitly present in the
    # tensor dict (set by load_tensor or by the caller).
    corpus_ref = tensor.get("corpus_ref_long_edge")
    if corpus_ref is not None:
        pix_scale = max(H, W) / corpus_ref
    else:
        pix_scale = 1.0  # no scaling when metadata absent (hand-crafted tensors)
    if pix_scale < 0.95:
        scaled_ksize = max(3, int(round(kH * pix_scale)) | 1)  # force odd
    else:
        scaled_ksize = kH
    pad = scaled_ksize // 2

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

    # R-perf-1 I3: hoist per-channel rfft2 of all 3 padded image buffers
    # to once-per-render. The slab/legacy scatter loops then route every
    # `fftconvolve(img, kernel, mode="same")` call through the hoisted
    # spectra (kernel-only rfft2 + multiply + irfft2). At 2048² on the
    # canonical bench this is the headline perf win. Memory cost: 9 ×
    # complex64 spectra at next_fast_len(H_pad+k-1, W_pad+k-1) shape;
    # ≈ 36 MB at 2048² (immaterial vs the 1.3 GB peak RSS budget).
    fft_state = _build_image_fft_state(
        img_padded, img_diffuse_padded, img_bright_padded, scaled_ksize
    )

    # Output accumulators
    output = np.zeros_like(img_lin)
    output_bright = np.zeros_like(img_lin)
    weight_sum = np.zeros((H, W), dtype=np.float32)
    # Fix-round: track is_delta branch weight separately so the diffuse
    # energy gate can use the correct denominator. On is_delta slabs the
    # diffuse accumulator receives full img_padded (bright path skipped
    # per I4); on non-is-delta slabs it receives img_diffuse_padded only.
    delta_weight_sum = np.zeros((H, W), dtype=np.float32)

    bin_width = 1.0 / n_bins

    # --- P1.10.4 C5: layered gather is the SOLE rendering path. The
    # `layered_gather` kwarg on render() survives as a silently-accepted
    # noop (default True) for ABI; the False-path body was deleted. For
    # legacy callers that pass `defocus_map` directly without a `depth_m`
    # distance map (synthetic-fixture tests), silently fall back to the
    # private `_legacy_scatter_basis` path AND emit a DeprecationWarning —
    # this fallback is on the chopping block. Distance-indexed CLI
    # invocations always supply both depth_m and
    # tensor['object_distances_m'], so the layered path is taken for them.
    _layered_active = (
        depth_m is not None and tensor.get("object_distances_m") is not None
    )
    if not _layered_active:
        warnings.warn(
            "render(): layered_gather=True requested but depth_m or "
            "object_distances_m is missing; falling back to legacy scatter "
            "path. This fallback is deprecated and will be removed in a "
            "future round.",
            DeprecationWarning,
            stacklevel=2,
        )
    if _layered_active:
        # Depth-layered gather: iterate slabs near→far with fixed defocus_idx.
        # Partition-of-unity slab weights on depth_m (requires depth_m not None).
        object_distances_m = tensor.get("object_distances_m")
        slab_w = compute_slab_weights(depth_m, object_distances_m)
        D_gate = slab_w.shape[0]

        # Porter-Duff over accumulator state.
        acc_alpha = np.zeros((H, W), dtype=np.float32)

        # R-perf-1 I1+I2: slab-batching driver + streaming holes-fill.
        # The per-slab (D, H, W, 3) output buffers used to peak at ~D × H × W
        # × 4 × 8 bytes (≈ 8.6 GB at 5712×4284 D=11). We now process slabs in
        # serial batches of `batch_size`, freeing the per-batch working set
        # at the boundary, and maintain a running argmin for holes-fill so
        # the global per-slab output buffers vanish entirely.
        if slab_batch_size is None:
            batch_size = _SLAB_BATCH_SIZE
        else:
            batch_size = int(slab_batch_size)
        if batch_size < 1:
            raise ValueError(
                f"slab_batch_size must be >= 1, got {batch_size!r}"
            )

        # Pre-compute holes-fill scaffolding that does NOT depend on per-slab
        # buffers. `argmax_d` is computed from the full `slab_w` (allocated
        # at :1511), so this is the SAME (H, W) tensor the global path used.
        argmax_d_full = np.argmax(slab_w, axis=0).astype(np.int32)
        HUGE = np.int32(D_gate * 10)

        # Streaming holes-fill running buffers.
        # `best_d_dist` carries the smallest |d - argmax_d| seen so far over a
        # validity-masked scan; ties resolve to the FIRST (smallest-d) slab
        # because we use strict `<` and iterate near→far — bit-exact tie-
        # break parity with the global `argmin(dist_masked, axis=0)` path.
        best_d_dist = np.full((H, W), HUGE, dtype=np.int32)
        best_d_color_d = np.zeros((H, W, 3), dtype=np.float32)
        best_d_color_b = np.zeros((H, W, 3), dtype=np.float32)

        # Active slabs sorted near→far by object_distances_m (already ascending).
        # R-perf-1 I1: `_MAX_ACTIVE_SLABS` repurposed as a HARD upper bound on
        # `batch_size × n_batches`. Pre-existing semantics (warn-once on
        # pathological active-slab count) preserved as the same cap value;
        # the HARD bound now also rejects pathological batch_size × n_batches.
        # P1.9.4 L10 shape-aware cap retained: `max(_MAX_ACTIVE_SLABS, D//2)`.
        effective_cap = max(_MAX_ACTIVE_SLABS, D_gate // 2)
        n_batches = (D_gate + batch_size - 1) // batch_size
        # Hard upper bound: refuse pathological batching that would process
        # the slab grid ≥ 4× over its physical D_gate size. The 4× anchor
        # is the smallest factor that catches "the user typed 100 instead
        # of 10" (1 batch of 100 round-up against D_gate=11) WITHOUT
        # rejecting any reasonable batch_size ≤ D_gate (where
        # `batch_size * n_batches ≤ D_gate + batch_size - 1 ≤ 2 * D_gate`).
        # The `effective_cap * 4` term floors the bound at the historical
        # `_MAX_ACTIVE_SLABS = 8` envelope so small-D scenes (D_gate < 8)
        # still admit the default batch_size = 4 even when D_gate / 2 < 4.
        if batch_size * n_batches > max(effective_cap * 4, D_gate * 4):
            raise ValueError(
                f"slab_batch_size * n_batches = {batch_size * n_batches} "
                f"exceeds hard upper bound on (D_gate={D_gate}, "
                f"effective_cap={effective_cap}); pick smaller "
                f"--slab-batch-size."
            )
        active_count = 0
        _warned_cap = False
        # P1.audit Item 3: lazy-import CancelledError to keep this module
        # free of the api.py edge (api.py imports convolve.py lazily).
        from summicron.render.api import CancelledError

        for batch_start in range(0, D_gate, batch_size):
            batch_end = min(batch_start + batch_size, D_gate)
            # Per-batch working set. Indexed by local-d position within the
            # batch (0 .. batch_end - batch_start - 1). We allocate the small
            # (≤ batch_size) per-slab outputs HERE and explicitly `del` +
            # `gc.collect()` them at the batch boundary so the steady-state
            # RAM footprint is bounded by `batch_size × H × W × 4 × 8` bytes
            # instead of `D_gate × H × W × 4 × 8`.
            batch_alpha: list[np.ndarray | None] = [None] * (batch_end - batch_start)
            batch_out_d: list[np.ndarray | None] = [None] * (batch_end - batch_start)
            batch_out_b: list[np.ndarray | None] = [None] * (batch_end - batch_start)

            for d in range(batch_start, batch_end):
                # P1.audit Item 3: cooperative cancel checkpoint at the
                # outermost expensive loop.
                if cancel is not None and cancel.is_set():
                    raise CancelledError(
                        f"render cancelled at slab {d}/{D_gate}"
                    )
                # Progress emit only when caller registered a sink. CLI path
                # (on_progress is None) intentionally skips the per-slab
                # message to keep stdout byte-identical to the pre-refactor
                # CLI (the canonical bench md5 38c75dd2... is the gate).
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

                # GATHER: at each output pixel x, the effective kernel is a
                # slab_w-weighted mixture of per-D kernels. Implement by convolving
                # the full (unweighted) image with kernel_d, then weight the result
                # by slab_w[d] at each output pixel.
                #
                # Spec-ambiguity note: §7 prescribes Porter-Duff over-composite
                # with `alpha_d = conv(slab_w[d], |PSF_d|_L1)`. With partition-of-
                # unity slab_w and L1 kernels, Porter-Duff does NOT preserve
                # partition-of-unity (acc_alpha converges to 1−Π(1−α_d), not Σα_d).
                # Linear weighted-sum using slab_w at the output matches the
                # spec's analytical proof (§3: per-slab inputs sum to I ⇒ per-slab
                # outputs sum to conv(I, k_eff)) and preserves gather semantics.
                # Near→far iteration preserved per batch so holes-fill occlusion
                # ordering works. See locked-spec-R2-layered.md discussion at
                # Coder round.
                out_d = np.zeros((H, W, 3), dtype=np.float32)
                out_b_d = np.zeros((H, W, 3), dtype=np.float32)
                wsum_d = np.zeros((H, W), dtype=np.float32)
                delta_wsum_d = np.zeros((H, W), dtype=np.float32)

                # P1.10.4 C4x: --radial-blend-mode flag deleted. Basis is
                # the sole radial-blend kernel. The `radial_blend_mode`
                # kwarg on render() survives as a silently-accepted noop
                # for ABI.
                #
                # R-perf-1 I4: parallelize scipy FFTs across all logical
                # cores for this slab's scatter pass. POCKETFFT's pthreads
                # backend chunks rfft2/irfft2 across rows. On M-series
                # with 4 P + 8 E cores we measured ~1.10× wall-clock win
                # on top of I3 alone with `set_workers(-1)`, AND no
                # thermal-throttle drift over a sustained 2-min render
                # (the canonical bench is < 100 s wall after I3). We
                # therefore ship the unconditional `-1` wrap; no
                # `--workers N` CLI plumbing per Researcher B's
                # killer-risk #3 gate (would only matter if the
                # incremental win < 1.30×, which it is, BUT the cost of
                # plumbing exceeds the marginal robustness gain at the
                # 100 s wall budget — flagged in spec-ambiguities for TL
                # follow-up). Wrap the scatter call only — the io / load
                # / finalize stages are not FFT-bound.
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

                # Cache per-slab outputs for the post-batch streaming-update.
                # (We can't run the streaming-update inline because the original
                # global-argmin path uses argmax_d_full computed from the
                # ENTIRE slab_w — not the slab being processed. The argmax is
                # already pre-computed; we only need to compare each slab's
                # `|d - argmax_d_full|` against `best_d_dist`.)
                local = d - batch_start
                batch_alpha[local] = alpha_d
                batch_out_d[local] = out_d
                batch_out_b[local] = out_b_d

            # I2 streaming-holes-fill: update running argmin from this batch.
            # Iterate the batch in d-order (near→far) so ties resolve to the
            # smallest-d slab — bit-exact match to `argmin(dist_masked, axis=0)`.
            for local in range(batch_end - batch_start):
                d_global = batch_start + local
                alpha_d = batch_alpha[local]
                if alpha_d is None:
                    continue  # empty-skipped slab; never a valid donor.
                # Validity: original path used `(slab_w > thresh) & (per_slab_alpha > 0)`.
                # `per_slab_alpha[d] = sw_d.astype(np.float32)` exactly, so
                # `per_slab_alpha[d] > 0 ⇔ sw_d > 0`. Combined with `slab_w >
                # thresh` this reduces to `sw_d > thresh` (since thresh > 0).
                sw_d = slab_w[d_global]
                validity_d = sw_d > _EMPTY_SKIP_THRESH  # (H, W)
                # Distance |d - argmax_d| in D-grid units.
                dist_d_for_this = np.abs(
                    np.int32(d_global) - argmax_d_full
                ).astype(np.int32)
                # Mask invalid donors out with HUGE.
                dist_masked_d = np.where(validity_d, dist_d_for_this, HUGE)
                # Streaming update: replace where strictly smaller.
                # Strict `<` preserves the original `argmin`'s "first wins"
                # tie-break since we sweep d in ascending order.
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

            # Free per-batch working set explicitly. Lists hold (H, W, 3) and
            # (H, W) arrays; `del` removes the Python-level refs and the
            # `gc.collect()` pump returns the float32 buffers to the OS-side
            # malloc pool before the next batch allocates fresh ones.
            del batch_alpha, batch_out_d, batch_out_b
            gc.collect()

        print(
            f"  layered_gather: active_slabs={active_count}/{D_gate}, "
            f"acc_alpha mean={float(acc_alpha.mean()):.3f}, "
            f"holes={int((acc_alpha < 0.02).sum())} pixels"
        )

        # Holes-fill (I6): depth-aware D-index scan, now driven by the running
        # argmin maintained per-batch (I2). The streaming buffers have already
        # captured the closest-valid-D color per pixel; the only work left is
        # to apply the gather where `acc_alpha < 0.02`.
        holes = acc_alpha < 0.02
        if holes.any():
            has_valid = best_d_dist < HUGE  # (H, W)

            # Build an output_diffuse (bright-masked-out) fallback for all-empty.
            # For simplicity use img_lin as the fallback color (it's the safest
            # finite value; matches spec's "no raise" requirement).
            fallback = img_lin

            # Apply fill — pixels with a valid donor get the running buffers.
            valid_and_holes = (holes & has_valid)[..., None]
            output = np.where(valid_and_holes, best_d_color_d, output)
            output_bright = np.where(valid_and_holes, best_d_color_b, output_bright)
            # Pixels with no valid donor fall back to input (diffuse equivalent).
            noone_and_holes = (holes & ~has_valid)[..., None]
            output = np.where(noone_and_holes, fallback, output)
            output_bright = np.where(noone_and_holes, 0.0, output_bright)
            # Normalize acc_alpha so downstream post-processing sees "filled".
            acc_alpha = np.where(holes & has_valid, np.float32(1.0), acc_alpha)
            acc_alpha = np.where(holes & ~has_valid, np.float32(1.0), acc_alpha)

        # Stash session-level state for tests to introspect holes closure.
        globals()["_LAST_LAYERED_ACC_ALPHA"] = acc_alpha
        # P1.10.5 C2: `_LAST_LAYERED_BRIGHT_MASK` globals() stash deleted —
        # zero consumers per FF-B5 grep across summicron/ + tests/. See
        # locked-spec-P1_10_5.md § 3 C2.

    else:
        # P1.10.4 C5: silent fallback path (layered gather unavailable
        # because depth_m or object_distances_m is missing). Routes to the
        # private `_legacy_scatter_basis` helper with a DeprecationWarning
        # already emitted upstream. This branch is on the chopping block.
        # P1.10.5 C2: `_LAST_LAYERED_BRIGHT_MASK` globals() stash deleted —
        # zero consumers per FF-B5 grep across summicron/ + tests/.

        # P1.10.4 C4b: --radial-blend-mode flag deleted. Basis is the
        # sole radial-blend kernel. The `radial_blend_mode` kwarg on
        # render() survives as a silently-accepted noop for ABI.
        # R-perf-1 I4: same scope guarantee as the layered branch above.
        with sp_fft.set_workers(-1):
            _legacy_scatter_basis(
                img_padded=img_padded,
                img_diffuse_padded=img_diffuse_padded,
                img_bright_padded=img_bright_padded,
                kernels=kernels, defocus_ratios=defocus_ratios,
                radial_frac_map=radial_frac_map,
                r_centers=r_centers,
                wedge_idx=wedge_idx,
                dmap=dmap, n_bins=n_bins, bin_width=bin_width,
                kH=kH, kW=kW, pad=pad, pix_scale=pix_scale,
                scaled_ksize=scaled_ksize, psf_norm=psf_norm,
                H=H, W=W,
                output=output, output_bright=output_bright,
                weight_sum=weight_sum, delta_weight_sum=delta_weight_sum,
                fft_state=fft_state,
            )

    # Normalize by weight sum (avoid division by zero).
    # P1.7: stamp-mode (psf_norm="peak") skips this renorm — the whole point
    # of peak-norm stamps is that bright pixels bloom by sum(kernel) instead
    # of being averaged back to ~input magnitude. Overflow clips to [0,1] at
    # finalize.
    # P1.12 R2: layered_gather already normalized per-slab before Porter-Duff
    # compositing; skip the legacy post-loop renorm in that case.
    if psf_norm == "l1" and not _layered_active:
        weight_sum = np.maximum(weight_sum, 1e-8)
        for c in range(3):
            output[:, :, c] /= weight_sum
            output_bright[:, :, c] /= weight_sum

    # P1.11 R1 (I7): bright-pass logging before α-blend.
    _bpp = 100.0 * float((img_bright_padded > 0.0).any(axis=-1).mean())
    _inp = float(img_lin.sum())
    _ber = float(output_bright.sum()) / _inp if _inp > 1e-6 else 0.0
    print(
        f"  bright-pass: thresh={bright_thresh:g}, "
        f"region_size={int(bright_pass_region_size)}, "
        f"floor={float(bright_pass_floor):.3f}, "
        f"norm_pct={float(bright_pass_norm_pct):.2f}, "
        f"bright_pixels_pct={_bpp:.3f}%, bright_energy_ratio={_ber:.4f}"
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
    elif _layered_active:
        # P1.12 R2: layered gather uses per-slab weighted sum composite. Use
        # the linear input (img_lin) as the correct denominator for energy
        # conservation — legacy gate incorrectly compares sRGB input.sum to
        # linear output.sum (gamma factor ~0.43 at mid-gray). L1 kernels +
        # partition-of-unity slab_w preserve Σ energy under the composite.
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
    else:
        # P1.11 R1 (I5): split gate. Diffuse ∈ [0.85, 1.15] (skip when the
        # diffuse input is ~empty, i.e. bright_thresh near 0); total ∈ [0.85, 100].
        # Fix-round: the diffuse accumulator receives (a) the FULL image on
        # is_delta slabs (I4 routes bright through the delta path too,
        # "bright + diffuse = original") and (b) only img_diffuse on
        # non-is-delta slabs. So the denominator is the weight-blended
        # effective input, not img_lin.sum() (original bug: trips gate on
        # bright content) nor img_diffuse.sum() alone (trips G6 where all
        # slabs are deltas). Uses UNPADDED sums (numerator is cropped H×W).
        diffuse_in_sum = float(img_diffuse_padded.sum())
        # P1.11 R1 fix-round 2: match the luminance-gate split used for the
        # actual bright/diffuse accumulators. Per-channel clip (old formula)
        # mismatches the new split on warm HDR content → diffuse_ratio
        # denominator is wrong, falsely trips the gate on real images.
        # P1.11 R1-PATCH: match the soft-knee bright mask used by the inner
        # loop (hard `<=` here would mismatch the primary split and trip the
        # diffuse-ratio gate on near-threshold content).
        lum_unpadded = (
            0.2126 * img_lin[..., 0]
            + 0.7152 * img_lin[..., 1]
            + 0.0722 * img_lin[..., 2]
        )
        # P1.10: same region-aware function on the unpadded site (no two-site drift).
        _bw_u = _compute_region_aware_bright_weight(
            lum_unpadded, bright_thresh,
            bright_pass_region_size, bright_pass_floor, bright_pass_norm_pct,
        )
        diffuse_mask_unpadded = (1.0 - _bw_u).astype(np.float32)[..., None]
        img_diffuse_unpadded = img_lin * diffuse_mask_unpadded
        total_weight_sum = float(weight_sum.sum()) + 1e-12
        delta_frac = float(delta_weight_sum.sum()) / total_weight_sum
        effective_diffuse_in = (
            delta_frac * float(img_lin.sum())
            + (1.0 - delta_frac) * float(img_diffuse_unpadded.sum())
        )
        diffuse_ratio = float((output - output_bright).sum()) / max(effective_diffuse_in, 1e-6)
        energy_ratio = float(output.sum()) / input_sum
        if psf_norm == "l1":
            if diffuse_in_sum > 1e-6 * float(img_padded.sum() + 1e-12):
                if not (0.85 <= diffuse_ratio <= 1.15):
                    raise RuntimeError(
                        f"Gate FAIL: diffuse energy ratio "
                        f"{diffuse_ratio:.4f} outside [0.85, 1.15]"
                    )
            if not (0.85 <= energy_ratio <= 100.0):
                raise RuntimeError(
                    f"Gate FAIL: total energy ratio "
                    f"{energy_ratio:.4f} outside [0.85, 100.0]"
                )
        else:  # psf_norm == "peak"
            # P1.7 stamp-mode: no lower bound (dim scenes legitimately lose
            # energy); upper bound is a bloom-runaway sanity ceiling. Matches
            # spec §3's own 10-500 estimate for wide-kernel peak-sum; purpose
            # is to catch runaway infinity / bad tensor loads, not police bloom.
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
                print(f"  Center SSIM: {ssim:.3f} (PASS)")
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

    if _layered_active:
        # P1.12 R2 / TL REVISION 1: α-blend as post-layered IDENTITY SNAP.
        # The α-blend was NOT truly subsumed by layered gather — it's
        # load-bearing for in-focus sharpness (G12 G_FACE_FALLOFF). At
        # dmap≈0 (face pixels), smoothstep POU spreads weight across ~4
        # non-identity bins, producing ~5× face softening vs legacy. This
        # snap restores sharpness at the focal plane while preserving the
        # layered composite everywhere else. Net mathematical effect:
        # in-focus (α=1) → pure img_lin (Portrait-Mode-differentiating
        # sharpness); out-of-focus (α=0) → pure layered composite
        # (organic bokeh). `output` at this point already = diffuse +
        # bright (line ~1002), so snap applies to their sum, which is
        # equivalent to snapping diffuse and zeroing bright at in-focus.
        alpha = (np.cos(0.5 * np.pi * np.minimum(dmap / D_KNEE_LAYERED, 1.0)) ** 2).astype(np.float32)
        output = alpha[:, :, None] * img_lin + (1.0 - alpha[:, :, None]) * output
        if np.any(np.isnan(output)):
            raise RuntimeError("Gate FAIL: NaN in layered output")
        print(f"  α-blend: D_KNEE_LAYERED={D_KNEE_LAYERED}, alpha_mean={float(alpha.mean()):.3f} (layered identity snap)")
    else:
        alpha = (np.cos(0.5 * np.pi * np.minimum(dmap / D_KNEE, 1.0)) ** 2).astype(np.float32)
        output = alpha[:, :, None] * img_lin + (1.0 - alpha[:, :, None]) * output
        if np.any(np.isnan(output)):
            raise RuntimeError("Gate FAIL: NaN in post-blend output")
        print(f"  α-blend: D_KNEE={D_KNEE}, alpha_mean={float(alpha.mean()):.3f}")

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
                  downres: int | None = None) -> None:
    """Save rendered image as JPEG (q95) and PNG."""
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


# ---------------------------------------------------------------------------
# CLI — argparse builder + thin shim + lifted render pipeline (P1.audit Item 3)
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Construct the convolve.py CLI parser.

    Lifted out of ``main()`` in P1.audit Item 3 so the GUI / tests can
    introspect flags without exec-side effects.
    """
    parser = argparse.ArgumentParser(
        description="Bin-and-convolve renderer: apply 4D PSF tensor to an image"
    )
    parser.add_argument("--image", required=True, help="Input image path (HEIC/JPEG/PNG)")
    parser.add_argument(
        "--tensor",
        default=None,
        help="4D tensor .npz path. When omitted, defaults to the wheel-shipped "
             "canonical tensor (summicron.render._artifacts.default_tensor_path()).",
    )
    parser.add_argument(
        "--out-dir",
        default=str(Path(__file__).resolve().parents[2] / "runs" / "render"),
        help="Output directory",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Explicit output JPG path. Overrides --out-dir/auto-naming when set.",
    )
    parser.add_argument(
        "--depth-scale",
        type=float,
        default=1.0,
        help="Multiplier for depth_m array before render (default 1.0; only applies with --depth-source=lidar)",
    )
    parser.add_argument(
        "--depth-source",
        choices=["lidar", "monocular"],
        default="lidar",
        help="Depth backend; 'lidar' decodes HEIC depth_images[0], "
             "'monocular' runs Depth Anything V2 (downloads ~100 MB DA2 "
             "model on first invocation; requires the [depth] extra).",
    )
    parser.add_argument(
        "--focus-xy",
        nargs=2,
        type=float,
        default=None,
        metavar=("X", "Y"),
        help="Normalized image coords [0,1] (axis: x=col, y=row) for tap-to-focus. "
             "Omit to use closest-subject (p2 of depth_m).",
    )
    parser.add_argument(
        "--focus-target-m",
        type=float,
        default=None,
        help="Tensor focus distance in meters (default: from tensor's focus_distance_m, else 0.7).",
    )
    parser.add_argument(
        "--dof-m",
        type=float,
        default=0.0,
        help="DoF band half-width in meters (Item C). 0 = off.",
    )
    parser.add_argument(
        "--downres",
        type=int,
        default=None,
        help="Downsample so the long edge equals this value (e.g. 2048). "
             "Never upscales. Affects output filename suffix (_r<value>). "
             "P1.10.4: replaces former `--working-long-edge` (collapsed; behavior "
             "preserved via downres_array's HDR-aware scipy bilinear path).",
    )
    parser.add_argument(
        "--psf-norm",
        choices=["l1", "peak"],
        default="l1",
        help="PSF normalization mode. 'l1' (default) keeps kernels L1-normed "
             "and averages via post-conv weight-sum divide (bit-identical to "
             "pre-P1.7 render path). 'peak' peak-normalizes kernels (stamp "
             "mode): bright pixels bloom by sum(kernel) and overflow clips at "
             "finalize. See doc/plan/phase1/section7/locked-spec-stamp-mode.md.",
    )
    # P1.10.4 C5: --layered-gather / --no-layered-gather flag removed.
    # Layered gather is the SOLE rendering path. The render() kwarg
    # survives as a silently-accepted noop (default True) for ABI.
    parser.add_argument(
        "--bright-pass-thresh",
        type=float,
        default=1.0,
        help="Dual-norm bright-pass threshold in linear luminance (P1.11 R1). "
             "Pixels above T convolve with a peak-renormed kernel (bloom "
             "gain); pixels below T use the existing L1 path. Default 1.0 "
             "targets HDR highlights. Pass 1e9 to disable (backward-compat "
             "— bit-identical to pre-R1). See "
             "doc/plan/phase1/section11/locked-spec-R1-brightpass.md.",
    )
    parser.add_argument(
        "--bright-pass-region-size",
        type=int,
        default=31,
        help="P1.10: morphological top-hat structuring-element size for "
             "region-aware bright-pass. Default 31 since P1.10.2 (was 0 "
             "pre-P1.10.2; pass 0 to disable, bit-identical to P1.11 "
             "R1-PATCH). Recommended: 15 (small lights only — string"
             "lights, candle), 31 (canonical), 63 (large lights). Detector "
             "fires on pixels with high local-contrast (true highlights); "
             "flat-bright midtones (face, white shirt) get suppressed.",
    )
    parser.add_argument(
        "--bright-pass-floor",
        type=float,
        default=0.10,
        help="P1.10: floor multiplier on the region-detector signal. 0.0 = "
             "strict (only true local-contrast highlights bloom). 1.0 = no "
             "region gate. Default 0.10 preserves a small bloom for "
             "large-uniform-bright safety net (LED panel, sun-disc filling "
             "frame). Only consulted when --bright-pass-region-size > 0.",
    )
    parser.add_argument(
        "--bright-pass-norm-pct",
        type=float,
        default=99.0,
        help="P1.10: percentile of detector values used as scene-scale "
             "anchor. Default 99.0 (top 1%% of pixels define bloom ceiling). "
             "Range [50.0, 100.0]. Only consulted when --bright-pass-region-size > 0.",
    )
    # P1.10.4 C4b: --radial-blend-mode flag deleted. Basis is the sole
    # radial-blend kernel; the `radial_blend_mode` kwarg survives on
    # render() as a silently-accepted noop (default "basis") to preserve
    # call-site ABI for tests not enumerated in spec § 6. See
    # doc/plan/phase1/section18/locked-spec-P1_10_4.md § 9 C4b.
    parser.add_argument(
        "--hdr-source",
        choices=["none", "macos"],
        default="none",
        help="HDR input path. 'none' (default) = LDR sRGB via pillow/HEIF "
             "(bit-identical to pre-P1.8 render). 'macos' = extended-linear "
             "float32 via Core Image expandToHDR, then Apple's Core Image "
             "CIToneMapHeadroom HDR→SDR step after render() (P1.10.2 retired "
             "the prior Reinhard-tonemap stage). Requires macOS + the "
             "[hdr-macos] extra. See doc/plan/phase1/section8/"
             "locked-spec-hdr-access.md.",
    )
    # ---- P1.13 K11 LDR-anchored per-channel shadow match (locked-spec § 2)
    parser.add_argument(
        "--shadow-match-mode",
        choices=["off", "on"],
        default="off",
        help="P1.13 K11 LDR-anchored per-channel shadow match. PRE-tonemap "
             "only (the post-tonemap variant from K10 was rejected -- killed "
             "dynamic range). When 'on': preview-tonemap to derive per-channel "
             "shadow ratios in display-linear space, then apply on linear-HDR "
             "buffer pre-final-tonemap (full headroom). 'off' (default) = "
             "bit-identical pre-K11. Requires --hdr-source != none.",
    )
    parser.add_argument(
        "--shadow-band-hi",
        type=float,
        default=0.70,
        help="P1.13 K11 LDR-luma cutoff defining 'shadow' band for "
             "--shadow-match-mode. Default 0.70 (covers shadow + mid; "
             "G11-confirmed on Demo_input as the right coverage to fix "
             "render's mid-band warm cast). 0.30 = shadow only (corrects "
             "shadow blue cast but leaves mid uncorrected). Range "
             "[0.0, 1.0]. Inert when --shadow-match-mode=off.",
    )
    parser.add_argument(
        "--shadow-match-max-gain",
        type=float,
        default=4.0,
        help="P1.13 K11 upper clamp on per-channel gain. Prevents pathological "
             "over-lift on degenerate scenes. Default 4.0. Range [1.0, 16.0]. "
             "Inert when --shadow-match-mode=off.",
    )
    parser.add_argument(
        "--shadow-match-strength",
        type=float,
        default=1.0,
        help="P1.13 K11 strength multiplier on auto-derived per-channel "
             "shadow gain. Default 1.0 (exact LDR match; 1.2 was tried and "
             "deemed too aggressive on G11 review). >1.0 overshoots; <1.0 "
             "undershoots. Multiplied BEFORE clamp. Range [0.0, 4.0]. "
             "Inert when --shadow-match-mode=off.",
    )
    parser.add_argument(
        "--slab-batch-size",
        type=int,
        default=None,
        help="R-perf-1 I1: depth-layered gather slab-batching size. The "
             "render() slab loop processes D depth slabs in serial batches "
             "of this size, freeing the per-batch working set at each batch "
             "boundary. Default (None → 4) tuned for 18 GB Macs at 2048² "
             "downres; smaller values reduce peak RSS at a small wall-clock "
             "cost; larger values trade RAM for fewer GC pumps. Bit-exact "
             "to the pre-batching path (uint8 max-abs-diff ≤ 1; verified by "
             "256² synthetic equivalence proof under tests/unit).",
    )
    return parser


def _validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Argparse-time validation lifted from the original ``main()`` body.

    Calls ``parser.error()`` (exit 2) on any out-of-range flag so the CLI
    surface stays unchanged. The GUI/in-process callers should construct
    ``RenderOptions`` directly and skip this — the validators downstream
    in :func:`render` raise ``ValueError`` on the same invariants.
    """
    if args.focus_xy is not None:
        x, y = args.focus_xy
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            parser.error("--focus-xy values must be in [0, 1]")
    if args.focus_target_m is not None and args.focus_target_m <= 0:
        parser.error("--focus-target-m must be > 0")
    if args.dof_m < 0:
        parser.error("--dof-m must be >= 0")
    # P1.13 K11 range gates
    if not (0.0 <= args.shadow_band_hi <= 1.0):
        parser.error("--shadow-band-hi must be in [0.0, 1.0]")
    if not (1.0 <= args.shadow_match_max_gain <= 16.0):
        parser.error("--shadow-match-max-gain must be in [1.0, 16.0]")
    if not (0.0 <= args.shadow_match_strength <= 4.0):
        parser.error("--shadow-match-strength must be in [0.0, 4.0]")
    # R-perf-1 I1: --slab-batch-size must be a positive integer when set.
    if getattr(args, "slab_batch_size", None) is not None:
        if int(args.slab_batch_size) < 1:
            parser.error("--slab-batch-size must be >= 1")
    # P1.audit Item 8: macOS-HDR preflight. Surface platform mismatch BEFORE
    # the deferred Quartz/CoreImage import fires inside the renderer; placed
    # ahead of the shadow-match-mode validator so platform issue surfaces
    # first if both apply.
    if args.hdr_source == "macos" and sys.platform != "darwin":
        parser.error(
            f"--hdr-source macos requires macOS; got {sys.platform}. "
            f"Use --hdr-source none on this platform."
        )
    # F3: early-error guard (was at L1953 post-render — wasted minutes
    # of compute before erroring).
    if args.shadow_match_mode == "on" and args.hdr_source == "none":
        parser.error(
            "--shadow-match-mode on requires --hdr-source != none "
            "(needs HDR linear buffer + CI preview pass)."
        )


def _run_render_pipeline(
    opts,
    *,
    on_progress: _OnProgress = None,
    cancel: threading.Event | None = None,
):
    """Execute the v0 render pipeline against a :class:`RenderOptions`.

    P1.audit Item 3: lifted from the original ``main()`` body as a pure
    refactor. ``on_progress=None`` (CLI default) routes status emits
    through ``print()`` so stdout is byte-identical to the pre-refactor
    path; the canonical bench md5 ``38c75dd2…`` is the gate. Returns a
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
        Path(__file__).resolve().parents[2] / "runs" / "render"
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
        slab_batch_size=getattr(opts, "slab_batch_size", None),
        on_progress=on_progress,
        cancel=cancel,
    )

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
    if opts.out_path is not None:
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
        save_results(
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
        center_ssim=None,
        bright_pixels_pct=0.0,
        alpha_mean=0.0,
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
    _run_render_pipeline(opts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
