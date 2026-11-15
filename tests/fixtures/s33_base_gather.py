"""FROZEN oracle for the S33 Tier A (byte-exact) speed round.

VERBATIM copy of the base hot-path functions and module constants taken from
``git show 2a741a8:summicron/render/convolve.py`` (main at the cut of the speed work). The function bodies below are pasted unchanged, extracted
with ``ast`` by node span (``_emit``, ``_make_delta``, ``_bright_stamp``,
``_fft_geometry``, ``_radial_basis_weights``, ``compute_slab_weights``,
``_build_basis_kernel``, ``_build_image_fft_state``, ``_fftconvolve_hoisted``,
``_apply_kernel``, ``_scatter_one_slab_fixed_d_basis``,
``_layered_gather_diffuse`` and the constants ``_MAX_ACTIVE_SLABS``,
``_EMPTY_SKIP_THRESH``, ``_SLAB_BATCH_SIZE``, ``N_ANG``). Only this docstring
and the import block are new.

Purpose: tests/adversarial/test_s33_exact_adversarial.py asserts that the
current ``summicron.render.convolve`` gather is ``tobytes``-equal to this
frozen copy. Do NOT edit the bodies to follow later changes in convolve.py:
the whole point is that this file does not move. It needs no ``runs/``
goldens and no other checkout, so it runs on main and in CI.
"""
# ruff: noqa
from __future__ import annotations

import gc
import warnings

import numpy as np
import scipy.fft as sp_fft
from scipy.ndimage import rotate as ndimage_rotate
from scipy.ndimage import zoom as ndimage_zoom
from scipy.signal import fftconvolve

_MAX_ACTIVE_SLABS = 8
_EMPTY_SKIP_THRESH = 0.01
_SLAB_BATCH_SIZE = 1
N_ANG = 8



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


def _make_delta(size: int) -> np.ndarray:
    """Exact single-pixel delta at the geometric centre (``size // 2``),
    float32 (renderer's odd-grid convention)."""
    k = np.zeros((size, size), dtype=np.float32)
    k[size // 2, size // 2] = 1.0
    return k


def _bright_stamp(rotated, gain):
    """Bright-pass stamp kernel. ``gain is None`` = peak stamp (today's
    arithmetic, verbatim: the divide, NOT a reciprocal multiply). A float
    ``gain`` = L1 stamp: the L1-normalised kernel times ``gain`` (energy
    conserving; ``rotated`` is L1 = 1)."""
    if gain is None:
        return rotated / max(float(rotated.max()), 1e-10)
    return rotated * np.float32(gain)


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


def _build_basis_kernel(
    ri, d_center_or_idx, use_idx, defocus_ratios, kernels,
    kH, kW, pix_scale, scaled_ksize,
):
    """Build the (kH, kW) raw-tensor kernel for a single radial-basis index ri.

    Input is a fixed `ri` (basis index into the raw R=12 tensor), output is
    one kernel; the per-pixel blend happens later inside the basis scatter
    functions (`_scatter_one_slab_fixed_d_basis`)
    via `_radial_basis_weights[ri, y, x]`.

    D-axis stance: D-axis remains 2-tap bilinear at the same scalar
    `d_center` as legacy. Per-pixel D-axis blend is OUT OF SCOPE for P1.10.1
    (preserves the R-cubic / D-linear interp scheme). Layered path pins one `d_idx` per
    slab via `use_idx=True`.

    Tail (pix_scale zoom + L1 renorm) preserves per-kernel L1
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

    # --- Tail: zoom + L1 renorm. ---
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
            kernel_base /= ksum
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
    wedge_idx, pad, H, W,
    output, output_bright, weight_sum, delta_weight_sum,
    *,
    fft_state: dict | None = None,
    bright_stamp_gain: float | None = None,
):
    """Apply kernel_base via the is_delta/angular-scatter body.

    S22 S1: ``bright_stamp_gain`` selects the bright-pass stamp (None = peak
    stamp, today's arithmetic; a float = L1 stamp ``rotated * gain``).

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
                rotated /= rotated.sum() + 1e-12
                rotated_bright = _bright_stamp(rotated, bright_stamp_gain)
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
            rotated /= rotated.sum() + 1e-12
            rotated_bright = _bright_stamp(rotated, bright_stamp_gain)
            # rotated_bright differs from rotated (stamp), so two rfft2s.
            ker_diffuse_F = sp_fft.rfft2(rotated, s=fshape)
            ker_bright_F = sp_fft.rfft2(rotated_bright, s=fshape)
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
    kH, kW, pad, pix_scale, scaled_ksize,
    H, W,
    output, output_bright, weight_sum, delta_weight_sum,
    cancel: threading.Event | None = None,
    *,
    fft_state: dict | None = None,
    bright_stamp_gain: float | None = None,
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
            kH, kW, pix_scale, scaled_ksize,
        )
        _apply_kernel(
            kernel_base, w, img_full_pad, img_diffuse_pad, img_bright_pad,
            wedge_idx, pad, H, W,
            output, output_bright, weight_sum, delta_weight_sum,
            fft_state=fft_state, bright_stamp_gain=bright_stamp_gain,
        )


def _layered_gather_diffuse(
    *,
    depth_m, slab_w, img_lin,
    img_padded, img_diffuse_padded, img_bright_padded,
    kernels, defocus_ratios, radial_frac_map, r_centers, wedge_idx,
    kH, kW, pad, pix_scale, scaled_ksize, H, W,
    fft_state, slab_batch_size, on_progress, cancel,
    emit_log=True,
    bright_stamp_gain=None,
    active_slab_cap=None,
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

    effective_cap = (
        max(_MAX_ACTIVE_SLABS, D_gate // 2) if active_slab_cap is None
        else int(active_slab_cap)
    )
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
                    scaled_ksize=scaled_ksize,
                    H=H, W=W,
                    output=out_d, output_bright=out_b_d,
                    weight_sum=wsum_d, delta_weight_sum=delta_wsum_d,
                    cancel=cancel,
                    fft_state=fft_state,
                    bright_stamp_gain=bright_stamp_gain,
                )

            # Per-slab normalize by spatial-slab weight_sum (radial partition).
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
