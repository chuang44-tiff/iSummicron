"""R-occlusion-layering — premultiplied-alpha layered OVER compositor.

This module owns the ENTIRE ``--occlusion-compositing on`` branch. The
canonical ``convolve.py`` slab-fold and ``_apply_kernel`` stay byte-untouched,
so the OFF path (and therefore the G1 canonical-bench md5) holds *by
construction*: the ON branch is reached only via a single
``if occlusion_compositing == "on":`` dispatch in ``render()``.

Fix target
----------
The occlusion-boundary **milky halo** at a foreground/background depth
discontinuity (canonical: IMG_4343 arm vs blurred background). In the
gather renderer the whole image is convolved per slab then weighted AFTER,
so background luminance is *averaged into* foreground-edge pixels and no
post-weighting can subtract it back out. The premultiplied-alpha layered
formulation makes background "show through" the silhouette by exactly
``(1 − α_blur)`` — correct partial coverage, not contamination.

Three-regime ∇depth-gated matte (research §1 Stage A-2 + §4 reconciliation)
---------------------------------------------------------------------------
A continuous selector routes each pixel between two composites so the round
fixes the halo WITHOUT re-introducing the slab-banding P1.12's PoU
smoothstep was built to prevent:

* **Selector** ``s = smoothstep(|∇depth| / (k · CoC_local))`` — ``scipy.ndimage``
  Sobel magnitude on ``depth_m``, CoC-scaled, continuous (no inter-regime
  seam).
* **Regime (a)** ``s ≈ 0`` (smooth depth): the existing PoU ``slab_w``
  linear-weighted-sum — reproduced here with the SAME per-slab scatter +
  ``wsum_d`` renorm + ``output += out_d * slab_w[d]`` fold as the OFF diffuse
  path. No banding on tilted-floor / cheek-ear ramps, no halo risk.
* **Regime (b)** ``s ≈ 1`` (true silhouette): hard nearest-slab interior matte
  ``H_d = (argmax_k slab_w[k] == d)`` (each pixel opaque in exactly ONE slab →
  solids get ``acc_α = 1``, no bleed-through) + guided-feather boundary band
  (dilate ~2·CoC) → premultiplied OVER, folded back-to-front.
* **Final diffuse composite** ``out_diffuse = s·[OVER] + (1−s)·[PoU-linear]``.

Premultiplied path = the decomposition. ``C'_d = H_d · img_diffuse`` is a
genuinely isolated layer; the matte ``H_d`` is convolved with the SAME slab
PSF as the color (blurring the matte → soft edges). Un-premultiply is
``(C'_d ⊛ K) / (H_d ⊛ K)`` with the Kosloff-Barsky α-bias clamp
``acc_c / max(acc_a, 1e-3)`` to avoid speckle where blurred α → 0.

The per-slab ``wsum_d`` radial-partition renorm CANCELS in the un-premultiply
(``weight_sum`` depends only on ``(d_idx, basis_w, wedge_idx)``, NOT the
convolved buffer → ``wsum_C == wsum_M``), so it commutes with the coverage
divide — no double-normalization, the P1.10.1 radial-basis baseline is
untouched.

``output_bright`` is assembled EXACTLY as the OFF path in BOTH regimes (the
peak-stamp at ``convolve.py:1096/:1141`` is preserved — no bright-path edit
this round). The §3 A×B bright-disc-on-silhouette bleed is a documented
R-highlight-guard carry-forward.

Deferred → R-highlight-guard
----------------------------
The snowflake / energy-honest-bright fix (peak-stamp kill, global L1 bright
buffer, sub-PSF spike scrub, π·r_min² area gate). ``_layered_composite`` runs
the diffuse decomposition BEFORE the ``:2028`` bright recombine, so
R-highlight-guard can insert the global L1 bright buffer + per-layer scrub
without re-plumbing the composite. ``r_min`` MUST be plumbed from the
*pre-delta* focus PSF at ``codev_to_render.py:145`` — NOT ``kernels[:, 0]``
(the S2 delta-guard trap). See the ``r_min`` plumbing stub below.
"""
from __future__ import annotations

import threading

import numpy as np
import scipy.fft as sp_fft
from scipy import ndimage

# ---------------------------------------------------------------------------
# Selector tuning (hardcoded — NO new user knob this round, per locked spec
# Item 2). `_SELECTOR_K` scales the local CoC: the regime crossover sits at
# `|∇depth| ≈ k · CoC_local`. k = 1.0 means "one CoC-radius of depth change
# per pixel reads as a true silhouette"; below that the gradient is treated as
# a smooth ramp (regime a). 1.0 is the documented default — a depth step that
# spreads over more than a blur-disc width is a ramp, a step sharper than the
# blur disc is an occlusion edge. Exposed only as this module constant.
# ---------------------------------------------------------------------------
_SELECTOR_K = 1.0

# Floor on the local CoC (in defocus-ratio units → pixels via _COC_PX_SCALE)
# so the selector denominator never collapses to 0 in perfectly-in-focus
# regions (where it would make every micro-gradient read as a silhouette).
_COC_FLOOR_PX = 1.0

# Maps a defocus-ratio magnitude [0, 1] to an approximate CoC radius in
# pixels for the selector + boundary-band dilation. The kernel half-width
# (`pad`) is the largest CoC the tensor renders, so CoC_px ≈ defocus · pad.
# This is a selector-scale heuristic only (not an energy path) — its only job
# is to set the regime crossover and dilation width sensibly.
def _coc_px(defocus_map: np.ndarray, pad: int) -> np.ndarray:
    """Approximate per-pixel circle-of-confusion radius in pixels."""
    coc = np.abs(defocus_map).astype(np.float32) * float(max(pad, 1))
    return np.maximum(coc, np.float32(_COC_FLOOR_PX))


def _smoothstep(x: np.ndarray) -> np.ndarray:
    """Cubic smoothstep ``3t² − 2t³`` on a value already mapped to [0, 1]."""
    t = np.clip(x, 0.0, 1.0).astype(np.float32)
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32)


def compute_edge_selector(
    depth_m: np.ndarray,
    defocus_map: np.ndarray,
    pad: int,
) -> np.ndarray:
    """Continuous ∇depth selector ``s ∈ [0, 1]`` (Item 2).

    ``s = smoothstep(|∇depth| / (k · CoC_local))``. ``|∇depth|`` is the
    ``scipy.ndimage.sobel`` gradient magnitude on ``depth_m`` (meters);
    ``CoC_local`` is the per-pixel CoC in pixels (``_coc_px``); ``k`` is the
    hardcoded ``_SELECTOR_K``. Returns ``(H, W)`` float32. ``s ≈ 0`` →
    regime-a (PoU diffuse, OFF-verbatim); ``s ≈ 1`` → regime-b (premultiplied
    OVER silhouette).

    The normalization makes the crossover depth-scale-aware: a step counts as
    a silhouette only when the depth changes by more than ``k`` CoC-radii
    worth of meters across the disc. We approximate "meters per CoC-pixel" via
    the median in-frame depth so the unitless ratio is stable across scenes.
    """
    depth = np.asarray(depth_m, dtype=np.float32)
    # R-occ-on-fixes F6: +/-Inf (sensor "infinity") would make Sobel emit NaN
    # and trip the NaN gate. Clamp +/-Inf only to the map's finite max/min;
    # NaN is left alone so it keeps failing loudly. No-op on finite input.
    if np.isinf(depth).any():
        fin = depth[np.isfinite(depth)]
        if fin.size:
            depth = np.where(
                np.isposinf(depth), np.float32(fin.max()),
                np.where(np.isneginf(depth), np.float32(fin.min()), depth),
            ).astype(np.float32)
    gx = ndimage.sobel(depth, axis=1, mode="reflect")
    gy = ndimage.sobel(depth, axis=0, mode="reflect")
    grad_mag = np.hypot(gx, gy).astype(np.float32)  # meters per (Sobel) pixel

    coc = _coc_px(defocus_map, pad)  # pixels
    # meters-per-pixel reference: median absolute depth keeps the ratio
    # scene-relative (a 0.1 m step on a 0.5 m subject is a silhouette; the
    # same 0.1 m step on a 20 m background is noise). Guard against an
    # all-zero depth map.
    depth_ref = float(np.median(np.abs(depth)))
    if depth_ref < 1e-6:
        depth_ref = 1.0
    # Normalized gradient: |∇depth| / depth_ref gives a unitless per-pixel
    # fractional depth change; divide by (k · CoC) so a sharper-than-disc step
    # saturates the smoothstep.
    denom = np.maximum(_SELECTOR_K * coc, np.float32(1e-6))
    ratio = (grad_mag / depth_ref) / denom
    return _smoothstep(ratio)


def _disocclusion_fill(premult_color: np.ndarray, matte: np.ndarray) -> np.ndarray:
    """EDT nearest-valid fill of the ``matte == 0`` hole (Item 7).

    Where a layer's matte is 0 (the pixel belongs to a nearer occluder), the
    premultiplied color is a black hole; convolving it would pull that black
    into the blurred disc at the silhouette (trading a milky halo for a dark
    one). ``scipy.ndimage.distance_transform_edt(return_indices=True)`` gathers
    the nearest *valid* (matte > 0) color into the hole BEFORE convolution.

    Kraus-Strengert pyramid push-pull is the named follow-up if the nearest
    fill streaks under large-CoC backgrounds (research §4 stage 5 / §5 risk 7).
    """
    valid = matte > 1e-6
    if valid.all() or not valid.any():
        # No holes (or nothing valid to pull from) — return premult as-is.
        return premult_color
    # Indices of the nearest valid pixel for every position.
    _, (iy, ix) = ndimage.distance_transform_edt(
        ~valid, return_indices=True
    )
    filled = premult_color[iy, ix, :]
    out = premult_color.copy()
    holes = ~valid
    out[holes, :] = filled[holes, :]
    return out


def _over_scatter_slab(
    *, d_idx, premult_pad, matte_pad, basis_w, wedge_idx,
    kernels, defocus_ratios, kH, kW, pad, pix_scale, scaled_ksize,
    psf_norm, H, W, fft_state, conv_c, conv_a, cancel,
):
    """Q-e: fused colour+matte OVER scatter for ONE slab (the only OVER scatter).

    Bit-identical (``ON_MD5``) to the former pair of
    ``convolve._scatter_one_slab_fixed_d_basis(..., fft_state=None)`` calls
    (colour pass, then the 3-channel-replicated matte pass read at ``[:, :, 0]``)
    with the discarded bright / ``wsum`` outputs removed. It MIRRORS the
    ``fft_state is None`` branch of ``convolve._apply_kernel`` minus the bright
    pass: same ``_build_basis_kernel`` call, same ``theta_a``/``rot_deg``,
    ``ndimage_rotate(order=1)``, ``np.maximum(., 0)`` and ``psf_norm`` renorm,
    same ``ri``-outer / ``a``-inner loop and skip predicates.

    DRIFT WARNING: any change to that branch's kernel build, rotate or
    normalise MUST be mirrored here by hand. The oracle test
    (``tests/unit/test_over_scatter_fused.py``) checks bit-identity against a
    FROZEN copy of the pre-Q-e two-pass code, so drift in ``_apply_kernel``
    (the OFF path) is NOT caught by it.

    One image spectrum per slab (3 colour channels + matte) is shared by every
    kernel spectrum: 5 FFTs per (ri, a) unit instead of 36. ``fshape`` /
    ``slice_y`` / ``slice_x`` come from the render-level ``fft_state`` built
    from the padded image, so the FFT geometry is exactly what
    ``fftconvolve(mode="same")`` chose.

    Accumulates into ``conv_c`` (H, W, 3) and the 2-D ``conv_a`` (H, W).
    Weights are applied in the spatial domain only; slabs are folded by the
    caller sequentially near->far.
    """
    from scipy.ndimage import rotate as ndimage_rotate

    from summicron.render.api import CancelledError
    from summicron.render.convolve import N_ANG, _build_basis_kernel

    fshape = fft_state["fshape"]
    sy = fft_state["slice_y"]
    sx = fft_state["slice_x"]
    # Geometry guard: fshape/slices were derived from img_padded; the OVER
    # buffers must be the same padded shape or the crop below is wrong.
    ref_shape = (H + 2 * pad, W + 2 * pad)
    if not (
        premult_pad.shape[:2] == matte_pad.shape == ref_shape
        and (sy.stop - sy.start, sx.stop - sx.start) == ref_shape
    ):
        raise AssertionError(
            "OVER scatter geometry mismatch: premult_pad "
            f"{premult_pad.shape}, matte_pad {matte_pad.shape}, expected "
            f"{ref_shape} matching fft_state slices"
        )

    # dtype guard: the complex64 ``out=`` buffer below would silently downcast
    # float64 spectra; production inputs are float32.
    if premult_pad.dtype != np.float32 or matte_pad.dtype != np.float32:
        raise TypeError(
            "OVER scatter requires float32 premult_pad/matte_pad, got "
            f"{premult_pad.dtype}/{matte_pad.dtype}"
        )

    # Cancel before any FFT work (per-ri polls below cover the rest).
    if cancel is not None and cancel.is_set():
        raise CancelledError(f"layered_composite OVER cancelled in slab {d_idx}")

    prem_F = [sp_fft.rfft2(premult_pad[:, :, c], s=fshape) for c in range(3)]
    matte_F = sp_fft.rfft2(matte_pad, s=fshape)
    buf = np.empty((fshape[0], fshape[1] // 2 + 1), dtype=np.complex64)
    crop = (slice(pad, pad + H), slice(pad, pad + W))

    def _acc(ker_F, w):
        # ON_MD5 (SK1): image spectrum is the explicit FIRST operand and
        # ker_F a NAMED local, product written into a reused buffer. complex64
        # multiply is not bitwise commutative (K*F != fftconvolve) and numpy
        # temporary elision on an inline operand moves bits; np.multiply(...,
        # out=buf) removes both hazards on every interpreter version.
        for c in range(3):
            np.multiply(prem_F[c], ker_F, out=buf)
            full = sp_fft.irfft2(buf, s=fshape)
            conv_c[:, :, c] += w * full[sy, sx][crop].astype(np.float32, copy=False)
        np.multiply(matte_F, ker_F, out=buf)
        full = sp_fft.irfft2(buf, s=fshape)
        conv_a[...] += w * full[sy, sx][crop].astype(np.float32, copy=False)

    Nr = basis_w.shape[0]
    for ri in range(Nr):
        if cancel is not None and cancel.is_set():
            raise CancelledError(
                f"layered_composite OVER cancelled in slab {d_idx}, "
                f"radial bin {ri}/{Nr}"
            )
        w = basis_w[ri].astype(np.float32)
        if w.max() < 0.01:
            continue
        kernel_base = _build_basis_kernel(
            ri, d_idx, True, defocus_ratios, kernels,
            kH, kW, psf_norm, pix_scale, scaled_ksize,
        )
        if kernel_base.max() > 0.5 * kernel_base.sum():  # is_delta
            if w.max() < 0.01:
                continue
            # ON_MD5: named local, never an inline rfft2 operand.
            ker_F = sp_fft.rfft2(kernel_base, s=fshape)
            _acc(ker_F, w)
            continue
        for a in range(N_ANG):
            w_ang = np.where(wedge_idx == a, w, np.float32(0)).astype(
                np.float32, copy=False
            )
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
            # ON_MD5: named local, never an inline rfft2 operand.
            ker_F = sp_fft.rfft2(rotated, s=fshape)
            _acc(ker_F, w_ang)


def _layered_composite(
    *,
    depth_m: np.ndarray,
    slab_w: np.ndarray,
    object_distances_m: np.ndarray,
    defocus_map: np.ndarray,
    img_lin: np.ndarray,
    img_padded: np.ndarray,
    img_diffuse_padded: np.ndarray,
    img_bright_padded: np.ndarray,
    kernels: np.ndarray,
    defocus_ratios: np.ndarray,
    radial_frac_map: np.ndarray,
    r_centers: np.ndarray,
    wedge_idx: np.ndarray,
    kH: int,
    kW: int,
    pad: int,
    pix_scale: float,
    scaled_ksize: int,
    psf_norm: str,
    H: int,
    W: int,
    fft_state: dict | None,
    slab_batch_size: int | None = None,
    cancel: threading.Event | None = None,
    on_progress=None,
    bright_stamp_gain: float | None = None,
    active_slab_cap: int | None = None,
):
    """Layered occlusion-aware diffuse compositor — owns the ON branch.

    Returns ``(output_diffuse, output_bright, weight_sum, delta_weight_sum,
    acc_alpha)``. round-2: ``output_diffuse = s·OVER + (1−s)·OFF_diffuse``
    where ``OFF_diffuse`` is the extracted OFF ``_layered_gather_diffuse``
    output (regime-a == OFF verbatim by reuse) and ``output_bright`` is
    assembled EXACTLY as the OFF path (no bright-path
    edit). The caller (``render()``) recombines ``output = output_diffuse +
    output_bright`` and runs the unchanged α-identity-snap + finalize, so the
    layered diffuse lands in the same linear space as OFF diffuse.

    r_min / S1 plumbing stub (deferred → R-highlight-guard)
    -------------------------------------------------------
    R-highlight-guard will insert the global energy-honest L1 bright buffer +
    sub-PSF spike scrub HERE, between the diffuse decomposition and the
    caller's bright recombine. ``r_min`` (the in-focus PSF support radius used
    by the π·r_min² area gate) MUST be read from the *pre-delta* focus PSF at
    ``codev_to_render.py:145`` — NOT ``kernels[:, 0]`` (the S2 delta-guard
    trap zeroes the focus slice to a single-pixel delta). The hook point is
    the ``output_bright`` assembly below; it is intentionally left
    OFF-verbatim this round so R-highlight-guard owns the bright path.
    """
    from summicron.render.convolve import (
        _layered_gather_diffuse,
        _radial_basis_weights,
        _EMPTY_SKIP_THRESH,
    )

    D = slab_w.shape[0]

    # --- regime-a == OFF VERBATIM (probe-round-1 PRIMARY FIX) ---------------
    # Round-1 reimplemented only the bare PoU sum and silently omitted the
    # layered_gather holes-fill / identity-snap machinery, so the ON path
    # re-rendered the whole diffuse image differently from OFF even where the
    # selector s≈0 (99.2% of IMG_4343 pixels changed). The fix: call the SAME
    # extracted helper the OFF path calls, so regime-a is literally the OFF
    # code (not a copy). Where s=0, ON diffuse == OFF diffuse to float tol.
    # `output_bright` here is the OFF-verbatim bright buffer (no bright-path
    # edit this round). `emit_log=False` keeps the ON stdout to its own
    # occlusion line (the OFF layered_gather log is suppressed).
    (
        off_diffuse, output_bright, weight_sum, delta_weight_sum, acc_alpha
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
        emit_log=False, bright_stamp_gain=bright_stamp_gain,
        active_slab_cap=active_slab_cap,
    )

    # R-native-mem item 3: drop the 9 image spectra (~1 GB at native) for the
    # whole OVER phase. ``_over_scatter_slab`` reads only ``fshape`` and
    # ``slice_y/slice_x`` from ``fft_state`` (:210-212) and builds its own
    # per-slab spectra; ``render()`` never reuses ``fft_state`` after the
    # dispatch (no ``fft_state`` reference in convolve.py after the call).
    if fft_state is not None:
        for _k in ("img_F", "img_diffuse_F", "img_bright_F"):
            fft_state.pop(_k, None)

    # --- selector (Item 2) + regime-b cost gate -----------------------------
    # Mechanism cost (locked spec §"Mechanism cost"): the layered two-pass OVER
    # scatter runs ONLY on regime-(b) pixels (thin silhouette band, <5% of
    # frame). If no pixel is in regime-b (s≈0 everywhere — no occlusion edge)
    # skip the OVER passes entirely and return the OFF diffuse unchanged.
    s_full = compute_edge_selector(depth_m, defocus_map, pad)  # (H, W)
    s_active = s_full > 1e-4
    any_regime_b = bool(s_active.any())
    if not any_regime_b:
        # Pure regime-a: ON diffuse is exactly the OFF diffuse.
        return off_diffuse, output_bright, weight_sum, delta_weight_sum, acc_alpha

    # --- regime-b: hard nearest-slab interior matte H_d (Item 4) ------------
    # argmax over slab_w → each pixel opaque in exactly ONE slab. Dilate the
    # discontinuity into a ~2·CoC unknown band so the matte is feathered (not
    # a razor cutout) — the boundary band is where OVER does its work.
    argmax_d = np.argmax(slab_w, axis=0).astype(np.int32)  # (H, W)
    coc_px = _coc_px(defocus_map, pad)
    band_px = int(np.clip(np.ceil(2.0 * float(np.median(coc_px))), 1, 64))
    # Boundary pixels: any pixel adjacent to a different argmax slab.
    edge = np.zeros((H, W), dtype=bool)
    edge[:-1, :] |= argmax_d[:-1, :] != argmax_d[1:, :]
    edge[1:, :] |= argmax_d[:-1, :] != argmax_d[1:, :]
    edge[:, :-1] |= argmax_d[:, :-1] != argmax_d[:, 1:]
    edge[:, 1:] |= argmax_d[:, :-1] != argmax_d[:, 1:]
    boundary_band = ndimage.binary_dilation(edge, iterations=band_px)

    # WRONG-POLARITY MATTE GUARD (Adversarial Surface 1 / Day-1 post-mortem).
    # The hard nearest-slab matte H_d must have the correct SENSE: the
    # foreground (NEAR, small object-distance, small argmax index) silhouette
    # interior must be OPAQUE in its own slab, NOT transparent. We verify the
    # partition is non-degenerate and a sentinel near-side interior pixel is
    # claimed by a near (low-index) slab — so a future refactor that flips the
    # matte construction (e.g. `argmin` instead of `argmax`, or `1 - H_d`)
    # fails loudly here instead of silently bleeding background through the
    # subject. Cheap: argmax over a PoU tensor that sums to 1 → exactly one
    # winner per pixel, so Σ_d (argmax==d) ≡ 1 (no holes, no double-cover).
    _interior_cover = np.zeros((H, W), dtype=np.float32)
    for _d in range(D):
        _interior_cover += (argmax_d == _d).astype(np.float32)
    if not np.allclose(_interior_cover, 1.0, atol=1e-4):
        raise RuntimeError(
            "occlusion matte polarity/partition guard FAILED: hard nearest-"
            "slab interior matte does not partition the frame "
            f"(Σ_d H_d range [{_interior_cover.min():.3f}, "
            f"{_interior_cover.max():.3f}], expected ≡1). A flipped/duplicated "
            "matte sense would bleed background through the foreground."
        )

    # Back-to-front OVER accumulators (premultiplied; Item 8).
    acc_c = np.zeros((H, W, 3), dtype=np.float32)  # premultiplied color
    acc_a = np.zeros((H, W), dtype=np.float32)     # coverage alpha

    # `img_diffuse` (unpadded) is the layer source for the premultiplied path
    # — the SAME diffuse split the OFF path convolves.
    img_diffuse = img_diffuse_padded[pad:pad + H, pad:pad + W, :]

    # Per-slab cost gate region: a slab's OVER contribution is only ever READ
    # at a regime-b OUTPUT pixel (s>0) — the final composite masks OVER by s,
    # so OVER values outside the s>0 region are discarded. A slab's matte
    # therefore only matters if, after blurring, it lands ON an s>0 output
    # pixel. The blur reach is ~pad, but the matte's OWN support already
    # extends across the silhouette; dilating the *active* region by the
    # feather band width (band_px, ~2·CoC) — NOT the full kernel pad — is
    # sufficient and ~pad/band_px× cheaper. round-2 perf: pad=33 dilation
    # made s_reach cover most of the frame so every slab ran the 2-pass; the
    # band_px gate restricts it to slabs whose matte actually touches the
    # silhouette band. Correctness preserved: any matte energy that blurs
    # FURTHER than band_px into an s>0 pixel comes from a slab whose interior
    # is adjacent to the silhouette (its argmax region borders the edge), and
    # that slab's matte already intersects the band by construction.
    s_reach = ndimage.binary_dilation(s_active, iterations=int(max(band_px, 1)))

    near_to_far = list(range(D))  # object_distances_m ascending → near→far

    # Q-e hoist: the radial-basis weights depend only on (radial_frac_map,
    # r_centers), not on the slab. Dense float32, identical to the value each
    # per-call build produced before.
    basis_w = _radial_basis_weights(radial_frac_map, r_centers)

    for d in near_to_far:
        if cancel is not None and cancel.is_set():
            from summicron.render.api import CancelledError
            raise CancelledError(f"layered_composite cancelled at slab {d}/{D}")
        sw_d = slab_w[d]  # (H, W)
        if float(sw_d.max()) < _EMPTY_SKIP_THRESH:
            continue
        if on_progress is not None:
            on_progress(f"  [occlusion] OVER slab {d + 1}/{D}")

        # ---- regime-b premultiplied OVER layer (Items 4/6/7/8) -------------
        # Hard nearest-slab interior matte: opaque where this slab is the
        # argmax, feathered across the dilated boundary band.
        H_d = (argmax_d == d).astype(np.float32)  # (H, W) interior matte
        # Feather: inside the boundary band, soften the binary matte toward
        # the PoU membership so the silhouette edge is semitransparent (the
        # blurred-matte soft-edge mechanism). Outside the band it stays a hard
        # solid (acc_a → 1 on solids, no bleed-through).
        H_d = np.where(boundary_band, np.maximum(H_d, sw_d), H_d).astype(np.float32)
        if float(H_d.max()) < _EMPTY_SKIP_THRESH:
            continue
        # Per-slab cost gate: if this slab's matte support does not reach the
        # blur-dilated active selector region, its OVER contribution can never
        # affect a regime-b output pixel (s=0 there → masked out by the final
        # composite s·OVER + (1−s)·OFF). Skip the (fused) OVER scatter for it.
        if not bool(((H_d > _EMPTY_SKIP_THRESH) & s_reach).any()):
            continue

        # Premultiply color. round-3 ENERGY FIX: the EDT disocclusion fill is
        # REMOVED here. probe-round-2 proved it was the sole cause of 2–3×
        # linear energy creation at silhouettes: it filled the matte==0 color
        # hole with nearest-valid color (≈1) but did NOT fill the matte (still
        # 0), so the premultiplied-color buffer carried color WITHOUT matching
        # coverage. After the two-pass blur, `conv_c` (blurred filled color)
        # exceeded `conv_a` (blurred matte ≈0 in the hole) → the un-premultiply
        # `conv_c/conv_a` broke the `conv_c ≤ conv_a` invariant and produced
        # >1 output (a silhouette BRIGHT FRINGE + global Core-Image-headroom
        # hijack). Without the fill, `conv_c = Σ K·H_d·color ≤ Σ K·H_d·1 =
        # conv_a` per channel (color ≤ 1) → the un-premultiply is bounded by
        # max(color) BY CONSTRUCTION (Barsky-Kosloff: the divide is the
        # H_d-weighted mean of color over the kernel support). The "don't pull
        # black into the disc" intent the fill served is handled correctly by
        # the OVER weighting itself: a hole pixel has layer_alpha=conv_a≈0, so
        # it contributes ≈0 to the OVER fold and the already-composited
        # background shows through by (1−acc_a) — correct partial coverage, not
        # a black hole. (`_disocclusion_fill` is retained for its unit tests +
        # a future Kraus-Strengert push-pull that fills coverage-consistently.)
        premult = H_d[..., None] * img_diffuse  # (H, W, 3)

        # Pad the premult colour + the 2-D matte; ONE fused scatter convolves
        # both with the slab PSF (colour and matte share the loop, so their
        # skip decisions are identical by construction).
        premult_pad = np.pad(
            premult, ((pad, pad), (pad, pad), (0, 0)), mode="reflect"
        )
        matte_pad = np.pad(H_d, ((pad, pad), (pad, pad)), mode="reflect")

        conv_c = np.zeros((H, W, 3), dtype=np.float32)
        conv_a = np.zeros((H, W), dtype=np.float32)
        with sp_fft.set_workers(-1):
            _over_scatter_slab(
                d_idx=d, premult_pad=premult_pad, matte_pad=matte_pad,
                basis_w=basis_w, wedge_idx=wedge_idx,
                kernels=kernels, defocus_ratios=defocus_ratios,
                kH=kH, kW=kW, pad=pad, pix_scale=pix_scale,
                scaled_ksize=scaled_ksize, psf_norm=psf_norm, H=H, W=W,
                fft_state=fft_state, conv_c=conv_c, conv_a=conv_a,
                cancel=cancel,
            )
        # The wsum_d radial renorm is IDENTICAL for color + matte (depends
        # only on (d_idx, basis_w, wedge_idx), not the convolved buffer), so
        # it cancels in (C'⊛K)/(H_d⊛K). We therefore do NOT renorm either by
        # wsum here — the cancellation IS the renorm. conv_c is premultiplied
        # blurred color; conv_a is the blurred matte α_blur.
        layer_premult_c = conv_c                      # (H, W, 3) premult color
        layer_alpha = conv_a                          # (H, W) blurred matte

        # Back-to-front OVER fold (premultiplied; near→far visited so we
        # composite the NEW (nearer-or-equal) layer UNDER the accumulator's
        # already-nearer content — standard back-to-front: acc holds the
        # composite so far of nearer layers; the farther/new layer goes
        # BEHIND). Porter-Duff "over" of acc OVER layer:
        #   acc_c = acc_c + (1 − acc_a) · layer_c
        #   acc_a = acc_a + (1 − acc_a) · layer_a
        one_minus_acc = (1.0 - acc_a)[..., None]
        acc_c = acc_c + one_minus_acc * layer_premult_c
        acc_a = acc_a + (1.0 - acc_a) * layer_alpha

    del basis_w

    # --- un-premultiply (Kosloff-Barsky α-bias clamp; Item 8) ---------------
    # acc_c / max(acc_a, ε). round-3: with the disocclusion fill removed the
    # premultiplied OVER now satisfies `acc_c ≤ acc_a · max_diffuse` per channel
    # by construction, so the divide is bounded by `max_diffuse ≈ 1`. The floor
    # ε is purely a divide-by-zero / speckle guard where blurred α → 0; it must
    # NOT become an amplifier. The old 1e-3 floor permitted up to 1000× gain on
    # thin edges — the Kosloff-Barsky bias is supposed to bias toward the
    # IDENTITY/background (so a vanishing-coverage pixel falls back to ~its own
    # diffuse value), NOT toward unbounded amplification. We (a) raise the floor
    # to a coverage-meaningful 1e-2, and (b) add the spec-sanctioned hard
    # energy-bound safety net: a diffuse-only layered composite can never exceed
    # the input diffuse bound, so clamp out_over to [0, max_input_diffuse].
    # Belt-and-suspenders: the construction bound makes the clamp a no-op on
    # correct inputs (verified by the energy-bound unit test), but it guarantees
    # no future regression can resurrect the bright-fringe / headroom-hijack.
    _diffuse_bound = float(max(img_diffuse.max(), 1e-6))
    out_over = acc_c / np.maximum(acc_a, np.float32(1e-2))[..., None]
    out_over = np.clip(out_over, 0.0, _diffuse_bound).astype(np.float32)

    # --- final diffuse composite: s·OVER + (1−s)·OFF_diffuse (Item 5) -------
    # off_diffuse is the layered_gather OFF diffuse (regime-a, bit-faithful).
    # Where s≈0 (99%+ of a real frame) the output == OFF diffuse to float tol;
    # only the s>0 silhouette band takes the premultiplied-OVER overlay.
    s = s_full[..., None]  # (H, W, 1) selector
    output_diffuse = (s * out_over + (1.0 - s) * off_diffuse).astype(np.float32)

    return output_diffuse, output_bright, weight_sum, delta_weight_sum, acc_alpha
