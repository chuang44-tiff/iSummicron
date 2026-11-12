"""Q-a R-eval-gates: paired focus-fidelity metrics (PRIVATE, dev/QA).

Measurement only; nothing here touches the render path. All image inputs are
float linear ``(H, W, 3)`` arrays (HDR values > 1 allowed); luma is Rec.709 Y.

Design (see ``doc/plan/phase2/section13_eval-gates/locked-spec-Q-a.md``):

* Metrics live on the *band* ``0.02 < alpha < 0.98`` (the transition zone of
  the identity-snap alpha), minus a 3-px dilation of ``(ref>1)|(out>1)``
  (bloom exclusion) and a border trim. ``alpha > 0`` is NOT used: the float32
  ``cos**2`` ramp is strictly positive everywhere (~1e-15), so ``alpha > 0``
  covers 100% of the frame.
* Gates are *paired* (variant - baseline on the same scene/res/placement/
  reference) except the single absolute sanity ceiling ``ger_band <= 1.02``.
"""
from __future__ import annotations

from typing import Any, Literal

import numpy as np
from scipy import ndimage

__all__ = [
    "luma",
    "focus_window",
    "band_mask",
    "knee_mask",
    "bg_mask",
    "q1_metrics",
    "bg_ger",
    "window_metrics",
    "focus_fidelity",
    "paired",
    "matte_spill_dp",
    "ALPHA_LO",
    "ALPHA_HI",
    "MIN_PX",
]

ALPHA_LO = 0.02
ALPHA_HI = 0.98
MIN_PX = 100
_EPS = 1e-6
_PAD = 8  # context pixels around the window so gradients/SSIM are valid at its edge

# Gate thresholds (spec §(c)).
GATE_OV = 0.006
GATE_CEIL = 1.02
GATE_BG = 0.01
GATE_CH = 0.02
GATE_NONBAND = 0.01


def luma(x: np.ndarray) -> np.ndarray:
    """Rec.709 luma of an ``(..., 3)`` array."""
    x = np.asarray(x)
    return (
        0.2126 * x[..., 0] + 0.7152 * x[..., 1] + 0.0722 * x[..., 2]
    ).astype(np.float64)


# ---------------------------------------------------------------------------
# Windows and masks
# ---------------------------------------------------------------------------


def focus_window(
    shape: tuple[int, ...],
    focus_xy: tuple[float, float],
    side_2048: int = 256,
    offset_px: tuple[int, int] = (0, 0),
) -> tuple[slice, slice]:
    """Square window around ``focus_xy`` as ``(slice_y, slice_x)``.

    ``focus_xy`` is (x, y) in [0, 1] (same convention as
    ``apply_tap_focus_shift``). Side = ``round(side_2048 * long_edge / 2048)``
    rounded to even, capped at the short edge. ``offset_px`` = (dx, dy) shifts
    the centre (already in frame pixels). The window is clamped inside the
    frame.
    """
    h, w = int(shape[0]), int(shape[1])
    side = int(round(side_2048 * max(h, w) / 2048.0))
    side += side % 2
    side = max(2, min(side, min(h, w)))
    cx = float(focus_xy[0]) * w + offset_px[0]
    cy = float(focus_xy[1]) * h + offset_px[1]
    x0 = int(round(cx - side / 2.0))
    y0 = int(round(cy - side / 2.0))
    x0 = min(max(x0, 0), w - side)
    y0 = min(max(y0, 0), h - side)
    return slice(y0, y0 + side), slice(x0, x0 + side)


def _dilate(m: np.ndarray, iterations: int) -> np.ndarray:
    if iterations <= 0:
        return m
    return ndimage.binary_dilation(m, iterations=iterations)


def _border_trim(shape: tuple[int, int], border: int) -> np.ndarray:
    keep = np.zeros(shape, dtype=bool)
    if border * 2 < min(shape):
        keep[border:shape[0] - border, border:shape[1] - border] = True
    return keep


def band_mask(
    alpha: np.ndarray,
    ref_y: np.ndarray,
    out_y: np.ndarray,
    lo: float = ALPHA_LO,
    hi: float = ALPHA_HI,
    bloom_dilate: int = 3,
    border: int = 4,
) -> np.ndarray:
    """Band ``lo < alpha < hi`` minus bloom dilation minus border trim."""
    m = (alpha > lo) & (alpha < hi)
    bloom = _dilate((ref_y > 1.0) | (out_y > 1.0), bloom_dilate)
    return m & ~bloom & _border_trim(alpha.shape, border)


def knee_mask(alpha: np.ndarray, dmap: np.ndarray, d_knee: float) -> np.ndarray:
    """Full knee zone: ``dmap < d_knee`` and ``alpha < 1``."""
    return (dmap < d_knee) & (alpha < 1.0)


def bg_mask(
    alpha: np.ndarray,
    ref_y: np.ndarray,
    out_y: np.ndarray,
    lo: float = ALPHA_LO,
    bloom_dilate: int = 3,
    border: int = 4,
) -> np.ndarray:
    """Background ``alpha <= lo`` minus bloom dilation minus border trim."""
    # Erode 1 px so the central-difference stencil never reads a band pixel
    # (a band-only lever must leave bg_ger untouched).
    m = ndimage.binary_erosion(alpha <= lo, iterations=1)
    bloom = _dilate((ref_y > 1.0) | (out_y > 1.0), bloom_dilate)
    return m & ~bloom & _border_trim(alpha.shape, border)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _grad_energy(y: np.ndarray, mask: np.ndarray) -> float:
    gy, gx = np.gradient(y)
    return float(((gy * gy + gx * gx)[mask]).sum())


def _ger(out_c: np.ndarray, ref_c: np.ndarray, mask: np.ndarray) -> float | None:
    den = _grad_energy(ref_c, mask)
    if den <= 0.0:
        return None
    return _grad_energy(out_c, mask) / den


def _none_metrics(mask: np.ndarray) -> dict[str, Any]:
    return {
        "ger_band": None, "ssim_band": None, "ovfrac": None, "ger_rgb": None,
        "chroma_err": None, "coverage": float(mask.mean()) if mask.size else 0.0,
        "n_px": int(mask.sum()),
    }


def q1_metrics(
    out_lin: np.ndarray,
    ref_lin: np.ndarray,
    alpha: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
    """Band metrics (see module docstring). ``None`` entries if ``n_px < 100``.

    ``alpha`` is accepted for signature symmetry with the spec; the mask has
    already been derived from it.
    """
    n_px = int(mask.sum())
    if n_px < MIN_PX:
        return _none_metrics(mask)
    out_y = luma(out_lin)
    ref_y = luma(ref_lin)

    ger_band = _ger(out_y, ref_y, mask)
    ger_rgb = tuple(
        _ger(out_lin[..., c].astype(np.float64), ref_lin[..., c].astype(np.float64), mask)
        for c in range(3)
    )
    if ger_band is None or any(g is None for g in ger_rgb):
        res = _none_metrics(mask)
        return res

    # Overshoot fraction vs the reference's 5x5 local [min, max].
    lmax = ndimage.maximum_filter(ref_y, size=5)
    lmin = ndimage.minimum_filter(ref_y, size=5)
    tol = 0.05 * (lmax - lmin) + _EPS
    viol = (out_y > lmax + tol) | (out_y < lmin - tol)
    ovfrac = float(viol[mask].mean())

    # Chroma error (report only).
    e = 1e-4
    lo_ = np.log((out_lin.astype(np.float64) + e) / (out_y[..., None] + e))
    lr_ = np.log((ref_lin.astype(np.float64) + e) / (ref_y[..., None] + e))
    chroma_err = float(np.abs(lo_ - lr_)[mask].mean())

    ssim_band: float | None
    try:
        from skimage.metrics import structural_similarity
    except ImportError:
        ssim_band = None
    else:
        vals = ref_y[mask]
        dr = float(np.percentile(vals, 99.5) - np.percentile(vals, 0.5))
        dr = max(dr, 1e-6)
        _, smap = structural_similarity(
            ref_y, out_y, data_range=dr, gaussian_weights=True, sigma=1.5,
            use_sample_covariance=False, full=True,
        )
        ssim_band = float(smap[mask].mean())

    return {
        "ger_band": float(ger_band),
        "ssim_band": ssim_band,
        "ovfrac": ovfrac,
        "ger_rgb": tuple(float(g) for g in ger_rgb),
        "chroma_err": chroma_err,
        "coverage": float(mask.mean()),
        "n_px": n_px,
    }


def _crop(win: tuple[slice, slice], shape: tuple[int, ...], pad: int):
    h, w = shape[:2]
    ys, xs = win
    y0, y1 = max(ys.start - pad, 0), min(ys.stop + pad, h)
    x0, x1 = max(xs.start - pad, 0), min(xs.stop + pad, w)
    inner = np.zeros((y1 - y0, x1 - x0), dtype=bool)
    b = 4
    # Clamp to >= 0 so a tiny/clamped window never wraps (negative stop).
    iy0, iy1 = max(ys.start - y0 + b, 0), max(ys.stop - y0 - b, 0)
    ix0, ix1 = max(xs.start - x0 + b, 0), max(xs.stop - x0 - b, 0)
    if iy1 > iy0 and ix1 > ix0:
        inner[iy0:iy1, ix0:ix1] = True
    return (slice(y0, y1), slice(x0, x1)), inner


def bg_ger(
    out_lin: np.ndarray,
    ref_lin: np.ndarray,
    alpha: np.ndarray,
    window: tuple[slice, slice],
    mask_out: np.ndarray | None = None,
) -> float | None:
    """GER over the background (``alpha <= 0.02``) inside ``window``.

    The global-weakening counter. ``mask_out`` (default ``out_lin``) supplies
    the bloom-exclusion luma so base/variant share one mask.
    """
    crop, inner = _crop(window, out_lin.shape, _PAD)
    o = out_lin[crop]
    r = ref_lin[crop]
    mo = o if mask_out is None else mask_out[crop]
    m = bg_mask(alpha[crop], luma(r), luma(mo), border=0) & inner
    if int(m.sum()) < MIN_PX:
        return None
    return _ger(luma(o), luma(r), m)


def window_metrics(
    out_lin: np.ndarray,
    ref_lin: np.ndarray,
    alpha: np.ndarray,
    window: tuple[slice, slice],
    mask_out: np.ndarray | None = None,
) -> dict[str, Any]:
    """:func:`q1_metrics` over ``window`` plus ``bg_ger`` (key ``bg_ger``).

    ``mask_out`` (default ``out_lin``) is the array whose luma feeds the bloom
    exclusion; pass the BASE output for both base and variant so the paired
    comparison uses an identical mask.
    """
    crop, inner = _crop(window, out_lin.shape, _PAD)
    o = out_lin[crop]
    r = ref_lin[crop]
    a = alpha[crop]
    mo = o if mask_out is None else mask_out[crop]
    mask = band_mask(a, luma(r), luma(mo), border=0) & inner
    res = q1_metrics(o, r, a, mask)
    res["bg_ger"] = bg_ger(out_lin, ref_lin, alpha, window, mask_out=mask_out)
    return res


def focus_fidelity(
    out_lin: np.ndarray,
    ref_lin: np.ndarray,
    alpha: np.ndarray,
    focus_xy: tuple[float, float],
    side_2048: int = 256,
) -> dict[str, Any] | None:
    """Pipeline entry: band metrics in the focus window; ``None`` if skimage
    is unavailable (``ssim_band`` could not be computed)."""
    try:
        import skimage.metrics  # noqa: F401
    except ImportError:
        return None
    win = focus_window(out_lin.shape, focus_xy, side_2048)
    crop, inner = _crop(win, out_lin.shape, _PAD)
    o = out_lin[crop]
    r = ref_lin[crop]
    a = alpha[crop]
    if min(o.shape[:2]) < 7:
        return None
    mask = band_mask(a, luma(r), luma(o), border=0) & inner
    return q1_metrics(o, r, a, mask)


# ---------------------------------------------------------------------------
# Paired gates
# ---------------------------------------------------------------------------


def paired(
    base: dict[str, Any],
    var: dict[str, Any],
    lever: Literal["band", "nonband"],
) -> dict[str, Any]:
    """Deltas (var - base) and per-gate pass/fail for one window.

    ``base``/``var`` come from :func:`window_metrics` (same window, same
    mask). Returns ``{"valid": False}`` if either side has ``None`` metrics.
    """
    if lever not in ("band", "nonband"):
        raise ValueError(f"lever must be 'band' or 'nonband', got {lever!r}")
    keys = ("ger_band", "ssim_band", "ovfrac", "ger_rgb")
    if any(base.get(k) is None or var.get(k) is None for k in keys):
        return {"valid": False}
    d_ger = var["ger_band"] - base["ger_band"]
    d_ssim = var["ssim_band"] - base["ssim_band"]
    d_ov = var["ovfrac"] - base["ovfrac"]
    d_ch = max(
        abs((v - b) - d_ger) for v, b in zip(var["ger_rgb"], base["ger_rgb"])
    )
    ce_b, ce_v = base.get("chroma_err"), var.get("chroma_err")
    d_chroma = None if (ce_b is None or ce_v is None) else ce_v - ce_b
    bg_b, bg_v = base.get("bg_ger"), var.get("bg_ger")
    d_bg = None if (bg_b is None or bg_v is None) else bg_v - bg_b
    gates = {
        "OV": d_ov <= GATE_OV,
        "CEIL": var["ger_band"] <= GATE_CEIL,
        # No measurable bg pixels -> gate cannot be evaluated -> FAIL
        # ("invalid: no background pixels"), never a vacuous PASS.
        "BG": False if d_bg is None else abs(d_bg) <= GATE_BG,
    }
    if lever == "band":
        gates["Q1-B"] = bool(d_ger > 0 and d_ssim > 0)
    else:
        gates["Q1-N"] = abs(d_ger) <= GATE_NONBAND
    return {
        "valid": True,
        "d_ger": float(d_ger),
        "d_ssim": float(d_ssim),
        "d_ovfrac": float(d_ov),
        "d_bg": None if d_bg is None else float(d_bg),
        "d_ch": float(d_ch),
        # Report-only: change in chroma_err (var - base); not a gate.
        "d_chroma": None if d_chroma is None else float(d_chroma),
        # Report-only: d_ch cannot separate channel-neutral levers from tints.
        "ch_flag": bool(d_ch > GATE_CH),
        "bg_invalid": d_bg is None,
        "ger_var": float(var["ger_band"]),
        "gates": gates,
    }


def bg_fullframe(
    base: np.ndarray,
    var: np.ndarray,
    ref: np.ndarray,
    alpha: np.ndarray,
    footprint: np.ndarray | None = None,
) -> dict[str, Any]:
    """Full-frame BG gate: |dGER| over (bg minus ``footprint``) <= GATE_BG.

    The bloom mask is built from ``base`` (shared by base/variant). No
    measurable bg pixels (or zero reference gradient energy) -> ``valid``
    False and ``gate`` False (invalid: no background pixels), never a PASS.
    """
    yr, yb, yv = luma(ref), luma(base), luma(var)
    m = bg_mask(alpha, yr, yb)
    if footprint is not None:
        m = m & ~footprint
    n_px = int(m.sum())
    gb = _ger(yb, yr, m) if n_px >= MIN_PX else None
    gv = _ger(yv, yr, m) if n_px >= MIN_PX else None
    if gb is None or gv is None:
        return {"valid": False, "d_bg": None, "n_px": n_px, "gate": False}
    d = float(gv - gb)
    return {"valid": True, "d_bg": d, "n_px": n_px, "gate": abs(d) <= GATE_BG}


def matte_spill_dp(
    dp: np.ndarray,
    matte: np.ndarray,
    thresholds_d: tuple[float, ...] = (0.41, 0.9),
    matte_min: float = 0.5,
) -> dict[float, float]:
    """S22 G4: share of subject pixels (``matte > matte_min``) whose effective
    diopter defocus ``|dp|`` exceeds each threshold (diopters), as fractions.

    ``dp`` is ``_eval_sink["dp"]`` (render-effective ``g * dp`` after the cap);
    a matte of a different size is nearest-resized to ``dp``. Returns
    ``{threshold: share}``; empty subject gives NaN.
    """
    dp = np.abs(np.asarray(dp, dtype=np.float32))
    m = np.asarray(matte, dtype=np.float32)
    if m.shape != dp.shape:
        ys = (np.arange(dp.shape[0]) * m.shape[0] // dp.shape[0]).astype(int)
        xs = (np.arange(dp.shape[1]) * m.shape[1] // dp.shape[1]).astype(int)
        m = m[np.ix_(ys, xs)]
    sel = m > matte_min
    n = int(sel.sum())
    vals = dp[sel]
    return {
        float(t): (float((vals > t).sum()) / n if n else float("nan"))
        for t in thresholds_d
    }
