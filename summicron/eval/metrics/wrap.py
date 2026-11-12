"""M7 — wrap-probe sanity.

Per LOCKED SPEC §3.M7: for every disc flagged ``wrap_applied=true``, assert
the ROI is finite, has non-zero support, and its diameter ratio relative to
a reference disc (same post-wrap kernel cell; for ``bubble_grid`` that is
disc #3 at 0.98m) is approximately 1.0.
"""
from __future__ import annotations

import numpy as np

from .common import fit_ellipse_guarded


ROI_HALF = 32
SUPPORT_MIN = 20
DIAMETER_RATIO_EXPECTED = (0.80, 1.20)


def _crop_roi(rendered_mono: np.ndarray, cx: float, cy: float) -> np.ndarray:
    h, w = rendered_mono.shape
    x0 = max(0, int(round(cx)) - ROI_HALF)
    x1 = min(w, int(round(cx)) + ROI_HALF)
    y0 = max(0, int(round(cy)) - ROI_HALF)
    y1 = min(h, int(round(cy)) + ROI_HALF)
    return rendered_mono[y0:y1, x0:x1]


def _diameter_from_fit(fit: dict) -> float:
    if fit["flag"] is not None:
        return float("nan")
    return 2.0 * float(np.sqrt(fit["area"] / np.pi))


def _find_reference_disc(annotation: dict, wrap_disc: dict) -> dict | None:
    """Find a non-wrap disc whose depth matches the wrapped effective depth.

    For ``bubble_grid``: focus=0.70, wrap disc depth=0.40 -> effective 1.00m;
    nearest tensor cell = 0.98m -> disc #3 (at 0.98m, r_frac=0.40).
    General rule: pick non-wrap disc with depth closest to the wrapped
    effective depth (``2*focus - depth``).
    """
    focus = float(annotation.get("focus_distance_m", 0.70))
    wrap_depth = float(wrap_disc["depth_m"])
    effective = 2.0 * focus - wrap_depth
    best = None
    best_dist = float("inf")
    for d in annotation["discs"]:
        if d.get("wrap_applied", False):
            continue
        dd = abs(float(d["depth_m"]) - effective)
        if dd < best_dist:
            best_dist = dd
            best = d
    return best


def m7_wrap_probe(rendered_rgb: np.ndarray, annotation: dict) -> dict:
    """Return {finite_ok, support_px, diameter_ratio, flag} for the wrap probe.

    If multiple wrap-applied discs exist, the first one is reported and the
    flag notes 'multiple_wrap_discs' when that happens.
    """
    mono = rendered_rgb[..., 0]
    wrap_discs = [d for d in annotation["discs"] if d.get("wrap_applied", False)]
    if not wrap_discs:
        return {
            "finite_ok": False,
            "support_px": 0,
            "diameter_ratio": float("nan"),
            "flag": "no_wrap_disc",
        }
    wrap = wrap_discs[0]
    cx, cy = wrap["center_xy_px"]
    roi = _crop_roi(mono, cx, cy)
    finite_ok = bool(np.isfinite(roi).all())
    if not finite_ok:
        return {
            "finite_ok": False,
            "support_px": 0,
            "diameter_ratio": float("nan"),
            "flag": "wrap_produced_nonfinite",
        }
    # Fit to measure support and diameter.
    try:
        fit = fit_ellipse_guarded(roi)
    except ValueError:
        return {
            "finite_ok": False,
            "support_px": 0,
            "diameter_ratio": float("nan"),
            "flag": "wrap_produced_nonfinite",
        }
    support = int(fit["area"])
    if support < SUPPORT_MIN:
        return {
            "finite_ok": True,
            "support_px": support,
            "diameter_ratio": float("nan"),
            "flag": "wrap_no_blob",
        }
    wrap_diam = _diameter_from_fit(fit)

    ref = _find_reference_disc(annotation, wrap)
    if ref is None:
        return {
            "finite_ok": True,
            "support_px": support,
            "diameter_ratio": float("nan"),
            "flag": "no_reference_disc",
        }
    rcx, rcy = ref["center_xy_px"]
    ref_roi = _crop_roi(mono, rcx, rcy)
    try:
        ref_fit = fit_ellipse_guarded(ref_roi)
    except ValueError:
        return {
            "finite_ok": True,
            "support_px": support,
            "diameter_ratio": float("nan"),
            "flag": "reference_nonfinite",
        }
    if ref_fit["flag"] is not None:
        return {
            "finite_ok": True,
            "support_px": support,
            "diameter_ratio": float("nan"),
            "flag": f"reference_{ref_fit['flag']}",
        }
    ref_diam = _diameter_from_fit(ref_fit)
    if ref_diam <= 0 or not np.isfinite(ref_diam):
        return {
            "finite_ok": True,
            "support_px": support,
            "diameter_ratio": float("nan"),
            "flag": "reference_zero_diameter",
        }
    ratio = wrap_diam / ref_diam
    flag = None
    if len(wrap_discs) > 1:
        flag = "multiple_wrap_discs"
    return {
        "finite_ok": True,
        "support_px": support,
        "diameter_ratio": float(ratio),
        "flag": flag,
    }
