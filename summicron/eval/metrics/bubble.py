"""M2/M3/M4 — bubble-disc metrics.

M2 on-axis roundness; M3 off-axis sagittal/tangential (signed, radial-vector
projection per LOCKED SPEC §3.M3); M4 per-column signed Spearman of
diameter-vs-|depth - focus|. All three filter discs to ``wrap_applied=false``.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

from .common import (
    fit_ellipse_guarded,
    project_covariance_radial,
)


ROI_HALF = 32  # 64x64 ROI
R_FRAC_NEAR_CENTER = 0.3


def _crop_roi(
    rendered_mono: np.ndarray, cx: float, cy: float, half: int = ROI_HALF
) -> np.ndarray:
    h, w = rendered_mono.shape
    x0 = max(0, int(round(cx)) - half)
    x1 = min(w, int(round(cx)) + half)
    y0 = max(0, int(round(cy)) - half)
    y1 = min(h, int(round(cy)) + half)
    return rendered_mono[y0:y1, x0:x1]


def _non_wrap_discs(annotation: dict) -> list[dict]:
    return [d for d in annotation["discs"] if not d.get("wrap_applied", False)]


def m2_roundness(
    rendered_rgb: np.ndarray, annotation: dict
) -> tuple[float, str | None]:
    """On-axis roundness (disc with r_frac == 0 and wrap_applied=False)."""
    mono = rendered_rgb[..., 0]
    center_discs = [
        d for d in _non_wrap_discs(annotation) if float(d.get("r_frac", 0.0)) == 0.0
    ]
    if not center_discs:
        return float("nan"), "no_center_disc"
    disc = center_discs[0]
    cx, cy = disc["center_xy_px"]
    roi = _crop_roi(mono, cx, cy)
    fit = fit_ellipse_guarded(roi)
    if fit["flag"] is not None:
        return float("nan"), fit["flag"]
    return fit["minor"] / fit["major"], None


def m3_sag_tan(
    rendered_rgb: np.ndarray, annotation: dict
) -> list[dict]:
    """Off-axis sagittal/tangential ratio per disc.

    Returns a list of dicts (one per eligible disc): ``{idx, sag_tan,
    theta_deg, flag}``. Discs with ``r_frac < 0.3`` or wrap_applied are
    filtered/flagged.
    """
    mono = rendered_rgb[..., 0]
    opt_center = tuple(annotation["optical_center_xy"])
    out: list[dict] = []

    for disc in _non_wrap_discs(annotation):
        idx = int(disc["idx"])
        r_frac = float(disc.get("r_frac", 0.0))
        if r_frac < R_FRAC_NEAR_CENTER:
            out.append(
                {
                    "idx": idx,
                    "sag_tan": float("nan"),
                    "theta_deg": float("nan"),
                    "flag": "near_center_degenerate",
                }
            )
            continue
        cx, cy = disc["center_xy_px"]
        roi = _crop_roi(mono, cx, cy)
        fit = fit_ellipse_guarded(roi)
        if fit["flag"] is not None:
            out.append(
                {
                    "idx": idx,
                    "sag_tan": float("nan"),
                    "theta_deg": float("nan"),
                    "flag": fit["flag"],
                }
            )
            continue
        proj = project_covariance_radial(
            major=fit["major"],
            minor=fit["minor"],
            orientation=fit["orientation"],
            center_xy=(float(cx), float(cy)),
            optical_center_xy=opt_center,
        )
        out.append(
            {
                "idx": idx,
                "sag_tan": float(proj["sag_tan"]),
                "theta_deg": float(proj["theta_major_vs_radial_deg"]),
                "flag": None,
            }
        )
    return out


def _disc_diameter(rendered_mono: np.ndarray, cx: float, cy: float) -> tuple[float, str | None]:
    roi = _crop_roi(rendered_mono, cx, cy)
    fit = fit_ellipse_guarded(roi)
    if fit["flag"] is not None:
        return float("nan"), fit["flag"]
    # Diameter via area: 2*sqrt(area/pi).
    return 2.0 * float(np.sqrt(fit["area"] / np.pi)), None


def _column_band(r_frac: float) -> str | None:
    # Three bands: center (0.0/0.25), mid (0.40/0.45), corner (0.77).
    if r_frac < 0.30:
        return "center"
    if r_frac < 0.60:
        return "mid"
    return "corner"


def m4_diameter_depth_monotonicity(
    rendered_rgb: np.ndarray, annotation: dict
) -> tuple[dict, dict]:
    """Per-column signed Spearman. Returns (rhos, flags_by_col)."""
    mono = rendered_rgb[..., 0]
    focus = float(annotation.get("focus_distance_m", 0.70))
    cols: dict[str, list[tuple[float, float]]] = {
        "center": [],
        "mid": [],
        "corner": [],
    }
    col_flags: dict[str, str | None] = {
        "center": None,
        "mid": None,
        "corner": None,
    }

    for disc in _non_wrap_discs(annotation):
        band = _column_band(float(disc.get("r_frac", 0.0)))
        if band is None:
            continue
        cx, cy = disc["center_xy_px"]
        diam, flag = _disc_diameter(mono, cx, cy)
        if flag is not None:
            col_flags[band] = "upstream_nan"
        cols[band].append((abs(float(disc["depth_m"]) - focus), diam))

    rhos: dict[str, float] = {}
    for band, pairs in cols.items():
        if col_flags[band] == "upstream_nan":
            rhos[band] = float("nan")
            continue
        if len(pairs) < 2:
            rhos[band] = float("nan")
            col_flags[band] = "insufficient_points"
            continue
        depths = np.array([p[0] for p in pairs], dtype=np.float64)
        diams = np.array([p[1] for p in pairs], dtype=np.float64)
        # Flat response guard: spearmanr with nan_policy="raise".
        if np.allclose(diams, diams[0]):
            rhos[band] = float("nan")
            col_flags[band] = "flat_response"
            continue
        if np.allclose(depths, depths[0]):
            rhos[band] = float("nan")
            col_flags[band] = "flat_response"
            continue
        res = spearmanr(depths, diams, nan_policy="raise")
        rho_val = float(res.statistic if hasattr(res, "statistic") else res[0])
        rhos[band] = rho_val

    return rhos, col_flags
