"""Shared numerical helpers for eval metrics.

`fit_ellipse_guarded` returns a dict with degeneracy guards (min-support,
min-minor-axis). `gradient_energy` is Sobel-based. `project_covariance_radial`
projects a fitted ellipse's covariance onto sagittal/tangential unit vectors.

Axis convention: image arrays are (H, W); centers are (x, y) = (col, row).
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import sobel


# Ellipse-fit degeneracy thresholds (LOCKED SPEC §3).
MIN_SUPPORT_PX = 20
MIN_MINOR_PX = 0.5
ROI_THRESHOLD_FRAC = 0.1  # threshold = frac * ROI.max()


def fit_ellipse_guarded(roi: np.ndarray) -> dict:
    """Threshold + largest CC + ellipse fit with degeneracy guards.

    Parameters
    ----------
    roi : np.ndarray, 2D float
        Monochrome intensity ROI. NaN/Inf raises ValueError.

    Returns
    -------
    dict with keys:
        value : str, "ok" or error tag (low_support, no_blob, degenerate_minor,
                input_flat)
        flag  : str | None, same as value except None when "ok"
        major, minor : float (pixels; 0.0 if fit failed)
        orientation  : float (radians, skimage convention; 0.0 if failed)
        area, centroid : int, (row, col) — from regionprops
    """
    if not np.isfinite(roi).all():
        raise ValueError("ROI contains NaN/Inf")
    roi = np.asarray(roi, dtype=np.float32)
    if roi.ndim != 2:
        raise ValueError(f"ROI must be 2D, got shape {roi.shape}")

    peak = float(roi.max())
    if peak <= 0.0 or not np.isfinite(peak):
        return _empty_fit("input_flat")

    threshold = ROI_THRESHOLD_FRAC * peak
    mask = roi > threshold
    if mask.sum() == 0:
        return _empty_fit("no_blob")

    from skimage.measure import label, regionprops
    labeled = label(mask)
    props = regionprops(labeled)
    if not props:
        return _empty_fit("no_blob")
    # largest CC
    largest = max(props, key=lambda p: p.area)
    if largest.area < MIN_SUPPORT_PX:
        return _empty_fit("low_support")

    major = float(largest.axis_major_length)
    minor = float(largest.axis_minor_length)
    if minor < MIN_MINOR_PX:
        return _empty_fit("degenerate_minor")

    return {
        "value": "ok",
        "flag": None,
        "major": major,
        "minor": minor,
        "orientation": float(largest.orientation),
        "area": int(largest.area),
        "centroid": tuple(float(c) for c in largest.centroid),
    }


def _empty_fit(flag: str) -> dict:
    return {
        "value": flag,
        "flag": flag,
        "major": 0.0,
        "minor": 0.0,
        "orientation": 0.0,
        "area": 0,
        "centroid": (0.0, 0.0),
    }


def gradient_energy(tile: np.ndarray) -> float:
    """Mean squared Sobel gradient magnitude on a 2D monochrome tile."""
    tile = np.asarray(tile, dtype=np.float32)
    if tile.ndim != 2:
        raise ValueError(f"tile must be 2D, got shape {tile.shape}")
    if not np.isfinite(tile).all():
        raise ValueError("tile contains NaN/Inf")
    gx = sobel(tile, axis=1)
    gy = sobel(tile, axis=0)
    return float((gx * gx + gy * gy).mean())


def project_covariance_radial(
    major: float,
    minor: float,
    orientation: float,
    center_xy: tuple[float, float],
    optical_center_xy: tuple[float, float],
) -> dict:
    """Project a fitted ellipse's covariance onto sagittal/tangential axes.

    skimage orientation: angle (radians) of the major axis w.r.t. the
    column-axis (x), in [-pi/2, pi/2]. We build the ellipse covariance as
    ``R diag(major/2, minor/2)^2 R^T`` with R from orientation, then project
    onto the unit vector from the optical center to the disc center
    (sagittal) and its perpendicular (tangential).

    Returns dict with keys ``sigma_sag``, ``sigma_tan``, ``sag_tan``,
    ``theta_major_vs_radial_deg``. Sigmas are in pixels (sqrt of projected
    covariance eigenvalue).
    """
    cx, cy = center_xy
    ox, oy = optical_center_xy
    dx = cx - ox
    dy = cy - oy
    r = float(np.hypot(dx, dy))
    if r < 1e-9:
        # Undefined radial direction; caller should guard upstream.
        r_hat = np.array([1.0, 0.0], dtype=np.float64)
    else:
        r_hat = np.array([dx / r, dy / r], dtype=np.float64)
    t_hat = np.array([-r_hat[1], r_hat[0]], dtype=np.float64)

    # Semi-axes: regionprops returns full lengths; covariance eigenvalues are
    # (semi-axis)^2 for a uniform ellipse scaled by 1/4 (as in moments), but
    # we only use relative projection -> use semi-axis squared.
    a = major / 2.0
    b = minor / 2.0
    # skimage orientation: angle between major axis and +x (columns).
    # Build rotation from major-axis frame to image (x, y).
    c = np.cos(orientation)
    s = np.sin(orientation)
    # Major axis direction in (x, y):
    maj_vec = np.array([c, s], dtype=np.float64)
    min_vec = np.array([-s, c], dtype=np.float64)
    # Covariance = a^2 * maj_vec maj_vec^T + b^2 * min_vec min_vec^T.
    cov = (a * a) * np.outer(maj_vec, maj_vec) + (b * b) * np.outer(
        min_vec, min_vec
    )

    sigma_sag = float(np.sqrt(max(r_hat @ cov @ r_hat, 0.0)))
    sigma_tan = float(np.sqrt(max(t_hat @ cov @ t_hat, 0.0)))
    if sigma_tan < 1e-9:
        sag_tan = float("nan")
    else:
        sag_tan = sigma_sag / sigma_tan

    # angle between major axis and radial direction, wrapped to [-90, 90] deg
    theta_rad = np.arctan2(maj_vec[1], maj_vec[0]) - np.arctan2(
        r_hat[1], r_hat[0]
    )
    theta_deg = float(np.degrees(theta_rad))
    # wrap to [-90, 90] since ellipse orientation is mod pi
    theta_deg = ((theta_deg + 90.0) % 180.0) - 90.0

    return {
        "sigma_sag": sigma_sag,
        "sigma_tan": sigma_tan,
        "sag_tan": sag_tan,
        "theta_major_vs_radial_deg": theta_deg,
    }


def assert_monochrome(rgb: np.ndarray, name: str = "rgb") -> None:
    """Assert (H, W, 3) float with R=G=B."""
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError(f"{name} must be (H, W, 3), got {rgb.shape}")
    if not np.isfinite(rgb).all():
        raise ValueError(f"{name} contains NaN/Inf")
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    if not (np.allclose(r, g, atol=1e-6) and np.allclose(g, b, atol=1e-6)):
        raise AssertionError(f"{name} is not monochrome (R=G=B violation)")
