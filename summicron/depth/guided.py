"""Guided-filter (He et al., TPAMI 2013) depth upsampling refinement.

Leaf module: numpy + scipy only. The filter runs in DISPARITY (1/depth)
space, where LiDAR / perspective depth is approximately piecewise affine in
image space and an edge-aware linear model ``q = a*I + b`` fits well; the
result is converted back to meters.

Guide choice: a single luminance channel (Rec.709 weights on the sRGB-ish
image). Rationale: one scalar covariance per window makes the filter exactly
the O(N) box-filter formulation with 6 box passes, vs a 3x3 covariance solve
per pixel for the RGB variant; hair/silhouette edges against the backdrop
are luminance edges in nearly all portrait scenes. Known limit: isoluminant
colour edges are not snapped (the RGB variant would help there).
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter

_DEPTH_MIN_M = 0.1
_DEPTH_MAX_M = 1e3
_DISP_FLOOR = 1.0 / _DEPTH_MAX_M


def luminance(rgb: np.ndarray) -> np.ndarray:
    """Rec.709 luma of an (H, W, 3) array; (H, W) input passes through."""
    a = np.asarray(rgb, dtype=np.float32)
    if a.ndim == 2:
        return a
    if a.ndim != 3 or a.shape[2] < 3:
        raise ValueError(f"guide must be (H,W) or (H,W,3), got {a.shape}")
    return a[..., 0] * 0.2126 + a[..., 1] * 0.7152 + a[..., 2] * 0.0722


def _box(x: np.ndarray, r: int) -> np.ndarray:
    return uniform_filter(x, size=2 * r + 1, mode="nearest")


def guided_filter(guide: np.ndarray, src: np.ndarray, r: int, eps: float) -> np.ndarray:
    """Gray-guide guided filter. ``guide`` and ``src`` are (H, W) float."""
    I = np.asarray(guide, dtype=np.float64)  # noqa: E741 (paper notation)
    p = np.asarray(src, dtype=np.float64)
    if I.shape != p.shape or I.ndim != 2:
        raise ValueError(f"guide {I.shape} and src {p.shape} must be equal (H,W)")
    if r < 1:
        raise ValueError("r must be >= 1")
    if eps <= 0:
        raise ValueError("eps must be > 0")
    mean_I = _box(I, r)
    mean_p = _box(p, r)
    cov_Ip = _box(I * p, r) - mean_I * mean_p
    var_I = _box(I * I, r) - mean_I * mean_I
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    return _box(a, r) * I + _box(b, r)


def guided_upsample_depth(
    depth_m: np.ndarray,
    guide: np.ndarray,
    r: int = 16,
    eps: float = 1e-3,
) -> np.ndarray:
    """Refine an already-upsampled depth map (meters) with an image guide.

    Parameters
    ----------
    depth_m : (H, W) meters, small = near (bilinearly upsampled LiDAR).
    guide   : (H, W) or (H, W, 3) image in [0, 1] at the same H, W.
    r       : box radius in pixels at working resolution.
    eps     : regulariser on guide variance (guide in [0, 1] units, so
              eps ~ 1e-3..1e-2 means "edges with contrast > ~0.03-0.1").

    Returns (H, W) float32 meters, finite, clipped to [0.1, 1e3].
    """
    d = np.asarray(depth_m, dtype=np.float32)
    if d.ndim != 2:
        raise ValueError(f"depth_m must be (H,W), got {d.shape}")
    g = luminance(guide)
    if g.shape != d.shape:
        raise ValueError(f"guide {g.shape} != depth {d.shape}")
    disp = 1.0 / np.clip(d, _DEPTH_MIN_M, _DEPTH_MAX_M)
    q = guided_filter(g, disp, r, eps)
    # Keep the output inside the input disparity range (a/b extrapolation
    # near edges can overshoot) so no new near/far extremes appear.
    q = np.clip(q, float(disp.min()), float(disp.max()))
    q = np.maximum(q, _DISP_FLOOR)
    out = 1.0 / q
    out = np.clip(out, _DEPTH_MIN_M, _DEPTH_MAX_M).astype(np.float32)
    if not np.all(np.isfinite(out)):
        raise ValueError("guided depth produced non-finite values")
    return out


# ---------------------------------------------------------------------------
# A1: LiDAR staircase smoothing (section15 locked spec, ruling 13 / 2.0)
# ---------------------------------------------------------------------------

_ASPECT_TOL = 0.02


def load_luma_guide(heic_path, target_hw: tuple[int, int]) -> np.ndarray | None:
    """Luminance guide from the HEIC primary image, or ``None`` (warns).

    The primary image is decoded as 8-bit sRGB in its STORED orientation (no
    ``exif_transpose``, matching the depth aux and the portrait matte),
    bilinearly resized to ``target_hw`` = (H, W), then Rec.709 luma on the
    encoded values. Never raises on bad input; every warning starts with
    ``depth-guided:``.
    """
    import warnings

    msg = None
    try:
        import pillow_heif
        from PIL import Image

        heif = pillow_heif.open_heif(str(heic_path), convert_hdr_to_8bit=True)
        im = heif.to_pillow().convert("RGB")
        height, width = int(target_hw[0]), int(target_hw[1])
        ratio = (im.width / im.height) / (width / height)
        if abs(ratio - 1.0) > _ASPECT_TOL:
            msg = (
                f"depth-guided: guide aspect {im.width}x{im.height} does not match "
                f"target {width}x{height}; guide skipped."
            )
        else:
            im = im.resize((width, height), Image.BILINEAR)
            rgb = np.asarray(im, dtype=np.float32) / 255.0
    except Exception as exc:  # never raise on bad input
        warnings.warn(
            f"depth-guided: could not decode guide ({exc!r}); guide skipped.", stacklevel=2
        )
        return None
    if msg is not None:
        warnings.warn(msg, stacklevel=2)
        return None
    return luminance(rgb).astype(np.float32)


def smooth_lidar_staircase(
    depth_m: np.ndarray,
    guide: np.ndarray,
    *,
    r_px_2048: int = 4,
    eps: float = 1e-4,
) -> np.ndarray:
    """Light guided smoothing of the LiDAR-grid staircase (fixed r/eps).

    ``r`` scales with resolution: ``max(1, round(r_px_2048 * max(H, W) / 2048))``.
    Returns (H, W) float32 meters; disparity stays inside the input's range.
    """
    d = np.asarray(depth_m)
    if d.ndim != 2:
        raise ValueError(f"depth_m must be (H,W), got {d.shape}")
    g = np.asarray(guide)
    if g.shape[:2] != d.shape:
        raise ValueError(f"guide {g.shape} != depth {d.shape}")
    r = max(1, round(r_px_2048 * max(d.shape) / 2048))
    return guided_upsample_depth(d, g, r=r, eps=eps)
