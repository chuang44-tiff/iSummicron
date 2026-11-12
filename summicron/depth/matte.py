"""Apple portrait-matte correction of LiDAR depth (section15 locked spec, 2.1-2.2).

Leaf module: numpy + scipy.ndimage + PIL; ``pillow_heif`` is imported
function-locally; no import from ``summicron.render``.

The correction blends near/far reference DISPARITIES inside a narrow band
around the matte's soft zone. Non-band pixels are never assigned, so the
output equals the input there bit for bit.
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt

_ASPECT_TOL = 0.02
_MATTE_URN_SUFFIX = ":portraiteffectsmatte"
_CORE_MIN_PX = 2000
_AGREE_RATIO = 1.3
_AGREE_FRAC = 0.95
_CURVES = ("linear", "sqrt", "hard")
_POLARITIES = ("normal", "inverted")


def _warn(msg: str) -> None:
    warnings.warn(f"depth-matte: {msg}", stacklevel=3)


def load_portrait_matte(heic_path, target_hw, *, depth_m=None):
    """Portrait matte alpha (H, W) float32 in [0, 1], or ``None``.

    Silent exactly when the aux is absent; warns (``depth-matte:`` prefix) on
    every other anomaly. Never raises on bad input.
    """
    from PIL import Image

    height, width = int(target_hw[0]), int(target_hw[1])
    try:
        import pillow_heif

        heif = pillow_heif.open_heif(str(heic_path), convert_hdr_to_8bit=False)
        aux = (heif.info or {}).get("aux") or {}
        matte_id = None
        for urn, ids in aux.items():
            if str(urn).endswith(_MATTE_URN_SUFFIX) and ids:
                matte_id = ids[0]
                break
        if matte_id is None:
            return None
        im = heif.get_aux_image(matte_id).to_pillow()
        if im.mode == "I;16":
            arr = np.asarray(im, dtype=np.float32) / 65535.0
        elif im.mode == "L":
            arr = np.asarray(im, dtype=np.float32) / 255.0
        elif im.mode in ("I", "F"):
            raw = np.nan_to_num(np.asarray(im, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
            mx = float(raw.max()) if raw.size else 0.0
            scale = 1.0 if mx <= 1.0 else 255.0 if mx <= 255.0 else 65535.0 if mx <= 65535.0 else mx
            arr = (raw / scale).astype(np.float32)
        else:
            arr = np.asarray(im.convert("L"), dtype=np.float32) / 255.0
        arr = np.clip(np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    except Exception as exc:
        _warn(f"could not decode portrait matte ({exc!r}); LiDAR depth unchanged.")
        return None

    mh, mw = arr.shape
    if abs((mw / mh) / (width / height) - 1.0) > _ASPECT_TOL:
        _warn(f"matte aspect {mw}x{mh} does not match target {width}x{height}; matte skipped.")
        return None
    if (mh, mw) != (height, width):
        pil = Image.fromarray(arr.astype(np.float32), mode="F").resize(
            (width, height), Image.BILINEAR
        )
        arr = np.asarray(pil, dtype=np.float32)
    alpha = np.clip(arr, 0.0, 1.0).astype(np.float32)

    if depth_m is not None:
        d = np.asarray(depth_m)
        if d.shape != (height, width):
            _warn(f"depth shape {d.shape} != target {(height, width)}; matte skipped.")
            return None
        core = alpha > 0.9
        n = int(core.sum())
        if n < _CORE_MIN_PX:
            _warn(f"matte has only {n} core pixels (< {_CORE_MIN_PX}); matte skipped.")
            return None
        dc = d[core]
        frac = float(np.mean(dc <= _AGREE_RATIO * np.median(dc)))
        if frac < _AGREE_FRAC:
            _warn(f"matte/LiDAR agreement {frac:.3f} < {_AGREE_FRAC}; matte skipped.")
            return None
    return alpha


def matte_band(alpha, *, r_px_2048=12, a_lo=0.02, a_hi=0.98):
    """Return ``(soft, band, r)``: soft zone, its dilation, and the radius."""
    a = np.asarray(alpha)
    soft = (a > a_lo) & (a < a_hi)
    r = max(1, round(r_px_2048 * max(a.shape) / 2048))
    band = binary_dilation(soft, iterations=r) if soft.any() else np.zeros_like(soft)
    return soft, band, int(r)


def _weight(alpha: np.ndarray, curve: str) -> np.ndarray:
    a = alpha.astype(np.float64)
    if curve == "linear":
        return a
    if curve == "sqrt":
        return np.sqrt(a)
    return (a >= 0.5).astype(np.float64)


def _report(curve, polarity, r, band, changed, soft_px, hi, lo, near_px, far_px):
    return {
        "changed_px": int(changed.sum()),
        "soft_px": int(soft_px),
        "hi_rewritten": int(hi),
        "lo_rewritten": int(lo),
        "band_px": int(band.sum()),
        "band_frac": float(band.mean()) if band.size else 0.0,
        "r": int(r),
        "near_ref_px": int(near_px),
        "far_ref_px": int(far_px),
        "curve": curve,
        "polarity": polarity,
        "band": band,
        "changed": changed,
    }


def correct_depth_with_matte(
    depth_m,
    alpha,
    *,
    r_px_2048=12,
    a_lo=0.02,
    a_hi=0.98,
    curve="linear",
    _polarity="normal",
):
    """Correct LiDAR depth near the matte's soft zone. Returns ``(depth, report)``."""
    if curve not in _CURVES:
        raise ValueError(f"unknown curve {curve!r}; expected one of {_CURVES}")
    if _polarity not in _POLARITIES:
        raise ValueError(f"unknown _polarity {_polarity!r}; expected one of {_POLARITIES}")
    d = np.asarray(depth_m)
    a = np.asarray(alpha)
    if d.ndim != 2 or d.shape != a.shape:
        raise ValueError(f"depth {d.shape} and alpha {a.shape} must be equal (H,W)")

    soft, band, r = matte_band(a, r_px_2048=r_px_2048, a_lo=a_lo, a_hi=a_hi)
    none = np.zeros(d.shape, dtype=bool)
    if not soft.any():
        return depth_m, _report(curve, _polarity, r, band, none, 0, 0, 0, 0, 0)

    near_ref = (a >= a_hi) & ~band
    far_ref = (a <= a_lo) & ~band
    n_near, n_far = int(near_ref.sum()), int(far_ref.sum())
    if n_near == 0 or n_far == 0:
        _warn("no near or far reference outside the band; LiDAR depth unchanged.")
        return depth_m, _report(curve, _polarity, r, band, none, 0, 0, 0, n_near, n_far)

    disp = 1.0 / d.astype(np.float64)
    idx = distance_transform_edt(~near_ref, return_distances=False, return_indices=True)
    disp_near = disp[idx[0], idx[1]]
    idx = distance_transform_edt(~far_ref, return_distances=False, return_indices=True)
    disp_far = disp[idx[0], idx[1]]

    inv = _polarity == "inverted"
    w = _weight(a, curve)
    if inv:
        w = 1.0 - w
    disp_soft = w * disp_near + (1.0 - w) * disp_far

    ld, ln, lf = np.log(disp), np.log(disp_near), np.log(disp_far)
    closer_far = np.abs(ld - lf) < np.abs(ld - ln)
    closer_near = np.abs(ld - ln) < np.abs(ld - lf)
    hi = band & (a >= a_hi) & closer_far
    lo = band & (a <= a_lo) & closer_near
    soft_b = band & soft
    changed = soft_b | hi | lo

    disp_out = np.where(soft_b, disp_soft, disp)
    disp_out = np.where(hi, disp_far if inv else disp_near, disp_out)
    disp_out = np.where(lo, disp_near if inv else disp_far, disp_out)

    out = d.astype(np.float32, copy=True)
    out[changed] = np.clip(1.0 / disp_out[changed], 0.1, 1e3).astype(np.float32)
    if not np.all(np.isfinite(out)):
        raise ValueError("Gate FAIL: NaN/Inf in matte-corrected depth")
    rep = _report(
        curve, _polarity, r, band, changed, soft_b.sum(), hi.sum(), lo.sum(), n_near, n_far
    )
    return out, rep
