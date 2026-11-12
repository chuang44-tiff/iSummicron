"""Measured input-tone LUT for ``--tonemap input-lut`` (S1 tone fix G).

The iPhone Portrait HEIC carries an SDR base and an HDR gain map. The pair
(``expandToHDR`` image, base SDR image) *is* the Apple tone mapping; this module
fits a monotone luma LUT ``H -> S`` from that pair and applies it to the render
buffer, so the render reproduces the input's OWN SDR tone. No curve is authored.

numpy only (no Quartz); importable on any platform.
Spec: ``doc/plan/phase2/section20_depth-diopter/locked-spec-S1-plumbing-tone.md`` s4.
"""
from __future__ import annotations

import warnings

import numpy as np

LUMA_W = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)

_N_BINS = 64
_LO_EDGE = 1e-3
_SDR_FLOOR = 0.02
_MIN_PIX = 200

# Auto bright-pass threshold: thr = max(1.0, 1.1 * H(0.9)).
_AUTO_Y = 0.9
_AUTO_MARGIN = 1.1


def _identity() -> tuple[np.ndarray, np.ndarray]:
    return np.array([0.0, 1.0]), np.array([0.0, 1.0])


def _srgb_encode(linear: np.ndarray) -> np.ndarray:
    """sRGB OETF; same formula as ``convolve.linear_to_srgb`` (kept local so this
    module stays numpy-only and import-light)."""
    linear = np.asarray(linear, dtype=np.float32)
    return np.where(
        linear <= 0.0031308,
        12.92 * linear,
        1.055 * np.power(np.maximum(linear, 0.0), 1.0 / 2.4) - 0.055,
    ).astype(np.float32)


def fit_input_lut(hdr_lin: np.ndarray, sdr_lin: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit the monotone luma LUT ``(xs, ys)`` mapping HDR linear luma to SDR linear luma.

    Both inputs are ``(H, W, 3)`` linear, same grid. Knots are the geometric bin
    centres of 64 geometric bins over ``[1e-3, 1.01 * hdr.max()]`` (bins with
    ``<= 200`` pixels whose SDR luma is ``> 0.02`` are dropped), ordinate the
    bin median of the SDR luma. ``(0, 0)`` is prepended and ``ys`` is made
    monotone. Pure and deterministic.
    """
    hdr = np.asarray(hdr_lin)
    sdr = np.asarray(sdr_lin)
    if hdr.shape != sdr.shape:
        raise ValueError(
            f"fit_input_lut: shape mismatch hdr {hdr.shape} vs sdr {sdr.shape}"
        )
    if not np.isfinite(hdr).all() or not np.isfinite(sdr).all():
        raise ValueError("fit_input_lut: non-finite values in hdr/sdr input")
    hmax = float(hdr.max()) if hdr.size else 0.0
    if hmax <= 1e-3:
        warnings.warn(
            "fit_input_lut: HDR image is (near-)black; using identity LUT",
            RuntimeWarning,
            stacklevel=2,
        )
        return _identity()
    hy = hdr.astype(np.float32) @ LUMA_W
    sy = sdr.astype(np.float32) @ LUMA_W
    edges = np.geomspace(_LO_EDGE, hmax * 1.01, _N_BINS + 1)
    idx = np.digitize(hy, edges)
    xs = [0.0]
    ys = [0.0]
    for i in range(1, len(edges)):
        s = (idx == i) & (sy > _SDR_FLOOR)
        if s.sum() > _MIN_PIX:
            xs.append(float(np.sqrt(edges[i - 1] * edges[i])))
            ys.append(float(np.median(sy[s])))
    if len(xs) < 3:
        warnings.warn(
            "fit_input_lut: fewer than 3 LUT knots survived; using identity LUT",
            RuntimeWarning,
            stacklevel=2,
        )
        return _identity()
    return np.array(xs), np.maximum.accumulate(np.array(ys))


def _identity_sat() -> tuple[np.ndarray, np.ndarray]:
    return np.array([_LO_EDGE, 1.0]), np.array([1.0, 1.0])


def fit_input_sat(hdr_lin: np.ndarray, sdr_lin: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit the per-luma-bin saturation scale ``(kx, kk)`` (rev 3, C3).

    Apple's HDR->SDR mapping is not luma-only: the SDR base is ~15% less
    saturated than the HDR expansion. Per luma bin (same bins/rules as
    :func:`fit_input_lut`, binned by HDR luma) fit ``r' - 1 = k (r - 1)`` on the
    unit-luma ratios ``r = rgb / Y`` by least squares:
    ``k = sum(r * r') / sum(r * r)`` (``r``, ``r'`` already minus 1), clamped to
    ``[0, 1.2]``. Bins with ``<= 200`` px, SDR luma ``<= 0.02`` or no chroma
    (``sum r*r <= 1e-9``) are skipped. Fewer than 2 knots or no HDR headroom
    (``hdr.max() <= 1.001``) -> identity ``k = 1`` (no warning).
    """
    hdr = np.asarray(hdr_lin)
    sdr = np.asarray(sdr_lin)
    if hdr.shape != sdr.shape:
        raise ValueError(
            f"fit_input_sat: shape mismatch hdr {hdr.shape} vs sdr {sdr.shape}"
        )
    if not np.isfinite(hdr).all() or not np.isfinite(sdr).all():
        raise ValueError("fit_input_sat: non-finite values in hdr/sdr input")
    hmax = float(hdr.max()) if hdr.size else 0.0
    if hmax <= 1.001:
        return _identity_sat()
    hy = hdr.astype(np.float32) @ LUMA_W
    sy = sdr.astype(np.float32) @ LUMA_W
    edges = np.geomspace(_LO_EDGE, hmax * 1.01, _N_BINS + 1)
    idx = np.digitize(hy, edges)
    kx: list[float] = []
    kk: list[float] = []
    for i in range(1, len(edges)):
        s = (idx == i) & (sy > _SDR_FLOOR)
        if s.sum() <= _MIN_PIX:
            continue
        r = hdr[s].astype(np.float64) / np.maximum(hy[s], 1e-6).astype(np.float64)[:, None] - 1.0
        rp = sdr[s].astype(np.float64) / sy[s].astype(np.float64)[:, None] - 1.0
        den = float((r * r).sum())
        if den <= 1e-9:
            continue
        kx.append(float(np.sqrt(edges[i - 1] * edges[i])))
        kk.append(float(np.clip((r * rp).sum() / den, 0.0, 1.2)))
    if len(kx) < 2:
        return _identity_sat()
    return np.array(kx), np.array(kk)


def fit_input_tone(
    hdr_lin: np.ndarray, sdr_lin: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convenience: ``(xs, ys, kx, kk)`` = luma LUT + saturation scale."""
    xs, ys = fit_input_lut(hdr_lin, sdr_lin)
    kx, kk = fit_input_sat(hdr_lin, sdr_lin)
    return xs, ys, kx, kk


def auto_bright_threshold(xs: np.ndarray, ys: np.ndarray) -> tuple[float, float]:
    """Return ``(thr, h09)`` with ``thr = max(1.0, 1.1 * H(0.9))``.

    ``H(0.9)`` is the HDR luma at which the LUT reaches SDR 0.9 (interpolated
    crossing; first index of tied ``ys``). If the LUT never reaches 0.9,
    ``H = xs[-1]`` and ``thr = 1.1 * xs[-1]`` (bright pass effectively off).
    """
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    u, ix = np.unique(ys, return_index=True)
    if u.max() < _AUTO_Y:
        h = float(xs[-1])
        return float(_AUTO_MARGIN * h), h
    h = float(np.interp(_AUTO_Y, u, xs[ix]))
    return float(max(1.0, _AUTO_MARGIN * h)), h


def apply_input_lut(
    rgb_lin: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    sat: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    """Hue-preserving luma LUT -> sRGB ``uint8`` ``(H, W, 3)``.

    ``Y = luma(rgb)``; ``rgb *= LUT(Y) / max(Y, 1e-6)``; per-pixel divide by
    ``max(1, max_c)``; clip to [0, 1]; sRGB encode; same u8 cast as the LDR
    finalize path. Negatives clamp to 0 (FFT ringing); NaN -> 0, +Inf -> large
    finite so the result is always finite.

    ``sat=(kx, kk)`` (from :func:`fit_input_sat`) adds the rev 3 chroma
    correction: unit-luma ratios ``r = rgb / Y`` become
    ``max(1 + k(Y) * (r - 1), 0)`` (``k`` interpolated in log Y), the colour is
    scaled to ``LUT(Y)`` and its luma restored per pixel. ``sat=None`` is the
    exact luma-only behaviour.
    """
    rgb = np.asarray(rgb_lin, dtype=np.float32)
    rgb = np.nan_to_num(rgb, nan=0.0, posinf=1e6, neginf=0.0)
    rgb = np.maximum(rgb, 0.0)
    y = rgb @ LUMA_W
    fy = np.interp(y, xs, ys).astype(np.float32)
    if sat is None:
        scale = fy / np.maximum(y, 1e-6)
        out = rgb * scale[..., None]
    else:
        kx, kk = sat
        ly = np.log(np.maximum(y, 1e-6))
        k = np.interp(ly, np.log(np.asarray(kx, np.float64)), kk).astype(np.float32)
        r = rgb / np.maximum(y, 1e-6)[..., None]
        out = fy[..., None] * np.maximum(1.0 + k[..., None] * (r - 1.0), 0.0)
        out = out * (fy / np.maximum(out @ LUMA_W, 1e-6))[..., None]
    out = out / np.maximum(out.max(-1, keepdims=True), 1.0)
    out = _srgb_encode(np.clip(out, 0.0, 1.0))
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)
