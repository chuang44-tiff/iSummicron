"""Test-local copy of the S33 tolerance metric (Tier B / C ceilings).

Same arithmetic as ``scripts/s33/gate_tolerance.py`` (the frozen gate): with
``e = |d| / max(ref, 1)`` per element in the linear domain, the full frame and
the field-edge annulus (radial_frac > 0.8) are each judged on max e, p99.9 of
e, energy ratio and u8 max |d|; Tier C adds PSNR >= 70 dB (peak 1.0).

This copy exists so the public FAST suite can judge an engine without
``scripts/s33`` or ``runs/``. ``tests/unit/test_s33_tiled_core.py`` cross-checks
it against the frozen gate when that is present. Ceilings may not be loosened.
"""
from __future__ import annotations

import math

import numpy as np

TIERS = {
    "B": dict(max_e=1e-5, p999_e=1e-6, energy=1e-5, u8=1, psnr=None),
    "C": dict(max_e=1e-4, p999_e=1e-5, energy=1e-5, u8=1, psnr=70.0),
}
ANNULUS_MIN = 0.8


def annulus_mask(h: int, w: int) -> np.ndarray:
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    y = (np.arange(h, dtype=np.float64) - cy)[:, None]
    x = (np.arange(w, dtype=np.float64) - cx)[None, :]
    hd = math.hypot(cy, cx)
    if hd == 0:
        return np.zeros((h, w), dtype=bool)
    return np.hypot(y, x) / hd > ANNULUS_MIN


def to_u8(linear: np.ndarray) -> np.ndarray:
    """A shared stand-in tonemap (clip + sRGB OETF + round) for u8 checks."""
    from summicron.render.convolve import linear_to_srgb

    s = linear_to_srgb(np.clip(linear, 0.0, 1.0).astype(np.float32))
    return np.clip(np.rint(s * 255.0), 0, 255).astype(np.uint8)


def region_metrics(ref, test, ref8, test8, mask):
    if mask is not None:
        ref, test, ref8, test8 = ref[mask], test[mask], ref8[mask], test8[mask]
    ref64 = ref.astype(np.float64)
    d = np.abs(test.astype(np.float64) - ref64)
    e = d / np.maximum(ref64, 1.0)
    se = float(ref64.sum())
    mse = float((d ** 2).mean())
    return dict(
        finite=bool(np.isfinite(test).all()),
        max_e=float(e.max()),
        p999_e=float(np.percentile(e, 99.9)),
        energy=abs(float(test.astype(np.float64).sum()) / se - 1.0) if se else float("inf"),
        psnr=float("inf") if mse == 0 else 10 * math.log10(1.0 / mse),
        u8=int(np.abs(test8.astype(np.int16) - ref8.astype(np.int16)).max()),
    )


def evaluate(ref, test, tier="B", ref8=None, test8=None):
    """``{'full'|'edge': {metric: (value, ceiling, ok)}, 'ok': bool}``."""
    lim = TIERS[tier]
    ref = np.asarray(ref)
    test = np.asarray(test)
    if ref.shape != test.shape:
        return dict(ok=False, error=f"shape {test.shape} != {ref.shape}")
    if not np.isfinite(test).all():
        return dict(ok=False, error="non-finite values in the linear output")
    if ref8 is None:
        ref8 = to_u8(ref)
    if test8 is None:
        test8 = to_u8(test)
    m = annulus_mask(*ref.shape[:2])
    if not m.any():
        return dict(ok=False, error="field-edge annulus is empty for this frame size")
    out, ok = {}, True
    for reg, mask in (("full", None), ("edge", m)):
        r = region_metrics(ref, test, ref8, test8, mask)
        row = {"finite": (r["finite"], True, r["finite"])}
        for k in ("max_e", "p999_e", "energy", "u8"):
            row[k] = (r[k], lim[k], r[k] <= lim[k])
        row["psnr"] = (r["psnr"], lim["psnr"], lim["psnr"] is None or r["psnr"] >= lim["psnr"])
        out[reg] = row
        ok &= all(v[2] for v in row.values())
    out["ok"] = bool(ok)
    return out


def local_max_e(ref, test, centers, half=3):
    """Max e in a (2*half+1)^2 box around each (y, x) light centre (report)."""
    e = np.abs(test.astype(np.float64) - ref.astype(np.float64)) / np.maximum(
        ref.astype(np.float64), 1.0
    )
    h, w = e.shape[:2]
    res = []
    for (cy, cx) in centers:
        y0, y1 = max(0, cy - half), min(h, cy + half + 1)
        x0, x1 = max(0, cx - half), min(w, cx + half + 1)
        res.append(float(e[y0:y1, x0:x1].max()))
    return res


def summary(res) -> str:
    if "error" in res:
        return f"FAIL {res['error']}"
    parts = []
    for reg in ("full", "edge"):
        r = res[reg]
        parts.append(
            f"{reg}: " + " ".join(
                f"{k}={r[k][0]:.3g}{'' if r[k][2] else '!'}"
                for k in ("max_e", "p999_e", "energy", "u8")
            )
        )
    return "; ".join(parts) + (" PASS" if res["ok"] else " FAIL")
