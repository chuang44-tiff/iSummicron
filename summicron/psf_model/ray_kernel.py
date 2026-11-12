"""Ray-trace spot diagram -> render kernel helpers (Section 24, numpy only).

Convention (verified end to end by ``tests/unit/test_raytrace_orientation.py``):
the renderer's meridian kernel has its OUTWARD direction toward decreasing row
(up). ``OUTWARD_ROW_SIGN`` encodes this: outward mass sits at
``row = centre + OUTWARD_ROW_SIGN * d_out / pitch``. Columns are tangential:
``col = centre + d_tan / pitch``.

This module must not import optiland; the tracer scripts feed it plain arrays.
"""
from __future__ import annotations

import numpy as np

OUTWARD_ROW_SIGN = -1
KERNEL_SIZE = 183
SCHEMA_VERSION = "1.0"

# Tensor axes (schema constants; inlined, bit-identical to the shipped artifacts).
OBJECT_DISTANCES_M = np.array([
    0.7, 0.9787938582686772, 1.3686248814064048, 1.9137166117061226,
    2.6759058085781366, 3.741657386773941, 5.231873242742804,
    7.315607710338405, 10.2292455662603, 14.303318192825259, 20.0,
], dtype=np.float64)
DEFOCUS_RATIOS = np.linspace(0, 1, 11)
# Equal-area annulus edges on the normalised field radius (R = 12 bins).
RADIAL_EDGES = np.array([
    0.0, 0.15075567228888181, 0.30151134457776363, 0.4264014327112209,
    0.5222329678670935, 0.6030226891555273, 0.674199862463242,
    0.7385489458759964, 0.7977240352174656, 0.8528028654224418,
    0.9045340337332909, 0.9534625892455924, 1.0,
], dtype=np.float64)


def make_delta(size: int = KERNEL_SIZE) -> np.ndarray:
    """Exact centred single-pixel delta, float64, L1 = 1."""
    k = np.zeros((size, size), dtype=np.float64)
    k[size // 2, size // 2] = 1.0
    return k


def rays_to_kernel(d_out_mm, d_tan_mm, pitch_mm: float, size: int = KERNEL_SIZE,
                   return_outside: bool = False):
    """Bilinear-splat weight-1 rays into a ``(size, size)`` float64 kernel.

    ``d_out_mm``: radial (outward-positive) displacement from the centring
    reference; ``d_tan_mm``: tangential displacement. Sub-pixel weight that
    would land outside the box is dropped and counted: the returned kernel is
    renormalised to L1 = 1 over the in-box energy, and ``outside`` (the
    dropped fraction of total ray weight, in [0, 1]) is returned when
    ``return_outside=True``. No blur is applied.
    """
    d_out = np.asarray(d_out_mm, dtype=np.float64).ravel()
    d_tan = np.asarray(d_tan_mm, dtype=np.float64).ravel()
    if d_out.shape != d_tan.shape:
        raise ValueError("d_out_mm and d_tan_mm must have the same length")
    if not (pitch_mm > 0 and np.isfinite(pitch_mm)):
        raise ValueError("pitch_mm must be positive finite")
    if size < 3 or size % 2 == 0:
        raise ValueError("size must be odd and >= 3")
    n = d_out.size
    if n == 0:
        raise ValueError("no rays")
    if not (np.all(np.isfinite(d_out)) and np.all(np.isfinite(d_tan))):
        raise ValueError("non-finite ray displacement")
    c = size // 2
    row = c + OUTWARD_ROW_SIGN * d_out / pitch_mm
    col = c + d_tan / pitch_mm
    r0 = np.floor(row)
    c0 = np.floor(col)
    fr = row - r0
    fc = col - c0
    r0 = r0.astype(np.int64)
    c0 = c0.astype(np.int64)
    acc = np.zeros(size * size, dtype=np.float64)
    inside_w = 0.0
    for dr, wr in ((0, 1.0 - fr), (1, fr)):
        for dc, wc in ((0, 1.0 - fc), (1, fc)):
            rr = r0 + dr
            cc = c0 + dc
            w = wr * wc
            ok = (rr >= 0) & (rr < size) & (cc >= 0) & (cc < size)
            if ok.any():
                np.add.at(acc, rr[ok] * size + cc[ok], w[ok])
                inside_w += float(w[ok].sum())
    outside = max(0.0, 1.0 - inside_w / n)
    if inside_w <= 0:
        raise ValueError("all ray energy fell outside the kernel box")
    k = (acc / acc.sum()).reshape(size, size)
    return (k, outside) if return_outside else k


def shape_metrics(kernel) -> tuple[float, float]:
    """``(core_aspect, tail)`` of a kernel.

    core_aspect = sqrt(I_rr / I_tt) over pixels inside the 90 %-energy radius
    about the centroid (rr along axis 0 = radial, tt along axis 1).
    tail = third standardised moment along axis 0 over all energy, sign
    converted so that positive = OUTWARD (per ``OUTWARD_ROW_SIGN``).
    """
    k = np.asarray(kernel, dtype=np.float64)
    if k.ndim != 2:
        raise ValueError("kernel must be 2-D")
    total = k.sum()
    if not total > 0:
        raise ValueError("kernel has no energy")
    w = k / total
    ys, xs = np.indices(k.shape, dtype=np.float64)
    cy = float((w * ys).sum())
    cx = float((w * xs).sum())
    dy = ys - cy
    dx = xs - cx
    dist = np.hypot(dy, dx)
    order = np.argsort(dist.ravel(), kind="stable")
    cum = np.cumsum(w.ravel()[order])
    idx = min(int(np.searchsorted(cum, 0.9)), cum.size - 1)
    r90 = dist.ravel()[order][idx]
    m = dist <= r90
    irr = float((w[m] * dy[m] ** 2).sum())
    itt = float((w[m] * dx[m] ** 2).sum())
    aspect = float(np.sqrt(irr / itt)) if itt > 0 else float("inf")
    var = float((w * dy ** 2).sum())
    if var > 0:
        skew = float((w * dy ** 3).sum()) / var ** 1.5
    else:
        skew = 0.0
    return aspect, float(OUTWARD_ROW_SIGN * skew)


def assemble_tensor(kernels, meta: dict | None = None) -> dict:
    """Pack ``(R, D, size, size)`` kernels + metadata into an npz-ready dict.

    d0 is forced to an exact centred delta (``delta_guard_applied=True``,
    always: a caller meta value is discarded). ``kernels`` in meta raises. An all-zero
    d >= 1 kernel raises ValueError naming (r, d).
    ``object_distances_m``, ``defocus_ratios`` and ``radial_edges`` default to
    the module constants above. Every other ``meta`` entry (e.g.
    ``transmission``, ``fu``, ``lens_citation``, ``prescription_variant``) is
    stored as a numpy array. Values are float32 kernels + numpy scalars so
    ``np.savez_compressed`` -> ``convolve.load_tensor`` works (allow_pickle=False).
    """
    meta = dict(meta or {})
    reserved = sorted({"kernels"} & set(meta))
    if reserved:
        raise ValueError(f"meta keys are schema-reserved: {reserved}")
    k = np.asarray(kernels)
    if k.ndim != 4 or k.shape[2] != k.shape[3] or k.shape[2] % 2 == 0:
        raise ValueError(f"kernels must be (R, D, n, n) with odd n; got {k.shape}")
    R, D, size, _ = k.shape
    if not np.all(np.isfinite(k)):
        raise ValueError("kernels contain non-finite values")
    if np.any(k < 0):
        raise ValueError("kernels contain negative values")
    k32 = k.astype(np.float32).copy()
    k32[:, 0] = make_delta(size).astype(np.float32)
    sums = k32[:, 1:].sum(axis=(2, 3), dtype=np.float64)
    if not np.all(sums > 0):
        r_, d_ = np.argwhere(~(sums > 0))[0]
        raise ValueError(f"kernel (r={r_}, d={d_ + 1}) has no energy (all zero); cannot L1-normalise")
    k32[:, 1:] /= k32[:, 1:].sum(axis=(2, 3), keepdims=True, dtype=np.float64).astype(np.float32)

    defaults = {
        "object_distances_m": OBJECT_DISTANCES_M,
        "defocus_ratios": DEFOCUS_RATIOS,
        "radial_edges": RADIAL_EDGES,
    }
    out = {"kernels": k32}
    for key, dflt in defaults.items():
        out[key] = np.asarray(meta.pop(key, dflt), dtype=np.float64)
    if out["radial_edges"].shape != (R + 1,):
        raise ValueError(f"radial_edges length {out['radial_edges'].shape} != R+1={R + 1}")
    if out["object_distances_m"].shape != (D,) or out["defocus_ratios"].shape != (D,):
        raise ValueError(f"object_distances_m/defocus_ratios must have length D={D}")

    fixed = {
        "schema_version": np.array(SCHEMA_VERSION),
        "corpus_ref_long_edge": np.array(5712, dtype=np.int64),
        "focus_distance_m": np.array(0.7),
        "wavelength_nm": np.array(550.0),
        "otf_norm_applied": np.array(False),
        "delta_guard_applied": np.array(True),
    }
    # Forced schema values win: a caller value for delta_guard_applied is discarded
    # (d0 is always forced to a delta, so it is always True).
    meta.pop("delta_guard_applied", None)
    for key, v in fixed.items():
        out[key] = v if key == "delta_guard_applied" else np.asarray(meta.pop(key, v))
    for key, v in meta.items():
        if v is None:
            continue
        out[key] = np.asarray(v)
    return out


def save_tensor(path, tensor_dict: dict) -> None:
    """``np.savez_compressed`` (lossless) the dict from :func:`assemble_tensor` (loadable by ``load_tensor``)."""
    np.savez_compressed(str(path), **tensor_dict)
