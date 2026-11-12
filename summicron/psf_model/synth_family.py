"""Section 22 S0: synthetic defocus-disc kernel family (leaf module).

Builds a tensor with the SAME schema as ``summicron/_artifacts/codev_render_v1.npz``
where each ``(r, d)`` kernel is a physically sized defocus disc (thin-lens,
same-framing CoC) convolved with the CodeV aberration PSF of that ``(r, d)``
shrunk to scale ``psf_scale``::

    K(r, d) = disc(CoC_d [, rim beta, cat-eye]) (*) S_s[P(r, d')]

Research: ``doc/plan/phase2/section22_blur-magnitude/research.md`` (sections 1,
2, 6). numpy/scipy only; does not import any ``summicron.render`` module. The
file is stored at the tensor's reference scale (``corpus_ref_long_edge`` =
5712) and the renderer's untouched zoom tail shrinks it at render time.

Scale rule: ``CoC_px(L) = L * A_mm * 24 / 36 / 1000 * dp`` (diameter), with
``dp = 1/0.7 - 1/z`` diopters, i.e. ``1.365 * A * dp`` at L = 2048.

D axis: ``axis="geometric11"`` reuses the shipped ``object_distances_m`` (D=11);
``axis="diopter16"`` uses 16 nodes uniform in diopter (a deliberate, documented
departure from the geometric rule: a synthetic family has no CodeV samples to
interpolate, and uniform-diopter nodes make the slab blend uniform in disc
size). ``defocus_ratios = linspace(0, 1, D)`` in both cases, consumed by
``np.interp(depth_m, object_distances_m, defocus_ratios)``.

D-axis exception (S22 S1, research 2.6): the ``diopter16`` axis (16 nodes uniform
in diopter, first node 0.7 m, last 20 m) deliberately departs from the project
rule "D axis is geometric; densify with ``geomspace``, never ``linspace``". That
rule protects the interpolation of CodeV *samples* along z (R-cubic / D-linear in
``log1p``). A synthetic family has no samples to interpolate: its only consumers
are ``np.interp(depth_m, object_distances_m, defocus_ratios)`` and
``compute_slab_weights`` on the meters axis, which need only a strictly
increasing ``object_distances_m``. CoC is linear in diopter offset, so
uniform-diopter nodes make the 2-3 slab blend uniform in disc size (max step
2.2 px at A = 17.5 / 2048; no double rim).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import signal
from scipy.ndimage import zoom as ndimage_zoom

from summicron.psf_model.downscale_tensor import _make_delta

FF_WIDTH_MM = 36.0
IPHONE_EQ_FOCAL_MM = 24.0
FAMILY_NAME = "synthetic_disc_v0"
# Bump on ANY change to the disc / shrink / convolve code: part of the render
# cache key (summicron.render._family), so stale cached families are rebuilt.
BUILDER_VERSION = 1
AXES = ("geometric11", "diopter16")
DEFAULT_SOURCE = Path(__file__).resolve().parents[1] / "_artifacts" / "codev_render_v1.npz"


# ------------------------------------------------------------------ geometry
def coc_diameter_px(aperture_mm: float, dp_diopter, long_edge_px: float):
    """Same-framing thin-lens CoC diameter in px at ``long_edge_px`` (research 1.1)."""
    return (
        np.asarray(dp_diopter, dtype=np.float64)
        * aperture_mm
        * IPHONE_EQ_FOCAL_MM
        * 1e-3
        / FF_WIDTH_MM
        * long_edge_px
    )


def object_distances_for_axis(
    axis: str, source_od: np.ndarray, *, n_d: int | None = None
) -> np.ndarray:
    """D-axis object distances (m). Node 0 is the 0.7 m focus plane."""
    if axis not in AXES:
        raise ValueError(f"axis must be one of {AXES}; got {axis!r}")
    if axis == "geometric11":
        return np.asarray(source_od, dtype=np.float64).copy()
    n = 16 if n_d is None else int(n_d)
    if n < 2:
        raise ValueError("n_d must be >= 2")
    inv = 1.0 / float(source_od[0]) - np.linspace(
        0.0, 1.0 / float(source_od[0]) - 1.0 / float(source_od[-1]), n
    )
    od = 1.0 / inv
    od[0] = float(source_od[0])
    od[-1] = float(source_od[-1])
    return od


def disc_kernel(
    diam_px: float,
    size: int,
    *,
    ss: int = 8,
    cat_eye: float = 0.0,
    rim_beta: float = 0.0,
    rim_rho0: float = 0.6,
    rim_pow: float = 2.0,
) -> np.ndarray:
    """Anti-aliased (ss x ss coverage) disc, L1-normalised float32, centred at size//2.

    ``cat_eye`` s > 0: intersection of two circles of the disc radius whose
    centres are offset by +-s*R/2 along y (lens shape, centroid stays central).
    ``rim_beta``: ``1 + beta*((rho-rho0)/(1-rho0))^pow`` for rho > rho0.
    """
    R = max(diam_px / 2.0, 0.5)
    n = size * ss
    c = (size - 1) / 2.0
    ys = (np.arange(n) + 0.5) / ss - 0.5 - c
    yy, xx = np.meshgrid(ys, ys, indexing="ij")
    rho = np.hypot(yy, xx) / R
    if cat_eye > 0:
        h = 0.5 * cat_eye * R
        m = ((np.hypot(yy - h, xx) / R <= 1.0) & (np.hypot(yy + h, xx) / R <= 1.0)).astype(
            np.float64
        )
    else:
        m = (rho <= 1.0).astype(np.float64)
    if rim_beta > 0:
        prof = 1.0 + rim_beta * np.clip((rho - rim_rho0) / (1.0 - rim_rho0), 0, 1) ** rim_pow
        m *= prof
    k = m.reshape(size, ss, size, ss).mean(axis=(1, 3))
    return (k / max(k.sum(), 1e-12)).astype(np.float32)


def shrink_psf(k: np.ndarray, factor: float) -> np.ndarray:
    """Scale a PSF's spatial extent by ``factor`` (<1 = smaller), L1-renormed, same box."""
    k64 = np.maximum(k.astype(np.float64), 0.0)
    if factor >= 0.999:
        return k64 / k64.sum()
    kz = np.maximum(ndimage_zoom(k64, factor, order=3), 0.0)
    out = np.zeros_like(k64)
    zh, zw = kz.shape
    H, W = k64.shape
    oy, ox = (H - zh) // 2, (W - zw) // 2
    out[oy : oy + zh, ox : ox + zw] = kz
    return out / out.sum()


def _convolve_same(a: np.ndarray, b: np.ndarray, size: int) -> np.ndarray:
    full = signal.fftconvolve(a.astype(np.float64), b.astype(np.float64), mode="full")
    cy, cx = full.shape[0] // 2, full.shape[1] // 2
    h = size // 2
    out = np.maximum(full[cy - h : cy - h + size, cx - h : cx - h + size], 0.0)
    return out / out.sum()


# ------------------------------------------------------------------ metrics
def render_scale_kernel(k_ref: np.ndarray, pix_scale: float, ksize: int) -> np.ndarray:
    """Replicates the renderer's zoom tail (order-3, clip>=0, centre crop, L1)."""
    kz = np.maximum(ndimage_zoom(k_ref.astype(np.float32), pix_scale, order=3), 0.0)
    zh, zw = kz.shape
    out = np.zeros((ksize, ksize), np.float32)
    cy, cx = zh // 2, zw // 2
    sy, sx = max(0, cy - ksize // 2), max(0, cx - ksize // 2)
    ey, ex = min(zh, sy + ksize), min(zw, sx + ksize)
    dy, dx = (ksize - (ey - sy)) // 2, (ksize - (ex - sx)) // 2
    out[dy : dy + (ey - sy), dx : dx + (ex - sx)] = kz[sy:ey, sx:ex]
    out /= max(float(out.sum()), 1e-12)
    return out


def crisp_metrics(k: np.ndarray) -> dict:
    """Crispness metrics of an L1 kernel (research 2.4): r50/r90, flatness,
    rim contrast, 10-90 edge width (px), haze (energy beyond 1.25*r90)."""
    k = np.maximum(k, 0.0).astype(np.float64)
    k = k / k.sum()
    H, W = k.shape
    yy, xx = np.mgrid[0:H, 0:W]
    cy, cx = (k * yy).sum(), (k * xx).sum()
    rr = np.hypot(yy - cy, xx - cx)
    order = np.argsort(rr.ravel())
    cum = np.cumsum(k.ravel()[order])
    rs = rr.ravel()[order]
    r50 = float(rs[np.searchsorted(cum, 0.5)])
    r90 = float(rs[np.searchsorted(cum, 0.9)])
    peak = float(k.max())
    rb = np.floor(rr).astype(int)
    prof = np.bincount(rb.ravel(), weights=k.ravel()) / np.maximum(np.bincount(rb.ravel()), 1)
    if r90 >= 2:
        core = prof[: max(1, int(0.5 * r90))].mean()
        lo_i = int(0.8 * r90)
        rim = prof[lo_i : max(lo_i + 1, int(r90) + 1)].mean()
    else:
        core = rim = prof[0]
    p = prof / max(prof.max(), 1e-12)
    hi = np.where(p >= 0.9)[0]
    i90 = int(hi.max()) if len(hi) else 0
    lo = np.where(p[i90:] <= 0.1)[0]
    i10 = int(i90 + lo.min()) if len(lo) else len(p) - 1
    r_eq = r90 / np.sqrt(0.9)
    peak_eq = 1.0 / (np.pi * max(r_eq, 1e-9) ** 2)
    return dict(
        r50=r50,
        r90=r90,
        peak=peak,
        flatness=float(peak_eq / peak),
        rim_contrast=float(rim / max(core, 1e-12)),
        edge_w=float(i10 - i90),
        haze=float(k[rr > 1.25 * r90].sum()),
        prof=prof,
    )


# ------------------------------------------------------------------ builder
def _load_source(path) -> dict:
    d = np.load(str(path), allow_pickle=False)
    return {k: d[k] for k in d.files}


def _passthrough_meta(src: dict, n_d: int) -> dict:
    """Metadata keys of the shipped schema, resized to the new D."""
    R = src["kernels"].shape[0]
    out = {
        "radial_edges": src["radial_edges"],
        "pixel_pitch_um": np.repeat(src["pixel_pitch_um"][:, :1], n_d, axis=1),
        "full_frame_dim_px": np.repeat(src["full_frame_dim_px"][:, :1, :], n_d, axis=1),
        "h_max_mm": src["h_max_mm"],
        "wavelength_nm": src["wavelength_nm"],
        "corpus_ref_long_edge": src["corpus_ref_long_edge"],
        "otf_norm_applied": src["otf_norm_applied"],
        "delta_guard_applied": np.array(True),
    }
    assert out["pixel_pitch_um"].shape == (R, n_d)
    return out


def build_synthetic_tensor(
    source_tensor_path=None,
    *,
    aperture_mm: float = 17.5,
    n_d: int | None = None,
    axis: str = "diopter16",
    psf_scale: float = 0.25,
    rim_beta: float = 0.3,
    cat_eye: float = 0.0,
    out_path=None,
) -> dict:
    """Build the synthetic disc family. Returns the npz payload dict; writes it
    to ``out_path`` when given. ``cat_eye`` is the kappa of ``s_r = kappa*r_center``
    (orientation sign unverified, default 0 = round)."""
    if aperture_mm <= 0:
        raise ValueError("aperture_mm must be > 0")
    if psf_scale <= 0:
        raise ValueError("psf_scale must be > 0")
    src = _load_source(source_tensor_path or DEFAULT_SOURCE)
    K = src["kernels"]
    R, _d_src, size, _ = K.shape
    od_src = src["object_distances_m"].astype(np.float64)
    focus = float(src["focus_distance_m"])
    ref_edge = int(src["corpus_ref_long_edge"])

    od = object_distances_for_axis(axis, od_src, n_d=n_d)
    n = len(od)
    dp = 1.0 / focus - 1.0 / od
    dp[0] = 0.0
    dp_src = 1.0 / focus - 1.0 / od_src
    coc = coc_diameter_px(aperture_mm, dp, ref_edge)
    edges = src["radial_edges"]
    r_center = 0.5 * (edges[:-1] + edges[1:])

    out = np.zeros((R, n, size, size), np.float32)
    for r in range(R):
        s_r = cat_eye * float(r_center[r])
        for d in range(n):
            if d == 0:
                out[r, d] = _make_delta(size)
                continue
            d_src = max(1, int(np.argmin(np.abs(dp_src - dp[d]))))
            psf = shrink_psf(K[r, d_src], psf_scale)
            disc = disc_kernel(float(coc[d]), size, cat_eye=s_r, rim_beta=rim_beta)
            out[r, d] = _convolve_same(disc, psf, size).astype(np.float32)
            # float32 round-off: renormalise so the float32 sum is 1 to ~1e-7
            out[r, d] /= np.float32(out[r, d].sum(dtype=np.float64))

    payload = {
        "kernels": out,
        "object_distances_m": od,
        "focus_distance_m": np.array(focus),
        "defocus_ratios": np.linspace(0.0, 1.0, n),
        **_passthrough_meta(src, n),
        "synth_family": np.array(FAMILY_NAME),
        "synth_builder_version": np.array(BUILDER_VERSION),
        "synth_axis": np.array(axis),
        "synth_aperture_mm": np.array(float(aperture_mm)),
        "synth_psf_scale": np.array(float(psf_scale)),
        "synth_rim_beta": np.array(float(rim_beta)),
        "synth_cat_eye": np.array(float(cat_eye)),
        "synth_coc_px_ref": coc.astype(np.float64),
    }
    if out_path is not None:
        _save(payload, out_path)
    return payload


def build_zoom_control(
    source_tensor_path=None, *, zoom: float = 2.0, out_path=None
) -> dict:
    """Negative control: the shipped tensor with every d>=1 kernel zoomed by
    ``zoom`` (order-3, clip, centre crop, L1). Node 0 stays an exact delta."""
    src = _load_source(source_tensor_path or DEFAULT_SOURCE)
    K = src["kernels"]
    R, D, size, _ = K.shape
    out = np.zeros_like(K)
    for r in range(R):
        for d in range(D):
            if d == 0:
                out[r, d] = _make_delta(size)
                continue
            kz = np.maximum(ndimage_zoom(K[r, d].astype(np.float64), zoom, order=3), 0.0)
            o = (kz.shape[0] - size) // 2
            c = kz[o : o + size, o : o + size]
            out[r, d] = (c / c.sum()).astype(np.float32)
    payload = {k: v for k, v in src.items() if k != "kernels"}
    payload["kernels"] = out
    payload["synth_family"] = np.array("zoom_control")
    payload["synth_zoom"] = np.array(float(zoom))
    if out_path is not None:
        _save(payload, out_path)
    return payload


def _save(payload: dict, out_path) -> None:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(p, **payload)


# ------------------------------------------------------------------ gates
EDGE_QUANT_SLACK_PX = 1.0  # edge width is measured on 1-px azimuthal bins (+-1 px resolution)
GATE_MIN_COC_PX = 12.0  # discs smaller than this (at 2048) are pixel-quantised; metrics reported, not gated


def gate_table(kernels: np.ndarray, long_edge_px: int = 2048, ref_edge_px: int = 5712) -> dict:
    """G-crisp / G-haze / G-energy numbers for every (r, d >= 1) kernel, measured
    after the renderer's zoom tail at ``long_edge_px``.

    Returns arrays ``flatness, rim, edge_w, haze, r90, coc_eff`` of shape (R, D)
    (node 0 left NaN), a boolean ``gated`` mask (coc_eff >= GATE_MIN_COC_PX),
    ``crisp_ok`` (per-kernel pass of flatness >= 0.85, rim >= 0.8,
    edge <= max(3, 0.12*CoC) + 1 px bin slack, haze <= 0.003; NaN-safe False where not gated) and
    the L1 deviation ``l1_err``.
    """
    R, D, kh, _ = kernels.shape
    ps = long_edge_px / ref_edge_px
    ksize = max(3, int(round(kh * ps)) | 1)
    names = ("flatness", "rim", "edge_w", "haze", "r90")
    t = {n: np.full((R, D), np.nan) for n in names}
    for r in range(R):
        for d in range(1, D):
            m = crisp_metrics(render_scale_kernel(kernels[r, d], ps, ksize))
            t["flatness"][r, d] = m["flatness"]
            t["rim"][r, d] = m["rim_contrast"]
            t["edge_w"][r, d] = m["edge_w"]
            t["haze"][r, d] = m["haze"]
            t["r90"][r, d] = m["r90"]
    coc_eff = 2.0 * t["r90"] / np.sqrt(0.9)  # flat-disc equivalent diameter
    gated = np.nan_to_num(coc_eff, nan=0.0) >= GATE_MIN_COC_PX
    with np.errstate(invalid="ignore"):
        ok = (
            (t["flatness"] >= 0.85)
            & (t["rim"] >= 0.8)
            & (t["edge_w"] <= np.maximum(3.0, 0.12 * coc_eff) + EDGE_QUANT_SLACK_PX)
            & (t["haze"] <= 0.003)
        )
    t["coc_eff"] = coc_eff
    t["gated"] = gated
    t["crisp_ok"] = ok & gated
    t["l1_err"] = np.abs(kernels.astype(np.float64).sum(axis=(2, 3)) - 1.0)
    return t
