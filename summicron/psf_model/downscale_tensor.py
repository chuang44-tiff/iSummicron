"""Downscale a render-ready PSF tensor to match a target image long-edge.

Instead of taking an arbitrary kernel-grid size, this utility takes a
**target image long-edge in pixels** (i.e. the working resolution of
the renderer) and computes the corresponding kernel grid so the kernel's
pixel pitch matches the image's pixel pitch at that resolution.

The result: ONE saved tensor sized to the iPhone native long edge
(default 5712 px -> ~183 px kernels). At render time, the renderer
scales further DOWN for smaller working resolutions via its existing
pix_scale path. We never upscale.

Per P1.9 (`doc/plan/phase1/section9/locked-spec-p1.9.md`): after cubic-zoom
+ L1 renorm, install an exact target-grid single-pixel delta at every
`(r, d=0)` slice (the S2 delta guard at build time). Defocused slices
(d >= 1) keep their L1-renormalized bilinear-downscaled PSF shape. The
delta is born at target resolution -- no cubic-zoom survival required.
OTF normalization was removed in P1.9 per user directive *"do not run
OTF normalized tensor anymore"* (P1.5/P1.6 confirmed OTF degrades
in-focus subject).

P1.9.4 R5 PIVOT (2026-04-25, user-authorized): R/D interpolation lives
HERE (post-resampling) rather than upstream in ``codev_to_render`` so
the cubic-RGI fits operate on ~44×44 downscaled kernels rather than
2048×2048 raw kernels (~2000× cheaper). Algorithmic locks (L18-L23)
unchanged: log1p-space, R-cubic + D-linear sequential 1-D RGIs,
post-overwrite raw nodes, per-(r, d) L1 renorm. Pipeline order at
render-tensor build:
    1. ``codev_to_render.convert()`` → raw (R, D, 2048, 2048) tensor
    2. ``downscale_tensor()`` initial bilinear/cubic resample
       → (R, D, ~44, ~44)
    3. R/D densification (this module's interp branch, default no-op)
       → (target_R, target_D, ~44, ~44)
    4. Delta install at d=0 (per L24) — runs LAST so post-interp
       overwrite of d=0 doesn't affect downstream renderer behavior.

CLI::

    python -m summicron.psf_model.downscale_tensor \
        --in  runs/ag_dblgauss/ag_dblgauss_f2.0_render.npz \
        --target-long-edge 5712 \
        [--target-R N] [--target-D N] [--mem-threshold-gb 8.0]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import zoom as _ndi_zoom

from summicron.psf_model.codev_to_render import (
    _compute_corpus_ref,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "runs" / "ag_dblgauss" / "ag_dblgauss_f2.0_render.npz"


def _make_delta(size: int) -> np.ndarray:
    """Return an exact single-pixel delta at the geometric center.

    Geometric center is ``size // 2`` on both axes (matching the renderer's
    odd-grid convention and ``_center_crop_or_pad`` from this module).

    P1.10.2 Cut 2: inlined from the deleted ``summicron.psf_model.otf_norm``
    module; sole production caller is the d=0 S2 delta install at :481 below.
    """
    k = np.zeros((size, size), dtype=np.float32)
    k[size // 2, size // 2] = 1.0
    return k

# 36x24 mm full-frame reference used throughout the pipeline.
FULL_FRAME_WIDTH_UM = 36000.0
FULL_FRAME_HEIGHT_UM = 24000.0

# P1.9.4 L11 / L15 — bake-time transient-memory guard default.
MEM_THRESHOLD_GB_DEFAULT = 8.0


def _center_crop_or_pad(k: np.ndarray, target: int) -> np.ndarray:
    """Force a 2D array to exactly (target, target) via center crop/pad.

    Mirrors the convention in ``summicron.render.convolve`` (see
    convolve.py:306-317) so downscaled kernels match the renderer's own
    on-the-fly zoom path.
    """
    h, w = k.shape
    if h > target:
        top = (h - target) // 2
        k = k[top:top + target, :]
    elif h < target:
        pad_top = (target - h) // 2
        pad_bot = target - h - pad_top
        k = np.pad(k, ((pad_top, pad_bot), (0, 0)), mode="constant")
    h, w = k.shape
    if w > target:
        left = (w - target) // 2
        k = k[:, left:left + target]
    elif w < target:
        pad_left = (target - w) // 2
        pad_right = target - w - pad_left
        k = np.pad(k, ((0, 0), (pad_left, pad_right)), mode="constant")
    return k


# ---------------------------------------------------------------------------
# P1.9.4 R5 PIVOT — R/D interp lives here (post-downscale, pre-delta-install).
# ---------------------------------------------------------------------------

def _read_radial_centers_from_codev_dir(
    input_dir: Path, expected_R: int
) -> tuple[np.ndarray, float]:
    """Read ``radial_bin_centers_mm`` + ``h_max_mm`` from the first CodeV
    ``ag_dblgauss_f2.0_focus*_d*.npz`` file in ``input_dir``.

    The committed raw render tensor (md5 ``60e39bb8a4...``) does NOT
    serialize ``radial_bin_centers_mm`` — that field would change the
    md5. Per L23 it must be read VERBATIM from the raw CodeV ``.npz``,
    NOT formula-derived. So when interp is requested we look up the
    source files in the same directory the raw tensor came from.
    """
    files = sorted(input_dir.glob("ag_dblgauss_f2.0_focus*_d*.npz"))
    if not files:
        raise RuntimeError(
            f"R/D interp requested but no CodeV source files found in "
            f"{input_dir}; need ag_dblgauss_f2.0_focus*_d*.npz to read "
            "radial_bin_centers_mm (cannot be formula-derived per L23)."
        )
    d = np.load(str(files[0]), allow_pickle=True)
    centers = d["radial_bin_centers_mm"].astype(np.float64)
    h_max_mm = float(d["h_max_mm"])
    if centers.shape != (expected_R,):
        raise RuntimeError(
            f"radial_bin_centers_mm length {centers.shape[0]} != R={expected_R} "
            f"(read from {files[0]})"
        )
    return centers, h_max_mm


def _interp_rd(
    kernels_raw: np.ndarray,
    radial_bin_centers_mm: np.ndarray,
    h_max_mm: float,
    object_distances_m_raw: np.ndarray,
    target_R: int,
    target_D: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """R/D interp for a (downscaled) kernel tensor (P1.9.4 L18/L19/L20/L22/L23).

    NON-NEGOTIABLE ORDER:
        1. ``log1p(kernels_raw * 1e4)``
        2. Pass 1 — R-axis 1-D RGI cubic over ``radial_bin_centers_mm``,
           queried at ``r_interp_mm`` (sqrt-quadrature inside source range).
        3. Pass 2 — D-axis 1-D RGI **linear** over ``object_distances_m``,
           queried at ``np.geomspace(od_raw[0], od_raw[-1], target_D)``.
        4. ``expm1 / 1e4`` then ``np.clip(0, None)``.
        5. Post-overwrite raw nodes with ``kernels_raw[r, d]`` (deterministic
           bit-identity at all raw nodes that map onto the target grid).
        6. Per-(r, d) slice L1 renorm (compensates the linear-in-log
           concavity bias — without this, midpoint slices drop to L1=0.90).

    P1.9.4 R5 PIVOT: this helper lives in ``downscale_tensor`` so it
    operates on already-downscaled kernels (kH ≈ 44 at iPhone-scale)
    rather than the raw 2048-grid (~2000× cheaper). Algorithm is
    identical to the original R4-FINAL spec — only LOCATION moved.

    Returns
    -------
    kernels_interp : (target_R, target_D, kH, kW) float32
    r_interp_mm : (target_R,) float64 — new radial bin centers in mm
    od_interp : (target_D,) float64 — new object distances in m
    """
    raw_R, raw_D, kH, kW = kernels_raw.shape
    assert radial_bin_centers_mm.shape == (raw_R,)
    assert object_distances_m_raw.shape == (raw_D,)

    # L23 source axis = radial_bin_centers_mm read verbatim from raw .npz.
    r_raw_centers_mm = radial_bin_centers_mm.astype(np.float64)
    od_raw = object_distances_m_raw.astype(np.float64)

    # L23 target axis: sqrt-quadrature INSIDE source range
    # (upper bound = (centers[-1]/h_max)**2 keeps queries <= centers[-1]
    # and avoids violating bounds_error=True; centers[-1] < h_max_mm).
    h_max = float(h_max_mm)
    r_interp_mm = h_max * np.sqrt(
        np.linspace(0.0, (r_raw_centers_mm[-1] / h_max) ** 2, target_R)
    )
    # L23 runtime assert — catches drift in the formula or raw centers.
    assert np.all(
        (r_interp_mm >= r_raw_centers_mm[0]) & (r_interp_mm <= r_raw_centers_mm[-1])
    ), (
        "r_interp_mm out of source range "
        f"[{r_raw_centers_mm[0]}, {r_raw_centers_mm[-1]}]: "
        f"min={r_interp_mm.min()}, max={r_interp_mm.max()}"
    )

    # L8 D-axis target = geomspace (preserves the log-uniform CodeV step).
    od_interp = np.geomspace(od_raw[0], od_raw[-1], target_D)

    # L2 / Fact 6 — interp in log1p(PSF * 1e4) space.
    log_kernels = np.log1p(kernels_raw.astype(np.float64) * 1e4)

    # ---- PASS 1 — R-axis cubic RGI over r_raw_centers_mm ----
    # Vectorized: RGI accepts ND values where axis 0 matches the points
    # tuple; trailing axes (raw_D, kH, kW) are processed in one call.
    rgi_r = RegularGridInterpolator(
        (r_raw_centers_mm,),
        log_kernels,  # (raw_R, raw_D, kH, kW) — axis 0 is the R-source axis
        method="cubic",
        bounds_error=True,
    )
    # Query at all r_interp_mm points; rgi_r expects shape (Nq, n_points_dim).
    log_r = rgi_r(r_interp_mm[:, None])  # (target_R, raw_D, kH, kW)

    # ---- PASS 2 — D-axis linear RGI over od_raw ----
    # Move D-axis to position 0 for RGI consumption, then back.
    log_r_dfirst = np.moveaxis(log_r, 1, 0)  # (raw_D, target_R, kH, kW)
    rgi_d = RegularGridInterpolator(
        (od_raw,),
        log_r_dfirst,
        method="linear",
        bounds_error=True,
    )
    log_interp_dfirst = rgi_d(od_interp[:, None])  # (target_D, target_R, kH, kW)
    log_interp = np.moveaxis(log_interp_dfirst, 0, 1)  # (target_R, target_D, kH, kW)

    # ---- DECOMPRESS + CLIP (per L22, mandatory order) ----
    kernels_interp = np.expm1(log_interp) / 1e4
    kernels_interp = np.clip(kernels_interp, 0.0, None)

    # ---- POST-OVERWRITE RAW NODES (L18/L20 deterministic node fix) ----
    def _node_map(src: np.ndarray, tgt: np.ndarray) -> dict[int, int]:
        """For each src index i, the tgt index j such that |src[i]-tgt[j]|
        is below a small tolerance; otherwise omit i."""
        out: dict[int, int] = {}
        for i, s in enumerate(src):
            j = int(np.argmin(np.abs(tgt - s)))
            tol = max(1e-9, 1e-9 * max(abs(s), 1.0))
            if abs(tgt[j] - s) <= tol:
                out[i] = j
        return out

    r_node_map = _node_map(r_raw_centers_mm, r_interp_mm)
    d_node_map = _node_map(od_raw, od_interp)
    # Always force the 4 corners (raw_R/raw_D first/last) regardless of
    # tolerance — they correspond exactly by construction (sqrt(0)=0,
    # sqrt(1)=1, geomspace endpoints are exact under IEEE 754).
    r_node_map[0] = 0
    r_node_map[raw_R - 1] = target_R - 1
    d_node_map[0] = 0
    d_node_map[raw_D - 1] = target_D - 1

    for r_raw_idx, r_tgt_idx in r_node_map.items():
        for d_raw_idx, d_tgt_idx in d_node_map.items():
            kernels_interp[r_tgt_idx, d_tgt_idx] = kernels_raw[r_raw_idx, d_raw_idx]

    # ---- PER-(r, d) SLICE L1 RENORM (L22 — mandatory after overwrite) ----
    for r_i in range(target_R):
        for d_i in range(target_D):
            s = float(kernels_interp[r_i, d_i].sum())
            if s > 0:
                kernels_interp[r_i, d_i] /= s
            # else: leave zero slab as zero (G_L1_CONSERVATION will fire
            # if this happens unexpectedly).

    return kernels_interp.astype(np.float32, copy=False), r_interp_mm, od_interp


def _interp_pixel_pitch(
    pixel_pitch_um_raw: np.ndarray,
    target_R: int,
    target_D: int,
) -> np.ndarray:
    """Interp pixel_pitch_um from (raw_R, raw_D) to (target_R, target_D).

    L6 — pitch is linear-meaningful (NOT energy-bearing); use linear-order
    zoom (no log transform, no L1 renorm). Use scipy.ndimage.zoom(order=1).
    """
    raw_R, raw_D = pixel_pitch_um_raw.shape
    if (target_R, target_D) == (raw_R, raw_D):
        return pixel_pitch_um_raw.copy()
    zoom_R = target_R / raw_R
    zoom_D = target_D / raw_D
    out = _ndi_zoom(pixel_pitch_um_raw, (zoom_R, zoom_D), order=1, mode="nearest")
    # Defensive: zoom can be off-by-one on non-integer factors; clip shape.
    if out.shape != (target_R, target_D):
        # Fall back to RGI linear on the meshgrid; this rarely fires.
        rr_src = np.arange(raw_R, dtype=np.float64)
        dd_src = np.arange(raw_D, dtype=np.float64)
        rr_tgt = np.linspace(0, raw_R - 1, target_R)
        dd_tgt = np.linspace(0, raw_D - 1, target_D)
        rgi = RegularGridInterpolator(
            (rr_src, dd_src), pixel_pitch_um_raw.astype(np.float64),
            method="linear", bounds_error=True,
        )
        Rgrid, Dgrid = np.meshgrid(rr_tgt, dd_tgt, indexing="ij")
        out = rgi(np.stack([Rgrid.ravel(), Dgrid.ravel()], axis=-1)).reshape(
            target_R, target_D
        )
    return out.astype(np.float64, copy=False)


def downscale_tensor(
    in_path: str | Path,
    out_path: str | Path,
    target_long_edge: int = 5712,
    target_R: int | None = None,
    target_D: int | None = None,
    mem_threshold_gb: float = MEM_THRESHOLD_GB_DEFAULT,
    codev_input_dir: Path | None = None,
) -> Path:
    """Downscale every (R, D) kernel to match an image pixel pitch.

    Pipeline:
      1. Bilinear/cubic resample each (r, d) kernel from raw kH→target_kH.
      2. *Optional* R/D densification (P1.9.4 R5 pivot): if
         ``target_R`` and ``target_D`` are provided AND differ from
         ``(raw_R, raw_D)``, run the R-cubic / D-linear two-pass RGI
         pipeline on the downscaled kernels.
      3. Install the S2 single-pixel delta at every ``(r, d=0)`` slice
         (LAST step — overwrites whatever interp produced at d=0).

    Default invocation (``target_R is None and target_D is None``) is a
    NO-OP through the interp branch — output is byte-identical to the
    P1.9.3 canonical iPhone tensor (md5 ``e6f501546c...``).

    Parameters
    ----------
    in_path : path to input .npz (produced by codev_to_render.convert).
    out_path : path to write new .npz.
    target_long_edge : the working-resolution long edge in pixels. The
        kernel is resampled so its pitch equals the iPhone's pitch at
        this resolution, i.e. ``36000 / target_long_edge`` micrometers.
    target_R : int or None
        Densified R-axis bin count. ``None`` (default) means "use raw R"
        and skips the interp branch entirely.
    target_D : int or None
        Densified D-axis bin count. ``None`` (default) means "use raw D"
        and skips the interp branch entirely.
    mem_threshold_gb : float
        Bake-time transient memory cap (per L11/L15). Raises
        ``RuntimeError`` if estimated transient > threshold. At
        post-downscale kH≈44 this almost never fires — it stays as a
        defensive guard for pathological R/D requests.
    codev_input_dir : Path or None
        Source directory of the original CodeV ``.npz`` files (needed
        when interp is requested, to source ``radial_bin_centers_mm``).
        Defaults to the directory of ``in_path``.

    Returns
    -------
    Path to the written file.

    Raises
    ------
    ValueError
        If the requested pitch implies ``target_kH > orig_kH`` (upscaling).
        The CodeV tensor is the upper bound — we never upscale.
    """
    in_path, out_path = Path(in_path), Path(out_path)
    data = np.load(str(in_path), allow_pickle=True)

    kernels = data["kernels"]  # (R, D, kH, kW)
    R, D, orig_kH, orig_kW = kernels.shape
    if orig_kH != orig_kW:
        raise RuntimeError(
            f"Non-square kernel grid {orig_kH}x{orig_kW}; not supported."
        )

    orig_pitch = np.asarray(data["pixel_pitch_um"], dtype=np.float64)
    orig_pitch_mean = float(orig_pitch.mean())

    target_pitch_um = FULL_FRAME_WIDTH_UM / float(target_long_edge)
    scale = orig_pitch_mean / target_pitch_um
    target_kH = int(round(orig_kH * scale))

    if target_kH < 1:
        raise ValueError(
            f"target_long_edge={target_long_edge} implies target_kH={target_kH} "
            f"(< 1); pick a larger target_long_edge."
        )
    if target_kH > orig_kH:
        raise ValueError(
            f"target_long_edge={target_long_edge} implies target_kH={target_kH} "
            f"> orig_kH={orig_kH}; upscaling is forbidden (CodeV tensor is the "
            f"upper bound). Pick a smaller target_long_edge."
        )

    # --- Stage 1: per-kernel cubic-zoom + L1 renorm to target grid ---
    out_kernels = np.zeros((R, D, target_kH, target_kH), dtype=np.float32)
    zoom_factor = target_kH / orig_kH
    for ri in range(R):
        for di in range(D):
            k_hi = kernels[ri, di].astype(np.float32)
            k_lo = _ndi_zoom(k_hi, zoom_factor, order=3).astype(np.float32)
            k_lo = _center_crop_or_pad(k_lo, target_kH)
            # Cubic can undershoot; PSFs are non-negative by physics.
            k_lo = np.maximum(k_lo, 0.0)
            s = float(k_lo.sum())
            if s <= 0.0:
                raise RuntimeError(
                    "Downscaled kernel has non-positive sum at bin "
                    f"(R={ri}, D={di}); cannot L1-renormalize."
                )
            k_lo /= s
            out_kernels[ri, di] = k_lo

    # --- Stage 2: P1.9.4 R5 PIVOT — optional R/D densification ---
    # Default no-op when both targets are None or equal raw → output
    # is byte-identical to the P1.9.3 canonical iPhone tensor (G_NO_OP_IDENTITY).
    eff_R = R if target_R is None else int(target_R)
    eff_D = D if target_D is None else int(target_D)
    is_no_op = (eff_R == R) and (eff_D == D)

    # Object distances + radial_edges + pixel_pitch_um — start from the
    # input tensor's serialized values; replace if interp runs.
    out_radial_edges = np.asarray(data["radial_edges"], dtype=np.float64)
    out_object_distances = np.asarray(
        data["object_distances_m"], dtype=np.float64
    )
    out_defocus_ratios = np.asarray(data["defocus_ratios"], dtype=np.float64)

    if not is_no_op:
        # L11 / L15 — bake-time transient memory guard at the post-downscale
        # kH (typically ~44; threshold rarely fires here, but kept as a
        # defensive cap on pathological R/D requests).
        kH = target_kH
        estimated_bytes = eff_R * eff_D * kH * kH * 4 * 5  # 4=float32, 5=scratch
        estimated_gb = estimated_bytes / 1e9
        if estimated_gb > mem_threshold_gb:
            raise RuntimeError(
                f"interp would allocate ~{estimated_gb:.1f} GB transient "
                f"(R={eff_R}, D={eff_D}, kH={kH}); raise --mem-threshold-gb "
                f"to override (current threshold {mem_threshold_gb} GB; "
                f"interp exceeds {mem_threshold_gb} GB)"
            )

        # Source axes for interp.
        # L23: radial_bin_centers_mm MUST be read verbatim from raw CodeV
        # files (NOT formula-derived). Look in codev_input_dir (defaults
        # to the directory of in_path).
        codev_dir = (
            Path(codev_input_dir) if codev_input_dir is not None
            else in_path.parent
        )
        radial_bin_centers_mm, h_max_mm_for_interp = (
            _read_radial_centers_from_codev_dir(codev_dir, expected_R=R)
        )

        focus_distance_m = float(data["focus_distance_m"])

        # L18-L23 — sequential 1-D RGI passes on kernels.
        out_kernels, r_interp_mm, out_object_distances = _interp_rd(
            out_kernels,
            radial_bin_centers_mm,
            h_max_mm_for_interp,
            out_object_distances,
            eff_R,
            eff_D,
        )

        # L17 — focus_distance_m endpoint defensive assert (geomspace
        # preserves od_raw[0] exactly under IEEE 754).
        assert np.isclose(focus_distance_m, out_object_distances[0], rtol=1e-9), (
            f"focus_distance_m={focus_distance_m} drifted from "
            f"object_distances_m[0]={out_object_distances[0]} after geomspace interp"
        )

        # L5 — re-derive radial_edges from the NEW (interpolated) centers.
        # raw .npz edges follow sqrt(linspace(0, 1, R)); rebuild the
        # post-midpoint-split edges array for the densified R.
        new_raw_edges = np.sqrt(np.linspace(0.0, 1.0, eff_R))
        new_midpoint = (new_raw_edges[0] + new_raw_edges[1]) / 2.0
        out_radial_edges = np.concatenate(
            [[new_raw_edges[0], new_midpoint], new_raw_edges[1:]]
        )
        assert np.all(np.diff(out_radial_edges) > 0), (
            "Interpolated radial_edges not monotonically increasing"
        )

        # L16 — defocus_ratios re-derived as linspace at the new D.
        out_defocus_ratios = np.linspace(0.0, 1.0, eff_D, dtype=np.float64)

        # Update R/D so the rest of the function operates on the densified shape.
        R, D = eff_R, eff_D

    # --- Stage 3: S2 delta install at d=0 (P1.9.4 L24 — runs LAST) ---
    # Install an exact single-pixel delta at d=0 for every (densified)
    # radial bin. Overwrites whatever interp produced at d=0.
    for ri in range(R):
        out_kernels[ri, 0] = _make_delta(target_kH)

    # New pitch: uniform across all (R,D) bins. Per-bin variation in the
    # 0.5235-0.5796 range is lost; the renderer uses .mean() anyway.
    new_pitch = np.full((R, D), target_pitch_um, dtype=np.float64)

    full_frame_dim_px = np.zeros((R, D, 2), dtype=np.int32)
    full_frame_dim_px[:, :, 0] = int(round(FULL_FRAME_WIDTH_UM / target_pitch_um))
    full_frame_dim_px[:, :, 1] = int(round(FULL_FRAME_HEIGHT_UM / target_pitch_um))

    # corpus_ref_long_edge MUST equal target_long_edge by construction.
    corpus_ref_long_edge = int(target_long_edge)
    # Sanity: matches the helper given uniform pitch (within int-rounding).
    helper_ref = _compute_corpus_ref(new_pitch)
    assert helper_ref == corpus_ref_long_edge, (
        f"corpus_ref drift: helper={helper_ref} target={corpus_ref_long_edge}"
    )

    # --- Gates at the downscaled, delta-guarded output (P1.9) ---
    # Shape sanity (pre-existing).
    if out_kernels.shape != (R, D, target_kH, target_kH):
        raise RuntimeError(
            f"Shape FAIL: kernels shape {out_kernels.shape} != "
            f"{(R, D, target_kH, target_kH)}"
        )
    # G1: center kernel is exact single-pixel delta at target grid.
    center = out_kernels[0, 0]
    cy = cx = target_kH // 2
    g1_peak = float(center[cy, cx])
    g1_sum = float(center.sum())
    g1_nonzero = int((center != 0).sum())
    if not (g1_peak == 1.0 and g1_sum == 1.0 and g1_nonzero == 1):
        raise RuntimeError(
            f"G1 FAIL: center kernel not exact delta at target grid "
            f"(peak={g1_peak}, sum={g1_sum}, nonzero={g1_nonzero})"
        )
    # G2: per-kernel L1 = 1.0 ± 0.01
    sums = out_kernels.reshape(R * D, -1).sum(axis=1)
    if not np.all(np.abs(sums - 1.0) <= 0.01):
        raise RuntimeError(
            f"G2 FAIL: per-kernel L1 out of [0.99, 1.01]; "
            f"min={sums.min():.6f} max={sums.max():.6f}"
        )
    # G3: no NaN/Inf
    if np.any(np.isnan(out_kernels)) or np.any(np.isinf(out_kernels)):
        raise RuntimeError("G3 FAIL: NaN/Inf in downscaled+normalized kernels")
    # Corpus-ref consistency (retained from the pre-existing gate-suite).
    if corpus_ref_long_edge != int(target_long_edge):
        raise RuntimeError(
            f"corpus_ref FAIL: corpus_ref_long_edge {corpus_ref_long_edge} != "
            f"target_long_edge {target_long_edge}"
        )
    # Physical-extent ratio (kept from prior impl; not in v3 spec but a
    # useful invariant — allows 10% tolerance due to per-bin pitch spread).
    # Only enforce on the no-op path: under R/D densification the orig_pitch
    # array shape (R_raw, D_raw) no longer matches (R, D), so the elementwise
    # comparison is ill-defined. The renderer relies on the mean pitch only.
    if is_no_op:
        extent_ratio = (new_pitch * target_kH) / (orig_pitch * orig_kH)
        if not np.all((extent_ratio >= 0.90) & (extent_ratio <= 1.10)):
            raise RuntimeError(
                f"Extent FAIL: physical-extent ratio out of [0.90, 1.10]; "
                f"min={extent_ratio.min():.6f} max={extent_ratio.max():.6f}"
            )

    # --- Write (mirror codev_to_render field order) ---
    # P1.10.2 Cut 2: otf_norm_applied write removed (OTF deprecated since P1.9).
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        str(out_path),
        kernels=out_kernels,
        radial_edges=out_radial_edges,
        defocus_ratios=out_defocus_ratios,
        pixel_pitch_um=new_pitch,
        full_frame_dim_px=full_frame_dim_px,
        object_distances_m=out_object_distances,
        focus_distance_m=np.float64(data["focus_distance_m"]),
        h_max_mm=np.float64(data["h_max_mm"]),
        wavelength_nm=np.float64(data["wavelength_nm"]),
        corpus_ref_long_edge=np.int64(corpus_ref_long_edge),
        delta_guard_applied=np.bool_(True),
    )

    # File size sanity. For target_long_edge=5712 (kH ~183) raw
    # 12*11*183^2*4 ~ 17.7 MB; np.savez stores uncompressed so the on-disk
    # size lands close to the raw bound. Cap at 20 MB to catch real bloat.
    # Skip this gate when interp doubles/triples the (R, D) bin count.
    size_mb = out_path.stat().st_size / (1024 * 1024)
    if target_long_edge == 5712 and is_no_op and size_mb >= 20.0:
        raise RuntimeError(
            f"Size FAIL: output size {size_mb:.2f} MB >= 20 MB for "
            f"target_long_edge=5712"
        )

    print(
        f"Wrote downscaled+delta-guarded tensor {out_kernels.shape} "
        f"(pitch {orig_pitch_mean:.4f} -> {target_pitch_um:.4f} um, "
        f"target_long_edge={target_long_edge}, "
        f"corpus_ref={corpus_ref_long_edge}, {size_mb:.2f} MB)"
    )
    return out_path


def _default_out(in_path: Path, target_long_edge: int) -> Path:
    stem = in_path.stem
    # strip an existing _iphone{N} or legacy _k{N} suffix if present
    for marker in ("_iphone", "_k"):
        if marker in stem:
            head, _, tail = stem.rpartition(marker)
            if tail.isdigit():
                stem = head
                break
    return in_path.with_name(f"{stem}_iphone{target_long_edge}.npz")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Downscale render-ready PSF tensor to match an image long-edge.",
    )
    parser.add_argument(
        "--in",
        dest="in_path",
        type=Path,
        default=DEFAULT_INPUT,
        help="Input .npz path.",
    )
    parser.add_argument(
        "--input",
        dest="in_path",
        type=Path,
        help="Alias for --in.",
    )
    parser.add_argument(
        "--out",
        dest="out_path",
        type=Path,
        default=None,
        help="Output .npz (default: <in_stem>_iphone<target_long_edge>.npz).",
    )
    parser.add_argument(
        "--output",
        dest="out_path",
        type=Path,
        help="Alias for --out.",
    )
    parser.add_argument(
        "--target-long-edge",
        type=int,
        default=5712,
        help="Target image long-edge in pixels (default: 5712, iPhone native).",
    )
    # P1.9.4 R5 PIVOT — interp flags moved here from codev_to_render.
    parser.add_argument(
        "--target-R",
        type=int,
        default=None,
        dest="target_R",
        help=(
            "Densified R-axis bin count. Default: raw R (no interp, "
            "byte-identical to canonical iPhone tensor)."
        ),
    )
    parser.add_argument(
        "--target-D",
        type=int,
        default=None,
        dest="target_D",
        help=(
            "Densified D-axis bin count. Default: raw D (no interp, "
            "byte-identical to canonical iPhone tensor)."
        ),
    )
    parser.add_argument(
        "--mem-threshold-gb",
        type=float,
        default=MEM_THRESHOLD_GB_DEFAULT,
        dest="mem_threshold_gb",
        help=(
            f"Bake-time transient memory cap in GB "
            f"(default {MEM_THRESHOLD_GB_DEFAULT})."
        ),
    )
    parser.add_argument(
        "--codev-input-dir",
        type=Path,
        default=None,
        dest="codev_input_dir",
        help=(
            "Directory of source CodeV .npz files (needed to read "
            "radial_bin_centers_mm under R/D interp). Defaults to the "
            "directory of --in."
        ),
    )
    args = parser.parse_args()

    out_path = args.out_path if args.out_path is not None else _default_out(
        args.in_path, args.target_long_edge
    )
    downscale_tensor(
        args.in_path,
        out_path,
        target_long_edge=args.target_long_edge,
        target_R=args.target_R,
        target_D=args.target_D,
        mem_threshold_gb=args.mem_threshold_gb,
        codev_input_dir=args.codev_input_dir,
    )


if __name__ == "__main__":
    main()
