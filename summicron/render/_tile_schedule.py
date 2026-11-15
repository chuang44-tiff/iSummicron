"""S33 Tier B: tile schedule shared by the CPU tiled engine (and, later, Tier C).

Holds everything that decides WHAT a tiled gather computes, with no FFT and no
threads: the FFT-side table, the output-tile plan, per-tile weight crops, the
global-skip statistics, the per-slab kernel bank (``slab_bank``) and the hole
prediction. ``summicron.render._engine_tiled`` executes it.

Rules (locked-spec-S33-tiled.md section 2):

* K is the side of the built kernel; halo and window origin come ONLY from K
  (never from measured kernel support).
* The rotation / renormalise / stamp lines in :func:`slab_bank` are verbatim
  copies of ``convolve._apply_kernel``; they appear nowhere else in the tiled
  engine, so the kernel bank is the reference's bank byte for byte.
* Skips keep the reference meaning: global (full-frame) 0.01 maxima, exact-zero
  occupancy per tile. No threshold is ever evaluated per tile.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from summicron.render import convolve as _cv

MAX_K = 255          # K above this is "not available" (custom --tensor only)
_N_SMALL = 324       # 7-smooth, K <= 99
_N_LARGE = 512       # 7-smooth, 99 < K <= 255


def tile_n_for_ksize(K: int) -> int:
    """FFT side N from the fixed table (no autotuning)."""
    K = int(K)
    if K < 1:
        raise ValueError(f"kernel size must be >= 1, got {K}")
    if K <= 99:
        return _N_SMALL
    if K <= MAX_K:
        return _N_LARGE
    raise ValueError(
        f"engine 'tiled' is not available for kernel size {K} (> {MAX_K}); "
        "use --engine reference"
    )


def plan_tiles(H: int, W: int, K: int, N: int):
    """Row-major output tiles ``(y0, y1, x0, x1)`` partitioning the H x W frame.

    Output tile side T = N - K + 1 (the wrap-free valid region of an
    N x N circular convolution with a K x K kernel).
    """
    T = int(N) - int(K) + 1
    if T < 1:
        raise ValueError(f"FFT side N={N} must be >= K+1 (K={K})")
    return T, [
        (y0, min(y0 + T, H), x0, min(x0 + T, W))
        for y0 in range(0, H, T)
        for x0 in range(0, W, T)
    ]


def input_window(tile, K: int):
    """Padded-image window ``(py0, py1, px0, px1)`` read by an output tile.

    Output pixel y reads padded rows [y, y+K-1] (odd and even K, with
    pad = K // 2); the tile reads [y0, y0 + t + K - 1). The slice is clipped to
    the padded frame by numpy.
    """
    y0, y1, x0, x1 = tile
    return y0, y1 + K - 1, x0, x1 + K - 1


def tile_weights(radial_frac_map, wedge_idx, r_centers, tile):
    """``(basis (Nr, th, tw) float32, wedge (th, tw))`` on crops of the
    render's own maps (never recomputed from tile-local coordinates)."""
    y0, y1, x0, x1 = tile
    basis = _cv._radial_basis_weights(radial_frac_map[y0:y1, x0:x1], r_centers)
    return basis, wedge_idx[y0:y1, x0:x1]


def wedge_weight(basis_ri, wedge, a):
    """The reference expression for ``w_ang``."""
    angular_mask = (wedge == a).astype(np.float32)
    return (basis_ri * angular_mask).astype(np.float32)


@dataclass
class TileStats:
    """Per-render statistics over the tile grid (one row per tile)."""

    bmax: np.ndarray       # (nt, Nr) float32   max of basis[ri] in the tile
    bpos: np.ndarray       # (nt, Nr) bool      (basis[ri] > 0).any()
    wmax: np.ndarray       # (nt, Nr, N_ANG) float32   max of w_ang
    wpos: np.ndarray       # (nt, Nr, N_ANG) bool      (w_ang > 0).any()
    has_bright: np.ndarray  # (nt,) bool  input window (halo included) != 0
    gmax_ri: np.ndarray = field(init=False)
    gmax_ria: np.ndarray = field(init=False)

    def __post_init__(self):
        # max over tiles == full-frame max (max is exact, tiles partition).
        self.gmax_ri = self.bmax.max(axis=0)
        self.gmax_ria = self.wmax.max(axis=0)


def _wedge_masks(wt):
    """``[(a, wt == a)]`` for the wedges present in the tile (an absent wedge has
    an all-zero ``w_ang``: weight 0, never positive)."""
    return [(a, m) for a in range(_cv.N_ANG) if (m := (wt == a)).any()]


def _compact_ok(rf_crop, wt, nr):
    """The compact (two-plane) form applies: >= 2 radial bins and wedge ids in
    ``0 .. N_ANG - 1`` (always true for the renderer's own maps)."""
    return nr >= 2 and wt.size > 0 and int(wt.min()) >= 0 and int(wt.max()) < _cv.N_ANG


def _tile_compact(rf_crop, r_centers):
    """``(ri_lo int32, lo f32, hi f32)``: the at-most-two non-zero planes of
    ``convolve._radial_basis_weights`` per pixel. The bracket lines are that
    function's own, verbatim, and ``basis[ri_lo] = lo``, ``basis[ri_lo+1] = hi``
    is its scatter (checked plane for plane against the shared function in
    tests/unit/test_s33_fixed_costs.py), so the values are identical."""
    rc = np.asarray(r_centers, dtype=np.float32)
    Nr = int(rc.shape[0])
    rf = np.asarray(rf_crop, dtype=np.float32)
    # (np.minimum(np.maximum(.)) is np.clip's own definition; spelled out so the
    # static "no clip in the tiled engine" guard stays literal.)
    ri_lo = np.minimum(
        np.maximum(np.searchsorted(rc, rf, side="right") - 1, 0), Nr - 2
    ).astype(np.int32)
    ri_hi = ri_lo + 1
    rc_lo = rc[ri_lo]
    rc_hi = rc[ri_hi]
    span = rc_hi - rc_lo
    span_safe = np.where(span < 1e-12, np.float32(1.0), span)
    t = np.minimum(np.maximum((rf - rc_lo) / span_safe, 0.0), 1.0).astype(np.float32)
    return ri_lo, (1.0 - t).astype(np.float32), t


def _segments(ri_lo, wt, lo, hi, nr, nz=None, want_max=True):
    """Per-(plane, wedge) reductions of the compact planes, as arrays shaped
    ``(nr, N_ANG)``: ``(wmax, wpos)`` where ``wmax[r, a]`` is the max of
    ``basis[r]`` over wedge ``a`` (0 if none) and ``wpos[r, a]`` is
    ``(basis[r] > 0 [& nz]).any()`` over it. One stable radix sort by
    ``(ri_lo, wedge)`` groups the pixels; ``reduceat`` reduces each group."""
    na = _cv.N_ANG
    key = (ri_lo.ravel() * na + wt.ravel()).astype(np.uint16)
    order = np.argsort(key, kind="stable")
    ng = (nr - 1) * na
    counts = np.bincount(key, minlength=ng)
    live = np.nonzero(counts)[0]
    starts = (np.cumsum(counts) - counts)[live]
    wpos = np.zeros((nr, na), bool)
    wmax = np.zeros((nr, na), np.float32)
    for plane_off, v in ((0, lo), (1, hi)):
        vs = v.ravel()[order]
        pos = vs > 0
        if nz is not None:
            pos &= nz.ravel()[order]
        seg_pos = np.zeros(ng, bool)
        seg_pos[live] = np.logical_or.reduceat(pos, starts)
        wpos[plane_off:plane_off + nr - 1] |= seg_pos.reshape(nr - 1, na)
        if want_max:
            seg_max = np.zeros(ng, np.float32)
            seg_max[live] = np.maximum.reduceat(vs, starts)
            blk = wmax[plane_off:plane_off + nr - 1]
            np.maximum(blk, seg_max.reshape(nr - 1, na), out=blk)
    return wmax, wpos


def tile_stats_row(radial_frac_map, wedge_idx, r_centers, tile, bright_padded, K):
    """Statistics of one tile: ``(bmax, bpos, wmax, wpos, has_bright)``.

    WP5: identical outputs to the per-(ri, a) loop over ``wedge_weight``.
    ``w_ang = basis * (wedge == a)`` is ``basis`` inside wedge ``a`` and exactly
    0 elsewhere, and ``basis >= 0``, so ``max(w_ang)`` is the max of ``basis``
    over the wedge's pixels (0 for an absent wedge) and ``(w_ang > 0).any()`` is
    ``(basis > 0).any()`` over them. Because every wedge id is in range, the
    plane statistics are the max / any over the wedge axis. The fast path
    works on the two non-zero planes per pixel; anything else takes the
    full-basis form.
    """
    y0, y1, x0, x1 = tile
    rf = radial_frac_map[y0:y1, x0:x1]
    wt = wedge_idx[y0:y1, x0:x1]
    nr = int(np.shape(r_centers)[0])
    if _compact_ok(rf, wt, nr):
        ri_lo, lo, hi = _tile_compact(rf, r_centers)
        wmax, wpos = _segments(ri_lo, wt, lo, hi, nr)
        bmax = wmax.max(axis=1).astype(np.float32)
        bpos = wpos.any(axis=1)
    else:
        basis, wt = tile_weights(radial_frac_map, wedge_idx, r_centers, tile)
        flat = basis.reshape(nr, -1)
        bmax = flat.max(axis=1).astype(np.float32)
        bpos = (flat > 0).any(axis=1)
        wmax = np.zeros((nr, _cv.N_ANG), np.float32)
        wpos = np.zeros((nr, _cv.N_ANG), bool)
        for a, m in _wedge_masks(wt):
            v = basis[:, m]                       # (nr, n_a)
            wmax[:, a] = np.maximum(v.max(axis=1), np.float32(0.0))
            wpos[:, a] = (v > 0).any(axis=1)
    py0, py1, px0, px1 = input_window(tile, K)
    # `!= 0`, not `> 0`: extended-range input can be negative. count_nonzero
    # counts exactly the `!= 0` elements (NaN included).
    has_b = bool(np.count_nonzero(bright_padded[py0:py1, px0:px1]) > 0)
    return bmax, bpos, wmax, wpos, has_b


def assemble_stats(rows) -> TileStats:
    bmax, bpos, wmax, wpos, hb = zip(*rows)
    return TileStats(
        bmax=np.stack(bmax), bpos=np.stack(bpos),
        wmax=np.stack(wmax), wpos=np.stack(wpos),
        has_bright=np.array(hb, dtype=bool),
    )


def joint_support_row(stats: TileStats, ti, sw_tile, radial_frac_map,
                      wedge_idx, r_centers, tile):
    """Per-tile bank-support flags for one slab: ``(nr, N_ANG + 1)`` bool.

    Column ``a`` < N_ANG is ``((w_ang > 0) & (sw != 0)).any()``; the last
    column is the is_delta flag ``((w > 0) & (sw != 0)).any()`` (the reference
    ``_support_bbox`` set, non-empty).
    """
    nr = stats.bpos.shape[1]
    flags = np.zeros((nr, _cv.N_ANG + 1), bool)
    sw_nz = sw_tile != 0
    if not sw_nz.any():
        return flags
    if sw_nz.all():
        flags[:, : _cv.N_ANG] = stats.wpos[ti]
        flags[:, _cv.N_ANG] = stats.bpos[ti]
        return flags
    y0, y1, x0, x1 = tile
    rf = radial_frac_map[y0:y1, x0:x1]
    wt = wedge_idx[y0:y1, x0:x1]
    if _compact_ok(rf, wt, nr):
        ri_lo, lo, hi = _tile_compact(rf, r_centers)
        _, wpos = _segments(ri_lo, wt, lo, hi, nr, nz=sw_nz, want_max=False)
        flags[:, : _cv.N_ANG] = wpos
        flags[:, _cv.N_ANG] = wpos.any(axis=1)
        return flags
    basis, wt = tile_weights(radial_frac_map, wedge_idx, r_centers, tile)
    pos = (basis > 0) & sw_nz[None]           # (nr, th, tw)
    # Rows with bpos False are all-False already (basis > 0 never holds there).
    flags[:, _cv.N_ANG] = pos.reshape(nr, -1).any(axis=1)
    for a, m in _wedge_masks(wt):
        flags[:, a] = pos[:, m].any(axis=1)
    return flags


@dataclass
class Unit:
    """One weight unit of a slab (ri, a) or the is_delta unit of ri (a None).

    ``bank`` units carry the kernels the reference FFTs; a weight unit that is
    not a bank unit contributes weight only.
    """

    ri: int
    a: int | None
    bank: bool
    kernel: np.ndarray | None = None          # rotated (or kernel_base if delta)
    kernel_bright: np.ndarray | None = None   # stamped; None for delta
    spec: np.ndarray | None = None            # (N, N//2+1) complex64
    spec_bright: np.ndarray | None = None


def slab_bank(d, *, nr, kernels, defocus_ratios, kH, kW, pix_scale,
              scaled_ksize, bright_stamp_gain, gmax_ri, gmax_ria, bank_flags):
    """Weight units of slab ``d`` in reference order (ri ascending, then a
    ascending), with the reference's rotated, renormalised, stamped kernels
    for bank units.

    ``gmax_ri`` / ``gmax_ria`` are the full-frame maxima (global skips);
    ``bank_flags[ri, a]`` (a == N_ANG for is_delta) is True when the unit's
    ``(w > 0) & (sw != 0)`` set is non-empty anywhere in the frame.
    """
    K = int(scaled_ksize)
    units: list[Unit] = []
    for ri in range(nr):
        if gmax_ri[ri] < 0.01:
            continue
        kernel_base = _cv._build_basis_kernel(
            ri, d, True, defocus_ratios, kernels,
            kH, kW, pix_scale, scaled_ksize,
        )
        if kernel_base.shape != (K, K):
            raise ValueError(
                f"engine 'tiled' is not available for a kernel of shape "
                f"{kernel_base.shape} (expected {(K, K)}); use --engine reference"
            )
        is_delta = kernel_base.max() > 0.5 * kernel_base.sum()
        if is_delta:
            bank = bool(bank_flags[ri, _cv.N_ANG])
            units.append(Unit(ri, None, bank, kernel_base if bank else None))
            continue
        for a in range(_cv.N_ANG):
            if gmax_ria[ri, a] < 0.01:
                continue
            bank = bool(bank_flags[ri, a])
            if not bank:
                units.append(Unit(ri, a, False))
                continue
            theta_a = (a + 0.5) * (2 * np.pi / _cv.N_ANG) - np.pi
            rot_deg = -(theta_a * 180.0 / np.pi + 90.0)
            rotated = _cv.ndimage_rotate(kernel_base, rot_deg, reshape=False, order=1)
            rotated = np.maximum(rotated, 0.0)
            rotated /= rotated.sum() + 1e-12
            rotated_bright = _cv._bright_stamp(rotated, bright_stamp_gain)
            units.append(Unit(ri, a, True, rotated, rotated_bright))
    return units


def predict_holes(slab_w, thresh):
    """``holes_predicted``: ``(acc_alpha < 0.02).any()`` for the reference's
    ascending-d float32 sum over the non-empty-skipped slabs."""
    acc = np.zeros(slab_w.shape[1:], dtype=np.float32)
    for d in range(slab_w.shape[0]):
        sw_d = slab_w[d]
        if float(sw_d.max()) < thresh:
            continue
        acc += sw_d.astype(np.float32)
    return bool((acc < 0.02).any())
