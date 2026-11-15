"""S33 Tier B: opt-in tiled overlap-save CPU engine for the layered gather.

``tiled_layered_gather`` takes the kwargs of ``convolve._layered_gather_diffuse``
(plus test-only ``threads`` and ``_tile_n``) and returns the same 5-tuple.
``convolve.render(engine="tiled")`` / ``--engine tiled`` call it (the switch is
``_resolve.resolve_engine``); the default path is the reference engine,
byte-identical. Tier B contract (see the tier spec): hard ceilings against the
reference on real scenes and charts at L1 gain <= 4; for the stress family, no
worse than 1.5x the reference's float32 error against a float64 truth (p99.9 meets
this everywhere measured; a few single-pixel maxima do not, see
``tests/unit/test_s33_tiled_stress_ratio.py``).

Design (locked-spec-S33-tiled.md section 2):

* Output tiles of side T = N - K + 1 partition the frame; each tile is one
  N x N circular convolution (overlap-save) per kernel on the reflect-padded
  image window, valid rows/cols [K-1, K-1+t).
* The same kernel bank as the reference (``_tile_schedule.slab_bank``).
* Weights on the fly from crops of the render's own maps; skips use the
  reference's global thresholds and exact-zero tile occupancy only.
* Bright-sparse (provisional, ruling R-B3): a tile whose input window holds no
  non-zero bright pixel has no bright inverse.
* One thread pool per gather call; every pixel is owned by exactly one tile job
  with a fixed operation order, so the output does not depend on the thread
  count or on completion order.
"""
from __future__ import annotations

import os
import warnings
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from types import SimpleNamespace

import numpy as np
import scipy.fft as sp_fft

from summicron.render import _tile_schedule as ts
from summicron.render import convolve as _cv

_CHUNK = 2                      # C: units per batched inverse FFT (small: bounds per-thread
                                # workspace; tiled peak RSS stays under the reference, R-B4)
_SPEC_BATCH = 16                # kernels per batched forward FFT


def _tile_spec(img, tile, K, N):
    """rfft2 of the tile's input window (copied to the origin of a zero
    N x N buffer), per channel: (3, N, N // 2 + 1) complex64."""
    py0, py1, px0, px1 = ts.input_window(tile, K)
    src = img[py0:py1, px0:px1]
    buf = np.zeros((3, N, N), dtype=np.float32)
    buf[:, : src.shape[0], : src.shape[1]] = np.moveaxis(src, -1, 0)
    return sp_fft.rfft2(buf, workers=1)


def _kernel_spectra(arrays, n, labels=None, workers=1):
    """Spectra of K x K kernels placed at the origin of zero N x N buffers.

    ``labels`` (d, ri, a | "delta", "diffuse" | "bright") is informational
    (the bank-identity test reads it). Returns (len(arrays), N, N // 2 + 1)
    complex64.
    """
    out = np.empty((len(arrays), n, n // 2 + 1), dtype=np.complex64)
    for i0 in range(0, len(arrays), _SPEC_BATCH):
        chunk = arrays[i0:i0 + _SPEC_BATCH]
        # SPEC AMENDMENT (user-ratified 2026-11-13, locked-spec-S33-tiled.md
        # section 12, amending 2.3): kernel spectra may be float64 transforms
        # stored as complex64. The image-side FFTs and the accumulation stay
        # float32 / complex64 (see _tile_spec).
        # float64 transform, rounded once to complex64: the kernel spectra are
        # reused by every tile, so their float32 round-off would be a systematic
        # error in each output (measured: lowers the stress-family error vs a
        # float64 truth to the reference's level or below).
        buf = np.zeros((len(chunk), n, n), dtype=np.float64)
        for j, k in enumerate(chunk):
            buf[j, : k.shape[0], : k.shape[1]] = k
        out[i0:i0 + len(chunk)] = sp_fft.rfft2(buf, workers=workers)  # -> complex64
    return out


def _convolve_chunk(spectra_img, spectra_ker, K, th, tw, N):
    """Batched inverse of ``S_img * K_spec`` (image spectrum first) for a chunk
    of units, cropped to the valid region: (C, 3, th, tw) float32 view."""
    c = len(spectra_ker)
    prod = np.empty((c, 3, N, N // 2 + 1), dtype=np.complex64)
    for j in range(c):
        np.multiply(spectra_img[j], spectra_ker[j][None], out=prod[j])
    full = sp_fft.irfft2(prod, s=(N, N), workers=1)
    return full[:, :, K - 1:K - 1 + th, K - 1:K - 1 + tw]


def tile_convolve_full(img_padded, kernel, H, W, *, N=None, threads=1):
    """Convolve the reflect-padded image with ONE kernel through the engine's
    tile primitives. Returns (H, W, 3) float32.

    Test helper for the geometric halo oracle: same window origin, FFT
    placement, product order and crop as the production job.
    """
    K = int(kernel.shape[0])
    N = ts.tile_n_for_ksize(K) if N is None else int(N)
    T, tiles = ts.plan_tiles(H, W, K, N)
    spec = _kernel_spectra([kernel], N)[0]
    out = np.zeros((H, W, 3), dtype=np.float32)

    def job(ti):
        tile = tiles[ti]
        y0, y1, x0, x1 = tile
        S = _tile_spec(img_padded, tile, K, N)
        conv = _convolve_chunk([S], [spec], K, y1 - y0, x1 - x0, N)[0]
        out[y0:y1, x0:x1] = np.moveaxis(conv, 0, -1)

    with ThreadPoolExecutor(max_workers=max(1, int(threads))) as ex:
        list(ex.map(job, range(len(tiles))))
    return out


def _update_best_donor(best_dist, best_d, best_b, d, sw_d, argmax_d_full,
                       out_d, out_b, huge):
    """The reference's streaming holes-fill update for one slab (verbatim
    arithmetic; strict ``<`` so the lowest d wins ties)."""
    validity_d = sw_d > _cv._EMPTY_SKIP_THRESH
    dist_d_for_this = np.abs(np.int32(d) - argmax_d_full).astype(np.int32)
    dist_masked_d = np.where(validity_d, dist_d_for_this, huge)
    better = dist_masked_d < best_dist
    if better.any():
        best_dist = np.where(better, dist_masked_d, best_dist)
        better_3 = better[..., None]
        best_d = np.where(better_3, out_d, best_d)
        best_b = np.where(better_3, out_b, best_b)
    return best_dist, best_d, best_b


def _tile_job(ctx, ti):
    """One output tile of one slab: weights, units, finishing. Writes only the
    tile's own disjoint slices."""
    from summicron.render.api import CancelledError

    if ctx.cancel is not None and ctx.cancel.is_set():
        raise CancelledError(
            f"render cancelled inside slab {ctx.d} (tiled engine)"
        )
    st = ctx.stats
    tile = ctx.tiles[ti]
    y0, y1, x0, x1 = tile
    th, tw = y1 - y0, x1 - x0
    sl = (slice(y0, y1), slice(x0, x1))
    K, N = ctx.K, ctx.N
    sw_tile = ctx.sw_d[sl]
    tile_has_sw = bool((sw_tile != 0).any())

    basis, wt = ts.tile_weights(ctx.radial_frac_map, ctx.wedge_idx, ctx.r_centers, tile)
    wsum = np.zeros((th, tw), dtype=np.float32)
    dwsum = np.zeros((th, tw), dtype=np.float32)
    active = []                      # (unit, w) of tile-active bank units
    for u in ctx.units:
        if u.a is None:
            if not st.bpos[ti, u.ri]:
                continue
            w = basis[u.ri].astype(np.float32)
            wsum += w
            dwsum += w
        else:
            if not st.wpos[ti, u.ri, u.a]:
                continue
            w = ts.wedge_weight(basis[u.ri], wt, u.a)
            wsum += w
        if u.bank and tile_has_sw:
            active.append((u, w))

    ctx.weight_sum[sl] += wsum
    ctx.delta_weight_sum[sl] += dwsum
    if not active:
        return

    has_b = bool(st.has_bright[ti])
    out = np.zeros((3, th, tw), dtype=np.float32)
    out_b = np.zeros((3, th, tw), dtype=np.float32) if has_b else None
    nd = [x for x in active if x[0].a is not None]
    S_d = _tile_spec(ctx.img_diffuse_padded, tile, K, N) if nd else None
    S_b = _tile_spec(ctx.img_bright_padded, tile, K, N) if (nd and has_b) else None
    S_f = (
        _tile_spec(ctx.img_padded, tile, K, N)
        if len(nd) != len(active) else None
    )

    for i0 in range(0, len(active), _CHUNK):
        if ctx.cancel is not None and ctx.cancel.is_set():
            raise CancelledError(
                f"render cancelled inside slab {ctx.d} (tiled engine)"
            )
        chunk = active[i0:i0 + _CHUNK]
        conv = _convolve_chunk(
            [S_f if u.a is None else S_d for u, _ in chunk],
            [u.spec for u, _ in chunk], K, th, tw, N,
        )
        for j, (u, w) in enumerate(chunk):
            out += w * conv[j]
    if has_b:
        nd_chunks = [x for x in active if x[0].a is not None]
        for i0 in range(0, len(nd_chunks), _CHUNK):
            if ctx.cancel is not None and ctx.cancel.is_set():
                raise CancelledError(
                    f"render cancelled inside slab {ctx.d} (tiled engine)"
                )
            chunk = nd_chunks[i0:i0 + _CHUNK]
            conv = _convolve_chunk(
                [S_b] * len(chunk), [u.spec_bright for u, _ in chunk],
                K, th, tw, N,
            )
            for j, (u, w) in enumerate(chunk):
                out_b += w * conv[j]

    wsafe = np.maximum(wsum, 1e-8)
    for c in range(3):
        out[c] /= wsafe
        if has_b:
            out_b[c] /= wsafe
    alpha = sw_tile.astype(np.float32)
    mix = alpha[..., None]
    out_cl = np.moveaxis(out, 0, -1)
    ctx.output[sl] += out_cl * mix
    if has_b:
        out_b_cl = np.moveaxis(out_b, 0, -1)
        ctx.output_bright[sl] += out_b_cl * mix
    if ctx.hole_d is not None:
        ctx.hole_d[sl] = out_cl
        if has_b:
            ctx.hole_b[sl] = out_b_cl


def tiled_layered_gather(
    *,
    depth_m, slab_w, img_lin,
    img_padded, img_diffuse_padded, img_bright_padded,
    kernels, defocus_ratios, radial_frac_map, r_centers, wedge_idx,
    kH, kW, pad, pix_scale, scaled_ksize, H, W,
    fft_state=None, slab_batch_size=None, on_progress=None, cancel=None,
    emit_log=True,
    bright_stamp_gain=None,
    active_slab_cap=None,
    slab_support_skip=True,
    basis_hoist=True,
    threads=None,
    _tile_n=None,
):
    """Tiled overlap-save counterpart of ``convolve._layered_gather_diffuse``.

    Same kwargs (``fft_state``, ``slab_support_skip`` and ``basis_hoist`` are
    accepted and ignored) and the same return
    ``(output, output_bright, weight_sum, delta_weight_sum, acc_alpha)``.
    ``threads`` and ``_tile_n`` are test-only.
    """
    from summicron.render.api import CancelledError

    K = int(scaled_ksize)
    N = ts.tile_n_for_ksize(K) if _tile_n is None else int(_tile_n)
    if _tile_n is not None and N < K + 1:
        raise ValueError(f"_tile_n={N} must be >= K+1 (K={K})")
    if int(pad) != K // 2:
        raise ValueError(
            f"engine 'tiled' is not available for pad={pad} with kernel size "
            f"{K} (expected pad == K // 2); use --engine reference"
        )
    D_gate = slab_w.shape[0]
    nr = int(len(r_centers))

    if slab_batch_size is None:
        batch_size = _cv._SLAB_BATCH_SIZE
    else:
        batch_size = int(slab_batch_size)
    if batch_size < 1:
        raise ValueError(f"slab_batch_size must be >= 1, got {batch_size!r}")
    effective_cap = (
        max(_cv._MAX_ACTIVE_SLABS, D_gate // 2) if active_slab_cap is None
        else int(active_slab_cap)
    )
    n_batches = (D_gate + batch_size - 1) // batch_size
    if batch_size * n_batches > max(effective_cap * 4, D_gate * 4):
        raise ValueError(
            f"slab_batch_size * n_batches = {batch_size * n_batches} "
            f"exceeds hard upper bound on (D_gate={D_gate}, "
            f"effective_cap={effective_cap}); pick smaller "
            f"--slab-batch-size."
        )

    if cancel is not None and cancel.is_set():
        raise CancelledError(f"render cancelled at slab 0/{D_gate}")

    P = max(1, int(threads)) if threads is not None else min(8, os.cpu_count() or 1)

    output = np.zeros_like(img_lin)
    output_bright = np.zeros_like(img_lin)
    weight_sum = np.zeros((H, W), dtype=np.float32)
    delta_weight_sum = np.zeros((H, W), dtype=np.float32)
    acc_alpha = np.zeros((H, W), dtype=np.float32)

    T, tiles = ts.plan_tiles(H, W, K, N)
    nt = len(tiles)
    holes_predicted = ts.predict_holes(slab_w, _cv._EMPTY_SKIP_THRESH)

    argmax_d_full = HUGE = best_dist = best_d = best_b = None
    hole_d = hole_b = None
    if holes_predicted:
        argmax_d_full = np.argmax(slab_w, axis=0).astype(np.int32)
        HUGE = np.int32(D_gate * 10)
        best_dist = np.full((H, W), HUGE, dtype=np.int32)
        best_d = np.zeros((H, W, 3), dtype=np.float32)
        best_b = np.zeros((H, W, 3), dtype=np.float32)

    active_count = 0
    _warned_cap = False
    ex = ThreadPoolExecutor(max_workers=P)
    try:
        rows = list(ex.map(
            lambda ti: ts.tile_stats_row(
                radial_frac_map, wedge_idx, r_centers, tiles[ti],
                img_bright_padded, K),
            range(nt),
        ))
        stats = ts.assemble_stats(rows)
        del rows
        any_bright = bool(stats.has_bright.any())

        ctx = SimpleNamespace(
            K=K, N=N, tiles=tiles, stats=stats,
            radial_frac_map=radial_frac_map, wedge_idx=wedge_idx,
            r_centers=r_centers, cancel=cancel,
            img_padded=img_padded, img_diffuse_padded=img_diffuse_padded,
            img_bright_padded=img_bright_padded,
            output=output, output_bright=output_bright,
            weight_sum=weight_sum, delta_weight_sum=delta_weight_sum,
            d=0, sw_d=None, units=None, hole_d=None, hole_b=None,
        )

        for d in range(D_gate):
            if cancel is not None and cancel.is_set():
                raise CancelledError(f"render cancelled at slab {d}/{D_gate}")
            if on_progress is not None:
                on_progress(f"  Slab {d + 1}/{D_gate}")
            sw_d = slab_w[d]
            if float(sw_d.max()) < _cv._EMPTY_SKIP_THRESH:
                continue
            active_count += 1
            if active_count > effective_cap and not _warned_cap:
                warnings.warn(
                    f"layered_gather: exceeded _MAX_ACTIVE_SLABS={effective_cap} "
                    f"(D={D_gate}, active_count={active_count}); continuing — "
                    f"possible pathological depth spread.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                _warned_cap = True

            # Bank-support flags (full-frame OR of per-tile joint support).
            joint = list(ex.map(
                lambda ti: ts.joint_support_row(
                    stats, ti, sw_d[tiles[ti][0]:tiles[ti][1],
                                    tiles[ti][2]:tiles[ti][3]],
                    radial_frac_map, wedge_idx, r_centers, tiles[ti]),
                range(nt),
            ))
            bank_flags = np.logical_or.reduce(joint, axis=0)
            del joint
            units = ts.slab_bank(
                d, nr=nr, kernels=kernels, defocus_ratios=defocus_ratios,
                kH=kH, kW=kW, pix_scale=pix_scale, scaled_ksize=scaled_ksize,
                bright_stamp_gain=bright_stamp_gain,
                gmax_ri=stats.gmax_ri, gmax_ria=stats.gmax_ria,
                bank_flags=bank_flags,
            )
            bank = [u for u in units if u.bank]
            if bank:
                arrays, labels, owners = [], [], []
                for u in bank:
                    if u.a is None:
                        arrays.append(u.kernel)
                        labels.append((d, u.ri, "delta", "diffuse"))
                        owners.append((u, "spec"))
                    else:
                        arrays.append(u.kernel)
                        labels.append((d, u.ri, u.a, "diffuse"))
                        owners.append((u, "spec"))
                        if any_bright:
                            arrays.append(u.kernel_bright)
                            labels.append((d, u.ri, u.a, "bright"))
                            owners.append((u, "spec_bright"))
                specs = _kernel_spectra(arrays, N, labels, workers=P)
                for (u, attr), s in zip(owners, specs):
                    setattr(u, attr, s)
                del specs, arrays
                # The spectra own the data; free the spatial kernels.
                for u in bank:
                    u.kernel = u.kernel_bright = None

            if holes_predicted:
                hole_d = np.zeros((H, W, 3), dtype=np.float32)
                hole_b = np.zeros((H, W, 3), dtype=np.float32)
            ctx.d, ctx.sw_d, ctx.units = d, sw_d, units
            ctx.hole_d, ctx.hole_b = hole_d, hole_b
            job = partial(_tile_job, ctx)
            for _ in ex.map(job, range(nt)):
                pass

            acc_alpha += sw_d.astype(np.float32)

            if holes_predicted:
                best_dist, best_d, best_b = _update_best_donor(
                    best_dist, best_d, best_b, d, sw_d, argmax_d_full,
                    hole_d, hole_b, HUGE,
                )
                hole_d = hole_b = None
            ctx.units = None
            del units, bank
    finally:
        ex.shutdown(wait=True, cancel_futures=True)

    if emit_log:
        _cv._emit(
            on_progress,
            f"  layered_gather: active_slabs={active_count}/{D_gate}, "
            f"acc_alpha mean={float(acc_alpha.mean()):.3f}, "
            f"holes={int((acc_alpha < 0.02).sum())} pixels"
        )

    holes = acc_alpha < 0.02
    if holes.any():
        if not holes_predicted:
            raise RuntimeError(
                "tiled engine: hole prediction disagrees with the accumulated "
                "alpha (internal error)"
            )
        has_valid = best_dist < HUGE
        fallback = img_lin
        valid_and_holes = (holes & has_valid)[..., None]
        output = np.where(valid_and_holes, best_d, output)
        output_bright = np.where(valid_and_holes, best_b, output_bright)
        noone_and_holes = (holes & ~has_valid)[..., None]
        output = np.where(noone_and_holes, fallback, output)
        output_bright = np.where(noone_and_holes, 0.0, output_bright)
        acc_alpha = np.where(holes & has_valid, np.float32(1.0), acc_alpha)
        acc_alpha = np.where(holes & ~has_valid, np.float32(1.0), acc_alpha)

    return output, output_bright, weight_sum, delta_weight_sum, acc_alpha
