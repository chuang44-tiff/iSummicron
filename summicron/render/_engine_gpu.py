"""S33 Tier C: opt-in MLX (Apple GPU) engine for the layered gather.

``gpu_layered_gather`` takes the kwargs of ``convolve._layered_gather_diffuse``
(plus test-only ``threads``, ``_tile_n``, ``_pairs``, ``_tile_chunk``,
``_host_acc``, ``_cache_limit``) and returns the same 5-tuple. It is the Tier B
tiled schedule with the executor moved to the GPU:

* Every decision (tile plan, statistics, joint support, kernel bank, skips,
  holes, donors) is made on the CPU by ``_tile_schedule`` / ``_engine_tiled``,
  imported and not copied. MLX only executes: image spectra, products, inverse
  FFTs, weighted accumulation, the closed-form weight sums and the epilogue.
* The SAME kernel bank: the same K x K float32 kernels from the same bank
  units. Their spectra are float32 transforms computed ON the GPU
  (``_gpu_kernel_spectra``, one ``mx.fft.rfft2`` over the zero-padded stack;
  user ruling 2026-11-14, precision per engine: gpu float32, tiled float64).
  ``_engine_tiled._kernel_spectra`` (CPU float64) is not used by this engine.
* FFT sides are 324 or 512 (``_tile_schedule.tile_n_for_ksize``), never more.
* ``mlx`` is imported only inside functions: the package imports and renders
  without it.
* Batch and chunk sizes are fixed functions of N (MLX ``irfft2`` is not
  batch-invariant, so the output is deterministic per configuration).
* Accumulation is scatter-free (a take-based dense add): one IEEE float32 add
  per pixel, in the reference unit order.

Spec: doc/plan/phase2/section33_speed-052/locked-spec-S33-gpu.md.
"""
from __future__ import annotations

import os
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np

from summicron.render import _engine_tiled as et
from summicron.render import _tile_schedule as ts
from summicron.render import convolve as _cv

# Fixed batching constants, a pure function of N (spec 2.4). Provisional cache
# limits (spec 2.6, ruling R-C2).
_PAIRS_SMALL, _CHUNK_SMALL, _HOST_ACC_SMALL, _CACHE_SMALL = 64, 16, False, 1 << 30
_PAIRS_LARGE, _CHUNK_LARGE, _HOST_ACC_LARGE, _CACHE_LARGE = 16, 8, True, 1 << 29

_LAST_RUN: dict = {}        # execution record (tests / probe only; no production reader)
_TIMES: dict = {}           # INDICATIVE per-stage wall split of the last gather
_PROFILE_SPLIT = False      # tests/probes only: evaluate contributions before the add

_WS_BLOCK = 64              # tiles per block of the weight-sum graphs

# User ruling 2026-11-14: gpu kernel spectra are float32 transforms computed on
# the GPU. The constant exists so the contract is one greppable switch; False
# is not supported by the shipped engine (it would reintroduce the CPU float64
# path the ruling retired).
_GPU_F32_SPECTRA = True

# Cache-limit guard (audit r1 bug 2): overlapping gathers must restore the
# caller's MLX cache limit exactly once, by the last one out.
_CACHE_LOCK = threading.Lock()
_CACHE_DEPTH = 0
_CACHE_PREV = None


def _gpu_kernel_spectra(arrays, n, labels=None):
    """Spectra of K x K float32 kernels placed at the origin of zero N x N
    buffers: one ``mx.fft.rfft2`` over the stack, on the GPU stream. Returns an
    evaluated ``mx.array`` (len(arrays), N, N // 2 + 1) complex64.

    ``labels`` (d, ri, a | "delta", "diffuse" | "bright") is informational (the
    bank-identity tests read it). The kernels are transformed unchanged: no
    rotation, crop, clip, flip or renormalisation here."""
    import mlx.core as mx

    assert _GPU_F32_SPECTRA
    buf = np.zeros((len(arrays), n, n), dtype=np.float32)
    for j, k in enumerate(arrays):
        buf[j, : k.shape[0], : k.shape[1]] = k
    with mx.stream(mx.gpu):
        spec = mx.fft.rfft2(mx.array(buf))
        mx.eval(spec)
    return spec


def _config_for(n: int):
    if n <= 324:
        return _PAIRS_SMALL, _CHUNK_SMALL, _HOST_ACC_SMALL, _CACHE_SMALL
    return _PAIRS_LARGE, _CHUNK_LARGE, _HOST_ACC_LARGE, _CACHE_LARGE


def availability():
    """None when the engine is usable, else the reason it is not."""
    try:
        import mlx.core as mx
    except ImportError:
        return "mlx is not importable"
    try:
        if not mx.metal.is_available():
            return "no Metal GPU device"
    except Exception:
        return "no Metal GPU device"
    return None


def _retag(msg: str) -> str:
    return msg.replace("engine 'tiled'", "engine 'gpu'")


def _t(key, t0):
    _TIMES[key] = _TIMES.get(key, 0.0) + (time.perf_counter() - t0)


# --------------------------------------------------------------------------
# tile layout (host)
# --------------------------------------------------------------------------

def to_tiles(a, ny, nx, T):
    """(H, W[, C]) -> (nt, [C,] T, T), zero padded, row-major tiles."""
    H, W = a.shape[:2]
    if a.ndim == 2:
        p = np.zeros((ny * T, nx * T), a.dtype)
        p[:H, :W] = a
        return np.ascontiguousarray(
            p.reshape(ny, T, nx, T).transpose(0, 2, 1, 3).reshape(ny * nx, T, T))
    C = a.shape[2]
    p = np.zeros((ny * T, nx * T, C), a.dtype)
    p[:H, :W] = a
    return np.ascontiguousarray(
        p.reshape(ny, T, nx, T, C).transpose(0, 2, 4, 1, 3).reshape(ny * nx, C, T, T))


def from_tiles(t, ny, nx, T, H, W):
    """(nt, [C,] T, T) -> (H, W[, C])."""
    if t.ndim == 3:
        return np.ascontiguousarray(
            t.reshape(ny, nx, T, T).transpose(0, 2, 1, 3).reshape(ny * T, nx * T)[:H, :W])
    C = t.shape[1]
    return np.ascontiguousarray(
        t.reshape(ny, nx, C, T, T).transpose(0, 3, 1, 4, 2).reshape(ny * T, nx * T, C)[:H, :W])


def _compact_basis(radial_frac_map, r_centers, tiles, *, wedge_idx=None, T=None, ex=None):
    """Compact weight planes in tile layout.

    Returns ``(ri_lo uint8, w_lo f32, w_hi f32, wedge uint8 | None)``, each
    ``(nt, T, T)`` (``wedge`` is None when ``wedge_idx`` is not given).
    ``ri_lo`` uses the verbatim bracket lines of ``convolve._radial_basis_weights``;
    the VALUES are gathered from ``ts.tile_weights`` (the shared function), so no
    second weight formula exists. At most two basis planes are non-zero per pixel
    (``ri_lo`` and ``ri_lo + 1``).
    """
    rc = np.asarray(r_centers, dtype=np.float32)
    nr = int(rc.shape[0])
    if nr < 2:
        raise ValueError(
            "engine 'gpu' is not available for fewer than 2 radial bins; "
            "use --engine reference"
        )
    nt = len(tiles)
    if T is None:
        T = max(max(t[1] - t[0], t[3] - t[2]) for t in tiles)
    wi = wedge_idx if wedge_idx is not None else np.zeros(np.shape(radial_frac_map), np.uint8)
    o_ri = np.zeros((nt, T, T), np.uint8)
    o_lo = np.zeros((nt, T, T), np.float32)
    o_hi = np.zeros((nt, T, T), np.float32)
    o_wd = np.zeros((nt, T, T), np.uint8)

    def one(ti):
        y0, y1, x0, x1 = tiles[ti]
        th, tw = y1 - y0, x1 - x0
        rf = np.asarray(radial_frac_map[y0:y1, x0:x1], dtype=np.float32)
        ri_lo = np.clip(
            np.searchsorted(rc, rf, side="right") - 1, 0, nr - 2
        ).astype(np.int32)
        basis, wt = ts.tile_weights(radial_frac_map, wi, r_centers, tiles[ti])
        o_ri[ti, :th, :tw] = ri_lo
        o_lo[ti, :th, :tw] = np.take_along_axis(basis, ri_lo[None], axis=0)[0]
        o_hi[ti, :th, :tw] = np.take_along_axis(basis, (ri_lo + 1)[None], axis=0)[0]
        o_wd[ti, :th, :tw] = wt

    if ex is None:
        for ti in range(nt):
            one(ti)
    else:
        for _ in ex.map(one, range(nt)):
            pass
    return o_ri, o_lo, o_hi, (o_wd if wedge_idx is not None else None)


def _build_take_src(local_tiles, c, count):
    """Source index vector for one rank's take-based add.

    ``local_tiles[i]`` is the chunk-local tile of the i-th contribution row of
    this rank. Returns an int32 vector of length ``c``: the contribution row for
    each tile, or ``count`` (the appended all-zero row) for a tile with no pair
    in this rank. A tile (hence a row) referenced twice is a hard error.
    """
    local_tiles = np.asarray(local_tiles, dtype=np.int64)
    if len(np.unique(local_tiles)) != len(local_tiles):
        raise RuntimeError(
            "gpu engine: duplicate tile in one accumulation rank (internal error)"
        )
    src = np.full(int(c), int(count), dtype=np.int32)
    src[local_tiles] = np.arange(len(local_tiles), dtype=np.int32)
    return src


def _pair_weights(rilo, wlo, whi, wedge, tid, ri, a):
    """Per-pair weight planes ``(P, T, T)``, bit-equal to ``ts.wedge_weight``.

    ``a < 0`` is the is_delta unit (weight = basis[ri]). ``rilo``/``wlo``/``whi``/
    ``wedge`` are the uploaded compact planes (mx arrays, tile layout); ``tid``,
    ``ri``, ``a`` are int32 mx vectors.
    """
    import mlx.core as mx

    rl = mx.take(rilo, tid, axis=0).astype(mx.int32)
    wl = mx.take(wlo, tid, axis=0)
    wh = mx.take(whi, tid, axis=0)
    wd = mx.take(wedge, tid, axis=0).astype(mx.int32)
    rr = ri[:, None, None]
    aa = a[:, None, None]
    zero = mx.zeros((1, 1, 1), mx.float32)
    one = mx.ones((1, 1, 1), mx.float32)
    b = mx.where(rl == rr, wl, mx.where(rl + 1 == rr, wh, zero))
    m = mx.where(aa < 0, one, (wd == aa).astype(mx.float32))
    return b * m


def _windows(g, img, tids):
    """(n, 3, N, N) float32 windows of ``img`` for output tiles ``tids`` (GPU).

    One contiguous row band is uploaded; each window is copied to the origin of
    a zero N x N buffer (overlap-save).
    """
    import mlx.core as mx

    wins = [ts.input_window(g.tiles[ti], g.K) for ti in tids]
    r0 = min(w[0] for w in wins)
    r1 = max(w[1] for w in wins)
    band = mx.array(np.ascontiguousarray(img[r0:r1]))
    out = []
    for (py0, py1, px0, px1) in wins:
        src = band[py0 - r0:min(py1, r1) - r0, px0:px1].transpose(2, 0, 1)
        hh, ww = src.shape[1], src.shape[2]
        if hh != g.N or ww != g.N:
            src = mx.pad(src, ((0, 0), (0, g.N - hh), (0, g.N - ww)))
        out.append(src)
    return mx.stack(out)


def _check_cancel(g, d):
    if g.cancel is not None and g.cancel.is_set():
        from summicron.render.api import CancelledError

        raise CancelledError(f"render cancelled inside slab {d} (gpu engine)")


def _run_waves(g, cols, acc, kspec, S, d):
    """Execute the pairs of one chunk in waves and accumulate rank by rank.

    ``cols`` = (loc, tid, sr, kr, ri, a, rank) int32 vectors sorted by
    (rank, tid); ``acc`` is the chunk accumulator ``(c, 3, T, T)``. After every
    wave the result is evaluated (no graph outlives a wave).
    """
    import mlx.core as mx

    loc, tid, sr, kr, ri, aa, rank = cols
    n = len(loc)
    T, N, K = g.T, g.N, g.K
    c = acc.shape[0]
    zero_row = mx.zeros((1, 3, T, T), mx.float32)
    for w0 in range(0, n, g.pairs):
        _check_cancel(g, d)
        sl = slice(w0, min(n, w0 + g.pairs))
        t0 = time.perf_counter()
        prod = mx.take(S, mx.array(sr[sl]), axis=0) * mx.take(
            kspec, mx.array(kr[sl]), axis=0)[:, None]
        full = mx.fft.irfft2(prod, s=(N, N))
        w = _pair_weights(g.rilo, g.wlo, g.whi, g.wedge,
                          mx.array(tid[sl]), mx.array(ri[sl]), mx.array(aa[sl]))
        contrib = full[:, :, K - 1:K - 1 + T, K - 1:K - 1 + T] * w[:, None]
        if _PROFILE_SPLIT:
            mx.eval(contrib)
            _t("waves_conv", t0)
            t0 = time.perf_counter()
        r_w = rank[sl]
        loc_w = loc[sl]
        for r in np.unique(r_w):
            sel = np.nonzero(r_w == r)[0]
            s0, s1 = int(sel[0]), int(sel[-1]) + 1
            if s1 - s0 != len(sel):
                raise RuntimeError("gpu engine: rank rows not contiguous (internal error)")
            src = _build_take_src(loc_w[s0:s1], c, s1 - s0)
            blk = mx.concatenate([contrib[s0:s1], zero_row])
            acc = acc + mx.take(blk, mx.array(src), axis=0)
        mx.eval(acc)
        _t("waves_add" if _PROFILE_SPLIT else "waves", t0)
        g.n_waves += 1
    return acc


def _chunk_columns(units, chunk, dpairs, bpairs, srow, kidx_d, kidx_b, is_b):
    """Pair table of one chunk as int32 columns
    ``(loc, tid, sr, kr, ri, a, rank)`` sorted by (rank, tid), or None.

    ``rank`` is the pair's index in its tile's reference unit order (bright
    pairs: in the bright sub-sequence, which is what the reference adds)."""
    rec = []
    for j, ti in enumerate(chunk):
        lst = bpairs[ti] if is_b else dpairs[ti]
        for rank, ui in enumerate(lst):
            u = units[ui]
            kind = "b" if is_b else ("f" if u.a is None else "d")
            rec.append((
                j, ti, srow[(kind, ti)],
                (kidx_b if is_b else kidx_d)[ui], u.ri,
                -1 if u.a is None else u.a, rank))
    if not rec:
        return None
    arr = np.array(rec, dtype=np.int32)
    arr = arr[np.lexsort((arr[:, 1], arr[:, 6]))]
    return tuple(np.ascontiguousarray(arr[:, k]) for k in range(7))


def gpu_tile_convolve_full(img_padded, kernel, H, W, *, N=None):
    """Convolve the reflect-padded image with ONE kernel through the engine's
    GPU primitives. Returns (H, W, 3) float32. Test helper for the halo oracle:
    same window origin, FFT placement, product order and crop as production."""
    import mlx.core as mx

    K = int(kernel.shape[0])
    N = ts.tile_n_for_ksize(K) if N is None else int(N)
    T, tiles = ts.plan_tiles(H, W, K, N)
    spec = _gpu_kernel_spectra([kernel], N)[0]
    out = np.zeros((H, W, 3), dtype=np.float32)
    g = SimpleNamespace(tiles=tiles, K=K, N=N)
    with mx.stream(mx.gpu):
        for ti, (y0, y1, x0, x1) in enumerate(tiles):
            S = mx.fft.rfft2(_windows(g, img_padded, [ti]))
            full = mx.fft.irfft2(S * spec[None, None], s=(N, N))
            crop = full[0, :, K - 1:K - 1 + (y1 - y0), K - 1:K - 1 + (x1 - x0)]
            mx.eval(crop)
            out[y0:y1, x0:x1] = np.moveaxis(np.array(crop), 0, -1)
    return out


def gpu_layered_gather(
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
    _pairs=None,
    _tile_chunk=None,
    _host_acc=None,
    _cache_limit=None,
):
    """MLX counterpart of ``convolve._layered_gather_diffuse``.

    Same kwargs (``fft_state``, ``slab_support_skip`` and ``basis_hoist`` are
    accepted and ignored) and the same return
    ``(output, output_bright, weight_sum, delta_weight_sum, acc_alpha)``.
    The underscore keywords are test-only.
    """
    from summicron.render.api import CancelledError

    K = int(scaled_ksize)
    try:
        N = ts.tile_n_for_ksize(K) if _tile_n is None else int(_tile_n)
    except ValueError as e:
        raise ValueError(_retag(str(e))) from None
    if _tile_n is not None and N < K + 1:
        raise ValueError(f"_tile_n={N} must be >= K+1 (K={K})")
    if int(pad) != K // 2:
        raise ValueError(
            f"engine 'gpu' is not available for pad={pad} with kernel size "
            f"{K} (expected pad == K // 2); use --engine reference"
        )
    nr = int(len(r_centers))
    if nr < 2:
        raise ValueError(
            "engine 'gpu' is not available for fewer than 2 radial bins; "
            "use --engine reference"
        )
    D_gate = slab_w.shape[0]

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

    import mlx.core as mx

    pairs, chunk_n, host_acc, cache_limit = _config_for(N)
    if _pairs is not None:
        pairs = int(_pairs)
    if _tile_chunk is not None:
        chunk_n = int(_tile_chunk)
    if _host_acc is not None:
        host_acc = bool(_host_acc)
    if _cache_limit is not None:
        cache_limit = int(_cache_limit)

    P = max(1, int(threads)) if threads is not None else min(8, os.cpu_count() or 1)

    _TIMES.clear()
    t_all = time.perf_counter()
    global _CACHE_DEPTH, _CACHE_PREV
    with _CACHE_LOCK:
        if _CACHE_DEPTH == 0:
            _CACHE_PREV = mx.set_cache_limit(cache_limit)
        _CACHE_DEPTH += 1
    ex = ThreadPoolExecutor(max_workers=P)
    try:
        entry_active = mx.get_active_memory()
        mx.reset_peak_memory()
        with mx.stream(mx.gpu):
            res, n_waves = _gather_body(
                mx, ex, P, K=K, N=N, nr=nr, D_gate=D_gate,
                pairs=pairs, chunk_n=chunk_n, host_acc=host_acc,
                effective_cap=effective_cap,
                depth_m=depth_m, slab_w=slab_w, img_lin=img_lin,
                img_padded=img_padded, img_diffuse_padded=img_diffuse_padded,
                img_bright_padded=img_bright_padded,
                kernels=kernels, defocus_ratios=defocus_ratios,
                radial_frac_map=radial_frac_map, r_centers=r_centers,
                wedge_idx=wedge_idx, kH=kH, kW=kW, pix_scale=pix_scale,
                scaled_ksize=scaled_ksize, H=H, W=W, on_progress=on_progress,
                cancel=cancel, emit_log=emit_log,
                bright_stamp_gain=bright_stamp_gain,
            )
        _LAST_RUN.clear()
        _LAST_RUN.update({
            "device": "gpu", "waves": n_waves,
            "peak_bytes": int(mx.get_peak_memory()),
            "cache_limit": int(cache_limit),
            "entry_active_bytes": int(entry_active),
        })
    finally:
        ex.shutdown(wait=True, cancel_futures=True)
        with _CACHE_LOCK:
            _CACHE_DEPTH -= 1
            if _CACHE_DEPTH == 0:
                mx.set_cache_limit(_CACHE_PREV)
                mx.clear_cache()
    _TIMES["gather_total"] = time.perf_counter() - t_all
    return res


def _gather_body(mx, ex, P, *, K, N, nr, D_gate, pairs, chunk_n, host_acc,
                 effective_cap, depth_m, slab_w, img_lin, img_padded,
                 img_diffuse_padded, img_bright_padded, kernels, defocus_ratios,
                 radial_frac_map, r_centers, wedge_idx, kH, kW, pix_scale,
                 scaled_ksize, H, W, on_progress, cancel, emit_log,
                 bright_stamp_gain):
    """The gather proper. Returns ``(5-tuple, n_waves)``; every MLX array is
    local to this call, so it is released when it returns."""
    from summicron.render.api import CancelledError

    NA = _cv.N_ANG
    T, tiles = ts.plan_tiles(H, W, K, N)
    nt = len(tiles)
    ny, nx = (H + T - 1) // T, (W + T - 1) // T
    if ny * nx != nt:
        raise RuntimeError("gpu engine: tile grid mismatch (internal error)")
    holes_predicted = ts.predict_holes(slab_w, _cv._EMPTY_SKIP_THRESH)

    argmax_d_full = HUGE = best_dist = best_d = best_b = None
    if holes_predicted:
        argmax_d_full = np.argmax(slab_w, axis=0).astype(np.int32)
        HUGE = np.int32(D_gate * 10)
        best_dist = np.full((H, W), HUGE, dtype=np.int32)
        best_d = np.zeros((H, W, 3), dtype=np.float32)
        best_b = np.zeros((H, W, 3), dtype=np.float32)

    # ---- per-render CPU planning -----------------------------------------
    t0 = time.perf_counter()
    rows = list(ex.map(
        lambda ti: ts.tile_stats_row(
            radial_frac_map, wedge_idx, r_centers, tiles[ti],
            img_bright_padded, K),
        range(nt),
    ))
    stats = ts.assemble_stats(rows)
    del rows
    any_bright = bool(stats.has_bright.any())
    _t("plan_stats", t0)

    t0 = time.perf_counter()
    h_ri, h_lo, h_hi, h_wd = _compact_basis(
        radial_frac_map, r_centers, tiles, wedge_idx=wedge_idx, T=T, ex=ex)
    g = SimpleNamespace(
        K=K, N=N, T=T, tiles=tiles, cancel=cancel, pairs=pairs, n_waves=0,
        rilo=mx.array(h_ri), wlo=mx.array(h_lo), whi=mx.array(h_hi),
        wedge=mx.array(h_wd),
    )
    del h_ri, h_lo, h_hi, h_wd
    if host_acc:
        h_output = np.zeros((nt, 3, T, T), np.float32)
        h_output_b = np.zeros((nt, 3, T, T), np.float32)
        g_output = g_output_b = None
    else:
        g_output = mx.zeros((nt, 3, T, T), mx.float32)
        g_output_b = mx.zeros((nt, 3, T, T), mx.float32)
    g_wsum_tot = mx.zeros((nt, T, T), mx.float32)
    g_dwsum_tot = mx.zeros((nt, T, T), mx.float32)
    mx.eval(g.rilo, g.wlo, g.whi, g.wedge, g_wsum_tot, g_dwsum_tot,
            *( [g_output, g_output_b] if not host_acc else []))
    _t("plan_upload", t0)

    acc_alpha = np.zeros((H, W), dtype=np.float32)
    active_count = 0
    warned_cap = False
    zero_out = mx.zeros((1, 3, T, T), mx.float32)

    for d in range(D_gate):
        if cancel is not None and cancel.is_set():
            raise CancelledError(f"render cancelled at slab {d}/{D_gate}")
        if on_progress is not None:
            on_progress(f"  Slab {d + 1}/{D_gate}")
        sw_d = slab_w[d]
        if float(sw_d.max()) < _cv._EMPTY_SKIP_THRESH:
            continue
        active_count += 1
        if active_count > effective_cap and not warned_cap:
            warnings.warn(
                f"layered_gather: exceeded _MAX_ACTIVE_SLABS={effective_cap} "
                f"(D={D_gate}, active_count={active_count}); continuing — "
                f"possible pathological depth spread.",
                RuntimeWarning,
                stacklevel=2,
            )
            warned_cap = True

        # Bank-support flags (full-frame OR of per-tile joint support).
        t0 = time.perf_counter()
        joint = list(ex.map(
            lambda ti: ts.joint_support_row(
                stats, ti, sw_d[tiles[ti][0]:tiles[ti][1],
                                tiles[ti][2]:tiles[ti][3]],
                radial_frac_map, wedge_idx, r_centers, tiles[ti]),
            range(nt),
        ))
        bank_flags = np.logical_or.reduce(joint, axis=0)
        del joint
        _t("plan_slab", t0)
        t0 = time.perf_counter()
        try:
            units = ts.slab_bank(
                d, nr=nr, kernels=kernels, defocus_ratios=defocus_ratios,
                kH=kH, kW=kW, pix_scale=pix_scale, scaled_ksize=scaled_ksize,
                bright_stamp_gain=bright_stamp_gain,
                gmax_ri=stats.gmax_ri, gmax_ria=stats.gmax_ria,
                bank_flags=bank_flags,
            )
        except ValueError as e:
            raise ValueError(_retag(str(e))) from None
        _t("bank_build", t0)

        # Kernel spectra: float32 transforms computed on the GPU (ruling 2026-11-14).
        t0 = time.perf_counter()
        bank = [u for u in units if u.bank]
        kidx_d, kidx_b = {}, {}
        g_kspec = None
        if bank:
            arrays, labels = [], []
            for ui, u in enumerate(units):
                if not u.bank:
                    continue
                if u.a is None:
                    kidx_d[ui] = len(arrays)
                    arrays.append(u.kernel)
                    labels.append((d, u.ri, "delta", "diffuse"))
                else:
                    kidx_d[ui] = len(arrays)
                    arrays.append(u.kernel)
                    labels.append((d, u.ri, u.a, "diffuse"))
                    if any_bright:
                        kidx_b[ui] = len(arrays)
                        arrays.append(u.kernel_bright)
                        labels.append((d, u.ri, u.a, "bright"))
            g_kspec = _gpu_kernel_spectra(arrays, N, labels)
            _t("kernel_spectra", t0)
            t0 = time.perf_counter()
            del arrays
            for u in bank:
                u.kernel = u.kernel_bright = None
        _t("spectra_upload", t0)

        # Per-slab weight sums, closed form of the reference's sequential sum.
        t0 = time.perf_counter()
        L = np.zeros((nr, NA), np.float32)
        Ld = np.zeros((nr, NA), np.float32)
        for u in units:
            if u.a is None:
                L[u.ri, :] = 1.0
                Ld[u.ri, :] = 1.0
            else:
                L[u.ri, u.a] = 1.0
        gL = mx.array(L.reshape(-1))
        gLd = mx.array(Ld.reshape(-1))
        wblocks, dblocks = [], []
        for b0 in range(0, nt, _WS_BLOCK):
            b1 = min(nt, b0 + _WS_BLOCK)
            f0 = (g.rilo[b0:b1].astype(mx.int32) * NA
                  + g.wedge[b0:b1].astype(mx.int32))
            wl, wh = g.wlo[b0:b1], g.whi[b0:b1]
            wb = wl * gL[f0] + wh * gL[f0 + NA]
            db = wl * gLd[f0] + wh * gLd[f0 + NA]
            mx.eval(wb, db)
            wblocks.append(wb)
            dblocks.append(db)
        g_wsum = mx.concatenate(wblocks) if len(wblocks) > 1 else wblocks[0]
        g_dw = mx.concatenate(dblocks) if len(dblocks) > 1 else dblocks[0]
        g_wsum_tot = g_wsum_tot + g_wsum
        g_dwsum_tot = g_dwsum_tot + g_dw
        sw_t = to_tiles(sw_d.astype(np.float32), ny, nx, T)
        g_sw = mx.array(sw_t)
        mx.eval(g_wsum_tot, g_dwsum_tot, g_sw)
        del wblocks, dblocks, g_dw, gL, gLd
        _t("gpu_wsum", t0)

        # Per-tile pair lists, in the reference unit order (rank = list index).
        t0 = time.perf_counter()
        tile_has_sw = (sw_t != 0).reshape(nt, -1).any(axis=1)
        dpairs, bpairs, act_tiles = {}, {}, []
        for ti in range(nt):
            if not tile_has_sw[ti]:
                continue
            dl, bl = [], []
            for ui, u in enumerate(units):
                if not u.bank:
                    continue
                if u.a is None:
                    if not stats.bpos[ti, u.ri]:
                        continue
                elif not stats.wpos[ti, u.ri, u.a]:
                    continue
                dl.append(ui)
                if u.a is not None and stats.has_bright[ti]:
                    bl.append(ui)
            if dl:
                dpairs[ti], bpairs[ti] = dl, bl
                act_tiles.append(ti)
        _t("plan_pairs", t0)

        if holes_predicted:
            hole_t = np.zeros((nt, 3, T, T), np.float32)
            hole_bt = np.zeros((nt, 3, T, T), np.float32)

        for c0 in range(0, len(act_tiles), chunk_n):
            _check_cancel(g, d)
            chunk = act_tiles[c0:c0 + chunk_n]
            c = len(chunk)
            t0 = time.perf_counter()
            need_d = [ti for ti in chunk
                      if any(units[ui].a is not None for ui in dpairs[ti])]
            need_f = [ti for ti in chunk
                      if any(units[ui].a is None for ui in dpairs[ti])]
            need_b = [ti for ti in chunk if bpairs[ti]]
            srow, stacks, base = {}, [], 0
            for kind, img, need in (("d", img_diffuse_padded, need_d),
                                    ("f", img_padded, need_f),
                                    ("b", img_bright_padded, need_b)):
                if not need:
                    continue
                stacks.append(_windows(g, img, need))
                for j, ti in enumerate(need):
                    srow[(kind, ti)] = base + j
                base += len(need)
            _t("windows", t0)
            t0 = time.perf_counter()
            S = mx.fft.rfft2(mx.concatenate(stacks) if len(stacks) > 1 else stacks[0])
            mx.eval(S)
            _t("tile_spec", t0)

            acc_d = mx.zeros((c, 3, T, T), mx.float32)
            acc_b = mx.zeros((c, 3, T, T), mx.float32)
            cd = _chunk_columns(units, chunk, dpairs, bpairs, srow, kidx_d, kidx_b, False)
            cb = _chunk_columns(units, chunk, dpairs, bpairs, srow, kidx_d, kidx_b, True)
            if cd is not None:
                acc_d = _run_waves(g, cd, acc_d, g_kspec, S, d)
            if cb is not None:
                acc_b = _run_waves(g, cb, acc_b, g_kspec, S, d)

            t0 = time.perf_counter()
            cids = mx.array(np.array(chunk, dtype=np.int32))
            wsafe = mx.maximum(mx.take(g_wsum, cids, axis=0), 1e-8)[:, None]
            mix = mx.take(g_sw, cids, axis=0)[:, None]
            o = acc_d / wsafe
            osw = o * mix
            ob = obsw = None
            if cb is not None:
                ob = acc_b / wsafe
                obsw = ob * mix
            if host_acc:
                mx.eval(osw, *( [obsw] if obsw is not None else [] ))
                ids = np.array(chunk, dtype=np.int64)
                h_output[ids] += np.array(osw)
                if obsw is not None:
                    h_output_b[ids] += np.array(obsw)
            else:
                src_nt = np.full(nt, c, dtype=np.int32)
                src_nt[np.array(chunk)] = np.arange(c, dtype=np.int32)
                s_nt = mx.array(src_nt)
                g_output = g_output + mx.take(
                    mx.concatenate([osw, zero_out]), s_nt, axis=0)
                if obsw is not None:
                    g_output_b = g_output_b + mx.take(
                        mx.concatenate([obsw, zero_out]), s_nt, axis=0)
                mx.eval(g_output, g_output_b)
            if holes_predicted:
                ids = np.array(chunk, dtype=np.int64)
                mx.eval(o)
                hole_t[ids] = np.array(o)
                if ob is not None:
                    mx.eval(ob)
                    hole_bt[ids] = np.array(ob)
            del S, acc_d, acc_b, o, osw, ob, obsw, stacks
            _t("epilogue", t0)

        acc_alpha += sw_d.astype(np.float32)
        if holes_predicted:
            t0 = time.perf_counter()
            hd = from_tiles(hole_t, ny, nx, T, H, W)
            hb = from_tiles(hole_bt, ny, nx, T, H, W)
            best_dist, best_d, best_b = et._update_best_donor(
                best_dist, best_d, best_b, d, sw_d, argmax_d_full, hd, hb, HUGE)
            del hole_t, hole_bt, hd, hb
            _t("holes", t0)
        del g_kspec, g_wsum, g_sw, units, bank

    t0 = time.perf_counter()
    if host_acc:
        output = from_tiles(h_output, ny, nx, T, H, W)
        output_bright = from_tiles(h_output_b, ny, nx, T, H, W)
    else:
        output = from_tiles(np.array(g_output), ny, nx, T, H, W)
        output_bright = from_tiles(np.array(g_output_b), ny, nx, T, H, W)
    weight_sum = from_tiles(np.array(g_wsum_tot), ny, nx, T, H, W)
    delta_weight_sum = from_tiles(np.array(g_dwsum_tot), ny, nx, T, H, W)
    _t("download", t0)
    n_waves = g.n_waves

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
                "gpu engine: hole prediction disagrees with the accumulated "
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

    return (output, output_bright, weight_sum, delta_weight_sum, acc_alpha), n_waves
