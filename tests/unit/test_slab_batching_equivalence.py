"""R-perf-1 I1+I2 — slab-batching + streaming-holes-fill equivalence proof.

Killer-risk gate per locked-spec-R-perf-1.md § I2 ("Coder must prove
streaming-equivalence to the global argmin(dist_masked, axis=0) +
take_along_axis path on a 256² synthetic with D=11, INCLUDING tie-break
behavior, BEFORE running the canonical bench").

The streaming holes-fill maintains a running per-pixel argmin of
``|d - argmax_d|`` over a validity-masked scan instead of materializing
the full ``(D, H, W, 3)`` per-slab output buffers. Strict ``<`` in the
streaming update + near→far slab iteration guarantee that ties resolve
to the SMALLEST-d slab — bit-exact match to ``np.argmin(dist_masked,
axis=0)`` (which itself returns the smallest tie-breaking index).

This file exercises the path two ways:

1. ``test_streaming_argmin_matches_global_argmin_synthetic`` — pure-numpy
   shadow of the streaming + global paths on a hand-crafted (D=11, 256²)
   ``slab_w`` + per-slab output volume. Confirms streaming's running
   argmin reproduces ``np.argmin(dist_masked, axis=0)`` and the gathered
   color matches ``take_along_axis(per_slab_outputs_d, ...)`` bit-exactly.

2. ``test_render_invariant_to_slab_batch_size`` — drives the real
   ``render()`` entrypoint on the same 256² synthetic with several
   batch sizes (1, 2, 4, 11) and asserts every output is bit-identical to
   the batch_size=D_gate=11 reference (which is the pre-batching path
   in everything-but-name).

Tie-break is exercised explicitly in (1) by constructing a depth column
with two slabs equidistant from ``argmax_d``.
"""
from __future__ import annotations

import numpy as np
import pytest

from summicron.render.convolve import (
    _EMPTY_SKIP_THRESH,
    compute_slab_weights,
    render,
)


# Production D=11 object_distances grid (matches test_layered_gather.py).
_OD_D11 = np.array(
    [0.7, 0.98, 1.37, 1.91, 2.68, 3.74, 5.23, 7.32, 10.23, 14.30, 20.0],
    dtype=np.float64,
)
_RE = np.array([0.0, 0.15, 0.30, 0.45, 0.60, 0.80, 1.01], dtype=np.float64)
_DR_D11 = np.linspace(0, 1, 11, dtype=np.float64)


def _synthetic_tensor_d11():
    """Minimal D=11 tensor matching production object_distances_m grid."""
    R, D = 6, 11
    k = 33
    kern = np.zeros((R, D, k, k), dtype=np.float32)
    c = k // 2
    # d=0 is a delta kernel (P1.9 S2 delta-guard convention).
    kern[:, 0, c, c] = 1.0
    ys, xs = np.mgrid[-c:c + 1, -c:c + 1].astype(np.float32)
    for di in range(1, D):
        g = np.exp(-(xs ** 2 + ys ** 2) / (2 * (1.5 + 1.5 * di) ** 2))
        kern[:, di] = (g / g.sum()).astype(np.float32)
    return {
        "kernels": kern,
        "radial_edges": _RE.copy(),
        "defocus_ratios": _DR_D11.copy(),
        "object_distances_m": _OD_D11.copy(),
        "focus_distance_m": 0.7,
        "corpus_ref_long_edge": None,
        "otf_norm_applied": False,
        "delta_guard_applied": True,
    }


def _global_argmin_holes_fill(
    slab_w: np.ndarray,
    per_slab_outputs_d: np.ndarray,
    per_slab_outputs_b: np.ndarray,
):
    """Reference (pre-batching) holes-fill: global argmin on a (D, H, W, 3)
    output volume.

    Mirrors lines 1619-1640 of the pre-R-perf-1 convolve.py exactly:

        validity = (slab_w > _EMPTY_SKIP_THRESH) & (per_slab_alpha > 0.0)
        argmax_d = np.argmax(slab_w, axis=0)
        dist_d = np.abs(D_idx[:,None,None] - argmax_d[None,:,:])
        dist_masked = np.where(validity, dist_d, HUGE)
        best_d = np.argmin(dist_masked, axis=0)
        gather_d = take_along_axis(per_slab_outputs_d, best_d_exp, axis=0)[0]
    """
    D, H, W = slab_w.shape
    HUGE = D * 10
    # per_slab_alpha = sw_d for non-empty slabs, 0 for empty-skipped.
    # Reproduce: per_slab_alpha[d] = slab_w[d] when slab_w[d].max() >= thresh
    # (else stays zero).
    per_slab_alpha = np.zeros_like(slab_w)
    for d in range(D):
        if float(slab_w[d].max()) >= _EMPTY_SKIP_THRESH:
            per_slab_alpha[d] = slab_w[d]

    validity = (slab_w > _EMPTY_SKIP_THRESH) & (per_slab_alpha > 0.0)
    argmax_d = np.argmax(slab_w, axis=0).astype(np.int32)
    D_idx = np.arange(D, dtype=np.int32)
    dist_d = np.abs(D_idx[:, None, None] - argmax_d[None, :, :]).astype(np.int32)
    dist_masked = np.where(validity, dist_d, HUGE).astype(np.int32)
    best_d = np.argmin(dist_masked, axis=0).astype(np.int32)
    has_valid = dist_masked.min(axis=0) < HUGE
    best_d_dist = dist_masked.min(axis=0).astype(np.int32)
    best_d_dist = np.where(has_valid, best_d_dist, np.int32(HUGE))
    best_d_exp = best_d[None, :, :, None]
    gather_d = np.take_along_axis(per_slab_outputs_d, best_d_exp, axis=0)[0]
    gather_b = np.take_along_axis(per_slab_outputs_b, best_d_exp, axis=0)[0]
    return best_d, best_d_dist, has_valid, gather_d, gather_b


def _streaming_argmin_holes_fill(
    slab_w: np.ndarray,
    per_slab_outputs_d: np.ndarray,
    per_slab_outputs_b: np.ndarray,
    batch_size: int,
):
    """Pure-numpy shadow of the in-place streaming-holes-fill the renderer
    runs after each batch. Iterates slabs near→far in batches of
    ``batch_size``, updating running ``best_d_dist / best_d_color_*`` with
    strict ``<`` so ties resolve to the smallest-d slab — bit-exact to
    ``np.argmin(dist_masked, axis=0)``."""
    D, H, W = slab_w.shape
    HUGE = np.int32(D * 10)
    argmax_d = np.argmax(slab_w, axis=0).astype(np.int32)
    best_d = np.zeros((H, W), dtype=np.int32)  # initialized below
    best_d_dist = np.full((H, W), HUGE, dtype=np.int32)
    best_d_color_d = np.zeros((H, W, 3), dtype=np.float32)
    best_d_color_b = np.zeros((H, W, 3), dtype=np.float32)

    for batch_start in range(0, D, batch_size):
        batch_end = min(batch_start + batch_size, D)
        for d in range(batch_start, batch_end):
            sw_d = slab_w[d]
            # Skip empty slabs the same way the renderer does.
            if float(sw_d.max()) < _EMPTY_SKIP_THRESH:
                continue
            validity_d = sw_d > _EMPTY_SKIP_THRESH
            dist_d_for_this = np.abs(np.int32(d) - argmax_d).astype(np.int32)
            dist_masked_d = np.where(validity_d, dist_d_for_this, HUGE)
            better = dist_masked_d < best_d_dist
            if better.any():
                best_d_dist = np.where(better, dist_masked_d, best_d_dist)
                best_d = np.where(better, np.int32(d), best_d)
                better_3 = better[..., None]
                best_d_color_d = np.where(
                    better_3, per_slab_outputs_d[d], best_d_color_d
                )
                best_d_color_b = np.where(
                    better_3, per_slab_outputs_b[d], best_d_color_b
                )

    has_valid = best_d_dist < HUGE
    return best_d, best_d_dist, has_valid, best_d_color_d, best_d_color_b


# ---------------------------------------------------------------------------
# (1) Pure-numpy streaming-vs-global equivalence proof on a 256² synthetic.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("batch_size", [1, 2, 3, 4, 7, 11])
def test_streaming_argmin_matches_global_argmin_synthetic(batch_size):
    """Streaming holes-fill must reproduce global argmin BIT-EXACTLY,
    including tie-break (ties resolve to the smallest-d slab)."""
    rng = np.random.default_rng(0xBADD0FF1)
    H = W = 256
    D = 11
    # Synthetic depth — uniform random across the full [0.7, 20] range so
    # most slabs have non-empty support.
    depth_m = rng.uniform(0.5, 25.0, (H, W)).astype(np.float32)
    slab_w = compute_slab_weights(depth_m, _OD_D11)
    assert slab_w.shape == (D, H, W)
    # NOTE on tie-break: with the validity definition `(slab_w > thresh)`,
    # tie-break in `argmin(dist_masked, axis=0)` is structurally vacuous
    # at the pixel level — the argmax_d slab is always valid (it has the
    # max value at that pixel, so it's also above threshold whenever any
    # slab is), giving dist=0 and a unique min. Tie-break still matters at
    # the IMPLEMENTATION level: the streaming path uses strict `<` while
    # iterating near→far, which is bit-exact to `argmin`'s "first wins"
    # semantics for any future change to the validity function. We verify
    # that with a separate fixture below where we override the validity
    # mask directly.

    per_d = rng.standard_normal((D, H, W, 3)).astype(np.float32)
    per_b = rng.standard_normal((D, H, W, 3)).astype(np.float32)

    g_best_d, g_dist, g_has_valid, g_gather_d, g_gather_b = (
        _global_argmin_holes_fill(slab_w, per_d, per_b)
    )
    s_best_d, s_dist, s_has_valid, s_gather_d, s_gather_b = (
        _streaming_argmin_holes_fill(slab_w, per_d, per_b, batch_size)
    )

    # has_valid mask is bit-identical.
    assert np.array_equal(g_has_valid, s_has_valid), (
        f"has_valid mismatch at batch_size={batch_size}: "
        f"{(g_has_valid != s_has_valid).sum()} pixels differ"
    )
    # best_d index is bit-identical.
    assert np.array_equal(g_best_d, s_best_d), (
        f"best_d mismatch at batch_size={batch_size}: "
        f"{(g_best_d != s_best_d).sum()} pixels differ"
    )

    # Distance and gathered colors are bit-identical on valid pixels.
    np.testing.assert_array_equal(g_dist[g_has_valid], s_dist[s_has_valid])
    # Gather colors must agree on valid-and-some-pixel where holes-fill
    # would actually consult them. Compare the full HxWx3 because both
    # paths zero-init invalid positions identically (zeros).
    np.testing.assert_array_equal(g_gather_d, s_gather_d)
    np.testing.assert_array_equal(g_gather_b, s_gather_b)


# ---------------------------------------------------------------------------
# (1b) Synthetic tie-break: validity mask hand-crafted to produce ties.
# ---------------------------------------------------------------------------


def _global_argmin_tiebreak(validity, argmax_d, per_d, per_b, D):
    """Reference: argmin(dist_masked, axis=0) — first wins on ties."""
    H, W = argmax_d.shape
    HUGE = D * 10
    D_idx = np.arange(D, dtype=np.int32)
    dist_d = np.abs(D_idx[:, None, None] - argmax_d[None, :, :]).astype(np.int32)
    dist_masked = np.where(validity, dist_d, HUGE).astype(np.int32)
    best_d = np.argmin(dist_masked, axis=0).astype(np.int32)
    has_valid = dist_masked.min(axis=0) < HUGE
    best_d_exp = best_d[None, :, :, None]
    g_d = np.take_along_axis(per_d, best_d_exp, axis=0)[0]
    g_b = np.take_along_axis(per_b, best_d_exp, axis=0)[0]
    return best_d, has_valid, g_d, g_b


def _streaming_argmin_tiebreak(validity, argmax_d, per_d, per_b, D, batch_size):
    """Streaming: strict `<` + near→far order = first wins on ties."""
    H, W = argmax_d.shape
    HUGE = np.int32(D * 10)
    best_d = np.zeros((H, W), dtype=np.int32)
    best_dist = np.full((H, W), HUGE, dtype=np.int32)
    best_color_d = np.zeros((H, W, 3), dtype=np.float32)
    best_color_b = np.zeros((H, W, 3), dtype=np.float32)
    for bs in range(0, D, batch_size):
        be = min(bs + batch_size, D)
        for d in range(bs, be):
            dist = np.abs(np.int32(d) - argmax_d).astype(np.int32)
            dist_masked = np.where(validity[d], dist, HUGE)
            better = dist_masked < best_dist
            if better.any():
                best_dist = np.where(better, dist_masked, best_dist)
                best_d = np.where(better, np.int32(d), best_d)
                better_3 = better[..., None]
                best_color_d = np.where(better_3, per_d[d], best_color_d)
                best_color_b = np.where(better_3, per_b[d], best_color_b)
    has_valid = best_dist < HUGE
    return best_d, has_valid, best_color_d, best_color_b


@pytest.mark.parametrize("batch_size", [1, 2, 3, 4, 7, 11])
def test_streaming_tiebreak_matches_argmin_first_wins(batch_size):
    """Hand-crafted validity mask that produces argmin ties — streaming
    must pick the SAME (smallest-d) winner as `np.argmin`."""
    rng = np.random.default_rng(0xCAFEFEED)
    H = W = 64
    D = 11
    # All slabs valid uniformly except where we tie-break.
    validity = np.ones((D, H, W), dtype=bool)
    # argmax_d = 5 everywhere — but we INVALIDATE slab 5 + neighbors at
    # one column, so dist=2 ties between slab 3 and slab 7 there.
    argmax_d = np.full((H, W), 5, dtype=np.int32)
    validity[5, :, 0] = False
    validity[4, :, 0] = False
    validity[6, :, 0] = False
    # Slabs 3 and 7 still valid; equidistant (dist=2). Smallest-d → 3.

    per_d = rng.standard_normal((D, H, W, 3)).astype(np.float32)
    per_b = rng.standard_normal((D, H, W, 3)).astype(np.float32)

    g_best_d, g_hv, g_gd, g_gb = _global_argmin_tiebreak(
        validity, argmax_d, per_d, per_b, D
    )
    s_best_d, s_hv, s_gd, s_gb = _streaming_argmin_tiebreak(
        validity, argmax_d, per_d, per_b, D, batch_size
    )

    # Tie-break sanity: at the salted column both must be 3 (smallest-d).
    assert int(g_best_d[0, 0]) == 3, (
        f"global argmin tie-break broke: expected 3, got {int(g_best_d[0, 0])}"
    )
    assert int(s_best_d[0, 0]) == 3, (
        f"streaming argmin tie-break broke (batch_size={batch_size}): "
        f"expected 3, got {int(s_best_d[0, 0])}"
    )

    np.testing.assert_array_equal(g_best_d, s_best_d)
    np.testing.assert_array_equal(g_hv, s_hv)
    np.testing.assert_array_equal(g_gd, s_gd)
    np.testing.assert_array_equal(g_gb, s_gb)


# ---------------------------------------------------------------------------
# (2) End-to-end: render() output must be invariant to slab_batch_size.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("batch_size", [1, 2, 4, 11])
def test_render_invariant_to_slab_batch_size(batch_size):
    """Drive render() at several batch sizes on a 256² synthetic; the
    output uint8 array must match the batch_size=11 (no-batching) path
    bit-exactly (max-abs-diff == 0 on float32 output, which is stronger
    than the canonical-bench uint8 ≤ 1 contract)."""
    rng = np.random.default_rng(20260506)
    H = W = 256
    img = rng.uniform(0.0, 0.8, (H, W, 3)).astype(np.float32)
    # Wide depth that activates most slabs so I1+I2 are exercised.
    depth_m = rng.uniform(0.7, 20.0, (H, W)).astype(np.float32)
    tensor = _synthetic_tensor_d11()

    # Reference: batch_size = D_gate = 11 (single batch, equivalent to the
    # pre-R-perf-1 path in everything-but-name).
    ref_u8, ref_lin = render(
        img, tensor, depth_m=depth_m, slab_batch_size=11
    )
    test_u8, test_lin = render(
        img, tensor, depth_m=depth_m, slab_batch_size=batch_size
    )

    # uint8 output: bit-identical (max-abs-diff == 0).
    diff_u8 = int(np.max(np.abs(ref_u8.astype(np.int16) - test_u8.astype(np.int16))))
    assert diff_u8 == 0, (
        f"slab_batch_size={batch_size} drifted uint8 vs no-batching: "
        f"max-abs-diff={diff_u8}"
    )
    # Linear float32 output: bit-identical (every batch_size collapses to
    # the same reduction order under streaming holes-fill).
    np.testing.assert_array_equal(ref_lin, test_lin)


def test_render_default_batch_size_matches_explicit_1():
    """slab_batch_size=None (module default = 1, R-native-mem) must be
    bit-identical to explicit 1 AND to explicit 4 (the former default)."""
    rng = np.random.default_rng(20260507)
    H = W = 256
    img = rng.uniform(0.0, 0.8, (H, W, 3)).astype(np.float32)
    depth_m = rng.uniform(0.7, 20.0, (H, W)).astype(np.float32)
    tensor = _synthetic_tensor_d11()

    out_default, lin_default = render(img, tensor, depth_m=depth_m)
    for b in (1, 4):
        out_explicit, lin_explicit = render(
            img, tensor, depth_m=depth_m, slab_batch_size=b
        )
        np.testing.assert_array_equal(out_default, out_explicit)
        np.testing.assert_array_equal(lin_default, lin_explicit)


def test_render_rejects_zero_batch_size():
    """slab_batch_size < 1 must raise ValueError."""
    rng = np.random.default_rng(0)
    img = rng.uniform(0, 0.5, (32, 32, 3)).astype(np.float32)
    depth_m = np.full((32, 32), 1.0, dtype=np.float32)
    tensor = _synthetic_tensor_d11()
    with pytest.raises(ValueError, match="slab_batch_size"):
        render(img, tensor, depth_m=depth_m, slab_batch_size=0)
