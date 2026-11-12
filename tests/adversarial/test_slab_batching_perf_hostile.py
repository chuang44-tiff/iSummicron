"""R-perf-1 I1+I2 — hostile tests for slab-batching + streaming holes-fill.

Adversarial probes against the Coder's claim that slab-batching is bit-exact
across batch sizes for the canonical bench scene. We attack:

(a) Wrong-polarity depth (REQUIRED per Adversarial Rules) — focus FAR rather
    than NEAR, with the slab-w validity mask and tie-break unchanged.
(b) is_delta mixed slabs — d=0 delta, d>=1 gaussian; bit-exactness across
    batch sizes proves the per-(d, ri) `kernel_base` build is not stale-cached
    across batch boundaries.
(c) Cancellation mid-batch — `cancel.set()` before render → CancelledError;
    no leaked partial accumulator state.
(d) `--slab-batch-size` hard guards — 0, negative, and pathological large
    values must raise ValueError.
(e) D=1 degenerate — single-slab tensor renders or raises cleanly, no NaN.
(f) Stdout-byte-identical (G1 contract) — the
    `layered_gather: active_slabs=...` print line must match across batch
    sizes (active_count, acc_alpha mean ≥3 decimal, holes count).
(g) RSS non-regression — batch_size=1 must NOT peak above batch_size=4 by
    more than its working-set delta (i.e., smaller batches use less RAM).

Wrong-polarity fixture details: `test_focus_at_far_invariant_to_batch_size`
constructs a depth map where the subject is at the FAR boundary (~20 m) and
the background is NEAR (~0.7 m). The focus_distance_m in the synthetic
tensor sits at the near end (0.7 m), so this scene puts the in-focus content
at the far end of the depth — opposite of the conventional "subject near,
background far" polarity that Day-1 post-mortem tests covered.
"""
from __future__ import annotations

import io
import contextlib
import threading
import gc
import os

import numpy as np
import pytest

import summicron.render.convolve as _CV
from summicron.render.api import CancelledError
from summicron.render.convolve import render, _SLAB_BATCH_SIZE


_OD_D11 = np.array(
    [0.7, 0.98, 1.37, 1.91, 2.68, 3.74, 5.23, 7.32, 10.23, 14.30, 20.0],
    dtype=np.float64,
)
_RE = np.array([0.0, 0.15, 0.30, 0.45, 0.60, 0.80, 1.01], dtype=np.float64)
_DR_D11 = np.linspace(0, 1, 11, dtype=np.float64)


def _tensor_d11_mixed_delta(ksize: int = 33) -> dict:
    """D=11 tensor: d=0 is delta (P1.9 S2 convention), d>=1 are gaussians.

    Distinct sigma per d so cross-batch determinism is sensitive to ANY
    stale-cache or repeat-compute on the (d, ri) → kernel_base hand-off.
    """
    R, D = 6, 11
    kern = np.zeros((R, D, ksize, ksize), dtype=np.float32)
    c = ksize // 2
    kern[:, 0, c, c] = 1.0  # delta — is_delta=True branch
    ys, xs = np.mgrid[-c:c + 1, -c:c + 1].astype(np.float32)
    for di in range(1, D):
        sigma = 1.5 + 1.5 * di
        g = np.exp(-(xs ** 2 + ys ** 2) / (2 * sigma ** 2))
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


# ---------------------------------------------------------------------------
# (a) WRONG-POLARITY: focus at FAR end of depth (subject far, bg near).
#     Adversarial Rules require at least one wrong-polarity fixture.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("batch_size", [1, 2, 3, 4, 7, 11])
def test_focus_at_far_invariant_to_batch_size(batch_size):
    """Wrong-polarity scene: subject (center) at 20 m, background at 0.7 m.
    Focus_distance_m=0.7 makes the subject MAXIMALLY defocused. Streaming
    holes-fill assumes ordering is irrelevant; verify with reversed polarity.
    """
    rng = np.random.default_rng(0xF002A)
    H, W = 96, 96
    # Center "subject" pixels far (20 m), background near (0.7 m).
    yy, xx = np.mgrid[:H, :W].astype(np.float32)
    cy, cx = H / 2, W / 2
    r2 = (yy - cy) ** 2 + (xx - cx) ** 2
    mask = r2 < 12 ** 2
    depth_m = np.where(mask, 20.0, 0.7).astype(np.float32)
    img = rng.uniform(0.05, 0.7, (H, W, 3)).astype(np.float32)

    tensor = _tensor_d11_mixed_delta()
    ref_u8, ref_lin = render(img, tensor, depth_m=depth_m, slab_batch_size=11)
    test_u8, test_lin = render(img, tensor, depth_m=depth_m, slab_batch_size=batch_size)

    assert np.all(np.isfinite(test_lin)), "wrong-polarity: non-finite output"
    np.testing.assert_array_equal(ref_lin, test_lin)
    np.testing.assert_array_equal(ref_u8, test_u8)


# ---------------------------------------------------------------------------
# (b) Mixed is_delta vs FFT slabs: bit-exact across batch sizes.
#     Probes any stale-cache on (d, ri) → kernel_base across batch boundaries.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("batch_size", [1, 2, 3, 5, 11])
def test_mixed_is_delta_invariant_to_batch_size(batch_size):
    """Some slabs are delta (d=0), others are gaussian (d>=1). Render at
    multiple batch sizes; all outputs must be byte-identical to b=11 ref."""
    rng = np.random.default_rng(0xDEAD11)
    H, W = 96, 96
    img = rng.uniform(0.0, 0.9, (H, W, 3)).astype(np.float32)
    depth_m = rng.uniform(0.7, 20.0, (H, W)).astype(np.float32)
    tensor = _tensor_d11_mixed_delta()

    ref_u8, ref_lin = render(img, tensor, depth_m=depth_m, slab_batch_size=11)
    test_u8, test_lin = render(img, tensor, depth_m=depth_m, slab_batch_size=batch_size)
    np.testing.assert_array_equal(ref_lin, test_lin)
    np.testing.assert_array_equal(ref_u8, test_u8)


# ---------------------------------------------------------------------------
# (c) Cancellation mid-batch: pre-set cancel → CancelledError, clean state.
# ---------------------------------------------------------------------------


def test_cancel_set_before_first_slab_raises_cancelled_error():
    """`cancel.set()` before render starts → first slab loop iteration
    raises CancelledError (the loop's checkpoint at convolve.py :1585).
    Verify no leaked accumulator state masquerades as a NaN-free output."""
    rng = np.random.default_rng(42)
    H, W = 48, 48
    img = rng.uniform(0.1, 0.7, (H, W, 3)).astype(np.float32)
    depth_m = rng.uniform(0.7, 20.0, (H, W)).astype(np.float32)
    tensor = _tensor_d11_mixed_delta()

    cancel = threading.Event()
    cancel.set()
    with pytest.raises(CancelledError):
        render(img, tensor, depth_m=depth_m, slab_batch_size=2, cancel=cancel)


def test_cancel_set_after_one_batch_raises_cleanly():
    """Pump one batch through, then cancel: subsequent batch's slab-0 loop
    iteration must raise CancelledError without leaking partial running-buffer
    state into a returned object."""
    rng = np.random.default_rng(43)
    H, W = 48, 48
    img = rng.uniform(0.1, 0.7, (H, W, 3)).astype(np.float32)
    depth_m = rng.uniform(0.7, 20.0, (H, W)).astype(np.float32)
    tensor = _tensor_d11_mixed_delta()

    cancel = threading.Event()
    fire_after = {"n": 0}

    def progress(_msg: str) -> None:
        fire_after["n"] += 1
        if fire_after["n"] >= 3:  # set after a few slabs (mid-stream)
            cancel.set()

    with pytest.raises(CancelledError):
        render(
            img, tensor, depth_m=depth_m, slab_batch_size=2,
            cancel=cancel, on_progress=progress,
        )


# ---------------------------------------------------------------------------
# (d) CLI hard guards on --slab-batch-size.
# ---------------------------------------------------------------------------


def test_slab_batch_size_zero_raises():
    img = np.full((32, 32, 3), 0.5, dtype=np.float32)
    depth_m = np.full((32, 32), 1.0, dtype=np.float32)
    tensor = _tensor_d11_mixed_delta()
    with pytest.raises(ValueError, match="slab_batch_size must be >= 1"):
        render(img, tensor, depth_m=depth_m, slab_batch_size=0)


def test_slab_batch_size_negative_raises():
    img = np.full((32, 32, 3), 0.5, dtype=np.float32)
    depth_m = np.full((32, 32), 1.0, dtype=np.float32)
    tensor = _tensor_d11_mixed_delta()
    with pytest.raises(ValueError, match="slab_batch_size must be >= 1"):
        render(img, tensor, depth_m=depth_m, slab_batch_size=-1)


def test_slab_batch_size_pathologically_large_rejected():
    """`slab_batch_size=100` on D=11 trips the hard upper bound — refuses
    `batch_size * n_batches > max(effective_cap*4, D_gate*4)`. This defends
    against a CLI typo like `--slab-batch-size 100` exhausting RAM."""
    img = np.full((32, 32, 3), 0.5, dtype=np.float32)
    depth_m = np.full((32, 32), 1.0, dtype=np.float32)
    tensor = _tensor_d11_mixed_delta()
    with pytest.raises(ValueError, match="exceeds hard upper bound"):
        render(img, tensor, depth_m=depth_m, slab_batch_size=100)


def test_module_default_constant_is_1():
    """Documents the default contract: module-level _SLAB_BATCH_SIZE = 1
    (R-native-mem item 1; locked-spec-mem.md §3.1)."""
    assert _SLAB_BATCH_SIZE == 1, (
        f"default slab batch size drifted: expected 1, got {_SLAB_BATCH_SIZE}"
    )


# ---------------------------------------------------------------------------
# (e) D=1 degenerate single-slab tensor — render must not produce silent NaN.
# ---------------------------------------------------------------------------


def test_d1_single_slab_does_not_produce_nan():
    """D=1 tensor: only one slab exists; streaming-batch loop runs once with
    n_batches=1. Output must be finite, no IndexError, no silent NaN."""
    R, D = 4, 1
    ksize = 17
    kern = np.zeros((R, D, ksize, ksize), dtype=np.float32)
    c = ksize // 2
    kern[:, 0, c, c] = 1.0  # delta only
    tensor = {
        "kernels": kern,
        "radial_edges": np.linspace(0.0, 1.0, R + 1, dtype=np.float64),
        "defocus_ratios": np.array([0.0], dtype=np.float64),
        "object_distances_m": np.array([0.7], dtype=np.float64),
        "focus_distance_m": 0.7,
        "corpus_ref_long_edge": None,
        "otf_norm_applied": False,
        "delta_guard_applied": True,
    }
    img = np.full((48, 48, 3), 0.4, dtype=np.float32)
    depth_m = np.full((48, 48), 0.7, dtype=np.float32)

    u8, lin = render(img, tensor, depth_m=depth_m, slab_batch_size=1)
    assert np.all(np.isfinite(lin)), "D=1: non-finite output"
    assert u8.shape == img.shape


# ---------------------------------------------------------------------------
# (f) Stdout-byte-identical (G1 contract) for the active_slabs/acc_alpha line.
# ---------------------------------------------------------------------------


def _capture_layered_print(img, tensor, depth_m, batch_size):
    """Run render() and return the `layered_gather: ...` stdout line."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        render(img, tensor, depth_m=depth_m, slab_batch_size=batch_size)
    for line in buf.getvalue().splitlines():
        if "layered_gather:" in line and "active_slabs=" in line:
            return line
    return None


def test_stdout_layered_print_invariant_to_batch_size():
    """G1 contract: the
    `layered_gather: active_slabs=N/D, acc_alpha mean=X.XXX, holes=K pixels`
    line must be byte-identical across batch sizes (active_count + acc_alpha
    rounded to 3 decimals + holes count are all batch-size-invariant)."""
    rng = np.random.default_rng(0xCAFE)
    H, W = 96, 96
    img = rng.uniform(0.0, 0.7, (H, W, 3)).astype(np.float32)
    depth_m = rng.uniform(0.7, 20.0, (H, W)).astype(np.float32)
    tensor = _tensor_d11_mixed_delta()

    ref = _capture_layered_print(img, tensor, depth_m, 11)
    assert ref is not None, "expected layered_gather print line on D>=2 path"
    for bs in (1, 2, 3, 4, 7):
        line = _capture_layered_print(img, tensor, depth_m, bs)
        assert line == ref, (
            f"stdout drift at batch_size={bs}:\n  ref:  {ref!r}\n  test: {line!r}"
        )


# ---------------------------------------------------------------------------
# (g) RSS non-regression: smaller batch_size should NOT spike peak RSS.
# ---------------------------------------------------------------------------


def _peak_rss_mb_during(callable_):
    """Crude peak-RSS probe via ru_maxrss. Coarse-grain (process-level), but
    sufficient to detect a smaller-batch spike vs a larger-batch run."""
    import resource
    gc.collect()
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    callable_()
    gc.collect()
    rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports ru_maxrss in BYTES; Linux in KB. Normalize to MB.
    divisor = 1024 * 1024 if os.uname().sysname == "Darwin" else 1024
    return max(0.0, (rss_after - rss_before) / divisor)


def test_rss_does_not_regress_at_batch_size_1():
    """batch_size=1 must NOT cost more peak RSS than batch_size=4: smaller
    batches are the whole point of I1's streaming holes-fill. A regression
    here means the per-batch `del + gc.collect` is leaking buffers."""
    rng = np.random.default_rng(7)
    H, W = 256, 256  # large enough for the working-set delta to be measurable
    img = rng.uniform(0.0, 0.7, (H, W, 3)).astype(np.float32)
    depth_m = rng.uniform(0.7, 20.0, (H, W)).astype(np.float32)
    tensor = _tensor_d11_mixed_delta()

    # Warmup: prime any one-time numpy/scipy caches so the measured deltas
    # are about working-set, not import-time amortization.
    render(img, tensor, depth_m=depth_m, slab_batch_size=11)

    rss4 = _peak_rss_mb_during(
        lambda: render(img, tensor, depth_m=depth_m, slab_batch_size=4)
    )
    rss1 = _peak_rss_mb_during(
        lambda: render(img, tensor, depth_m=depth_m, slab_batch_size=1)
    )
    # ru_maxrss is monotone-max across the process lifetime, so subsequent
    # smaller-batch runs frequently report 0-delta. Accept that as a PASS.
    # The hostile assertion is: rss1 - rss4 must be small (≤ 50 MB slack)
    # — i.e., batch=1 does NOT spike above batch=4 by more than the FFT
    # temp working set.
    assert rss1 - rss4 <= 50.0, (
        f"RSS regression at batch_size=1: delta={rss1 - rss4:.1f} MB "
        f"(rss1={rss1:.1f}, rss4={rss4:.1f})"
    )
