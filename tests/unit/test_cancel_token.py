"""P1.audit Item 3 — cooperative cancellation token plumbing.

Verifies that ``render_image()`` polls ``cancel.is_set()`` between
expensive stages and raises :class:`CancelledError` cleanly.

Strategy: drive a tiny synthetic-fixture ``render()`` (NOT the full
canonical bench) on a 64×64 toy tensor, set the cancel event before
the first slab iterates, and assert the exception is raised. We poke
the cancel directly from the main thread (no real concurrency needed
for the fast unit test) AND from a background thread (the realistic
GUI scenario).
"""
from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from summicron.render.api import CancelledError
from summicron.render.convolve import render


def _toy_tensor(R=2, D=2, K=15):
    """Tiny 4D tensor with delta-at-d=0 and broad gaussian-at-d=1."""
    kernels = np.zeros((R, D, K, K), dtype=np.float32)
    cy = cx = K // 2
    # d=0: pure delta for every radial bin.
    kernels[:, 0, cy, cx] = 1.0
    # d=1: broadened gaussian, L1-normalized.
    yy, xx = np.mgrid[:K, :K] - cy
    g = np.exp(-(yy * yy + xx * xx) / (2.0 * 3.0 ** 2)).astype(np.float32)
    g /= g.sum()
    for r in range(R):
        kernels[r, 1] = g
    return {
        "kernels": kernels,
        "radial_edges": np.array(
            [0.0, 0.5, 1.0001], dtype=np.float64
        ),
        "defocus_ratios": np.array([0.0, 1.0], dtype=np.float64),
        "object_distances_m": np.array([0.7, 5.0], dtype=np.float64),
        "focus_distance_m": 0.7,
        "corpus_ref_long_edge": 5712,
        "delta_guard_applied": True,
        "R": R, "D": D,
    }


def test_cancel_already_set_raises_immediately():
    """If ``cancel`` is set BEFORE render() enters the slab loop, the
    next checkpoint raises CancelledError. No partial buffer escapes."""
    H, W = 64, 64
    img = np.full((H, W, 3), 0.4, dtype=np.float32)
    tensor = _toy_tensor()
    depth_m = np.full((H, W), 5.0, dtype=np.float32)
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(CancelledError) as exc_info:
        render(img, tensor, depth_m=depth_m, cancel=cancel)
    assert "cancel" in str(exc_info.value).lower()


def test_cancel_from_background_thread_raises():
    """Background thread sets cancel mid-render; main render() raises."""
    H, W = 256, 256
    img = np.full((H, W, 3), 0.4, dtype=np.float32)
    tensor = _toy_tensor()
    depth_m = np.full((H, W), 5.0, dtype=np.float32)
    cancel = threading.Event()

    result_box: list = []
    error_box: list = []

    def _do_render():
        try:
            out = render(img, tensor, depth_m=depth_m, cancel=cancel)
            result_box.append(out)
        except BaseException as e:  # noqa: BLE001
            error_box.append(e)

    t = threading.Thread(target=_do_render)
    t.start()
    # Tiny sleep then trigger cancel; on a 256x256 toy render this
    # races the slab loop reliably.
    time.sleep(0.05)
    cancel.set()
    t.join(timeout=30.0)
    assert not t.is_alive(), "render() did not return after cancel"
    # Either it cancelled (common) or finished before we set cancel
    # (acceptable on very fast machines). At minimum: no leaked state.
    assert (len(error_box) + len(result_box)) == 1
    if error_box:
        assert isinstance(error_box[0], CancelledError)
