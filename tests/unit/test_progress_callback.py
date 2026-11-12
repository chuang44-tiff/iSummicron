"""P1.audit Item 3 — on_progress callback emits per-slab status.

Verifies (a) the renderer calls ``on_progress`` at least D_gate times
inside the slab loop, and (b) when ``on_progress is None`` the renderer
keeps the per-slab message OFF stdout (preserves CLI bit-id; the
canonical bench md5 ``38c75dd2…`` is the gate).
"""
from __future__ import annotations

import io
from contextlib import redirect_stdout

import numpy as np

from summicron.render.convolve import render


def _toy_tensor(R=2, D=11, K=15):
    kernels = np.zeros((R, D, K, K), dtype=np.float32)
    cy = cx = K // 2
    kernels[:, 0, cy, cx] = 1.0  # delta at d=0
    yy, xx = np.mgrid[:K, :K] - cy
    g = np.exp(-(yy * yy + xx * xx) / (2.0 * 3.0 ** 2)).astype(np.float32)
    g /= g.sum()
    for r in range(R):
        for d in range(1, D):
            kernels[r, d] = g
    return {
        "kernels": kernels,
        "radial_edges": np.array(
            [0.0, 0.5, 1.0001], dtype=np.float64
        ),
        "defocus_ratios": np.linspace(0.0, 1.0, D, dtype=np.float64),
        "object_distances_m": np.linspace(0.7, 5.0, D, dtype=np.float64),
        "focus_distance_m": 0.7,
        "corpus_ref_long_edge": 5712,
        "delta_guard_applied": True,
        "R": R, "D": D,
    }


def test_progress_callback_collects_at_least_one_message_per_slab():
    """Caller-supplied progress sink receives one 'Slab d/D_gate' line
    per slab loop iteration."""
    H, W = 64, 64
    img = np.full((H, W, 3), 0.4, dtype=np.float32)
    tensor = _toy_tensor(D=11)
    depth_m = np.full((H, W), 5.0, dtype=np.float32)
    msgs: list[str] = []
    render(img, tensor, depth_m=depth_m, on_progress=msgs.append)
    slab_msgs = [m for m in msgs if m.lstrip().startswith("Slab ")]
    # All 11 slabs are active on a flat-far depth map, so we expect
    # exactly D_gate messages — but allow >= for robustness against
    # future per-stage emits.
    assert len(slab_msgs) >= 11, (
        f"Expected >=11 slab progress messages, got {len(slab_msgs)}: "
        f"{slab_msgs!r}"
    )


def test_no_progress_means_no_slab_lines_on_stdout():
    """When on_progress=None, the per-slab message MUST NOT print to
    stdout — that would break the canonical bench md5 invariant."""
    H, W = 64, 64
    img = np.full((H, W, 3), 0.4, dtype=np.float32)
    tensor = _toy_tensor(D=4)
    depth_m = np.full((H, W), 5.0, dtype=np.float32)
    buf = io.StringIO()
    with redirect_stdout(buf):
        render(img, tensor, depth_m=depth_m)
    out = buf.getvalue()
    assert "Slab 1/" not in out, (
        f"Per-slab progress leaked to stdout in CLI mode: {out!r}"
    )
