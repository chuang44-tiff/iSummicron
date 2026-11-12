"""Synthetic PSF tensor factory for shape-parameterization tests (P1.9.3 L9).

Produces a minimal valid render-ready tensor with arbitrary ``(R, D)``
shape: ``d=0`` slices are pure single-pixel deltas (mirroring the P1.9
build-time delta guard), ``d >= 1`` slices are small normalized
Gaussians. Used by tests that need to drive ``load_tensor`` +
``render`` through non-default tensor shapes without depending on a
shipped tensor.

The factory intentionally produces kernels that pass through
``load_tensor`` without triggering the
"otf_norm_applied=False AND delta_guard_applied=False" warning —
``delta_guard_applied=True`` is written.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def make_synthetic_tensor(
    R: int,
    D: int,
    H: int = 33,
    W: int = 33,
    *,
    corpus_ref_long_edge: int = 5712,
) -> dict:
    """Return an in-memory dict shaped like ``load_tensor`` output.

    Parameters
    ----------
    R : int
        Radial bin count. Must be >= 1.
    D : int
        Defocus / object-distance steps. Must be >= 1.
    H, W : int
        Kernel grid size. Default 33 (odd so delta can sit at the
        geometric center).
    corpus_ref_long_edge : int
        Written into the metadata field unchanged.

    Returns
    -------
    dict with keys:
        kernels, radial_edges, defocus_ratios, corpus_ref_long_edge,
        otf_norm_applied, delta_guard_applied, object_distances_m,
        focus_distance_m, R, D
    """
    if R < 1 or D < 1:
        raise ValueError(f"R and D must be >= 1; got R={R}, D={D}")
    if H < 1 or W < 1:
        raise ValueError(f"H and W must be >= 1; got H={H}, W={W}")

    kernels = np.zeros((R, D, H, W), dtype=np.float32)
    cy, cx = H // 2, W // 2

    # d=0: single-pixel delta per radial bin (P1.9 delta-guard shape).
    for r in range(R):
        kernels[r, 0, cy, cx] = 1.0

    # d>=1: small isotropic Gaussian, L1-normalized. Sigma grows mildly
    # with d so defocused slices are distinguishable.
    if D >= 2:
        ys = np.arange(H, dtype=np.float32) - cy
        xs = np.arange(W, dtype=np.float32) - cx
        yy, xx = np.meshgrid(ys, xs, indexing="ij")
        for d in range(1, D):
            sigma = 1.0 + 0.5 * d
            g = np.exp(-(yy * yy + xx * xx) / (2.0 * sigma * sigma)).astype(np.float32)
            g /= g.sum()
            for r in range(R):
                kernels[r, d] = g

    # radial_edges on [0, 1] with R+1 entries, monotone.
    radial_edges = np.linspace(0.0, 1.0, R + 1, dtype=np.float64)

    # defocus_ratios on [0, 1] with D entries; D=1 degenerates to [0.0].
    defocus_ratios = (
        np.linspace(0.0, 1.0, D, dtype=np.float64) if D > 1 else np.array([0.0], dtype=np.float64)
    )

    # Monotone object_distances_m from 0.7m (tensor's canonical focus) out.
    # Use 0.7, 1.0, 1.5, 2.2, 3.2 ... geometric-ish spread so triangulate
    # slab weighting has non-degenerate bin widths.
    object_distances_m = np.linspace(0.7, 0.7 + 0.3 * D, D, dtype=np.float64)
    focus_distance_m = float(object_distances_m[0])

    return {
        "kernels": kernels,
        "radial_edges": radial_edges,
        "defocus_ratios": defocus_ratios,
        "corpus_ref_long_edge": int(corpus_ref_long_edge),
        "otf_norm_applied": False,
        "delta_guard_applied": True,
        "object_distances_m": object_distances_m,
        "focus_distance_m": focus_distance_m,
        "R": int(R),
        "D": int(D),
    }


def write_synthetic_tensor(
    path: str | Path,
    R: int,
    D: int,
    H: int = 33,
    W: int = 33,
    *,
    corpus_ref_long_edge: int = 5712,
) -> Path:
    """Persist a synthetic tensor dict as an .npz compatible with load_tensor.

    Useful for tests that need to exercise ``load_tensor`` from disk
    (validation paths, I/O round-trips).
    """
    path = Path(path)
    tensor = make_synthetic_tensor(
        R, D, H=H, W=W, corpus_ref_long_edge=corpus_ref_long_edge
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        str(path),
        kernels=tensor["kernels"],
        radial_edges=tensor["radial_edges"],
        defocus_ratios=tensor["defocus_ratios"],
        object_distances_m=tensor["object_distances_m"],
        focus_distance_m=np.float64(tensor["focus_distance_m"]),
        corpus_ref_long_edge=np.int64(tensor["corpus_ref_long_edge"]),
        otf_norm_applied=np.bool_(tensor["otf_norm_applied"]),
        delta_guard_applied=np.bool_(tensor["delta_guard_applied"]),
    )
    return path
