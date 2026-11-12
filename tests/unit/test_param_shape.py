"""P1.9.3 L9 / G2 / G3 — synthetic shape parameterization smoke tests.

Drives ``load_tensor`` + ``render`` through non-default ``(R, D)`` tensor
shapes to prove the pipeline no longer gates on the legacy 12×11 literals.

Each synthetic shape runs through the (sole) layered-gather render path;
a shape that breaks shape-derivation fails loudly here.
"""
from __future__ import annotations

import numpy as np
import pytest

from summicron.render.convolve import load_tensor, render
from tests.fixtures.synthetic_tensor import make_synthetic_tensor, write_synthetic_tensor


# Shapes: include both the R<12/D<11 regime (synthetic (3,2)) and the
# R>12 regime (synthetic (24,11)). Default (12, 11) is covered by the
# canonical bench; here we add it as a sanity sample on a synthetic
# build to prove the shape-derivation path is self-consistent.
_SHAPES = [(3, 2), (12, 11), (24, 11)]


@pytest.mark.parametrize("shape", _SHAPES)
def test_render_smoke_non_default_shape(tmp_path, shape):
    """G3 — render() completes, output is finite, shape matches input image.

    Runs through a small image (64x64) to keep wall-time tiny. Depth map
    spans the tensor's object_distances_m range so layered gather has
    non-empty active slabs.
    """
    R, D = shape
    path = write_synthetic_tensor(
        tmp_path / f"syn_{R}x{D}.npz", R, D, H=21, W=21
    )
    tensor = load_tensor(path)
    assert tensor["R"] == R
    assert tensor["D"] == D

    H, W = 48, 64
    rng = np.random.default_rng(seed=0)
    img = rng.uniform(0.0, 1.0, size=(H, W, 3)).astype(np.float32)
    # Depth spans the full object_distances_m range so slab activations
    # are spread across D bins (layered_gather=True exercises > 1 bin).
    od = tensor["object_distances_m"]
    depth_m = np.linspace(
        float(od[0]), float(od[-1]), H * W, dtype=np.float32
    ).reshape(H, W)

    out_u8, out_lin = render(
        img, tensor, depth_m=depth_m
    )
    assert out_u8.shape == (H, W, 3)
    assert out_lin.shape == (H, W, 3)
    assert np.all(np.isfinite(out_lin)), "non-finite output"
    assert np.all(np.isfinite(out_u8.astype(np.float32))), "non-finite uint8"


def test_in_memory_synthetic_tensor_roundtrips_through_render():
    """L9 — make_synthetic_tensor (without on-disk round-trip) also works.

    Proves the factory's dict schema is directly usable as a render()
    tensor argument, not just as a disk artifact.
    """
    tensor = make_synthetic_tensor(R=4, D=5, H=21, W=21)
    H, W = 32, 32
    img = np.full((H, W, 3), 0.3, dtype=np.float32)
    depth_m = np.full((H, W), 1.0, dtype=np.float32)
    out_u8, out_lin = render(img, tensor, depth_m=depth_m)
    assert out_u8.shape == (H, W, 3)
    assert np.all(np.isfinite(out_lin))
