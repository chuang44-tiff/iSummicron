"""End-to-end HDR render smoke test (macOS-only).

Exercises the full CLI path `main()` with `--hdr-source macos` against the
canonical vibe-bench HEIC and a synthetic minimal tensor. Verifies:
  - No guard RuntimeError / ValueError raised.
  - Output file exists.
  - Output uint8 values in [0, 255].
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
HDR_FIXTURE = REPO_ROOT / "iphoneImage" / "IMG_6651.HEIC"


def _make_minimal_tensor(path: Path) -> Path:
    """Identity-ish tensor: center-delta kernels so render is near-identity.

    P1.10.2 Cut 5: include object_distances_m + focus_distance_m so the
    new default --depth-source=lidar pipeline can interp depth_m → defocus.
    """
    R, D, K = 3, 4, 11
    kernels = np.zeros((R, D, K, K), dtype=np.float32)
    kernels[:, :, K // 2, K // 2] = 1.0
    radial_edges = np.linspace(0.0, 1.0, R + 1).astype(np.float64)
    defocus_ratios = np.linspace(0.0, 1.0, D).astype(np.float64)
    object_distances_m = np.linspace(0.7, 5.0, D).astype(np.float64)
    np.savez(
        str(path),
        kernels=kernels,
        radial_edges=radial_edges,
        defocus_ratios=defocus_ratios,
        corpus_ref_long_edge=8000,
        otf_norm_applied=True,
        object_distances_m=object_distances_m,
        focus_distance_m=np.float64(0.7),
    )
    return path


@pytest.mark.skipif(sys.platform != "darwin", reason="HDR path is macOS-only")
@pytest.mark.skipif(not HDR_FIXTURE.exists(), reason=f"missing {HDR_FIXTURE}")
def test_render_hdr_path_end_to_end(tmp_path: Path) -> None:
    """Full main() with --hdr-source macos on a real HDR HEIC.

    To keep the test fast we monkey-patch `load_hdr_heic` to return a small
    synthetic HDR array (values up to 4.0, `isfinite`-clean); the real
    `expandToHDR` decode is exercised in `test_hdr_load_macos.py`. This
    test's job is the end-to-end main() pipeline: loader branch, relaxed
    guard, skipped sRGB linearization, tonemap + gamma + save.
    """
    tensor_path = _make_minimal_tensor(tmp_path / "tensor.npz")
    out_path = tmp_path / "out.jpg"

    rng = np.random.default_rng(0)
    hdr_img = rng.uniform(0.0, 4.0, size=(128, 128, 3)).astype(np.float32)

    with (
        patch(
            "summicron.io.hdr_load_macos.load_hdr_heic",
            return_value=hdr_img,
        ),
        patch("sys.argv", [
            "convolve",
            "--image", str(HDR_FIXTURE),
            "--tensor", str(tensor_path),
            "--hdr-source", "macos",
            "--output", str(out_path),
        ]),
    ):
        from summicron.render.convolve import main
        main()

    assert out_path.exists(), f"Expected output at {out_path}"
    with Image.open(out_path) as im:
        arr = np.asarray(im.convert("RGB"))
    assert arr.dtype == np.uint8
    assert arr.max() <= 255
    assert arr.min() >= 0
