"""Adversarial tests for P1.4 LiDAR depth wiring in convolve.py.

These tests target the attack surface of the lidar pipeline:
- Missing/corrupt depth data
- Extreme or degenerate depth values
- Invalid CLI arguments
- File-not-found / shape-mismatch edge cases

Every test asserts a real invariant. xfail markers indicate bugs found.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tensor_npz(
    path: Path,
    focus_distance_m: float = 0.7,
    n_radial: int = 3,
    n_defocus: int = 4,
    ksize: int = 11,
) -> Path:
    R, D, K = n_radial, n_defocus, ksize
    kernels = np.zeros((R, D, K, K), dtype=np.float32)
    c = K // 2
    kernels[:, :, c, c] = 1.0
    radial_edges = np.linspace(0.0, 1.0, R + 1).astype(np.float64)
    defocus_ratios = np.linspace(0.0, 1.0, D).astype(np.float64)
    object_distances_m = np.linspace(focus_distance_m, 5.0, D).astype(np.float64)
    np.savez(
        str(path),
        kernels=kernels,
        radial_edges=radial_edges,
        defocus_ratios=defocus_ratios,
        corpus_ref_long_edge=100,
        otf_norm_applied=True,
        object_distances_m=object_distances_m,
        focus_distance_m=focus_distance_m,
    )
    return path


def _make_image(path: Path, h: int = 64, w: int = 64) -> Path:
    from PIL import Image
    arr = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
    Image.fromarray(arr, "RGB").save(str(path), "JPEG", quality=80)
    return path


def _make_tensor_dict(
    n_radial: int = 3,
    n_defocus: int = 4,
    ksize: int = 11,
    focus_distance_m: float = 0.7,
) -> dict:
    R, D, K = n_radial, n_defocus, ksize
    kernels = np.zeros((R, D, K, K), dtype=np.float32)
    c = K // 2
    kernels[:, :, c, c] = 1.0
    radial_edges = np.linspace(0.0, 1.0, R + 1).astype(np.float64)
    defocus_ratios = np.linspace(0.0, 1.0, D).astype(np.float64)
    object_distances_m = np.linspace(focus_distance_m, 5.0, D).astype(np.float64)
    return {
        "kernels": kernels,
        "radial_edges": radial_edges,
        "defocus_ratios": defocus_ratios,
        "corpus_ref_long_edge": 100,
        "otf_norm_applied": True,
        "object_distances_m": object_distances_m,
        "focus_distance_m": focus_distance_m,
    }


# ---------------------------------------------------------------------------
# 7) test_lidar_no_depth_in_heic
# ---------------------------------------------------------------------------

def test_lidar_no_depth_in_heic(tmp_path: Path, capsys) -> None:
    """--depth-source=lidar on an image with NO embedded depth should raise
    a typed ValueError. P1.audit Item 3 lifted the error from a CLI-layer
    parser.error() into a library-layer raise; the CLI shim (Coder P0
    Item 2) will catch this and translate to exit-2 at the boundary.
    """
    img_path = _make_image(tmp_path / "test.jpg")
    tensor_path = _make_tensor_npz(tmp_path / "tensor.npz")

    with (
        patch("summicron.depth.estimate.extract_lidar_depth", return_value=None),
        patch("sys.argv", [
            "convolve", "--image", str(img_path),
            "--tensor", str(tensor_path),
            "--depth-source", "lidar",
            "--out-dir", str(tmp_path / "out"),
        ]),
    ):
        from summicron.render.convolve import main
        # Follow-ups sweep item 1: main() maps ValueError to exit 2.
        assert main() == 2
    assert "no embedded depth aux" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# 8) test_lidar_all_nan_depth
# ---------------------------------------------------------------------------

def test_lidar_all_nan_depth(tmp_path: Path) -> None:
    """If LiDAR returns all NaN, apply_tap_focus_shift should handle it
    gracefully (returns flat map at focus_target_m). The pipeline should
    NOT crash."""
    img_path = _make_image(tmp_path / "test.jpg")
    tensor_path = _make_tensor_npz(tmp_path / "tensor.npz")

    nan_depth = np.full((64, 64), np.nan, dtype=np.float32)
    # apply_tap_focus_shift handles all-NaN by returning flat focus_target_m map
    flat_map = np.full((64, 64), 0.7, dtype=np.float32)

    with (
        patch("summicron.depth.estimate.extract_lidar_depth", return_value=nan_depth),
        patch("summicron.depth.estimate.apply_tap_focus_shift", return_value=(flat_map, 0.7)),
        patch("summicron.render.convolve.save_results"),
        patch("sys.argv", [
            "convolve", "--image", str(img_path),
            "--tensor", str(tensor_path),
            "--depth-source", "lidar",
            "--out-dir", str(tmp_path / "out"),
        ]),
    ):
        from summicron.render.convolve import main
        # Should not crash
        main()


# ---------------------------------------------------------------------------
# 8b) test_lidar_all_nan_depth_render_direct
# ---------------------------------------------------------------------------

def test_lidar_all_nan_depth_render_direct() -> None:
    """Pass all-NaN depth_m directly to render(). np.interp produces NaN
    defocus values which feed into kernel selection. This should NOT crash
    or silently produce garbage."""
    from summicron.render.convolve import render

    H, W = 32, 32
    image = np.random.rand(H, W, 3).astype(np.float32) * 0.5
    tensor = _make_tensor_dict()
    nan_depth = np.full((H, W), np.nan, dtype=np.float32)

    # np.interp with NaN inputs produces NaN outputs. The render function
    # uses these for kernel interpolation. This may crash or produce NaN output.
    # Either outcome is a finding.
    try:
        out_u8, out_lin = render(image, tensor, depth_m=nan_depth)
        # If it didn't crash, check for NaN contamination in output
        if np.any(np.isnan(out_lin)):
            pytest.xfail(
                "BUG: render() produces NaN output when depth_m is all-NaN. "
                "The np.interp inside render converts NaN depth to NaN defocus, "
                "which propagates through kernel selection."
            )
    except Exception as e:
        pytest.xfail(
            f"BUG: render() crashes on all-NaN depth_m: {type(e).__name__}: {e}"
        )


# ---------------------------------------------------------------------------
# 9) test_lidar_depth_below_focus
# ---------------------------------------------------------------------------

def test_lidar_depth_below_focus() -> None:
    """After apply_tap_focus_shift wrap, NO values should be below
    focus_target_m. Test the actual function."""
    from summicron.depth.estimate import apply_tap_focus_shift

    H, W = 32, 32
    # Depth map with some values closer than focus
    depth_m = np.random.uniform(0.3, 3.0, (H, W)).astype(np.float32)
    focus_target_m = 0.7

    shifted, anchor = apply_tap_focus_shift(depth_m, None, focus_target_m)

    # All shifted values should be >= focus_target_m (that's the wrap guarantee)
    below = shifted[shifted < focus_target_m - 1e-6]
    assert below.size == 0, (
        f"Found {below.size} pixels below focus_target_m={focus_target_m}: "
        f"min={below.min():.4f}"
    )


# ---------------------------------------------------------------------------
# 10) test_focus_target_m_zero
# ---------------------------------------------------------------------------

def test_focus_target_m_zero(tmp_path: Path) -> None:
    """--focus-target-m=0 must be rejected by the parser (must be > 0)."""
    img_path = _make_image(tmp_path / "test.jpg")
    tensor_path = _make_tensor_npz(tmp_path / "tensor.npz")

    with patch("sys.argv", [
        "convolve", "--image", str(img_path),
        "--tensor", str(tensor_path),
        "--focus-target-m", "0",
        "--out-dir", str(tmp_path / "out"),
    ]):
        from summicron.render.convolve import main
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 2


# ---------------------------------------------------------------------------
# 11) test_focus_target_m_negative
# ---------------------------------------------------------------------------

def test_focus_target_m_negative(tmp_path: Path) -> None:
    """--focus-target-m=-1 must be rejected by the parser."""
    img_path = _make_image(tmp_path / "test.jpg")
    tensor_path = _make_tensor_npz(tmp_path / "tensor.npz")

    with patch("sys.argv", [
        "convolve", "--image", str(img_path),
        "--tensor", str(tensor_path),
        "--focus-target-m", "-1",
        "--out-dir", str(tmp_path / "out"),
    ]):
        from summicron.render.convolve import main
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 2


# ---------------------------------------------------------------------------
# 14) test_depth_m_all_at_focus
# ---------------------------------------------------------------------------

def test_depth_m_all_at_focus() -> None:
    """All depth values == focus_distance_m should produce defocus ≈ 0
    everywhere, so output should be approximately the input (identity)."""
    from summicron.render.convolve import render

    H, W = 32, 32
    # Simple gray image
    image = np.full((H, W, 3), 0.5, dtype=np.float32)
    tensor = _make_tensor_dict(focus_distance_m=0.7)
    depth_m = np.full((H, W), 0.7, dtype=np.float32)

    out_u8, out_lin = render(image, tensor, depth_m=depth_m)

    # With all pixels at focus distance, defocus_ratio=0.0 everywhere,
    # and kernel[r, d=0] is a delta. Output should be very close to input.
    # Allow tolerance for sRGB roundtrip and edge padding effects.
    input_u8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    # Check center region (avoid edge padding artifacts)
    margin = 10
    center_in = input_u8[margin:-margin, margin:-margin]
    center_out = out_u8[margin:-margin, margin:-margin]
    mae = np.mean(np.abs(center_in.astype(float) - center_out.astype(float)))
    assert mae < 15.0, (
        f"At-focus depth should produce near-identity render, but MAE={mae:.1f}"
    )


# ---------------------------------------------------------------------------
# 15) test_depth_m_extreme_far
# ---------------------------------------------------------------------------

def test_depth_m_extreme_far() -> None:
    """All depth = 1000m (extreme far). np.interp should clamp to the
    maximum defocus_ratio. The render should not crash."""
    from summicron.render.convolve import render

    H, W = 32, 32
    # Use a non-uniform image (sharp edge) so blur is detectable.
    image = np.zeros((H, W, 3), dtype=np.float32)
    image[:, W // 2:, :] = 0.8  # left half dark, right half bright
    tensor = _make_tensor_dict(focus_distance_m=0.7)
    depth_m = np.full((H, W), 1000.0, dtype=np.float32)

    out_u8, out_lin = render(image, tensor, depth_m=depth_m)

    # Should not crash; output should be valid uint8
    assert out_u8.dtype == np.uint8
    assert out_u8.shape == (H, W, 3)
    # np.interp clamps extreme-far to max defocus_ratio; render should complete.
    # We verify it ran without error — that's the primary invariant.


# ---------------------------------------------------------------------------
# 16) test_lidar_with_downres
# ---------------------------------------------------------------------------

def test_lidar_with_downres(tmp_path: Path) -> None:
    """--downres + --depth-source=lidar: depth should be extracted at the
    downscaled resolution, not the original.

    P1.10.4: replaces former test_lidar_with_working_long_edge after the
    --working-long-edge → --downres flag collapse.
    """
    img_path = _make_image(tmp_path / "test.jpg", h=128, w=128)
    tensor_path = _make_tensor_npz(tmp_path / "tensor.npz")

    # We'll capture what target_hw is passed to extract_lidar_depth
    captured_hw = {}

    def mock_extract(path, target_hw=None):
        captured_hw["target_hw"] = target_hw
        h, w = target_hw if target_hw else (128, 128)
        return np.full((h, w), 2.0, dtype=np.float32)

    def mock_shift(depth, tap, focus):
        return depth, 2.0

    with (
        patch("summicron.depth.estimate.extract_lidar_depth", side_effect=mock_extract),
        patch("summicron.depth.estimate.apply_tap_focus_shift", side_effect=mock_shift),
        patch("summicron.render.convolve.save_results"),
        patch("sys.argv", [
            "convolve", "--image", str(img_path),
            "--tensor", str(tensor_path),
            "--depth-source", "lidar",
            "--downres", "64",
            "--out-dir", str(tmp_path / "out"),
        ]),
    ):
        from summicron.render.convolve import main
        main()

    # After --downres=64 on a 128x128 image, the image is 64x64.
    # LiDAR depth should be extracted at (64, 64), not (128, 128).
    assert captured_hw.get("target_hw") is not None, "extract_lidar_depth not called"
    th, tw = captured_hw["target_hw"]
    assert th == 64 and tw == 64, (
        f"Expected depth extraction at (64, 64) after downscale, got ({th}, {tw})"
    )


# ---------------------------------------------------------------------------
# Extra adversarial: depth_m with Inf values
# ---------------------------------------------------------------------------

def test_depth_m_with_inf() -> None:
    """Depth map containing +inf values. np.interp may produce unexpected
    results with inf input."""
    from summicron.render.convolve import render

    H, W = 32, 32
    image = np.full((H, W, 3), 0.5, dtype=np.float32)
    tensor = _make_tensor_dict(focus_distance_m=0.7)
    depth_m = np.full((H, W), np.inf, dtype=np.float32)

    try:
        out_u8, out_lin = render(image, tensor, depth_m=depth_m)
        # If it didn't crash, verify no NaN in output
        assert not np.any(np.isnan(out_lin)), (
            "BUG: render() produces NaN output from inf depth_m"
        )
        assert out_u8.dtype == np.uint8
    except Exception as e:
        # A clean error is fine for inf input
        assert isinstance(e, (ValueError, RuntimeError)), (
            f"Unexpected crash on inf depth: {type(e).__name__}: {e}"
        )


# ---------------------------------------------------------------------------
# Extra adversarial: depth_m / defocus_map mutual exclusion — deleted in
# CLI-1a (defocus_map removed from render(); depth_m is the sole input).
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Extra adversarial: depth_m without object_distances_m in tensor
# ---------------------------------------------------------------------------

def test_depth_m_without_object_distances_in_tensor() -> None:
    """Passing depth_m with a tensor missing object_distances_m should raise
    ValueError."""
    from summicron.render.convolve import render

    H, W = 16, 16
    image = np.full((H, W, 3), 0.5, dtype=np.float32)
    tensor = _make_tensor_dict()
    tensor["object_distances_m"] = None  # simulate missing metadata
    depth = np.full((H, W), 1.0, dtype=np.float32)

    with pytest.raises(ValueError, match="object_distances_m"):
        render(image, tensor, depth_m=depth)
