"""Unit tests for P1.2 distance-indexed renderer (depth_m entry-point)."""
from __future__ import annotations

import numpy as np
import pytest

from summicron.render.convolve import load_tensor, render, srgb_to_linear


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_distance_tensor(ksize: int = 33) -> dict:
    """Build a synthetic tensor with object_distances_m metadata.

    6 radial bins, 5 defocus levels mapped to distances
    [0.7, 1.0, 3.0, 10.0, 20.0] meters.
    defocus_ratios = [0.0, 0.25, 0.5, 0.75, 1.0].
    Every kernel is a centered delta so we can reason about identity.
    """
    R, D = 6, 5
    kernels = np.zeros((R, D, ksize, ksize), dtype=np.float32)
    c = ksize // 2
    kernels[:, :, c, c] = 1.0
    radial_edges = np.array([0.0, 0.15, 0.30, 0.45, 0.60, 0.80, 1.01], dtype=np.float64)
    defocus_ratios = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float64)
    object_distances_m = np.array([0.7, 1.0, 3.0, 10.0, 20.0], dtype=np.float64)
    focus_distance_m = 0.7
    return {
        "kernels": kernels,
        "radial_edges": radial_edges,
        "defocus_ratios": defocus_ratios,
        "object_distances_m": object_distances_m,
        "focus_distance_m": focus_distance_m,
        "corpus_ref_long_edge": 256,
        "otf_norm_applied": True,
    }


def _make_plain_tensor(ksize: int = 33) -> dict:
    """Tensor WITHOUT distance metadata (hand-crafted / legacy)."""
    R, D = 6, 4
    kernels = np.zeros((R, D, ksize, ksize), dtype=np.float32)
    c = ksize // 2
    kernels[:, :, c, c] = 1.0
    radial_edges = np.array([0.0, 0.15, 0.30, 0.45, 0.60, 0.80, 1.01], dtype=np.float64)
    defocus_ratios = np.array([0.0, 0.33, 0.67, 1.0], dtype=np.float64)
    return {
        "kernels": kernels,
        "radial_edges": radial_edges,
        "defocus_ratios": defocus_ratios,
        "object_distances_m": None,
        "focus_distance_m": None,
        "corpus_ref_long_edge": 256,
        "otf_norm_applied": True,
    }


def _flat_image(sz: int = 64, val: float = 0.5) -> np.ndarray:
    """Uniform sRGB image (sz, sz, 3) float32 in [0, 1]."""
    return np.full((sz, sz, 3), val, dtype=np.float32)


# ---------------------------------------------------------------------------
# T1: depth_m is required (CLI-1a)
# ---------------------------------------------------------------------------

def test_missing_depth_m_raises_typeerror():
    with pytest.raises(TypeError):
        render(_flat_image(64, 0.5), _make_distance_tensor())


# ---------------------------------------------------------------------------
# T2: depth_m basic — no crash, no NaN, correct shape
# ---------------------------------------------------------------------------

class TestDepthMBasic:
    """Render with depth_m across a range of distances."""

    def test_multi_distance_rows(self):
        """256x256 image with rows at different depths renders cleanly."""
        sz = 64  # keep small for speed
        img = _flat_image(sz, 0.5)
        tensor = _make_distance_tensor()

        # Build depth map: rows at different distances
        distances = [0.7, 1.0, 3.0, 10.0, 20.0]
        depth_m = np.zeros((sz, sz), dtype=np.float32)
        rows_per_band = sz // len(distances)
        for i, d in enumerate(distances):
            r0 = i * rows_per_band
            r1 = (i + 1) * rows_per_band if i < len(distances) - 1 else sz
            depth_m[r0:r1, :] = d

        out_u8, out_lin = render(img, tensor, depth_m=depth_m)
        assert out_u8.shape == (sz, sz, 3)
        assert out_u8.dtype == np.uint8
        assert not np.any(np.isnan(out_lin))
        assert not np.any(np.isinf(out_lin))


# ---------------------------------------------------------------------------
# T3: Focus distance = delta (near-identity)
# ---------------------------------------------------------------------------

class TestFocusDistanceDelta:
    """depth_m == focus_distance_m everywhere => d=0 => near-identity."""

    def test_focus_ssim(self):
        """depth_m = 0.7 everywhere. Center crop SSIM > 0.95 vs input.

        Uses a 256x256 image so the near-axis slab covers a meaningful
        central area; centre crop at 1/3 margin avoids edge/corner
        degradation from the radial slab cos^2 taper.
        """
        sz = 256
        np.random.seed(123)
        img = np.random.rand(sz, sz, 3).astype(np.float32) * 0.8 + 0.1
        tensor = _make_distance_tensor()

        depth_m = np.full((sz, sz), 0.7, dtype=np.float32)
        out_u8, out_lin = render(img, tensor, depth_m=depth_m)

        input_lin = srgb_to_linear(img)

        try:
            from skimage.metrics import structural_similarity
            # Tight center crop (inner 1/3) to stay inside near-axis slab
            margin = sz // 3
            c_in = input_lin[margin:-margin, margin:-margin]
            c_out = out_lin[margin:-margin, margin:-margin]
            ssim = structural_similarity(
                c_in, c_out, channel_axis=2, data_range=1.0
            )
            assert ssim > 0.95, f"SSIM {ssim:.4f} <= 0.95 for focus-distance depth"
        except ImportError:
            # Fallback: L-inf check on center crop
            margin = sz // 3
            c_in = input_lin[margin:-margin, margin:-margin]
            c_out = out_lin[margin:-margin, margin:-margin]
            diff = np.abs(c_out - c_in).max()
            assert diff < 0.05, f"Max diff {diff:.4f} at focus distance"


# ---------------------------------------------------------------------------
# T4: Clamp beyond max distance
# ---------------------------------------------------------------------------

class TestClampBeyondMax:
    """depth_m = 100m everywhere should clamp to d=1.0 without crashing."""

    def test_far_depth_no_crash(self):
        sz = 64
        img = _flat_image(sz, 0.5)
        tensor = _make_distance_tensor()

        depth_m = np.full((sz, sz), 100.0, dtype=np.float32)
        out_u8, out_lin = render(img, tensor, depth_m=depth_m)
        assert out_u8.shape == (sz, sz, 3)
        assert not np.any(np.isnan(out_lin))
        assert not np.any(np.isinf(out_lin))

    def test_far_depth_max_blur(self):
        """Far-clamp invariant: d=100m (beyond the last object distance)
        renders identically to d=20m (= object_distances_m[-1], max blur)."""
        sz = 64
        img = _flat_image(sz, 0.5)
        tensor = _make_distance_tensor()

        depth_far = np.full((sz, sz), 100.0, dtype=np.float32)
        out_far_u8, _ = render(img, tensor, depth_m=depth_far)

        depth_max = np.full((sz, sz), 20.0, dtype=np.float32)
        out_max_u8, _ = render(img, tensor, depth_m=depth_max)

        np.testing.assert_array_equal(out_far_u8, out_max_u8)


# T5 (mutual exclusion depth_m vs defocus_map) deleted in CLI-1a:
# defocus_map is removed from render(); depth_m is the only depth input.


# ---------------------------------------------------------------------------
# T6: Monotonicity gate on load_tensor
# ---------------------------------------------------------------------------

class TestMonotonicityGate:
    """object_distances_m must be monotonic and [0] == focus_distance_m."""

    def test_valid_tensor_passes(self, tmp_path):
        """Well-formed npz passes monotonicity gate."""
        ksize = 17
        R, D = 4, 3
        kernels = np.zeros((R, D, ksize, ksize), dtype=np.float32)
        c = ksize // 2
        kernels[:, :, c, c] = 1.0

        npz_path = tmp_path / "test_tensor.npz"
        np.savez(
            str(npz_path),
            kernels=kernels,
            radial_edges=np.array([0.0, 0.3, 0.6, 0.9, 1.01]),
            defocus_ratios=np.array([0.0, 0.5, 1.0]),
            object_distances_m=np.array([0.7, 3.0, 20.0]),
            focus_distance_m=np.float64(0.7),
            otf_norm_applied=True,
        )

        t = load_tensor(npz_path)
        assert t["object_distances_m"] is not None
        assert t["focus_distance_m"] == 0.7
        assert np.all(np.diff(t["object_distances_m"]) > 0)
        assert t["object_distances_m"][0] == t["focus_distance_m"]

    def test_non_monotonic_raises(self, tmp_path):
        """Non-monotonic object_distances_m triggers assertion."""
        ksize = 17
        R, D = 4, 3
        kernels = np.zeros((R, D, ksize, ksize), dtype=np.float32)
        c = ksize // 2
        kernels[:, :, c, c] = 1.0

        npz_path = tmp_path / "bad_tensor.npz"
        np.savez(
            str(npz_path),
            kernels=kernels,
            radial_edges=np.array([0.0, 0.3, 0.6, 0.9, 1.01]),
            defocus_ratios=np.array([0.0, 0.5, 1.0]),
            object_distances_m=np.array([0.7, 20.0, 3.0]),  # not monotonic!
            focus_distance_m=np.float64(0.7),
            otf_norm_applied=True,
        )

        # P1.9.4 L11: was AssertionError, now ValueError (uniform validation type).
        with pytest.raises(ValueError, match="monotonically increasing"):
            load_tensor(npz_path)

    def test_focus_mismatch_raises(self, tmp_path):
        """object_distances_m[0] != focus_distance_m triggers assertion."""
        ksize = 17
        R, D = 4, 3
        kernels = np.zeros((R, D, ksize, ksize), dtype=np.float32)
        c = ksize // 2
        kernels[:, :, c, c] = 1.0

        npz_path = tmp_path / "mismatch_tensor.npz"
        np.savez(
            str(npz_path),
            kernels=kernels,
            radial_edges=np.array([0.0, 0.3, 0.6, 0.9, 1.01]),
            defocus_ratios=np.array([0.0, 0.5, 1.0]),
            object_distances_m=np.array([0.7, 3.0, 20.0]),
            focus_distance_m=np.float64(1.0),  # mismatch!
            otf_norm_applied=True,
        )

        # P1.audit Item 4: assert→TensorValidationError(ValueError).
        with pytest.raises((AssertionError, ValueError), match="focus_distance_m"):
            load_tensor(npz_path)

    def test_missing_metadata_graceful(self, tmp_path):
        """Tensor without distance metadata loads with None fields."""
        ksize = 17
        R, D = 4, 3
        kernels = np.zeros((R, D, ksize, ksize), dtype=np.float32)
        c = ksize // 2
        kernels[:, :, c, c] = 1.0

        npz_path = tmp_path / "plain_tensor.npz"
        np.savez(
            str(npz_path),
            kernels=kernels,
            radial_edges=np.array([0.0, 0.3, 0.6, 0.9, 1.01]),
            defocus_ratios=np.array([0.0, 0.5, 1.0]),
            otf_norm_applied=True,
        )

        t = load_tensor(npz_path)
        assert t["object_distances_m"] is None
        assert t["focus_distance_m"] is None


# ---------------------------------------------------------------------------
# Extra: depth_m with plain tensor raises ValueError
# ---------------------------------------------------------------------------

class TestDepthMRequiresMetadata:
    """depth_m on tensor without object_distances_m raises ValueError."""

    def test_no_metadata_raises(self):
        sz = 32
        img = _flat_image(sz, 0.5)
        tensor = _make_plain_tensor()
        depth_m = np.full((sz, sz), 1.0, dtype=np.float32)

        with pytest.raises(ValueError, match="object_distances_m"):
            render(img, tensor, depth_m=depth_m)
