"""Unit tests for summicron.depth.estimate.extract_lidar_depth.

Real iPhone fixture: IMG_6651.HEIC. Gates G1–G4 from the locked spec.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from summicron.depth.estimate import extract_lidar_depth

HEIC = Path(__file__).resolve().parents[2] / "iphoneImage" / "IMG_6651.HEIC"
TARGET_HW = (768, 576)


@pytest.fixture(scope="module")
def z_native() -> np.ndarray:
    if not HEIC.exists():
        pytest.skip("IMG_6651.HEIC fixture not present.")
    z = extract_lidar_depth(HEIC, target_hw=TARGET_HW)
    assert z is not None, "No depth aux image returned for IMG_6651."
    return z


def test_g1_non_degenerate_anchor(z_native):
    # 2nd-percentile (in meters) should be > 0.05m on a real LiDAR scene.
    p2 = float(np.percentile(z_native, 2))
    assert p2 > 0.05, f"2nd-percentile depth_m = {p2:.6f}m (degenerate)"


def test_g2_foreground_near_background_far(z_native):
    # Bottom-right near subject should be a small distance (< 1m);
    # center (far wall) should be larger (> 1m).
    assert z_native[-20, -20] < 1.0, (
        f"BR near subject = {z_native[-20, -20]:.3f}m; expected < 1.0m"
    )
    assert z_native[384, 288] > 1.0, (
        f"Center far = {z_native[384, 288]:.3f}m; expected > 1.0m"
    )


def test_g3_shape_dtype_meters_range(z_native):
    """Spec G2: 0.40 <= min <= 0.45 and 2.50 <= max <= 2.70 on IMG_6651."""
    assert z_native.shape == TARGET_HW
    assert z_native.dtype == np.float32
    assert 0.40 <= float(z_native.min()) <= 0.45, (
        f"depth_m.min={z_native.min():.4f} outside [0.40, 0.45]"
    )
    assert 2.50 <= float(z_native.max()) <= 2.70, (
        f"depth_m.max={z_native.max():.4f} outside [2.50, 2.70]"
    )


def test_g4_monotonicity_spot_check():
    """raw[i] > raw[j]  =>  depth_m[i] < depth_m[j] (higher raw = nearer = smaller meters)."""
    if not HEIC.exists():
        pytest.skip("IMG_6651.HEIC fixture not present.")
    import pillow_heif
    heif = pillow_heif.open_heif(str(HEIC), convert_hdr_to_8bit=False)
    di = heif.info["depth_images"][0]
    raw = np.asarray(di).astype(np.int32)
    # Extract depth_m at the native depth-aux shape (no resize, to compare directly).
    z_raw_shape = extract_lidar_depth(HEIC, target_hw=raw.shape)
    assert z_raw_shape.shape == raw.shape

    rng = np.random.default_rng(0)
    H, W = raw.shape
    n_pairs = 100
    n_ok = 0
    n_checked = 0
    tries = 0
    while n_checked < n_pairs and tries < 5000:
        tries += 1
        i = (int(rng.integers(H)), int(rng.integers(W)))
        j = (int(rng.integers(H)), int(rng.integers(W)))
        if raw[i] == raw[j]:
            continue
        n_checked += 1
        if raw[i] > raw[j]:
            if z_raw_shape[i] < z_raw_shape[j]:
                n_ok += 1
        else:
            if z_raw_shape[i] > z_raw_shape[j]:
                n_ok += 1
    # Expect 100% monotone given deterministic decode (allow tiny slack for ties).
    assert n_ok == n_checked, f"Monotonicity failed: {n_ok}/{n_checked}"


def test_returns_none_when_no_depth(tmp_path: Path):
    """HEIC without a depth aux image must return None, not raise."""
    # Build a minimal HEIC with no depth by re-encoding a dummy RGB array.
    import pillow_heif
    from PIL import Image

    pillow_heif.register_heif_opener()
    rgb = (np.random.default_rng(0).integers(0, 256, size=(64, 64, 3))
           .astype(np.uint8))
    pil = Image.fromarray(rgb, mode="RGB")
    out = tmp_path / "no_depth.heic"
    pil.save(str(out), format="HEIF")
    assert extract_lidar_depth(out, target_hw=(64, 64)) is None
