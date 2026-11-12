"""Adversarial tests for Item A: meters end-to-end (extract_lidar_depth +
find_anchor_depth + build_depth_defocus_map unit-agnostic invariants).

These tests use the StubHeif infrastructure pattern from
test_extract_lidar_depth_adversarial.py to inject arbitrary HEIC depth
metadata (pillow_heif's encoder cannot write d_min/d_max directly).
"""
from __future__ import annotations

import sys
import types
import warnings
from pathlib import Path

import numpy as np
import pytest

from summicron.depth import estimate as depth_mod
from summicron.depth.estimate import (
    build_depth_defocus_map,
    extract_lidar_depth,
    find_anchor_depth,
)

HEIC_REAL = Path(__file__).resolve().parents[2] / "iphoneImage" / "IMG_6651.HEIC"


# ---------------------------------------------------------------------------
# Stub infrastructure (mirror of test_extract_lidar_depth_adversarial)
# ---------------------------------------------------------------------------

class _StubDepthImage:
    def __init__(self, raw: np.ndarray, metadata: dict | None):
        self._raw = raw
        self.info = {"metadata": metadata} if metadata is not None else {}

    def __array__(self, dtype=None):
        if dtype is not None:
            return self._raw.astype(dtype)
        return self._raw


class _StubHeif:
    def __init__(self, depth_imgs):
        self.info = {"depth_images": depth_imgs} if depth_imgs is not None else {}


class _StubModule(types.ModuleType):
    def __init__(self, depth_imgs):
        super().__init__("pillow_heif")
        self._depth_imgs = depth_imgs

    def open_heif(self, _path, convert_hdr_to_8bit=False):  # noqa: ARG002
        return _StubHeif(self._depth_imgs)


@pytest.fixture
def stub_heif(monkeypatch):
    def _install(depth_imgs):
        stub = _StubModule(depth_imgs)
        monkeypatch.setitem(sys.modules, "pillow_heif", stub)
    return _install


# ---------------------------------------------------------------------------
# 1. d_min == d_max -> degenerate disparity range
# ---------------------------------------------------------------------------

def test_degenerate_dmin_equals_dmax(stub_heif):
    """Equal d_min/d_max -> disp constant -> depth_m constant. find_anchor_depth
    on a constant array should return that same value."""
    raw = np.full((16, 16), 128, dtype=np.uint8)
    di = _StubDepthImage(raw, {"d_min": 2.0, "d_max": 2.0, "representation_type": 1})
    stub_heif([di])
    depth_m = extract_lidar_depth("dummy.heic")
    assert depth_m is not None
    # disp = 2 + (raw/255)*0 = 2 -> depth_m = 1/2 = 0.5
    assert np.allclose(depth_m, 0.5, atol=1e-5), f"expected 0.5m flat, got {depth_m.mean()}"
    anchor = find_anchor_depth(depth_m, percentile=2.0)
    assert anchor == pytest.approx(0.5, abs=1e-5)


# ---------------------------------------------------------------------------
# 2. Negative d_min -> negative disparity -> 1/maximum(disp,1e-6) == 1e6 -> clipped to 1000
#    Production silently produces clipped values; adversarial: should reject.
# ---------------------------------------------------------------------------

@pytest.mark.xfail(
    reason="BUG: extract_lidar_depth accepts negative d_min/d_max (corrupt metadata) "
           "and silently clips depths to far-cap. No validation/warning."
)
def test_negative_dmin_rejected(stub_heif):
    raw = np.full((8, 8), 0, dtype=np.uint8)
    di = _StubDepthImage(raw, {"d_min": -1.0, "d_max": 2.0, "representation_type": 1})
    stub_heif([di])
    with pytest.raises((ValueError, AssertionError)):
        extract_lidar_depth("dummy.heic")


@pytest.mark.xfail(
    reason="BUG: extract_lidar_depth accepts negative d_max (corrupt metadata) silently."
)
def test_negative_dmax_rejected(stub_heif):
    raw = np.full((8, 8), 255, dtype=np.uint8)
    di = _StubDepthImage(raw, {"d_min": -3.0, "d_max": -0.5, "representation_type": 1})
    stub_heif([di])
    with pytest.raises((ValueError, AssertionError)):
        extract_lidar_depth("dummy.heic")


# ---------------------------------------------------------------------------
# 3. Very large d_max -> near-zero depth -> within far-cap [0.1, 1000]
# ---------------------------------------------------------------------------

def test_huge_dmax_clipped_to_floor(stub_heif):
    """d_max=1000 1/m at raw=255 -> depth ~ 1/1000 = 0.001m ; clipped to 0.1m floor."""
    raw = np.full((8, 8), 255, dtype=np.uint8)
    di = _StubDepthImage(raw, {"d_min": 0.5, "d_max": 1000.0, "representation_type": 1})
    stub_heif([di])
    depth_m = extract_lidar_depth("dummy.heic")
    assert depth_m is not None
    assert depth_m.min() >= 0.1 - 1e-5, f"floor not enforced: min={depth_m.min()}"
    assert depth_m.max() <= 1e3 + 1e-3
    assert np.isfinite(depth_m).all()


# ---------------------------------------------------------------------------
# 4. All-zero raw uint8 with positive d_min,d_max -> disp=d_min everywhere
#    -> constant depth_m -> find_anchor_depth handles constant input
# ---------------------------------------------------------------------------

def test_all_zero_raw_constant_depth_anchor(stub_heif):
    raw = np.zeros((24, 32), dtype=np.uint8)
    di = _StubDepthImage(raw, {"d_min": 1.0, "d_max": 5.0, "representation_type": 1})
    stub_heif([di])
    depth_m = extract_lidar_depth("dummy.heic")
    assert depth_m is not None
    # disp = 1 + (0/255)*4 = 1 -> depth = 1/1 = 1.0 (no clip)
    assert np.allclose(depth_m, 1.0, atol=1e-5)
    anchor = find_anchor_depth(depth_m, percentile=2.0)
    assert anchor == pytest.approx(1.0, abs=1e-4)


# ---------------------------------------------------------------------------
# 5. Real-data: anchor on IMG_6651 must be in METERS (~0.4-0.5m), NOT [0,1]
# ---------------------------------------------------------------------------

def test_real_anchor_is_meters_not_normalized():
    if not HEIC_REAL.exists():
        pytest.skip("IMG_6651.HEIC fixture not present.")
    depth_m = extract_lidar_depth(HEIC_REAL, target_hw=(768, 576))
    anchor = find_anchor_depth(depth_m, percentile=2.0)
    # If it were normalized [0,1], p2 of typical depth would be ~0.0..0.05.
    # We *demand* metric: > 0.1m floor and < 5m for IMG_6651 stuffed-animal scene.
    assert anchor > 0.1, (
        f"anchor {anchor:.4f} <= 0.1 looks normalized, not meters"
    )
    assert anchor < 5.0, f"anchor {anchor:.4f} > 5m unexpected for IMG_6651"
    # Per spec: ~0.43m
    assert 0.30 <= anchor <= 0.55, f"anchor {anchor:.4f} far from spec'd 0.43m"


# ---------------------------------------------------------------------------
# 6. Reciprocal-difference defocus is unit-agnostic: scaling depth by k should
#    produce the *same* defocus map (units cancel after p98 normalisation).
# ---------------------------------------------------------------------------

def test_defocus_unit_agnostic_under_scalar_multiply():
    """build_depth_defocus_map: scale depth+anchor by k -> identical defocus."""
    rng = np.random.default_rng(seed=42)
    z1 = (rng.uniform(0.5, 5.0, size=(40, 60))).astype(np.float32)
    anchor1 = 1.2
    d1 = build_depth_defocus_map(z1, anchor1)

    k = 7.3  # arbitrary scalar
    z2 = (z1 * k).astype(np.float32)
    anchor2 = anchor1 * k
    d2 = build_depth_defocus_map(z2, anchor2)

    # Reciprocal-difference: |1/(kz) - 1/(k*a)| = (1/k) * |1/z - 1/a|
    # -> raw scales by 1/k, p98 also scales by 1/k -> ratio identical.
    assert np.allclose(d1, d2, atol=1e-4), (
        f"defocus map not unit-invariant under x{k}: max|Δ|="
        f"{np.abs(d1 - d2).max():.5f}"
    )


# ---------------------------------------------------------------------------
# 7. Constant depth feeding build_depth_defocus_map: triggers flat-scene path
# ---------------------------------------------------------------------------

def test_defocus_constant_depth_flat_scene_warning():
    z = np.full((32, 32), 1.5, dtype=np.float32)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        d = build_depth_defocus_map(z, 1.5)
    msgs = " ".join(str(w.message).lower() for w in caught)
    assert "flat" in msgs, f"expected flat-scene warning; got {[str(w.message) for w in caught]}"
    # Output should be ~zero (small * 0.01 in flat path)
    assert d.max() < 1e-4
