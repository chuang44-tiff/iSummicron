"""Adversarial tests for summicron.depth.estimate.extract_lidar_depth.

Focuses on the wrong-polarity fixture mandated by the Day-1 post-mortem
and canonical workflow: any gate that makes a brightness/polarity
assumption MUST have a flipped-convention test. Also covers degenerate
metadata, wrong format, missing file, extreme target_hw, and FULL-image
monotonicity (not just a 100-pair spot check).

Uses a stub HEIF object injected via sys.modules to avoid writing real
HEIC fixtures with arbitrary metadata (pillow_heif's encoder doesn't
support writing d_min/d_max/representation_type directly).
"""
from __future__ import annotations

import sys
import types
import warnings
from pathlib import Path

import numpy as np
import pytest

from summicron.depth import estimate as depth_mod
from summicron.depth.estimate import extract_lidar_depth

HEIC_REAL = Path(__file__).resolve().parents[2] / "iphoneImage" / "IMG_6651.HEIC"


# ---------------------------------------------------------------------------
# Stub infrastructure: inject a fake pillow_heif.open_heif for one call.
# ---------------------------------------------------------------------------

class _StubDepthImage:
    def __init__(self, raw: np.ndarray, metadata: dict | None):
        self._raw = raw
        if metadata is None:
            self.info = {}
        else:
            self.info = {"metadata": metadata}

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
    """Factory to install a stub pillow_heif for the duration of a test."""
    def _install(depth_imgs):
        stub = _StubModule(depth_imgs)
        monkeypatch.setitem(sys.modules, "pillow_heif", stub)
    return _install


# ---------------------------------------------------------------------------
# MANDATORY wrong-polarity fixture (per Day-1 post-mortem #1)
# ---------------------------------------------------------------------------

@pytest.mark.xfail(
    reason="BUG: extract_lidar_depth silently inverts the scene for a "
           "wrong-polarity HEIC; no detection / warning. A file whose raw "
           "convention is 'raw=0 means near' will produce inverted depth "
           "and the function has no way to flag it."
)
def test_wrong_polarity_fixture_should_warn(stub_heif):
    """Build a synthetic depth where raw=0 encodes NEAR (flipped from Apple's
    convention of raw=0 = far). The current formula
    ``disp = d_min + raw/255 * (d_max - d_min)`` treats raw=255 as high
    disparity (near), so a scene where raw ramps 0->255 left-to-right is
    decoded with left=far, right=near. If the author intended the flipped
    convention, the function returns inverted depth with no warning.

    This test asserts the function should either (a) emit a warning when it
    detects a suspicious/unknown convention, or (b) require an explicit
    convention parameter. Today it does neither -> xfail."""
    H, W = 32, 32
    raw = np.tile(np.linspace(0, 255, W, dtype=np.uint8), (H, 1))  # 0..255 L->R
    # Apple convention metadata: representation_type=1. We're feeding data
    # that was produced under the FLIPPED convention (raw=0 means near).
    # The caller has no way to tell the function this; the decoder should
    # warn about possible convention mismatch (e.g., when d_min, d_max are
    # both near-field values, or when distribution is unusual).
    meta = {"d_min": 0.01, "d_max": 0.05, "representation_type": 1}
    stub_heif([_StubDepthImage(raw, meta)])

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        z = extract_lidar_depth("/ignored.heic")
    # Demand SOME kind of polarity/convention warning be emitted.
    msgs = " ".join(str(wi.message).lower() for wi in w)
    assert any(
        kw in msgs for kw in ("polarity", "convention", "flipped", "inverted")
    ), f"No polarity warning emitted; warnings were: {[str(x.message) for x in w]}"
    assert z is not None


def test_wrong_polarity_decodes_something_finite(stub_heif):
    """Even under a flipped-convention fixture, the function must not crash
    and must return finite output. This is the minimum safety contract."""
    H, W = 32, 32
    raw = np.tile(np.linspace(0, 255, W, dtype=np.uint8), (H, 1))
    meta = {"d_min": 0.01, "d_max": 0.05, "representation_type": 1}
    stub_heif([_StubDepthImage(raw, meta)])
    z = extract_lidar_depth("/ignored.heic")
    assert z is not None
    assert z.shape == (H, W)
    assert np.isfinite(z).all()
    # Meters output: clipped to [0.1, 1e3]; sanity bounds (no [0,1] cap).
    assert 0.1 <= z.min() and z.max() <= 1e3


# ---------------------------------------------------------------------------
# File / format errors
# ---------------------------------------------------------------------------

def test_missing_file_clean_error(tmp_path: Path):
    """Nonexistent HEIC path must raise a clean IO error."""
    with pytest.raises((FileNotFoundError, OSError, Exception)):
        extract_lidar_depth(tmp_path / "does_not_exist.HEIC")


def test_jpg_file_does_not_crash_bizarrely(tmp_path: Path):
    """A JPG passed where HEIC is expected should raise a clean error,
    not a segfault or silent None."""
    from PIL import Image
    p = tmp_path / "not_a_heic.jpg"
    Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8)).save(p, format="JPEG")
    with pytest.raises(Exception):
        extract_lidar_depth(p)


# ---------------------------------------------------------------------------
# Missing / empty / alternate depth_images
# ---------------------------------------------------------------------------

def test_empty_depth_images_list_returns_none(stub_heif):
    """depth_images=[] (present but empty) must return None, same as missing."""
    stub_heif([])
    assert extract_lidar_depth("/ignored.heic") is None


def test_missing_depth_images_key_returns_none(stub_heif):
    """No depth_images key at all must return None."""
    stub_heif(None)
    assert extract_lidar_depth("/ignored.heic") is None


def test_representation_type_2_warns_and_returns(stub_heif):
    """rep_type=2 (Z-axis depth, unsupported) must emit a warning and fall
    back to a heuristic without crashing."""
    raw = np.arange(256, dtype=np.uint8).reshape(16, 16)
    meta = {"d_min": 0.1, "d_max": 2.0, "representation_type": 2}
    stub_heif([_StubDepthImage(raw, meta)])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        z = extract_lidar_depth("/ignored.heic")
    assert z is not None
    assert any("representation_type" in str(wi.message) for wi in w), \
        "Expected warning about unsupported representation_type"


def test_representation_type_missing_defaults_to_1(stub_heif):
    """Missing representation_type key should default to 1 (not crash)."""
    raw = np.arange(256, dtype=np.uint8).reshape(16, 16)
    meta = {"d_min": 0.1, "d_max": 2.0}  # no representation_type
    stub_heif([_StubDepthImage(raw, meta)])
    z = extract_lidar_depth("/ignored.heic")
    assert z is not None
    assert z.shape == (16, 16)


# ---------------------------------------------------------------------------
# Degenerate / invalid metadata
# ---------------------------------------------------------------------------

def test_d_min_equals_d_max_flat_depth(stub_heif):
    """Degenerate disparity (d_min == d_max) must not divide-by-zero;
    the 1e-8 guard in the normalize step should produce flat output."""
    raw = np.full((16, 16), 100, dtype=np.uint8)
    meta = {"d_min": 1.0, "d_max": 1.0, "representation_type": 1}
    stub_heif([_StubDepthImage(raw, meta)])
    z = extract_lidar_depth("/ignored.heic")
    assert z is not None
    assert np.isfinite(z).all()
    # Flat disparity -> flat depth_m (= 1/d_min). All pixels equal.
    assert z.max() - z.min() < 1e-5, f"Expected flat output, got ptp={z.max()-z.min()}"


def test_negative_d_min_does_not_crash(stub_heif):
    """Negative d_min is invalid metadata but must not crash the decoder —
    the 1e-6 guard on disp before inversion should protect."""
    raw = np.arange(256, dtype=np.uint8).reshape(16, 16)
    meta = {"d_min": -0.5, "d_max": 1.0, "representation_type": 1}
    stub_heif([_StubDepthImage(raw, meta)])
    z = extract_lidar_depth("/ignored.heic")
    assert z is not None
    assert np.isfinite(z).all()


def test_both_d_min_d_max_negative(stub_heif):
    """Both negative: all disp values negative -> clamped to 1e-6 -> uniform
    large depth -> flat output after normalize. Must not NaN/Inf."""
    raw = np.arange(256, dtype=np.uint8).reshape(16, 16)
    meta = {"d_min": -2.0, "d_max": -1.0, "representation_type": 1}
    stub_heif([_StubDepthImage(raw, meta)])
    z = extract_lidar_depth("/ignored.heic")
    assert z is not None
    assert np.isfinite(z).all()


# ---------------------------------------------------------------------------
# target_hw edge cases
# ---------------------------------------------------------------------------

def test_target_hw_extreme_aspect(stub_heif):
    """1x100 target — extreme anisotropic resize must not crash."""
    raw = np.arange(256, dtype=np.uint8).reshape(16, 16)
    meta = {"d_min": 0.1, "d_max": 2.0, "representation_type": 1}
    stub_heif([_StubDepthImage(raw, meta)])
    z = extract_lidar_depth("/ignored.heic", target_hw=(1, 100))
    assert z is not None
    assert z.shape == (1, 100)
    assert np.isfinite(z).all()


def test_target_hw_zero_raises(stub_heif):
    """target_hw=(0, 0) is invalid; PIL resize to 0 should error cleanly."""
    raw = np.arange(256, dtype=np.uint8).reshape(16, 16)
    meta = {"d_min": 0.1, "d_max": 2.0, "representation_type": 1}
    stub_heif([_StubDepthImage(raw, meta)])
    with pytest.raises(Exception):
        extract_lidar_depth("/ignored.heic", target_hw=(0, 0))


def test_target_hw_negative_raises(stub_heif):
    raw = np.arange(256, dtype=np.uint8).reshape(16, 16)
    meta = {"d_min": 0.1, "d_max": 2.0, "representation_type": 1}
    stub_heif([_StubDepthImage(raw, meta)])
    with pytest.raises(Exception):
        extract_lidar_depth("/ignored.heic", target_hw=(-8, 8))


# ---------------------------------------------------------------------------
# Full-image monotonicity (real fixture, not just 100 pairs)
# ---------------------------------------------------------------------------

def test_full_image_monotonicity_strict():
    """Per canonical workflow: verify monotonicity across the FULL image,
    not just a random 100-pair sample. For a strictly monotone decode
    (1/disp with disp linear in raw) higher raw must imply smaller z
    EVERYWHERE. We verify by sorting both arrays and checking rank
    correlation is -1 (perfectly anti-correlated)."""
    if not HEIC_REAL.exists():
        pytest.skip("IMG_6651.HEIC fixture not present.")
    import pillow_heif
    heif = pillow_heif.open_heif(str(HEIC_REAL), convert_hdr_to_8bit=False)
    di = heif.info["depth_images"][0]
    raw = np.asarray(di).astype(np.int32).ravel()
    z = extract_lidar_depth(HEIC_REAL, target_hw=np.asarray(di).shape).ravel()
    # For every unique raw value, z must be exactly one value (deterministic)
    # and the mapping raw -> z must be strictly decreasing.
    unique_raw = np.unique(raw)
    means = np.array([z[raw == r].mean() for r in unique_raw])
    # Strictly decreasing across unique raw values.
    diffs = np.diff(means)
    assert np.all(diffs < 0), (
        f"Mapping raw->z is not strictly decreasing everywhere; "
        f"num non-decreasing diffs = {int((diffs >= 0).sum())} / {len(diffs)}"
    )


# ---------------------------------------------------------------------------
# Polarity sanity with controlled synthetic fixture
# ---------------------------------------------------------------------------

def test_synthetic_ramp_polarity_is_apple_convention(stub_heif):
    """Canonical Apple convention: raw=255 -> high disparity -> small depth
    (near subject) -> z near 0 after min-max normalize. Feed a ramp 0->255
    and verify z decreases as raw increases."""
    H, W = 8, 256
    raw = np.tile(np.arange(256, dtype=np.uint8), (H, 1))
    meta = {"d_min": 0.1, "d_max": 2.0, "representation_type": 1}
    stub_heif([_StubDepthImage(raw, meta)])
    z = extract_lidar_depth("/ignored.heic")
    assert z is not None
    # Column 0 is far (large depth_m), column 255 is near (small depth_m).
    # With d_min=0.1, d_max=2.0: depth_m[col0] = 1/0.1 = 10m, depth_m[col255] = 1/2.0 = 0.5m.
    assert z[:, 0].mean() > z[:, -1].mean()
    assert z[:, 0].mean() > 5.0, f"Far column mean {z[:,0].mean():.3f}m; expected > 5m"
    assert z[:, -1].mean() < 1.0, f"Near column mean {z[:,-1].mean():.3f}m; expected < 1m"
