"""Adversarial tests for the HDR `render()` + `main()` path.

Targets:
  - `hdr_source` argument coercion (None, 'NONE', bad strings).
  - Guard boundary conditions on LDR path (max exactly 1.001, 1.0012, etc.).
  - HDR path non-finite gate (NaN, +Inf, -Inf, mixed).
  - Tonemap ORDERING: render() HDR must return linear pre-tonemap; main()
    applies the configured tonemap BEFORE linear_to_srgb.
  - CLI parsing: `--hdr-source=macos` equals-form, case sensitivity,
    default = none, mutual exclusion of hdr-source + various modes.
  - Dual-tensor + HDR end-to-end (no second-call LDR guard trip).
  - LDR regression: hdr_source='none' with legitimate [0, 0.95] input must
    remain bit-identical to pre-P1.8 path.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from summicron.render.convolve import render
from tests.fixtures.defocus import render_with_defocus

REPO_ROOT = Path(__file__).resolve().parents[2]
HDR_FIXTURE = REPO_ROOT / "iphoneImage" / "IMG_6651.HEIC"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _mini_tensor_dict() -> dict:
    R, D, K = 3, 4, 11
    kernels = np.zeros((R, D, K, K), dtype=np.float32)
    kernels[:, :, K // 2, K // 2] = 1.0
    return {
        "kernels": kernels,
        "radial_edges": np.linspace(0.0, 1.0, R + 1).astype(np.float64),
        "defocus_ratios": np.linspace(0.0, 1.0, D).astype(np.float64),
        "corpus_ref_long_edge": 8000,
        "otf_norm_applied": True,
        "object_distances_m": None,
        "focus_distance_m": None,
    }


def _mini_tensor_npz(path: Path) -> Path:
    R, D, K = 3, 4, 11
    kernels = np.zeros((R, D, K, K), dtype=np.float32)
    kernels[:, :, K // 2, K // 2] = 1.0
    # P1.10.2 Cut 5: include object_distances_m + focus_distance_m so the
    # new default --depth-source=lidar pipeline can interp depth_m → defocus.
    np.savez(
        str(path),
        kernels=kernels,
        radial_edges=np.linspace(0.0, 1.0, R + 1).astype(np.float64),
        defocus_ratios=np.linspace(0.0, 1.0, D).astype(np.float64),
        corpus_ref_long_edge=8000,
        otf_norm_applied=True,
        object_distances_m=np.linspace(0.7, 5.0, D).astype(np.float64),
        focus_distance_m=np.float64(0.7),
    )
    return path


# ---------------------------------------------------------------------------
# Guard-boundary tests (Linux-runnable; render() is platform-agnostic)
# ---------------------------------------------------------------------------


def test_ldr_guard_accepts_exact_one():
    """LDR path: max = 1.0 exactly must pass."""
    tensor = _mini_tensor_dict()
    img = np.ones((32, 32, 3), dtype=np.float32)
    r, _ = render_with_defocus(img, tensor, hdr_source="none")
    assert r.dtype == np.uint8


def test_ldr_guard_accepts_1_001():
    """LDR path: max = 1.001 is the declared boundary (<=); must pass."""
    tensor = _mini_tensor_dict()
    img = np.full((32, 32, 3), 1.001, dtype=np.float32)
    r, _ = render_with_defocus(img, tensor, hdr_source="none")
    assert r.dtype == np.uint8


def test_ldr_guard_rejects_just_above_boundary():
    """LDR path: max = 1.0015 must raise."""
    tensor = _mini_tensor_dict()
    img = np.full((32, 32, 3), 1.0015, dtype=np.float32)
    with pytest.raises(ValueError, match="Float input max"):
        render_with_defocus(img, tensor, hdr_source="none")


def test_ldr_guard_rejects_extreme_hdr_through_ldr_path():
    """LDR path: max = 5.0 must raise (the whole point of the guard)."""
    tensor = _mini_tensor_dict()
    img = np.full((32, 32, 3), 5.0, dtype=np.float32)
    with pytest.raises(ValueError, match="Float input max"):
        render_with_defocus(img, tensor, hdr_source="none")


def test_ldr_path_095_regression():
    """Regression: hdr_source='none' on LDR polarity (max=0.95) works as
    before. This is the wrong-polarity counterpart for the LDR guard."""
    tensor = _mini_tensor_dict()
    img = np.full((32, 32, 3), 0.95, dtype=np.float32)
    r, lin = render_with_defocus(img, tensor, hdr_source="none")
    assert r.dtype == np.uint8
    assert r.max() <= 255
    # Linear output non-trivially bright
    assert lin.max() > 0.5


def test_hdr_guard_accepts_max_well_above_one():
    """HDR path: max = 10.0 must pass (that's the whole point)."""
    tensor = _mini_tensor_dict()
    img = np.full((32, 32, 3), 10.0, dtype=np.float32)
    r, lin = render_with_defocus(img, tensor, hdr_source="macos")
    # render() HDR returns linear float32 identity with max preserved
    assert r.dtype == np.float32
    assert r.max() > 9.0  # no clipping inside render()


def test_hdr_guard_rejects_nan():
    tensor = _mini_tensor_dict()
    img = np.full((32, 32, 3), 2.0, dtype=np.float32)
    img[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        render_with_defocus(img, tensor, hdr_source="macos")


def test_hdr_guard_rejects_positive_infinity():
    tensor = _mini_tensor_dict()
    img = np.full((32, 32, 3), 2.0, dtype=np.float32)
    img[0, 0, 0] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        render_with_defocus(img, tensor, hdr_source="macos")


def test_hdr_guard_rejects_negative_infinity():
    tensor = _mini_tensor_dict()
    img = np.full((32, 32, 3), 2.0, dtype=np.float32)
    img[0, 0, 0] = -np.inf
    with pytest.raises(ValueError, match="non-finite"):
        render_with_defocus(img, tensor, hdr_source="macos")


def test_hdr_path_accepts_all_zero():
    """Zero HDR array must not crash the energy gate (trivial-conserve branch)."""
    tensor = _mini_tensor_dict()
    img = np.zeros((32, 32, 3), dtype=np.float32)
    r, _ = render_with_defocus(img, tensor, hdr_source="macos")
    assert np.isfinite(r).all()
    assert r.max() == 0.0


def test_hdr_path_accepts_leq_one_values():
    """HDR path must also accept <=1 values (SDR content decoded by expandToHDR
    still stays <=1). This is the loader wrong-polarity counterpart at the
    render level."""
    tensor = _mini_tensor_dict()
    img = np.full((32, 32, 3), 0.3, dtype=np.float32)
    r, lin = render_with_defocus(img, tensor, hdr_source="macos")
    assert np.isfinite(r).all()
    # No sRGB linearization in HDR path — output should track input directly
    assert abs(float(r.mean()) - 0.3) < 0.01


# ---------------------------------------------------------------------------
# Tonemap ordering
# ---------------------------------------------------------------------------


def test_render_hdr_does_not_apply_tonemap_internally():
    """Spec §3 load-bearing: render() must NOT inline tonemap when hdr_source
    != 'none'. Output should be linear float32 with values > 1.0."""
    tensor = _mini_tensor_dict()
    img = np.full((32, 32, 3), 4.0, dtype=np.float32)
    r, lin = render_with_defocus(img, tensor, hdr_source="macos")
    # If tonemap were inlined, max would be < 1.0 (4/(1+4) = 0.8).
    assert r.max() > 1.0, (
        f"render() HDR max={r.max()} <= 1.0 — tonemap appears to be inlined"
    )
    # And the two return values point at the same array by the spec contract.
    assert r is lin


# ---------------------------------------------------------------------------
# hdr_source argument coercion
# ---------------------------------------------------------------------------


def test_render_hdr_source_python_none_raises_or_behaves_as_ldr():
    """Passing Python None must not silently switch to HDR mode. This is a
    common slip (forgetting to set the default); the contract should be
    explicit. P1.8: render() now raises ValueError on invalid hdr_source."""
    tensor = _mini_tensor_dict()
    img = np.full((32, 32, 3), 0.5, dtype=np.float32)
    with pytest.raises(ValueError, match="hdr_source"):
        render_with_defocus(img, tensor, hdr_source=None)  # type: ignore[arg-type]


def test_render_hdr_source_unknown_string_falls_through_to_hdr():
    """P1.8: render() now restricts hdr_source to {'none', 'macos'} and
    raises ValueError on anything else (the conscious tightening the
    previous version of this test anticipated)."""
    tensor = _mini_tensor_dict()
    img = np.full((32, 32, 3), 2.0, dtype=np.float32)
    with pytest.raises(ValueError, match="hdr_source"):
        render_with_defocus(img, tensor, hdr_source="macos-gainmap")


def test_render_hdr_source_case_sensitivity_at_render_level():
    """render() is case-sensitive on hdr_source. P1.8: 'NONE' is rejected
    as invalid rather than silently falling into the HDR branch."""
    tensor = _mini_tensor_dict()
    img = np.full((32, 32, 3), 2.0, dtype=np.float32)
    with pytest.raises(ValueError, match="hdr_source"):
        render_with_defocus(img, tensor, hdr_source="NONE")


# ---------------------------------------------------------------------------
# CLI flag parsing
# ---------------------------------------------------------------------------


def test_cli_hdr_source_default_is_none(tmp_path: Path):
    """Missing --hdr-source must default to 'none' (LDR path)."""
    tensor_path = _mini_tensor_npz(tmp_path / "tensor.npz")
    out_path = tmp_path / "out.jpg"

    # Build a mini JPEG
    from PIL import Image

    arr = (np.ones((32, 32, 3), dtype=np.uint8) * 128)
    img_path = tmp_path / "in.jpg"
    Image.fromarray(arr, mode="RGB").save(str(img_path), "JPEG")

    # P1.10.2 Cut 5: --depth-source default flipped to "lidar"; mock the
    # LiDAR extraction since the input is a synthetic JPEG with no aux.
    depth_fake = np.full((32, 32), 0.7, dtype=np.float32)

    with (
        patch("summicron.depth.estimate.extract_lidar_depth", return_value=depth_fake),
        patch("summicron.depth.estimate.apply_tap_focus_shift", return_value=(depth_fake, 0.7)),
        patch("sys.argv", [
            "convolve",
            "--image", str(img_path),
            "--tensor", str(tensor_path),
            "--output", str(out_path),
        ]),
    ):
        from summicron.render.convolve import main

        main()
    assert out_path.exists()


def test_cli_hdr_source_equals_form(tmp_path: Path):
    """`--hdr-source=macos` equals-form must work identically to space form."""
    tensor_path = _mini_tensor_npz(tmp_path / "tensor.npz")
    out_path = tmp_path / "out.jpg"
    hdr_img = np.full((32, 32, 3), 2.0, dtype=np.float32)
    # P1.10.2 Cut 5: mock LiDAR extraction (default --depth-source is lidar).
    depth_fake = np.full((32, 32), 0.7, dtype=np.float32)

    with (
        patch("summicron.io.hdr_load_macos.load_hdr_heic", return_value=hdr_img),
        patch("summicron.depth.estimate.extract_lidar_depth", return_value=depth_fake),
        patch("summicron.depth.estimate.apply_tap_focus_shift", return_value=(depth_fake, 0.7)),
        patch("sys.argv", [
            "convolve",
            "--image", "/tmp/dummy.heic",
            "--tensor", str(tensor_path),
            "--hdr-source=macos",  # equals form
            "--output", str(out_path),
        ]),
    ):
        from summicron.render.convolve import main

        main()
    assert out_path.exists()


def test_cli_hdr_source_uppercase_rejected(tmp_path: Path, capsys):
    """argparse choices are case-sensitive — 'MACOS' must be rejected."""
    tensor_path = _mini_tensor_npz(tmp_path / "tensor.npz")
    out_path = tmp_path / "out.jpg"

    with patch("sys.argv", [
        "convolve",
        "--image", "/tmp/dummy",
        "--tensor", str(tensor_path),
        "--hdr-source", "MACOS",
        "--output", str(out_path),
    ]):
        from summicron.render.convolve import main

        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 2


def test_cli_hdr_source_linux_rejected(tmp_path: Path):
    """'--hdr-source linux' is not an offered choice."""
    tensor_path = _mini_tensor_npz(tmp_path / "tensor.npz")

    with patch("sys.argv", [
        "convolve",
        "--image", "/tmp/dummy",
        "--tensor", str(tensor_path),
        "--hdr-source", "linux",
    ]):
        from summicron.render.convolve import main

        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 2


# ---------------------------------------------------------------------------
# load_image regression
# ---------------------------------------------------------------------------


def test_load_image_ldr_path_is_bit_identical(tmp_path: Path):
    """`load_image` on a JPEG must return the exact same bytes as before
    P1.8 — no tone-curve shift, no pre-multiplication. Pillow-roundtrip is
    the oracle."""
    from PIL import Image

    from summicron.render.convolve import load_image

    arr = np.random.default_rng(42).integers(0, 255, (64, 48, 3), dtype=np.uint8)
    jpg_path = tmp_path / "probe.jpg"
    Image.fromarray(arr, mode="RGB").save(str(jpg_path), "JPEG", quality=95)

    loaded = load_image(jpg_path)
    # JPEG is lossy, so compare via round-trip of the SAME Pillow path.
    with Image.open(jpg_path) as im:
        oracle = np.asarray(im.convert("RGB"), dtype=np.uint8).astype(np.float32) / 255.0
    np.testing.assert_array_equal(loaded, oracle)
    assert loaded.dtype == np.float32
    assert loaded.min() >= 0.0
    assert loaded.max() <= 1.0
