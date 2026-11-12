"""R0 P1.11 — Adversarial probes of `_canonicalize_orientation`.

Builds synthetic HEIC fixtures with EXIF Orientation values 1-8 and probes the
HDR-vs-LDR pixel-position parity contract: the spec G1 says "HDR loader shape
must match `load_image` shape" but the user-facing requirement is stronger —
the renderer's per-pixel kernel grid must align with image content. Shape
parity alone is insufficient when stored shape happens to equal display shape
(EXIF=2/3/4 all preserve H,W; square images preserve H,W under any rotation).

Findings encoded as `pytest.mark.xfail(strict=True)` so they fail loud the
moment a future fix lands.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

MACOS_ONLY = pytest.mark.skipif(
    sys.platform != "darwin", reason="Core Image is macOS-only"
)


def _build_heic(path: Path, arr: np.ndarray, exif_orientation: int | None) -> None:
    """Save `arr` (H, W, 3) uint8 as HEIC with optional EXIF Orientation."""
    import pillow_heif
    from PIL import Image

    pillow_heif.register_heif_opener()
    if exif_orientation is None:
        Image.fromarray(arr, mode="RGB").save(str(path), "HEIF", quality=100)
    else:
        exif = Image.Exif()
        exif[0x0112] = exif_orientation
        Image.fromarray(arr, mode="RGB").save(
            str(path), "HEIF", exif=exif.tobytes(), quality=100
        )


def _quad_palette_array(h: int, w: int) -> np.ndarray:
    """Build (H, W, 3) uint8 with 4 colored quadrants (R/G/B/Y) for parity probes."""
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[: h // 2, : w // 2] = [255, 0, 0]
    arr[: h // 2, w // 2 :] = [0, 255, 0]
    arr[h // 2 :, : w // 2] = [0, 0, 255]
    arr[h // 2 :, w // 2 :] = [255, 255, 0]
    return arr


def _classify(px) -> str:
    r, g, b = px[0], px[1], px[2]
    if r > 0.5 and g < 0.3 and b < 0.3:
        return "R"
    if r < 0.3 and g > 0.5 and b < 0.3:
        return "G"
    if r < 0.3 and g < 0.3 and b > 0.5:
        return "B"
    if r > 0.5 and g > 0.5 and b < 0.3:
        return "Y"
    return "?"


def _quads(im: np.ndarray) -> tuple[str, str, str, str]:
    """Sample center of each canonical-display quadrant. HDR is gamma'd for cmp."""
    if im.dtype == np.float32 and im.max() > 0.1:
        im_v = np.clip(im, 0, 1) ** (1.0 / 2.4)
    else:
        im_v = im.astype(np.float32) / 255.0 if im.dtype == np.uint8 else im
    h, w = im_v.shape[:2]
    cy, cx = h // 4, w // 4
    return (
        _classify(im_v[cy, cx]),
        _classify(im_v[cy, 3 * cx]),
        _classify(im_v[3 * cy, cx]),
        _classify(im_v[3 * cy, 3 * cx]),
    )


# ---------------------------------------------------------------------------
# Axis 1+9: EXIF 2/4/5/7 fallback — pixel-layout parity (real bugs)
# ---------------------------------------------------------------------------


@MACOS_ONLY
def test_exif5_pixel_layout_matches_load_image(tmp_path: Path):
    from summicron.io.hdr_load_macos import load_hdr_heic
    from summicron.render.convolve import load_image

    arr = _quad_palette_array(96, 64)
    src = tmp_path / "exif5.heic"
    _build_heic(src, arr, 5)
    ldr = load_image(src)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        hdr = load_hdr_heic(src)
    assert _quads(hdr) == _quads(ldr), (
        f"EXIF=5 layout mismatch: HDR {_quads(hdr)} vs LDR {_quads(ldr)}"
    )


@MACOS_ONLY
def test_exif7_pixel_layout_matches_load_image(tmp_path: Path):
    from summicron.io.hdr_load_macos import load_hdr_heic
    from summicron.render.convolve import load_image

    arr = _quad_palette_array(96, 64)
    src = tmp_path / "exif7.heic"
    _build_heic(src, arr, 7)
    ldr = load_image(src)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        hdr = load_hdr_heic(src)
    assert _quads(hdr) == _quads(ldr)


# ---------------------------------------------------------------------------
# Axis 1+9: EXIF 2/4 (mirrors only, H==W not preserved oddly) - shape-match
# shortcut hides the mirror entirely. Real bug for non-iPhone HEIC.
# ---------------------------------------------------------------------------


@MACOS_ONLY
def test_exif2_mirror_pixel_layout_matches_load_image(tmp_path: Path):
    from summicron.io.hdr_load_macos import load_hdr_heic
    from summicron.render.convolve import load_image

    arr = _quad_palette_array(96, 64)
    src = tmp_path / "exif2.heic"
    _build_heic(src, arr, 2)
    ldr = load_image(src)
    hdr = load_hdr_heic(src)
    assert _quads(hdr) == _quads(ldr)


@MACOS_ONLY
def test_exif4_mirror_pixel_layout_matches_load_image(tmp_path: Path):
    from summicron.io.hdr_load_macos import load_hdr_heic
    from summicron.render.convolve import load_image

    arr = _quad_palette_array(96, 64)
    src = tmp_path / "exif4.heic"
    _build_heic(src, arr, 4)
    ldr = load_image(src)
    hdr = load_hdr_heic(src)
    assert _quads(hdr) == _quads(ldr)


# ---------------------------------------------------------------------------
# Axis 1+9: EXIF=3 — IS in the iPhone emit set. Same shape-match bug.
# ---------------------------------------------------------------------------


@MACOS_ONLY
def test_exif3_180_rotation_pixel_layout_matches_load_image(tmp_path: Path):
    from summicron.io.hdr_load_macos import load_hdr_heic
    from summicron.render.convolve import load_image

    arr = _quad_palette_array(96, 64)
    src = tmp_path / "exif3.heic"
    _build_heic(src, arr, 3)
    ldr = load_image(src)
    hdr = load_hdr_heic(src)
    assert _quads(hdr) == _quads(ldr), (
        f"EXIF=3 (iPhone emits this!): HDR {_quads(hdr)} vs LDR {_quads(ldr)}"
    )


# ---------------------------------------------------------------------------
# Axis 5: square HEIC + EXIF=6 — shape-match always wins, rotation skipped.
# ---------------------------------------------------------------------------


@MACOS_ONLY
def test_square_heic_exif6_pixel_layout_matches_load_image(tmp_path: Path):
    from summicron.io.hdr_load_macos import load_hdr_heic
    from summicron.render.convolve import load_image

    arr = _quad_palette_array(64, 64)
    src = tmp_path / "sq_exif6.heic"
    _build_heic(src, arr, 6)
    ldr = load_image(src)
    hdr = load_hdr_heic(src)
    assert _quads(hdr) == _quads(ldr)


# ---------------------------------------------------------------------------
# Axis 4: positive verification — EXIF=6 (non-square) k=3 is correct
# (asymmetric L-pattern catches off-by-2 = 180-deg confusion).
# ---------------------------------------------------------------------------


@MACOS_ONLY
def test_exif6_k_value_correct_via_asymmetric_L(tmp_path: Path):
    """An L-shape rotated wrong by 180 deg lands in the opposite corner.
    Validates the empirical `k=3` for EXIF=6 isn't off-by-2."""
    from summicron.io.hdr_load_macos import load_hdr_heic
    from summicron.render.convolve import load_image

    arr = np.full((96, 64, 3), 30, np.uint8)
    # L-shape in storage TL corner (asymmetric — sensitive to 180 flip).
    arr[0:20, 0:8] = [255, 0, 0]
    arr[12:20, 0:30] = [255, 0, 0]
    src = tmp_path / "L_exif6.heic"
    _build_heic(src, arr, 6)

    ldr = load_image(src)
    hdr = load_hdr_heic(src)
    assert ldr.shape == hdr.shape

    ldr_red = (ldr[..., 0] > 0.5) & (ldr[..., 1] < 0.3) & (ldr[..., 2] < 0.3)
    hdr_g = np.clip(hdr, 0, 1) ** (1.0 / 2.4)
    hdr_red = (hdr_g[..., 0] > 0.5) & (hdr_g[..., 1] < 0.3) & (hdr_g[..., 2] < 0.3)
    inter = (ldr_red & hdr_red).sum()
    union = (ldr_red | hdr_red).sum()
    iou = inter / max(int(union), 1)
    assert iou > 0.95, (
        f"EXIF=6 L-shape IoU {iou:.3f} < 0.95 — k value off (off-by-2 = 180 flip)"
    )


# ---------------------------------------------------------------------------
# Axis 2: missing-EXIF-tag path
# ---------------------------------------------------------------------------


@MACOS_ONLY
def test_no_exif_tag_defaults_to_identity(tmp_path: Path):
    """No EXIF metadata at all -> `getexif().get(0x0112, 1)` returns 1 ->
    table k=0 -> identity. Shape-match guard keeps it a no-op anyway."""
    from summicron.io.hdr_load_macos import load_hdr_heic
    from summicron.render.convolve import load_image

    arr = _quad_palette_array(96, 64)
    src = tmp_path / "no_exif.heic"
    _build_heic(src, arr, None)
    ldr = load_image(src)
    hdr = load_hdr_heic(src)
    assert hdr.shape == ldr.shape
    assert _quads(hdr) == _quads(ldr)


# ---------------------------------------------------------------------------
# Axis 6: JPEG / PNG inputs — load_hdr_heic accepts via Core Image. Verify
# the canonicalizer doesn't blow up on non-HEIC inputs (Core Image happily
# decodes JPEG/PNG via ImageIO).
# ---------------------------------------------------------------------------


@MACOS_ONLY
def test_load_hdr_heic_accepts_jpeg(tmp_path: Path):
    from PIL import Image
    from summicron.io.hdr_load_macos import load_hdr_heic

    arr = (np.ones((16, 24, 3), dtype=np.uint8) * 100)
    src = tmp_path / "test.jpg"
    Image.fromarray(arr, mode="RGB").save(str(src), "JPEG", quality=95)
    out = load_hdr_heic(src)
    assert out.dtype == np.float32
    assert out.shape == (16, 24, 3)
    assert out.flags.c_contiguous


@MACOS_ONLY
def test_load_hdr_heic_accepts_png(tmp_path: Path):
    from PIL import Image
    from summicron.io.hdr_load_macos import load_hdr_heic

    arr = (np.ones((16, 24, 3), dtype=np.uint8) * 100)
    src = tmp_path / "test.png"
    Image.fromarray(arr, mode="RGB").save(str(src), "PNG")
    out = load_hdr_heic(src)
    assert out.dtype == np.float32
    assert out.shape == (16, 24, 3)


# ---------------------------------------------------------------------------
# Axis 7: memory layout — explicit C-contiguity post-rotation
# ---------------------------------------------------------------------------


@MACOS_ONLY
@pytest.mark.parametrize("orientation", [1, 3, 6, 8])
def test_canonicalized_array_is_c_contiguous(tmp_path: Path, orientation: int):
    """np.rot90 returns a non-contig view; `np.ascontiguousarray` must fix it."""
    from summicron.io.hdr_load_macos import load_hdr_heic

    arr = _quad_palette_array(96, 64)
    src = tmp_path / f"contig_exif{orientation}.heic"
    _build_heic(src, arr, orientation)
    out = load_hdr_heic(src)
    assert out.flags["C_CONTIGUOUS"], (
        f"EXIF={orientation}: post-canonicalize array not C-contiguous"
    )


# ---------------------------------------------------------------------------
# Axis 8: determinism — same path -> bit-identical buffer across calls
# ---------------------------------------------------------------------------


@MACOS_ONLY
def test_load_hdr_heic_deterministic(tmp_path: Path):
    from summicron.io.hdr_load_macos import load_hdr_heic

    arr = _quad_palette_array(96, 64)
    src = tmp_path / "det_exif6.heic"
    _build_heic(src, arr, 6)
    a = load_hdr_heic(src)
    b = load_hdr_heic(src)
    assert np.array_equal(a, b), "two HDR loads of same file differ"


# ---------------------------------------------------------------------------
# Axis 10: LDR-path side effect — calling HDR loader (which calls
# `pillow_heif.register_heif_opener()` inside `_canonicalize_orientation`)
# must not perturb subsequent `load_image()` results.
# ---------------------------------------------------------------------------


@MACOS_ONLY
def test_hdr_call_does_not_perturb_load_image(tmp_path: Path):
    from summicron.io.hdr_load_macos import load_hdr_heic
    from summicron.render.convolve import load_image

    arr = _quad_palette_array(96, 64)
    src = tmp_path / "side_effect.heic"
    _build_heic(src, arr, 6)
    ldr_before = load_image(src)
    _ = load_hdr_heic(src)
    ldr_after = load_image(src)
    assert np.array_equal(ldr_before, ldr_after), (
        "HDR loader perturbed LDR load (likely register_heif_opener side effect)"
    )
    assert ldr_before.shape == ldr_after.shape


REPO_ROOT = Path(__file__).resolve().parents[2]
HDR_FIXTURE = REPO_ROOT / "iphoneImage" / "IMG_6651.HEIC"
HDR_FIXTURE_6915 = REPO_ROOT / "iphoneImage" / "IMG_6915.HEIC"


@MACOS_ONLY
@pytest.mark.skipif(not HDR_FIXTURE.exists(), reason=f"missing {HDR_FIXTURE}")
def test_real_data_determinism_IMG_6651():
    from summicron.io.hdr_load_macos import load_hdr_heic

    a = load_hdr_heic(HDR_FIXTURE)
    b = load_hdr_heic(HDR_FIXTURE)
    assert np.array_equal(a, b)
    assert a.flags["C_CONTIGUOUS"]


@MACOS_ONLY
@pytest.mark.skipif(
    not HDR_FIXTURE_6915.exists(), reason=f"missing {HDR_FIXTURE_6915}"
)
def test_real_data_load_image_unchanged_after_hdr_call():
    """Canonical contract: IMG_6915 LDR shape (5712, 4284, 3) regardless of
    whether the HDR loader has been called previously."""
    from summicron.io.hdr_load_macos import load_hdr_heic
    from summicron.render.convolve import load_image

    ldr_before = load_image(HDR_FIXTURE_6915)
    _ = load_hdr_heic(HDR_FIXTURE_6915)
    ldr_after = load_image(HDR_FIXTURE_6915)
    assert ldr_before.shape == (5712, 4284, 3)
    assert ldr_after.shape == (5712, 4284, 3)
    assert np.array_equal(ldr_before, ldr_after)
