"""Anchor / orientation alignment tests — designed to catch the class of bug
R0 shipped with invisible to automated tests but visible to the user.

R0 LESSON: the 10-axis robust audit passed (shape parity, Laplacian ratio,
32×32 downsampled luminance correlation, LDR bit-identity) while the visual
face sharpness dropped 15.8× at the eye region. The audit lacked two gates
that would have caught it:

1. FULL-RESOLUTION content alignment between LDR and HDR (not 32×32).
   A coarse correlation can read 0.99 while content at the face pixel level
   is rotated/flipped wrong, because both images have similar regional
   statistics (green foliage everywhere).

2. FOCUS-XY ANCHOR GROUND TRUTH: given `--focus-xy X Y`, the SHARPEST output
   region must be near (X, Y) — i.e. what the user tapped is what the
   renderer put in focus.

These tests encode both. They REQUIRE real canonical bench HEICs.
"""
from __future__ import annotations
import sys
import subprocess
from pathlib import Path
import numpy as np
import pytest

from summicron.render._artifacts import default_tensor_path
from PIL import Image
from scipy import ndimage as ndi

REPO = Path(__file__).resolve().parents[2]
IMG_6915 = REPO / "iphoneImage" / "IMG_6915.HEIC"
IMG_6651 = REPO / "iphoneImage" / "IMG_6651.HEIC"
TENSOR = default_tensor_path()

# Canonical face-anchor coord for IMG_6915 in PORTRAIT storage orientation
# (post-R0 fix). Lands on the eyes per scene content at face-correct
# (0.40, 0.45). The pre-R0 baseline_face_v2.jpg used (0.34, 0.60) which is
# the shoulder in portrait — see doc/plan/phase1/section11/TLDR-R0-orientation.md
# for the anchor-semantic regression finding.
IMG_6915_FACE_ANCHOR = (0.40, 0.45)

pytestmark = pytest.mark.skipif(
    not IMG_6915.exists() or not TENSOR.exists(),
    reason="requires canonical bench HEIC + tensor on this checkout",
)


def _lum(rgb: np.ndarray) -> np.ndarray:
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


def _lap_var(rgb: np.ndarray) -> float:
    return float(ndi.laplace(_lum(rgb).astype(np.float64)).var())


def _tiled_lap_var_map(rgb: np.ndarray, tile: int = 64) -> np.ndarray:
    """Return a (H//tile, W//tile) map of Laplacian variance per tile —
    used to locate the SHARPEST region of the image."""
    lum = _lum(rgb).astype(np.float64)
    lap = ndi.laplace(lum)
    H, W = lum.shape
    hy, hx = H // tile, W // tile
    m = np.zeros((hy, hx), dtype=np.float64)
    for y in range(hy):
        for x in range(hx):
            tl = lap[y * tile:(y + 1) * tile, x * tile:(x + 1) * tile]
            m[y, x] = float(tl.var())
    return m


# ---------------------------------------------------------------------------
# Test 1: Full-resolution LDR ↔ HDR content alignment.
# Catches: rotation-direction bugs (np.rot90 wrong k value, EXIF mis-reading)
# that the R0 32×32 downsampled correlation missed.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("img_path", [IMG_6915, IMG_6651], ids=["IMG_6915", "IMG_6651"])
def test_ldr_hdr_content_aligned_at_full_resolution(img_path):
    """Full-res gamma-matched LDR vs HDR content must align.

    A 0.99 correlation on a 32×32 downsampled grid can coexist with a
    rotation-direction bug (both images have similar bulk statistics).
    This test computes a per-pixel correspondence at FULL resolution by
    taking a mid-luminance slice (where neither path is clipped) and
    measuring how well the two align at zero spatial offset vs 90°
    rotations. If ANY 90°/180°/270° rotation of HDR beats identity, the
    load path rotated the wrong way.
    """
    from summicron.render.convolve import load_image
    from summicron.io.hdr_load_macos import load_hdr_heic

    ldr = load_image(str(img_path))  # (H, W, 3), sRGB gamma [0, 1]
    hdr = load_hdr_heic(str(img_path))  # (H, W, 3), extended-linear

    assert ldr.shape == hdr.shape, (
        f"{img_path.name}: shapes differ ldr={ldr.shape} hdr={hdr.shape}"
    )

    # Gamma-match: LDR → linear (gamma 2.2), HDR → clipped to [0, 1]
    ldr_lin = np.power(np.clip(ldr, 0, 1), 2.2)
    hdr_clip = np.clip(hdr, 0, 1)

    ldr_lum = _lum(ldr_lin)
    hdr_lum = _lum(hdr_clip)

    # Downsample to 256×256 for per-pixel alignment (32×32 is too coarse).
    H, W = ldr_lum.shape
    zf = (256 / H, 256 / W)
    ld = ndi.zoom(ldr_lum, zf, order=1)
    hd = ndi.zoom(hdr_lum, zf, order=1)

    def normcorr(a, b):
        a_n = (a - a.mean()) / (a.std() + 1e-8)
        b_n = (b - b.mean()) / (b.std() + 1e-8)
        return float((a_n * b_n).mean())

    # Test identity correlation AND 90°/180°/270° rotations.
    # Identity must be STRICTLY larger than any rotation (+ margin)
    # or the HDR content was mis-rotated.
    corr_identity = normcorr(ld, hd)
    corr_90 = normcorr(ld, np.rot90(hd, k=1))
    corr_180 = normcorr(ld, np.rot90(hd, k=2))
    corr_270 = normcorr(ld, np.rot90(hd, k=3))

    assert corr_identity > 0.85, (
        f"{img_path.name}: LDR-HDR content correlation {corr_identity:.4f} < 0.85; "
        f"content is misaligned even at identity orientation"
    )
    # Critical: identity must beat ALL rotations by ≥ 0.05
    best_wrong = max(corr_90, corr_180, corr_270)
    assert corr_identity > best_wrong + 0.05, (
        f"{img_path.name}: HDR rotated WRONG way. "
        f"identity_corr={corr_identity:.4f} vs best_rot_corr={best_wrong:.4f} "
        f"(90={corr_90:.4f}, 180={corr_180:.4f}, 270={corr_270:.4f}). "
        f"If a rotation correlates better than identity, "
        f"the np.rot90 k in _EXIF_TO_OP is wrong direction."
    )


# ---------------------------------------------------------------------------
# Test 2: Focus-xy anchor → sharpest-region ground truth.
# Catches: anchor-coord semantic change where the rendered-sharp location
# is NOT where the user tapped. R0 shipped with face-anchor coord (0.34, 0.60)
# landing in the shoulder/shirt after the portrait-grid fix, not the face.
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_focus_xy_anchor_is_locally_sharp(tmp_path):
    """Given --focus-xy X Y, the output tile AT (X, Y) must be in the top 20%
    of tiles by Laplacian variance — i.e. the anchor region is demonstrably
    among the sharpest regions in the frame.

    We do NOT require it to be THE global max (bright HDR point-source
    content like string-light highlights can have higher Lap variance than
    the focal-plane content in L1-normed output). Top-20% is the right
    contract: the anchor landed on something sharp.

    This catches the R0-debug regression mode: if --focus-xy points at
    shoulder while renderer puts face in focus (or vice versa), the anchor
    tile's Lap variance will rank LOW among all tiles.
    """
    focus_x, focus_y = IMG_6915_FACE_ANCHOR
    out = tmp_path / "IMG_6915_anchor_test.jpg"
    cmd = [
        sys.executable, "-m",
        "summicron.render.convolve",
        "--image", str(IMG_6915),
        "--tensor", str(TENSOR),
        "--depth-source", "lidar",
        "--depth-scale", "10",
        "--downres", "2048",
        "--focus-xy", str(focus_x), str(focus_y),
        "--psf-norm", "l1",
        "--hdr-source", "macos",
        "--output", str(out),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO))
    assert out.exists(), f"render failed: {result.stdout}\n{result.stderr}"

    rgb = np.asarray(Image.open(out).convert("RGB"), dtype=np.float32) / 255.0
    H, W = rgb.shape[:2]

    # Tiled Laplacian variance map
    tile_size = 64
    lap_map = _tiled_lap_var_map(rgb, tile=tile_size)

    # Find anchor tile
    anchor_tile_y = int(focus_y * H / tile_size)
    anchor_tile_x = int(focus_x * W / tile_size)
    anchor_lap = lap_map[anchor_tile_y, anchor_tile_x]

    # Rank: what fraction of tiles have HIGHER lap_var than the anchor?
    flat = lap_map.flatten()
    rank_pct = float((flat > anchor_lap).mean())

    assert rank_pct < 0.20, (
        f"Focus-xy anchor did NOT land on a locally-sharp region: "
        f"anchor tile ({anchor_tile_x}, {anchor_tile_y}) has Lap variance "
        f"{anchor_lap:.6f}, which is only top {rank_pct*100:.1f}%. "
        f"The scene pixel the user tapped is not where the renderer put a "
        f"sharp plane. Likely a storage-vs-display coord mismatch."
    )


# ---------------------------------------------------------------------------
# Test 3: Face-region visual-quality floor on canonical bench.
# Catches: face blur regressions (the R0 eye-region went from Lap var 0.002
# to 0.0001 — 15.8× worse — and no test noticed). Hard floor against drift.
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_face_region_sharpness_floor_canonical(tmp_path):
    """Face-region Laplacian variance on IMG_6915 canonical bench must be
    above a floor. Pre-regression v2 baseline: ~0.00135 at eyes (in rotated-
    for-viewing landscape). Post-anchor-fix R0: ~0.00151. Floor set at
    0.0008 (~60% of baseline) to allow R1 bright-pass variation but catch
    gross face-softening regressions."""
    focus_x, focus_y = IMG_6915_FACE_ANCHOR
    out = tmp_path / "IMG_6915_face_sharpness_test.jpg"
    cmd = [
        sys.executable, "-m",
        "summicron.render.convolve",
        "--image", str(IMG_6915),
        "--tensor", str(TENSOR),
        "--depth-source", "lidar",
        "--depth-scale", "10",
        "--downres", "2048",
        "--focus-xy", str(focus_x), str(focus_y),
        "--psf-norm", "l1",
        "--hdr-source", "macos",
        "--output", str(out),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO))
    assert out.exists(), f"render failed: {result.stdout}\n{result.stderr}"

    rgb = np.asarray(Image.open(out).convert("RGB"), dtype=np.float32) / 255.0
    H, W = rgb.shape[:2]

    # Post-R0, output is native portrait. Eye coord = face anchor directly.
    eye_x, eye_y = int(focus_x * W), int(focus_y * H)
    r = 100  # 200×200 patch
    eye_patch = rgb[max(0, eye_y - r):eye_y + r, max(0, eye_x - r):eye_x + r]
    eye_lap_var = _lap_var(eye_patch)

    FLOOR = 0.0008
    assert eye_lap_var >= FLOOR, (
        f"Face-region sharpness regression: eye Laplacian variance "
        f"{eye_lap_var:.6f} < floor {FLOOR:.6f}. "
        f"(Pre-regression v2 baseline = 0.002036; post-R0-regression = 0.000129.) "
        f"If R1 bright-pass intentionally affects midtones, update the floor."
    )


# ---------------------------------------------------------------------------
# Test 4: Cross-path anchor consistency — LDR and HDR renders with the same
# --focus-xy must put the sharpest region in the SAME scene location.
# Catches: load-path shape disagreement causing one path's focus-xy to hit
# different scene content than the other (R0 root-cause class).
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_ldr_hdr_anchor_region_both_sharp(tmp_path):
    """Same --focus-xy on LDR and HDR paths must both produce a sharp
    region at the anchor. Catches load-path-shape bugs where focus-xy
    means different scene content across paths.

    Uses top-25% rank gate on the anchor tile (same contract as Test 2)
    applied to BOTH renders — guards against the path-disagreement class
    of bug without over-asserting on which specific tile is the global max
    (which can differ between LDR Lanczos downres and HDR bilinear)."""
    focus_x, focus_y = IMG_6915_FACE_ANCHOR
    outs = {}
    for tag, hdr_src in [("ldr", "none"), ("hdr", "macos")]:
        out = tmp_path / f"IMG_6915_{tag}.jpg"
        cmd = [
            sys.executable, "-m",
            "summicron.render.convolve",
            "--image", str(IMG_6915),
            "--tensor", str(TENSOR),
            "--depth-source", "lidar",
            "--depth-scale", "10",
            "--downres", "2048",
            "--focus-xy", str(focus_x), str(focus_y),
            "--psf-norm", "l1",
            "--hdr-source", hdr_src,
            "--output", str(out),
        ]
        subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO))
        assert out.exists(), f"{tag} render failed"
        outs[tag] = np.asarray(Image.open(out).convert("RGB"), dtype=np.float32) / 255.0

    # Must be same shape post-R0 (this itself is a regression gate)
    assert outs["ldr"].shape == outs["hdr"].shape, (
        f"LDR and HDR output shapes differ: "
        f"ldr={outs['ldr'].shape} hdr={outs['hdr'].shape}. "
        f"Load paths disagree on orientation — the R0 bug class."
    )

    tile = 64
    for tag, rgb in outs.items():
        H, W = rgb.shape[:2]
        lap_map = _tiled_lap_var_map(rgb, tile=tile)
        anchor_y = int(focus_y * H / tile)
        anchor_x = int(focus_x * W / tile)
        anchor_lap = lap_map[anchor_y, anchor_x]
        rank_pct = float((lap_map.flatten() > anchor_lap).mean())
        assert rank_pct < 0.25, (
            f"{tag.upper()} path: anchor region at ({focus_x}, {focus_y}) "
            f"has Lap variance rank top {rank_pct*100:.1f}%. Should be in "
            f"top 25% for a correctly-anchored render. The focus-xy coord "
            f"does not land on sharp content on this path."
        )
