"""Adversarial tests for `summicron.io.hdr_tonemap_macos.tonemap_core_image`.

Covers buffer lifetime, negative input clamping, extreme-HDR headroom clamp,
single-pixel edge case, non-contiguous input, and a wrong-polarity fixture
(LDR input must not be darkened artificially by the tonemap).
"""
from __future__ import annotations

import gc
import sys

import numpy as np
import pytest

from summicron.io.hdr_tonemap_macos import tonemap_core_image

MACOS_ONLY = pytest.mark.skipif(
    sys.platform != "darwin", reason="Core Image is macOS-only"
)


@MACOS_ONLY
def test_buffer_lifetime():
    """Returned uint8 array must own its memory and survive GC of any
    underlying CIImage/CIContext (enforced by `.copy()` in the module)."""
    rgb = np.full((8, 8, 3), 0.3, dtype=np.float32)
    out = tonemap_core_image(rgb)
    assert out.flags.owndata, "returned array does not own its buffer"
    # Force GC of any transient ObjC refs, then mutate and confirm ownership.
    gc.collect()
    out[0, 0, 0] = 42
    assert out[0, 0, 0] == 42


@MACOS_ONLY
def test_negative_input_clamped():
    """Input with min=-0.01 must pass without crash; output is valid uint8."""
    rng = np.random.default_rng(2)
    rgb = rng.uniform(0.0, 1.5, size=(8, 8, 3)).astype(np.float32)
    rgb[0, 0, 0] = -0.01
    out = tonemap_core_image(rgb)
    assert out.dtype == np.uint8
    assert out.shape == (8, 8, 3)
    assert out.min() >= 0
    assert out.max() <= 255


@MACOS_ONLY
def test_extreme_hdr_clamp():
    """Input with max=100.0 (>> filter's 32.0 ceiling) must clamp headroom
    to 32.0 without raising NSInvalidArgumentException."""
    rgb = np.full((8, 8, 3), 0.5, dtype=np.float32)
    rgb[0, 0, 0] = 100.0
    out = tonemap_core_image(rgb)
    assert out.dtype == np.uint8
    assert out.shape == (8, 8, 3)
    assert not np.isnan(out).any()


@MACOS_ONLY
def test_single_pixel():
    """(1, 1, 3) input → (1, 1, 3) uint8."""
    rgb = np.array([[[0.5, 0.5, 0.5]]], dtype=np.float32)
    out = tonemap_core_image(rgb)
    assert out.shape == (1, 1, 3)
    assert out.dtype == np.uint8


@MACOS_ONLY
def test_non_contiguous_input():
    """Reversed-stride input (not C-contiguous) must be handled via
    `np.ascontiguousarray` inside the function."""
    rng = np.random.default_rng(3)
    base = rng.uniform(0.0, 1.5, size=(8, 8, 3)).astype(np.float32)
    view = base[:, :, ::-1]
    assert not view.flags.c_contiguous
    out = tonemap_core_image(view)
    assert out.shape == (8, 8, 3)
    assert out.dtype == np.uint8


@MACOS_ONLY
def test_ldr_input_not_darkened_artificially():
    """POST-MORTEM-1 WRONG-POLARITY FIXTURE: a uniform LDR array (max=0.5)
    passed through CI tonemap must NOT come out visibly darker than
    `linear_to_srgb(0.5)*255 ≈ 188`. A too-eager tonemap that compresses
    below the headroom threshold would fail this."""
    rgb = np.full((32, 32, 3), 0.5, dtype=np.float32)
    out = tonemap_core_image(rgb)
    mean = float(out.mean())
    assert 185 <= mean <= 192, (
        f"LDR 0.5 uniform tonemapped to mean {mean}; "
        "tonemap is over-eager below headroom threshold"
    )


# ---------------------------------------------------------------------------
# P1.8.1 Round-2 adversarial additions — probes for edge cases the Coder's
# synthetic fixtures did NOT cover.
# ---------------------------------------------------------------------------


@MACOS_ONLY
@pytest.mark.xfail(
    strict=True,
    reason=(
        "BUG: NaN in a single pixel propagates into rgb_hdr.max() → "
        "src_headroom becomes NaN (since max(1.0, nan)=nan in Python), "
        "which is then passed to CIToneMapHeadroom without validation. "
        "There is no explicit guard; the filter either silently produces "
        "garbage or the NaN is memcpy'd into the CIImage buffer and blooms "
        "across the render. Production code should either raise ValueError "
        "on non-finite input or sanitize via np.nan_to_num before .max()."
    ),
)
def test_nan_pixel_rejected_or_sanitized():
    """A single NaN pixel must not silently corrupt the whole render.

    Real-world path: FFT convolution in render() can, under very specific
    bug conditions (divide-by-zero in a future weight-sum renorm, corrupt
    tensor kernels, etc.) produce a NaN that reaches tonemap. The tonemap
    should either raise or sanitize — NOT propagate silently.
    """
    rgb = np.full((16, 16, 3), 0.3, dtype=np.float32)
    rgb[0, 0, 0] = np.float32("nan")
    out = tonemap_core_image(rgb)
    # Strict contract: no NaN, and non-NaN bulk pixels not blown to black.
    assert not np.isnan(out).any(), "NaN input propagated to uint8 output"
    # The non-NaN bulk should render as roughly linear_to_srgb(0.3) ≈ 149.
    bulk_mean = float(out[1:, 1:].mean())
    assert 120 <= bulk_mean <= 170, (
        f"NaN pixel poisoned bulk: bulk_mean={bulk_mean}"
    )


@MACOS_ONLY
@pytest.mark.xfail(
    strict=True,
    reason=(
        "BUG: +Inf in a single pixel makes rgb_hdr.max() = +inf. "
        "min(32.0, inf) = 32.0 so the headroom clamp works, BUT the inf "
        "value itself is still written into the CIImage float32 buffer. "
        "The filter's behaviour on +inf input is undefined by the spec "
        "and observed to produce white-saturation bleed across nearby "
        "pixels in the Core Image output. Production code should "
        "np.clip / np.nan_to_num before the buffer copy."
    ),
)
def test_posinf_pixel_does_not_bleed():
    """+Inf must not bleed beyond its own pixel."""
    rgb = np.full((16, 16, 3), 0.3, dtype=np.float32)
    rgb[0, 0, 0] = np.float32("inf")
    out = tonemap_core_image(rgb)
    # Non-inf bulk should still render near linear_to_srgb(0.3).
    bulk_mean = float(out[2:, 2:].mean())
    assert 120 <= bulk_mean <= 170, (
        f"+Inf pixel bled into bulk: bulk_mean={bulk_mean}"
    )


@MACOS_ONLY
def test_empty_zero_size_raises():
    """A zero-size (0, 0, 3) array should raise a clear error rather than
    segfault, hang, or return garbage. Any of ValueError / RuntimeError is
    acceptable — silent success or native crash is not."""
    rgb = np.zeros((0, 0, 3), dtype=np.float32)
    with pytest.raises((ValueError, RuntimeError)):
        tonemap_core_image(rgb)


@MACOS_ONLY
def test_all_zero_input_headroom_floor():
    """All-zero input → rgb_hdr.max() = 0 → headroom clamped to 1.0.
    Filter must accept inputSourceHeadroom=1.0 without raising, and output
    must be all-zero (no spurious offset)."""
    rgb = np.zeros((8, 8, 3), dtype=np.float32)
    out = tonemap_core_image(rgb)
    assert out.shape == (8, 8, 3)
    assert out.dtype == np.uint8
    # sRGB(0) = 0; a single-bit-off offset would signal bogus gamma.
    assert int(out.max()) == 0, f"all-zero input produced nonzero output max={out.max()}"


@MACOS_ONLY
def test_grayscale_input_raises_but_message_is_cryptic():
    """Grayscale (H, W) input raises ValueError via a numpy-broadcast
    error in rgba_in[..., :3] = rgb_hdr. It IS rejected (good), but the
    message is a cryptic numpy broadcast error, not a helpful
    'expected (H, W, 3)' message. We pin the current behaviour; if the
    module ever adds a nicer guard this test will need updating.
    SMELL (not a bug): message quality, reported separately."""
    gray = np.full((8, 8), 0.5, dtype=np.float32)
    with pytest.raises(ValueError, match="could not broadcast"):
        tonemap_core_image(gray)


@MACOS_ONLY
def test_rgba_input_raises_but_message_is_cryptic():
    """RGBA (H, W, 4) input raises ValueError via numpy broadcast. Same
    pinning as grayscale — rejection works, message is cryptic."""
    rgba = np.full((8, 8, 4), 0.5, dtype=np.float32)
    with pytest.raises(ValueError, match="could not broadcast"):
        tonemap_core_image(rgba)


@MACOS_ONLY
def test_uint8_input_behaviour():
    """uint8 input (already-LDR) — not in the signature's contract.

    We test what CURRENTLY happens. The Coder's implementation forces
    ``np.ascontiguousarray(rgb_hdr, dtype=np.float32)`` which will coerce
    uint8 [0..255] to float32 [0..255] (NOT [0, 1]). That means a uint8
    input of 128 becomes 128.0 linear, max=255 → headroom=32.0, and the
    tonemap squashes the whole image toward black. This is silent wrong
    output. We document the failure mode: output.max() must not be <20 for
    a mid-gray uint8 input."""
    rgb_u8 = np.full((8, 8, 3), 128, dtype=np.uint8)
    # Should either raise TypeError (explicit reject) OR coerce to [0, 1].
    # Current behaviour is neither → silent dark output.
    try:
        out = tonemap_core_image(rgb_u8)
    except (TypeError, ValueError):
        # Acceptable: module rejected the unsupported dtype.
        return
    # If it accepted the uint8, the output must at least be reasonable
    # (mid-gray uint8 input should not render as black).
    assert out.max() > 20, (
        f"uint8 mid-gray input silently rendered as near-black "
        f"(max={out.max()}); module accepted unsupported dtype without "
        "coercing [0..255] → [0, 1]."
    )


@MACOS_ONLY
def test_realistic_ldr_scene_wrong_polarity():
    """POST-MORTEM-1 WRONG-POLARITY (second fixture): a realistic LDR scene
    with NON-uniform content (dark background + bright-highlight region,
    max=0.95, mean ~0.3). The render pipeline's assumption is "HDR input
    with possibly > 1.0"; the opposite polarity is "non-HDR HEIC passed
    through expandToHDR" which yields strictly-[0, 1] float32. Failure
    modes being probed:
      (a) hard clip at 1.0 — would crush the bright-highlight region,
      (b) midtone crush — would drop global mean far below expected,
      (c) spurious raise — must never happen.
    """
    H, W = 64, 64
    rgb = np.full((H, W, 3), 0.05, dtype=np.float32)  # dark background
    # Bright-highlight patch in upper-left quadrant (~1/16 of area, ~0.9 val).
    rgb[0:16, 0:16, :] = 0.9
    rgb[5:8, 5:8, :] = 0.95  # sub-patch peak
    assert rgb.min() >= 0.0 and rgb.max() <= 1.0
    # For reference: uniform-0.3 input → ~148 uint8 via linear_to_srgb;
    # this scene is much darker (mean 0.05 + small bright patch) →
    # expected mean band is [60, 130] as described in the brief.
    out = tonemap_core_image(rgb)
    assert out.shape == (H, W, 3)
    assert out.dtype == np.uint8
    mean = float(out.mean())
    assert 60 <= mean <= 130, (
        f"realistic LDR scene tonemapped to mean {mean}; "
        "expected [60, 130] — failure suggests midtone crush or hard clip"
    )
    # Highlight region must not be crushed flat: brightest pixel in the
    # 0.95 sub-patch should still be distinctly brighter than the dark bg.
    highlight_mean = float(out[5:8, 5:8].mean())
    bg_mean = float(out[32:, 32:].mean())
    assert highlight_mean > bg_mean + 30, (
        f"highlight ({highlight_mean}) not meaningfully brighter than "
        f"background ({bg_mean}) — midtone/highlight contrast crushed"
    )


@MACOS_ONLY
def test_single_bright_spike_does_not_kill_bulk():
    """One 1e6 pixel in an otherwise-normal scene. Headroom clamp engages
    at 32.0 (good); the spike should go to ~255. The REAL risk: does the
    global headroom choice make the bulk pixels unusably dark?"""
    rgb = np.full((16, 16, 3), 0.3, dtype=np.float32)
    rgb[0, 0, :] = 1e6
    out = tonemap_core_image(rgb)
    spike = out[0, 0]
    # The spike should saturate (or near-saturate) to white.
    assert int(spike.min()) >= 240, (
        f"1e6 spike did not saturate: pixel={spike}"
    )
    # Bulk must still be visible. uniform 0.3 linear → sRGB ≈ 149 without
    # tonemap. With src_headroom=32 the filter will darken bulk some, but
    # a well-behaved tonemap keeps the mid-gray visible (> 40).
    bulk = out[1:, 1:]
    bulk_mean = float(bulk.mean())
    assert bulk_mean > 40, (
        f"1e6 spike crushed bulk to bulk_mean={bulk_mean}; "
        "headroom choice is too aggressive for extreme spikes"
    )


@MACOS_ONLY
def test_repeated_call_bit_identical():
    """Back-to-back calls on the same input must produce bit-identical
    output. This catches context-invalidation, shared-buffer, and caching
    bugs that would only show up under repeated use."""
    rng = np.random.default_rng(42)
    rgb = rng.uniform(0.0, 3.0, size=(32, 32, 3)).astype(np.float32)
    out1 = tonemap_core_image(rgb)
    out2 = tonemap_core_image(rgb)
    out3 = tonemap_core_image(rgb)
    assert np.array_equal(out1, out2), "call#1 != call#2 (non-deterministic)"
    assert np.array_equal(out2, out3), "call#2 != call#3 (non-deterministic)"


def test_lazy_import_not_leaked_on_none_path(tmp_path, monkeypatch):
    """The hdr_tonemap_macos module imports Quartz lazily inside
    tonemap_core_image(); main() with --hdr-source=none MUST NOT import
    the tonemap module AT ALL. We verify by stubbing
    ``hdr_tonemap_macos`` in sys.modules with a sentinel that raises if
    its tonemap_core_image is accessed, then running the LDR path.

    Runnable on any platform (the none path must not touch macOS APIs)."""
    import sys as _sys
    from pathlib import Path as _Path
    from unittest.mock import patch

    repo_root = _Path(__file__).resolve().parents[2]
    small_img = repo_root / "iphoneImage" / "IMG_6651.HEIC"
    if not small_img.exists():
        pytest.skip("no small fixture image available")

    # Create a poisoned module that would fail if imported for its
    # tonemap_core_image symbol.
    class _Poisoned:
        def __getattr__(self, name):
            if name == "tonemap_core_image":
                raise ImportError(
                    f"none-path leaked tonemap import: looked up {name}"
                )
            raise AttributeError(name)

    # Pop any cached real module, install the poison.
    _sys.modules.pop("summicron.io.hdr_tonemap_macos", None)
    monkeypatch.setitem(
        _sys.modules, "summicron.io.hdr_tonemap_macos", _Poisoned()
    )

    # Build a minimal tensor file.
    # P1.10.2 Cut 5: include object_distances_m + focus_distance_m so the
    # new default --depth-source=lidar pipeline can interp depth_m → defocus.
    tensor_path = tmp_path / "tensor.npz"
    R, D, K = 3, 4, 11
    kernels = np.zeros((R, D, K, K), dtype=np.float32)
    kernels[:, :, K // 2, K // 2] = 1.0
    np.savez(
        str(tensor_path),
        kernels=kernels,
        radial_edges=np.linspace(0.0, 1.0, R + 1).astype(np.float64),
        defocus_ratios=np.linspace(0.0, 1.0, D).astype(np.float64),
        corpus_ref_long_edge=8000,
        otf_norm_applied=True,
        object_distances_m=np.linspace(0.7, 5.0, D).astype(np.float64),
        focus_distance_m=np.float64(0.7),
    )

    out_path = tmp_path / "out.jpg"

    # P1.10.2 Cut 5: mock LiDAR extraction. Image is real HEIC downresed
    # to 64x48 (long-edge 64). Match the post-downres shape.
    depth_fake = np.full((64, 48), 0.7, dtype=np.float32)

    with (
        patch("summicron.depth.estimate.extract_lidar_depth", return_value=depth_fake),
        patch("summicron.depth.estimate.apply_tap_focus_shift", return_value=(depth_fake, 0.7)),
        patch("sys.argv", [
            "convolve",
            "--image", str(small_img),
            "--tensor", str(tensor_path),
            "--hdr-source", "none",
            "--downres", "64",
            "--output", str(out_path),
        ]),
    ):
        # Force re-import of convolve in case tonemap was already cached
        _sys.modules.pop("summicron.render.convolve", None)
        from summicron.render.convolve import main
        main()

    assert out_path.exists(), "LDR main() produced no output"


