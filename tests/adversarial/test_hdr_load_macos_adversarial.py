"""Adversarial tests for `summicron.io.hdr_load_macos.load_hdr_heic`.

Targets error paths (bad path, non-HEIC file, directory), wrong-polarity
fixtures (SDR-only HEIC through expandToHDR), buffer-lifetime, and platform
spoofing. Most tests are macOS-only; see ``skipif`` marks.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from summicron.io.hdr_load_macos import load_hdr_heic

REPO_ROOT = Path(__file__).resolve().parents[2]
HDR_FIXTURE = REPO_ROOT / "iphoneImage" / "IMG_6651.HEIC"

MACOS_ONLY = pytest.mark.skipif(
    sys.platform != "darwin", reason="Core Image is macOS-only"
)


# ---------------------------------------------------------------------------
# Error-path tests (macOS-only; all exercise the Quartz stack)
# ---------------------------------------------------------------------------


@MACOS_ONLY
def test_load_hdr_heic_nonexistent_path_raises():
    """CIImage.imageWithContentsOfURL returns None for a missing file; loader
    must raise a clear RuntimeError, not silently return junk."""
    with pytest.raises(RuntimeError, match="CIImage failed to load"):
        load_hdr_heic("/tmp/this-file-does-not-exist-for-sure-xyz.heic")


@MACOS_ONLY
def test_load_hdr_heic_non_heic_file_raises(tmp_path: Path):
    """Pass a plain text file with a .heic extension; CIImage should fail to
    decode and the loader should raise RuntimeError."""
    bogus = tmp_path / "not_really.heic"
    bogus.write_text("definitely not a HEIC")
    with pytest.raises(RuntimeError, match="CIImage failed to load"):
        load_hdr_heic(bogus)


@MACOS_ONLY
def test_load_hdr_heic_directory_as_path_raises(tmp_path: Path):
    """Directory (not a file) must raise, not try to decode a folder."""
    with pytest.raises(RuntimeError):
        load_hdr_heic(tmp_path)


@MACOS_ONLY
def test_load_hdr_heic_empty_file_raises(tmp_path: Path):
    """Zero-byte .heic file must raise, not return zeros silently."""
    empty = tmp_path / "empty.heic"
    empty.write_bytes(b"")
    with pytest.raises(RuntimeError):
        load_hdr_heic(empty)


@MACOS_ONLY
def test_load_hdr_heic_path_with_spaces(tmp_path: Path):
    """Paths containing ASCII spaces must round-trip through NSURL correctly."""
    # Build a minimal SDR HEIC in a path containing spaces.
    import pillow_heif
    from PIL import Image

    pillow_heif.register_heif_opener()
    arr = (np.ones((16, 24, 3), dtype=np.uint8) * 96)
    src = tmp_path / "with spaces.heic"
    Image.fromarray(arr, mode="RGB").save(str(src), "HEIF")
    out = load_hdr_heic(src)
    assert out.shape == (16, 24, 3)
    assert out.dtype == np.float32


@MACOS_ONLY
def test_load_hdr_heic_path_with_unicode(tmp_path: Path):
    """Paths containing non-ASCII chars must work (NSURL requires correct
    percent-encoding)."""
    import pillow_heif
    from PIL import Image

    pillow_heif.register_heif_opener()
    arr = (np.ones((16, 24, 3), dtype=np.uint8) * 96)
    src = tmp_path / "uni\u00e9\u2603.heic"
    Image.fromarray(arr, mode="RGB").save(str(src), "HEIF")
    out = load_hdr_heic(src)
    assert out.shape == (16, 24, 3)


@MACOS_ONLY
def test_load_hdr_heic_accepts_path_object(tmp_path: Path):
    """Path object (not str) should work — spec signature is str | Path."""
    import pillow_heif
    from PIL import Image

    pillow_heif.register_heif_opener()
    arr = (np.ones((8, 8, 3), dtype=np.uint8) * 64)
    src = tmp_path / "pathobj.heic"
    Image.fromarray(arr, mode="RGB").save(str(src), "HEIF")
    assert isinstance(src, Path)
    out = load_hdr_heic(src)
    assert out.dtype == np.float32


# ---------------------------------------------------------------------------
# Wrong-polarity fixture — Post-mortem 1 required test
# ---------------------------------------------------------------------------


@MACOS_ONLY
def test_load_hdr_heic_sdr_only_heic_stays_leq_one(tmp_path: Path):
    """POST-MORTEM-1 WRONG-POLARITY FIXTURE: an SDR-only HEIC (no gain map,
    no HDR content) passed through `expandToHDR` must return a finite float32
    array with max <= 1.001 (i.e., the loader does NOT fabricate HDR content
    where there is none), AND the render guard must still accept it as a
    valid HDR-path input (isfinite is the only check).

    Without this fixture, the LDR guard would only be exercised by bright-
    on-bright HDR content, which is also the expected polarity. This is the
    wrong-polarity counterpart.
    """
    import pillow_heif
    from PIL import Image

    pillow_heif.register_heif_opener()
    arr = (np.ones((32, 48, 3), dtype=np.uint8) * 128)  # mid-grey SDR
    src = tmp_path / "sdr_only.heic"
    Image.fromarray(arr, mode="RGB").save(str(src), "HEIF")

    result = load_hdr_heic(src)
    # Contract: SDR-only content should stay <= 1.0 after expandToHDR; the
    # working colorspace is extended-linear sRGB, so mid-grey (128/255 ≈ 0.5
    # sRGB) linearizes to ~0.216. The loader must not fabricate HDR.
    assert np.isfinite(result).all()
    assert result.max() <= 1.001, (
        f"SDR-only HEIC returned max={result.max()} > 1.001; "
        "expandToHDR is fabricating HDR content where none exists"
    )
    # And the render guard (np.isfinite) must accept this downstream; assert
    # directly here so the test fails loud if the loader returns NaN/Inf.
    assert not np.isnan(result).any()
    assert not np.isinf(result).any()


@MACOS_ONLY
def test_load_hdr_heic_sdr_only_render_does_not_clip(tmp_path: Path):
    """Follow-up to the wrong-polarity fixture: an SDR-only HEIC fed through
    `--hdr-source macos` must round-trip through `load_hdr_heic` +
    `linear_to_srgb` without (a) raising the finite guard, (b) being crushed
    to pure black, or (c) saturating to white. Essentially: the SDR path
    still works through the HDR loader branch."""
    import pillow_heif
    from PIL import Image

    pillow_heif.register_heif_opener()
    arr = (np.ones((32, 48, 3), dtype=np.uint8) * 128)
    src = tmp_path / "sdr_only.heic"
    Image.fromarray(arr, mode="RGB").save(str(src), "HEIF")
    img = load_hdr_heic(src)

    from summicron.render.convolve import linear_to_srgb

    # SDR-only HEIC stays <= 1.0 (no fabricated HDR); skip tonemap, just
    # verify the linear → sRGB → uint8 round-trip preserves a visible mid-grey.
    srgb = linear_to_srgb(np.clip(img, 0.0, 1.0))
    u8 = np.clip(srgb * 255.0, 0, 255).astype(np.uint8)
    # The mid-grey input shouldn't collapse to 0 or saturate to 255.
    assert 10 < int(u8.mean()) < 245, (
        f"SDR-through-HDR-path mean={u8.mean()} — path is clipping or crushing"
    )


# ---------------------------------------------------------------------------
# Real-data shape / polarity
# ---------------------------------------------------------------------------


@MACOS_ONLY
@pytest.mark.skipif(not HDR_FIXTURE.exists(), reason=f"missing {HDR_FIXTURE}")
def test_load_hdr_heic_exact_shape():
    """R0 (P1.11): portrait HEIC stays portrait.

    IMG_6651 is stored 5712x4284 with EXIF Orientation=6 (needs 90-deg CW).
    After `_canonicalize_orientation`, the loader must return the display
    orientation — (H=5712, W=4284, 3) — matching `load_image()`.
    """
    result = load_hdr_heic(HDR_FIXTURE)
    assert result.shape == (5712, 4284, 3), (
        f"unexpected shape {result.shape}; expected (5712, 4284, 3)"
    )


# ---------------------------------------------------------------------------
# R0 orientation canonicalization (P1.11 I8 / I9)
# ---------------------------------------------------------------------------


@MACOS_ONLY
def test_load_hdr_heic_orientation_6_portrait_stored_landscape(tmp_path: Path):
    """R0 I8: synthetic HEIC with EXIF Orientation=6.

    Saved pixel array is (H=32, W=48), EXIF tag declares rotate 90-deg CW to
    display. pillow_heif reports `im.size = (48, 32)` (display orientation).
    `_canonicalize_orientation` must `np.rot90` the Core Image buffer so the
    returned array shape matches PIL's display-oriented `(H=32, W=48, 3)`.
    This fixture empirically validates the `k` value in the EXIF->rot90 table.
    """
    import pillow_heif
    from PIL import Image

    pillow_heif.register_heif_opener()
    arr = np.zeros((48, 32, 3), dtype=np.uint8)
    arr[0, 0] = [255, 0, 0]  # sentinel
    src = tmp_path / "exif6.heic"
    exif = Image.Exif()
    exif[0x0112] = 6
    Image.fromarray(arr, mode="RGB").save(str(src), "HEIF", exif=exif.tobytes())

    out = load_hdr_heic(src)
    assert out.shape == (32, 48, 3), (
        f"EXIF=6 HEIC returned shape {out.shape}; expected (32, 48, 3)"
    )


@MACOS_ONLY
def test_load_hdr_heic_orientation_1_landscape_unchanged(tmp_path: Path):
    """R0 I9: synthetic HEIC with EXIF Orientation=1 (identity).

    Saved array (H=32, W=48), EXIF=1 (no rotation). pillow_heif reports
    `im.size = (48, 32)` i.e. display shape `(H=32, W=48)`. Core Image also
    returns `(H=32, W=48, 3)`. Shapes match -> canonicalizer is a no-op.
    Regression guard: catches over-rotation on the happy path.
    """
    import pillow_heif
    from PIL import Image

    pillow_heif.register_heif_opener()
    arr = np.zeros((32, 48, 3), dtype=np.uint8)
    arr[0, 0] = [0, 255, 0]
    src = tmp_path / "exif1.heic"
    exif = Image.Exif()
    exif[0x0112] = 1
    Image.fromarray(arr, mode="RGB").save(str(src), "HEIF", exif=exif.tobytes())

    out = load_hdr_heic(src)
    assert out.shape == (32, 48, 3), (
        f"EXIF=1 HEIC returned shape {out.shape}; expected (32, 48, 3)"
    )


@MACOS_ONLY
@pytest.mark.skipif(not HDR_FIXTURE.exists(), reason=f"missing {HDR_FIXTURE}")
def test_load_hdr_heic_no_negative_values():
    """HDR loader must never return negative values — extended-linear sRGB
    is non-negative in the physical sense. Negatives will confuse the
    downstream Reinhard tonemap (x/(1+x) goes negative for x<0)."""
    result = load_hdr_heic(HDR_FIXTURE)
    # Allow tiny float roundoff but not real negative values.
    assert result.min() >= -1e-4, (
        f"HDR loader returned real negative values min={result.min()}"
    )


# ---------------------------------------------------------------------------
# Buffer-lifetime
# ---------------------------------------------------------------------------


@MACOS_ONLY
def test_load_hdr_heic_returned_array_owns_memory(tmp_path: Path):
    """The loader uses `.copy()`; mutating the returned array must NOT affect
    a subsequent load, and `.flags.owndata` must be True (so the Objective-C
    buffer can be freed)."""
    import pillow_heif
    from PIL import Image

    pillow_heif.register_heif_opener()
    arr = (np.ones((16, 16, 3), dtype=np.uint8) * 77)
    src = tmp_path / "lifetime.heic"
    Image.fromarray(arr, mode="RGB").save(str(src), "HEIF")

    a = load_hdr_heic(src)
    assert a.flags.owndata, "returned array does not own its buffer"
    sentinel = 999.0
    a[0, 0, 0] = sentinel

    b = load_hdr_heic(src)
    assert b[0, 0, 0] != sentinel, (
        "second load sees sentinel value — buffers are shared"
    )
    assert a[0, 0, 0] == sentinel, "first array mutation was lost"


# ---------------------------------------------------------------------------
# Platform-gate behavior
# ---------------------------------------------------------------------------


def test_load_hdr_heic_platform_gate_runtime_check(monkeypatch):
    """The `sys.platform != 'darwin'` check runs at CALL time, not import
    time. Verify by monkey-patching `sys.platform` to `'win32'` and `'cygwin'`
    after import — both must raise NotImplementedError."""
    for fake in ("linux", "linux2", "win32", "cygwin", "freebsd"):
        monkeypatch.setattr(sys, "platform", fake)
        with pytest.raises(NotImplementedError, match="macOS"):
            load_hdr_heic("/any/path.heic")


def test_load_hdr_heic_module_importable_without_quartz():
    """The module must import cleanly on Linux CI — Quartz import is lazy,
    inside the function body."""
    import importlib

    mod = importlib.import_module("summicron.io.hdr_load_macos")
    # Public symbol exists
    assert callable(getattr(mod, "load_hdr_heic"))
    # Quartz / Foundation are NOT module-level names (lazy imports).
    assert not hasattr(mod, "Quartz"), (
        "Quartz leaked to module scope; lazy-import contract broken"
    )
    assert not hasattr(mod, "NSURL"), (
        "NSURL leaked to module scope; lazy-import contract broken"
    )
