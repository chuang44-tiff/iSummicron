"""macOS Core Image HDR→SDR tonemap via `CIToneMapHeadroom`.

Takes extended-linear float32 RGB `(H, W, 3)` and returns uint8 sRGB `(H, W, 3)`
by wrapping the buffer as a `CIImage`, applying the `CIToneMapHeadroom` filter
with `inputSourceHeadroom=max(rgb_hdr)` clamped to `[1.0, 32.0]`, and rendering
to 8-bit sRGB via a cached `CIContext`.

The module is safe to import on non-macOS platforms; the single public call
`tonemap_core_image()` raises `NotImplementedError` there. Heavy Objective-C
frameworks (Quartz) are imported lazily *inside* the function so Linux CI can
import this module for test discovery without installing pyobjc-framework-Quartz.

Requires macOS 15+ (Sequoia) for `CIToneMapHeadroom`; on older macOS the filter
lookup returns `None` and we raise `RuntimeError` with the detected version.
"""
from __future__ import annotations

import platform
import sys

import numpy as np

# Module-level CIContext cache (populated lazily on first call).
_CI_CONTEXT = None


def _get_context():
    """Return the cached CIContext, creating it on first call."""
    global _CI_CONTEXT
    if _CI_CONTEXT is None:
        import Quartz

        _CI_CONTEXT = Quartz.CIContext.context()
    return _CI_CONTEXT


def tonemap_core_image(rgb_hdr: np.ndarray) -> np.ndarray:
    """Tonemap extended-linear HDR → uint8 sRGB via Core Image.

    Parameters
    ----------
    rgb_hdr : np.ndarray
        `(H, W, 3)` float32 extended-linear RGB. Values may exceed 1.0.

    Returns
    -------
    np.ndarray
        `(H, W, 3)` uint8 sRGB. Alpha is dropped.

    Raises
    ------
    NotImplementedError
        On any non-macOS platform.
    RuntimeError
        On macOS < 15 where `CIToneMapHeadroom` is unavailable.
    """
    if sys.platform != "darwin":
        raise NotImplementedError(
            "tonemap_core_image requires macOS 15+ "
            "(Core Image CIToneMapHeadroom)"
        )

    # Lazy imports: keep this module importable on Linux CI.
    import Quartz

    # Clamp negatives — FFT convolution can reintroduce tiny negatives at
    # kernel support edges.
    rgb_hdr = np.maximum(rgb_hdr, 0.0)
    # Ensure contiguous float32 — `imageWithBitmapData_` reads the raw buffer.
    rgb_hdr = np.ascontiguousarray(rgb_hdr, dtype=np.float32)
    height, width = rgb_hdr.shape[:2]

    # Pad RGB → RGBA (float32): CIImage expects 4 channels in kCIFormatRGBAf.
    rgba_in = np.empty((height, width, 4), dtype=np.float32)
    rgba_in[..., :3] = rgb_hdr
    rgba_in[..., 3] = 1.0
    rgba_in = np.ascontiguousarray(rgba_in)

    row_bytes = width * 4 * 4  # RGBAf = 16 bytes/pixel
    size = Quartz.CGSizeMake(width, height)
    linear_cs = Quartz.CGColorSpaceCreateWithName(
        Quartz.kCGColorSpaceExtendedLinearSRGB
    )

    ci_img = Quartz.CIImage.imageWithBitmapData_bytesPerRow_size_format_colorSpace_(
        rgba_in.tobytes(),
        row_bytes,
        size,
        Quartz.kCIFormatRGBAf,
        linear_cs,
    )
    if ci_img is None:
        raise RuntimeError("CIImage.imageWithBitmapData_ returned None")

    # Headroom clamped to filter's valid range [1.0, 32.0].
    src_headroom = max(1.0, min(32.0, float(rgb_hdr.max())))

    tm_filter = Quartz.CIFilter.filterWithName_("CIToneMapHeadroom")
    if tm_filter is None:
        raise RuntimeError(
            "CIToneMapHeadroom unavailable; requires macOS 15+. "
            f"Current platform reports macOS {platform.mac_ver()[0]}."
        )
    tm_filter.setValue_forKey_(ci_img, "inputImage")
    tm_filter.setValue_forKey_(src_headroom, "inputSourceHeadroom")
    tm_filter.setValue_forKey_(1.0, "inputTargetHeadroom")
    out_img = tm_filter.valueForKey_("outputImage")
    if out_img is None:
        raise RuntimeError("CIToneMapHeadroom produced no outputImage")

    # Render to 8-bit sRGB via cached CIContext.
    ci_context = _get_context()
    srgb_cs = Quartz.CGColorSpaceCreateWithName(Quartz.kCGColorSpaceSRGB)
    out_row_bytes = width * 4  # RGBA8 = 4 bytes/pixel
    buf = bytearray(out_row_bytes * height)
    bounds = Quartz.CGRectMake(0, 0, width, height)
    ci_context.render_toBitmap_rowBytes_bounds_format_colorSpace_(
        out_img,
        buf,
        out_row_bytes,
        bounds,
        Quartz.kCIFormatRGBA8,
        srgb_cs,
    )
    rgba = np.frombuffer(bytes(buf), dtype=np.uint8).reshape(height, width, 4)
    # Drop alpha; copy so the backing buffer can be freed.
    return rgba[..., :3].copy()
