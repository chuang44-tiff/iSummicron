"""macOS HDR HEIC loader via Core Image's `expandToHDR` option.

Returns extended-linear float32 RGB in `(H, W, 3)`. Values may exceed 1.0 on
true HDR captures (iPhone 12+ HEIC with HDR gain-map). SDR-only captures are
NOT guaranteed to stay <= 1.0: a Display-P3-tagged SDR image loads with max
~1.34 (P3 -> extended-linear-sRGB gamut excursion), so ``max > 1`` alone is
not an HDR discriminator.

The module is safe to import on non-macOS platforms; the single public call
`load_hdr_heic()` raises `NotImplementedError` there. Heavy Objective-C
frameworks (Quartz / Foundation) are imported lazily *inside* the function so
Linux CI can import this module for test discovery without installing
pyobjc-framework-Quartz.

The CIImage `options:` argument must be a real `NSDictionary` (built via
`NSDictionary.dictionaryWithObject_forKey_`): on macOS 27 + pyobjc 12 any
bridged Python `dict` there raises `NSInvalidArgumentException`.
"""
from __future__ import annotations

import platform
import sys
import warnings
from pathlib import Path

import numpy as np

# EXIF Orientation (tag 0x0112) -> numpy op mapping stored -> display.
# Each entry is a callable taking a (H, W, 3) array and returning the
# display-oriented array. k=3 for EXIF=6 verified empirically (I8); other
# entries follow the canonical EXIF Orientation table.
# iPhone emits only {1, 3, 6, 8}, but the table covers all 8 for correctness
# on non-iPhone HEIC and for the R0 adversarial probes.
_EXIF_TO_OP = {
    1: lambda a: a,                                 # identity
    2: lambda a: np.fliplr(a),                      # mirror horizontal
    3: lambda a: np.rot90(a, k=2),                  # rotate 180
    4: lambda a: np.flipud(a),                      # mirror vertical
    5: lambda a: np.rot90(np.fliplr(a), k=1),       # transpose (TL-BR diag)
    6: lambda a: np.rot90(a, k=3),                  # 90 CW (display undo)
    7: lambda a: np.rot90(np.fliplr(a), k=3),       # transverse (TR-BL diag)
    8: lambda a: np.rot90(a, k=1),                  # 90 CCW
}


def _canonicalize_orientation(arr: np.ndarray, path: str | Path) -> np.ndarray:
    """Rotate `arr` so (H, W) matches PIL's canonical (display) shape.

    Core Image returns the raw stored bitmap ignoring EXIF; pillow_heif
    applies EXIF and exposes display orientation via `PIL.Image.size`.

    Only EXIF=1 is a sound shape-equality short-circuit. For 2/3/4 the
    rotation/mirror preserves (H, W) so a naive shape match would silently
    skip the transform; for square images, every rotation preserves shape.
    Therefore we read the EXIF tag *first* and only short-circuit on EXIF==1.
    """
    from PIL import Image

    try:
        import pillow_heif
        pillow_heif.register_heif_opener()
    except ImportError:
        pass
    try:
        with Image.open(str(path)) as pil:
            w_canon, h_canon = pil.size
            # `original_orientation` is sometimes present but None on HEICs
            # without an EXIF Orientation tag; treat both missing-key and
            # None-valued as "fall back to EXIF tag, then identity".
            orientation = pil.info.get("original_orientation")
            if orientation is None:
                orientation = pil.getexif().get(0x0112, 1)
    except Exception as exc:
        raise RuntimeError(
            f"_canonicalize_orientation: PIL failed to open {path}: {exc}"
        ) from exc

    if orientation == 1:
        return np.ascontiguousarray(arr)
    op = _EXIF_TO_OP.get(orientation)
    if op is None:
        warnings.warn(
            f"unexpected EXIF Orientation={orientation} for {path}; "
            "defaulting to identity",
            RuntimeWarning, stacklevel=2,
        )
        op = _EXIF_TO_OP[1]
    arr = op(arr)
    if arr.shape != (h_canon, w_canon, 3):
        raise RuntimeError(
            f"_canonicalize_orientation: post-transform shape {arr.shape} != "
            f"({h_canon}, {w_canon}, 3) for {path} (EXIF={orientation})"
        )
    return np.ascontiguousarray(arr)


def load_hdr_heic(path: str | Path) -> np.ndarray:
    """Load a HEIC as extended-linear float32 RGB via Core Image expandToHDR.

    Parameters
    ----------
    path : str | Path
        Filesystem path to a .HEIC file.

    Returns
    -------
    np.ndarray
        `(H, W, 3)` float32, extended-linear. Values may exceed 1.0 for HDR
        captures; alpha is dropped. No sRGB linearization downstream — the
        pixels are already in linear light.

    Raises
    ------
    NotImplementedError
        On any non-macOS platform.
    """
    if sys.platform != "darwin":
        raise NotImplementedError(
            "load_hdr_heic requires macOS (Core Image)"
        )

    # Lazy imports: keep this module importable on Linux CI.
    # P1.audit Item 1: actionable error if the `[hdr-macos]` extra is missing
    # (e.g. user installed plain wheel and passed --hdr-source macos).
    try:
        import Quartz
        from Foundation import NSURL, NSDictionary, NSNumber
    except ImportError as e:
        raise ImportError(
            "macOS HDR loader requires `pip install isummicron[hdr-macos]` "
            "(installs pyobjc-framework-Quartz, which provides Core Image). "
            "Use --hdr-source none to fall back to LDR sRGB load."
        ) from e

    url = NSURL.fileURLWithPath_(str(path))
    key = getattr(Quartz, "kCIImageExpandToHDR", "kCIImageExpandToHDR")
    # macOS 27 + pyobjc 12: a bridged Python dict raises
    # NSInvalidArgumentException ("key does not exist"); pass a real NSDictionary.
    # Do NOT fall back to options=None on failure: that loads without HDR
    # expansion (render succeeds but silently loses headroom / drifts).
    options = NSDictionary.dictionaryWithObject_forKey_(
        NSNumber.numberWithBool_(True), key
    )
    try:
        ci_image = Quartz.CIImage.imageWithContentsOfURL_options_(url, options)
    except Exception as exc:  # objc bridge raises ValueError / objc.error
        raise RuntimeError(
            f"Core Image rejected HDR load of {path} "
            f"(macOS {platform.mac_ver()[0]}): {exc}. "
            "Use --hdr-source none to fall back to LDR sRGB load."
        ) from exc
    if ci_image is None:
        raise RuntimeError(f"CIImage failed to load: {path}")

    extent = ci_image.extent()
    # extent.size is a CGSize; cast to ints for ndarray dims.
    width = int(extent.size.width)
    height = int(extent.size.height)
    if width <= 0 or height <= 0:
        raise RuntimeError(
            f"CIImage has invalid extent: {width}x{height} (path={path})"
        )

    # Default CIContext — working colorspace is extended-linear sRGB.
    ci_context = Quartz.CIContext.context()

    row_bytes = width * 4 * 4  # RGBA float32 = 16 bytes/pixel
    buf = bytearray(row_bytes * height)
    bounds = Quartz.CGRectMake(0, 0, width, height)

    ci_context.render_toBitmap_rowBytes_bounds_format_colorSpace_(
        ci_image,
        buf,
        row_bytes,
        bounds,
        Quartz.kCIFormatRGBAf,
        None,  # colorSpace=None: passthrough of CIContext working space
    )

    rgba = np.frombuffer(buf, dtype=np.float32).reshape(height, width, 4)
    # Drop alpha; np.maximum below allocates a fresh array (no explicit copy).
    # P1.8 bug 4: clip extended-linear out-of-gamut negatives (Core Image
    # returns ~-0.004 on IMG_6651). Below optical noise floor; keeps downstream
    # Reinhard well-defined.
    arr = np.maximum(rgba[:, :, :3], 0.0)
    # R0 (P1.11): Core Image ignores EXIF Orientation; align to PIL canonical
    # display shape so renderer sees the same grid regardless of load path.
    return _canonicalize_orientation(arr, path)
