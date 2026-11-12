"""Cheap detection for ``--hdr-source auto`` (section20 Rev 3).

No Objective-C framework is imported and no pixel is decoded here: the
macOS loader is probed with ``importlib.util.find_spec`` and the photo's
HDR gain map with a HEIF header parse (about 1 ms).
"""
from __future__ import annotations

import sys
from pathlib import Path


def macos_hdr_loader_available() -> bool:
    """True iff on macOS and both pyobjc modules the HDR loader imports
    (``Quartz``, ``Foundation``) are findable."""
    if sys.platform != "darwin":
        return False
    import importlib.util

    try:
        return all(
            importlib.util.find_spec(m) is not None for m in ("Quartz", "Foundation")
        )
    except (ImportError, ValueError):
        return False


def has_hdr_gainmap(path: str | Path) -> bool:
    """True iff ``path`` is a HEIC/HEIF carrying an Apple HDR gain-map aux image.

    Any failure returns False; the subsequent real load reports the actual
    error.
    """
    p = Path(path)
    if p.suffix.lower() not in (".heic", ".heif"):
        return False
    try:
        import pillow_heif

        info = pillow_heif.open_heif(str(p)).info
        return any("hdrgainmap" in str(k).lower() for k in (info.get("aux") or {}))
    except Exception:
        return False
