"""R-hdr-macos27: CIImage `options:` must be a real NSDictionary.

On macOS 27.2 + pyobjc 12.2 a bridged Python ``dict`` passed as the
``options:`` argument of ``CIImage.imageWithContentsOfURL_options_`` raises
``NSInvalidArgumentException - key does not exist``. The loader must build
the options via ``NSDictionary.dictionaryWithObject_forKey_`` and must wrap
any bridge exception in an actionable ``RuntimeError`` (never silently fall
back to ``options=None``, which loads without HDR expansion).

The fake-module tests run on any platform; the real-HEIC test is slow-marked
and darwin-only.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

from summicron.io.hdr_load_macos import load_hdr_heic

REPO_ROOT = Path(__file__).resolve().parents[2]
HDR_FIXTURE_6915 = REPO_ROOT / "iphoneImage" / "IMG_6915.HEIC"


class _FakeNSDict:
    """Stand-in for an NSDictionary built via dictionaryWithObject_forKey_."""

    def __init__(self, obj, key):
        self.obj = obj
        self.key = key


def _install_fakes(monkeypatch, *, raise_exc: Exception | None = None):
    record: dict = {}

    class NSDictionary:
        @staticmethod
        def dictionaryWithObject_forKey_(obj, key):
            record["constructor"] = "dictionaryWithObject_forKey_"
            return _FakeNSDict(obj, key)

    class NSNumber:
        @staticmethod
        def numberWithBool_(b):
            record["nsnumber_bool"] = b
            return ("NSNumber", bool(b))

    class NSURL:
        @staticmethod
        def fileURLWithPath_(p):
            return ("NSURL", p)

    class CIImage:
        @staticmethod
        def imageWithContentsOfURL_options_(url, options):
            record["url"] = url
            record["options"] = options
            record["calls"] = record.get("calls", 0) + 1
            if raise_exc is not None:
                raise raise_exc
            return None

    quartz = types.ModuleType("Quartz")
    quartz.kCIImageExpandToHDR = "kCIImageExpandToHDR"
    quartz.CIImage = CIImage
    foundation = types.ModuleType("Foundation")
    foundation.NSURL = NSURL
    foundation.NSDictionary = NSDictionary
    foundation.NSNumber = NSNumber

    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setitem(sys.modules, "Quartz", quartz)
    monkeypatch.setitem(sys.modules, "Foundation", foundation)
    return record


def test_options_is_nsdictionary_with_expand_to_hdr(monkeypatch):
    record = _install_fakes(monkeypatch)
    with pytest.raises(RuntimeError, match="CIImage failed to load"):
        load_hdr_heic("/nonexistent/fake.heic")

    assert record["constructor"] == "dictionaryWithObject_forKey_"
    options = record["options"]
    assert not isinstance(options, dict)
    assert isinstance(options, _FakeNSDict)
    assert options.key == "kCIImageExpandToHDR"
    assert record["nsnumber_bool"] is True
    assert options.obj == ("NSNumber", True)


def test_bridge_exception_wrapped_with_hdr_source_hint(monkeypatch):
    exc = ValueError("NSInvalidArgumentException - key does not exist")
    record = _install_fakes(monkeypatch, raise_exc=exc)
    with pytest.raises(RuntimeError, match="--hdr-source none") as ei:
        load_hdr_heic("/nonexistent/fake.heic")
    assert ei.value.__cause__ is exc
    # Loud failure: exactly one CIImage call, no silent options=None retry.
    assert record["calls"] == 1
    assert isinstance(record["options"], _FakeNSDict)


@pytest.mark.slow
@pytest.mark.skipif(sys.platform != "darwin", reason="Core Image is macOS-only")
@pytest.mark.skipif(
    not HDR_FIXTURE_6915.exists(), reason=f"missing {HDR_FIXTURE_6915}"
)
def test_real_heic_hdr_expansion_present():
    from summicron.render.convolve import load_image

    hdr = load_hdr_heic(HDR_FIXTURE_6915)
    assert hdr.dtype == np.float32
    assert hdr.ndim == 3 and hdr.shape[-1] == 3
    assert hdr.shape == load_image(HDR_FIXTURE_6915).shape
    # Without expandToHDR the max is ~1.35 / p99 ~0.93 (no headroom).
    assert float(hdr.max()) > 3.0
    assert float(np.percentile(hdr, 99)) > 1.5
