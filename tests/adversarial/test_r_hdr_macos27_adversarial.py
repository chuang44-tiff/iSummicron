"""R-hdr-macos27 adversarial probes for ``summicron.io.hdr_load_macos``.

Targets the macOS 27 / pyobjc 12 fix: the CIImage ``options:`` argument is
now a real ``NSDictionary`` and bridge exceptions are wrapped in an actionable
``RuntimeError`` (no silent ``options=None`` fallback, which would load
without HDR expansion and drift ~16 LSB from the G1 pin).

Three groups:
  * fake-module probes (any platform): key-constant absence, silent-fallback
    trap, exception taxonomy (wrapped vs. propagated), missing Foundation
    symbols;
  * real Core Image probes (darwin-only): odd inputs, WRONG-POLARITY LDR
    fixtures (headroom must NOT be fabricated; ``max > 1`` alone is not an
    HDR discriminator), every repo HEIC carries headroom, orientation,
    cross-process determinism;
  * CLI preflight on non-darwin.
"""
from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from summicron.io import hdr_load_macos
from summicron.io.hdr_load_macos import load_hdr_heic

REPO_ROOT = Path(__file__).resolve().parents[2]
IMG_DIR = REPO_ROOT / "iphoneImage"
REAL_HEICS = sorted(
    p for p in IMG_DIR.glob("*") if p.suffix.lower() == ".heic"
) if IMG_DIR.exists() else []
HDR_6915 = IMG_DIR / "IMG_6915.HEIC"

darwin_only = pytest.mark.skipif(
    sys.platform != "darwin", reason="Core Image is macOS-only"
)


# --------------------------------------------------------------------------
# Fake Quartz / Foundation harness
# --------------------------------------------------------------------------
class _FakeNSDict:
    def __init__(self, obj, key):
        self.obj = obj
        self.key = key


def _install_fakes(
    monkeypatch,
    *,
    ciimage_impl=None,
    with_key_constant: bool = True,
    foundation_has_nsdictionary: bool = True,
):
    record: dict = {"calls": []}

    class NSDictionary:
        @staticmethod
        def dictionaryWithObject_forKey_(obj, key):
            return _FakeNSDict(obj, key)

    class NSNumber:
        @staticmethod
        def numberWithBool_(b):
            return ("NSNumber", bool(b))

    class NSURL:
        @staticmethod
        def fileURLWithPath_(p):
            return ("NSURL", p)

    def _default_impl(url, options):
        return None

    impl = ciimage_impl or _default_impl

    class CIImage:
        @staticmethod
        def imageWithContentsOfURL_options_(url, options):
            record["calls"].append(options)
            return impl(url, options)

    quartz = types.ModuleType("Quartz")
    if with_key_constant:
        quartz.kCIImageExpandToHDR = "kCIImageExpandToHDR"
    quartz.CIImage = CIImage
    foundation = types.ModuleType("Foundation")
    foundation.NSURL = NSURL
    if foundation_has_nsdictionary:
        foundation.NSDictionary = NSDictionary
    foundation.NSNumber = NSNumber

    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setitem(sys.modules, "Quartz", quartz)
    monkeypatch.setitem(sys.modules, "Foundation", foundation)
    return record


def test_key_constant_absent_falls_back_to_literal_key(monkeypatch):
    """Spec: ``getattr(Quartz, 'kCIImageExpandToHDR', 'kCIImageExpandToHDR')``.
    A pyobjc build lacking the constant must still request expansion via
    the literal key (verified == the constant's value), in an NSDictionary."""
    record = _install_fakes(monkeypatch, with_key_constant=False)
    with pytest.raises(RuntimeError, match="CIImage failed to load"):
        load_hdr_heic("/nonexistent/x.heic")
    (opts,) = record["calls"]
    assert isinstance(opts, _FakeNSDict) and not isinstance(opts, dict)
    assert opts.key == "kCIImageExpandToHDR"
    assert opts.obj == ("NSNumber", True)


def test_silent_fallback_trap_options_none_never_tried(monkeypatch):
    """A Core Image that REJECTS the expand options but would happily load
    with ``options=None`` is exactly the silent-headroom-loss scenario. The
    loader must fail loudly and must never issue an options=None retry."""
    sentinel = object()

    def impl(url, options):
        if options is None:
            return sentinel  # would 'succeed' without HDR expansion
        raise ValueError("NSInvalidArgumentException - key does not exist")

    record = _install_fakes(monkeypatch, ciimage_impl=impl)
    with pytest.raises(RuntimeError, match="--hdr-source none"):
        load_hdr_heic("/nonexistent/x.heic")
    assert None not in record["calls"]
    assert len(record["calls"]) == 1


def test_non_valueerror_bridge_exception_wrapped_with_context(monkeypatch):
    """objc.error is not a ValueError; any Exception subclass from the
    bridge must be wrapped with path, macOS version and the hint, and
    preserve ``__cause__``."""

    class FakeObjcError(Exception):
        pass

    exc = FakeObjcError("NSInternalInconsistencyException - boom")

    def impl(url, options):
        raise exc

    _install_fakes(monkeypatch, ciimage_impl=impl)
    path = "/some dir/ünï 照片.HEIC"
    with pytest.raises(RuntimeError) as ei:
        load_hdr_heic(path)
    msg = str(ei.value)
    assert "--hdr-source none" in msg
    assert path in msg
    assert "boom" in msg
    assert "macOS" in msg
    assert ei.value.__cause__ is exc


def test_keyboardinterrupt_not_swallowed_into_runtimeerror(monkeypatch):
    """``except Exception`` must not capture BaseException: Ctrl-C during a
    slow Core Image decode must propagate as KeyboardInterrupt."""

    def impl(url, options):
        raise KeyboardInterrupt

    _install_fakes(monkeypatch, ciimage_impl=impl)
    with pytest.raises(KeyboardInterrupt):
        load_hdr_heic("/nonexistent/x.heic")


def test_foundation_without_nsdictionary_gives_actionable_importerror(
    monkeypatch,
):
    """Older/partial pyobjc lacking NSDictionary must surface the
    ImportError hint path, not an AttributeError/NameError mid-load."""
    _install_fakes(monkeypatch, foundation_has_nsdictionary=False)
    with pytest.raises(ImportError, match="--hdr-source none"):
        load_hdr_heic("/nonexistent/x.heic")


def test_non_darwin_loader_raises_notimplemented(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    with pytest.raises(NotImplementedError):
        load_hdr_heic("/nonexistent/x.heic")


def test_cli_preflight_rejects_macos_hdr_on_non_darwin(monkeypatch, capsys):
    from summicron.render.convolve import _validate_args, build_parser

    monkeypatch.setattr(sys, "platform", "linux")
    parser = build_parser()
    args = parser.parse_args(["--image", "x.heic", "--hdr-source", "macos"])
    with pytest.raises(SystemExit) as ei:
        _validate_args(args, parser)
    assert ei.value.code == 2
    err = capsys.readouterr().err
    assert "--hdr-source macos requires macOS" in err
    assert "--hdr-source none" in err


# --------------------------------------------------------------------------
# Real Core Image: odd inputs (fast — CIImage returns None quickly)
# --------------------------------------------------------------------------
def _odd_input(kind: str, tmp_path: Path) -> Path:
    if kind == "missing":
        return tmp_path / "nope.HEIC"
    if kind == "directory":
        d = tmp_path / "a_dir.HEIC"
        d.mkdir()
        return d
    if kind == "zero_byte":
        p = tmp_path / "zero.HEIC"
        p.write_bytes(b"")
        return p
    if kind == "text_as_heic":
        p = tmp_path / "text.HEIC"
        p.write_text("not an image\n" * 200)
        return p
    if kind == "truncated_heic":
        p = tmp_path / "trunc.HEIC"
        p.write_bytes(HDR_6915.read_bytes()[:4096])
        return p
    raise AssertionError(kind)


@darwin_only
@pytest.mark.parametrize(
    "kind",
    ["missing", "directory", "zero_byte", "text_as_heic", "truncated_heic"],
)
def test_odd_inputs_raise_clear_runtimeerror(kind, tmp_path):
    if kind == "truncated_heic" and not HDR_6915.exists():
        pytest.skip("needs IMG_6915.HEIC")
    p = _odd_input(kind, tmp_path)
    with pytest.raises(RuntimeError) as ei:
        load_hdr_heic(p)
    assert str(p) in str(ei.value)


# --------------------------------------------------------------------------
# WRONG-POLARITY fixtures: LDR inputs must NOT show fabricated headroom
# --------------------------------------------------------------------------
@darwin_only
def test_ldr_srgb_png_no_icc_never_exceeds_one(tmp_path):
    """Untagged 8-bit sRGB with a saturated white patch: expandToHDR on a
    gain-map-less image must not invent headroom. Max is exactly 1.0."""
    from PIL import Image

    rng = np.random.default_rng(0)
    arr = rng.integers(0, 256, size=(96, 128, 3), dtype=np.uint8)
    arr[:20, :20] = 255
    p = tmp_path / "ldr.png"
    Image.fromarray(arr).save(p)
    out = load_hdr_heic(p)
    assert out.shape == (96, 128, 3) and out.dtype == np.float32
    assert float(out.max()) <= 1.0 + 1e-6
    assert float(out[:20, :20].min()) >= 1.0 - 1e-6  # white maps to 1.0


@darwin_only
def test_ldr_low_key_dark_frame_stays_dark(tmp_path):
    """Inverted polarity: a low-key frame whose brightest pixel is ~10% of
    full scale. A '>1.0 means HDR' gate would fail here by design; the
    loader must neither crash nor lift the frame into fake headroom."""
    from PIL import Image

    rng = np.random.default_rng(1)
    arr = rng.integers(0, 26, size=(64, 80, 3), dtype=np.uint8)
    p = tmp_path / "dark.png"
    Image.fromarray(arr).save(p)
    out = load_hdr_heic(p)
    assert np.isfinite(out).all()
    # sRGB 25/255 -> linear ~0.0097.
    assert float(out.max()) < 0.02
    assert float(out.min()) >= 0.0


@darwin_only
@pytest.mark.slow
@pytest.mark.skipif(not HDR_6915.exists(), reason="needs IMG_6915.HEIC")
def test_p3_tagged_ldr_exceeds_one_but_is_not_hdr(tmp_path):
    """Display-P3-tagged SDR derived from the HDR HEIC (gain map dropped):
    gamut conversion to extended-linear sRGB pushes a few saturated pixels
    above 1.0, so ``max > 1`` is NOT an HDR-expansion discriminator. The
    bulk (p99) must stay < 1 while the true HDR load has p99 > 1.5."""
    from PIL import Image
    import pillow_heif

    pillow_heif.register_heif_opener()
    with Image.open(HDR_6915) as im:
        icc = im.info.get("icc_profile")
        rgb = im.convert("RGB").resize((535, 714))
    p = tmp_path / "p3_ldr.jpg"
    rgb.save(p, quality=95, icc_profile=icc)
    ldr = load_hdr_heic(p)
    assert float(np.percentile(ldr, 99)) < 1.0
    assert float(ldr.max()) < 2.0
    hdr = load_hdr_heic(HDR_6915)
    assert float(np.percentile(hdr, 99)) > 1.5


# (HYG-0 item 16: strict-xfail `test_docstring_contract_sdr_stays_le_one`
# deleted together with the module-docstring fix that dropped the
# "SDR stays <= 1.0" overclaim.)


# --------------------------------------------------------------------------
# Real HEICs: headroom present on every repo HEIC + orientation
# --------------------------------------------------------------------------
@darwin_only
@pytest.mark.slow
@pytest.mark.parametrize("path", REAL_HEICS, ids=[p.name for p in REAL_HEICS])
def test_every_repo_heic_has_hdr_headroom_and_display_shape(path):
    """All repo HEICs are gain-map HDR captures (probe: max 2.54-6.51,
    p99 1.42-5.34). Without expandToHDR IMG_6915 gives max 1.35 / p99 0.93,
    so p99 > 1.2 AND max > 2.0 discriminates a silent no-expansion load."""
    from summicron.render.convolve import load_image

    hdr = load_hdr_heic(path)
    assert hdr.dtype == np.float32 and hdr.ndim == 3 and hdr.shape[-1] == 3
    assert hdr.shape == load_image(path).shape
    assert np.isfinite(hdr).all() and float(hdr.min()) >= 0.0
    assert float(hdr.max()) > 2.0
    assert float(np.percentile(hdr, 99)) > 1.2


@darwin_only
@pytest.mark.slow
@pytest.mark.skipif(not HDR_6915.exists(), reason="needs IMG_6915.HEIC")
def test_space_and_unicode_path_loads_identically(tmp_path):
    d = tmp_path / "dir with space ünï"
    d.mkdir()
    p = d / "照片 copy.HEIC"
    shutil.copy(HDR_6915, p)
    a = load_hdr_heic(p)
    b = load_hdr_heic(HDR_6915)
    assert a.shape == b.shape
    assert np.array_equal(a, b)


# Each fresh process loads twice, saves only its FIRST load to argv[2] (tmp
# volume is tight; one .npy per process), and reports its own within-process
# max-abs / differing-pixel count so a mismatch explains itself.
_DET_SCRIPT = r"""
import hashlib, sys
import numpy as np
from summicron.io.hdr_load_macos import load_hdr_heic
p, out = sys.argv[1], sys.argv[2]
a = load_hdr_heic(p)
b = load_hdr_heic(p)
np.save(out, a)
d = np.abs(a.astype(np.float64) - b.astype(np.float64))
diff_px = int(np.any(d > 0, axis=-1).sum()) if d.ndim == 3 else int((d > 0).sum())
print(hashlib.md5(a.tobytes()).hexdigest(), hashlib.md5(b.tobytes()).hexdigest(),
      repr(float(d.max())), diff_px)
"""


def _platform_diag() -> str:
    """macOS build + pyobjc version for determinism-failure messages."""
    import platform
    try:
        build = subprocess.run(
            ["sw_vers", "-buildVersion"], capture_output=True, text=True, timeout=10
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        build = "?"
    try:
        import objc
        pyobjc = getattr(objc, "__version__", "?")
    except ImportError:
        pyobjc = "not installed"
    return f"macOS {platform.mac_ver()[0]} build {build}, pyobjc {pyobjc}"


@darwin_only
@pytest.mark.slow
@pytest.mark.skipif(not HDR_6915.exists(), reason="needs IMG_6915.HEIC")
def test_determinism_within_and_across_fresh_processes(tmp_path):
    outs, npys = [], []
    for i in range(2):
        npy = tmp_path / f"proc{i}.npy"
        r = subprocess.run(
            [sys.executable, "-c", _DET_SCRIPT, str(HDR_6915), str(npy)],
            capture_output=True, text=True, timeout=300, cwd=str(REPO_ROOT),
        )
        assert r.returncode == 0, r.stderr
        outs.append(r.stdout.split())
        npys.append(npy)
    (a1, a2, a_max, a_px), (b1, b2, b_max, b_px) = outs
    assert a1 == a2, (
        f"two loads in one fresh process differ: max-abs={a_max}, "
        f"differing pixels={a_px}; {_platform_diag()}"
    )
    if not (a1 == b1 == b2):
        x = np.load(npys[0]).astype(np.float64)
        y = np.load(npys[1]).astype(np.float64)
        d = np.abs(x - y) if x.shape == y.shape else None
        detail = (
            f"shape {x.shape} vs {y.shape}" if d is None else
            f"max-abs={float(d.max())!r}, differing pixels="
            f"{int(np.any(d > 0, axis=-1).sum()) if d.ndim == 3 else int((d > 0).sum())}"
        )
        pytest.fail(
            f"loads differ across fresh processes ({a1} vs {b1}/{b2}; "
            f"within-proc-2 max-abs={b_max}, px={b_px}): {detail}; {_platform_diag()}"
        )


_NEG_SCRIPT = r"""
import sys, numpy as np, Quartz
from summicron.io.hdr_load_macos import load_hdr_heic
real = Quartz.CIImage
class _NoExpand:
    @staticmethod
    def imageWithContentsOfURL_options_(url, options):
        return real.imageWithContentsOfURL_options_(url, None)
Quartz.CIImage = _NoExpand
for p in sys.argv[1:]:
    a = load_hdr_heic(p)
    print(f"{float(a.max())!r}\t{float(np.percentile(a, 99))!r}")
"""


@darwin_only
@pytest.mark.slow
@pytest.mark.skipif(not REAL_HEICS, reason="no repo HEICs")
def test_headroom_gate_rejects_silent_no_expansion_load_on_every_heic():
    """Negative control for the headroom gate above: simulate the forbidden
    silent fallback (options=None, everything else real) in a fresh
    subprocess (avoids the post-None-load state effect noted in the spec).
    The gate (max > 2.0 AND p99 > 1.2) must FAIL for every file, proving it
    discriminates per-file, not just on IMG_6915. Probe: no-expansion max
    1.18-1.45, p99 0.90-0.97."""
    r = subprocess.run(
        [sys.executable, "-c", _NEG_SCRIPT, *map(str, REAL_HEICS)],
        capture_output=True, text=True, timeout=600, cwd=str(REPO_ROOT),
    )
    assert r.returncode == 0, r.stderr
    rows = [ln.split("\t") for ln in r.stdout.strip().splitlines()]
    assert len(rows) == len(REAL_HEICS)
    for path, (mx, p99) in zip(REAL_HEICS, rows):
        gate = float(mx) > 2.0 and float(p99) > 1.2
        assert not gate, f"gate passes WITHOUT expansion on {path.name}"
