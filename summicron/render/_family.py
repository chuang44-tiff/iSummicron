"""Synthetic disc kernel family: run-time build + user-cache (section 22 S1).

``load_family_tensor(aperture_mm)`` returns a ``load_tensor``-validated tensor
dict for the physically sized defocus-disc family of
``summicron.psf_model.synth_family`` (research: ``doc/plan/phase2/
section22_blur-magnitude/``). The family is generated from the wheel-shipped
CodeV artifact at run time and cached under a user cache dir; nothing is ever
written under ``summicron/_artifacts/`` (spec I3).

Cache dir: ``$ISUMMICRON_CACHE_DIR``, else ``~/Library/Caches/isummicron`` on
darwin, else ``$XDG_CACHE_HOME/isummicron`` or ``~/.cache/isummicron``; the
family lives in the ``synth/`` subdir as ``synth_<key16>.npz``.

Numpy is imported lazily (this module is cheap to import).
"""
from __future__ import annotations

import hashlib
import os
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Callable

FAMILY_NAME = "synthetic_disc_v0"
# Fixed family parameters (spec section 2: no flags, part of the cache key).
PSF_SCALE = 0.25
RIM_BETA = 0.3
CAT_EYE = 0.0
AXIS = "diopter16"
N_D = 16
DEFAULT_APERTURE_MM = 17.5
DEFAULT_BRIGHT_STAMP_GAIN = 2.0
# Aperture range (mm) accepted by RenderOptions: the 183 px kernel box holds
# the far-node disc up to ~34 mm; 30 mm leaves margin.
APERTURE_MIN_MM = 3.0
APERTURE_MAX_MM = 30.0

_CACHE_ERRORS = (OSError, ValueError, zipfile.BadZipFile, KeyError, EOFError)


def default_cache_dir() -> Path:
    """The ``synth/`` cache directory (not created)."""
    env = os.environ.get("ISUMMICRON_CACHE_DIR")
    if env:
        base = Path(env)
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Caches" / "isummicron"
    else:
        xdg = os.environ.get("XDG_CACHE_HOME")
        base = (Path(xdg) if xdg else Path.home() / ".cache") / "isummicron"
    return base / "synth"


def _source_md5() -> str:
    from summicron.render._artifacts import default_tensor_path

    h = hashlib.md5()  # noqa: S324 - content id, not security
    with open(default_tensor_path(), "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def family_key(aperture_mm: float) -> str:
    """First 16 hex of sha256 over the canonical family description."""
    from summicron.psf_model import synth_family

    s = (
        f"family={FAMILY_NAME}|builder={synth_family.BUILDER_VERSION}"
        f"|src={_source_md5()}|A={float(aperture_mm):.6f}|s={PSF_SCALE:g}"
        f"|beta={RIM_BETA:g}|cat={CAT_EYE:.1f}|axis={AXIS}|n_d={N_D}"
    )
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def _kernels_sha256(kernels) -> str:
    import numpy as np

    return hashlib.sha256(np.ascontiguousarray(kernels).tobytes()).hexdigest()


def _build_payload(aperture_mm: float, key: str) -> dict:
    import numpy as np

    from summicron.psf_model import synth_family

    payload = synth_family.build_synthetic_tensor(
        synth_family.DEFAULT_SOURCE,
        aperture_mm=float(aperture_mm), n_d=N_D, axis=AXIS,
        psf_scale=PSF_SCALE, rim_beta=RIM_BETA, cat_eye=CAT_EYE,
    )
    payload["schema_version"] = np.array("1.0")
    payload["synth_key"] = np.array(key)
    payload["synth_kernels_sha256"] = np.array(_kernels_sha256(payload["kernels"]))
    return payload


def _write_atomic(payload: dict, directory: Path, final: Path, key: str) -> None:
    import numpy as np

    directory.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=f"synth_{key}.", suffix=".tmp.npz")
    try:
        with os.fdopen(fd, "wb") as f:
            np.savez(f, **payload)
        os.replace(tmp, final)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _try_cached(path: Path, key: str):
    """Return the validated tensor dict of an existing cache file, or None."""
    import numpy as np

    from summicron.render.convolve import TensorValidationError, load_tensor

    if not path.is_file():
        return None
    try:
        with np.load(str(path), allow_pickle=False) as data:
            if str(data["synth_key"]) != key:
                return None
            if "synth_family" not in data.files or str(data["synth_family"]) != FAMILY_NAME:
                return None
            if str(data["synth_kernels_sha256"]) != _kernels_sha256(data["kernels"]):
                return None
        return load_tensor(path)
    except (*_CACHE_ERRORS, TensorValidationError):
        return None


def load_family_tensor(
    aperture_mm: float,
    *,
    cache_dir: Path | str | None = None,
    on_progress: Callable[[str], None] | None = None,
) -> tuple[dict, Path, bool]:
    """Build (or load from cache) the synthetic family for ``aperture_mm``.

    Returns ``(tensor, path, cache_hit)``; ``tensor`` is the
    ``load_tensor``-validated dict. A corrupt, truncated or foreign-key cache
    file is rebuilt (at most one build per call). An unwritable cache dir
    falls back to a throw-away temp dir with one progress note.
    """
    import math

    from summicron.render.convolve import load_tensor

    if not math.isfinite(float(aperture_mm)):
        raise ValueError(f"aperture_mm must be finite, got {aperture_mm!r}")
    directory = Path(cache_dir) if cache_dir is not None else default_cache_dir()
    key = family_key(aperture_mm)
    final = directory / f"synth_{key}.npz"

    hit = _try_cached(final, key)
    if hit is not None:
        return hit, final, True

    if on_progress is not None:
        on_progress(f"  building synthetic family A={float(aperture_mm):g} mm (~3 s)...")
    payload = _build_payload(aperture_mm, key)
    try:
        _write_atomic(payload, directory, final, key)
    except OSError:
        if on_progress is not None:
            on_progress(
                f"  note: cache dir {directory} is not writable; "
                "using a temporary family file"
            )
        import numpy as np

        tmpdir = tempfile.TemporaryDirectory(prefix="isummicron_synth_")
        try:
            tmp_path = Path(tmpdir.name) / f"synth_{key}.npz"
            np.savez(str(tmp_path), **payload)
            # Report the intended (unwritable) cache path, not the cleaned temp one.
            return load_tensor(tmp_path), final, False
        finally:
            tmpdir.cleanup()
    return load_tensor(final), final, False
