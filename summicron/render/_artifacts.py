"""Wheel-shipped tensor artifact resolver.

Two ray-traced PSF tensors are bundled inside the wheel at
``summicron/_artifacts/``:

* ``"35"``: 35 mm f/2, ray-traced from the expired US patent 3,006,249,
  Example II, using optiland (the default);
* ``"50"``: 50 mm f/2 double-Gauss (optiland sample objective), ray-traced
  with optiland.

Both are built by ``scripts/build_psf``. `default_tensor_path()` resolves a
lens key to the bundled file via ``importlib.resources``; a missing artifact
(mis-built wheel) raises a typed :class:`FileNotFoundError` with an
actionable reinstall message.
"""
from __future__ import annotations

from importlib import resources
from pathlib import Path

LENS_ARTIFACTS: dict[str, str] = {
    "35": "psf_lens35_us3006249_fu0.4.npz",
    "50": "psf_lens50_doublegauss_fu0.4.npz",
}
DEFAULT_LENS = "35"

# Literal file name of the default artifact (read by the release privacy gate).
_ARTIFACT_NAME = "psf_lens35_us3006249_fu0.4.npz"

# sha256 of the float32 ``kernels`` array (``kernels.tobytes()``) of each
# shipped artifact: ties the artifact to the ``scripts/build_psf`` build.
KERNELS_SHA256: dict[str, str] = {
    "35": "827a4473fa5858ac98051c88e050a797601a35a530decd9472187d5fd2b89629",
    "50": "3ee0545c80784b007fc480f722cf87cd037ccdbeb54ce228f71762f9db778308",
}


def default_tensor_path(lens: str = DEFAULT_LENS) -> Path:
    """Return the filesystem path to the wheel-shipped tensor for ``lens``.

    Raises
    ------
    ValueError
        If ``lens`` is not a key of :data:`LENS_ARTIFACTS`.
    FileNotFoundError
        If the artifact is not present on the importable
        ``summicron._artifacts`` package (a wheel built without the
        ``force-include`` entry, or a working tree missing the file).
        Message includes a "reinstall" action.
    """
    if lens not in LENS_ARTIFACTS:
        raise ValueError(
            f"unknown lens {lens!r}; choose one of {sorted(LENS_ARTIFACTS)}"
        )
    name = LENS_ARTIFACTS[lens]
    try:
        ref = resources.files("summicron._artifacts") / name
    except ModuleNotFoundError as e:
        raise FileNotFoundError(
            "summicron._artifacts package missing; reinstall isummicron "
            "(`pip install --force-reinstall isummicron`)."
        ) from e

    # ``resources.files`` returns a Traversable; for filesystem packages this
    # is a real Path. ``.is_file()`` works on every Traversable.
    if not ref.is_file():
        raise FileNotFoundError(
            f"Tensor artifact `{name}` is missing from the installed "
            f"`summicron._artifacts` package; reinstall isummicron "
            f"(`pip install --force-reinstall isummicron`)."
        )
    return Path(str(ref))

