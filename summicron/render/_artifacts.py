"""Wheel-shipped tensor artifact resolver (P1.audit Item 7).

The canonical CodeV render tensor (`codev_render_v1.npz`) is bundled
inside the wheel at ``summicron/_artifacts/`` so installed users get a
working render pipeline without separately downloading the 17.7 MB tensor
from the dev tree.

`default_tensor_path()` resolves to that bundled file via
``importlib.resources``; missing artifact (mis-built wheel) raises a
typed :class:`FileNotFoundError` with an actionable reinstall message
rather than ``importlib.resources.MissingResource``.
"""
from __future__ import annotations

from importlib import resources
from pathlib import Path

_ARTIFACT_NAME = "codev_render_v1.npz"


def default_tensor_path() -> Path:
    """Return the filesystem path to the wheel-shipped canonical tensor.

    Raises
    ------
    FileNotFoundError
        If the artifact is not present on the importable
        ``summicron._artifacts`` package — typically a wheel built without
        the ``[tool.hatch.build.targets.wheel.force-include]`` entry, or
        an editable-install missing the file from the working tree.
        Message includes a "reinstall" action.
    """
    try:
        ref = resources.files("summicron._artifacts") / _ARTIFACT_NAME
    except ModuleNotFoundError as e:
        raise FileNotFoundError(
            "summicron._artifacts package missing; reinstall isummicron "
            "(`pip install --force-reinstall isummicron`)."
        ) from e

    # ``resources.files`` returns a Traversable; for filesystem packages
    # this is a real Path, but for zipped wheels it may be a MultiplexedPath
    # or a temp-file-backed wrapper. ``.is_file()`` works on both; convert to
    # a real path via ``.as_posix()`` only after confirming presence.
    if not ref.is_file():
        raise FileNotFoundError(
            f"Canonical tensor artifact `{_ARTIFACT_NAME}` is missing from "
            f"the installed `summicron._artifacts` package; reinstall "
            f"isummicron (`pip install --force-reinstall isummicron`)."
        )
    return Path(str(ref))
