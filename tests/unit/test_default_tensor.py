"""Wheel-shipped tensor resolver (P1.audit Item 7).

Asserts ``default_tensor_path()`` returns the bundled artifact and the
canonical render tensor loads with R=12, D=11. Missing-artifact case is
simulated by monkeypatching ``importlib.resources.files`` to return a
path that does not exist.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from summicron.render._artifacts import default_tensor_path
from summicron.render.convolve import load_tensor


def test_default_tensor_path_resolves_to_existing_file():
    """The bundled tensor is on-disk in the working tree (and in the wheel)."""
    p = default_tensor_path()
    assert isinstance(p, Path)
    assert p.is_file(), f"missing canonical tensor at {p}"
    assert p.name == "codev_render_v1.npz"


def test_default_tensor_loads_as_R12_D11():
    """Round-trip through load_tensor: the shipped artifact is the canonical
    R=12, D=11 CodeV tensor."""
    t = load_tensor(default_tensor_path())
    assert t["R"] == 12
    assert t["D"] == 11
    assert t["delta_guard_applied"] is True


def test_missing_artifact_raises_actionable_error(monkeypatch, tmp_path):
    """Monkeypatching the resolver to return a non-existent path must
    surface a typed FileNotFoundError that says 'reinstall'."""
    from summicron.render import _artifacts as art_mod

    class FakeRef:
        def is_file(self):
            return False
        def __str__(self):
            return str(tmp_path / "nope.npz")

    def fake_files(_pkg):
        class _Pkg:
            def __truediv__(self, _name):
                return FakeRef()
        return _Pkg()

    monkeypatch.setattr(art_mod.resources, "files", fake_files)
    with pytest.raises(FileNotFoundError, match="reinstall"):
        art_mod.default_tensor_path()
