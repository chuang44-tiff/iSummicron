"""CLI-1a proof-of-removal: dead render() fallback + vestigial kwargs are gone."""
import inspect

import numpy as np
import pytest

from summicron.render import convolve
from summicron.render.convolve import render


def test_removed_params_and_symbols():
    params = inspect.signature(render).parameters
    for s in ("defocus_map", "n_bins", "layered_gather", "r_blend_strength",
              "radial_blend_mode"):
        assert s not in params
    for s in ("_legacy_scatter_basis", "build_defocus_map", "_UNSET", "N_BINS",
              "D_KNEE"):
        assert not hasattr(convolve, s)
    assert "falling back to legacy" not in inspect.getsource(convolve)


def test_removed_call_shapes_raise_typeerror():
    img = np.zeros((8, 8, 3), np.float32)
    with pytest.raises(TypeError):
        render(img, {}, defocus_map=np.zeros((8, 8), np.float32))
    with pytest.raises(TypeError):
        render(img, {})
