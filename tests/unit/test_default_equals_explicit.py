"""0.5.0 SLOW: a flagless default render equals the explicit-flag render.

The default look (physical gain, l1 bright stamp, diopter mapping under
LiDAR) must be byte-identical to spelling every flag out against the same
shipped artifact. Compared on IMG_6915 (canonical flags) and on the kitchen
image (flagless, HDR auto), for both lenses: JPEG md5 and ``output_linear``
max abs difference 0.
"""
from __future__ import annotations

import hashlib
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from summicron.render._artifacts import default_tensor_path
from summicron.render._bench import G1_FOCUS_XY, G1_IMAGE

ROOT = Path(__file__).resolve().parents[2]
KITCHEN = "iphoneImage/Demo_input.HEIC"
KITCHEN_XY = (0.40, 0.39)

pytestmark = pytest.mark.slow


def _scene(name):
    from summicron.render.api import RenderOptions

    if name == "g1":
        return RenderOptions.canonical_bench(ROOT / G1_IMAGE, G1_FOCUS_XY)
    return RenderOptions(image_path=ROOT / KITCHEN, focus_xy=KITCHEN_XY)


@pytest.mark.parametrize("lens", ["35", "50"])
@pytest.mark.parametrize("scene", ["g1", "kitchen"])
def test_default_equals_explicit(tmp_path, scene, lens):
    if sys.platform != "darwin":
        pytest.skip("needs macOS (HDR loader)")
    img = ROOT / (G1_IMAGE if scene == "g1" else KITCHEN)
    if not img.exists():
        pytest.skip("sample images not shipped")
    from summicron.render.api import render_image

    base = _scene(scene)
    out_d, out_e = tmp_path / "d.jpg", tmp_path / "e.jpg"
    default = replace(base, lens=lens, out_path=out_d, return_linear=True)
    explicit = replace(
        base, lens=None, tensor_path=default_tensor_path(lens),
        depth_mapping="diopter", depth_gain=1.0,
        bright_stamp="l1", out_path=out_e, return_linear=True,
    )
    r_d = render_image(default, on_progress=lambda m: None)
    r_e = render_image(explicit, on_progress=lambda m: None)
    assert hashlib.md5(out_d.read_bytes()).hexdigest() == \
        hashlib.md5(out_e.read_bytes()).hexdigest()
    assert float(np.abs(r_d.output_linear - r_e.output_linear).max()) == 0.0
