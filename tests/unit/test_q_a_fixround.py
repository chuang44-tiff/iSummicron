"""Q-a fix round: diagnostics hashability + Q1 failure isolation. FAST."""
from __future__ import annotations

from pathlib import Path

import pytest

from summicron.render._artifacts import default_tensor_path
from summicron.render.api import RenderDiagnostics, RenderOptions, render_image

REPO = Path(__file__).resolve().parents[2]
IMG = REPO / "iphoneImage" / "IMG_6915.HEIC"


def test_diagnostics_hashable_with_focus_fidelity():
    d = RenderDiagnostics(center_ssim=0.9, focus_fidelity={"ssim_band": 0.9})
    assert isinstance(hash(d), int)
    # compare=False: the dict does not participate in equality.
    assert d == RenderDiagnostics(center_ssim=0.9, focus_fidelity={"x": 1.0})


@pytest.mark.skipif(not IMG.exists(), reason="canonical image missing")
def test_quality_exception_does_not_fail_render(tmp_path, monkeypatch):
    from summicron.render import _quality

    def boom(*a, **k):
        raise RuntimeError("synthetic Q1 failure")

    monkeypatch.setattr(_quality, "focus_fidelity", boom)
    opts = RenderOptions(
        image_path=IMG, tensor_path=default_tensor_path(),
        out_path=tmp_path / "o.jpg", depth_source="lidar",
        focus_xy=(0.40, 0.45), depth_scale=10.0, downres=256, hdr_source="none",
    )
    with pytest.warns(RuntimeWarning, match="Q1 focus-fidelity"):
        res = render_image(opts)
    assert res.output_u8.shape[2] == 3
    assert res.diagnostics.focus_fidelity is None
    assert res.diagnostics.center_ssim is None
    hash(res.diagnostics)
