"""Q-a: slow G1 pin for the focus-fidelity metrics (one canonical render).

Measured 2026-11-09 on IMG_6915 @ (0.40, 0.45), canonical bench, 2048:
    ger_band=0.95272  ssim_band=0.99030  ovfrac=0.0
    ger_rgb=(0.96266, 0.94796, 0.92835)  coverage=0.45398  n_px=33587
Re-baselined for 0.5.0 (ray-traced 35 mm default tensor, approved default look;
measured on the committed artifact, tolerance 1e-4):
    ger_band=0.99842  ssim_band=0.99824  ovfrac=0.0
    ger_rgb=(0.99872, 0.99832, 0.99831)  coverage=0.03221  n_px=2383 History (older tensors):
Re-baselined 2026-11-09 by Q-b-flip (exact d0 delta default;
the old values above are the pre-flip numbers; coverage / n_px unchanged):
    ger_band=0.98002  ssim_band=0.99796  ovfrac=0.0
    ger_rgb=(0.99872, 0.99832, 0.99831)  coverage=0.45398  n_px=33587
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from summicron.render._bench import G1_FOCUS_XY, G1_IMAGE, G1_MD5

ROOT = Path(__file__).resolve().parents[2]
IMAGE = ROOT / G1_IMAGE

pytestmark = pytest.mark.slow


def test_q1_metrics_pinned_on_g1(tmp_path):
    pytest.importorskip("skimage")
    if sys_platform_not_darwin() or not IMAGE.exists():
        pytest.skip("needs macOS + canonical HEIC")
    from summicron.render.api import RenderOptions, render_image

    out = tmp_path / "g1.jpg"
    opts = RenderOptions.canonical_bench(
        IMAGE, G1_FOCUS_XY, out_path=out, return_linear=True
    )
    sink: dict = {}
    res = render_image(opts, _eval_sink=sink, on_progress=lambda m: None)

    assert hashlib.md5(out.read_bytes()).hexdigest() == G1_MD5
    a1 = sink["alpha"] == 1.0
    assert a1.any()
    diff = np.abs(res.output_linear - sink["ref_lin"]).max(axis=2)
    assert float(diff[a1].max()) == 0.0

    ff = res.diagnostics.focus_fidelity
    assert ff is not None
    assert res.diagnostics.center_ssim == ff["ssim_band"]
    tol = 1e-4
    assert ff["ger_band"] == pytest.approx(0.99842, abs=tol)
    assert ff["ssim_band"] == pytest.approx(0.99824, abs=tol)
    assert ff["ovfrac"] == pytest.approx(0.0, abs=tol)
    for got, want in zip(ff["ger_rgb"], (0.99872, 0.99832, 0.99831)):
        assert got == pytest.approx(want, abs=tol)
    assert ff["coverage"] == pytest.approx(0.03221, abs=tol)
    assert ff["n_px"] == 2383


def sys_platform_not_darwin() -> bool:
    import sys
    return sys.platform != "darwin"
