"""Run 5 gate tests needing a real HEIC + Core Image (SLOW): G4(iii), input-lut T1 smoke.

G4(iv) (ys.max() < 0.9 branch) is a FAST test below.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from summicron.io.tone_lut import LUMA_W, auto_bright_threshold

ROOT = Path(__file__).resolve().parents[2]
DEMO = ROOT / "iphoneImage" / "Demo_input.HEIC"
DEMO_XY = (0.40, 0.39)


def test_g4iv_ys_max_below_09_threshold_rule_and_bright_pixels_reported():
    """Spec 4.3: LUT never reaching 0.9 -> thr = 1.1 * xs[-1].

    The bright-pass is NOT guaranteed disabled (skeptic: bright_pixels_pct == 0 is
    impossible in general), so the fraction is REPORTED, not asserted zero.
    """
    xs = np.array([0.0, 0.5, 1.0, 3.0])
    ys = np.array([0.0, 0.3, 0.6, 0.8])
    thr, h = auto_bright_threshold(xs, ys)
    assert h == pytest.approx(3.0)
    assert thr == pytest.approx(1.1 * xs[-1])


def _need_demo():
    if not DEMO.exists():
        pytest.skip(f"missing {DEMO}")
    if sys.platform != "darwin":
        pytest.skip("needs macOS Core Image")


@pytest.mark.slow
def test_g4iii_explicit_depth_gain_equals_auto_bit_identical(tmp_path, capsys):
    _need_demo()
    from unittest.mock import patch

    import summicron.depth.diopter as dio
    from summicron.render.api import RenderOptions, render_image

    seen: list[float] = []
    real = dio.auto_gain

    def spy(*a, **k):
        g = real(*a, **k)
        seen.append(float(g))
        return g

    base = dict(depth_guided="on", depth_mapping="diopter", return_linear=True, downres=1024)
    with patch.object(dio, "auto_gain", spy):
        a = render_image(RenderOptions.canonical_bench(
            DEMO, DEMO_XY, out_path=tmp_path / "a.jpg", depth_gain="auto", **base))
    assert len(seen) == 1
    b = render_image(RenderOptions.canonical_bench(
        DEMO, DEMO_XY, out_path=tmp_path / "b.jpg", depth_gain=seen[0], **base))
    assert np.array_equal(a.output_u8, b.output_u8)
    assert np.array_equal(a.output_linear, b.output_linear)
    print(f"G4(iii): auto gain {seen[0]!r}; bright_pixels_pct {a.diagnostics.bright_pixels_pct:.4f}")


@pytest.mark.slow
def test_input_lut_smoke_t1_demo(tmp_path):
    """T1: mean over p1..p99 of |P_out - P_in| (display luma) <= 0.02 on Demo, canonical argv."""
    _need_demo()
    from summicron.render.api import RenderOptions, render_image
    from summicron.render.convolve import downres_array, load_image

    opts = RenderOptions.canonical_bench(
        DEMO, DEMO_XY, out_path=tmp_path / "t.jpg", tonemap="input-lut",
        depth_guided="on", depth_mapping="diopter",
    )
    res = render_image(opts)
    out = res.output_u8.astype(np.float32) / 255.0
    sdr = downres_array(load_image(str(DEMO)), 2048).astype(np.float32)
    if sdr.max() > 1.5:
        sdr /= 255.0
    q = np.arange(1, 100)
    err = np.abs(np.percentile(out @ LUMA_W, q) - np.percentile(sdr @ LUMA_W, q)).mean()
    assert out.shape == sdr.shape
    assert err <= 0.02, err
    # R/B display ratio within 1.5% of the input (rev 3 chroma gate)
    rb = lambda x: x[..., 0].mean() / x[..., 2].mean()  # noqa: E731
    assert abs(rb(out) / rb(sdr) - 1) <= 0.015
