"""CLI-1b C4 -- in-process canonical bench via ``RenderOptions.canonical_bench``.

SLOW (real HEIC + Core Image). ``render_image(canonical_bench(...))`` must
reproduce the pinned CLI md5s (G1 OFF / ON), surface non-zero diagnostics,
plumb ``center_ssim`` (Q-a: ``== focus_fidelity['ssim_band']``), and ``return_linear``
must not perturb the u8 output.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from summicron.render._bench import G1_FOCUS_XY, G1_IMAGE, G1_MD5, ON_MD5

ROOT = Path(__file__).resolve().parents[2]
IMAGE = ROOT / G1_IMAGE

pytestmark = pytest.mark.slow


def _md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def _skip_unless_ready():
    import sys

    if not IMAGE.exists():
        pytest.skip(f"canonical image missing: {IMAGE}")
    if sys.platform != "darwin":
        pytest.skip("canonical bench needs macOS Core Image")


@pytest.mark.parametrize(
    "mode,expected", [("off", G1_MD5), ("on", ON_MD5)]
)
def test_canonical_bench_api_md5_and_diagnostics(tmp_path, mode, expected):
    _skip_unless_ready()
    from summicron.render.api import RenderOptions, render_image

    out = tmp_path / "o.jpg"
    opts = RenderOptions.canonical_bench(
        IMAGE, G1_FOCUS_XY, out_path=out, occlusion_compositing=mode
    )
    res = render_image(opts)
    assert out.exists() and res.output_path == out
    assert _md5(out) == expected
    assert res.diagnostics.bright_pixels_pct > 0.0
    assert res.diagnostics.alpha_mean > 0.0
    ff = res.diagnostics.focus_fidelity
    assert ff is not None
    assert res.diagnostics.center_ssim is not None
    assert res.diagnostics.center_ssim == ff["ssim_band"]


def test_canonical_bench_api_return_linear_keeps_u8_md5(tmp_path):
    _skip_unless_ready()
    from summicron.render.api import RenderOptions, render_image

    out = tmp_path / "lin.jpg"
    opts = RenderOptions.canonical_bench(
        IMAGE, G1_FOCUS_XY, out_path=out, return_linear=True
    )
    res = render_image(opts)
    assert res.output_linear is not None
    assert _md5(out) == G1_MD5
