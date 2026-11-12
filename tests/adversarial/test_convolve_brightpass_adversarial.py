"""P1.11 R1 bright-pass adversarial edge cases."""
from __future__ import annotations
import io
from contextlib import redirect_stdout
import numpy as np
import pytest
from summicron.render.convolve import render
from tests.fixtures.defocus import render_with_defocus

_RE = np.array([0.0, 0.15, 0.30, 0.45, 0.60, 0.80, 1.01], dtype=np.float64)
_DR = np.array([0.0, 0.33, 0.67, 1.0], dtype=np.float64)

def _blur_tensor(k=33, zero_slab=None):
    kern = np.zeros((6, 4, k, k), dtype=np.float32); c = k // 2
    kern[:, 0, c, c] = 1.0
    ys, xs = np.mgrid[-c:c + 1, -c:c + 1].astype(np.float32)
    for di in range(1, 4):
        g = np.exp(-(xs ** 2 + ys ** 2) / (2 * (1.5 + 1.5 * di) ** 2))
        kern[:, di] = (g / g.sum()).astype(np.float32)
    if zero_slab is not None: kern[zero_slab] = 0.0
    return {"kernels": kern, "radial_edges": _RE.copy(), "defocus_ratios": _DR.copy()}

def test_bright_thresh_zero_blooms_all_pixels():
    """thresh=0 → every pixel in bright path. Thin defocus (0.2) keeps the stamp
    bloom inside the [0.85, 50] total gate; result stays finite. Wide-defocus
    + thresh=0 legitimately trips the 50× runaway gate — that's by design."""
    img = np.full((64, 64, 3), 0.5, dtype=np.float32)
    dm = np.full((64, 64), 0.2, dtype=np.float32)
    _u, lin = render_with_defocus(img, _blur_tensor(), dmap=dm, hdr_source="macos", bright_thresh=0.0)
    assert np.isfinite(lin).all() and lin.sum() > 0

def test_bright_thresh_negative_coerced_to_zero():
    """Negative → coerced to 0 (documented choice). Bit-identical to thresh=0."""
    img = np.full((48, 48, 3), 0.4, dtype=np.float32)
    dm = np.full((48, 48), 0.2, dtype=np.float32); t = _blur_tensor()
    _a, ln = render_with_defocus(img, t, dmap=dm, hdr_source="macos", bright_thresh=-1.0)
    _b, lz = render_with_defocus(img, t, dmap=dm, hdr_source="macos", bright_thresh=0.0)
    assert np.max(np.abs(ln - lz)) < 1e-5

def test_bright_thresh_non_finite_raises():
    with pytest.raises(ValueError, match="finite"):
        render_with_defocus(np.zeros((32, 32, 3), dtype=np.float32), _blur_tensor(),
               dmap=np.zeros((32, 32), dtype=np.float32), bright_thresh=float("nan"))

def test_bright_pass_on_image_edge():
    """Bright pixel in (0,0) — FFT wrap must not splat onto opposite edge."""
    img = np.full((80, 80, 3), 0.1, dtype=np.float32); img[0, 0, :] = 5.0
    dm = np.full((80, 80), 0.6, dtype=np.float32)
    _u, lin = render_with_defocus(img, _blur_tensor(), dmap=dm, hdr_source="macos", bright_thresh=1.0)
    assert lin[70:, 70:, :].max() < 0.5 * lin[:10, :10, :].max()
    assert np.isfinite(lin).all()

def test_bright_energy_log_reports_nonzero_on_bright_content():
    img = np.full((64, 64, 3), 0.2, dtype=np.float32); img[30:34, 30:34, :] = 4.0
    dm = np.full((64, 64), 0.6, dtype=np.float32); buf = io.StringIO()
    with redirect_stdout(buf):
        render_with_defocus(img, _blur_tensor(), dmap=dm, hdr_source="macos", bright_thresh=1.0)
    out = buf.getvalue()
    assert "bright_energy_ratio=" in out
    for line in out.splitlines():
        if "bright_energy_ratio=" in line:
            assert float(line.split("bright_energy_ratio=")[-1].strip()) > 0.0
            return
    pytest.fail("bright-pass log line not found")

def test_kernel_max_zero_fallback():
    """Zeroed tensor slab → centred-delta fallback. 1e-10 floor on rotated.max()
    must prevent divide-by-zero in the bright path."""
    img = np.full((56, 56, 3), 0.5, dtype=np.float32); img[27:29, 27:29, :] = 3.0
    dm = np.full((56, 56), 0.6, dtype=np.float32)
    _u, lin = render_with_defocus(img, _blur_tensor(zero_slab=(2, 2)), dmap=dm,
                     hdr_source="macos", bright_thresh=1.0)
    assert np.isfinite(lin).all()
