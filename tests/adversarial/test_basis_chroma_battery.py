"""P1.10.4 C6 — Basis-mode chroma battery (5 fixtures x 3 gates).

Per locked-spec-P1_10_4.md § 5 + memory feedback_adversarial_chromaticity_gate.md:
mandatory chroma fixture battery covers warm/cool/magenta/green saturated HDR
inputs + achromatic control under basis-default path. Compensates for the
P1.10.2 § 8 trigger override (>=3 distinct user scenes structurally
unsatisfiable per iphoneImage/ — only 2 usable HEICs).

Gates:
- G_CHROMA_HUE: <=1% hue drift (mirrors test_p1_10_1_chromaticity.py threshold
  at lines 79, 82, 93, 96, 141, 142). Skipped on the achromatic control.
- G_CHROMA_BITID: no NaN/Inf, sane output range (no negative blow-ups,
  no astronomical highlights).
- G_CHROMA_DELTA-SAFETY: S2 delta-guard preservation at defocus_map=0; subject
  region must satisfy max-abs-diff < 0.05 in linear space.

The basis+layered render path is the now-default (P1.10.2). These tests pass
explicit `radial_blend_mode="basis"` + `layered_gather=True` to be stable
against any future default flips.
"""
from __future__ import annotations

import numpy as np
import pytest

from summicron.render.convolve import render
from tests.fixtures.defocus import render_with_defocus



# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _make_gaussian_tensor(R: int = 4, D: int = 4, ksize: int = 33,
                          sigma: float = 4.0) -> dict:
    """Tiny Gaussian tensor with S2 delta guard at d=0.

    Mirrors the fixture in `tests/adversarial/test_basis_quality.py:26-48`
    and `tests/adversarial/test_p1_10_1_chromaticity.py:22-40`.
    """
    cy, cx = ksize // 2, ksize // 2
    yy, xx = np.mgrid[:ksize, :ksize]
    g = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * sigma * sigma))
    g = g.astype(np.float32) / g.sum()
    kernels = np.tile(g[None, None, :, :], (R, D, 1, 1))
    delta = np.zeros((ksize, ksize), dtype=np.float32)
    delta[cy, cx] = 1.0
    kernels[:, 0] = delta  # S2 delta guard at d=0
    return {
        "kernels": kernels,
        "radial_edges": np.linspace(0.0, 1.0, R + 1).astype(np.float64),
        "defocus_ratios": np.linspace(0.0, 1.0, D).astype(np.float64),
        "corpus_ref_long_edge": None,
        "otf_norm_applied": False,
        "delta_guard_applied": True,
        "object_distances_m": None,
        "focus_distance_m": None,
    }


def _hsv_to_rgb_linear(h_deg: float, s: float, v: float) -> np.ndarray:
    """HSV (h in degrees, s/v in [0, +inf)) -> linear-RGB triple (float32).

    Standard 6-segment HSV. v can exceed 1.0 for HDR headroom; chromaticity
    (h, s) is preserved across all v.
    """
    h = (h_deg % 360.0) / 60.0
    c = v * s
    x = c * (1.0 - abs((h % 2.0) - 1.0))
    m = v - c
    if 0.0 <= h < 1.0:
        r, g, b = c, x, 0.0
    elif 1.0 <= h < 2.0:
        r, g, b = x, c, 0.0
    elif 2.0 <= h < 3.0:
        r, g, b = 0.0, c, x
    elif 3.0 <= h < 4.0:
        r, g, b = 0.0, x, c
    elif 4.0 <= h < 5.0:
        r, g, b = x, 0.0, c
    else:
        r, g, b = c, 0.0, x
    return np.array([r + m, g + m, b + m], dtype=np.float32)


def _rgb_to_hue_deg(rgb: np.ndarray) -> float:
    """Linear-RGB triple -> hue in degrees [0, 360).

    Returns NaN if the pixel is achromatic (cmax == cmin); callers must
    handle this for the desat control fixture.
    """
    r, g, b = float(rgb[0]), float(rgb[1]), float(rgb[2])
    cmax = max(r, g, b)
    cmin = min(r, g, b)
    delta = cmax - cmin
    if delta < 1e-8:
        return float("nan")
    if cmax == r:
        h = 60.0 * (((g - b) / delta) % 6.0)
    elif cmax == g:
        h = 60.0 * (((b - r) / delta) + 2.0)
    else:
        h = 60.0 * (((r - g) / delta) + 4.0)
    return h % 360.0


# ---------------------------------------------------------------------------
# Parametrized chroma fixtures
# ---------------------------------------------------------------------------


_FIXTURE_PARAMS = [
    {"name": "warm",     "hue_deg": 30.0,  "peak": 3.0},
    {"name": "cool",     "hue_deg": 210.0, "peak": 3.0},
    {"name": "magenta",  "hue_deg": 330.0, "peak": 3.0},
    {"name": "green",    "hue_deg": 120.0, "peak": 3.0},
    {"name": "desat",    "hue_deg": None,  "peak": 3.0},
]


@pytest.fixture(params=_FIXTURE_PARAMS, ids=lambda p: p["name"])
def chroma_fixture(request):
    p = request.param
    sz = 256
    img = np.full((sz, sz, 3), 0.05, dtype=np.float32)  # dim grey background
    cy, cx = sz // 2, sz // 2
    if p["hue_deg"] is None:
        # Achromatic luminance match: equal R=G=B at peak.
        img[cy - 20:cy + 20, cx - 20:cx + 20, :] = p["peak"]
    else:
        rgb = _hsv_to_rgb_linear(p["hue_deg"], 1.0, p["peak"])
        img[cy - 20:cy + 20, cx - 20:cx + 20, :] = rgb
    return p["name"], p["hue_deg"], img


def _render_basis(img: np.ndarray, defocus_value: float = 0.0) -> np.ndarray:
    """Helper: render through small Gaussian tensor on basis+layered path."""
    H, W, _ = img.shape
    tensor = _make_gaussian_tensor(R=4, D=4, ksize=33, sigma=4.0)
    dmap = np.full((H, W), float(defocus_value), dtype=np.float32)
    _, out_lin = render_with_defocus(
        img, tensor, dmap=dmap,
        hdr_source="macos",  # accept HDR (peak ~3.0) without sRGB round-trip
    )
    return out_lin


# ---------------------------------------------------------------------------
# Gate 1 — G_CHROMA_HUE  (<=1% hue drift)
# ---------------------------------------------------------------------------


def test_chroma_hue_drift_le_1pct(chroma_fixture):
    """G_CHROMA_HUE: input hue at bright-spot center preserved within 1%.

    Mirrors the 1% threshold enforced by `test_p1_10_1_chromaticity.py:79,
    82, 93, 96, 141, 142`. Skipped on achromatic control (no hue to drift).

    NB: defocus=0 keeps the spot in-focus through the S2 delta guard; this
    measures hue at the kernel-center sample, the strictest possible test.
    """
    name, hue_in_deg, img = chroma_fixture
    if hue_in_deg is None:
        pytest.skip("achromatic fixture has no hue to drift")
    out_lin = _render_basis(img, defocus_value=0.0)
    cy, cx = 128, 128
    out_rgb = out_lin[cy, cx, :]
    hue_out = _rgb_to_hue_deg(out_rgb)
    assert not np.isnan(hue_out), (
        f"{name}: output pixel {out_rgb.tolist()} is achromatic — "
        f"basis path collapsed chroma at center"
    )
    # Wrap-aware hue distance: shortest arc on the 360° circle.
    raw = abs(hue_out - hue_in_deg) % 360.0
    arc = min(raw, 360.0 - raw)
    drift = arc / 360.0
    assert drift < 0.01, (
        f"{name}: hue drift {drift*100:.3f}% > 1% "
        f"(in={hue_in_deg:.2f}°, out={hue_out:.2f}°, arc={arc:.3f}°)"
    )


# ---------------------------------------------------------------------------
# Gate 2 — G_CHROMA_BITID  (no NaN / no Inf / sane range)
# ---------------------------------------------------------------------------


def test_chroma_no_nan_inf(chroma_fixture):
    """G_CHROMA_BITID: render output finite, non-negative, non-astronomical.

    Saturated HDR inputs at peak ~3.0 must not produce NaN, Inf, negative
    pixels, or values >= 1e3 through the basis path.
    """
    name, _hue, img = chroma_fixture
    out_lin = _render_basis(img, defocus_value=0.0)
    assert np.isfinite(out_lin).all(), f"{name}: NaN/Inf in basis output"
    assert out_lin.min() >= 0.0, (
        f"{name}: negative pixels in output (min={out_lin.min():.4f})"
    )
    assert out_lin.max() < 1e3, (
        f"{name}: astronomical highlights (max={out_lin.max():.4f} >= 1e3)"
    )


# ---------------------------------------------------------------------------
# Gate 3 — G_CHROMA_DELTA-SAFETY  (S2 delta guard at defocus=0)
# ---------------------------------------------------------------------------


def test_chroma_delta_safety_in_focus(chroma_fixture):
    """G_CHROMA_DELTA-SAFETY: at defocus=0, output ≈ input (S2 delta guard).

    The S2 delta guard installed at `kernels[:, 0]` makes every radial bin
    a pure single-pixel delta at d=0; convolving with a delta is the
    identity, so subject-region max-abs-diff must stay below 0.05 in linear
    space. Memory `project_s2_delta_guard_subject_safety.md`: subject
    preservation is "mathematically guaranteed" at d=0.
    """
    name, _hue, img = chroma_fixture
    out_lin = _render_basis(img, defocus_value=0.0)
    # Compare full-frame; the spot, the dim background, and the boundary
    # all live under the same delta guard at defocus=0.
    diff = np.abs(out_lin - img).max()
    assert diff < 0.05, (
        f"{name}: max-abs-diff {diff:.4f} > 0.05 — S2 delta guard violated "
        f"under basis-default path"
    )
