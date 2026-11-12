"""P1.11 R1 bright-pass — Adversarial subagent hunt round (R1-ADV).

Tests here are Adversarial-subagent-written (separate from Coder's
``test_convolve_brightpass_adversarial.py``) so the next fix round can tell
them apart at a glance.

xfail(strict=True) tests demonstrate real bugs the Coder should fix. Plain
passing tests add coverage on axes the Coder did not probe.
"""
from __future__ import annotations

import io
from contextlib import redirect_stdout

import numpy as np
import pytest

from summicron.render.convolve import render
from tests.fixtures.defocus import render_with_defocus



_RE = np.array([0.0, 0.15, 0.30, 0.45, 0.60, 0.80, 1.01], dtype=np.float64)
_DR = np.array([0.0, 0.33, 0.67, 1.0], dtype=np.float64)


def _blur_tensor(k: int = 33) -> dict:
    """Tensor with delta at d=0 and expanding gaussians at d>0."""
    kern = np.zeros((6, 4, k, k), dtype=np.float32)
    c = k // 2
    kern[:, 0, c, c] = 1.0
    ys, xs = np.mgrid[-c:c + 1, -c:c + 1].astype(np.float32)
    for di in range(1, 4):
        g = np.exp(-(xs ** 2 + ys ** 2) / (2 * (1.5 + 1.5 * di) ** 2))
        kern[:, di] = (g / g.sum()).astype(np.float32)
    return {
        "kernels": kern,
        "radial_edges": _RE.copy(),
        "defocus_ratios": _DR.copy(),
    }


# ---------------------------------------------------------------------------
# Pass-through coverage (Coder did NOT cover these — add unguarded)
# ---------------------------------------------------------------------------

def test_bright_thresh_above_max_is_bit_identical():
    """Coder's G5 tests bit-identity at thresh=1e9 only.

    Real user guarantee is: if ``T > max(img)`` then
    ``clip(img, 0, T) == img`` and ``relu(img - T) == 0`` exactly,
    so output SHOULD be bit-identical to disabled (1e9) for ANY threshold
    above the image max. This probes a mid-range threshold on LDR input.
    Regression here would mean a signal is leaking through the bright path
    even when the bright accumulator should be all-zeros.
    """
    img = np.random.default_rng(42).uniform(0, 0.9, (64, 64, 3)).astype(np.float32)
    dm = np.full((64, 64), 0.6, dtype=np.float32)
    t = _blur_tensor()
    _a, la = render_with_defocus(img, t, dmap=dm, bright_thresh=1e9)
    _b, lb = render_with_defocus(img, t, dmap=dm, bright_thresh=100.0)
    assert np.max(np.abs(la - lb)) < 1e-6, (
        f"thresh=100 (> max=0.9) should equal thresh=1e9, "
        f"max diff = {np.max(np.abs(la - lb))}"
    )


def test_bright_thresh_int_dtype_accepted():
    """CLI --bright-pass-thresh is float, but the kwarg spec is loose.

    Passing Python int directly to render() should work (no dtype promotion
    surprise). Coder only tested float literals.
    """
    img = np.random.default_rng(3).uniform(0, 0.9, (48, 48, 3)).astype(np.float32)
    dm = np.full((48, 48), 0.6, dtype=np.float32)
    t = _blur_tensor()
    # int 1 vs float 1.0 must be numerically identical
    _a, la = render_with_defocus(img, t, dmap=dm, bright_thresh=1)
    _b, lb = render_with_defocus(img, t, dmap=dm, bright_thresh=1.0)
    assert np.max(np.abs(la - lb)) < 1e-7, (
        f"int vs float thresh diverged: {np.max(np.abs(la - lb))}"
    )


def test_bright_pass_log_printed_even_when_disabled():
    """Log line for grepping/monitoring — should be printed unconditionally."""
    img = np.full((48, 48, 3), 0.5, dtype=np.float32)
    dm = np.full((48, 48), 0.6, dtype=np.float32)
    t = _blur_tensor()
    buf = io.StringIO()
    with redirect_stdout(buf):
        render_with_defocus(img, t, dmap=dm, bright_thresh=1e9)
    assert "bright-pass:" in buf.getvalue(), (
        "Log line missing with thresh=1e9 (disabled) — breaks grep/monitor flows"
    )
    assert "bright_pixels_pct=0.000%" in buf.getvalue()
    assert "bright_energy_ratio=0.0000" in buf.getvalue()


def test_all_delta_slabs_bright_pass_is_identity():
    """When dmap=0 every slab trips is_delta → bright path skipped everywhere.

    Output must equal input EXACTLY (alpha=1 at dmap=0 blends back to img_lin)
    regardless of thresh. This is a tighter check than Coder's G6 — pixel-exact.
    """
    img = np.full((64, 64, 3), 0.2, dtype=np.float32)
    img[30:34, 30:34, :] = 5.0  # HDR-scale bright blob
    dm = np.zeros((64, 64), dtype=np.float32)  # all in focus
    t = _blur_tensor()
    _a, la = render_with_defocus(img, t, dmap=dm, hdr_source="macos", bright_thresh=1e9)
    _b, lb = render_with_defocus(img, t, dmap=dm, hdr_source="macos", bright_thresh=1.0)
    # Bit-identical because α=1 uses img_lin on all pixels
    assert np.array_equal(la, lb), (
        f"all-delta bright-pass should be pure passthrough; "
        f"max diff = {np.max(np.abs(la - lb))}"
    )
    # And both should equal the input (linearized — already linear since hdr_source='macos')
    assert np.max(np.abs(la - img)) < 1e-6, (
        f"all-delta scene: output should equal input; "
        f"max diff = {np.max(np.abs(la - img))}"
    )


def test_bright_pass_wedge_boundary_symmetry():
    """A single bright pixel on a wedge boundary (theta=0, due right of center).

    Bloom must remain left/right symmetric around the bright pixel
    within float tolerance. Wedge-boundary double-counting would show up as
    asymmetry (one side blooms stronger than the other).
    """
    H, W = 96, 96
    img = np.full((H, W, 3), 0.1, dtype=np.float32)
    cy, cx = (H - 1) / 2, (W - 1) / 2
    # Due-right of optical center: theta ≈ 0, on wedge 3|4 boundary
    img[int(cy), int(cx) + 20, :] = 8.0
    dm = np.full((H, W), 0.6, dtype=np.float32)
    t = _blur_tensor()
    _u, lin = render_with_defocus(img, t, dmap=dm, hdr_source="macos", bright_thresh=1.0)
    row = int(cy)
    col = int(cx) + 20
    for dx in (1, 2, 3, 5, 8):
        lft = float(lin[row, col - dx, 0])
        rgt = float(lin[row, col + dx, 0])
        assert abs(lft - rgt) < 1e-3 * max(lft, rgt, 1e-6), (
            f"wedge-boundary asymmetry at dx={dx}: left={lft}, right={rgt}"
        )


# ---------------------------------------------------------------------------
# Suspected real bug: diffuse energy gate denominator uses full input sum
# ---------------------------------------------------------------------------

def test_extreme_hdr_bright_pixel_passes_diffuse_gate():
    """Pathological but legitimate HDR: one pixel at 500x the background.

    Fireworks / direct sun / specular-on-chrome produces this. The bright path
    should carry the bloom, the diffuse path should conserve energy within
    [0.85, 1.15] *of the clipped input*. Coder's current gate divides by
    ``img_lin.sum()`` (FULL input), which drops diffuse_ratio below 0.85 as
    soon as the bright pixel dominates the mass.

    Measured on 80x80, bg=0.1, bright=500: diffuse_ratio ≈ 0.56, tripping
    Gate FAIL. This blocks legitimate HDR content through the render().
    """
    img = np.full((80, 80, 3), 0.1, dtype=np.float32)
    img[40, 40, :] = 500.0  # specular highlight, 5000× background
    dm = np.full((80, 80), 0.6, dtype=np.float32)
    t = _blur_tensor()
    # Must not raise AssertionError from the gate
    _u, lin = render_with_defocus(img, t, dmap=dm, hdr_source="macos", bright_thresh=1.0)
    assert np.isfinite(lin).all()


# ---------------------------------------------------------------------------
# Secondary passing coverage
# ---------------------------------------------------------------------------

def test_bright_pass_finite_on_hdr_headroom_range():
    """Sweep thresh across 0.1 .. 10 on moderate HDR content. No NaN/Inf
    anywhere, and total energy ratio monotonic-ish with decreasing thresh.

    Coder's tests only probed single thresholds. Sweeping catches bound-regime
    misbehavior (e.g. conditional that only fires at specific thresh values).
    """
    img = np.full((64, 64, 3), 0.3, dtype=np.float32)
    img[30:34, 30:34, :] = 4.0  # 4x peak, plausible HDR
    dm = np.full((64, 64), 0.5, dtype=np.float32)
    t = _blur_tensor()
    ratios = []
    for th in (0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 1e9):
        _u, lin = render_with_defocus(img, t, dmap=dm, hdr_source="macos",
                         bright_thresh=th)
        assert np.isfinite(lin).all(), f"non-finite at thresh={th}"
        ratios.append(float(lin.sum() / img.sum()))
    # Lower threshold → more bright path engagement → larger total energy
    # (peak-norm kernel stamps onto more pixels). ratio at 1e9 ≈ 1.0 baseline.
    assert ratios[-1] < ratios[0] + 1e-3, (
        f"expected monotonic total ratio as thresh drops, got {ratios}"
    )


def test_bright_pass_solid_1d_image_no_explosion():
    """Flat gray image well below the threshold. Identity to disabled, since
    bright is empty.

    R1-PATCH: use lum well below T-K to isolate diffuse behavior; lum == thresh
    now triggers smoothstep midpoint per soft-knee semantics, so the old
    `lum = thresh` fixture no longer tests the intended "flat image doesn't
    cause bloom explosion" invariant. Anchors to `lum << thresh` instead.
    """
    img = np.full((48, 48, 3), 0.01, dtype=np.float32)
    dm = np.full((48, 48), 0.6, dtype=np.float32)
    t = _blur_tensor()
    _a, la = render_with_defocus(img, t, dmap=dm, bright_thresh=0.5)
    _b, lb = render_with_defocus(img, t, dmap=dm, bright_thresh=1e9)
    # img max is 0.01 << T-K = 0.1; bright_weight = 0 exactly; result should
    # match disabled within float noise.
    assert np.max(np.abs(la - lb)) < 1e-5, (
        f"flat image well below thresh drifted: max diff = {np.max(np.abs(la - lb))}"
    )


def test_negative_float_input_with_bright_pass():
    """HDR Core Image LUT can momentarily produce tiny negative floats even
    after np.maximum(..., 0) clip. Verify bright-pass handles that without
    leaking into either path."""
    img = np.full((48, 48, 3), 0.3, dtype=np.float32)
    # Inject small negative — simulates Core Image LUT artifact
    img[10:12, 10:12, :] = -1e-4
    img[30, 30, :] = 4.0  # bright blob
    dm = np.full((48, 48), 0.6, dtype=np.float32)
    t = _blur_tensor()
    _u, lin = render_with_defocus(img, t, dmap=dm, hdr_source="macos", bright_thresh=1.0)
    assert np.isfinite(lin).all()


def test_bright_pass_preserves_chromaticity_on_warm_highlight():
    """Regression gate for fix-round 2 chromatic-halo bug (IMG_6915 umbrella).

    Per-channel threshold splats the R channel alone when R>T but G,B<T,
    producing a red bloom halo around warm highlights. Luminance gate +
    full-RGB split preserves chromaticity: the bloom's R:G:B ratio should
    match the source pixel's R:G:B ratio within ~10%.

    Fixture: warm central region (R=3.0, G=2.0, B=1.5, lum≈2.24) on a warm
    gray background matching the same hue (R=0.3, G=0.2, B=0.15). The gray
    bg stays below lum-threshold (lum≈0.22 < 1.0) so its contribution rides
    through the diffuse path. Bloom = rendered - rendered(bg_only): the
    bright pixels' excess over bg is (2.7, 1.8, 1.35) with the SAME 3:2:1.5
    ratio, and subtracting the bg-only render cancels the diffuse offset.
    Under the OLD per-channel thresholding (T=1.0), only R exceeds T →
    bright splat is (2.0, 0.0, 0.0), ratio R:G:B ≈ 2:0:0 → red halo.
    """
    H, W = 64, 64
    # Warm gray bg (same hue as bright region so chromaticity is clean)
    bg_r, bg_g, bg_b = 0.3, 0.2, 0.15
    img = np.empty((H, W, 3), dtype=np.float32)
    img[..., 0] = bg_r
    img[..., 1] = bg_g
    img[..., 2] = bg_b
    # 5x5 warm bright block in the center (big enough that bright energy
    # dominates the ring sum and the overall energy ratio stays sane).
    cy, cx = H // 2, W // 2
    img[cy - 2:cy + 3, cx - 2:cx + 3, 0] = 3.0
    img[cy - 2:cy + 3, cx - 2:cx + 3, 1] = 2.0
    img[cy - 2:cy + 3, cx - 2:cx + 3, 2] = 1.5
    dm = np.full((H, W), 0.6, dtype=np.float32)
    t = _blur_tensor()
    _u, lin = render_with_defocus(img, t, dmap=dm, hdr_source="macos", bright_thresh=1.0)
    # Baseline: same image without the bright block (pure warm-gray bg).
    baseline = np.empty_like(img)
    baseline[..., 0] = bg_r
    baseline[..., 1] = bg_g
    baseline[..., 2] = bg_b
    _u2, base_lin = render_with_defocus(baseline, t, dmap=dm, hdr_source="macos",
                           bright_thresh=1.0)
    bloom = lin - base_lin
    # Ring around the 5x5 bright block (radial distance 6-12 px from center).
    yy, xx = np.mgrid[:H, :W]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    ring = (r >= 6.0) & (r <= 12.0)
    bloom_r = float(bloom[..., 0][ring].sum())
    bloom_g = float(bloom[..., 1][ring].sum())
    bloom_b = float(bloom[..., 2][ring].sum())
    # Source excess over bg = (2.7, 1.8, 1.35) → ratios g/r=2/3, b/r=1/2.
    # Subtracting bg-only render cancels the diffuse offset from bg.
    ratio_gr = bloom_g / bloom_r
    ratio_br = bloom_b / bloom_r
    assert abs(ratio_gr - (2.0 / 3.0)) < 0.10 * (2.0 / 3.0), (
        f"Chromatic halo: bloom G/R = {ratio_gr:.3f}, expected {2.0 / 3.0:.3f} "
        f"(within 10%). R={bloom_r:.3f}, G={bloom_g:.3f}, B={bloom_b:.3f}"
    )
    assert abs(ratio_br - (1.5 / 3.0)) < 0.10 * (1.5 / 3.0), (
        f"Chromatic halo: bloom B/R = {ratio_br:.3f}, expected {1.5 / 3.0:.3f} "
        f"(within 10%). R={bloom_r:.3f}, G={bloom_g:.3f}, B={bloom_b:.3f}"
    )


def test_bright_pass_preserves_chromaticity_on_cool_highlight():
    """Companion to the warm-highlight chromaticity regression (GC3).

    Symmetric probe: cool bright highlight where B dominates. Luminance =
    0.2126*0.5 + 0.7152*1.5 + 0.0722*3.0 ≈ 1.40 > thresh=1.0, so the gate
    admits the full (R=0.5, G=1.5, B=3.0) triple into the bright path.
    Luminance gate + full-RGB split preserves the source's B-heavy hue in
    the bloom ring.

    Under OLD per-channel thresholding (T=1.0) only G and B exceed T →
    bright splat would be (0.0, 0.5, 2.0), shifting bloom cyan-magenta
    rather than keeping the 1:3:6 source hue. This test is the cool-side
    mirror of the warm test and together they pin chromaticity under
    asymmetric R and B dominance.

    Fixture follows the same pattern as the warm test: bg hue matches the
    bright block's hue (bg = bright / 10, giving bg≈(0.05, 0.15, 0.30))
    so the bg-only render cancels the diffuse offset cleanly under
    subtraction. Source excess over bg = (0.45, 1.35, 2.7) with the SAME
    0.5:1.5:3.0 = 1:3:6 ratio, which is what the luminance-gated bright
    path should preserve.
    """
    H, W = 64, 64
    # B-heavy warm-gray bg (same hue as bright region, i.e. bright / 10
    # so the bg-only subtraction cancels the diffuse offset cleanly —
    # mirrors the warm-test pattern exactly).
    bg_r, bg_g, bg_b = 0.05, 0.15, 0.30
    img = np.empty((H, W, 3), dtype=np.float32)
    img[..., 0] = bg_r
    img[..., 1] = bg_g
    img[..., 2] = bg_b
    # 5x5 cool bright block in the center. Lum = 1.40 > 1.0 (admitted by
    # luminance gate). Bg lum ≈ 0.14 < 1.0 (diffuse path).
    cy, cx = H // 2, W // 2
    img[cy - 2:cy + 3, cx - 2:cx + 3, 0] = 0.5
    img[cy - 2:cy + 3, cx - 2:cx + 3, 1] = 1.5
    img[cy - 2:cy + 3, cx - 2:cx + 3, 2] = 3.0
    dm = np.full((H, W), 0.6, dtype=np.float32)
    t = _blur_tensor()
    _u, lin = render_with_defocus(img, t, dmap=dm, hdr_source="macos", bright_thresh=1.0)
    # Baseline: same image without the bright block (pure B-heavy warm-gray).
    baseline = np.empty_like(img)
    baseline[..., 0] = bg_r
    baseline[..., 1] = bg_g
    baseline[..., 2] = bg_b
    _u2, base_lin = render_with_defocus(baseline, t, dmap=dm, hdr_source="macos",
                           bright_thresh=1.0)
    bloom = lin - base_lin
    # Ring around the 5x5 bright block (radial distance 6-12 px from center).
    yy, xx = np.mgrid[:H, :W]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    ring = (r >= 6.0) & (r <= 12.0)
    bloom_r = float(bloom[..., 0][ring].sum())
    bloom_g = float(bloom[..., 1][ring].sum())
    bloom_b = float(bloom[..., 2][ring].sum())
    # Source excess over bg = (0.45, 1.35, 2.7) → ratios r/b = 0.45/2.7 =
    # 1/6 ≈ 0.1667, g/b = 1.35/2.7 = 0.5. Ratio against the B channel
    # (dominant) rather than R, since R is the smallest excess here and
    # sensitive to accumulator float noise.
    expected_rb = 0.45 / 2.7
    expected_gb = 1.35 / 2.7
    ratio_rb = bloom_r / bloom_b
    ratio_gb = bloom_g / bloom_b
    assert abs(ratio_rb - expected_rb) < 0.10 * expected_rb, (
        f"Chromatic halo (cool): bloom R/B = {ratio_rb:.3f}, expected "
        f"{expected_rb:.3f} (within 10%). "
        f"R={bloom_r:.3f}, G={bloom_g:.3f}, B={bloom_b:.3f}"
    )
    assert abs(ratio_gb - expected_gb) < 0.10 * expected_gb, (
        f"Chromatic halo (cool): bloom G/B = {ratio_gb:.3f}, expected "
        f"{expected_gb:.3f} (within 10%). "
        f"R={bloom_r:.3f}, G={bloom_g:.3f}, B={bloom_b:.3f}"
    )
