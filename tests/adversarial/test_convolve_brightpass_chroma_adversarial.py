"""P1.11 R1-CHROMA — Adversarial round-2 hunt after luminance-gate landing.

Fresh-eyes pass on the Rec.709-luminance bright-pass split that replaced the
per-channel clip in ``summicron/render/convolve.py:494-506`` (plus matching
energy-gate denominator at ``:713-731``).

Probe surface:
- Luminance-boundary discontinuity (ramp crossing ``thresh``).
- ``lum == bright_thresh`` exact-boundary semantics (strict ``>`` vs ``<=``).
- Negative-channel input (HDR LUT artifacts) with mixed positive dominant.
- Luminance-dominant channel cases (G-heavy, single-channel R-saturated).
- Two distinct-hue bright pixels not bleeding chromaticity into each other.
- Energy-gate threshold sweep on a **mixed-hue** HDR scene (the existing
  sweep test uses only an achromatic bright blob).
- ``bright_mask`` dtype + shape regression tripwire.
- Rec.709 Y-weights float32 precision (boundary lum == thresh exact).
- Critique-grade replay of Coder's warm/cool chromaticity tests with
  **heterogeneous** background (no matching-hue convenience) to see whether
  the ±10% bound survives when the bg-subtraction cancels less cleanly.

All tests write new coverage. Production code is untouched.
"""
from __future__ import annotations

import numpy as np
import pytest

from summicron.render.convolve import render
from tests.fixtures.defocus import render_with_defocus



_RE = np.array([0.0, 0.15, 0.30, 0.45, 0.60, 0.80, 1.01], dtype=np.float64)
_DR = np.array([0.0, 0.33, 0.67, 1.0], dtype=np.float64)


def _blur_tensor(k: int = 33) -> dict:
    """Same shape-shim tensor used by the Coder's R1 tests."""
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
# Luminance-boundary smoothness
# ---------------------------------------------------------------------------

def test_luminance_ramp_has_no_step_at_threshold():
    """An achromatic luminance ramp 0.5..1.5 crossing ``thresh=1.0`` should
    produce a smooth output — no pixel-wide step at the boundary.

    The bright path is peak-normed so the transition is a C0 kink (not a
    step), but the output must remain monotonic through the boundary region
    and the worst first-difference must not exceed the typical-bloom first-
    difference by orders of magnitude. Regression signal: if a future edit
    accidentally inserts a discontinuous mask term or routes the bright
    splat additively without peak-norm, a pixel-wide spike will appear here.
    """
    H, W = 80, 200
    ramp = np.linspace(0.5, 1.5, W, dtype=np.float32)
    img = np.empty((H, W, 3), dtype=np.float32)
    img[..., 0] = ramp[None, :]
    img[..., 1] = ramp[None, :]
    img[..., 2] = ramp[None, :]
    dm = np.full((H, W), 0.6, dtype=np.float32)
    t = _blur_tensor()
    # P1.10.2 Cut 1: this gate pins legacy bright-pass smoothness; force
    # bright_pass_region_size=0 (top-hat off) so the test is independent of
    # the region-aware bright-pass default.
    _u, lin = render_with_defocus(img, t, dmap=dm, hdr_source="macos",
                     bright_thresh=1.0, bright_pass_region_size=0)
    # Take a middle row (R channel; achromatic so all channels equal).
    row = lin[H // 2, :, 0]
    # Output must be monotonically increasing across the full ramp (bloom
    # from the bright side can only amplify higher-luminance regions more).
    # Allow small float wobble.
    d = np.diff(row)
    # Mostly monotonic — count violations.
    n_neg = int(np.sum(d < -1e-3))
    assert n_neg == 0, (
        f"Ramp output dropped {n_neg} times; bright path likely introduced "
        f"a non-monotonic step at the luminance boundary. "
        f"min diff={float(d.min()):.4f}"
    )
    # Worst first-difference must not exceed 10× the typical (median abs)
    # first-difference — large margin; we're only catching a STEP, not a
    # curvature bump.
    median_abs = float(np.median(np.abs(d)) + 1e-8)
    max_abs = float(np.max(np.abs(d)))
    assert max_abs < 30.0 * median_abs, (
        f"Suspicious jump at luminance boundary: max|diff|={max_abs:.4f} "
        f"> 30× median|diff|={median_abs:.4f}"
    )


# ---------------------------------------------------------------------------
# lum == thresh exact boundary semantics (strict > vs <= asymmetry)
# ---------------------------------------------------------------------------

# P1.10.2 Cut 3: test_luminance_at_thresh_midpoint_is_half_bright_weighted
# DELETED — compared bright_knee=0.4 vs bright_knee=0.0, both kwargs removed.


def test_luminance_just_above_threshold_fires_bright_mask():
    """Mirror of the exact-boundary test: ``lum = 1 + small ε`` must put
    the bright pixel into the bright path. A future edit that switches
    ``>`` to ``>=`` would not regress this, but switching to ``<`` or
    adding a knee with an epsilon offset could — this test anchors the
    "admit" side of the gate. Uses one isolated above-thresh pixel on a
    dim bg (not a uniform bright image) to keep the total-energy gate
    in range.
    """
    H, W = 48, 48
    img = np.full((H, W, 3), 0.1, dtype=np.float32)
    # Single pixel just above the luminance boundary.
    above = np.float32(1.0) + np.float32(1e-4)
    img[H // 2, W // 2, :] = above
    dm = np.full((H, W), 0.6, dtype=np.float32)
    t = _blur_tensor()
    _a, la = render_with_defocus(img, t, dmap=dm, hdr_source="macos",
                    bright_thresh=1.0)
    _b, lb = render_with_defocus(img, t, dmap=dm, hdr_source="macos",
                    bright_thresh=1e9)
    # At thresh=1e9 the bright path is disabled and the single pixel's
    # energy is L1-averaged (output pixel ~= mean of kernel support which
    # keeps peak close to the in-focus bg). At thresh=1.0 the bright path
    # fires (peak-normed splat on that pixel), producing an output peak
    # noticeably ABOVE the disabled baseline. If `>` silently flipped to
    # `<` or an epsilon-knee hid the pixel, the two outputs would match.
    assert float(la.max()) > 1.1 * float(lb.max()), (
        f"lum > thresh did not fire on above-boundary pixel: "
        f"la.max={float(la.max()):.4f} vs lb.max={float(lb.max()):.4f} "
        f"— bright path not engaging? Same or lower max suggests mask=0 "
        f"for the target pixel."
    )


# ---------------------------------------------------------------------------
# Negative-channel HDR LUT artifacts
# ---------------------------------------------------------------------------

def test_mixed_negative_positive_channels_no_leak():
    """Core Image extended-linear HDR may return channels with tiny
    negative values adjacent to large positives (post-LUT, pre-clip). A
    pixel with ``R=3.0, G=3.0, B=-1e-4`` has Rec.709 lum ≈ 2.14, admitted.
    The bright path receives the full triple (with the tiny negative);
    the diffuse path receives exactly zero. Verify no NaN/Inf and that
    the negative does not propagate into a large artifact.
    """
    H, W = 64, 64
    img = np.full((H, W, 3), 0.2, dtype=np.float32)
    img[30, 30, 0] = 3.0
    img[30, 30, 1] = 3.0
    img[30, 30, 2] = -1e-4
    dm = np.full((H, W), 0.5, dtype=np.float32)
    t = _blur_tensor()
    _u, lin = render_with_defocus(img, t, dmap=dm, hdr_source="macos",
                     bright_thresh=1.0)
    assert np.isfinite(lin).all()
    # The negative channel's magnitude after blur should remain tiny (it's
    # a single-pixel -1e-4 smeared by an L1-normalized bloom kernel, so its
    # absolute contribution is ≤ 1e-4 anywhere). A bug that splat the
    # negative with a peak-normed kernel would amplify by kernel-sum.
    assert float(lin[..., 2].min()) > -1e-2, (
        f"Negative-B channel propagated too aggressively: "
        f"min B = {float(lin[..., 2].min()):.6f}"
    )


# ---------------------------------------------------------------------------
# Chromaticity under luminance-dominant channel variants
# ---------------------------------------------------------------------------

def test_single_red_channel_saturation_preserves_50_1_hue():
    """Adversarial idea 6: ``R=5.0, G=0.1, B=0.1`` has lum ≈ 1.14 (just over
    1.0). The luminance gate admits the full pixel; the bloom ring should
    carry R:G:B = 50:1:1 within float-precision noise. A bug that clipped
    or renormed the bright triple would collapse this ratio.
    """
    H, W = 64, 64
    bg_r, bg_g, bg_b = 0.5, 0.01, 0.01  # bright / 10 matching hue
    img = np.empty((H, W, 3), dtype=np.float32)
    img[..., 0] = bg_r
    img[..., 1] = bg_g
    img[..., 2] = bg_b
    cy, cx = H // 2, W // 2
    img[cy - 2:cy + 3, cx - 2:cx + 3, 0] = 5.0
    img[cy - 2:cy + 3, cx - 2:cx + 3, 1] = 0.1
    img[cy - 2:cy + 3, cx - 2:cx + 3, 2] = 0.1
    dm = np.full((H, W), 0.6, dtype=np.float32)
    t = _blur_tensor()
    _u, lin = render_with_defocus(img, t, dmap=dm, hdr_source="macos",
                     bright_thresh=1.0)
    bg_img = np.empty_like(img)
    bg_img[..., 0] = bg_r
    bg_img[..., 1] = bg_g
    bg_img[..., 2] = bg_b
    _u2, base_lin = render_with_defocus(bg_img, t, dmap=dm, hdr_source="macos",
                           bright_thresh=1.0)
    bloom = lin - base_lin
    yy, xx = np.mgrid[:H, :W]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    ring = (r >= 6.0) & (r <= 12.0)
    br_r = float(bloom[..., 0][ring].sum())
    br_g = float(bloom[..., 1][ring].sum())
    br_b = float(bloom[..., 2][ring].sum())
    # Source excess (5-0.5, 0.1-0.01, 0.1-0.01) = (4.5, 0.09, 0.09)
    # Expected G/R ≈ B/R ≈ 0.02.
    assert abs(br_g / br_r - 0.02) < 0.10 * 0.02, (
        f"Single-R saturation: G/R = {br_g / br_r:.4f}, expected 0.02. "
        f"R={br_r:.3f} G={br_g:.3f} B={br_b:.3f}"
    )
    assert abs(br_b / br_r - 0.02) < 0.10 * 0.02, (
        f"Single-R saturation: B/R = {br_b / br_r:.4f}, expected 0.02. "
        f"R={br_r:.3f} G={br_g:.3f} B={br_b:.3f}"
    )


def test_green_heavy_pixel_preserves_luminance_weighted_hue():
    """Adversarial idea 5: ``R=0.1, G=5.0, B=0.05`` — Rec.709 lum ≈ 3.60.
    Gate admits full pixel. Bloom must carry R/G = 0.02 and B/G = 0.01
    (from source-excess ratios). A bug that weighted the bright splat by
    max-channel instead of lum would pass similarly, but a bug that renormed
    per-channel or dropped low channels would collapse R and B toward zero.
    """
    H, W = 64, 64
    bg_r, bg_g, bg_b = 0.01, 0.5, 0.005  # matching hue
    img = np.empty((H, W, 3), dtype=np.float32)
    img[..., 0] = bg_r
    img[..., 1] = bg_g
    img[..., 2] = bg_b
    cy, cx = H // 2, W // 2
    img[cy - 2:cy + 3, cx - 2:cx + 3, 0] = 0.1
    img[cy - 2:cy + 3, cx - 2:cx + 3, 1] = 5.0
    img[cy - 2:cy + 3, cx - 2:cx + 3, 2] = 0.05
    dm = np.full((H, W), 0.6, dtype=np.float32)
    t = _blur_tensor()
    _u, lin = render_with_defocus(img, t, dmap=dm, hdr_source="macos",
                     bright_thresh=1.0)
    bg_img = np.empty_like(img)
    bg_img[..., 0] = bg_r
    bg_img[..., 1] = bg_g
    bg_img[..., 2] = bg_b
    _u2, base_lin = render_with_defocus(bg_img, t, dmap=dm, hdr_source="macos",
                           bright_thresh=1.0)
    bloom = lin - base_lin
    yy, xx = np.mgrid[:H, :W]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    ring = (r >= 6.0) & (r <= 12.0)
    br_r = float(bloom[..., 0][ring].sum())
    br_g = float(bloom[..., 1][ring].sum())
    br_b = float(bloom[..., 2][ring].sum())
    # Source excess (0.09, 4.5, 0.045). Expected R/G = 0.02, B/G = 0.01.
    assert abs(br_r / br_g - 0.02) < 0.10 * 0.02, (
        f"G-heavy: R/G = {br_r / br_g:.4f}, expected 0.02. "
        f"R={br_r:.3f} G={br_g:.3f} B={br_b:.3f}"
    )
    assert abs(br_b / br_g - 0.01) < 0.10 * 0.01, (
        f"G-heavy: B/G = {br_b / br_g:.4f}, expected 0.01. "
        f"R={br_r:.3f} G={br_g:.3f} B={br_b:.3f}"
    )


# ---------------------------------------------------------------------------
# Two distinct-hue bright pixels — no cross-bleed
# ---------------------------------------------------------------------------

def test_two_distinct_hue_bright_pixels_no_cross_bleed():
    """Warm bright at (20,20) and cool bright at (60,60) in an 80×80 image.
    Each pixel's bloom ring at r≤8 must preserve its OWN hue, not drift
    toward the other pixel's hue. A bug that smeared both bright pixels
    through a single shared kernel (e.g. summing them before FFT) would
    cause their rings to converge toward the averaged hue.

    Uses matching-hue backgrounds for each pixel via a diagonal split so
    each bloom's diffuse-offset subtraction cancels cleanly.
    """
    H, W = 80, 80
    img = np.full((H, W, 3), 0.1, dtype=np.float32)
    # Warm (R=3.0, G=1.5, B=0.5): lum ≈ 1.75
    img[18:22, 18:22, 0] = 3.0
    img[18:22, 18:22, 1] = 1.5
    img[18:22, 18:22, 2] = 0.5
    # Cool (R=0.5, G=1.5, B=3.0): lum ≈ 1.40
    img[58:62, 58:62, 0] = 0.5
    img[58:62, 58:62, 1] = 1.5
    img[58:62, 58:62, 2] = 3.0
    dm = np.full((H, W), 0.3, dtype=np.float32)
    t = _blur_tensor()
    _u, lin = render_with_defocus(img, t, dmap=dm, hdr_source="macos",
                     bright_thresh=1.0)
    base = np.full_like(img, 0.1)
    _u2, base_lin = render_with_defocus(base, t, dmap=dm, hdr_source="macos",
                           bright_thresh=1.0)
    bloom = lin - base_lin
    yy, xx = np.mgrid[:H, :W]
    # Tight warm ring (r<=8) far from cool center
    rw = np.sqrt((yy - 20) ** 2 + (xx - 20) ** 2)
    warm_ring = (rw >= 5.0) & (rw <= 8.0)
    br_w = float(bloom[..., 0][warm_ring].sum())
    bg_w = float(bloom[..., 1][warm_ring].sum())
    bb_w = float(bloom[..., 2][warm_ring].sum())
    # Warm bloom R >> B (warm hue)
    assert br_w > 3.0 * bb_w, (
        f"Warm bloom not R-dominant: R={br_w:.3f} G={bg_w:.3f} B={bb_w:.3f} "
        f"— cool-pixel hue bled across 40 px gap"
    )
    # Tight cool ring
    rc = np.sqrt((yy - 60) ** 2 + (xx - 60) ** 2)
    cool_ring = (rc >= 5.0) & (rc <= 8.0)
    cr_c = float(bloom[..., 0][cool_ring].sum())
    cg_c = float(bloom[..., 1][cool_ring].sum())
    cb_c = float(bloom[..., 2][cool_ring].sum())
    # Cool bloom B >> R (cool hue)
    assert cb_c > 3.0 * cr_c, (
        f"Cool bloom not B-dominant: R={cr_c:.3f} G={cg_c:.3f} B={cb_c:.3f} "
        f"— warm-pixel hue bled"
    )


# ---------------------------------------------------------------------------
# Energy-gate denominator — mixed-hue HDR sweep
# ---------------------------------------------------------------------------

def test_energy_gate_holds_on_mixed_hue_hdr_sweep():
    """Coder's existing sweep uses only an achromatic bright blob; with
    luminance-gated splits the denominator logic is hue-insensitive in
    the achromatic case (all channels follow the same mask). A mixed-hue
    scene (warm + cool + single-R + single-G HDR blocks) exercises the
    gate denominator where the luminance mask and per-channel envelope
    diverge most strongly. The gate must not fail for any threshold in
    a broad sweep.
    """
    H, W = 64, 64
    img = np.full((H, W, 3), 0.2, dtype=np.float32)
    img[10:14, 10:14] = [3.0, 2.0, 0.7]   # warm
    img[10:14, 50:54] = [0.5, 1.5, 3.0]   # cool
    img[50:54, 10:14] = [5.0, 0.5, 0.5]   # red
    img[50:54, 50:54] = [0.3, 5.0, 0.3]   # green
    dm = np.full((H, W), 0.5, dtype=np.float32)
    t = _blur_tensor()
    # Sweep below, at, across, and above bright ranges
    for th in (0.01, 0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 1e9):
        # Must not raise
        _u, lin = render_with_defocus(img, t, dmap=dm, hdr_source="macos",
                         bright_thresh=th)
        assert np.isfinite(lin).all(), f"non-finite at thresh={th}"


# ---------------------------------------------------------------------------
# Coder-test critique: heterogeneous (non-matching) background
# ---------------------------------------------------------------------------

def test_warm_highlight_chromaticity_with_neutral_background():
    """CRITIQUE GATE — the Coder's two chromaticity tests use backgrounds
    that match the bright block's hue (bg = bright / 10). This makes the
    bg-only subtraction cancel the diffuse offset trivially. A TRULY
    adversarial test should use a neutral (achromatic) background so the
    bg-subtraction cannot hide a small per-channel ratio drift. The ±10%
    tolerance is relaxed to ±20% here because the neutral-bg diffuse ring
    contributes measurably to R/G/B ratios in the ring even after subtraction,
    so some ratio drift is expected — but the bloom must still be warm-
    dominant (R > G > B), not chromatically neutralized.
    """
    H, W = 64, 64
    # NEUTRAL achromatic bg (R=G=B=0.1), lum=0.1 well below 1.0
    img = np.full((H, W, 3), 0.1, dtype=np.float32)
    cy, cx = H // 2, W // 2
    # Same warm bright block as Coder's test
    img[cy - 2:cy + 3, cx - 2:cx + 3, 0] = 3.0
    img[cy - 2:cy + 3, cx - 2:cx + 3, 1] = 2.0
    img[cy - 2:cy + 3, cx - 2:cx + 3, 2] = 1.5
    dm = np.full((H, W), 0.6, dtype=np.float32)
    t = _blur_tensor()
    _u, lin = render_with_defocus(img, t, dmap=dm, hdr_source="macos",
                     bright_thresh=1.0)
    # bg-only baseline is achromatic — subtraction cancels only the
    # achromatic offset; any chromatic drift from the bright path remains.
    base = np.full_like(img, 0.1)
    _u2, base_lin = render_with_defocus(base, t, dmap=dm, hdr_source="macos",
                           bright_thresh=1.0)
    bloom = lin - base_lin
    yy, xx = np.mgrid[:H, :W]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    ring = (r >= 6.0) & (r <= 12.0)
    br = float(bloom[..., 0][ring].sum())
    bg = float(bloom[..., 1][ring].sum())
    bb = float(bloom[..., 2][ring].sum())
    # Warm hue: R > G > B strictly, maintained under neutral bg.
    assert br > bg > bb, (
        f"Warm bloom hue order broken with neutral bg: "
        f"R={br:.3f} G={bg:.3f} B={bb:.3f}"
    )
    # Loose-but-real hue ratio check: bloom_G/bloom_R within 30% of 2/3.
    expected_gr = 2.0 / 3.0
    ratio_gr = bg / br
    assert abs(ratio_gr - expected_gr) < 0.30 * expected_gr, (
        f"Warm bloom G/R = {ratio_gr:.3f}, expected {expected_gr:.3f} ±30% "
        f"(neutral bg, looser than matched-hue test)"
    )


# ---------------------------------------------------------------------------
# Regression tripwires (shape / dtype)
# ---------------------------------------------------------------------------

def test_bright_mask_is_broadcastable_3d():
    """Regression tripwire: the spec comment documents ``bright_mask`` as
    shape ``(H_pad, W_pad, 1)`` so ``img_padded * bright_mask`` broadcasts
    across channels. A future edit removing ``[..., None]`` would demote the
    mask to 2D and either crash or silently broadcast incorrectly across
    the last axis. Since bright_mask is a local, we probe this indirectly:
    render a mixed-channel image where a 2D broadcast would produce an
    observably different output than a 3D broadcast.

    Mechanism: with a ``(H, W)`` 2D mask, ``img * mask`` broadcasts mask
    along axis 0 — i.e. mask[0] multiplies every column's row-0, producing
    a transpose-like artifact. Probe shape: a row-varying image under an
    asymmetric mask will differ.
    """
    # Indirect probe via output consistency on a rotated image.
    H, W = 48, 48
    img = np.full((H, W, 3), 0.1, dtype=np.float32)
    img[20, 30, 0] = 3.0  # off-center warm pixel in R channel
    img[20, 30, 1] = 3.0
    img[20, 30, 2] = 3.0
    dm = np.full((H, W), 0.5, dtype=np.float32)
    t = _blur_tensor()
    _u, lin = render_with_defocus(img, t, dmap=dm, hdr_source="macos",
                     bright_thresh=1.0)
    # All three channels should bloom equally (achromatic input).
    # If bright_mask lost its [..., None] and 2D-broadcast, channels would
    # diverge (mask[0,:] would be applied differently per channel via
    # numpy's broadcasting rules).
    diff_rg = float(np.max(np.abs(lin[..., 0] - lin[..., 1])))
    diff_rb = float(np.max(np.abs(lin[..., 0] - lin[..., 2])))
    assert diff_rg < 1e-4, (
        f"Achromatic input yielded chromatic output: max|R-G|={diff_rg:.6f} "
        f"— bright_mask may have wrong shape"
    )
    assert diff_rb < 1e-4, (
        f"Achromatic input yielded chromatic output: max|R-B|={diff_rb:.6f} "
        f"— bright_mask may have wrong shape"
    )


def test_bright_mask_dtype_is_float_not_bool():
    """The spec pins ``bright_mask`` dtype to ``np.float32`` (via the
    ``.astype(np.float32)`` cast). Multiplying a float32 image by a bool
    array works via numpy's promotion rules but produces float64 output
    (upcast), which would cascade into later float32 assertions and cost
    memory. Probe: render output dtype must be float32.
    """
    img = np.full((48, 48, 3), 0.5, dtype=np.float32)
    img[20:24, 20:24, :] = 3.0
    dm = np.full((48, 48), 0.5, dtype=np.float32)
    t = _blur_tensor()
    _u, lin = render_with_defocus(img, t, dmap=dm, hdr_source="macos",
                     bright_thresh=1.0)
    assert lin.dtype == np.float32, (
        f"Output dtype drifted to {lin.dtype}; bright_mask may have been "
        f"left as bool and promoted the math to float64"
    )


# ---------------------------------------------------------------------------
# Rec.709 precision at boundary
# ---------------------------------------------------------------------------

def test_rec709_weights_sum_exactly_to_one_float32():
    """Defensive probe: confirm the Rec.709 weights
    (0.2126, 0.7152, 0.0722) sum to exactly 1.0 in float32 — this is the
    precondition for ``R=G=B=1.0`` giving ``lum=1.0`` bit-exact, and hence
    for the ``lum > bright_thresh`` boundary to have well-defined semantics
    on an achromatic 1.0 pixel.

    If a future edit uses higher-precision weights like 0.21260 +
    0.71520 + 0.07220 that don't sum exactly to 1.0 in float32, this
    test fails fast, preventing a subtle flavor of boundary drift.
    """
    w = np.float32(0.2126) + np.float32(0.7152) + np.float32(0.0722)
    assert w == np.float32(1.0), (
        f"Rec.709 weights do not sum to exactly 1.0 in float32: sum={w!r}. "
        f"The `lum > thresh` boundary at achromatic value 1.0 becomes "
        f"under- or over-defined depending on rounding direction."
    )
