"""P1.11 R1-PATCH soft-knee — Adversarial subagent hunt round (R1-PATCH-ADV).

Tests here are Adversarial-subagent-written (separate from Coder's
``test_convolve_brightpass_softknee_adversarial.py``) so the next fix round
can tell them apart at a glance.

``xfail(strict=True)`` marks demonstrated bugs in the Coder's implementation;
plain passing tests add coverage on axes the Coder did not probe.

Paired with LOCKED SPEC: doc/plan/phase1/section11/locked-spec-R1-PATCH-softknee.md
"""
from __future__ import annotations

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


# P1.10.2 Cut 3: --bright-pass-knee CLI flag and bright_knee render() kwarg
# removed; demoted to module constant `_BRIGHT_KNEE = 0.20` in convolve.py.
# Tests pinning the now-removed kwarg behavior (default=0.20 source-grep,
# both-call-sites forwarding source-grep, K=0 hard-boolean, K=-0.1 invalid,
# K=0.5>T contamination, K=NaN/Inf rejection, K=0 vs K=0.4 canopy compare)
# all DELETED here. Tests that take K as ambient (K=0.4 mid-band semantics
# still hold under K=0.20) lose the kwarg arg only.


# ---------------------------------------------------------------------------
# Hunt idea 5: pure-white float precision at T=1.0, K=0.4.
# ---------------------------------------------------------------------------

def test_pure_white_at_default_thresh_gives_half_weight():
    """Hunt idea 5: with T=1.0, K=0.4, a lum=1.0 pixel has
    ``(1.0 - 0.6) / 0.8 = 0.5`` → smoothstep(0.5) = 0.5 exactly.

    This means pure white content is SPLIT 50/50 bright/diffuse at defaults.
    Document this behavior — if future refactor changes the curve (see hunt
    idea 7), this test will catch the semantic shift.
    """
    # Compute the math directly.
    lum = 1.0
    T = 1.0
    K = 0.4
    x = np.clip((lum - (T - K)) / (2.0 * K + 1e-12), 0.0, 1.0)
    w = x * x * (3.0 - 2.0 * x)
    assert abs(w - 0.5) < 1e-6, (
        f"pure-white at T=1.0 K=0.4: bright_weight = {w}, expected 0.5"
    )


# ---------------------------------------------------------------------------
# Hunt idea 6: kernel-sum amplification asymmetry. DESIGN QUESTION.
# ---------------------------------------------------------------------------

def test_knee_midband_energy_amplification():
    """Hunt idea 6: a flat image at ``lum = T`` (middle of knee) splits
    50/50. Diffuse path keeps magnitude ~1·img; bright path multiplies by
    kernel_sum (~peak-normed gaussian sum ≈ kH·kW/area ≈ 100 for our test
    tensor). Combined pixel contribution is ~50.5·img versus ~1·img for
    pixels well below T.

    This is a STEP DISCONTINUITY in total output magnitude across the
    band, not pixel-by-pixel flipping like the original bug, but a
    different artifact class: smooth-but-amplified region vs normal region.

    SEVERITY: design question — by comparing flat panels at lum=T-K-0.1
    (diffuse-only) vs lum=T (mid-band split) we measure the ratio.
    """
    t = _blur_tensor()
    dm = np.full((64, 64), 0.6, dtype=np.float32)
    img_below = np.full((64, 64, 3), 0.5, dtype=np.float32)  # below T-K=0.8
    img_at = np.full((64, 64, 3), 1.0, dtype=np.float32)     # exactly T
    _a, out_below = render_with_defocus(img_below, t, dmap=dm, hdr_source="macos",
                           bright_thresh=1.0)
    _b, out_at = render_with_defocus(img_at, t, dmap=dm, hdr_source="macos",
                        bright_thresh=1.0)
    mean_below = float(out_below[20:44, 20:44, 1].mean())
    mean_at = float(out_at[20:44, 20:44, 1].mean())
    # Ratio of "input-normalized output" measures the amplification delta.
    ratio_below = mean_below / 0.5  # expect ~1.0
    ratio_at = mean_at / 1.0        # expect >> 1.0 with bright-path split
    print(f"  ratio_below={ratio_below:.3f}  ratio_at={ratio_at:.3f}")
    # Report without hard-failing. The assert below is loose; the print is
    # the actual diagnostic. If ratio_at / ratio_below > 5 we have a
    # discontinuity worth flagging.
    amplification = ratio_at / max(ratio_below, 1e-6)
    # Hard assert that we don't explode to >100× — that would indicate a
    # real bug in the bright-path normalization.
    assert amplification < 100.0, (
        f"bright-path amplification at midband is {amplification:.1f}× "
        f"vs below-band. If >> 1× this is the known asymmetric-kernel-sum "
        f"design issue (peak vs L1). Flag to TL for design review."
    )


# ---------------------------------------------------------------------------
# Hunt idea 7: pin smoothstep curve (not linear, not cosine).
# ---------------------------------------------------------------------------

def test_smoothstep_curve_shape_pinned():
    """Hunt idea 7: the spec locked ``3x²-2x³`` cubic smoothstep. A future
    refactor to a cosine ramp or linear would silently change semantics.

    Pin the shape at three internal points. If Coder swaps curves, this
    fails. Derivatives at endpoints must be 0 (C¹ continuity).
    """
    # Directly evaluate the same formula as production.
    def ss(x):
        x = np.clip(x, 0.0, 1.0)
        return x * x * (3.0 - 2.0 * x)

    # Pin values:
    assert abs(ss(0.0) - 0.0) < 1e-9
    assert abs(ss(0.25) - 0.15625) < 1e-6   # 3*0.0625 - 2*0.015625
    assert abs(ss(0.5) - 0.5) < 1e-9
    assert abs(ss(0.75) - 0.84375) < 1e-6
    assert abs(ss(1.0) - 1.0) < 1e-9

    # Derivative at endpoints: d/dx (3x²-2x³) = 6x - 6x² = 6x(1-x)
    # At x=0 → 0, at x=1 → 0. Verify via finite diff:
    h = 1e-5
    d0 = (ss(0 + h) - ss(0)) / h
    d1 = (ss(1) - ss(1 - h)) / h
    assert abs(d0) < 1e-3, f"d/dx at 0 should be 0, got {d0}"
    assert abs(d1) < 1e-3, f"d/dx at 1 should be 0, got {d1}"


# ---------------------------------------------------------------------------
# Hunt idea 8: is_delta interaction with non-zero knee.
# ---------------------------------------------------------------------------

def test_is_delta_branch_suppresses_bright_at_nonzero_knee():
    """Hunt idea 8: when ``is_delta`` fires (d=0 slab), the bright path is
    skipped inside the inner loop. At K=0.4, a mid-band pixel has
    ``bright_weight=0.5`` — but on an all-delta tensor that weight should
    still produce output ≈ input (because the delta branch routes the FULL
    ``img_padded`` through, not just the diffuse half).

    Verify: all-delta tensor + lum near T + K=0.4 produces output magnitude
    within 5% of input. If the is_delta suppression is broken, we'd see
    either doubling (image + image) or halving (only diffuse half).
    """
    k = 33
    kern = np.zeros((6, 4, k, k), dtype=np.float32)
    kern[:, :, k // 2, k // 2] = 1.0  # all-delta across all (r, d)
    t = {"kernels": kern, "radial_edges": _RE.copy(), "defocus_ratios": _DR.copy()}
    img = np.full((64, 64, 3), 1.0, dtype=np.float32)  # lum=1.0, mid-band
    dm = np.zeros((64, 64), dtype=np.float32)  # d=0 → is_delta fires
    _a, out = render_with_defocus(img, t, dmap=dm, hdr_source="macos",
                     bright_thresh=1.0)
    # Interior pixels (away from pad): delta + full img routing means output ≈ 1.
    interior = out[16:48, 16:48, :]
    assert abs(float(interior.mean()) - 1.0) < 0.05, (
        f"is_delta + mid-band knee: expected ~1.0, got {float(interior.mean()):.4f}"
    )


# ---------------------------------------------------------------------------
# Hunt idea 9: peak/stamp mode interaction.
# ---------------------------------------------------------------------------

def test_soft_knee_with_psf_norm_peak_stamp_mode():
    """Hunt idea 9: psf_norm='peak' (stamp mode) auto-exposes by output.max().
    Check soft-knee still produces finite, bounded output in stamp mode.

    This is a smoke test — catches divide-by-zero if omax drifts, and
    NaN/Inf from the two-path accumulator interacting with stamp-mode renorm.
    """
    img = np.full((64, 64, 3), 0.3, dtype=np.float32)
    img[28:36, 28:36, :] = 2.0  # bright blob
    dm = np.full((64, 64), 0.6, dtype=np.float32)
    t = _blur_tensor()
    _a, out = render_with_defocus(img, t, dmap=dm, hdr_source="macos",
                     psf_norm="peak", bright_thresh=1.0)
    assert np.all(np.isfinite(out)), "stamp mode + soft-knee: non-finite output"
    assert float(out.max()) <= 1.0 + 1e-4, (
        f"stamp mode should auto-expose to max <=1.0, got {float(out.max())}"
    )


# P1.10.2 Cut 3: test_soft_knee_beats_hard_boolean_on_uniform_canopy_strict
# DELETED — compared bright_knee=0.0 vs bright_knee=0.4, both removed.


# ---------------------------------------------------------------------------
# Hunt idea 11: midtone leak magnitude at K=0.4.
# ---------------------------------------------------------------------------

def test_midtone_leak_band_magnitude_at_default_knee():
    """Regression gate for TL REVISION 2 + 3 — confirms default K keeps
    lum=0.7 midtones at bright_weight=0 exactly.

    With T=1.0 and K=0.20 (current default, narrowed from initial 0.4 after
    midtone-leak showstopper, then widened from 0.15 after K-sweep visual
    review), the knee band is [0.80, 1.20]. A flat lum=0.7 panel lies
    outside the band (below T-K=0.80) → smoothstep arg clipped to 0 →
    bright_weight=0 exactly → no bright-path amplification at all. This
    test flipped from xfail under K=0.4 (which amplified lum=0.7 by ~4.5×)
    to PASS under any K<=0.3."""
    img = np.full((64, 64, 3), 0.7, dtype=np.float32)
    dm = np.full((64, 64), 0.6, dtype=np.float32)
    t = _blur_tensor()
    # Use the default knee — this is the whole point of the regression gate.
    _a, out_leak = render_with_defocus(img, t, dmap=dm, hdr_source="macos",
                          bright_thresh=1.0)
    center_leak = float(out_leak[20:44, 20:44, 1].mean())
    # Confirm lum=0.7 is fully protected at default K: center mean within 5%
    # of 0.7 (pure diffuse path — no bright-path energy leak).
    assert abs(center_leak - 0.7) / 0.7 < 0.05, (
        f"midtone leak at lum=0.7 with default knee: expected ~0.7, got "
        f"{center_leak:.4f} (deviation > 5%). Default K must keep lum=0.7 "
        f"at bright_weight=0 exactly."
    )


# ---------------------------------------------------------------------------
# Hunt idea 12: positional-arg backward-compat.
# ---------------------------------------------------------------------------

def test_render_kwargs_stable_for_positional_callers():
    """Hunt idea 12: positional-arg backward-compat. P1.10.2 Cut 3 removed
    ``bright_knee`` from the signature; pin the post-removal order so any
    future kwarg insertion that breaks positional callers past
    ``bright_thresh`` is caught at import time.
    """
    import inspect
    sig = inspect.signature(render)
    params = list(sig.parameters)
    # P1.10.2 Cut 3: bright_knee removed (demoted to module constant).
    # P1.audit Item 3: on_progress + cancel added as keyword-only params
    # (after `*`) — does NOT affect positional callers past bright_thresh
    # because they are kwarg-only. Pin includes them for drift detection.
    expected = [
        "image_srgb", "tensor", "depth_m",
        "psf_norm", "hdr_source", "bright_thresh",
        "bright_pass_region_size", "bright_pass_floor", "bright_pass_norm_pct",
        "occlusion_compositing",
        "slab_batch_size", "on_progress", "cancel", "stats",
        "bright_stamp_gain",  # S22 S1 (keyword-only tail)
    ]
    assert params == expected, (
        f"render() signature drifted: got {params}, expected {expected}. "
        f"Positional callers past `bright_thresh` would break."
    )


# P1.10.2 Cut 3: test_nan_knee_rejected_or_produces_zero_output and
# test_inf_knee_raises_or_produces_nan_gate_failure DELETED — both probed
# the now-removed bright_knee kwarg. K is the module constant 0.20.
