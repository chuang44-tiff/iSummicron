"""Adversarial round-2 tests for eval harness (post-Coder).

Attacks beyond §5 items 1-13 already covered in
``test_{fixtures,metrics}_adversarial.py``. Focus: sign correctness, donut
PSFs, degenerate references, CLI end-to-end, numpy leaks in nested dicts,
empty metrics tuple, annotation polarity contract, monotonicity under noise.

Each test asserts either:
  - harness handles the edge correctly  (PASS → no real gap), or
  - harness silently does the wrong thing  (FAIL → real gap, Coder owns the fix).

If a test is written to document a baseline (not a bug), it is marked so.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from summicron.eval.metrics import (
    energy_ratio,
    m2_roundness,
    m3_sag_tan,
    m4_diameter_depth_monotonicity,
    m5_in_focus_sharpness,
    m7_wrap_probe,
)
from summicron.eval.metrics.common import (
    fit_ellipse_guarded,
    project_covariance_radial,
)
from summicron.eval.score import DEFAULT_METRICS, _py, score
from tests.unit.eval.conftest import (
    bubble_grid_annotation_stub,
    mono_rgb,
    rasterize_disc,
    rasterize_ellipse,
    standard_grid_discs,
)


# ============================================================================
# Attack A — M3 sign correctness (post-mortem 1 pattern)
# ============================================================================
#
# The spec says: "a flipped-polarity PSF shows up as sag_tan < 1 vs > 1, not
# merely non-1". The existing test only checks magnitude. Build a synthetic
# ellipse with KNOWN radial/tangential orientation and assert the sign.
#
# Corner SE at (904, 904) with optical center (512, 512): radial direction
# points outward, ~45 deg. A tangentially-elongated PSF at this corner (major
# axis perpendicular to the radial) should yield sag_tan < 1 (sagittal sigma
# smaller than tangential). A radially-elongated PSF should yield sag_tan > 1.


def _corner_se_annotation(r_frac=0.77):
    return bubble_grid_annotation_stub(
        [
            {
                "idx": 9,
                "center_xy_px": [904, 904],
                "r_frac": r_frac,
                "depth_m": 10.23,
                "source_diameter_px": 16,
                "wrap_applied": False,
            }
        ]
    )


def test_round2_m3_tangential_elongation_gives_sag_tan_lt_1():
    """Tangentially-elongated blob at corner SE → sag_tan < 1 (sign check)."""
    h = w = 1024
    # Radial direction at (904,904) relative to (512,512) is (1,1)/sqrt(2) ≈ 45°.
    # Tangential (perp) is (-1,1)/sqrt(2) ≈ 135°.
    # Elongate along TANGENTIAL: major axis at theta = 135° (= 3pi/4).
    theta = np.deg2rad(135.0)
    mono = rasterize_ellipse(h, w, 904, 904, a=24.0, b=8.0, theta_rad=theta)
    rgb = mono_rgb(mono)
    ann = _corner_se_annotation()
    out = m3_sag_tan(rgb, ann)
    assert out[0]["flag"] is None, out[0]
    # Tangential sigma > sagittal sigma → ratio < 1.
    assert out[0]["sag_tan"] < 0.9, (
        f"tangential elongation should yield sag_tan < 1, got {out[0]['sag_tan']}"
    )


def test_round2_m3_radial_elongation_gives_sag_tan_gt_1():
    """Radially-elongated blob at corner SE → sag_tan > 1 (sign check)."""
    h = w = 1024
    # Elongate along RADIAL: major axis at theta = 45°.
    theta = np.deg2rad(45.0)
    mono = rasterize_ellipse(h, w, 904, 904, a=24.0, b=8.0, theta_rad=theta)
    rgb = mono_rgb(mono)
    ann = _corner_se_annotation()
    out = m3_sag_tan(rgb, ann)
    assert out[0]["flag"] is None, out[0]
    assert out[0]["sag_tan"] > 1.1, (
        f"radial elongation should yield sag_tan > 1, got {out[0]['sag_tan']}"
    )


def test_round2_m3_radial_vs_tangential_are_reciprocal():
    """Radial and tangential elongations with same aspect → reciprocal ratios."""
    h = w = 1024
    ann = _corner_se_annotation()
    mono_rad = rasterize_ellipse(
        h, w, 904, 904, a=24.0, b=8.0, theta_rad=np.deg2rad(45.0)
    )
    mono_tan = rasterize_ellipse(
        h, w, 904, 904, a=24.0, b=8.0, theta_rad=np.deg2rad(135.0)
    )
    r_rad = m3_sag_tan(mono_rgb(mono_rad), ann)[0]["sag_tan"]
    r_tan = m3_sag_tan(mono_rgb(mono_tan), ann)[0]["sag_tan"]
    # Reciprocal within discretization noise.
    assert abs(r_rad * r_tan - 1.0) < 0.15, (
        f"radial*tangential should be ~1 (reciprocal), got {r_rad}*{r_tan}={r_rad*r_tan}"
    )


# ============================================================================
# Attack B — Donut / annular PSF (real defocused PSFs are ring-shaped)
# ============================================================================
#
# A defocused PSF can be a bright ring with a dark center. Above the
# 0.1*max threshold, the whole ring is one connected component. The ellipse
# fit via regionprops will report the ring's bbox but area will be JUST the
# ring area — M4's `2*sqrt(area/pi)` diameter thus UNDERESTIMATES the real
# disc diameter (which is the outer ring diameter). This matters for M4
# monotonicity: if some discs are donuts and others are filled, diameter
# comparison is apples-to-oranges.


def test_round2_annular_psf_does_not_crash_ellipse_fit():
    """Donut ROI → ellipse fit returns ok (no crash), but note the area."""
    # 40x40 ROI with a ring: outer radius 15, inner radius 8.
    roi = np.zeros((40, 40), dtype=np.float32)
    ys, xs = np.mgrid[0:40, 0:40]
    r = np.hypot(xs - 20, ys - 20)
    ring = (r >= 8) & (r <= 15)
    roi[ring] = 1.0
    fit = fit_ellipse_guarded(roi)
    assert fit["flag"] is None, f"annulus should fit, got {fit}"
    # Major axis length reflects the OUTER extent of the ring (~30 px),
    # but area is ring area (not disc area). This is a known numerical
    # quirk of the 2*sqrt(area/pi) diameter proxy — document it.
    assert fit["major"] >= 25.0, f"major axis should span ring, got {fit['major']}"
    # If someone expects diameter == 30 (outer diameter), they'll be
    # surprised: area-based diameter is much smaller than major-axis length.
    area_diam = 2.0 * np.sqrt(fit["area"] / np.pi)
    assert area_diam < fit["major"], (
        "baseline: donut PSF area-based diameter < major-axis length "
        f"({area_diam:.1f} vs {fit['major']:.1f}). If M4 uses area-diameter, "
        "filled vs donut discs compare incorrectly."
    )


# ============================================================================
# Attack C — M7 reference disc degenerate
# ============================================================================
#
# Spec M7 uses disc #3 as reference. If reference ROI is empty (renderer bug)
# M7 must flag, not divide by zero or a tiny value. Existing tests cover
# missing wrap disc but not missing reference.


def test_round2_m7_missing_reference_disc_flagged():
    """Wrap disc present, reference disc #3 rendered as zeros → reference_no_blob."""
    h = w = 1024
    mono = rasterize_disc(h, w, 720, 512, radius=10.0)  # disc #11 only
    # disc #3 (800, 512) region is all zeros
    rgb = mono_rgb(mono)
    ann = bubble_grid_annotation_stub(standard_grid_discs())
    result = m7_wrap_probe(rgb, ann)
    assert result["finite_ok"] is True
    assert result["support_px"] >= 20
    assert np.isnan(result["diameter_ratio"])
    # Flag must name the reference failure so TL probe can diagnose.
    assert "reference" in (result["flag"] or ""), (
        f"expected reference_* flag, got {result['flag']}"
    )


def test_round2_m7_tiny_reference_disc_does_not_explode_ratio():
    """Reference disc at minimum support (~21 px) → ratio is finite, not 1/tiny=huge."""
    h = w = 1024
    # Wrap disc: big blob (diameter ~20)
    wrap_blob = rasterize_disc(h, w, 720, 512, radius=10.0)
    # Reference disc: small but barely-passing blob (area ~25 px)
    ref_blob = rasterize_disc(h, w, 800, 512, radius=2.9)
    mono = np.clip(wrap_blob + ref_blob, 0.0, 1.0)
    rgb = mono_rgb(mono)
    ann = bubble_grid_annotation_stub(standard_grid_discs())
    result = m7_wrap_probe(rgb, ann)
    # Either ratio is finite (and large) with no crash, or flagged low_support.
    if result["flag"] is None:
        assert np.isfinite(result["diameter_ratio"])
        # Ratio will be large (wrap_diam ~ 20, ref_diam ~ 5.6), but not inf.
        assert result["diameter_ratio"] > 1.5
    else:
        assert "reference" in result["flag"] or "low_support" in result["flag"]


# ============================================================================
# Attack D — M4 monotonicity under measurement noise
# ============================================================================


def test_round2_m4_monotonic_with_noise_still_high_rho():
    """Diameters monotonic-with-noise (±1 px) → spearman should stay near 1."""
    # Craft a bubble_grid-shaped mono with specific diameters per mid-column disc.
    # Mid-column discs: #3 (0.98m), #4 (3.74m), #5 (1.37m), #6 (2.68m).
    # |depth-0.70| = 0.28, 3.04, 0.67, 1.98. Sort: 0.28 < 0.67 < 1.98 < 3.04.
    # So target diameters (sorted by |depth-0.70|) monotonic increasing:
    # disc #3 → smallest, disc #5 → 2nd, disc #6 → 3rd, disc #4 → largest.
    targets = {3: 6.0, 5: 7.0, 6: 8.0, 4: 9.0}
    h = w = 1024
    mono = np.zeros((h, w), dtype=np.float32)
    discs = standard_grid_discs()
    for d in discs:
        if d["idx"] in targets:
            cx, cy = d["center_xy_px"]
            # Add ±1 px noise to the radius
            noise = (d["idx"] * 0.37 - 0.5)  # deterministic "noise"
            r = targets[d["idx"]] + 0.5 * noise
            mono = np.clip(
                mono + rasterize_disc(h, w, cx, cy, r), 0.0, 1.0
            )
    rgb = mono_rgb(mono)
    ann = bubble_grid_annotation_stub(discs)
    rhos, flags = m4_diameter_depth_monotonicity(rgb, ann)
    # Mid column should be strongly monotonic (ρ ≥ 0.8 even with noise).
    assert flags["mid"] is None, f"mid column flagged: {flags}"
    assert rhos["mid"] >= 0.8, (
        f"noisy-monotonic mid column should have ρ ≥ 0.8, got {rhos['mid']}"
    )


# ============================================================================
# Attack E — score() with empty metrics tuple
# ============================================================================
# Spec §5 line 199 explicitly cut this as "solving a problem we don't have".
# Confirm the harness does not crash — returns only _meta.


def test_round2_score_empty_metrics_returns_meta_only():
    rgb = mono_rgb(np.zeros((64, 64), dtype=np.float32))
    depth = np.full((64, 64), 0.70, dtype=np.float32)
    ann = bubble_grid_annotation_stub(standard_grid_discs())
    out = score(rgb, depth, ann, input_rgb=rgb, metrics=())
    # Only _meta should be present. No crash.
    assert set(out.keys()) == {"_meta"}
    assert out["_meta"]["metrics_requested"] == []


# ============================================================================
# Attack F — numpy scalar leaks in nested M7 / M3 / M4 dicts
# ============================================================================


def test_round2_py_coerces_nested_numpy_scalars():
    """_py must coerce numpy scalars in nested dicts AND lists."""
    nasty = {
        "scalar_f32": np.float32(1.5),
        "scalar_i64": np.int64(7),
        "scalar_bool": np.bool_(True),
        "nested": {
            "arr": np.array([1, 2, 3], dtype=np.int32),
            "list": [np.float64(2.5), np.int16(3), {"inner": np.bool_(False)}],
        },
    }
    out = _py(nasty)
    # json.dumps is the litmus test: fails on numpy scalars.
    s = json.dumps(out)
    # Round-trip and check types.
    rt = json.loads(s)
    assert isinstance(rt["scalar_f32"], float)
    assert isinstance(rt["scalar_i64"], int)
    assert isinstance(rt["scalar_bool"], bool)
    assert isinstance(rt["nested"]["arr"], list)
    assert isinstance(rt["nested"]["list"][2]["inner"], bool)


def test_round2_score_output_json_dumpable_with_m3_list():
    """Full score() report with bubble_grid must be json.dumps()-safe,
    including the M3 per-disc list (which contains floats inside dicts)."""
    h = w = 1024
    mono = rasterize_disc(h, w, 904, 904, radius=12.0)  # corner SE only
    rgb = mono_rgb(mono)
    depth = np.full((h, w), 0.70, dtype=np.float32)
    ann = bubble_grid_annotation_stub(standard_grid_discs())
    report = score(rgb, depth, ann, input_rgb=rgb, metrics=("M3",))
    s = json.dumps(report)
    assert isinstance(s, str)


# ============================================================================
# Attack G — annotation polarity: wrap_applied hand-edited on wrong disc
# ============================================================================
#
# If a user hand-edits annotation so disc #3 has wrap_applied=true, what
# happens? Spec says M2/M3/M4 filter wrap_applied discs out (correct); M7
# iterates `wrap_applied=true` discs and reports the FIRST one. If disc #3 is
# flagged, M7 measures disc #3 as the "wrap probe" — silently wrong.


def test_round2_m7_multiple_wrap_discs_flag_is_fragile():
    """Two wrap-applied discs → 'multiple_wrap_discs' flag ONLY fires on the
    happy path. REAL GAP: in error paths (reference missing, low support,
    etc.), the multiple-wrap signal is silently overwritten.

    Scenario: disc #3 is hand-edited to wrap_applied=true. M7 picks disc #11
    (first in list) as the wrap disc; _find_reference_disc then EXCLUDES
    both #3 and #11, so the reference falls to the next-closest-depth disc
    (#5 at 1.37m, (740,284)). If that disc happens to be rendered (happy
    path), flag='multiple_wrap_discs' fires. If not, some other flag wins.

    Test both paths to document the gap.
    """
    h = w = 1024
    # Minimal 3-disc annotation: two wrap-applied, one reference.
    # Wrap disc A at (720, 512), wrap_depth=0.40 -> effective=1.00m.
    # Wrap disc B at (800, 512), wrap_depth=0.40 -> same effective.
    # Reference at (740, 284), depth=0.98m (closest to effective 1.00m).
    discs = [
        {"idx": 0, "center_xy_px": [740, 284], "r_frac": 0.45, "depth_m": 0.98, "source_diameter_px": 16, "wrap_applied": False},
        {"idx": 1, "center_xy_px": [720, 512], "r_frac": 0.29, "depth_m": 0.40, "source_diameter_px": 16, "wrap_applied": True},
        {"idx": 2, "center_xy_px": [800, 512], "r_frac": 0.40, "depth_m": 0.40, "source_diameter_px": 16, "wrap_applied": True},
    ]

    # Happy path: rasterize wrap_discs[0] (idx=1 at (720,512)) and reference (idx=0).
    m_wrap = rasterize_disc(h, w, 720, 512, radius=10.0)
    m_ref = rasterize_disc(h, w, 740, 284, radius=10.0)
    mono_happy = np.clip(m_wrap + m_ref, 0.0, 1.0)
    ann = bubble_grid_annotation_stub(discs)
    result_happy = m7_wrap_probe(mono_rgb(mono_happy), ann)
    # On the happy path, flag should surface.
    assert result_happy["flag"] == "multiple_wrap_discs", (
        f"happy path: expected 'multiple_wrap_discs', got {result_happy['flag']!r}"
    )

    # Error path: rasterize wrap disc only, no reference.
    # Real gap: 'multiple_wrap_discs' is silently lost.
    mono_err = m_wrap
    result_err = m7_wrap_probe(mono_rgb(mono_err), ann)
    # DOCUMENTED REAL GAP: flag is NOT 'multiple_wrap_discs' on error paths.
    # When this is fixed (e.g. by concatenating flags or promoting
    # multiple_wrap_discs to a hard ValueError), update this assertion.
    assert result_err["flag"] != "multiple_wrap_discs", (
        "If this starts passing, the multiple-wrap-discs signal is now "
        "preserved across error paths — update the test + spec."
    )


# ============================================================================
# Attack H — M2 on edge-cropped ellipse (center disc at the frame edge)
# ============================================================================
#
# B's proposal had an edge-crop guard. If the "on-axis" disc is moved to the
# edge (annotation hand-edited), the 64x64 ROI would clip. The ellipse fit
# would see a truncated blob and report a bogus roundness < 1. There is no
# M2 edge guard in the current impl. Whether this lands as a "real gap"
# depends on how strict we want M2's contract; here we document the baseline.


def test_round2_m2_clipped_on_frame_edge_silently_reports_nonround():
    """Hand-edit center disc to near edge → ROI clips → roundness drops silently.

    The current impl's _crop_roi clamps to frame bounds, so the ROI is
    smaller than 64x64 but still valid; the largest CC inside the partial
    ROI IS the clipped blob, which regionprops fits as an elongated shape.

    Documented as a baseline: if M2 is ever used with a hand-edited
    annotation putting the 'on-axis' disc at a wall, the harness silently
    reports a low roundness with no edge-crop warning. Not a P0 gap
    (spec §2.1 pins disc #0 at (512,512)), but worth surfacing.
    """
    h = w = 1024
    # Put a big disc ~radius 25 with center at (10, 512): half of it falls
    # outside the frame.
    mono = rasterize_disc(h, w, 10, 512, radius=25.0)
    rgb = mono_rgb(mono)
    ann = bubble_grid_annotation_stub(
        [
            {
                "idx": 0,
                "center_xy_px": [10, 512],  # hand-edit: on-axis claimed at edge
                "r_frac": 0.0,
                "depth_m": 0.70,
                "source_diameter_px": 16,
                "wrap_applied": False,
            }
        ]
    )
    val, flag = m2_roundness(rgb, ann)
    # Currently: no edge guard. Value will be < 1.0 (clipped half-disc).
    # This test DOCUMENTS the baseline — adjust expectation if Coder adds
    # edge guard in future round.
    if flag is None:
        # Silent wrong answer.
        assert val < 1.0, f"expected clipped-edge M2 < 1, got {val}"
    else:
        # If a guard was added, any flag value is acceptable.
        assert isinstance(flag, str)


# ============================================================================
# Attack I — M5 on saturated output (documented baseline)
# ============================================================================


def test_round2_m5_saturated_rendered_reports_low_sharpness_silently():
    """Rendered tile clipped to 1.0 over a wide area → gradient energy drops.

    Spec cut the saturation flag. Document that M5 silently returns a
    low ratio, NOT NaN. This is a known gap flagged in §5 cuts, not a
    regression.
    """
    h = w = 1024
    input_mono = np.zeros((h, w), dtype=np.float32)
    # Add a USAF-like tile at (512,512) 320x320 with gradients
    ys, xs = np.mgrid[352:672, 352:672]
    input_mono[352:672, 352:672] = ((xs // 16) % 2).astype(np.float32)
    # Saturated rendered: clip entire tile to 1.0 → no gradient.
    rendered_mono = input_mono.copy()
    rendered_mono[352:672, 352:672] = 1.0
    input_rgb = mono_rgb(input_mono)
    rendered_rgb = mono_rgb(rendered_mono)
    ann = {
        "scene": "usaf_depth",
        "frame_hw": [h, w],
        "focus_distance_m": 0.70,
        "tiles": [
            {
                "idx": 0,
                "bbox_xyxy": [352, 352, 672, 672],
                "depth_m": 0.70,
                "bar_widths_px": [16, 8, 4, 2],
                "group_bboxes_xyxy": [],
            }
        ],
    }
    val, flag = m5_in_focus_sharpness(input_rgb, rendered_rgb, ann)
    # Baseline: saturated render yields M5 ≈ 0 with no saturation flag.
    assert flag is None  # no saturation flag (cut per spec §12)
    assert val < 0.05, f"saturated render should yield M5 near 0, got {val}"


# ============================================================================
# Attack J — CLI end-to-end with all-zero rendered (integration smoke)
# ============================================================================


def _write_npz_pair(path: Path, rgb: np.ndarray, depth: np.ndarray, ann: dict):
    np.savez(str(path), rgb=rgb, depth_m=depth)
    with path.with_suffix(".json").open("w") as fh:
        json.dump(ann, fh)


def test_round2_cli_end_to_end_zero_rendered(tmp_path):
    """CLI with all-zero rendered must emit a well-formed report, not crash."""
    h = w = 256
    # Minimal valid bubble_grid input: one center disc.
    disc = {
        "idx": 0,
        "center_xy_px": [128, 128],
        "r_frac": 0.0,
        "depth_m": 0.70,
        "source_diameter_px": 16,
        "wrap_applied": False,
    }
    mono_in = rasterize_disc(h, w, 128, 128, radius=8.0)
    rgb_in = mono_rgb(mono_in)
    rgb_zero = np.zeros((h, w, 3), dtype=np.float32)
    depth = np.full((h, w), 0.70, dtype=np.float32)
    ann = bubble_grid_annotation_stub([disc])
    ann["frame_hw"] = [h, w]

    in_path = tmp_path / "input.npz"
    rn_path = tmp_path / "rendered.npz"
    out_path = tmp_path / "report.json"
    _write_npz_pair(in_path, rgb_in, depth, ann)
    _write_npz_pair(rn_path, rgb_zero, depth, ann)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "summicron.eval",
            "--input",
            str(in_path),
            "--rendered",
            str(rn_path),
            "--output",
            str(out_path),
            "--metrics",
            "M1,M2",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"CLI crashed on zero rendered. stderr:\n{result.stderr}"
    )
    assert out_path.exists()
    report = json.loads(out_path.read_text())
    # M1: rendered.sum()=0 over non-zero input → ratio=0.0 (NOT NaN, input
    # is not too dark — sum > 1.0 from the disc).
    assert report["M1"] == 0.0 or np.isclose(report["M1"], 0.0)
    assert report["M1__flag"] is None
    # M2: zero rendered → no_blob or input_flat flag.
    assert np.isnan(report["M2"]) or report["M2"] != report["M2"]  # NaN
    assert report["M2__flag"] in ("no_blob", "input_flat")


# ============================================================================
# Attack K — empty discs list on bubble_grid passes score() dispatcher
# ============================================================================


def test_round2_score_empty_discs_list_no_crash():
    """bubble_grid with zero discs: M2/M3/M4 must flag, not crash."""
    h = w = 256
    rgb = mono_rgb(np.zeros((h, w), dtype=np.float32))
    depth = np.full((h, w), 0.70, dtype=np.float32)
    ann = bubble_grid_annotation_stub([])
    report = score(rgb, depth, ann, input_rgb=rgb, metrics=("M2", "M3", "M4", "M7"))
    # Every metric should have a flag indicating the issue.
    assert np.isnan(report["M2"])
    assert report["M2__flag"] in ("no_center_disc", "no_blob", "input_flat")
    # M3 returns empty list (no discs to score), not a flagged scalar.
    assert report["M3"] == [] or report["M3"] == []
    # M4: insufficient_points per band.
    for band in ("center", "mid", "corner"):
        assert report.get(f"M4__flag_{band}") in ("insufficient_points", None)
    # M7 with no wrap disc → scene_has_no_wrap_probe.
    assert report["M7__flag"] == "scene_has_no_wrap_probe"
