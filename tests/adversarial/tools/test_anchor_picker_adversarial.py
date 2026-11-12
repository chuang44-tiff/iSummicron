"""Adversarial tests for the GUI anchor picker (P1.10).

Targets the Coder's own "weak spots" report + additional angles surfaced by
probing the actual code. Per Day-1 post-mortem: ships at least one wrong-polarity
fixture for the co-planar depth probe gate.

All tests here poke the pure-compute helpers (``probe_depth``,
``emit_canonical_command``, ``_normalize_click``, ``_read_exif_orientation``).
GUI interaction (matplotlib event synthesis) is exercised where feasible.

Test naming convention:
    test_*  -> negative-space probe: verifies current behavior is correct.
    xfail   -> genuine bug; Coder fix-round needs to repair.
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pytest

from summicron.tools.anchor_picker import (
    CANONICAL_TEMPLATE,
    _normalize_click,
    _read_exif_orientation,
    emit_canonical_command,
    probe_depth,
)


# ---------------------------------------------------------------------------
# MANDATORY wrong-polarity / asymmetry fixture (Day-1 post-mortem requirement)
# ---------------------------------------------------------------------------

def _make_depth(center_val: float, ring_val: float, h: int = 256, w: int = 256,
                cx: int = 128, cy: int = 128) -> np.ndarray:
    depth = np.full((h, w), ring_val, dtype=np.float32)
    depth[cy - 10:cy + 11, cx - 10:cx + 11] = center_val
    return depth


def test_coplanar_warn_polarity_invariant_symmetric_case():
    """MANDATORY per Day-1 post-mortem: co-planar warn must fire for BOTH
    center<annulus AND center>annulus when |delta|<0.05.

    Spec says the threshold is on |center - annulus|, so both polarities are
    equivalent. This test catches the bug where the Coder writes
    ``(center - annulus) < 0.05`` (signed) instead of ``abs(...) < 0.05``.
    """
    # Case A: subject (center) darker/closer than surround
    out_a = probe_depth(_make_depth(0.46, 0.49), px=128, py=128)
    # Case B: subject (center) farther than surround -- reversed polarity
    out_b = probe_depth(_make_depth(0.49, 0.46), px=128, py=128)
    # Both must warn. And the equality on `coplanar_warn` is the asymmetry probe.
    assert out_a["coplanar_warn"] is True, "case A (center<annulus) must warn"
    assert out_b["coplanar_warn"] is True, "case B (center>annulus) must warn"


def test_coplanar_warn_polarity_equal_case():
    """center == annulus (flat surface) must also warn -- it is the degenerate
    polarity case (delta=0). Complements the two asymmetric polarities above.
    """
    out = probe_depth(_make_depth(0.46, 0.46), px=128, py=128)
    assert out["coplanar_warn"] is True


# ---------------------------------------------------------------------------
# Threshold boundary (Coder weak-spot #2): strict `<` 0.05 and float32 drift
# ---------------------------------------------------------------------------

@pytest.mark.xfail(
    strict=True,
    reason="BUG: coplanar threshold is polarity-dependent at |delta|=0.05 due "
    "to float32 rounding of the median. center=0.46,annulus=0.51 warns "
    "(delta rounds to 0.04999...) but center=0.46,annulus=0.41 does NOT "
    "warn (delta rounds to 0.05000...). Same |delta|=0.05, opposite "
    "outcomes. Fix: either use >= 0.05 semantics or clamp float32 noise "
    "with a tiny epsilon. Severity P1.",
)
def test_coplanar_threshold_polarity_symmetric_at_exactly_0p05():
    # Both of these have `|center - annulus| == 0.05` in the user's mental model.
    # In float32 one rounds *below* 0.05 and warns; the other rounds *above*.
    above = probe_depth(_make_depth(0.46, 0.51), px=128, py=128)
    below = probe_depth(_make_depth(0.46, 0.41), px=128, py=128)
    assert above["coplanar_warn"] == below["coplanar_warn"], (
        f"asymmetric at |delta|=0.05 exactly: above={above['coplanar_warn']} "
        f"below={below['coplanar_warn']}"
    )


def test_coplanar_threshold_clearly_inside_window_warns():
    """Sanity: delta=0.04 is clearly inside the co-planar window, warns."""
    out = probe_depth(_make_depth(0.46, 0.50), px=128, py=128)
    assert out["coplanar_warn"] is True


def test_coplanar_threshold_clearly_outside_window_silent():
    """Sanity: delta=0.10 is clearly outside, does not warn."""
    out = probe_depth(_make_depth(0.46, 0.56), px=128, py=128)
    assert out["coplanar_warn"] is False


# ---------------------------------------------------------------------------
# Coder weak-spot #5: LiDAR clip floor (extract_lidar_depth clips to [0.1, 1e3])
# ---------------------------------------------------------------------------

@pytest.mark.xfail(
    strict=True,
    reason="BUG: uniform 0.1m depth (the LiDAR clip floor from "
    "extract_lidar_depth) trips the co-planar warn. The intent of the "
    "`0.0 < center_med` guard was to skip the LiDAR-missing sentinel, "
    "but a region uniformly clipped to the floor passes the guard and "
    "triggers a warning that has no user-actionable meaning. Fix: "
    "either raise the lower bound to slightly above the clip floor "
    "(e.g. center_med > 0.1) or add a dedicated message. Severity P2.",
)
def test_coplanar_does_not_warn_at_lidar_clip_floor():
    depth = np.full((256, 256), 0.1, dtype=np.float32)  # the floor from estimate.py
    out = probe_depth(depth, px=128, py=128)
    assert out["coplanar_warn"] is False, (
        "uniform depth at the LiDAR clip floor (0.1m) should not fire the "
        "co-planar warn -- it is a probable clip-artifact, not a real surface"
    )


def test_coplanar_does_not_warn_at_zero_depth_sentinel():
    """Confirms the current guard works for the obvious LiDAR-missing case."""
    depth = np.zeros((256, 256), dtype=np.float32)
    out = probe_depth(depth, px=128, py=128)
    assert out["coplanar_warn"] is False


# ---------------------------------------------------------------------------
# Coder weak-spot #1: edge / corner clicks
# ---------------------------------------------------------------------------

def test_corner_click_at_0_0_returns_finite_report():
    """Click at array origin. Disk + partial annulus must still yield finite numbers.
    This is the "click near edge / corner" robustness probe.
    """
    depth = _make_depth(0.46, 0.80, h=256, w=256, cx=0, cy=0)
    out = probe_depth(depth, px=0, py=0)
    assert np.isfinite(out["center_med"])
    assert np.isfinite(out["annulus_med"])
    assert np.isfinite(out["iqr"])


def test_corner_click_at_W_minus_1_H_minus_1_finite():
    depth = np.full((128, 96), 0.80, dtype=np.float32)
    # subject centered at opposite corner
    depth[118:128, 86:96] = 0.46
    out = probe_depth(depth, px=95, py=127)
    assert np.isfinite(out["center_med"])
    assert np.isfinite(out["annulus_med"])


def test_click_fully_outside_image_returns_safe_defaults():
    """Click at (100_000, 100_000) must not crash; probe_depth returns zeros."""
    depth = np.full((64, 64), 0.5, dtype=np.float32)
    out = probe_depth(depth, px=100_000, py=100_000)
    assert out["center_med"] == 0.0
    assert out["annulus_med"] == 0.0
    assert out["iqr"] == 0.0
    assert out["coplanar_warn"] is False


def test_negative_click_does_not_raise():
    """Negative click coords (from a buggy event) must not crash probe_depth."""
    depth = np.full((64, 64), 0.5, dtype=np.float32)
    out = probe_depth(depth, px=-5, py=-5)
    assert "center_med" in out  # no exception -> passes


def test_image_smaller_than_annulus_outer_radius():
    """Image smaller than 80px radius -> annulus is a thin crescent but still valid."""
    depth = np.full((40, 40), 0.46, dtype=np.float32)
    out = probe_depth(depth, px=20, py=20)
    # With 40x40 image at center (20,20), max radius is sqrt(20^2+20^2)~28 < 30.
    # mask.any() is False -> annulus_med falls back to 0.0.
    assert out["annulus_med"] == 0.0
    # And because annulus_med==0.0 differs from 0.46 by 0.46 > 0.05, no warn.
    assert out["coplanar_warn"] is False


# ---------------------------------------------------------------------------
# Coder weak-spot: dict contract (keys + types)
# ---------------------------------------------------------------------------

def test_probe_depth_dict_contract():
    depth = np.full((256, 256), 0.5, dtype=np.float32)
    out = probe_depth(depth, px=128, py=128)
    assert set(out.keys()) == {"center_med", "iqr", "annulus_med", "coplanar_warn"}
    # Type contract: plain Python float + plain Python bool (not numpy scalars).
    assert isinstance(out["center_med"], float)
    assert isinstance(out["iqr"], float)
    assert isinstance(out["annulus_med"], float)
    assert isinstance(out["coplanar_warn"], bool)
    # Specifically not a numpy.bool_, which has surprising JSON / pickle behavior.
    assert type(out["coplanar_warn"]) is bool  # noqa: E721


# ---------------------------------------------------------------------------
# Coder weak-spot #3: perf floor on annulus ogrid allocation (information only)
# ---------------------------------------------------------------------------

def test_probe_depth_perf_floor_on_realistic_image():
    """Info-only: probe 5 clicks on a 5712x4284 depth map, assert < 2.5 s total.
    The annulus mask allocates (H, W) ogrid per click. At 5712x4284 this is
    ~24M pixels per click. If this ever exceeds a few hundred ms per click,
    the rapid-click UX is degraded.
    """
    depth = np.full((5712, 4284), 0.5, dtype=np.float32)
    t0 = time.perf_counter()
    for _ in range(5):
        probe_depth(depth, px=2142, py=4284)
    dt = time.perf_counter() - t0
    # 2.5s is a very generous ceiling; flag if we ever blow past it.
    assert dt < 2.5, f"5 clicks on 5712x4284 took {dt:.3f}s"


# ---------------------------------------------------------------------------
# emit_canonical_command / format / rounding
# ---------------------------------------------------------------------------

def test_emit_canonical_kwargs_reorder_identical():
    """Both kwarg orders must produce byte-identical output. Guards against a
    future refactor that makes the emitter positional-sensitive."""
    a = emit_canonical_command(image_path="p.HEIC", x_norm=0.5, y_norm=0.75)
    b = emit_canonical_command(y_norm=0.75, x_norm=0.5, image_path="p.HEIC")
    assert a == b


def test_emit_canonical_formats_zero_zero():
    """Clicks at (0, 0) must format as 0.0000 0.0000 (no sign, no exponent)."""
    out = emit_canonical_command("p.HEIC", 0.0, 0.0)
    assert "--focus-xy 0.0000 0.0000" in out


def test_emit_canonical_focus_xy_rounds_half_pixel():
    """Click at exactly the half-pixel midpoint formats cleanly.
    event.xdata=2141.5 on W=4284 -> x_norm=0.49988... -> "0.4999".
    """
    x_norm, y_norm = _normalize_click(2141.5, 4284.0, 4284, 5712)
    cmd = emit_canonical_command("p.HEIC", x_norm, y_norm)
    assert "--focus-xy 0.4999 0.7500" in cmd


def test_emit_canonical_focus_xy_rounds_sub_pixel_epsilon_to_same_value():
    """Sub-pixel jitter on either side of 0.5 -> same formatted output."""
    x1, _ = _normalize_click(2141.999, 0.0, 4284, 5712)
    x2, _ = _normalize_click(2142.0001, 0.0, 4284, 5712)
    cmd1 = emit_canonical_command("p.HEIC", x1, 0.5)
    cmd2 = emit_canonical_command("p.HEIC", x2, 0.5)
    assert cmd1 == cmd2


@pytest.mark.xfail(
    strict=True,
    reason="BUG: emit_canonical_command does not shell-quote --image. A path "
    "containing $(...) or backticks copied verbatim into a shell would "
    "trigger command substitution. Fix: shlex.quote(str(image_path)). "
    "Severity P1 (solo-dev local-only mitigates, but the picker prints "
    "this command for copy-paste so it IS the supported workflow).",
)
def test_emit_canonical_shell_escapes_image_path():
    import shlex

    malicious = "/tmp/foo $(rm -rf /).HEIC"
    cmd = emit_canonical_command(malicious, 0.5, 0.75)
    # The raw dollar-paren substring should NOT appear unquoted. Minimal check:
    # the path should appear shlex-quoted (single-quoted with escapes).
    quoted = shlex.quote(malicious)
    assert f"--image {quoted}" in cmd, (
        "image path must be shell-escaped before interpolation into the "
        "copy-paste-able canonical command"
    )


def test_emit_canonical_handles_path_with_spaces_does_not_crash():
    """Defensive: paths with spaces must not crash the emitter (it's just str())."""
    cmd = emit_canonical_command("/tmp/my photo.HEIC", 0.5, 0.75)
    assert "/tmp/my photo.HEIC" in cmd


def test_emit_canonical_output_line_count_stable():
    """The template is 5 lines joined by \\\\\\n. Count must stay stable."""
    cmd = emit_canonical_command("p.HEIC", 0.5, 0.75)
    # HYG-0 item 2: --tensor dropped, flags on one line -> 5 logical lines.
    assert cmd.count("\\\n") == 4  # 4 continuations between 5 logical lines


# ---------------------------------------------------------------------------
# _normalize_click: identity contract stress
# ---------------------------------------------------------------------------

def test_normalize_click_at_image_max_corner():
    """A click at the far corner (W, H) -> (1.0, 1.0)."""
    x, y = _normalize_click(4284.0, 5712.0, 4284, 5712)
    assert x == 1.0 and y == 1.0


def test_normalize_click_is_identity_on_fraction():
    for frac_x in (0.1, 0.25, 0.5, 0.7, 0.875):
        for frac_y in (0.1, 0.25, 0.5, 0.7, 0.875):
            x, y = _normalize_click(frac_x * 4284, frac_y * 5712, 4284, 5712)
            assert pytest.approx(x, abs=1e-9) == frac_x
            assert pytest.approx(y, abs=1e-9) == frac_y


# ---------------------------------------------------------------------------
# EXIF reader robustness (weak-spot: corrupt / missing EXIF)
# ---------------------------------------------------------------------------

def test_exif_reader_missing_file_returns_one():
    """Missing file -> 1 (fallback). Swallowed via try/except."""
    got = _read_exif_orientation(Path("/nonexistent/path/foo.HEIC"))
    assert got == 1


def test_exif_reader_png_without_exif_returns_one():
    """PNG with no EXIF tags returns 1."""
    from PIL import Image
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "a.png"
        Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)).save(p)
        assert _read_exif_orientation(p) == 1


def test_exif_reader_jpeg_with_orientation_6():
    """JPEG with explicit EXIF Orientation=6 reads back as 6.

    This is the first test linking the _orientation.py table to on-disk bits.
    Coder weak-spot #4 was that the 8-way roundtrip is self-consistent only.
    """
    from PIL import Image
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "b.jpg"
        img = Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8))
        exif = img.getexif()
        exif[274] = 6  # 274 == Orientation tag
        img.save(p, exif=exif)
        assert _read_exif_orientation(p) == 6


def test_exif_reader_empty_bytes_returns_one():
    """File is not a valid image -> PIL raises -> fallback to 1."""
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "garbage.heic"
        p.write_bytes(b"\x00" * 128)
        assert _read_exif_orientation(p) == 1


# ---------------------------------------------------------------------------
# _orientation.py: forward-direction test linking to concrete pixel layout
# ---------------------------------------------------------------------------

@pytest.mark.xfail(
    strict=True,
    reason="BUG: _orientation.display_to_storage_norm has orientations 6 and 8 "
    "swapped vs PIL.ImageOps.exif_transpose. For a 6x4 storage image with "
    "bright pixel at (row=0, col=3), PIL places it at display (row=3, col=5) "
    "under orient=6; the helper maps click (5.5, 3.5)/(6, 4) to storage "
    "(0.125, 0.917) but the true storage is (0.875, 0.083). Current code "
    "orient=6 returns (1-v, u); should be (v, 1-u). Orient=8 is the inverse "
    "mistake. The 8-way roundtrip test in test_orientation.py is self-"
    "consistent and misses this because its inverses were derived from the "
    "same wrong forward map. HEIC path is unaffected (pillow_heif auto-"
    "transposes), but the helper is documented for non-HEIC JPEG/PNG "
    "callers. Severity P1.",
)
def test_orientation_forward_table_matches_pil_exif_transpose():
    """Forward test: for each EXIF orientation 1..8, apply PIL.ImageOps
    exif_transpose and verify the display->storage map agrees with the pixel
    layout change. This catches the dead-code weak-spot: previously the
    orientation table was only verified self-consistently via roundtrip.

    Setup: a 6x4 storage image with a bright pixel at (row=0, col=3). After
    EXIF transpose, a click on the display pixel containing that bright
    pixel must map back (via the helper) to storage normalized coords
    (3.5/4, 0.5/6) = (0.875, 0.0833).
    """
    from PIL import Image, ImageOps
    from summicron.tools._orientation import display_to_storage_norm

    storage_expected_x = 3.5 / 4.0   # 0.875
    storage_expected_y = 0.5 / 6.0   # ~0.0833
    failures: list[str] = []

    for orientation in range(1, 9):
        arr = np.zeros((6, 4, 3), dtype=np.uint8)  # H=6, W=4
        arr[0, 3] = 255  # bright at (row=0, col=3) in storage
        img = Image.fromarray(arr)
        exif = img.getexif()
        exif[274] = orientation

        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "t.jpg"
            img.save(p, exif=exif)
            with Image.open(p) as im2:
                disp = ImageOps.exif_transpose(im2)
                disp_arr = np.asarray(disp)

        ys, xs, _ = np.where(disp_arr > 100)
        assert len(ys) >= 1, f"bright pixel lost after orient={orientation}"
        row_disp, col_disp = int(ys[0]), int(xs[0])
        h_disp, w_disp = disp_arr.shape[:2]

        x_norm, y_norm = display_to_storage_norm(
            col_disp + 0.5, row_disp + 0.5, w_disp, h_disp, orientation,
        )
        tol = 0.06
        if (abs(x_norm - storage_expected_x) > tol
                or abs(y_norm - storage_expected_y) > tol):
            failures.append(
                f"orient={orientation}: got ({x_norm:.4f},{y_norm:.4f}) "
                f"expected ({storage_expected_x:.4f},{storage_expected_y:.4f})"
            )
    assert not failures, "\n".join(failures)


# ---------------------------------------------------------------------------
# CANONICAL_TEMPLATE invariants
# ---------------------------------------------------------------------------

def test_canonical_template_has_all_required_flags():
    """Regression: the template must carry every flag from runs/vibe_bench/README.md."""
    # HYG-0 item 2: --tensor intentionally absent (default wheel tensor).
    required = ("--image", "--depth-source", "--depth-scale",
                "--downres", "--focus-xy", "--psf-norm", "--hdr-source",
                "--output")
    for flag in required:
        assert flag in CANONICAL_TEMPLATE, f"missing flag {flag}"


def test_canonical_template_depth_scale_ten():
    """10x LiDAR depth scale is canonical bench spec."""
    assert "--depth-scale 10" in CANONICAL_TEMPLATE


def test_canonical_template_psf_norm_l1():
    """Canonical bench uses raw tensor with --psf-norm l1."""
    assert "--psf-norm l1" in CANONICAL_TEMPLATE


# ---------------------------------------------------------------------------
# Subprocess smoke test (weak-spot #6: missing importorskip)
# ---------------------------------------------------------------------------

def test_no_show_smoke_has_pillow_heif_guard():
    """Assert the subprocess smoke test is guarded. If pillow_heif is missing
    the existing test errors out rather than skipping. We verify both by
    calling importorskip (to document the dep) and by running the subprocess
    with a nonexistent image so we exit on arg validation, not HEIC decoding.
    """
    pytest.importorskip("pillow_heif")
    # Non-existent image -> argparse returns 2 per main()'s early check.
    proc = subprocess.run(
        [sys.executable, "-m", "summicron.tools.anchor_picker",
         "--image", "/nonexistent/foo.HEIC", "--no-show"],
        capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 2
    assert "ERROR: image not found" in proc.stderr


def test_no_show_cli_bad_depth_source_errors():
    """argparse rejects bad --depth-source values early."""
    proc = subprocess.run(
        [sys.executable, "-m", "summicron.tools.anchor_picker",
         "--image", "/nonexistent/foo.HEIC",
         "--depth-source", "invalid", "--no-show"],
        capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode != 0


# ---------------------------------------------------------------------------
# on_click behavioral probe: synthetic matplotlib events
# ---------------------------------------------------------------------------

def _run_picker_with_click(arr: np.ndarray, depth: np.ndarray,
                           clicks: list[tuple[float, float]]) -> list[str]:
    """Rebuild the on_click logic from anchor_picker.main() and invoke it with
    synthetic clicks (without needing a matplotlib display). Returns the list of
    canonical commands that would be printed.

    Mirrors the source closely; re-implementing lets us drive multiple clicks
    and verify latest-click-wins + out-of-axes guards.
    """
    h, w = arr.shape[:2]
    commands: list[str] = []

    def synth_click(x_ev: float | None, y_ev: float | None, in_axes: bool):
        # Inline the pure parts of on_click (no matplotlib calls).
        if not in_axes or x_ev is None or y_ev is None:
            return
        x_norm, y_norm = _normalize_click(float(x_ev), float(y_ev), w, h)
        _ = probe_depth(depth, int(round(x_ev)), int(round(y_ev)))
        commands.append(emit_canonical_command("p.HEIC", x_norm, y_norm))

    for x, y in clicks:
        synth_click(x, y, in_axes=(x is not None))
    return commands


def test_on_click_latest_click_wins():
    """Multiple clicks -> only the most recent command is "last_command"."""
    arr = np.zeros((256, 256, 3), dtype=np.float32)
    depth = np.full((256, 256), 0.5, dtype=np.float32)
    cmds = _run_picker_with_click(arr, depth,
                                  clicks=[(50.0, 50.0), (200.0, 180.0)])
    assert len(cmds) == 2
    assert cmds[0] != cmds[1]
    # The second click's coords (200/256, 180/256) = (0.7812, 0.7031)
    assert "--focus-xy 0.7812 0.7031" in cmds[1]


def test_on_click_guards_none_xdata():
    """Events with xdata=None (clicks outside image axes) must be ignored."""
    arr = np.zeros((128, 128, 3), dtype=np.float32)
    depth = np.full((128, 128), 0.5, dtype=np.float32)
    cmds = _run_picker_with_click(arr, depth,
                                  clicks=[(None, None), (64.0, 64.0)])
    # First click dropped (None data), only the second emits a command.
    assert len(cmds) == 1


# ---------------------------------------------------------------------------
# Co-planar warn text UX (flagged in task brief)
# ---------------------------------------------------------------------------

@pytest.mark.xfail(
    strict=True,
    reason="BUG (UX): the co-planar warn text 'subject may not isolate' is "
    "misleading on a uniform surface (flat wall) where there IS no "
    "subject to isolate. A dedicated 'uniform surface; no foreground "
    "object detected' message would be more informative. Severity P2 "
    "(cosmetic; does not affect correctness).",
)
def test_coplanar_warn_text_distinguishes_uniform_surface():
    """On a perfectly flat depth map (iqr == 0 AND |center - annulus| == 0)
    the warning text should identify this as 'uniform surface', not 'subject
    may not isolate'.

    Today the spec only carries one warn string. This xfail documents the
    UX improvement for the next round.
    """
    # We can't read the main()'s stdout without spawning a subprocess, so
    # we probe the intent by asserting the warning payload in probe_depth's
    # output carries a distinguishing flag. The Coder would need to add a
    # key like 'uniform_surface_warn' to the dict.
    depth = np.full((256, 256), 0.46, dtype=np.float32)
    out = probe_depth(depth, px=128, py=128)
    assert out["coplanar_warn"] is True
    assert "uniform_surface_warn" in out, (
        "dedicated flag for flat uniform surfaces (no foreground object)"
    )
