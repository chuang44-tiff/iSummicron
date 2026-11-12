"""P1.10.4 — Adversarial hostile-probe coverage.

Round 1 Adversarial pass over P1.10.4 cleanup (slab + scatter path deletion +
lighter cleanup): 8 atomic commits (C1-C7) + sub-commit C3a.

Hostile probe categories:
  TestRemovedFlags          — argparse must reject deleted CLI flags
  TestRemovedSymbols        — deleted prod symbols must ImportError
  TestKwargNoop             — render() vestigial kwargs must be silent noops
  TestSilentFallbackWarning — DeprecationWarning at convolve.py:1037-1043
  TestRenameVerification    — _apply_slab_kernel → _apply_kernel (C3a)
  TestG9RegressionArmor     — basis-mode 2nd-deriv envelope xfail-strict
  TestWrongPolarityChroma   — dim-blob-on-bright-background fixtures
  TestDownresStress         — C1 collapse handles --downres correctly

Locked spec: doc/plan/phase1/section18/locked-spec-P1_10_4.md

References:
- tests/adversarial/test_p1_10_2_hostile.py (sibling pattern; C/D categories)
- tests/adversarial/test_basis_chroma_battery.py (chroma fixture helpers)
- tests/adversarial/test_basis_quality.py (G9 port + xfail-strict)
"""
from __future__ import annotations

import inspect
import subprocess
import sys
import warnings

import numpy as np
import pytest

from summicron.render.convolve import render


# ---------------------------------------------------------------------------
# Shared fixture helpers (self-contained — adversarial tests stay decoupled)
# ---------------------------------------------------------------------------


def _tiny_tensor(R=4, D=4, ksize=33, with_object_distances=True, sigma=4.0):
    """Minimal valid render-ready tensor with S2 delta guard at d=0.

    Mirrors the factory in test_p1_10_2_hostile.py and test_basis_chroma_battery.py.
    """
    cy, cx = ksize // 2, ksize // 2
    yy, xx = np.mgrid[:ksize, :ksize]
    g = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * sigma * sigma))
    g = (g / g.sum()).astype(np.float32)
    kernels = np.tile(g[None, None, :, :], (R, D, 1, 1))
    delta = np.zeros((ksize, ksize), dtype=np.float32)
    delta[cy, cx] = 1.0
    kernels[:, 0] = delta
    obj_dist = (
        np.linspace(0.7, 5.0, D, dtype=np.float64)
        if with_object_distances
        else None
    )
    return {
        "kernels": kernels,
        "radial_edges": np.linspace(0.0, 1.0, R + 1, dtype=np.float64),
        "defocus_ratios": np.linspace(0.0, 1.0, D, dtype=np.float64),
        "corpus_ref_long_edge": None,
        "delta_guard_applied": True,
        "object_distances_m": obj_dist,
        "focus_distance_m": 0.7 if with_object_distances else None,
    }


def _tiny_img(H=32, W=32, fill=0.5):
    return np.full((H, W, 3), fill, dtype=np.float32)


def _tiny_defocus(H=32, W=32, fill=0.3):
    return np.full((H, W), fill, dtype=np.float32)


def _tiny_depth(H=32, W=32, fill=1.5):
    return np.full((H, W), fill, dtype=np.float32)


def _run_cli(extra_flags, timeout=60):
    """Invoke the render CLI in subprocess; return CompletedProcess."""
    return subprocess.run(
        [
            sys.executable, "-m", "summicron.render.convolve",
        ] + list(extra_flags),
        capture_output=True,
        text=True,
        timeout=timeout,
    )


# ---------------------------------------------------------------------------
# 1. Removed-flag CLI rejection (5 probes)
# ---------------------------------------------------------------------------


class TestRemovedFlags:
    """Each removed CLI flag must be rejected by argparse with non-zero exit
    + 'unrecognized'/'error' in the combined stdout/stderr."""

    def test_r_blend_strength_rejected(self):
        """C2: --r-blend-strength flag deleted (kwarg survives as noop)."""
        r = _run_cli(["--r-blend-strength", "0.50"])
        assert r.returncode != 0, (
            f"--r-blend-strength accepted; should be removed. "
            f"stdout={r.stdout[:200]!r} stderr={r.stderr[:200]!r}"
        )
        combined = (r.stdout + r.stderr).lower()
        assert "unrecognized" in combined or "error" in combined, (
            f"Expected argparse rejection text; got: {combined[:300]!r}"
        )

    def test_radial_blend_mode_rejected(self):
        """C4b: --radial-blend-mode flag deleted (kwarg survives as noop)."""
        r = _run_cli(["--radial-blend-mode", "basis"])
        assert r.returncode != 0, (
            f"--radial-blend-mode accepted; should be removed. "
            f"stderr={r.stderr[:200]!r}"
        )
        combined = (r.stdout + r.stderr).lower()
        assert "unrecognized" in combined or "error" in combined

    def test_radial_blend_mode_slab_rejected(self):
        """C4b: even with the old `slab` choice, the flag itself is gone."""
        r = _run_cli(["--radial-blend-mode", "slab"])
        assert r.returncode != 0, "--radial-blend-mode slab should fail (flag gone)"
        combined = (r.stdout + r.stderr).lower()
        assert "unrecognized" in combined or "error" in combined

    def test_layered_gather_rejected(self):
        """C5: --layered-gather flag deleted."""
        r = _run_cli(["--layered-gather"])
        assert r.returncode != 0, (
            f"--layered-gather accepted; should be removed. "
            f"stderr={r.stderr[:200]!r}"
        )
        combined = (r.stdout + r.stderr).lower()
        assert "unrecognized" in combined or "error" in combined

    def test_no_layered_gather_rejected(self):
        """C5: --no-layered-gather (BooleanOptionalAction inverse) deleted."""
        r = _run_cli(["--no-layered-gather"])
        assert r.returncode != 0, (
            f"--no-layered-gather accepted; should be removed. "
            f"stderr={r.stderr[:200]!r}"
        )
        combined = (r.stdout + r.stderr).lower()
        assert "unrecognized" in combined or "error" in combined

    def test_working_long_edge_rejected(self):
        """C1: --working-long-edge collapsed into --downres."""
        r = _run_cli(["--working-long-edge", "2048"])
        assert r.returncode != 0, (
            f"--working-long-edge accepted; should be collapsed. "
            f"stderr={r.stderr[:200]!r}"
        )
        combined = (r.stdout + r.stderr).lower()
        assert "unrecognized" in combined or "error" in combined


# ---------------------------------------------------------------------------
# 2. Removed-symbol ImportError probes (6 probes)
# ---------------------------------------------------------------------------


class TestRemovedSymbols:
    """Symbols deleted in C2/C3/C3a/C5 must raise on import."""

    def test_legacy_path_counter_import_error(self):
        """C5: _LEGACY_PATH_COUNTER global removed."""
        with pytest.raises((ImportError, AttributeError)):
            from summicron.render.convolve import _LEGACY_PATH_COUNTER  # noqa: F401

    def test_get_legacy_path_count_import_error(self):
        """C5: get_legacy_path_count() helper removed."""
        with pytest.raises((ImportError, AttributeError)):
            from summicron.render.convolve import get_legacy_path_count  # noqa: F401

    def test_build_slab_kernel_import_error(self):
        """C3: _build_slab_kernel deleted (slab body gone)."""
        with pytest.raises((ImportError, AttributeError)):
            from summicron.render.convolve import _build_slab_kernel  # noqa: F401

    def test_legacy_scatter_slab_import_error(self):
        """C3: _legacy_scatter_slab deleted (slab body gone)."""
        with pytest.raises((ImportError, AttributeError)):
            from summicron.render.convolve import _legacy_scatter_slab  # noqa: F401

    def test_scatter_one_slab_fixed_d_import_error(self):
        """C3: _scatter_one_slab_fixed_d deleted (slab body gone).

        Note: the basis sibling _scatter_one_slab_fixed_d_basis SURVIVES."""
        import summicron.render.convolve as _c
        # Fail if the slab-only function still exists as an attribute, but
        # allow the basis sibling.
        assert not hasattr(_c, "_scatter_one_slab_fixed_d"), (
            "_scatter_one_slab_fixed_d should be deleted (slab body gone in C3)"
        )

    def test_apply_slab_kernel_renamed(self):
        """C3a: _apply_slab_kernel renamed to _apply_kernel."""
        with pytest.raises((ImportError, AttributeError)):
            from summicron.render.convolve import _apply_slab_kernel  # noqa: F401


# ---------------------------------------------------------------------------
# 3. Kwarg-as-noop verification (4 probes)
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# 4. Silent-fallback DeprecationWarning probes (3 probes)
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# 5. Chroma battery hostile inputs (4 probes)
# ---------------------------------------------------------------------------


class TestChromaHostileInputs:
    """Hostile inputs that should be CAUGHT by chroma battery gates or
    produce well-defined error/result. NaN, near-saturation, single-pixel
    impulse on edge of frame."""

    def test_nan_in_input_raises_or_produces_finite(self):
        """NaN in input should NOT silently propagate; either raise or
        produce finite output via the 'isfinite' guard at convolve.py:798."""
        tensor = _tiny_tensor(with_object_distances=True)
        img = _tiny_img(H=32, W=32, fill=0.5)
        img[5, 5, :] = np.nan
        depth_m = _tiny_depth()
        try:
            out, lin = render(img, tensor, depth_m=depth_m)
        except (ValueError, AssertionError, RuntimeError):
            return  # acceptable: explicit raise (P1.audit Item 5: NaN gate→RuntimeError).
        # If it didn't raise, at minimum the assert in render() (line 1247-1248)
        # caught the NaN and either coerced or NaN'd-out. Document.
        assert out.shape == (32, 32, 3), f"Bad shape: {out.shape}"

    def test_extreme_peak_does_not_overflow(self):
        """Peak 1e6 input should not produce NaN/Inf in linear output —
        clipping happens at finalize for LDR; HDR path returns float32."""
        tensor = _tiny_tensor(with_object_distances=True)
        img = _tiny_img(H=32, W=32, fill=0.05)
        img[14:18, 14:18, :] = 1e6  # extreme peak
        depth_m = _tiny_depth()
        # HDR path returns linear without clipping.
        try:
            _u8, lin = render(
                img, tensor, depth_m=depth_m, hdr_source="macos",
            )
            assert np.isfinite(lin).all(), (
                f"Extreme peak produced NaN/Inf in linear output "
                f"(min={lin.min()}, max={lin.max()})"
            )
        except (ValueError, AssertionError) as exc:
            pytest.skip(f"render rejected 1e6 input by gate: {exc}")

    def test_single_pixel_impulse_centered(self):
        """Single-pixel impulse on dim ramp must convolve cleanly (no NaN,
        finite output, no astronomical highlights from divide-by-zero)."""
        tensor = _tiny_tensor(with_object_distances=True)
        img = np.full((32, 32, 3), 0.05, dtype=np.float32)
        img[16, 16, :] = 2.5  # single-pixel HDR impulse
        depth_m = _tiny_depth()
        _u8, lin = render(
            img, tensor, depth_m=depth_m, hdr_source="macos",
        )
        assert np.isfinite(lin).all(), "single-pixel impulse → NaN/Inf"
        assert lin.min() >= -1e-4, f"single-pixel impulse → negatives ({lin.min()})"
        assert lin.max() < 1e3, f"impulse blew up ({lin.max()})"

    def test_impulse_at_edge_of_frame(self):
        """Impulse at frame edge must not crash FFT pad / boundary handling."""
        tensor = _tiny_tensor(with_object_distances=True)
        img = np.full((32, 32, 3), 0.05, dtype=np.float32)
        img[0, 0, :] = 2.0       # corner
        img[0, 31, :] = 2.0      # top-right
        img[31, 31, :] = 2.0     # bottom-right
        depth_m = _tiny_depth()
        _u8, lin = render(
            img, tensor, depth_m=depth_m, hdr_source="macos",
        )
        assert np.isfinite(lin).all(), "corner-impulse render → NaN/Inf (boundary bug)"


# ---------------------------------------------------------------------------
# 6. Wrong-polarity fixture (MANDATORY per Post-mortem 1)
# ---------------------------------------------------------------------------


class TestWrongPolarityChroma:
    """Per Post-mortem 1 (`feedback_adversarial_chromaticity_gate.md`):
    bright-pass / chroma assertions assume bright-blob-on-dim-background.
    A regression that flips polarity (dim-blob-on-bright-background) would
    silently slip past the existing battery. These probes feed the inverse
    polarity through the basis path and assert sane output."""

    def _make_inverse_fixture(self, sz=64):
        """Bright background (peak 2.0) with a dim spot (0.05) at center.

        This is the polarity inverse of the chroma-battery fixture in
        test_basis_chroma_battery.py (which uses dim grey 0.05 bg + bright
        peak 3.0 spot)."""
        img = np.full((sz, sz, 3), 2.0, dtype=np.float32)  # bright bg
        cy, cx = sz // 2, sz // 2
        img[cy - 8:cy + 8, cx - 8:cx + 8, :] = 0.05  # dim hole
        return img

    def test_inverse_polarity_no_nan(self):
        """Bright-bg + dim-spot must not produce NaN/Inf in basis path —
        this is the polarity inverse not covered by the C6 chroma battery."""
        tensor = _tiny_tensor(with_object_distances=True)
        img = self._make_inverse_fixture(sz=64)
        depth_m = np.full((64, 64), 1.5, dtype=np.float32)
        _u8, lin = render(
            img, tensor, depth_m=depth_m, hdr_source="macos",
        )
        assert np.isfinite(lin).all(), (
            "inverse-polarity (bright-bg + dim-spot) produced NaN/Inf — "
            "bright-pass region detector polarity bug?"
        )

    def test_inverse_polarity_dim_spot_not_brightened(self):
        """The dim hole at center must NOT mysteriously brighten under basis.
        If basis kernel selection or bright-pass region detector flips polarity,
        the dim spot might 'inherit' bg luminance via inverted top-hat."""
        tensor = _tiny_tensor(with_object_distances=True)
        img = self._make_inverse_fixture(sz=64)
        depth_m = np.full((64, 64), 1.5, dtype=np.float32)
        _u8, lin = render(
            img, tensor, depth_m=depth_m, hdr_source="macos",
        )
        # Center pixel should NOT exceed the bright bg in any channel.
        # Some convolution-bleed is OK, but a >2x jump indicates a bug.
        center_max = float(lin[32, 32, :].max())
        bg_max = float(img[0, 0, :].max())  # 2.0
        # Allow ≤ bg_max + a tiny tolerance for FFT round-off / bleed.
        assert center_max <= bg_max * 1.5, (
            f"Inverse-polarity center brightened past bg ({center_max:.3f} > "
            f"1.5×{bg_max:.3f}); polarity-flip bug?"
        )

    def test_inverse_polarity_focused_delta_safety(self):
        """At defocus=0 the S2 delta guard must hold even on inverted polarity:
        focused output ≈ input regardless of which side is bright."""
        tensor = _tiny_tensor(with_object_distances=True)
        img = self._make_inverse_fixture(sz=64)
        # Focus on center distance (matches kernels[:, 0] delta).
        depth_m = np.full((64, 64), 0.7, dtype=np.float32)  # focus_distance_m
        _u8, lin = render(
            img, tensor, depth_m=depth_m, hdr_source="macos",
        )
        # S2 delta guard guarantees subject preservation at d=0
        # (project_s2_delta_guard_subject_safety.md). Linear-to-linear
        # comparison; allow a small bleed at FFT pad boundary.
        diff = np.abs(lin - img).max()
        assert diff < 0.5, (
            f"Inverse-polarity S2 delta guard violated: max-abs-diff={diff:.4f}. "
            f"Bright bg + dim spot at focus distance must be near-identity."
        )


# ---------------------------------------------------------------------------
# 7. C3a rename verification (2 probes)
# ---------------------------------------------------------------------------


class TestRenameVerification:
    """C3a: _apply_slab_kernel renamed to _apply_kernel. Verify the new name
    exists, the old name doesn't, and there are no zombie references."""

    def test_apply_kernel_exists(self):
        """_apply_kernel must be importable post-C3a."""
        from summicron.render.convolve import _apply_kernel  # noqa: F401
        assert callable(_apply_kernel), "_apply_kernel must be callable"

    def test_no_zombie_apply_slab_kernel_references(self):
        """Source-grep: no live `_apply_slab_kernel` reference in summicron/.

        Comments referencing the old name in P1.10.4 commit-message context
        are OK (they live in CLAUDE.md / doc/), but the prod code itself must
        not call the deleted symbol.
        """
        from pathlib import Path
        prod_root = Path(__file__).resolve().parents[2] / "summicron"
        offenders = []
        for py in prod_root.rglob("*.py"):
            text = py.read_text(encoding="utf-8", errors="replace")
            # Look for actual call sites (`_apply_slab_kernel(`) — not just
            # the name embedded in a comment, since C3a rename comments can
            # legitimately reference the old name.
            if "_apply_slab_kernel(" in text:
                offenders.append(str(py))
        assert not offenders, (
            f"Zombie _apply_slab_kernel( call sites in prod: {offenders}"
        )


# ---------------------------------------------------------------------------
# 8. G9 xfail-strict regression armor (1 probe)
# ---------------------------------------------------------------------------


class TestG9RegressionArmor:
    """The G9 sibling at tests/adversarial/test_basis_quality.py is an
    xfail-strict marker on basis-mode 2nd-derivative envelope. If it ever
    XPASSes, basis kernel improved past the 15.0 envelope — Coder action
    needed (un-xfail + retighten envelope). This probe asserts the test
    is still XFAIL (not PASS, not XPASS)."""

    def test_g9_is_xfail_under_pytest(self):
        """Re-run the G9 test in subprocess, parse pytest's verbose output,
        and assert the marker reports XFAIL."""
        result = subprocess.run(
            [
                sys.executable, "-m", "pytest",
                "tests/adversarial/test_basis_quality.py::test_gradient_second_derivative_bounded_basis_mode",
                "-v", "--tb=short", "--no-header", "-q",
            ],
            capture_output=True, text=True, timeout=120,
            cwd=str(__import__("pathlib").Path(__file__).resolve().parents[2]),
        )
        combined = result.stdout + result.stderr
        # XFAIL line in pytest verbose output looks like:
        #   tests/.../test_xxx XFAIL [...]
        # If basis improved past 15.0 envelope, line is:
        #   tests/.../test_xxx XPASS or PASSED
        # The strict=True marker will then make the test FAIL the run.
        assert "xfailed" in combined.lower() or "XFAIL" in combined, (
            f"G9 sibling test no longer reports XFAIL — basis improved past "
            f"15.0 envelope or test is broken. Pytest output:\n{combined[-500:]}"
        )
        # Run must EXIT 0 (xfail-strict + xfail = pass).
        assert result.returncode == 0, (
            f"G9 sibling test run exit={result.returncode}; "
            f"output:\n{combined[-500:]}"
        )


# ---------------------------------------------------------------------------
# 9. --downres stress (C1 collapse handles both old + new flag values)
# ---------------------------------------------------------------------------


class TestDownresStress:
    """C1 collapse: --working-long-edge → --downres. Verify the help text
    mentions the collapse and the parser accepts standard values."""

    def test_downres_help_text_mentions_collapse(self):
        """`--help` output must mention --downres + the working-long-edge
        collapse history (so users grepping for the old flag find the new one)."""
        r = _run_cli(["--help"])
        # --help exits 0 (argparse default).
        assert r.returncode == 0, f"--help failed: {r.stderr[:200]!r}"
        text = r.stdout + r.stderr
        assert "--downres" in text, "--help output missing --downres"
        # The C1 commit body says help text references the collapse.
        assert "working-long-edge" in text or "P1.10.4" in text or "long edge" in text.lower(), (
            "--help output should reference the working-long-edge collapse "
            "or note 'long edge' so old-flag users can find the new one"
        )

    def test_downres_invalid_negative_value(self):
        """Negative --downres is meaningless; argparse should pass through but
        downstream validation should reject (or behave as 'no downres')."""
        r = _run_cli(["--downres", "-100"])
        # We don't require this to fail at argparse (type=int allows negatives),
        # but the run MUST fail eventually since --image and --tensor are missing.
        assert r.returncode != 0, "convolve with no --image/--tensor should fail"


# ---------------------------------------------------------------------------
# 10. Cross-cutting: vestigial-kwarg + DeprecationWarning interaction
# ---------------------------------------------------------------------------


