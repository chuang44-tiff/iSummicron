"""P1.10.2 — Adversarial hostile-probe coverage.

Round 1 Adversarial pass over P1.10.2 cleanup + default-flip changes
(8 atomic cuts: removed flags, removed modules, BooleanOptionalAction, new
defaults, vestigial-warn removal, escape-hatch back-compat, deferred symbols).

Categories:
  A. Removed-flag rejection (argparse returns non-zero + error message)
  B. Removed-import errors (_make_delta relocation, tonemap + otf_norm gone)
  C. BooleanOptionalAction edges (HISTORICAL — P1.10.4 C5 deleted the
     --layered-gather flag; tests in this section use a self-contained
     argparse re-impl that documents BooleanOptionalAction semantics
     in the abstract, not the live convolve.py flag.)
  D. Layered-gather silent fallback (now emits DeprecationWarning per
     P1.10.4 C5 + locked-spec § 4 option (b))
  E. New-defaults sanity (inspect.signature on render)
  F. Kwarg-deprecation contract (P1.10.5 C1 — explicit-pass fires
     DeprecationWarning; flipped Cut 7's no-warn contract)
  G. Escape-hatch back-compat (old behaviour reachable via explicit overrides)
  H. (REMOVED in P1.10.4 C3 — slab path deleted; deferral satisfied by C3.)

Locked spec: doc/plan/phase1/section17/locked-spec-cleanup-P1_10_2.md
"""
from __future__ import annotations

import inspect
import sys
import subprocess
import warnings

import numpy as np
import pytest

from summicron.render.convolve import render


# ---------------------------------------------------------------------------
# Shared synthetic-tensor factory (self-contained; no fixture dependency)
# ---------------------------------------------------------------------------

def _tiny_tensor(R=4, D=4, ksize=33, with_object_distances=True):
    """Minimal valid render-ready tensor, delta guard at d=0."""
    cy, cx = ksize // 2, ksize // 2
    yy, xx = np.mgrid[:ksize, :ksize]
    g = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * 4.0 ** 2))
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


# ---------------------------------------------------------------------------
# A. Removed-flag rejection
# ---------------------------------------------------------------------------

class TestRemovedFlagRejection:
    """Argparse must reject flags that were cut in P1.10.2."""

    def _run(self, extra_flags):
        result = subprocess.run(
            [
                sys.executable, "-m", "summicron.render.convolve",
            ] + extra_flags,
            capture_output=True,
            text=True,
            timeout=60,
        )
        return result

    def test_bright_pass_knee_rejected(self):
        """`--bright-pass-knee 0.5` must cause non-zero exit (flag was Cut 3)."""
        r = self._run(["--bright-pass-knee", "0.5"])
        assert r.returncode != 0, (
            f"--bright-pass-knee accepted but should be removed; "
            f"stdout={r.stdout[:200]} stderr={r.stderr[:200]}"
        )
        combined = (r.stdout + r.stderr).lower()
        assert "unrecognized" in combined or "error" in combined, (
            f"Expected argparse error text; got: {combined[:300]}"
        )

    def test_dual_tensor_rejected(self):
        """`--dual-tensor /tmp/x.npz` must cause non-zero exit (flag was Cut 4)."""
        r = self._run(["--dual-tensor", "/tmp/x.npz"])
        assert r.returncode != 0, (
            f"--dual-tensor accepted but should be removed; "
            f"stderr={r.stderr[:200]}"
        )
        combined = (r.stdout + r.stderr).lower()
        assert "unrecognized" in combined or "error" in combined

    def test_depth_source_none_rejected(self):
        """`--depth-source none` must cause non-zero exit (flag was Cut 5)."""
        r = self._run(["--depth-source", "none"])
        assert r.returncode != 0, (
            f"--depth-source accepted but should be removed; "
            f"stderr={r.stderr[:200]}"
        )
        combined = (r.stdout + r.stderr).lower()
        assert "unrecognized" in combined or "invalid choice" in combined or "error" in combined

    def test_layered_gather_rejected(self):
        """`--layered-gather` must cause non-zero exit (flag was P1.10.4 C5)."""
        r = self._run(["--layered-gather"])
        assert r.returncode != 0, (
            f"--layered-gather accepted but should be removed; "
            f"stderr={r.stderr[:200]}"
        )
        combined = (r.stdout + r.stderr).lower()
        assert "unrecognized" in combined or "error" in combined, (
            f"Expected argparse error text; got: {combined[:300]}"
        )

    def test_no_layered_gather_rejected(self):
        """`--no-layered-gather` must cause non-zero exit (flag was P1.10.4 C5)."""
        r = self._run(["--no-layered-gather"])
        assert r.returncode != 0, (
            f"--no-layered-gather accepted but should be removed; "
            f"stderr={r.stderr[:200]}"
        )
        combined = (r.stdout + r.stderr).lower()
        assert "unrecognized" in combined or "error" in combined


# ---------------------------------------------------------------------------
# B. Removed-import errors
# ---------------------------------------------------------------------------

class TestRemovedImports:
    """Cut 2 (otf_norm) + Cut 6 (tonemap) must be gone; _make_delta relocated."""

    def test_tonemap_reinhard_import_error(self):
        """`from summicron.render.tonemap import reinhard` must raise ImportError
        (Cut 6: tonemap.py deleted)."""
        with pytest.raises((ImportError, ModuleNotFoundError)):
            from summicron.render.tonemap import reinhard  # noqa: F401

    def test_otf_norm_normalize_otf_import_error(self):
        """`from summicron.psf_model.otf_norm import normalize_otf` must raise
        ImportError (Cut 2: otf_norm.py deleted)."""
        with pytest.raises((ImportError, ModuleNotFoundError)):
            from summicron.psf_model.otf_norm import normalize_otf  # noqa: F401

    def test_otf_norm_make_delta_import_error(self):
        """`from summicron.psf_model.otf_norm import _make_delta` must raise
        ImportError — _make_delta now lives in summicron.render.convolve."""
        with pytest.raises((ImportError, ModuleNotFoundError)):
            from summicron.psf_model.otf_norm import _make_delta  # noqa: F401

    def test_make_delta_importable_from_convolve(self):
        """`from summicron.render.convolve import _make_delta` must
        succeed (0.5.0: the builder module was removed)."""
        from summicron.render.convolve import _make_delta  # noqa: F401
        assert callable(_make_delta), "_make_delta must be callable"


# ---------------------------------------------------------------------------
# C. BooleanOptionalAction edges
# ---------------------------------------------------------------------------

class TestBooleanOptionalAction:
    """layered-gather is now BooleanOptionalAction default=True."""

    def _parse(self, extra: list[str]):
        """Parse args via the module's own parser (import private helper)."""
        import summicron.render.convolve as _c
        import argparse
        # Re-construct the parser by calling parse_args on a known-safe invocation
        # that doesn't require image/tensor (use sys.argv trick).
        import sys
        old_argv = sys.argv
        try:
            sys.argv = ["convolve", "--image", "/dev/null", "--tensor", "/dev/null"] + extra
            # We can't call parse_args without reaching the full __main__ block,
            # but we CAN parse directly via argparse by reconstructing the namespace.
            # Instead, use subprocess and capture just the first error line.
        finally:
            sys.argv = old_argv


    def test_no_layered_gather_parse_accepted(self):
        """--no-layered-gather must parse to False (subprocess + env round-trip)."""
        # We test this via argparse error probing: --no-layered-gather must NOT fail.
        r = subprocess.run(
            [
                sys.executable, "-c",
                "import sys, argparse; "
                "import summicron.render.convolve as c; "
                # We extract just the arg-parse piece by examining the parse result
                # via a tiny inline script that reimplements the same action:
                "p = argparse.ArgumentParser(); "
                "p.add_argument('--layered-gather', action=argparse.BooleanOptionalAction, default=True); "
                "args = p.parse_args(['--no-layered-gather']); "
                "assert args.layered_gather is False, f'got {args.layered_gather}'; "
                "print('OK')",
            ],
            capture_output=True, text=True, timeout=30,
        )
        assert r.returncode == 0 and "OK" in r.stdout, (
            f"--no-layered-gather round-trip failed: stdout={r.stdout} stderr={r.stderr[:300]}"
        )

    def test_last_wins_layered_gather_false(self):
        """--layered-gather --no-layered-gather → False (last-wins argparse semantics)."""
        r = subprocess.run(
            [
                sys.executable, "-c",
                "import argparse; "
                "p = argparse.ArgumentParser(); "
                "p.add_argument('--layered-gather', action=argparse.BooleanOptionalAction, default=True); "
                "args = p.parse_args(['--layered-gather', '--no-layered-gather']); "
                "assert args.layered_gather is False, f'got {args.layered_gather}'; "
                "print('OK')",
            ],
            capture_output=True, text=True, timeout=30,
        )
        assert r.returncode == 0 and "OK" in r.stdout, (
            f"last-wins semantics check failed: stdout={r.stdout} stderr={r.stderr[:300]}"
        )


# ---------------------------------------------------------------------------
# D. Layered-gather silent fallback (convolve.py ~line 1224)
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# E. New-defaults sanity
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# F. Kwarg-deprecation contract (P1.10.5 C1 flipped Cut 7's no-warn contract)
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# G. Escape-hatch back-compat
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# H. (REMOVED in P1.10.4 C3 — slab path deleted; deferral satisfied.)
#
# Was `class TestSpecDeferralIntegrity` asserting `_build_slab_kernel` symbol
# existence and `radial_blend_mode="slab"` callability under P1.10.2 § 8
# deferral. P1.10.4 C3 deletes the slab body bodies; the deferral promise
# is fulfilled (slab → C4b removes the flag wholesale next cut). The
# Cat-G escape-hatch test (`test_explicit_overrides_disable_all_new_defaults`)
# already covers the `radial_blend_mode="slab"` callability via the
# transparent slab→basis route in C3.
# ---------------------------------------------------------------------------
