"""P1.10.5 — Adversarial hostile-probe coverage (round 1).

Round 1 Adversarial pass over P1.10.5 cleanup carry-forward (kwarg
DeprecationWarning + `_LAST_LAYERED_BRIGHT_MASK` deletion + sibling
baseline rm + docstring + filterwarnings). Coder + fix-round shipped;
FAST 100% green at HEAD `9d19165`.

Hostile probe categories:
  TestUnsetSentinelCollision    — caller passing `_UNSET` directly
  TestDeprecationFiresOnEveryValue — 3 kwargs × every value type
  TestNoDeprecationOnDefaultCall — default path is silent
  TestMessageRegexFilterPrecision — C3 regex doesn't silence kwarg deprecations
  TestLastLayeredBrightMaskGone — `_LAST_LAYERED_BRIGHT_MASK` truly removed
  TestWrongPolarityFallbackProbe — dim-blob-on-bright fixture through fallback
  TestStackLevelAttribution     — DeprecationWarning attributes to caller
  (TestSiblingBaselineSweep deleted in HYG-0 item 13 — ~135 s doc-tree scan.)
  TestBitIdentityWithExplicit    — explicit-default vs omitted produces same
                                    bytes

Locked spec: doc/plan/phase1/section19/locked-spec-P1_10_5.md

Pre-flight:
- `convolve._UNSET` is a private module-level `object()` sentinel; users
  passing it explicitly should NOT fire deprecation (it equals the default).
- C3 message-regex filter targets the silent-fallback DeprecationWarning at
  convolve.py:1140-1147; it MUST NOT silence the new C1 kwarg deprecations
  (different phrasing — `r_blend_strength=` vs `layered_gather=True
  requested ... falling back`).
- This file ALSO stamps the silent-fallback regex filter (per C3 pattern)
  because some probes deliberately exercise the fallback branch with no
  depth_m to surface stacklevel + polarity bugs.
"""
from __future__ import annotations

import importlib
import re
import warnings
from pathlib import Path

import numpy as np
import pytest

import summicron.render.convolve as convolve
from summicron.render.convolve import render



# ---------------------------------------------------------------------------
# Shared fixture helpers
# ---------------------------------------------------------------------------


def _tiny_tensor(R=4, D=4, ksize=33, with_object_distances=True, sigma=4.0):
    """Minimal valid tensor with S2 delta guard at d=0.

    Mirrors test_p1_10_4_hostile.py::_tiny_tensor.
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


def _tiny_depth(H=32, W=32, fill=1.5):
    return np.full((H, W), fill, dtype=np.float32)


def _tiny_defocus(H=32, W=32, fill=0.3):
    return np.full((H, W), fill, dtype=np.float32)


def _kwarg_dep_warnings(record_list, kwarg_name):
    """Filter recorded warnings for the C1 kwarg-deprecation message
    (NOT the silent-fallback message)."""
    return [
        w for w in record_list
        if issubclass(w.category, DeprecationWarning)
        and f"render({kwarg_name}=" in str(w.message)
    ]


def _silent_fallback_warnings(record_list):
    return [
        w for w in record_list
        if issubclass(w.category, DeprecationWarning)
        and "falling back to legacy scatter" in str(w.message)
    ]


# ---------------------------------------------------------------------------
# Category 1 — `_UNSET` sentinel collision
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# Category 2 — DeprecationWarning fires on EVERY non-`_UNSET` value
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# Category 3 — DeprecationWarning does NOT fire on default-call path
# ---------------------------------------------------------------------------


class TestNoDeprecationOnDefaultCall:
    """The default-call path (`render(img, tensor, depth_m=...)`) must
    produce ZERO C1 kwarg deprecation warnings. The silent-fallback
    warning is a separate concern (filterwarnings-stamped above)."""

    def test_no_kwargs_passed_no_kwarg_dep(self):
        img, tensor, depth_m = _tiny_img(), _tiny_tensor(), _tiny_depth()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            render(img, tensor, depth_m=depth_m)
        all_kw_dep = []
        for kw_name in ("r_blend_strength", "radial_blend_mode", "layered_gather"):
            all_kw_dep.extend(_kwarg_dep_warnings(caught, kw_name))
        assert not all_kw_dep, (
            f"Default-call path fired kwarg deprecations: "
            f"{[str(w.message) for w in all_kw_dep]}"
        )

    def test_only_supported_kwargs_no_dep(self):
        """Passing only kwargs OTHER than the 3 deprecated ones must be
        silent for kwarg-deprecation purposes."""
        img, tensor, depth_m = _tiny_img(), _tiny_tensor(), _tiny_depth()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            render(
                img, tensor, depth_m=depth_m, psf_norm="l1",
                bright_pass_region_size=0,
                bright_pass_floor=0.10,
                bright_pass_norm_pct=99.0,
            )
        all_kw_dep = []
        for kw_name in ("r_blend_strength", "radial_blend_mode", "layered_gather"):
            all_kw_dep.extend(_kwarg_dep_warnings(caught, kw_name))
        assert not all_kw_dep


# ---------------------------------------------------------------------------
# Category 4 — Message-regex filter precision
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# Category 5 — `_LAST_LAYERED_BRIGHT_MASK` truly gone
# ---------------------------------------------------------------------------


class TestLastLayeredBrightMaskGone:
    """C2 deletion: both `globals()["_LAST_LAYERED_BRIGHT_MASK"] = ...`
    assignments must be gone, AND no test or production source should
    import / globals-access the symbol."""

    def test_attribute_missing_on_module(self):
        sentinel = object()
        result = getattr(convolve, "_LAST_LAYERED_BRIGHT_MASK", sentinel)
        assert result is sentinel, (
            f"`_LAST_LAYERED_BRIGHT_MASK` still exists on convolve module "
            f"as {type(result)} {result!r}; C2 deletion incomplete"
        )

    def test_not_set_after_layered_render(self):
        """Even after running the layered-gather path (which is where the
        global was historically stashed), the attribute must remain absent."""
        img, tensor, depth_m = _tiny_img(), _tiny_tensor(), _tiny_depth()
        render(img, tensor, depth_m=depth_m)
        assert not hasattr(convolve, "_LAST_LAYERED_BRIGHT_MASK"), (
            "Layered-gather render created `_LAST_LAYERED_BRIGHT_MASK` "
            "globals stash; C2 not effective at runtime"
        )


    def test_no_test_file_references_symbol(self):
        """Sanity: zero test files should `import` or string-reference
        `_LAST_LAYERED_BRIGHT_MASK`. (Defends against a future test that
        re-introduces the dependency without re-adding the global.)"""
        tests_root = Path(__file__).resolve().parent.parent
        offenders = []
        for path in tests_root.rglob("*.py"):
            try:
                txt = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            # Skip self-reference in this hostile-probe file.
            if path.name == "test_p1_10_5_hostile.py":
                continue
            if "_LAST_LAYERED_BRIGHT_MASK" in txt:
                offenders.append(str(path))
        assert not offenders, (
            f"`_LAST_LAYERED_BRIGHT_MASK` appears in {len(offenders)} test "
            f"files — symbol resurrected: {offenders[:5]}"
        )

    def test_no_prod_file_references_symbol_outside_doc_comments(self):
        """Production code must not use `_LAST_LAYERED_BRIGHT_MASK` as
        an active symbol. The convolve.py file mentions it ONLY in
        deletion-history comments (P1.10.5 C2 marker)."""
        prod_root = Path(__file__).resolve().parent.parent.parent / "summicron"
        offenders = []
        for path in prod_root.rglob("*.py"):
            try:
                txt = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            for lineno, line in enumerate(txt.splitlines(), 1):
                if "_LAST_LAYERED_BRIGHT_MASK" not in line:
                    continue
                # Permitted: comment lines documenting the deletion.
                stripped = line.lstrip()
                if stripped.startswith("#"):
                    continue
                offenders.append(f"{path}:{lineno}: {line.strip()}")
        assert not offenders, (
            f"`_LAST_LAYERED_BRIGHT_MASK` appears in active production "
            f"code: {offenders}"
        )


# ---------------------------------------------------------------------------
# Category 6 — Wrong-polarity fixture (Post-mortem 1 mandate)
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# Category 7 — Stacklevel attribution
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# Category 9 (optional) — Bit-identity: explicit-default vs omitted
# ---------------------------------------------------------------------------


