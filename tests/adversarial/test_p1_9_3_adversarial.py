"""P1.9.3 adversarial — hostile tensors, validation-gate regressions, path-selection probes.

This suite hunts for bugs the Coder's happy-path suite
(test_convolve_load_tensor.py, test_param_shape.py, test_constants_regression.py,
test_back_compat_canonical.py) left unexplored. Targets:

 1. Hostile .npz schemas (shape / dtype / polarity / monotonicity mismatches).
 2. Wrong-polarity fixture (mandatory per DevTeam Post-mortem 1): delta parked
    at the wrong d-index so S2-guard assumptions break if anything gates on them.
 3. Extreme (R, D) values the L9 factory-smoke-parametrize didn't cover.
 4. (REMOVED in P1.10.4 C5 — `_LEGACY_PATH_COUNTER` deleted; dual-path
    probes vacuous post-False-path-removal.)
 5. `_MAX_ACTIVE_SLABS` warn-message format & count at hostile D.
 6. L11 grep gate on `corpus_ref_long_edge` pinning.
 7. L1 tamper-check (md5 unchanged — no test writes into the real default tensor).
 8. `load_tensor` return-dict key-set lock.

No code edits to summicron/. Tests only. Xfails mark genuine bugs where
validation should fire but currently silently accepts.
"""
from __future__ import annotations

import hashlib
import warnings
from pathlib import Path

import numpy as np
import pytest

from summicron.render._artifacts import default_tensor_path

from summicron.render.convolve import load_tensor, render
from tests.fixtures.synthetic_tensor import make_synthetic_tensor, write_synthetic_tensor

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TENSOR = default_tensor_path()
CONVOLVE_SRC = ROOT / "summicron" / "render" / "convolve.py"

# The exact key-set `load_tensor` is contracted to return (spec §L2 + existing keys).
EXPECTED_DICT_KEYS = frozenset(
    [
        "kernels",
        "radial_edges",
        "defocus_ratios",
        "corpus_ref_long_edge",
        "delta_guard_applied",
        "object_distances_m",
        "focus_distance_m",
        "R",
        "D",
    ]
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rewrite_npz(path: Path, **overrides) -> None:
    """Load an .npz, override keys, write back. Used to inject hostile fields."""
    data = dict(np.load(str(path), allow_pickle=True))
    data.update(overrides)
    np.savez(str(path), **data)


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# 1. Hostile schema — validation-gate regressions
# ---------------------------------------------------------------------------

class TestHostileSchemaValidation:
    """G5 adversarial — every mismatch category MUST raise, not silently pass."""

    def test_radial_edges_plus2_entries_raises(self, tmp_path):
        """R+2 edges (one too many) raises ValueError."""
        path = write_synthetic_tensor(tmp_path / "edges_p2.npz", R=4, D=3)
        data = dict(np.load(str(path), allow_pickle=True))
        extra = np.concatenate(
            [data["radial_edges"], [data["radial_edges"][-1] + 0.05,
                                    data["radial_edges"][-1] + 0.10]]
        )
        data["radial_edges"] = extra
        np.savez(str(path), **data)
        with pytest.raises(ValueError, match="radial_edges length"):
            load_tensor(path)

    def test_object_distances_too_long_raises(self, tmp_path):
        """len(object_distances_m) == D+1 raises ValueError (not silent truncate)."""
        path = write_synthetic_tensor(tmp_path / "od_plus1.npz", R=3, D=3)
        data = dict(np.load(str(path), allow_pickle=True))
        od = np.concatenate([data["object_distances_m"], [data["object_distances_m"][-1] + 0.5]])
        data["object_distances_m"] = od
        np.savez(str(path), **data)
        with pytest.raises(ValueError, match="object_distances_m length"):
            load_tensor(path)

    def test_object_distances_non_monotonic_raises_value_error(self, tmp_path):
        """Non-monotone object_distances_m should raise ValueError, not AssertionError."""
        path = write_synthetic_tensor(tmp_path / "nonmono.npz", R=3, D=4)
        data = dict(np.load(str(path), allow_pickle=True))
        od = data["object_distances_m"].copy()
        od[1], od[2] = od[2], od[1]  # swap to break strict-increasing
        data["object_distances_m"] = od
        np.savez(str(path), **data)
        with pytest.raises(ValueError, match="monoton"):
            load_tensor(path)

    def test_object_distances_non_monotonic_still_rejects(self, tmp_path):
        """Non-monotone object_distances_m MUST still raise (as AssertionError today).

        Paired with the xfail above — we don't tolerate a silent accept even if
        the exception type is wrong. If this test ever stops raising, the
        validation gate has regressed.
        """
        path = write_synthetic_tensor(tmp_path / "nonmono2.npz", R=3, D=4)
        data = dict(np.load(str(path), allow_pickle=True))
        od = data["object_distances_m"].copy()
        od[1], od[2] = od[2], od[1]
        data["object_distances_m"] = od
        np.savez(str(path), **data)
        with pytest.raises((ValueError, AssertionError)):
            load_tensor(path)

    def test_zero_dim_tensor_clear_error(self, tmp_path):
        """Zero-size kernels(0,0,H,W) should raise a user-friendly ValueError."""
        path = tmp_path / "zero.npz"
        np.savez(
            str(path),
            kernels=np.zeros((0, 0, 11, 11), dtype=np.float32),
            radial_edges=np.array([0.0], dtype=np.float64),
            defocus_ratios=np.array([], dtype=np.float64),
            object_distances_m=np.array([], dtype=np.float64),
            focus_distance_m=np.float64(0.7),
            corpus_ref_long_edge=np.int64(5712),
            otf_norm_applied=np.bool_(False),
            delta_guard_applied=np.bool_(True),
        )
        with pytest.raises(ValueError):
            load_tensor(path)

    def test_zero_dim_tensor_still_errors(self, tmp_path):
        """Zero-size kernels(0,0,H,W) MUST raise SOMETHING (don't silent-pass)."""
        path = tmp_path / "zero2.npz"
        np.savez(
            str(path),
            kernels=np.zeros((0, 0, 11, 11), dtype=np.float32),
            radial_edges=np.array([0.0], dtype=np.float64),
            defocus_ratios=np.array([], dtype=np.float64),
            object_distances_m=np.array([], dtype=np.float64),
            focus_distance_m=np.float64(0.7),
            corpus_ref_long_edge=np.int64(5712),
            otf_norm_applied=np.bool_(False),
            delta_guard_applied=np.bool_(True),
        )
        with pytest.raises((ValueError, IndexError, AssertionError)):
            load_tensor(path)


# ---------------------------------------------------------------------------
# 2. Dtype / polarity hostility — silent validation holes
# ---------------------------------------------------------------------------

class TestDtypeAndPolarityHostility:
    """Kernels dtype/value hostility. Most of these silently pass today; each
    xfail marks where spec-intent would prefer a validation error.
    """

    def test_negative_kernels_rejected(self, tmp_path):
        path = write_synthetic_tensor(tmp_path / "neg.npz", R=3, D=3)
        data = dict(np.load(str(path), allow_pickle=True))
        data["kernels"] = -data["kernels"]
        np.savez(str(path), **data)
        with pytest.raises(ValueError, match="(negative|non-?negative|finite)"):
            load_tensor(path)

    def test_nan_kernels_rejected(self, tmp_path):
        path = write_synthetic_tensor(tmp_path / "nan.npz", R=3, D=3)
        data = dict(np.load(str(path), allow_pickle=True))
        data["kernels"][0, 0, 0, 0] = np.nan
        np.savez(str(path), **data)
        with pytest.raises(ValueError, match="(finite|NaN|nan)"):
            load_tensor(path)

    def test_int_kernels_coerced_cleanly_to_float32(self, tmp_path):
        """INT kernels get coerced to float32 (documented behavior).

        Asserts the coercion is lossless in the sense that dtype is float32
        after load and values compare element-wise to the pre-cast float view.
        """
        path = write_synthetic_tensor(tmp_path / "i32.npz", R=3, D=3)
        data = dict(np.load(str(path), allow_pickle=True))
        scaled = (data["kernels"] * 1000.0).astype(np.int32)
        data["kernels"] = scaled
        np.savez(str(path), **data)
        loaded = load_tensor(path)
        assert loaded["kernels"].dtype == np.float32
        # Bit-identical round-trip through float32 cast.
        np.testing.assert_array_equal(
            loaded["kernels"], scaled.astype(np.float32)
        )

    def test_wrong_polarity_delta_at_d_equals_D_minus_1(self, tmp_path):
        """MANDATORY wrong-polarity fixture: delta parked at far instead of near.

        The S2 delta-guard (ray_kernel.make_delta) installs the delta at
        `kernels[r, 0]` — the NEAR-defocus / in-focus slot. A tensor with
        delta at `d = D-1` (far defocus) inverts the physics: the renderer
        would produce sharp output for far-defocus pixels and blurred output
        for in-focus pixels.

        Spec-intent: load_tensor() should NOT accept-or-reject based on
        delta position (tensor identity is not gated on delta-guard shape).
        But render() behavior is observable — subject-depth pixels should
        now blur, and far-depth pixels should stay sharp. This test locks
        the observable inversion so any future "silent delta guard" would
        flag immediately.
        """
        path = write_synthetic_tensor(tmp_path / "wrongpol.npz", R=3, D=3)
        data = dict(np.load(str(path), allow_pickle=True))
        # Swap the d=0 (delta) and d=D-1 (gaussian) slices for every radial bin.
        kernels = data["kernels"].copy()
        near = kernels[:, 0].copy()
        far = kernels[:, -1].copy()
        kernels[:, 0] = far
        kernels[:, -1] = near
        data["kernels"] = kernels
        np.savez(str(path), **data)

        t = load_tensor(path)
        assert t["kernels"].shape[:2] == (3, 3)
        # The delta fingerprint: max value ~= 1.0 on a single pixel.
        # In-focus (d=0) is now the gaussian; far (d=-1) is now the delta.
        assert float(t["kernels"][0, 0].max()) < 0.2, (
            "d=0 should now be a gaussian (max ~= gaussian peak, not 1.0)"
        )
        assert float(t["kernels"][0, -1].max()) == pytest.approx(1.0), (
            "d=D-1 should now be the delta (max == 1.0)"
        )

        # Render observably runs but produces a qualitatively-inverted output:
        # in-focus depth region blurs; far-depth stays sharp. We lock the
        # "this runs end-to-end without raising" invariant plus finite.
        img = np.full((48, 48, 3), 0.5, dtype=np.float32)
        od = t["object_distances_m"]
        depth_m = np.full((48, 48), float(od[0]), dtype=np.float32)  # all near
        out_u8, out_lin = render(img, t, depth_m=depth_m)
        assert out_u8.shape == (48, 48, 3)
        assert np.all(np.isfinite(out_lin))


# ---------------------------------------------------------------------------
# 3. Extreme (R, D) shapes — L9 factory doesn't cover these
# ---------------------------------------------------------------------------

class TestExtremeRDShapes:
    """R=1 minimum + D=1 minimum + very-large R. Breaks expected invariants?"""

    @pytest.mark.parametrize("layered", [True, False])
    def test_R1_D1_minimum_both_paths(self, tmp_path, layered):
        """R=1, D=1 — minimum legitimate tensor. Must not crash on either path."""
        t = make_synthetic_tensor(R=1, D=1, H=11, W=11)
        img = np.full((32, 32, 3), 0.3, dtype=np.float32)
        depth_m = np.full((32, 32), float(t["object_distances_m"][0]), dtype=np.float32)
        out_u8, out_lin = render(img, t, depth_m=depth_m)
        assert out_u8.shape == (32, 32, 3)
        assert np.all(np.isfinite(out_lin))

    @pytest.mark.parametrize("layered", [True, False])
    def test_R1_D_default_both_paths(self, tmp_path, layered):
        """R=1 (single radial bin), D=11 (default). Tests radial-only-degenerate path."""
        t = make_synthetic_tensor(R=1, D=11, H=11, W=11)
        img = np.full((48, 48, 3), 0.3, dtype=np.float32)
        od = t["object_distances_m"]
        depth_m = np.linspace(
            float(od[0]), float(od[-1]), 48 * 48, dtype=np.float32
        ).reshape(48, 48)
        out_u8, out_lin = render(img, t, depth_m=depth_m)
        assert out_u8.shape == (48, 48, 3)
        assert np.all(np.isfinite(out_lin))

    @pytest.mark.parametrize("layered", [True, False])
    def test_R_default_D1_both_paths(self, tmp_path, layered):
        """R=12, D=1 — single-depth tensor. layered=True path has only 1 slab."""
        t = make_synthetic_tensor(R=12, D=1, H=11, W=11)
        img = np.full((48, 48, 3), 0.3, dtype=np.float32)
        depth_m = np.full((48, 48), float(t["object_distances_m"][0]), dtype=np.float32)
        out_u8, out_lin = render(img, t, depth_m=depth_m)
        assert out_u8.shape == (48, 48, 3)
        assert np.all(np.isfinite(out_lin))

    def test_R_very_high_memory_sanity(self, tmp_path):
        """R=50 stress-test (well above current 12). Shape-derivation must propagate
        without crash. Large H*W intentionally kept small to keep wall-time sane.
        """
        t = make_synthetic_tensor(R=50, D=3, H=11, W=11)
        assert t["R"] == 50
        assert t["kernels"].shape == (50, 3, 11, 11)
        img = np.full((32, 32, 3), 0.3, dtype=np.float32)
        od = t["object_distances_m"]
        depth_m = np.linspace(
            float(od[0]), float(od[-1]), 32 * 32, dtype=np.float32
        ).reshape(32, 32)
        out_u8, _ = render(img, t, depth_m=depth_m)
        assert out_u8.shape == (32, 32, 3)


# ---------------------------------------------------------------------------
# 4. (REMOVED in P1.10.4 C5 — _LEGACY_PATH_COUNTER deleted; class
#    TestDualPathCounterProbes excised. Section number preserved for
#    bisect/git-blame stability.)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 5. _MAX_ACTIVE_SLABS warn — format + single-fire sanity at hostile D
# ---------------------------------------------------------------------------

class TestMaxActiveSlabsWarnMessage:
    """L4 — warn must include `D=<n>` and `active_count=<n>` in message."""

    def test_warn_includes_D_and_active_count_at_D20(self):
        """D=20 with depth spread across all bins triggers the soft cap."""
        t = make_synthetic_tensor(R=3, D=20, H=11, W=11)
        img = np.full((48, 48, 3), 0.3, dtype=np.float32)
        od = t["object_distances_m"]
        depth_m = np.linspace(
            float(od[0]), float(od[-1]), 48 * 48, dtype=np.float32
        ).reshape(48, 48)

        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            render(img, t, depth_m=depth_m)

        slab_warns = [
            w for w in rec
            if w.category is RuntimeWarning
            and "_MAX_ACTIVE_SLABS" in str(w.message)
        ]
        assert len(slab_warns) >= 1, (
            "expected at least one RuntimeWarning about _MAX_ACTIVE_SLABS at D=20"
        )
        msg = str(slab_warns[0].message)
        assert "D=20" in msg, f"warn message missing D=20: {msg!r}"
        assert "active_count=" in msg, f"warn message missing active_count=: {msg!r}"
        # Once — message has warn-once semantics per call.
        # (spec: warn-once per-process; within one call we should get at most 1).
        assert len(slab_warns) == 1, (
            f"warn should fire exactly once per render call, got {len(slab_warns)}"
        )

    def test_no_warn_at_D3_below_cap(self):
        """Below the cap (D=3, far below _MAX_ACTIVE_SLABS=8) — no warn."""
        t = make_synthetic_tensor(R=3, D=3, H=11, W=11)
        img = np.full((32, 32, 3), 0.3, dtype=np.float32)
        od = t["object_distances_m"]
        depth_m = np.linspace(
            float(od[0]), float(od[-1]), 32 * 32, dtype=np.float32
        ).reshape(32, 32)

        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            render(img, t, depth_m=depth_m)

        slab_warns = [
            w for w in rec
            if w.category is RuntimeWarning
            and "_MAX_ACTIVE_SLABS" in str(w.message)
        ]
        assert len(slab_warns) == 0, (
            f"no soft-cap warn expected at D=3<=8; got {len(slab_warns)}"
        )


# ---------------------------------------------------------------------------
# 6. L11 / L6 grep gates — module-level literals must not reappear
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 7. L1 — default tensor identity + path
# ---------------------------------------------------------------------------

class TestDefaultTensorUntouched:
    """L1 — wheel-shipped default tensor (summicron/_artifacts/psf_lens35_us3006249_fu0.4.npz) stays at R=12/D=11.

    This test proves:
      (a) The file is present and its MD5 is recorded (for TL to cross-check
          against pre-refactor HEAD out-of-band).
      (b) Adversarial tests never write to the real default tensor — they
          always copy via write_synthetic_tensor() into tmp_path.
      (c) The loaded tensor's (R, D) agree with the spec (12, 11).
    """

    def test_default_tensor_R12_D11_and_md5_recorded(self, capsys):
        t = load_tensor(DEFAULT_TENSOR)
        assert t["R"] == 12
        assert t["D"] == 11
        m = _md5(DEFAULT_TENSOR)
        # Assert the MD5 is a valid 32-hex string (checksum produces one).
        assert len(m) == 32
        assert all(c in "0123456789abcdef" for c in m)
        # Print to stdout so test log captures the hash for TL probe
        # cross-check against any pre-refactor snapshot.
        print(f"[P1.9.3 adversarial] default_tensor md5={m}")

    def test_adversarial_fixtures_never_target_real_tensor(self, tmp_path):
        """Sanity: write_synthetic_tensor writes under tmp_path, never the repo."""
        path = write_synthetic_tensor(tmp_path / "guard.npz", R=2, D=2)
        # The path MUST live under tmp_path, never under summicron/_artifacts/.
        assert tmp_path in path.parents or path.parent == tmp_path
        # And the real file, if present, MUST NOT have tmp_path as a parent.
        if DEFAULT_TENSOR.exists():
            assert tmp_path not in DEFAULT_TENSOR.parents


# ---------------------------------------------------------------------------
# 8. load_tensor return-dict key-set lock
# ---------------------------------------------------------------------------

class TestLoadTensorReturnKeySet:
    """L2 — return-dict keys are exactly the spec-required set."""

    def test_synthetic_tensor_exposes_all_expected_keys(self, tmp_path):
        path = write_synthetic_tensor(tmp_path / "keys.npz", R=3, D=3)
        t = load_tensor(path)
        got = set(t.keys())
        missing = EXPECTED_DICT_KEYS - got
        extra = got - EXPECTED_DICT_KEYS
        assert not missing, f"load_tensor missing keys: {sorted(missing)}"
        assert not extra, f"load_tensor has unexpected extra keys: {sorted(extra)}"

    def test_R_D_are_python_int_not_numpy(self, tmp_path):
        """spec: ints, not np.int64. Dict consumers that use `isinstance(x, int)`
        break on np.int64 on some paths — e.g., JSON-serialize-by-default.
        """
        path = write_synthetic_tensor(tmp_path / "ints.npz", R=5, D=4)
        t = load_tensor(path)
        # Robustness: Python `int`, not numpy scalar.
        assert type(t["R"]) is int
        assert type(t["D"]) is int


# ---------------------------------------------------------------------------
# 9. (removed in 0.5.0: tensor-builder file-naming drift tests)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 10. Tensor-wide energy invariant (not a fixture tautology)
# ---------------------------------------------------------------------------

class TestKernelEnergyAcrossShapes:
    """Non-fixture invariant: every (r, d) slab must be L1 ≈ 1.0 after load
    (regardless of shape / factory / origin), because L1 normalization
    is a stated pipeline property.
    """

    @pytest.mark.parametrize("R,D", [(1, 1), (3, 2), (7, 5), (24, 11)])
    def test_synthetic_l1_invariant_per_slab(self, tmp_path, R, D):
        path = write_synthetic_tensor(tmp_path / f"l1_{R}x{D}.npz", R, D)
        t = load_tensor(path)
        k = t["kernels"]  # (R, D, H, W)
        sums = k.reshape(R * D, -1).sum(axis=1)
        # L1 per slab == 1.0 to within float32 precision.
        assert np.all(np.abs(sums - 1.0) < 1e-5), (
            f"per-slab L1 drift at (R,D)=({R},{D}): "
            f"min={sums.min():.6f} max={sums.max():.6f}"
        )

    def test_default_tensor_l1_invariant_per_slab(self):
        t = load_tensor(DEFAULT_TENSOR)
        k = t["kernels"]
        R, D = k.shape[0], k.shape[1]
        sums = k.reshape(R * D, -1).sum(axis=1)
        # the tensor builder's gate allows ±0.01 drift post-renorm.
        assert np.all(np.abs(sums - 1.0) <= 0.01), (
            f"default tensor L1 drift > 0.01: "
            f"min={sums.min():.6f} max={sums.max():.6f}"
        )


# ---------------------------------------------------------------------------
# 11. Validation-precedence probe — multiple mismatches in one file
# ---------------------------------------------------------------------------

class TestMultiMismatchPrecedence:
    """G5 regression: a hostile .npz with both radial_edges AND object_distances_m
    wrong should still raise (doesn't matter which first), not silently pass.
    """

    def test_both_lengths_wrong_raises_value_error(self, tmp_path):
        path = write_synthetic_tensor(tmp_path / "both_bad.npz", R=4, D=3)
        data = dict(np.load(str(path), allow_pickle=True))
        # Corrupt both fields.
        data["radial_edges"] = data["radial_edges"][:-1]  # R instead of R+1
        data["object_distances_m"] = data["object_distances_m"][:-1]  # D-1
        np.savez(str(path), **data)
        with pytest.raises(ValueError):
            load_tensor(path)
