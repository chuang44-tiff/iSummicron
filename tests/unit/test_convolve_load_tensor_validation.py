"""P1.9.4 L11 — load_tensor validation hardening (positive-direction tests).

Companion to the adversarial-suite xfail-flips at
``tests/adversarial/test_p1_9_3_adversarial.py``: these tests pin the
NEW validation behavior introduced in P1.9.4 (the 4 bugs that previously
silently passed or raised cryptic errors).

The locked spec (`doc/plan/phase1/section14/locked-spec-P1_9_4.md` § L11) gates:

 1. Zero-dim kernels (R==0 or D==0) → clear ``ValueError`` BEFORE indexing.
 2. Non-finite kernels (NaN / +Inf / -Inf) → ``ValueError`` reporting first
    bad multi-index.
 3. Negative kernels (below the ``-1e-6`` ULP tolerance) → ``ValueError``
    naming the actual minimum. Tolerance specifically permits ``[-1e-6, 0)``
    so legitimate post-cubic-zoom-clamp tensors pass cleanly.
 4. Non-monotonic ``object_distances_m`` → ``ValueError`` (was
    ``AssertionError``; ValueError is the spec'd uniform validation type).

Plus tolerance-edge probes around L11 negative gate to lock the ``-1e-6``
boundary, and a smoke that the default canonical bench tensor still loads
clean (G_LOAD_TENSOR_DEFAULT).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from summicron.render._artifacts import default_tensor_path

from summicron.render.convolve import load_tensor
from tests.fixtures.synthetic_tensor import write_synthetic_tensor

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TENSOR = default_tensor_path()


def _resave(path: Path, **overrides) -> None:
    """Reopen an .npz, overlay overrides, and rewrite to the same path."""
    data = dict(np.load(str(path), allow_pickle=True))
    data.update(overrides)
    np.savez(str(path), **data)


# ---------------------------------------------------------------------------
# 1. Zero-dim gate (R==0 or D==0)
# ---------------------------------------------------------------------------

class TestZeroDimGate:
    """L11 — zero-size R or D axis raises ValueError BEFORE any indexing."""

    def _write_zero(self, path: Path, R: int, D: int) -> None:
        np.savez(
            str(path),
            kernels=np.zeros((R, D, 11, 11), dtype=np.float32),
            radial_edges=np.linspace(0.0, 1.0, max(R + 1, 1), dtype=np.float64),
            defocus_ratios=(
                np.linspace(0.0, 1.0, D, dtype=np.float64)
                if D >= 1 else np.array([], dtype=np.float64)
            ),
            object_distances_m=(
                np.linspace(0.7, 0.7 + 0.3 * D, D, dtype=np.float64)
                if D >= 1 else np.array([], dtype=np.float64)
            ),
            focus_distance_m=np.float64(0.7),
            corpus_ref_long_edge=np.int64(5712),
            otf_norm_applied=np.bool_(False),
            delta_guard_applied=np.bool_(True),
        )

    def test_R_zero_raises_value_error(self, tmp_path):
        path = tmp_path / "r0.npz"
        self._write_zero(path, R=0, D=3)
        with pytest.raises(ValueError, match="zero-size"):
            load_tensor(path)

    def test_D_zero_raises_value_error(self, tmp_path):
        path = tmp_path / "d0.npz"
        self._write_zero(path, R=3, D=0)
        with pytest.raises(ValueError, match="zero-size"):
            load_tensor(path)

    def test_both_zero_raises_value_error(self, tmp_path):
        path = tmp_path / "rd0.npz"
        self._write_zero(path, R=0, D=0)
        with pytest.raises(ValueError, match="zero-size"):
            load_tensor(path)

    def test_zero_dim_error_message_includes_shape(self, tmp_path):
        """Diagnostic — message must show the offending shape so users can
        figure out which axis collapsed."""
        path = tmp_path / "rd0_msg.npz"
        self._write_zero(path, R=0, D=4)
        with pytest.raises(ValueError, match=r"shape=\(0, 4"):
            load_tensor(path)


# ---------------------------------------------------------------------------
# 2. Non-finite gate (NaN / Inf)
# ---------------------------------------------------------------------------

class TestNonFiniteGate:
    """L11 — kernels containing NaN or +/-Inf raise ValueError with a first
    non-finite multi-index in the message."""

    def test_nan_raises_value_error(self, tmp_path):
        path = write_synthetic_tensor(tmp_path / "nan.npz", R=3, D=3)
        data = dict(np.load(str(path), allow_pickle=True))
        data["kernels"][1, 2, 5, 5] = np.nan
        np.savez(str(path), **data)
        with pytest.raises(ValueError, match="non-finite"):
            load_tensor(path)

    def test_pos_inf_raises_value_error(self, tmp_path):
        path = write_synthetic_tensor(tmp_path / "posinf.npz", R=3, D=3)
        data = dict(np.load(str(path), allow_pickle=True))
        data["kernels"][0, 1, 4, 4] = np.inf
        np.savez(str(path), **data)
        with pytest.raises(ValueError, match="non-finite"):
            load_tensor(path)

    def test_neg_inf_raises_value_error(self, tmp_path):
        path = write_synthetic_tensor(tmp_path / "neginf.npz", R=3, D=3)
        data = dict(np.load(str(path), allow_pickle=True))
        data["kernels"][2, 0, 6, 7] = -np.inf
        np.savez(str(path), **data)
        with pytest.raises(ValueError, match="non-finite"):
            load_tensor(path)

    def test_nan_message_reports_first_bad_index(self, tmp_path):
        """Diagnostic — first non-finite multi-index must appear in message."""
        path = write_synthetic_tensor(tmp_path / "nan_msg.npz", R=4, D=4)
        data = dict(np.load(str(path), allow_pickle=True))
        # Write at a known location; the gate reports the first lex order
        # (np.argwhere returns C-order).
        data["kernels"][0, 0, 0, 7] = np.nan
        np.savez(str(path), **data)
        # Synthetic d=0 slabs are deltas at center; argwhere finds (0,0,0,7) first.
        with pytest.raises(ValueError, match=r"first at index \(0, 0, 0, 7\)"):
            load_tensor(path)


# ---------------------------------------------------------------------------
# 3. Negative-value gate (with -1e-6 tolerance)
# ---------------------------------------------------------------------------

class TestNegativeGate:
    """L11 — negative kernels raise ValueError IF below the -1e-6 tolerance.

    The tolerance is intentionally NOT strict ``< 0``: cubic interpolation +
    np.clip(0, None) in the tensor builder legitimately leaves ULP noise
    in ``[-1e-7, 0)`` after expm1 decompression, and rejecting that would
    spuriously fail valid post-interp tensors.
    """

    def test_clearly_negative_kernels_rejected(self, tmp_path):
        """Sign-flipped tensor (all values negated) → reject."""
        path = write_synthetic_tensor(tmp_path / "neg.npz", R=3, D=3)
        data = dict(np.load(str(path), allow_pickle=True))
        data["kernels"] = -data["kernels"]
        np.savez(str(path), **data)
        with pytest.raises(ValueError, match="negative"):
            load_tensor(path)

    def test_single_large_negative_pixel_rejected(self, tmp_path):
        """One pixel at -1.0 (well past tolerance) → reject."""
        path = write_synthetic_tensor(tmp_path / "neg1.npz", R=3, D=3)
        data = dict(np.load(str(path), allow_pickle=True))
        data["kernels"][1, 1, 4, 4] = -1.0
        np.savez(str(path), **data)
        with pytest.raises(ValueError, match="negative"):
            load_tensor(path)

    def test_message_reports_min_value(self, tmp_path):
        """Diagnostic — min value must appear in message so user sees magnitude."""
        path = write_synthetic_tensor(tmp_path / "neg_msg.npz", R=2, D=2)
        data = dict(np.load(str(path), allow_pickle=True))
        data["kernels"][0, 0, 1, 1] = -0.25
        np.savez(str(path), **data)
        with pytest.raises(ValueError, match="min value"):
            load_tensor(path)

    def test_just_below_tolerance_rejected(self, tmp_path):
        """Value at ``-2e-6`` is below the ``-1e-6`` tolerance → reject."""
        path = write_synthetic_tensor(tmp_path / "neg_below.npz", R=2, D=2)
        data = dict(np.load(str(path), allow_pickle=True))
        # Subtract 2e-6 from one pixel; well below the -1e-6 gate.
        data["kernels"][0, 1, 5, 5] = -2e-6
        np.savez(str(path), **data)
        with pytest.raises(ValueError, match="negative"):
            load_tensor(path)

    def test_within_tolerance_passes_ulp_noise(self, tmp_path):
        """Value at ``-1e-7`` (ULP-level noise) is INSIDE the ``-1e-6`` gate
        and MUST load cleanly. Otherwise legitimate cubic-zoom-clamp tensors
        would fail.
        """
        path = write_synthetic_tensor(tmp_path / "neg_ulp.npz", R=2, D=2)
        data = dict(np.load(str(path), allow_pickle=True))
        # Inject a sub-tolerance negative blip at a non-delta location.
        data["kernels"][0, 1, 7, 8] = -1e-7
        np.savez(str(path), **data)
        # No raise expected.
        loaded = load_tensor(path)
        assert loaded["kernels"].shape == (2, 2, 33, 33)

    def test_exactly_zero_passes(self, tmp_path):
        """Exact zero is non-negative — must pass (sanity)."""
        path = write_synthetic_tensor(tmp_path / "zero_ok.npz", R=2, D=2)
        data = dict(np.load(str(path), allow_pickle=True))
        data["kernels"][0, 1, 7, 8] = 0.0
        np.savez(str(path), **data)
        loaded = load_tensor(path)
        assert loaded["kernels"].shape == (2, 2, 33, 33)


# ---------------------------------------------------------------------------
# 4. Non-monotonic object_distances_m gate (assert → ValueError)
# ---------------------------------------------------------------------------

class TestNonMonotonicObjectDistances:
    """L11 — non-monotone ``object_distances_m`` raises ValueError uniformly
    (was AssertionError pre-P1.9.4).
    """

    def test_swapped_pair_raises_value_error(self, tmp_path):
        path = write_synthetic_tensor(tmp_path / "swap.npz", R=3, D=4)
        data = dict(np.load(str(path), allow_pickle=True))
        od = data["object_distances_m"].copy()
        od[1], od[2] = od[2], od[1]
        data["object_distances_m"] = od
        np.savez(str(path), **data)
        with pytest.raises(ValueError, match="monoton"):
            load_tensor(path)

    def test_strictly_decreasing_raises_value_error(self, tmp_path):
        path = write_synthetic_tensor(tmp_path / "rev.npz", R=3, D=5)
        data = dict(np.load(str(path), allow_pickle=True))
        data["object_distances_m"] = data["object_distances_m"][::-1].copy()
        np.savez(str(path), **data)
        with pytest.raises(ValueError, match="monoton"):
            load_tensor(path)

    def test_equal_consecutive_raises_value_error(self, tmp_path):
        """Strict monotone increase — equality between consecutive entries
        must also raise (np.diff > 0 is strict)."""
        path = write_synthetic_tensor(tmp_path / "eq.npz", R=3, D=4)
        data = dict(np.load(str(path), allow_pickle=True))
        od = data["object_distances_m"].copy()
        od[2] = od[1]  # tie 1↔2
        data["object_distances_m"] = od
        np.savez(str(path), **data)
        with pytest.raises(ValueError, match="monoton"):
            load_tensor(path)

    def test_raise_type_is_value_error_not_assertion(self, tmp_path):
        """Type-precision lock: NOT AssertionError. Mixed `pytest.raises`
        on AssertionError vs ValueError is the legacy bug; verify it sticks."""
        path = write_synthetic_tensor(tmp_path / "type_lock.npz", R=2, D=4)
        data = dict(np.load(str(path), allow_pickle=True))
        od = data["object_distances_m"].copy()
        od[1], od[2] = od[2], od[1]
        data["object_distances_m"] = od
        np.savez(str(path), **data)
        try:
            load_tensor(path)
        except ValueError:
            pass
        except AssertionError as e:
            pytest.fail(
                f"load_tensor raised AssertionError instead of ValueError: {e!r}"
            )
        else:
            pytest.fail("load_tensor did not raise on non-monotonic input")


# ---------------------------------------------------------------------------
# 5. G_LOAD_TENSOR_DEFAULT — default canonical-bench tensor still loads clean
# ---------------------------------------------------------------------------

class TestDefaultTensorStillLoads:
    """No-false-positive sanity: the real bench tensor must still load after
    the new gates land. Catches any over-tight tolerance regressions.
    """

    def test_default_tensor_loads_without_raise(self):
        t = load_tensor(DEFAULT_TENSOR)
        assert t["R"] == 12
        assert t["D"] == 11
        # Sanity: kernels are post-clip, finite, non-negative-ish.
        k = t["kernels"]
        assert np.all(np.isfinite(k))
        assert float(k.min()) >= -1e-6
