"""``load_tensor()`` schema + metadata validation (P1.audit Item 4).

8+ malformed-tensor cases each MUST raise ``TensorValidationError``;
plus a happy-path schema_version='1.0' load + a missing-schema warn-only
load.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest

from summicron.render.convolve import TensorValidationError, load_tensor


def _good_kwargs(tmp: Path):
    """Construct a minimal-valid tensor payload (R=2, D=2, kH=kW=3)."""
    R, D, K = 2, 2, 3
    kernels = np.ones((R, D, K, K), dtype=np.float32)
    kernels /= kernels.sum(axis=(2, 3), keepdims=True)
    return {
        "kernels": kernels,
        "radial_edges": np.linspace(0, 1, R + 1).astype(np.float64),
        "defocus_ratios": np.linspace(0, 1, D).astype(np.float64),
        "object_distances_m": np.array([0.7, 1.4], dtype=np.float64),
        "focus_distance_m": np.float64(0.7),
        "corpus_ref_long_edge": np.int64(2048),
    }


def _save(tmp: Path, payload: dict, name: str = "t.npz") -> Path:
    p = tmp / name
    np.savez(str(p), **payload)
    return p


def test_happy_path_loads(tmp_path):
    """Schema_version + delta_guard_applied present → no warnings, R/D set."""
    kw = _good_kwargs(tmp_path)
    kw["schema_version"] = np.array("1.0")
    kw["delta_guard_applied"] = np.bool_(True)
    p = _save(tmp_path, kw, "good.npz")
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        t = load_tensor(p)
    assert t["R"] == 2 and t["D"] == 2


def test_missing_schema_warns_only(tmp_path):
    p = _save(tmp_path, _good_kwargs(tmp_path))
    with pytest.warns(UserWarning, match="schema_version"):
        load_tensor(p)


def test_unknown_schema_raises(tmp_path):
    kw = _good_kwargs(tmp_path)
    kw["schema_version"] = np.array("99.0")
    p = _save(tmp_path, kw)
    with pytest.raises(TensorValidationError, match="schema_version"):
        load_tensor(p)


@pytest.mark.parametrize("missing_key", [
    "kernels", "radial_edges", "defocus_ratios",
])
def test_missing_required_key(tmp_path, missing_key):
    """Schema-required keys: omitting any raises TensorValidationError.

    NOTE: object_distances_m / focus_distance_m / corpus_ref_long_edge are
    OPTIONAL on read (back-compat with intermediate pre-downscale tensors
    + many in-tree test fixtures); validated only when present.
    """
    kw = _good_kwargs(tmp_path)
    kw.pop(missing_key)
    p = _save(tmp_path, kw)
    with pytest.raises(TensorValidationError, match="missing"):
        load_tensor(p)


@pytest.mark.parametrize("absent_key", [
    "object_distances_m", "focus_distance_m", "corpus_ref_long_edge",
])
def test_absent_optional_metadata_loads_with_defaults(tmp_path, absent_key):
    """Absence of object_distances_m / focus_distance_m / corpus_ref_long_edge
    must not block loading — back-compat for intermediate tensors and
    legacy fixtures (P1.audit Item 4 narrowing)."""
    kw = _good_kwargs(tmp_path)
    kw.pop(absent_key)
    p = _save(tmp_path, kw)
    tensor = load_tensor(p)  # MUST NOT raise.
    assert "kernels" in tensor


def test_non_square_kernels(tmp_path):
    kw = _good_kwargs(tmp_path)
    kw["kernels"] = np.ones((2, 2, 3, 5), dtype=np.float32)
    p = _save(tmp_path, kw)
    with pytest.raises(TensorValidationError, match="square"):
        load_tensor(p)


def test_even_side_kernels_warns_only(tmp_path, recwarn):
    """Even-sided kernels are valid intermediate state (pre-downscale
    2048x2048 tensors); load_tensor warns but does not raise. The basis
    kernel builder errors localised at render time if it actually receives
    an even-sided tensor."""
    kw = _good_kwargs(tmp_path)
    kw["kernels"] = np.ones((2, 2, 4, 4), dtype=np.float32)
    p = _save(tmp_path, kw)
    tensor = load_tensor(p)  # MUST NOT raise.
    assert tensor["kernels"].shape == (2, 2, 4, 4)
    msgs = [str(w.message) for w in recwarn.list]
    assert any("even-sided" in m for m in msgs)


def test_non_finite_radial_edges(tmp_path):
    kw = _good_kwargs(tmp_path)
    kw["radial_edges"] = np.array([0.0, np.nan, 1.0], dtype=np.float64)
    p = _save(tmp_path, kw)
    with pytest.raises(TensorValidationError, match="radial_edges"):
        load_tensor(p)


def test_non_monotonic_radial_edges(tmp_path):
    kw = _good_kwargs(tmp_path)
    kw["radial_edges"] = np.array([0.0, 0.5, 0.4], dtype=np.float64)
    p = _save(tmp_path, kw)
    with pytest.raises(TensorValidationError, match="radial_edges"):
        load_tensor(p)


def test_nan_defocus_ratios(tmp_path):
    kw = _good_kwargs(tmp_path)
    kw["defocus_ratios"] = np.array([0.0, np.nan], dtype=np.float64)
    p = _save(tmp_path, kw)
    with pytest.raises(TensorValidationError, match="defocus_ratios"):
        load_tensor(p)


def test_focus_mismatch_isclose_boundary_pass(tmp_path):
    """Tiny ULP drift (rel 1e-7) is tolerated by math.isclose(rel_tol=1e-6)."""
    kw = _good_kwargs(tmp_path)
    kw["object_distances_m"] = np.array(
        [0.7 + 0.7 * 1e-7, 1.4], dtype=np.float64
    )
    p = _save(tmp_path, kw)
    # MUST succeed — within tolerance.
    load_tensor(p)


def test_focus_mismatch_isclose_boundary_fail(tmp_path):
    """A 1e-3 relative mismatch trips the gate."""
    kw = _good_kwargs(tmp_path)
    kw["object_distances_m"] = np.array([0.71, 1.4], dtype=np.float64)
    p = _save(tmp_path, kw)
    with pytest.raises(TensorValidationError, match="focus_distance"):
        load_tensor(p)


def test_negative_corpus_ref(tmp_path):
    kw = _good_kwargs(tmp_path)
    kw["corpus_ref_long_edge"] = np.int64(-100)
    p = _save(tmp_path, kw)
    with pytest.raises(TensorValidationError, match="corpus_ref"):
        load_tensor(p)


def test_load_tensor_passes_allow_pickle_false(tmp_path, monkeypatch):
    """P1.audit Item 4: ``np.load(..., allow_pickle=False)`` is the audit's
    CITED hardening surface (rejects pickled-object payloads from untrusted
    GUI/user-selected tensors). Pin the contract by intercepting np.load
    and asserting the flag is passed.
    """
    captured: dict[str, object] = {}
    real_load = np.load

    def spy_load(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = dict(kwargs)
        return real_load(*args, **kwargs)

    # Patch the np reference inside summicron.render.convolve so the spy
    # fires on load_tensor's actual call site.
    import summicron.render.convolve as _conv

    monkeypatch.setattr(_conv.np, "load", spy_load)

    p = _save(tmp_path, _good_kwargs(tmp_path))
    load_tensor(p)
    assert captured["kwargs"].get("allow_pickle") is False, (
        f"load_tensor must call np.load(allow_pickle=False); "
        f"got kwargs={captured['kwargs']}"
    )
