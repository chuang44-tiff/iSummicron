"""P1.9.3 L2 / G2 / G5 — load_tensor shape-derivation and validation tests.

Covers:
- ``R``/``D`` keys exposed in return dict, derived from ``kernels.shape[:2]``.
- Default tensor reports ``R==12, D==11`` and dict agrees with kernel shape.
- Synthetic tensors at arbitrary shapes round-trip through load_tensor.
- Mutated tensors with bad ``radial_edges`` / ``object_distances_m`` length
  raise ``ValueError``.
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


class TestLoadTensorShapeDerivation:
    """L2 — R, D keys derived from kernels.shape[:2]."""

    def test_default_tensor_R12_D11(self):
        t = load_tensor(DEFAULT_TENSOR)
        assert t["R"] == 12
        assert t["D"] == 11
        assert t["kernels"].shape[:2] == (t["R"], t["D"])

    def test_default_tensor_metadata_flags(self):
        """HYG-0 item 11: wheel tensor is RAW (no OTF norm) with S2 delta guard installed."""
        d = np.load(str(DEFAULT_TENSOR))
        assert "otf_norm_applied" not in d.files or not bool(d["otf_norm_applied"])
        assert "delta_guard_applied" in d.files and bool(d["delta_guard_applied"])

    @pytest.mark.parametrize("R,D", [(3, 2), (12, 11), (24, 11)])
    def test_synthetic_tensor_shape_keys(self, tmp_path, R, D):
        path = write_synthetic_tensor(tmp_path / f"syn_{R}x{D}.npz", R, D)
        t = load_tensor(path)
        assert t["R"] == R
        assert t["D"] == D
        assert t["kernels"].shape[:2] == (R, D)
        assert t["R"] == int(t["R"])  # int-typed, not np.int64
        assert isinstance(t["R"], int)
        assert isinstance(t["D"], int)


class TestLoadTensorValidation:
    """G5 — length mismatches raise ValueError (not AssertionError / KeyError)."""

    def test_radial_edges_too_short_raises(self, tmp_path):
        path = write_synthetic_tensor(tmp_path / "bad_edges.npz", R=4, D=3)
        # Mutate on disk: rewrite with truncated radial_edges (length 4 != R+1=5).
        data = dict(np.load(str(path), allow_pickle=True))
        data["radial_edges"] = data["radial_edges"][:-1]  # 5 -> 4
        np.savez(str(path), **data)
        with pytest.raises(ValueError, match="radial_edges length"):
            load_tensor(path)

    def test_radial_edges_too_long_raises(self, tmp_path):
        path = write_synthetic_tensor(tmp_path / "bad_edges2.npz", R=4, D=3)
        data = dict(np.load(str(path), allow_pickle=True))
        # Pad to length R+2 (one too many).
        extra = np.concatenate(
            [data["radial_edges"], [data["radial_edges"][-1] + 0.1]]
        )
        data["radial_edges"] = extra
        np.savez(str(path), **data)
        with pytest.raises(ValueError, match="radial_edges length"):
            load_tensor(path)

    def test_object_distances_wrong_length_raises(self, tmp_path):
        path = write_synthetic_tensor(tmp_path / "bad_od.npz", R=4, D=3)
        data = dict(np.load(str(path), allow_pickle=True))
        # Shrink to length D-1.
        data["object_distances_m"] = data["object_distances_m"][:-1]
        np.savez(str(path), **data)
        with pytest.raises(ValueError, match="object_distances_m length"):
            load_tensor(path)

    def test_non_4d_kernels_raises(self, tmp_path):
        # Write a 3D kernel array; load_tensor must reject it.
        path = tmp_path / "bad_3d.npz"
        np.savez(
            str(path),
            kernels=np.zeros((4, 3, 5), dtype=np.float32),
            radial_edges=np.linspace(0, 1, 5),
            defocus_ratios=np.linspace(0, 1, 3),
            object_distances_m=np.linspace(0.7, 2.0, 3),
            focus_distance_m=np.float64(0.7),
            corpus_ref_long_edge=np.int64(5712),
            otf_norm_applied=np.bool_(False),
            delta_guard_applied=np.bool_(True),
        )
        with pytest.raises(ValueError, match="4D"):
            load_tensor(path)


class TestKernelL1Intact:
    """G6 — per-kernel L1 sum is preserved bit-for-bit on round-trip."""

    def test_synthetic_l1_round_trip_preserved(self, tmp_path):
        path = write_synthetic_tensor(tmp_path / "l1.npz", R=5, D=4)
        # Read raw kernels from disk (pre-load_tensor).
        raw = np.load(str(path))["kernels"]
        t = load_tensor(path)
        loaded = t["kernels"]
        # Bit-identical after dtype cast (both were float32 on write).
        assert np.array_equal(raw.astype(np.float32), loaded)
        # L1 sanity.
        sums = loaded.reshape(loaded.shape[0] * loaded.shape[1], -1).sum(axis=1)
        assert np.all(np.abs(sums - 1.0) < 1e-5)
