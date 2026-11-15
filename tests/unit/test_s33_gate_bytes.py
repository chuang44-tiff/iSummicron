"""scripts/s33/gate_bytes.py wraps the frozen compare with a tobytes() check."""
import sys
from pathlib import Path

import numpy as np

import pytest

_S33 = Path(__file__).resolve().parents[2] / "scripts" / "s33"
if not (_S33 / "gate_bytes.py").exists():  # dev-only harness: not shipped in the public tree
    pytest.skip("scripts/s33 absent", allow_module_level=True)
sys.path.insert(0, str(_S33))
import gate_bytes  # noqa: E402
import gate_exact  # noqa: E402


def _g(lin):
    return {"linear": lin, "u8": np.zeros((2, 2, 3), np.uint8), "stdout": "", "stderr": ""}


def test_wrapper_installed_and_sees_signed_zero():
    assert gate_exact.compare is gate_bytes.compare
    a = np.zeros((2, 2, 3), np.float32)
    b = a.copy()
    b[0, 0, 0] = -0.0
    assert gate_bytes._frozen_compare(_g(a), _g(b)) == []
    assert "linear bytes differ" in gate_bytes.compare(_g(a), _g(b))


def test_identical_passes():
    a = np.ones((2, 2, 3), np.float32)
    assert gate_bytes.compare(_g(a), _g(a.copy())) == []
