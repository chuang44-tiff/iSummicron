"""``-O`` (optimized) mode safety (P1.audit Item 5).

Asserts swept to explicit raises; under ``python -O`` the ``assert``
statement is stripped, so any leftover ``assert`` would silently allow
NaN output. Each subprocess runs the renderer entrypoint with ``-O`` and
asserts: bad inputs exit 2 with a clean error message; happy path exits 0.
"""
from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

_REPO = str(Path(__file__).resolve().parents[2])


def _make_minimal_tensor(tmp: Path, focus_mismatch: bool = False,
                         scale_kernels: float = 1.0) -> Path:
    """Tiny kernel tensor for fast in-process render via api.render_image."""
    R, D, K = 2, 2, 3
    kernels = np.ones((R, D, K, K), dtype=np.float32)
    kernels /= kernels.sum(axis=(2, 3), keepdims=True)
    kernels *= float(scale_kernels)
    obj_dist = np.array([0.7, 1.4], dtype=np.float64)
    focus = np.float64(0.7 if not focus_mismatch else 1.0)
    p = tmp / "t.npz"
    np.savez(
        str(p),
        kernels=kernels,
        radial_edges=np.linspace(0, 1, R + 1).astype(np.float64),
        defocus_ratios=np.linspace(0, 1, D).astype(np.float64),
        object_distances_m=obj_dist,
        focus_distance_m=focus,
        corpus_ref_long_edge=np.int64(64),
        schema_version=np.array("1.0"),
        delta_guard_applied=np.bool_(True),
    )
    return p


def _run_under_O(snippet: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-O", "-c", snippet],
        capture_output=True, text=True, check=False,
    )


@pytest.mark.slow
def test_focus_mismatched_tensor_under_O_exits_clean(tmp_path):
    """A focus_distance mismatch trips TensorValidationError under -O."""
    t = _make_minimal_tensor(tmp_path, focus_mismatch=True)
    snippet = textwrap.dedent(f"""
        import sys
        from summicron.render.convolve import load_tensor, TensorValidationError
        try:
            load_tensor({str(t)!r})
        except TensorValidationError as e:
            print('caught:', e, file=sys.stderr)
            sys.exit(2)
        sys.exit(0)
    """)
    proc = _run_under_O(snippet)
    assert proc.returncode == 2, (
        f"expected exit 2 under -O; got {proc.returncode}; stderr={proc.stderr!r}"
    )
    assert "Traceback" not in proc.stdout


@pytest.mark.slow
def test_nan_input_under_O_raises(tmp_path):
    """NaN in input must raise (ValueError) — not silently produce NaN output."""
    t = _make_minimal_tensor(tmp_path)
    snippet = textwrap.dedent(f"""
        import sys
        import numpy as np
        sys.path.insert(0, {_REPO!r})
        from summicron.render.convolve import load_tensor
        from tests.fixtures.defocus import render_with_defocus
        tensor = load_tensor({str(t)!r})
        img = np.full((16, 16, 3), np.nan, dtype=np.float32)
        try:
            render_with_defocus(img, tensor)
        except (ValueError, RuntimeError) as e:
            sys.exit(2)
        sys.exit(0)
    """)
    proc = _run_under_O(snippet)
    assert proc.returncode == 2, (
        f"expected exit 2 (NaN guarded); got {proc.returncode}; stderr={proc.stderr!r}"
    )


@pytest.mark.slow
def test_energy_overflow_under_O_raises(tmp_path):
    """Kernels scaled 1000× should trip the energy ratio gate (RuntimeError)."""
    t = _make_minimal_tensor(tmp_path, scale_kernels=1000.0)
    snippet = textwrap.dedent(f"""
        import sys
        import numpy as np
        sys.path.insert(0, {_REPO!r})
        from summicron.render.convolve import load_tensor
        from tests.fixtures.defocus import render_with_defocus
        tensor = load_tensor({str(t)!r})
        img = np.full((16, 16, 3), 0.5, dtype=np.float32)
        try:
            render_with_defocus(img, tensor)
        except RuntimeError as e:
            sys.exit(2)
        except ValueError as e:
            sys.exit(2)
        sys.exit(0)
    """)
    proc = _run_under_O(snippet)
    assert proc.returncode == 2, (
        f"expected exit 2 (energy gate); got {proc.returncode}; stderr={proc.stderr!r}"
    )


@pytest.mark.slow
def test_happy_path_under_O_exits_zero(tmp_path):
    """Plain valid render under -O exits 0."""
    t = _make_minimal_tensor(tmp_path)
    snippet = textwrap.dedent(f"""
        import sys
        import numpy as np
        sys.path.insert(0, {_REPO!r})
        from summicron.render.convolve import load_tensor
        from tests.fixtures.defocus import render_with_defocus
        tensor = load_tensor({str(t)!r})
        img = np.full((16, 16, 3), 0.5, dtype=np.float32)
        out, lin = render_with_defocus(img, tensor)
        assert out.dtype == np.uint8 and out.shape == (16, 16, 3)
        sys.exit(0)
    """)
    proc = _run_under_O(snippet)
    assert proc.returncode == 0, (
        f"happy path failed under -O: stderr={proc.stderr!r}"
    )
