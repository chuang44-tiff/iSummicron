"""P1.9.4 L21 / G_R_QUADRATURE — pin the raw R-axis sqrt-quadrature contract.

The raw CodeV ``.npz`` stores ``radial_bin_edges_hmax`` as
``np.sqrt(np.linspace(0, 1, 12))`` to machine precision (verified
empirically; max-abs vs sqrt = 1.11e-16). This contract test locks
that invariant so future tensor regenerations either preserve sqrt-
quadrature OR break this test (and force the consumer to update).

Source of facts: ``doc/plan/phase1/section14/learnings-P1_9_4-phase1.md``
Fact 1.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
RAW_NPZ = ROOT / "runs" / "ag_dblgauss" / "ag_dblgauss_f2.0_focus0.70m_d00_0.70m.npz"


@pytest.mark.skipif(not RAW_NPZ.exists(), reason=f"raw CodeV file absent: {RAW_NPZ}")
def test_r_axis_is_sqrt_quadrature():
    """G_R_QUADRATURE — radial_bin_edges_hmax matches sqrt(linspace(0,1,12)) to ~machine ε."""
    raw = np.load(str(RAW_NPZ))
    edges = raw["radial_bin_edges_hmax"]
    expected = np.sqrt(np.linspace(0.0, 1.0, 12))
    # Empirical max-abs error against sqrt is ~1.11e-16; we lock at 1e-12.
    assert np.allclose(edges, expected, atol=1e-12), (
        f"R-axis edges drifted from sqrt-quadrature: "
        f"max-abs diff = {np.abs(edges - expected).max():.3e}"
    )


@pytest.mark.skipif(not RAW_NPZ.exists(), reason=f"raw CodeV file absent: {RAW_NPZ}")
def test_r_axis_edges_length_is_12():
    """Edges array has length 12 (matches the 12 raw PSF samples)."""
    raw = np.load(str(RAW_NPZ))
    assert raw["radial_bin_edges_hmax"].shape == (12,)


@pytest.mark.skipif(not RAW_NPZ.exists(), reason=f"raw CodeV file absent: {RAW_NPZ}")
def test_r_axis_centers_length_is_12():
    """Centers array has length 12 (paired with psf_r00..psf_r11 keys)."""
    raw = np.load(str(RAW_NPZ))
    assert raw["radial_bin_centers_mm"].shape == (12,)


@pytest.mark.skipif(not RAW_NPZ.exists(), reason=f"raw CodeV file absent: {RAW_NPZ}")
def test_r_axis_centers_are_not_edge_midpoints():
    """L23 / Fact 2 — centers are an INDEPENDENT length-12 sampling, not edge midpoints.

    Edge midpoints would be length 11; centers are length 12. The two
    arrays are different sqrt-related but not identical samplings.
    """
    raw = np.load(str(RAW_NPZ))
    edges = raw["radial_bin_edges_hmax"]
    centers = raw["radial_bin_centers_mm"]
    h_max = float(raw["h_max_mm"])
    centers_norm = centers / h_max
    # Edge midpoints (length 11) cannot equal centers (length 12) — sanity.
    edge_mids = (edges[:-1] + edges[1:]) / 2.0
    assert len(edge_mids) == len(centers) - 1
    # And centers[0] should be 0.0 (on-axis), distinct from edge_mids[0]≈0.151.
    assert float(centers_norm[0]) == 0.0
    assert float(edge_mids[0]) > 0.1
