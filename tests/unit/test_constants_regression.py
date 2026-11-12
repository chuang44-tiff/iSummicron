"""P1.9.3 L3 / L4 / L5 / G4 — pin module-level constants.

Invisible constants drift silently across refactors; lock them here so
a grep-gate test catches accidental changes.
"""
from __future__ import annotations

import summicron.render.convolve as convolve


def test_n_ang_is_8():
    """L3 — angular wedge count for kernel rotation."""
    assert convolve.N_ANG == 8


def test_max_active_slabs_is_8():
    """L4 — SOFT cap for layered-gather active-slab warn-once."""
    assert convolve._MAX_ACTIVE_SLABS == 8


def test_empty_skip_thresh_is_0_01():
    """L5 — layered-gather empty-slab skip threshold."""
    assert convolve._EMPTY_SKIP_THRESH == 0.01


def test_d_knee_layered_in_range():
    """Sanity: the layered α-blend knee stays on the (0, 1] interval.

    Degenerate values (0 or >1) would break the α-blend per comments
    in convolve.py at the declarations.
    """
    assert 0.0 < convolve.D_KNEE_LAYERED <= 1.0
