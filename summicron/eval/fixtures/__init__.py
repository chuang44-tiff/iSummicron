"""Synthetic controlled-ground-truth fixtures for the iSummicron eval harness.

Public API — three generators per LOCKED SPEC Section 9 / P0.3 §8:
``generate_bubble_grid``, ``generate_bubble_random``, ``generate_usaf_depth``.
"""
from __future__ import annotations

from summicron.eval.fixtures.bubble import (
    generate_bubble_grid,
    generate_bubble_random,
)
from summicron.eval.fixtures.usaf import generate_usaf_depth

__all__ = [
    "generate_bubble_grid",
    "generate_bubble_random",
    "generate_usaf_depth",
]
