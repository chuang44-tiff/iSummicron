"""End-to-end score() API + JSON-serializability round-trip."""
from __future__ import annotations

import json

import numpy as np
import pytest

from summicron.eval.score import score
from tests.unit.eval.conftest import (
    bubble_grid_annotation_stub,
    mono_rgb,
    render_discs,
    standard_grid_discs,
)


def test_score_bubble_grid_end_to_end():
    discs = standard_grid_discs()
    mono = render_discs(discs)
    rgb = mono_rgb(mono)
    depth = np.full((1024, 1024), 0.70, dtype=np.float32)
    ann = bubble_grid_annotation_stub(discs)

    report = score(
        rendered_rgb=rgb,
        depth_m=depth,
        annotation=ann,
        input_rgb=rgb,
        metrics=("M1", "M2", "M3", "M4", "M7"),
    )
    # JSON-serializable:
    text = json.dumps(report)
    round_tripped = json.loads(text)
    assert round_tripped["M1"] == pytest.approx(1.0, abs=1e-6)
    assert "M2" in round_tripped
    assert isinstance(round_tripped["M3"], list)
    assert isinstance(round_tripped["M4"], dict)
    assert round_tripped["M7"]["finite_ok"] in (True, False)
    assert "_meta" in round_tripped
    assert round_tripped["_meta"]["scene"] == "bubble_grid"


def test_score_rejects_unknown_metric():
    rgb = mono_rgb(np.zeros((64, 64), dtype=np.float32))
    depth = np.zeros((64, 64), dtype=np.float32)
    ann = bubble_grid_annotation_stub(standard_grid_discs())
    with pytest.raises(ValueError, match="unknown metric"):
        score(rgb, depth, ann, input_rgb=rgb, metrics=("M99",))


def test_score_m1_requires_input():
    rgb = mono_rgb(np.zeros((64, 64), dtype=np.float32))
    depth = np.zeros((64, 64), dtype=np.float32)
    ann = bubble_grid_annotation_stub(standard_grid_discs())
    with pytest.raises(ValueError, match="M1 requires input_rgb"):
        score(rgb, depth, ann, input_rgb=None, metrics=("M1",))


def test_score_shape_mismatch():
    rgb = mono_rgb(np.zeros((64, 64), dtype=np.float32))
    depth = np.zeros((32, 32), dtype=np.float32)
    ann = bubble_grid_annotation_stub(standard_grid_discs())
    with pytest.raises(ValueError, match="depth_m shape"):
        score(rgb, depth, ann, input_rgb=rgb, metrics=("M1",))


def test_score_no_numpy_scalar_leak():
    """_py() coercion: no numpy floats/ints should survive into the report."""
    discs = standard_grid_discs()
    mono = render_discs(discs)
    rgb = mono_rgb(mono)
    depth = np.full((1024, 1024), 0.70, dtype=np.float32)
    ann = bubble_grid_annotation_stub(discs)
    report = score(rgb, depth, ann, input_rgb=rgb)

    def _walk(obj):
        if isinstance(obj, dict):
            for v in obj.values():
                _walk(v)
        elif isinstance(obj, list):
            for v in obj:
                _walk(v)
        else:
            assert not isinstance(obj, (np.floating, np.integer, np.bool_, np.ndarray)), (
                f"numpy scalar leaked: {type(obj).__name__}"
            )

    _walk(report)
