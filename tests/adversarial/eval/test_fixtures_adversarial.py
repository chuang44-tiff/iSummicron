"""Adversarial tests for eval fixture generators (LOCKED SPEC §5, items 1-6, 11).

Items 7-10 and 12-13 are metric-side and owned by Coder B.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from summicron.eval.fixtures.bubble import generate_bubble_grid, generate_bubble_random
from summicron.eval.fixtures.common import (
    FOCUS_DISTANCE_M,
    FRAME_H,
    FRAME_W,
    TENSOR_DEPTHS_M,
    WRAP_PROBE_DEPTH_M,
    assert_depth_range,
    assert_monochrome,
    save_fixture,
)
from summicron.eval.fixtures.usaf import generate_usaf_depth


# §5 item 1 — negative seed → TypeError.
def test_bubble_random_rejects_negative_seed():
    with pytest.raises(TypeError):
        generate_bubble_random(seed=-1)


def test_bubble_random_rejects_non_int_seed():
    with pytest.raises(TypeError):
        generate_bubble_random(seed="forty-two")  # type: ignore[arg-type]


# §5 item 2 — depth outside in-scope set → ValueError naming the offender.
def test_assert_depth_range_rejects_unknown_depth():
    depth = np.full((16, 16), 2.5, dtype=np.float32)  # 2.5m not in tensor grid
    with pytest.raises(ValueError, match="2.5"):
        assert_depth_range(depth, discs=None)


def test_assert_depth_range_rejects_wrap_depth_without_flag():
    depth = np.full((4, 4), WRAP_PROBE_DEPTH_M, dtype=np.float32)
    # disc list lacks wrap_applied=true → 0.40m must be rejected
    with pytest.raises(ValueError, match="0.4"):
        assert_depth_range(depth, discs=[])


def test_assert_depth_range_accepts_wrap_depth_with_flag():
    depth = np.full((4, 4), WRAP_PROBE_DEPTH_M, dtype=np.float32)
    discs = [{"depth_m": WRAP_PROBE_DEPTH_M, "wrap_applied": True}]
    assert_depth_range(depth, discs=discs)  # no raise


# §5 item 3 — frame size too small for grid margin → ValueError.
def test_save_fixture_rejects_shape_mismatch(tmp_path):
    rgb = np.zeros((32, 32, 3), dtype=np.float32)
    depth = np.zeros((16, 16), dtype=np.float32)  # mismatched spatial
    with pytest.raises(ValueError, match="spatial shape"):
        save_fixture(tmp_path / "bad.npz", rgb, depth, {"scene": "x"})


# §5 item 4 — missing 0.7m tile in usaf_depth → ValueError.
def test_usaf_depth_requires_center_0p70():
    """If the USAF_TILE_TABLE is ever edited so center != 0.70m, generate_usaf_depth
    must refuse. Patch the module to witness the guard."""
    import summicron.eval.fixtures.usaf as u

    bad_table = (
        ((512, 512), 10.23),  # center tile is NOT 0.70m
        ((200, 200),  0.98),
        ((824, 200),  1.91),
        ((200, 824),  3.74),
        ((824, 824), 10.23),
    )
    orig = u.USAF_TILE_TABLE
    u.USAF_TILE_TABLE = bad_table  # type: ignore[assignment]
    try:
        with pytest.raises(ValueError, match="mandatory focus"):
            u.generate_usaf_depth()
    finally:
        u.USAF_TILE_TABLE = orig  # type: ignore[assignment]


def test_usaf_depth_rejects_off_tensor_depth():
    import summicron.eval.fixtures.usaf as u

    bad_table = (
        ((512, 512), FOCUS_DISTANCE_M),
        ((200, 200), 2.5),  # 2.5m not in tensor grid
        ((824, 200), 1.91),
        ((200, 824), 3.74),
        ((824, 824), 10.23),
    )
    orig = u.USAF_TILE_TABLE
    u.USAF_TILE_TABLE = bad_table  # type: ignore[assignment]
    try:
        with pytest.raises(ValueError, match="not in tensor-native"):
            u.generate_usaf_depth()
    finally:
        u.USAF_TILE_TABLE = orig  # type: ignore[assignment]


# §5 item 5 — monochrome contract violation (forced R != G path) → AssertionError.
def test_save_fixture_raises_on_mono_violation(tmp_path):
    rgb = np.zeros((16, 16, 3), dtype=np.float32)
    rgb[..., 0] = 0.5  # R != G, monochrome contract broken
    depth = np.full((16, 16), FOCUS_DISTANCE_M, dtype=np.float32)
    with pytest.raises(AssertionError, match="monochrome"):
        save_fixture(tmp_path / "bad.npz", rgb, depth, {"scene": "x"})


def test_assert_monochrome_raises_on_channel_mismatch():
    rgb = np.zeros((4, 4, 3), dtype=np.float32)
    rgb[..., 2] = 0.1
    with pytest.raises(AssertionError, match="monochrome"):
        assert_monochrome(rgb)


# §5 item 6 — NPZ+JSON round-trip: regenerate, reload, structurally equal.
def test_bubble_grid_npz_json_round_trip(tmp_path):
    rgb, depth, ann = generate_bubble_grid()
    out = save_fixture(tmp_path / "bg.npz", rgb, depth, ann)
    loaded = np.load(str(out))
    assert np.array_equal(loaded["rgb"], rgb)
    assert np.array_equal(loaded["depth_m"], depth)
    with open(out.with_suffix(".json")) as f:
        ann2 = json.load(f)
    # Structurally equal (json round-trip collapses tuple→list — our
    # annotation is already list-shaped, so equal dict compare works).
    assert ann2 == ann


def test_usaf_depth_npz_json_round_trip(tmp_path):
    rgb, depth, ann = generate_usaf_depth()
    out = save_fixture(tmp_path / "u.npz", rgb, depth, ann)
    loaded = np.load(str(out))
    assert np.array_equal(loaded["rgb"], rgb)
    assert np.array_equal(loaded["depth_m"], depth)
    with open(out.with_suffix(".json")) as f:
        ann2 = json.load(f)
    assert ann2 == ann


# §5 item 11 — wrong-polarity fixture standing order (Day-1 post-mortem).
def test_wrong_polarity_inverted_bubble_grid_is_distinguishable(tmp_path):
    """Build a bright-bg/dark-disc variant and assert it is structurally
    distinct from the correct-polarity generator output. The generator
    itself must never emit inverted polarity; this test pins that.

    See DevTeam.iSummicron.md:666-676 and LOCKED SPEC §5 item 11.
    """
    rgb_ok, depth_m, ann = generate_bubble_grid()
    # Correct polarity: sparse bright discs, mostly zero.
    assert float(rgb_ok.mean()) < 0.01
    # Inverted: bright background, dark discs.
    rgb_inv = 1.0 - rgb_ok
    assert float(rgb_inv.mean()) > 0.99
    # Both are monochrome (R=G=B); polarity is a content-level concept.
    assert_monochrome(rgb_ok)
    assert_monochrome(rgb_inv)
    # save_fixture does not silently auto-correct polarity — it writes
    # what it was given, and the two files differ. A downstream metric
    # (Coder B) must catch polarity; at the fixture layer we assert
    # only that the generator never emits the inverted one.
    out_ok = save_fixture(tmp_path / "ok.npz", rgb_ok, depth_m, ann)
    out_inv = save_fixture(tmp_path / "inv.npz", rgb_inv, depth_m, ann)
    import hashlib
    h_ok = hashlib.sha256(out_ok.read_bytes()).hexdigest()
    h_inv = hashlib.sha256(out_inv.read_bytes()).hexdigest()
    assert h_ok != h_inv


# Extra: layout-infeasibility on USAF (direct common.py helper).
def test_draw_usaf_tile_rejects_infeasible_layout():
    from summicron.eval.fixtures.common import draw_usaf_tile
    img = np.zeros((100, 100, 3), dtype=np.float32)
    # tile_size=50 is too small for 4 widths * 2 rows * 40 px = 320 rows.
    with pytest.raises(ValueError, match="tile layout infeasible"):
        draw_usaf_tile(img, 0, 0, tile_size=50)


def test_bubble_random_retry_cap_raises_when_infeasible():
    """Ask for more discs than the 96-px rejection margin can fit → cap trips."""
    # 64..960 = 896 px range; 96-px spacing → ~9x9 grid ≈ 80 slots; asking
    # for 400 will exhaust the 500-retry cap.
    with pytest.raises(RuntimeError, match="exceeded"):
        generate_bubble_random(seed=42, n=400)
