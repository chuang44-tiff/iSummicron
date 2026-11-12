"""Unit tests for usaf_depth fixture generator (LOCKED SPEC §2.3)."""
from __future__ import annotations

import json

import numpy as np

from summicron.eval.fixtures.checksums import EXPECTED_SHA256
from summicron.eval.fixtures.common import (
    FOCUS_DISTANCE_M,
    TENSOR_DEPTHS_M,
    assert_monochrome,
    save_fixture,
    sha256_file,
)
from summicron.eval.fixtures.usaf import (
    USAF_BAR_WIDTHS,
    USAF_TILE_SIZE,
    USAF_TILE_TABLE,
    generate_usaf_depth,
)


def test_usaf_depth_shape_and_dtype():
    rgb, depth_m, ann = generate_usaf_depth()
    assert rgb.shape == (1024, 1024, 3)
    assert rgb.dtype == np.float32
    assert depth_m.shape == (1024, 1024)
    assert depth_m.dtype == np.float32
    assert float(rgb.min()) >= 0.0 and float(rgb.max()) <= 1.0


def test_usaf_depth_monochrome_contract():
    rgb, _, _ = generate_usaf_depth()
    assert_monochrome(rgb)


def test_usaf_depth_annotation_has_5_tiles_with_center_at_0p70():
    _, _, ann = generate_usaf_depth()
    assert ann["scene"] == "usaf_depth"
    assert ann["frame_hw"] == [1024, 1024]
    assert len(ann["tiles"]) == 5
    assert ann["tiles"][0]["depth_m"] == FOCUS_DISTANCE_M
    # All tile depths must be tensor-native.
    for t in ann["tiles"]:
        assert float(t["depth_m"]) in set(TENSOR_DEPTHS_M)


def test_usaf_depth_bboxes_match_table_and_tile_size():
    _, _, ann = generate_usaf_depth()
    for t, ((cx, cy), depth) in zip(ann["tiles"], USAF_TILE_TABLE):
        x0, y0, x1, y1 = t["bbox_xyxy"]
        assert x1 - x0 == USAF_TILE_SIZE
        assert y1 - y0 == USAF_TILE_SIZE
        assert (x0 + x1) // 2 == cx
        assert (y0 + y1) // 2 == cy
        assert t["depth_m"] == depth
        assert t["bar_widths_px"] == list(USAF_BAR_WIDTHS)


def test_usaf_depth_deterministic():
    a_rgb, a_d, _ = generate_usaf_depth()
    b_rgb, b_d, _ = generate_usaf_depth()
    assert np.array_equal(a_rgb, b_rgb)
    assert np.array_equal(a_d, b_d)


def test_usaf_depth_depth_map_regions_match_annotation():
    """Depth interior (16 px inset from bbox) must equal tile depth.

    A small 8 px overlap between center tile bbox [352,672]^2 and the
    4 corner bboxes at 360 is accepted — later-written corner tiles
    win at the overlap. The 16 px inset exceeds the overlap cleanly.
    """
    _, depth_m, ann = generate_usaf_depth()
    inset = 16
    for t in ann["tiles"]:
        x0, y0, x1, y1 = t["bbox_xyxy"]
        tile_depth = depth_m[y0 + inset:y1 - inset, x0 + inset:x1 - inset]
        assert np.all(np.isclose(tile_depth, t["depth_m"], atol=1e-4)), (
            f"tile {t['idx']} depth interior != {t['depth_m']}"
        )


def test_usaf_depth_bars_only_inside_tiles():
    rgb, _, ann = generate_usaf_depth()
    # Build an OR-mask of all tile bboxes.
    mask = np.zeros(rgb.shape[:2], dtype=bool)
    for t in ann["tiles"]:
        x0, y0, x1, y1 = t["bbox_xyxy"]
        mask[y0:y1, x0:x1] = True
    outside = rgb[~mask]
    assert float(outside.max()) == 0.0, (
        "non-zero pixels outside USAF tiles — bars leaked"
    )


def test_usaf_depth_sha256_matches_pinned(tmp_path):
    rgb, depth_m, ann = generate_usaf_depth()
    out = save_fixture(tmp_path / "usaf_depth_input.npz", rgb, depth_m, ann)
    assert sha256_file(out) == EXPECTED_SHA256["usaf_depth"]


def test_usaf_depth_json_round_trip(tmp_path):
    rgb, depth_m, ann = generate_usaf_depth()
    out = save_fixture(tmp_path / "u.npz", rgb, depth_m, ann)
    data = np.load(str(out))
    assert np.array_equal(data["rgb"], rgb)
    with open(out.with_suffix(".json")) as f:
        loaded = json.load(f)
    assert loaded["scene"] == "usaf_depth"
    assert len(loaded["tiles"]) == 5
