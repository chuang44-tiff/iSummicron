"""Unit tests for bubble fixture generators (LOCKED SPEC §2.1, §2.2, §5)."""
from __future__ import annotations

import json

import numpy as np
import pytest

from summicron.eval.fixtures.bubble import (
    BUBBLE_GRID_TABLE,
    SOURCE_DIAMETER_PX,
    generate_bubble_grid,
    generate_bubble_random,
)
from summicron.eval.fixtures.checksums import EXPECTED_SHA256
from summicron.eval.fixtures.common import (
    FOCUS_DISTANCE_M,
    TENSOR_DEPTHS_M,
    WRAP_PROBE_DEPTH_M,
    assert_monochrome,
    save_fixture,
    sha256_file,
)


# ---------- bubble_grid -----------------------------------------------------


def test_bubble_grid_shape_and_dtype():
    rgb, depth_m, ann = generate_bubble_grid()
    assert rgb.shape == (1024, 1024, 3)
    assert rgb.dtype == np.float32
    assert depth_m.shape == (1024, 1024)
    assert depth_m.dtype == np.float32
    assert float(rgb.min()) >= 0.0 and float(rgb.max()) <= 1.0


def test_bubble_grid_monochrome_contract():
    rgb, _, _ = generate_bubble_grid()
    assert_monochrome(rgb)  # raises on violation


def test_bubble_grid_has_12_discs_and_wrap_probe():
    _, _, ann = generate_bubble_grid()
    assert ann["scene"] == "bubble_grid"
    assert ann["frame_hw"] == [1024, 1024]
    assert ann["optical_center_xy"] == [512, 512]
    assert ann["focus_distance_m"] == FOCUS_DISTANCE_M
    discs = ann["discs"]
    assert len(discs) == 12
    # Disc #11 = wrap-probe.
    d11 = discs[11]
    assert d11["idx"] == 11
    assert d11["depth_m"] == WRAP_PROBE_DEPTH_M
    assert d11["wrap_applied"] is True
    # All others have wrap_applied=False.
    for d in discs[:11]:
        assert d["wrap_applied"] is False


def test_bubble_grid_depth_values_match_table():
    _, _, ann = generate_bubble_grid()
    for d, (idx, x, y, r_frac, depth, wrap) in zip(ann["discs"], BUBBLE_GRID_TABLE):
        assert d["idx"] == idx
        assert d["center_xy_px"] == [x, y]
        assert d["r_frac"] == r_frac
        assert d["depth_m"] == depth
        assert d["wrap_applied"] == wrap
        assert d["source_diameter_px"] == SOURCE_DIAMETER_PX


def test_bubble_grid_deterministic():
    """Calling generate_bubble_grid twice yields bitwise-identical rgb+depth."""
    rgb_a, depth_a, _ = generate_bubble_grid()
    rgb_b, depth_b, _ = generate_bubble_grid()
    assert np.array_equal(rgb_a, rgb_b)
    assert np.array_equal(depth_a, depth_b)


def test_bubble_grid_sha256_matches_pinned(tmp_path):
    rgb, depth_m, ann = generate_bubble_grid()
    out = save_fixture(tmp_path / "bubble_grid_input.npz", rgb, depth_m, ann)
    assert sha256_file(out) == EXPECTED_SHA256["bubble_grid"]


def test_bubble_grid_depth_map_has_wrap_probe_and_tensor_values():
    """0.40m must appear in depth_m (disc #11 square) alongside tensor depths."""
    _, depth_m, _ = generate_bubble_grid()
    uniq = np.unique(depth_m)
    assert any(abs(v - WRAP_PROBE_DEPTH_M) < 1e-4 for v in uniq)
    assert any(abs(v - FOCUS_DISTANCE_M) < 1e-4 for v in uniq)
    # Every non-background/non-wrap value must be tensor-native.
    allowed = set(TENSOR_DEPTHS_M) | {WRAP_PROBE_DEPTH_M}
    for v in uniq:
        assert any(abs(float(v) - a) < 1e-4 for a in allowed), v


def test_bubble_grid_json_round_trip(tmp_path):
    rgb, depth_m, ann = generate_bubble_grid()
    out = save_fixture(tmp_path / "bg.npz", rgb, depth_m, ann)
    data = np.load(str(out))
    assert np.array_equal(data["rgb"], rgb)
    assert np.array_equal(data["depth_m"], depth_m)
    with open(out.with_suffix(".json")) as f:
        loaded = json.load(f)
    assert loaded["scene"] == "bubble_grid"
    assert len(loaded["discs"]) == 12
    assert loaded["discs"][11]["wrap_applied"] is True


# ---------- bubble_random ---------------------------------------------------


def test_bubble_random_shape_and_seed():
    rgb, depth_m, ann = generate_bubble_random(seed=42)
    assert rgb.shape == (1024, 1024, 3)
    assert depth_m.shape == (1024, 1024)
    assert ann["scene"] == "bubble_random"
    assert ann["seed"] == 42
    assert len(ann["discs"]) == 12
    # No wrap-probe in random.
    for d in ann["discs"]:
        assert d["wrap_applied"] is False
        assert d["depth_m"] != WRAP_PROBE_DEPTH_M


def test_bubble_random_deterministic_with_seed():
    a_rgb, a_d, _ = generate_bubble_random(seed=42)
    b_rgb, b_d, _ = generate_bubble_random(seed=42)
    assert np.array_equal(a_rgb, b_rgb)
    assert np.array_equal(a_d, b_d)


def test_bubble_random_different_seed_yields_different_output():
    a_rgb, _, _ = generate_bubble_random(seed=42)
    b_rgb, _, _ = generate_bubble_random(seed=7)
    assert not np.array_equal(a_rgb, b_rgb)


def test_bubble_random_minimum_pair_distance_96px():
    _, _, ann = generate_bubble_random(seed=42)
    centers = [d["center_xy_px"] for d in ann["discs"]]
    for i, (x1, y1) in enumerate(centers):
        for j, (x2, y2) in enumerate(centers):
            if i >= j:
                continue
            dx, dy = x1 - x2, y1 - y2
            assert dx * dx + dy * dy >= 96 * 96, (
                f"discs {i},{j} within 96 px: {(x1,y1)}, {(x2,y2)}"
            )


def test_bubble_random_sha256_matches_pinned(tmp_path):
    rgb, depth_m, ann = generate_bubble_random(seed=42)
    out = save_fixture(tmp_path / "bubble_random_seed42_input.npz", rgb, depth_m, ann)
    assert sha256_file(out) == EXPECTED_SHA256["bubble_random_seed42"]


# ---------- wrong-polarity fixture (Day-1 post-mortem standing order) ------


def test_wrong_polarity_fixture_refused_by_save_fixture(tmp_path):
    """Build a bright-bg/dark-disc (inverted) bubble_grid and assert the
    save_fixture I/O boundary refuses it because monochrome contract is
    still intact but a separate polarity check — enforced via the
    renderer contract — would mis-measure astigmatism.

    Standing order (Day-1 post-mortem, DevTeam.iSummicron.md:666-676):
    at least one wrong-polarity fixture per round. Here we verify the
    generator never emits the inverted polarity, and a hand-crafted
    inverted rgb passes save_fixture's monochrome assertion (since R=G=B
    still holds) but is measurably the *wrong* polarity: rgb.sum() far
    exceeds the sparse-disc expected total.
    """
    rgb, depth_m, ann = generate_bubble_grid()
    # Invert polarity: bright background, dark discs. R=G=B preserved.
    inverted = 1.0 - rgb
    assert_monochrome(inverted)  # still monochrome; the bug is polarity-level
    # Witness: the legitimate fixture has sparse bright discs.
    assert float(rgb.sum()) < 1e5, (
        "legitimate bubble_grid should be sparse bright-on-dark"
    )
    # Witness: the inverted fixture has dense bright background.
    assert float(inverted.sum()) > 3e6, (
        "inverted fixture should be dense bright-on-dark (wrong polarity)"
    )
    # save_fixture itself does not — and must not silently — promote the
    # inverted fixture. Save both and assert the digests differ.
    out_ok = save_fixture(tmp_path / "ok.npz", rgb, depth_m, ann)
    out_bad = save_fixture(tmp_path / "inv.npz", inverted, depth_m, ann)
    assert sha256_file(out_ok) != sha256_file(out_bad), (
        "inverted and correct-polarity fixtures must hash differently"
    )
