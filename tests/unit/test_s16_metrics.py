"""Unit tests for scripts/s16_metrics.py helpers (Q-d S0, spec section 7 item 11)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_P = Path(__file__).resolve().parents[2] / "scripts" / "s16_metrics.py"
if not _P.exists():
    pytest.skip("scripts/s16_metrics.py absent (staged release tree)", allow_module_level=True)
_spec = importlib.util.spec_from_file_location("s16_metrics", _P)
M = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(M)

# Re-pinned in 0.5.0 on the shipped 35 mm optiland tensor (the old vector was the removed
# legacy tensor's, a different and ~2x smaller blur); measured with M.r90_vector, 2-dp rounding.
EXPECT_R90 = [0.71, 4.84, 7.89, 10.16, 11.85, 13.06, 13.96, 14.56, 14.92, 15.4, 15.57]
BRIGHT_LINE = (
    "  bright-pass: thresh=1, region_size=15, floor=0.050, norm_pct=99.50, "
    "bright_pixels_pct=12.173%, bright_energy_ratio=4.0560"
)


@pytest.fixture(scope="module")
def tensor():
    return M.load_tensor_np()


def test_r90_vector_shipped_tensor(tensor):
    v = M.r90_vector(tensor)
    assert v.shape == (11,)
    assert abs(v[0] - 0.7) <= 0.05
    assert np.allclose(v, EXPECT_R90, atol=0.05)


def test_g5_pass_boundaries():
    assert M.g5_pass(7.9 + 0.10, 7.9)
    assert M.g5_pass(7.9 * 1.02, 7.9)
    assert not M.g5_pass(7.9 + 0.17, 7.9)
    assert M.g5_pass(5.0, 7.9)


def test_energy_split_sums():
    ed, eb = M.energy_split(100.0, 0.5, 40.0)
    assert eb == 20.0 and ed == 80.0 and ed + eb == 100.0


def test_parse_bright_line():
    r = M.parse_bright_line(["noise", BRIGHT_LINE, "tail"])
    assert r == {"ber": 4.056, "bright_pixels_pct": 12.173}
    with pytest.raises(ValueError):
        M.parse_bright_line(["no such line", "  alpha-blend: x"])


def test_g_match_two_plane(tensor):
    od, ratios = tensor["object_distances_m"], tensor["defocus_ratios"]
    r90 = M.r90_vector(tensor)
    raw = np.full((40, 40), 0.96, np.float32)
    raw[:, 20:] = 3.0
    bg = np.zeros((40, 40), bool)
    bg[:, 20:] = True
    anchor = 0.96
    p50_target = M.bg_r90_p50(1.2, raw, anchor, od, ratios, r90, bg)
    g = M.g_match(raw, anchor, od, ratios, r90, bg, p50_meters=p50_target)
    assert abs(g - 1.2) <= 0.01
    assert abs(M.bg_r90_p50(g, raw, anchor, od, ratios, r90, bg) - p50_target) <= 0.05


def test_entropy_norm_bounds():
    assert M.entropy_norm(np.zeros(100, np.float32)) == 0.0
    d = np.repeat(np.linspace(0, 1, 11), 10).astype(np.float32)
    assert M.entropy_norm(d) == pytest.approx(1.0, abs=1e-6)


def test_snowflake_count_background_mask():
    lum = np.zeros((40, 40), np.float32)
    lum[5:7, 5:7] = 2.0      # small, in "subject" half
    lum[5:7, 30:32] = 2.0    # small, in "background" half
    bg = np.zeros((40, 40), bool)
    bg[:, 20:] = True
    assert M.snowflake_count(lum) == 2
    assert M.snowflake_count(lum, bg) == 1


def test_analyze_markdown_g5_and_bfac_columns():
    ent = {"g": 1.0, "m1": None, "M4_lapvar_ratio": 1.0,
           "bg": {"r90_p10": 1, "r90_p50": 2, "r90_p90": 3, "spread": 0.1, "entropy": 0.2,
                  "F_cap": 0.0, "F_disc": 0.0},
           "G7": {"ED_ratio": 1.0, "D_ok": True, "ber_ratio": 1.0, "B_ok": True, "bpp_equal": True,
                  "E_total_ratio": 1.0, "max_ratio": 1.0},
           "M5": {"width_px": 1.0, "width_over_r": 1.0, "n": 1},
           "G5": {"margin": -1.51, "tol": 0.158, "ok": True}, "bfac_flag": True,
           "snowflakes": 10, "snowflakes_bg": 4}
    md = M.analyze_markdown({"r90": [0.7], "scenes": {"S": {"bg_kind": "x", "arms": {"a": ent}}}})
    assert "-1.51 (0.16)" in md and "FLAG" in md and "10/4" in md
    assert "G5 margin (tol)" in md and "bfac flag" in md


def test_ascii_label_no_middle_dot():
    assert M._ascii_label("PASS·V") == "PASS-V"
