"""S33 Tier B: the STRESS family, hard-gated by the ruled ratio.

User ruling R-B2 (revised, 2026-11-13). The hard Tier B ceilings cover real scenes
and the standard charts at L1 bright-stamp gain <= 4
(``test_s33_tiled_field_character.py``). Beyond that range both engines carry
float32 FFT round-off that exceeds the ceilings, so the contract is

    (tiled error vs float64 truth) / (reference error vs the same truth) <= 1.5

on p99.9 of e (HARD), full frame AND the field-edge annulus. FINAL rule (user,
2026-11-13 round 3): the per-pixel max (absolute e and max ratio) is printed,
not asserted. The stress
family: L1 gain 8 and above, 50x / 100x HDR lights, the peak stamp on HDR, LDR
peak with large (41 x 41) patches. (Also in the family, tested next to their
geometry: HDR 16x dense light fields at gain 2 in
``tests/adversarial/test_s33_tiled_numerics_adversarial.py`` and lens 50 LDR
peak on dense small frames in ``test_s33_tiled_engine_configs.py``.)

The raw Tier B numbers and the max ratio are printed. Amplitudes are the
stress amplitudes; none is reduced to pass.
"""
from __future__ import annotations

import pytest

from tests.fixtures.truth import assert_stress
from tests.unit.test_s33_tiled_core import (
    _gather_kwargs, _run_ref, _run_tiled, _tensor,
)
from tests.unit.test_s33_tiled_field_character import build_chart

# name -> (amplitude, bright_stamp_gain, patch side or None)
STRESS = {
    "l1g8_hdr16": (16.0, 8.0, None),
    "l1g16_hdr16": (16.0, 16.0, None),
    "hdr50_l1g2": (50.0, 2.0, None),
    "hdr100_l1g2": (100.0, 2.0, None),
    "peak_hdr8": (8.0, None, None),
    "peak_hdr50": (50.0, None, None),
    "ldr_peak_patch41": (1.0, None, 41),
}


def _stress_case(name, h, w, tile_n, lens="35", n_lights=16):
    amp, gain, patch = STRESS[name]
    t = dict(_tensor(lens))
    img, depth, lights = build_chart(h, w, amp, n_lights=n_lights, tensor=t)
    if patch:
        r = patch // 2
        for lt in lights:
            y, x = lt["y"], lt["x"]
            img[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1] = amp
    kw = _gather_kwargs(img, depth, K=67, thresh=0.5, bright_stamp_gain=gain,
                        lens=lens, tensor=t)
    ref5, _ = _run_ref(kw)
    got5, _ = _run_tiled(kw, **({"_tile_n": tile_n} if tile_n else {}))
    assert float(ref5[1].max()) > 0.0 or patch      # the bright image was used
    assert_stress(kw, ref5, got5, f"{name} {h}x{w} N={tile_n or 324} lens {lens}")


# Final rule (user, 2026-11-13 r3): only the p99.9 ratio is hard. Per-pixel absolute
# max is printed. For the record (measured, 1e-5 = the Tier B max ceiling), rows
# where the tiled absolute max exceeds 1e-5; the reference's own error is in
# brackets. The tiled engine is WORSE than the reference on: lens 50 peak_hdr8
# 700x640 (1.04e-5 vs 8.0e-6), lens 35 peak_hdr50 640x700 (2.83e-5 vs 2.5e-5),
# lens 50 peak_hdr50 both shapes (5.96e-5 vs 4.13e-5, 5.02e-5 vs 4.35e-5), lens 35
# ldr_peak_patch41 700x640 (2.54e-5 vs 2.02e-5), lens 50 ldr_peak_patch41
# 640x700 (1.4e-4 vs 8.7e-5), and, in the field-edge annulus only, lens 35
# ldr_peak_patch41 640x700 (1.16e-5 vs 7.72e-6) and lens 50 ldr_peak_patch41
# 700x640 (2.41e-5 vs 2.19e-5). It is equal or better on lens 35 peak_hdr50
# 700x640 (1.87e-5 vs 2.1e-5), lens 35 ldr_peak_patch41 640x700 full frame
# (2.18e-5 vs 2.31e-5), lens 50 ldr_peak_patch41 700x640 full frame (9.94e-5 vs
# 1.11e-4), the 1e3 extreme light row (4.16e-5 vs 4.96e-5), and the 260x220
# forced-tile peak_hdr50 / ldr_peak_patch41 rows (3.4e-5 vs 4.17e-5; 1.8e-5 vs
# 2.47e-5). Over all 21 measured region rows above 1e-5: 9 worse, 12 equal or
# better (runs/s33_speed/r4_tiled/stress.log). The ldr_peak_patch41 rows use 41 x 41 patches; peak_hdr rows use
# the HDR chart's own light sizes.


@pytest.mark.parametrize("name", [
    pytest.param(n) for n in STRESS
])
def test_stress_ratio_forced_tiles(name):
    """K=67, N=96 on a 260 x 220 frame (3 x 3 non-multiple tiles)."""
    _stress_case(name, 260, 220, 96)


@pytest.mark.slow
@pytest.mark.parametrize("lens,hw,name", [
    pytest.param(lens, hw, n)
    for lens in ("35", "50")
    for hw in ((700, 640), (640, 700))
    for n in STRESS
])
def test_stress_ratio_production_n(name, hw, lens):
    """Production N (324) on a frame holding 3 x 3 tiles."""
    _stress_case(name, *hw, None, lens=lens)
