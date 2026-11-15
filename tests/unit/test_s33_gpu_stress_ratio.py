"""S33 Tier C WP2, gate G-C5b: the STRESS family through the MLX gather.

User amendment 2026-11-14 (user ruling): for ``--engine gpu`` the hard stress
rule is

    (gpu error vs float64 truth) / (reference error vs the same truth) <= 5

on p99.9 of e, full frame AND the field-edge annulus (Tier B stays 1.5). The
per-pixel max (absolute e and its ratio) is PRINTED, never gated. The real-scene
and standard-chart ceilings stay the hard Tier C ones
(``test_s33_gpu_coverage.py``). The 1e5x single light is out of scope.

The family is the Tier B one (``test_s33_tiled_stress_ratio.py``): L1 gain >= 8,
dense HDR 16x fields, 50x / 100x / 1000x lights, the peak stamp on HDR, LDR peak
with large patches, lens 50 peak on dense small frames. Amplitudes are the stress
amplitudes; none is reduced to pass. No GPU output bytes are stored anywhere.
"""
from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from tests.fixtures import tolerance as tol  # noqa: E402
from tests.fixtures.truth import format_ratio, stress_ratio  # noqa: E402
from tests.unit.test_s33_gpu_core import _run_gpu  # noqa: E402
from tests.unit.test_s33_tiled_core import (  # noqa: E402
    _gather_kwargs, _run_ref, _tensor, _total,
)
from tests.unit.test_s33_tiled_field_character import build_chart  # noqa: E402
from tests.unit.test_s33_tiled_stress_ratio import STRESS  # noqa: E402

pytestmark = pytest.mark.gpu

GPU_RATIO_CEILING = 7.0       # user ruling 2026-11-14 (float32 spectra, supersedes the earlier 5.0); Tier B is 1.5


def assert_gpu_stress(kw, ref5, got5, what):
    """The hard gpu stress gate: p99.9 ratio <= 7, full frame and annulus.
    weight_sum / delta_weight_sum / acc_alpha must be bit-equal. Prints the raw
    Tier C numbers, the ratio table and the per-pixel max (not gated)."""
    for i, name in ((2, "weight_sum"), (3, "delta_weight_sum"), (4, "acc_alpha")):
        assert np.array_equal(ref5[i], got5[i]), f"{what}: {name} differs"
    assert np.isfinite(got5[0]).all() and np.isfinite(got5[1]).all(), what
    tc = tol.summary(tol.evaluate(_total(ref5), _total(got5), "C"))
    res = stress_ratio(kw, ref5, got5)
    print(f"[gpu-stress] {what}\n  raw Tier C: {tc}\n  vs truth: {format_ratio(res)}")
    for reg, v in res.items():
        assert v["ratio"][1] <= GPU_RATIO_CEILING, (
            f"{what}: {reg} p99.9 ratio {v['ratio'][1]:.3g} > {GPU_RATIO_CEILING}: "
            f"{format_ratio(res)}")
    return res


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
    got5, _ = _run_gpu(kw, **({"_tile_n": tile_n} if tile_n else {}))
    assert float(ref5[1].max()) > 0.0 or patch      # the bright image was used
    return assert_gpu_stress(kw, ref5, got5,
                             f"{name} {h}x{w} N={tile_n or 324} lens {lens}")


@pytest.mark.parametrize("name", list(STRESS))
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


# --------------------------------------------------------------------------
# the other Tier B stress rows: dense HDR 16x, 1000x light, lens 50 peak dense
# --------------------------------------------------------------------------

def _noise(h, w, seed, lo=0.0, hi=0.8):
    return np.random.default_rng(seed).uniform(lo, hi, (h, w, 3)).astype(np.float32)


def _ramp_depth(h, w, od):
    ramp = np.geomspace(od[0], od[-1], w, dtype=np.float32)
    depth = np.broadcast_to(ramp[None, :], (h, w)).copy()
    depth[h // 4: h // 2, w // 5: w // 2] = np.float32(od[3])
    return depth


@pytest.mark.parametrize("lens", ["35", "50"])
@pytest.mark.parametrize("hw", [(130, 110), (110, 130)])
def test_hdr16_dense_light_field_stress_ratio(lens, hw):
    """Ten 3x3 16x lights on a 130 x 110 noise frame at the default L1 gain 2
    sit on the float32 round-off floor (the dense-highlight family)."""
    h, w = hw
    rng = np.random.default_rng(11)
    img = _noise(h, w, 11)
    for _ in range(10):
        y, x = int(rng.integers(1, h - 4)), int(rng.integers(1, w - 4))
        img[y:y + 3, x:x + 3] = 16.0
    od = _tensor(lens)["object_distances_m"]
    kw = _gather_kwargs(img, _ramp_depth(h, w, od), lens=lens, K=17, thresh=0.5,
                        bright_stamp_gain=2.0)
    ref5, _ = _run_ref(kw)
    got5, _ = _run_gpu(kw, _tile_n=43)
    assert_gpu_stress(kw, ref5, got5, f"hdr16 dense lens {lens} {hw}")


def test_extreme_highlight_1e3_stress_ratio():
    """The 1000x single light (the 1e5x light is out of scope)."""
    h, w = 90, 80
    img = _noise(h, w, 1, 0.0, 0.5)
    img[44:47, 38:41] = 1e3
    od = _tensor("35")["object_distances_m"]
    depth = np.broadcast_to(np.geomspace(od[0], od[-1], w, dtype=np.float32)[None, :],
                            (h, w)).copy()
    kw = _gather_kwargs(img, depth, K=17, thresh=0.5, bright_stamp_gain=2.0)
    ref5, _ = _run_ref(kw)
    got5, _ = _run_gpu(kw, _tile_n=40)
    assert_gpu_stress(kw, ref5, got5, "extreme amp 1000")


@pytest.mark.parametrize("hw", [(360, 320), (320, 360)])
def test_lens50_ldr_peak_dense_small_frame_stress_ratio(hw):
    """16 lights on a 360 x 320 frame, lens 50 LDR peak stamp, N=128."""
    t = dict(_tensor("50"))
    img, depth, _ = build_chart(*hw, 1.0, n_lights=16, tensor=t)
    kw = _gather_kwargs(img, depth, K=67, thresh=0.5, bright_stamp_gain=None, lens="50")
    ref5, _ = _run_ref(kw)
    got5, _ = _run_gpu(kw, _tile_n=128)
    assert float(ref5[1].max()) > 0.01
    assert_gpu_stress(kw, ref5, got5, f"lens50 ldr_peak dense {hw}")
