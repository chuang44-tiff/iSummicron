"""S33 Tier B WP2, gate G-B5: field-character coverage, hard gate.

Multi-slab impulse charts: 3x3 point lights at all 12 radial bins, >= 16
azimuths (corner and wedge-boundary angles included), every slab node d1..d10
plus between-node depths (nearest-light cells), landscape and portrait, at
K=67 / N=324 and K=183 / N=512. The tiled engine's gather is compared with the
reference gather under the Tier B ceilings (full frame and the field-edge
annulus), via the test-local copy of the frozen metric.

Hard set (user ruling R-B2 revised, 2026-11-13): LDR 1.0 peak with 3x3 lights
(``bright_stamp_gain=None``) and HDR 16x at L1 gain 2 and 4, both lenses, both
orientations, hard Tier B ceilings (full frame and annulus), including at
production N. The stress cases (L1 gain >= 8, 50x/100x lights, peak stamp on HDR,
LDR 41x41 patches) are gated by the ruled stress ratio (tiled error vs a float64
truth) / (reference error vs the same truth) <= 1.5
(``tests/unit/test_s33_tiled_stress_ratio.py``, ``tests/fixtures/truth.py``).

FAST variants force ``_tile_n`` so a small frame holds >= 3x3 non-multiple
tiles; the production-N cases at 2048-class frames are SLOW.
"""
from __future__ import annotations

import numpy as np
import pytest

from summicron.render import _tile_schedule as ts
from tests.fixtures import tolerance as tol
from tests.fixtures.truth import truth_gather  # noqa: F401  (float64 truth; re-exported)
from tests.unit.test_s33_tiled_core import (  # noqa: F401  (shared helpers)
    _gather_kwargs, _geometry, _run_ref, _run_tiled, _tensor, _total,
)

AMPS = {                       # name -> (amplitude, bright_stamp_gain)
    "ldr_peak": (1.0, None),
    "hdr16_l1g2": (16.0, 2.0),
    "hdr16_l1g4": (16.0, 4.0),
}


# --------------------------------------------------------------------------
# chart
# --------------------------------------------------------------------------

def build_chart(h, w, amp, *, n_lights, tensor, margin=2):
    """Returns ``(img, depth, lights)``; ``lights`` = list of dicts with the
    pixel centre, achieved ri bin, azimuth (deg) and depth."""
    od = np.asarray(tensor["object_distances_m"], np.float64)
    edges = np.asarray(tensor["radial_edges"], np.float64)
    rc = 0.5 * (edges[:-1] + edges[1:])
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    maxr = float(np.hypot(cy, cx))
    # depths: every node d1..d10 first, then between-node geometric means
    nodes = [float(od[i]) for i in range(1, 11)]
    mids = [float(np.sqrt(od[i] * od[i + 1])) for i in range(0, 10)]
    depths = [x for pair in zip(nodes, mids) for x in pair]
    img = np.full((h, w, 3), 0.02, np.float32)
    lights = []
    for k in range(n_lights):
        ri = k % len(rc)
        if ri >= 5:
            # bins 5..11 mostly exist near the frame corners: one corner per
            # light (cycling), with a small per-light angle offset so the
            # azimuths stay distinct
            quad = (ri - 5 + k // len(rc)) % 4
            sy, sx = (1, 1, -1, -1)[quad], (1, -1, 1, -1)[quad]
            th = np.arctan2(sy * cy, sx * cx) + np.radians(2.5 * ((11 - ri) // 4))
        else:
            th = k * 2 * np.pi / n_lights
        R = rc[ri] * maxr
        # keep the light inside the frame (margin) along this azimuth
        sy_, cx_ = np.sin(th), np.cos(th)
        lim = min(
            (cy - margin) / abs(sy_) if abs(sy_) > 1e-9 else np.inf,
            (cx - margin) / abs(cx_) if abs(cx_) > 1e-9 else np.inf,
        )
        R = min(R, lim)
        y = int(round(cy + R * sy_))
        x = int(round(cx + R * cx_))
        img[y - 1:y + 2, x - 1:x + 2] = amp
        rf = float(np.hypot(y - cy, x - cx) / maxr)
        got_ri = int(np.clip(np.searchsorted(edges, rf, side="right") - 1, 0, len(rc) - 1))
        az = float(np.degrees(np.arctan2(y - cy, x - cx)))
        lights.append(dict(y=y, x=x, ri=got_ri, az=az, depth=depths[k % len(depths)]))
    ly = np.array([l_["y"] for l_ in lights], np.float32)
    lx = np.array([l_["x"] for l_ in lights], np.float32)
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    best = np.full((h, w), np.inf, np.float32)
    depth = np.empty((h, w), np.float32)
    for i, l_ in enumerate(lights):
        dd = (ys - ly[i]) ** 2 + (xs - lx[i]) ** 2
        m = dd < best
        best[m] = dd[m]
        depth[m] = np.float32(l_["depth"])
    return img, depth, lights


def chart_coverage(lights):
    ris = {l_["ri"] for l_ in lights}
    azs = {round(l_["az"]) for l_ in lights}
    return ris, azs


def judge(ref5, got5, lights, what):
    """Tier B on the total image plus per-light local max e (reported)."""
    res = tol.evaluate(_total(ref5), _total(got5), "B")
    assert res["ok"], f"{what}: {tol.summary(res)}"
    for i, name in ((2, "weight_sum"), (3, "delta_weight_sum"), (4, "acc_alpha")):
        assert np.array_equal(ref5[i], got5[i]), f"{what}: {name} differs"
    local = tol.local_max_e(_total(ref5), _total(got5),
                            [(l_["y"], l_["x"]) for l_ in lights])
    assert max(local) <= tol.TIERS["B"]["max_e"], f"{what}: light max e {max(local):.3g}"
    print(f"{what}: {tol.summary(res)}; local max e per light "
          f"max={max(local):.3g} min={min(local):.3g}")
    return res, local


def _run_chart(h, w, K, amp_name, *, n_lights, tile_n=None, lens="35"):
    amp, gain = AMPS[amp_name]
    t = dict(_tensor(lens))
    img, depth, lights = build_chart(h, w, amp, n_lights=n_lights, tensor=t)
    ris, azs = chart_coverage(lights)
    assert ris == set(range(12)), f"chart misses radial bins: {sorted(set(range(12)) - ris)}"
    assert len(azs) >= 16, f"chart has {len(azs)} azimuths"
    kw = _gather_kwargs(img, depth, K=K, thresh=0.5, bright_stamp_gain=gain, lens=lens)
    ref5, _ = _run_ref(kw)
    got5, _ = _run_tiled(kw, **({"_tile_n": tile_n} if tile_n else {}))
    # the chart must really exercise the bright image (not vacuous)
    assert float(ref5[1].max()) > 0.01
    return ref5, got5, lights, kw


# --------------------------------------------------------------------------
# FAST: forced small tiles, both orientations
# --------------------------------------------------------------------------

def test_chart_builder_covers_everything():
    t = _tensor("35")
    for (h, w) in ((260, 220), (220, 260), (1536, 2048), (2048, 1536)):
        for n in (16, 32):
            _, depth, lights = build_chart(h, w, 1.0, n_lights=n, tensor=t)
            ris, azs = chart_coverage(lights)
            assert ris == set(range(12)), (h, w, n, sorted(ris))
            assert len(azs) >= 16, (h, w, n, len(azs))
    # all 10 interior nodes and between-node depths appear (32 lights)
    od = np.asarray(t["object_distances_m"], np.float64)
    _, depth, _ = build_chart(300, 260, 1.0, n_lights=32, tensor=t)
    vals = np.unique(depth)
    for i in range(1, 11):
        assert np.isclose(vals, od[i], rtol=1e-5).any(), f"node d{i} missing"
    assert sum(not np.isclose(od, v, rtol=1e-4).any() for v in vals) >= 8


@pytest.mark.parametrize("hw", [(260, 220), (220, 260)])
@pytest.mark.parametrize("amp_name", ["ldr_peak", "hdr16_l1g2"])
def test_chart_k67_forced_tiles(hw, amp_name):
    ref5, got5, lights, _ = _run_chart(*hw, 67, amp_name, n_lights=16, tile_n=96)
    _, tiles = ts.plan_tiles(*hw, 67, 96)
    assert len({t[0] for t in tiles}) >= 3 and len({t[2] for t in tiles}) >= 3
    judge(ref5, got5, lights, f"K=67/N=96 {hw} {amp_name}")


# LDR peak is not run here: 12 lights on a 220x190 frame at K=183 put every light's
# 183 px bokeh on every pixel, the dense-highlight stress family (measured p99.9
# 1.24e-6 for the peak stamp, float32 round-off; R-B2). The sparse K=183 peak
# chart is the SLOW production-N case below.
@pytest.mark.parametrize("hw", [(220, 190), (190, 220)])
@pytest.mark.parametrize("amp_name", ["hdr16_l1g2"])
def test_chart_k183_forced_tiles(hw, amp_name):
    """K=183 with a tensor dict that has no corpus_ref_long_edge (pix_scale 1.0),
    _tile_n=256 (T=74) on a small frame: 3x3 tiles."""
    amp, gain = AMPS[amp_name]
    t = dict(_tensor("35"))
    t.pop("corpus_ref_long_edge", None)
    img, depth, lights = build_chart(*hw, amp, n_lights=12, tensor=t)
    kw = _gather_kwargs(img, depth, K=183, thresh=0.5, bright_stamp_gain=gain, tensor=t)
    ref5, _ = _run_ref(kw)
    got5, _ = _run_tiled(kw, _tile_n=256)
    assert float(ref5[1].max()) > 0.01
    judge(ref5, got5, lights, f"K=183/N=256 {hw} {amp_name}")


# --------------------------------------------------------------------------
# Production N (K=67 -> N=324, T=258), L1 gain 2 and 4 hard set
# --------------------------------------------------------------------------

def _production_n_case(lens, hw, amp_name):
    ref5, got5, lights, _ = _run_chart(*hw, 67, amp_name, n_lights=20, lens=lens)
    _, tiles = ts.plan_tiles(*hw, 67, ts.tile_n_for_ksize(67))
    assert len(tiles) >= 4
    judge(ref5, got5, lights, f"production N lens {lens} {hw} {amp_name}")


@pytest.mark.parametrize("lens,hw,amp_name", [
    ("50", (700, 640), "hdr16_l1g4"),
    ("35", (640, 700), "hdr16_l1g2"),
])
def test_chart_production_n_gain_2_and_4(lens, hw, amp_name):
    """Hard gate at production N (324) on a frame holding 3 x 3 tiles: HDR 16x
    at L1 gain 2 and 4 (two of the eight rows; the rest are SLOW)."""
    _production_n_case(lens, hw, amp_name)


@pytest.mark.slow
@pytest.mark.parametrize("lens", ["35", "50"])
@pytest.mark.parametrize("hw", [(700, 640), (640, 700)])
@pytest.mark.parametrize("amp_name", ["ldr_peak", "hdr16_l1g2", "hdr16_l1g4"])
def test_chart_production_n_matrix(lens, hw, amp_name):
    """Both lenses x both orientations x LDR peak / L1 gain 2 / L1 gain 4."""
    _production_n_case(lens, hw, amp_name)


# --------------------------------------------------------------------------
# SLOW: production N on 2048-class frames
# --------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.parametrize("hw", [(2048, 1536), (1536, 2048)])
@pytest.mark.parametrize("amp_name", ["ldr_peak", "hdr16_l1g2"])
def test_chart_k67_n324_2048(hw, amp_name):
    ref5, got5, lights, _ = _run_chart(*hw, 67, amp_name, n_lights=32)
    judge(ref5, got5, lights, f"K=67/N=324 {hw} {amp_name}")


@pytest.mark.slow
@pytest.mark.parametrize("hw", [(900, 820), (820, 900)])
@pytest.mark.parametrize("amp_name", ["ldr_peak", "hdr16_l1g2"])
def test_chart_k183_n512(hw, amp_name):
    """K=183 / N=512 (T=330): production table, 3x3 tiles, pix_scale 1.0."""
    amp, gain = AMPS[amp_name]
    t = dict(_tensor("35"))
    t.pop("corpus_ref_long_edge", None)
    img, depth, lights = build_chart(*hw, amp, n_lights=20, tensor=t)
    ris, azs = chart_coverage(lights)
    assert ris == set(range(12)) and len(azs) >= 16
    kw = _gather_kwargs(img, depth, K=183, thresh=0.5, bright_stamp_gain=gain, tensor=t)
    ref5, _ = _run_ref(kw)
    got5, _ = _run_tiled(kw)
    _, tiles = ts.plan_tiles(*hw, 183, 512)
    assert len(tiles) >= 4
    judge(ref5, got5, lights, f"K=183/N=512 {hw} {amp_name}")


# --------------------------------------------------------------------------
# float64 truth (``tests.fixtures.truth``), sanity-checked here
# --------------------------------------------------------------------------

def test_truth_gather_agrees_with_reference_on_a_small_chart():
    """Validates the float64 truth helper the probe relies on: reference and
    tiled both sit within ~1e-5 of it on a realistic-density chart."""
    ref5, got5, _, kw = _run_chart(220, 260, 67, "hdr16_l1g2", n_lights=16, tile_n=96)
    td, tb = truth_gather(kw)
    truth = (td + tb)
    for name, five in (("reference", ref5), ("tiled", got5)):
        tot = _total(five).astype(np.float64)
        e = np.abs(tot - truth) / np.maximum(truth, 1.0)
        assert float(e.max()) < 1e-4, (name, float(e.max()))
