"""S33 Tier C WP2: stdout/stats (G-C4), field character (G-C5a) and uncovered
configurations (G-C6), all through ``gpu_layered_gather`` directly.

The engine is not wired into ``render()`` yet (WP3). Where a whole ``render()``
is needed (G-C4) ``convolve._layered_gather_diffuse`` is monkeypatched to the MLX
gather, the same technique as ``test_s33_tiled_stdout_stats.py``.

Every case is judged against the in-tree reference gather with the Tier C
ceilings (full frame and field-edge annulus) via the test-local copy of the
frozen metric, plus the per-impulse local metric (R-C12: e_imp <= 1e-4 on the
impulse's K x K box, L1 within 1e-5, centroid within 0.01 px, tail sign equal).
No GPU output bytes or md5 are stored (G-C0).

The gather's ``_MAX_ACTIVE_SLABS`` RuntimeWarning is a Python warning issued
inside the gather: the pipeline does not copy it into
``RenderDiagnostics.warnings`` (only its own collected messages go there), so it
is compared through ``warnings.catch_warnings`` exactly as Tier B did.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

import summicron.render.convolve as cv  # noqa: E402
from summicron.render import _engine_gpu as eg  # noqa: E402
from summicron.render import _tile_schedule as ts  # noqa: E402
from tests.fixtures import tolerance as tol  # noqa: E402
from tests.fixtures.synthetic_tensor import make_synthetic_tensor  # noqa: E402
from tests.unit.test_s33_gpu_core import (  # noqa: E402
    _assert_impulses, _assert_tier_c, _imp_metrics, _run_gpu, _same5,
)
from tests.unit.test_s33_tiled_core import (  # noqa: E402
    _gather_kwargs, _run_ref, _tensor, _total,
)
from tests.unit.test_s33_tiled_field_character import (  # noqa: E402
    AMPS, build_chart, chart_coverage,
)

pytestmark = pytest.mark.gpu

C = tol.TIERS["C"]


# --------------------------------------------------------------------------
# G-C4: stdout, stats, progress, warnings (through render())
# --------------------------------------------------------------------------

def _scene(h=150, w=170, seed=3):
    rng = np.random.default_rng(seed)
    img = rng.uniform(0.0, 0.7, (h, w, 3)).astype(np.float32)
    for _ in range(6):
        y, x = int(rng.integers(3, h - 4)), int(rng.integers(3, w - 4))
        img[y:y + 3, x:x + 3] = 1.0
    od = _tensor("35")["object_distances_m"]
    ramp = np.geomspace(od[0], od[-1], w, dtype=np.float32)
    depth = np.broadcast_to(ramp[None, :], (h, w)).copy()
    depth[h // 4: h // 2, w // 5: w // 2] = np.float32(od[3])
    return img, depth


def _render(engine, img, depth, *, tile_n=96, cap=None, **kw):
    """render() with the gather chosen by ``engine`` ("reference" or "gpu")."""
    tensor = dict(_tensor("35"), corpus_ref_long_edge=464)   # pix_scale 0.366 -> K=67
    lines: list[str] = []
    stats: dict = {}
    got: dict = {}
    real_gather = cv._layered_gather_diffuse
    real_state = cv._build_image_fft_state
    state_calls = {"n": 0}

    def gather(**k):
        if cap is not None:
            k["active_slab_cap"] = cap
        if engine == "gpu":
            k["fft_state"] = None
            out = eg.gpu_layered_gather(**k, _tile_n=tile_n)
        else:
            out = real_gather(**k)
        got["five"] = out
        return out

    def state(*a, **k):
        state_calls["n"] += 1
        return None if engine == "gpu" else real_state(*a, **k)

    cv._layered_gather_diffuse = gather
    cv._build_image_fft_state = state
    try:
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            _, lin = cv.render(
                img, tensor, depth_m=depth, hdr_source="none",
                bright_thresh=kw.pop("bright_thresh", 0.9),
                bright_stamp_gain=kw.pop("bright_stamp_gain", 2.0),
                on_progress=lines.append, stats=stats, **kw,
            )
    finally:
        cv._layered_gather_diffuse = real_gather
        cv._build_image_fft_state = real_state
    warns = [(w.category, str(w.message)) for w in rec
             if "_MAX_ACTIVE_SLABS" in str(w.message)]
    return dict(lin=lin, stats=stats, lines=lines, warns=warns,
                five=got["five"], state_calls=state_calls["n"])


def _pair(**kw):
    img, depth = _scene()
    return _render("reference", img, depth, **kw), _render("gpu", img, depth, **kw)


def test_five_tuple_accumulators_array_equal_and_tier_c():
    ref, got = _pair()
    for i, name in ((2, "weight_sum"), (3, "delta_weight_sum"), (4, "acc_alpha")):
        assert np.array_equal(ref["five"][i], got["five"][i]), name
    assert np.array_equal(ref["stats"]["acc_alpha"], got["stats"]["acc_alpha"])
    res = tol.evaluate(ref["five"][0] + ref["five"][1], got["five"][0] + got["five"][1], "C")
    assert res["ok"], tol.summary(res)
    res = tol.evaluate(ref["lin"], got["lin"], "C")
    assert res["ok"], tol.summary(res)
    assert got["state_calls"] == 1                    # the swap really took effect


def test_progress_sequence_and_stdout_lines_identical():
    ref, got = _pair()
    assert ref["lines"] == got["lines"]
    joined = "\n".join(got["lines"])
    assert "layered_gather: active_slabs=" in joined
    assert "bright-pass: thresh=" in joined
    slab_lines = [x for x in got["lines"] if x.startswith("  Slab ")]
    assert slab_lines == [f"  Slab {i}/11" for i in range(1, 12)]
    # the gpu engine adds no stdout line at all (no version note)
    assert not any("mlx" in x.lower() or "gpu" in x.lower() for x in got["lines"])


def test_bright_energy_ratio_digits_unchanged():
    """A changed digit in bright_energy_ratio on a real scene is a STOP."""
    ref, got = _pair()
    pick = lambda r: [x for x in r["lines"] if "bright_energy_ratio=" in x]   # noqa: E731
    assert len(pick(ref)) == 1
    assert pick(ref) == pick(got)
    assert float(pick(ref)[0].rsplit("bright_energy_ratio=", 1)[1]) > 0.0


def test_stats_dict_keys_and_scalars_equal():
    ref, got = _pair()
    assert set(ref["stats"]) == set(got["stats"])
    for k, v in ref["stats"].items():
        g = got["stats"][k]
        if isinstance(v, np.ndarray):
            if k == "acc_alpha":
                assert np.array_equal(v, g), k
            else:
                assert np.allclose(v, g, rtol=0, atol=1e-4), k
        elif isinstance(v, float):
            assert g == pytest.approx(v, abs=1e-5), k
        else:
            assert g == v, k


def test_max_active_slabs_warning_text_and_count_identical_through_render():
    ref, got = _pair(cap=2)
    assert len(ref["warns"]) == 1
    assert ref["warns"] == got["warns"]
    assert ref["lines"] == got["lines"]


def test_progress_sequence_with_empty_slabs():
    """Slabs the reference skips (full-frame max < _EMPTY_SKIP_THRESH) still
    print their ``Slab i/D`` line and are not counted active."""
    img, _ = _scene()
    od = _tensor("35")["object_distances_m"]
    depth = np.full(img.shape[:2], od[4], np.float32)
    ref = _render("reference", img, depth)
    got = _render("gpu", img, depth)
    assert ref["lines"] == got["lines"]
    active = [x for x in got["lines"] if "layered_gather:" in x][0]
    assert int(active.split("active_slabs=")[1].split("/")[0]) < 11
    for i in (2, 3, 4):
        assert np.array_equal(ref["five"][i], got["five"][i])


# --------------------------------------------------------------------------
# G-C5a: charts at the Tier C ceilings
# --------------------------------------------------------------------------

def judge_c(ref5, got5, lights, K, what):
    """Tier C on the total and the diffuse image, exact accumulators, the light
    local max e, and the per-impulse metric on every light."""
    _assert_tier_c(ref5, got5, what)
    local = tol.local_max_e(_total(ref5), _total(got5),
                            [(lt["y"], lt["x"]) for lt in lights])
    assert max(local) <= C["max_e"], f"{what}: light max e {max(local):.3g}"
    res = tol.evaluate(_total(ref5), _total(got5), "C")
    print(f"{what}: {tol.summary(res)}; local max e per light max={max(local):.3g}")
    _assert_impulses(ref5, got5, [(lt["y"], lt["x"]) for lt in lights], K, what)


def _chart_case(h, w, K, amp_name, *, n_lights, tile_n=None, lens="35", amp_gain=None,
                tensor=None, depth_fn=None):
    amp, gain = amp_gain if amp_gain is not None else AMPS[amp_name]
    t = dict(tensor if tensor is not None else _tensor(lens))
    img, depth, lights = build_chart(h, w, amp, n_lights=n_lights, tensor=t)
    if depth_fn is not None:
        depth = depth_fn(h, w, np.asarray(t["object_distances_m"], np.float64))
    kw = _gather_kwargs(img, depth, K=K, thresh=0.5, bright_stamp_gain=gain,
                        lens=lens, tensor=t)
    ref5, _ = _run_ref(kw)
    got5, _ = _run_gpu(kw, **({"_tile_n": tile_n} if tile_n else {}))
    assert float(ref5[1].max()) > 0.0 or K <= 17          # the bright image was used
    return ref5, got5, lights, kw


def test_chart_builder_covers_everything():
    t = _tensor("35")
    for (h, w) in ((260, 220), (220, 260), (1536, 2048), (2048, 1536)):
        _, _, lights = build_chart(h, w, 1.0, n_lights=32, tensor=t)
        ris, azs = chart_coverage(lights)
        assert ris == set(range(12)) and len(azs) >= 16, (h, w)


@pytest.mark.parametrize("hw", [(260, 220), (220, 260)])
@pytest.mark.parametrize("amp_name", ["ldr_peak", "hdr16_l1g2", "hdr16_l1g4"])
def test_chart_k67_forced_tiles(hw, amp_name):
    ref5, got5, lights, _ = _chart_case(*hw, 67, amp_name, n_lights=16, tile_n=96)
    ris, azs = chart_coverage(lights)
    assert ris == set(range(12)) and len(azs) >= 16
    _, tiles = ts.plan_tiles(*hw, 67, 96)
    assert len({t[0] for t in tiles}) >= 3 and len({t[2] for t in tiles}) >= 3
    judge_c(ref5, got5, lights, 67, f"K=67/N=96 {hw} {amp_name}")


@pytest.mark.parametrize("hw", [(220, 190), (190, 220)])
@pytest.mark.parametrize("amp_name", ["hdr16_l1g2", "hdr16_l1g4"])
def test_chart_k183_forced_tiles(hw, amp_name):
    """K=183, pix_scale 1.0 (a tensor dict without corpus_ref_long_edge),
    _tile_n=256 on a small frame: 3x3 tiles."""
    t = dict(_tensor("35"))
    t.pop("corpus_ref_long_edge", None)
    ref5, got5, lights, _ = _chart_case(*hw, 183, amp_name, n_lights=12, tile_n=256,
                                        tensor=t)
    judge_c(ref5, got5, lights, 183, f"K=183/N=256 {hw} {amp_name}")


@pytest.mark.parametrize("lens,hw,amp_name", [
    ("50", (700, 640), "hdr16_l1g4"),
    ("35", (640, 700), "hdr16_l1g2"),
])
def test_chart_production_n_gain_2_and_4(lens, hw, amp_name):
    """Production N (324) on a frame holding 3 x 3 tiles."""
    ref5, got5, lights, _ = _chart_case(*hw, 67, amp_name, n_lights=20, lens=lens)
    _, tiles = ts.plan_tiles(*hw, 67, ts.tile_n_for_ksize(67))
    assert len(tiles) >= 4
    judge_c(ref5, got5, lights, 67, f"production N lens {lens} {hw} {amp_name}")


@pytest.mark.slow
@pytest.mark.parametrize("lens", ["35", "50"])
@pytest.mark.parametrize("hw", [(700, 640), (640, 700)])
@pytest.mark.parametrize("amp_name", ["ldr_peak", "hdr16_l1g2", "hdr16_l1g4"])
def test_chart_production_n_matrix(lens, hw, amp_name):
    ref5, got5, lights, _ = _chart_case(*hw, 67, amp_name, n_lights=20, lens=lens)
    judge_c(ref5, got5, lights, 67, f"production N lens {lens} {hw} {amp_name}")


@pytest.mark.slow
@pytest.mark.parametrize("hw", [(2048, 1536), (1536, 2048)])
@pytest.mark.parametrize("amp_name", ["ldr_peak", "hdr16_l1g2"])
def test_chart_k67_n324_2048(hw, amp_name):
    ref5, got5, lights, _ = _chart_case(*hw, 67, amp_name, n_lights=32)
    judge_c(ref5, got5, lights, 67, f"K=67/N=324 {hw} {amp_name}")


@pytest.mark.slow
@pytest.mark.parametrize("hw", [(900, 820), (820, 900)])
@pytest.mark.parametrize("amp_name", ["ldr_peak", "hdr16_l1g2"])
def test_chart_k183_n512(hw, amp_name):
    """K=183 / N=512 (T=330): production table, 3x3 tiles, pix_scale 1.0."""
    t = dict(_tensor("35"))
    t.pop("corpus_ref_long_edge", None)
    ref5, got5, lights, _ = _chart_case(*hw, 183, amp_name, n_lights=20, tensor=t)
    _, tiles = ts.plan_tiles(*hw, 183, 512)
    assert len(tiles) >= 4
    judge_c(ref5, got5, lights, 183, f"K=183/N=512 {hw} {amp_name}")


# --------------------------------------------------------------------------
# G-C5a: the 12-radius x 16-azimuth impulse grid (R-C12 per-impulse metric)
# --------------------------------------------------------------------------

RADII = np.array([0.08, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.88, 0.94, 0.98, 1.0])
NAZ = 16


def grid_positions(h, w, wedge_boundary=False):
    """Impulses on 12 radii x 16 azimuths out to the frame corners; on the
    wedge-boundary set the azimuths sit on / next to the 45-degree wedge edges."""
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    maxr = float(np.hypot(cy, cx))
    pos = []
    for rf in RADII:
        for ia in range(NAZ):
            if wedge_boundary:
                th = ia * 2 * np.pi / NAZ + 1e-3
            else:
                th = (ia + 0.5) * 2 * np.pi / NAZ
            y = int(np.clip(round(cy + rf * maxr * np.sin(th)), 6, h - 7))
            x = int(np.clip(round(cx + rf * maxr * np.cos(th)), 6, w - 7))
            if (y, x) not in pos:
                pos.append((y, x))
    return pos


def _groups(pos, sep):
    gs: list[list] = []
    for p in pos:
        for g in gs:
            if all(max(abs(p[0] - q[0]), abs(p[1] - q[1])) >= sep for q in g):
                g.append(p)
                break
        else:
            gs.append([p])
    return gs


def _grid_depth_vals(od):
    return [od[i] for i in range(1, 11)] + [np.sqrt(od[i] * od[i + 1]) for i in (1, 3, 5, 8)]


def _grid_case(h, w, K, amp, gain, *, wedge_boundary=False, tile_n=None, lens="35",
               max_groups=None):
    """Impulse groups (pairwise >= K+4 px apart, so no box is contaminated by a
    neighbour), depth patches cycling over every slab node and between-node
    values. Every group is compared with the reference gather; returns the
    worst per-impulse numbers."""
    od = _tensor(lens)["object_distances_m"]
    dvals = _grid_depth_vals(od)
    gs = _groups(grid_positions(h, w, wedge_boundary), K + 4)
    worst = dict(e_imp=0.0, l1=0.0, shift=0.0, n=0)
    for gi, g in enumerate(gs[:max_groups]):
        img = np.full((h, w, 3), 0.02, np.float32)
        depth = np.full((h, w), od[5], np.float32)
        for k, (y, x) in enumerate(g):
            img[y - 1:y + 2, x - 1:x + 2] = amp
            dv = dvals[(k + gi * 5) % len(dvals)]
            depth[max(0, y - K // 2 - 2):y + K // 2 + 3,
                  max(0, x - K // 2 - 2):x + K // 2 + 3] = np.float32(dv)
        kw = _gather_kwargs(img, depth, K=K, thresh=0.5, bright_stamp_gain=gain, lens=lens)
        ref5, _ = _run_ref(kw)
        got5, _ = _run_gpu(kw, **({"_tile_n": tile_n} if tile_n else {}))
        _assert_tier_c(ref5, got5, f"grid {h}x{w} grp{gi}")
        _assert_impulses(ref5, got5, g, K, f"grid {h}x{w} grp{gi} n={len(g)}")
        ms = _imp_metrics(_total(ref5), _total(got5), g, K)
        worst["e_imp"] = max(worst["e_imp"], max(m[0] for m in ms))
        worst["l1"] = max(worst["l1"], max(m[1] for m in ms))
        worst["shift"] = max(worst["shift"], max(m[2] for m in ms))
        worst["n"] += len(ms)
    print(f"grid {h}x{w} amp={amp} gain={gain} wb={wedge_boundary}: {worst}")
    return worst


def test_impulse_grid_covers_radii_azimuths_and_corners():
    for hw in ((300, 260), (260, 300), (760, 1000)):
        pos = grid_positions(*hw)
        assert len(pos) >= 100
        cy, cx = (hw[0] - 1) / 2.0, (hw[1] - 1) / 2.0
        maxr = np.hypot(cy, cx)
        rfs = [np.hypot(y - cy, x - cx) / maxr for y, x in pos]
        assert max(rfs) > 0.85 and min(rfs) < 0.12
        assert len({round(np.degrees(np.arctan2(y - cy, x - cx)) / 22.5) for y, x in pos}) >= 16


@pytest.mark.parametrize("hw", [(300, 260), (260, 300)])
def test_impulse_grid_forced_tiles_l1_stamp(hw):
    w = _grid_case(*hw, 67, 4.0, 2.0, tile_n=96, max_groups=3)
    assert w["n"] >= 30


def test_impulse_grid_forced_tiles_peak_stamp():
    w = _grid_case(300, 260, 67, 1.0, None, tile_n=96, max_groups=3)
    assert w["n"] >= 30


def test_impulse_grid_forced_tiles_wedge_boundary_set():
    w = _grid_case(260, 300, 67, 4.0, 2.0, wedge_boundary=True, tile_n=96, max_groups=3)
    assert w["n"] >= 30


@pytest.mark.slow
@pytest.mark.parametrize("hw", [(760, 1000), (1000, 760)])
@pytest.mark.parametrize("wb,name,amp,gain", [
    (False, "L1g2_amp4", 4.0, 2.0), (False, "peak_ldr1", 1.0, None),
    (False, "hdr50_l1g2", 50.0, 2.0), (False, "hdr100_l1g2", 100.0, 2.0),
    (True, "L1g2_amp4", 4.0, 2.0),      # the wedge-boundary set runs with the L1 stamp only
])
def test_impulse_grid_production_n(hw, wb, name, amp, gain):
    """Production N (324): the full grid, portrait and landscape, L1 and peak
    stamps, 50x and 100x HDR impulses, the wedge-boundary set (L1 stamp only)."""
    _grid_case(*hw, 67, amp, gain, wedge_boundary=wb)


@pytest.mark.slow
def test_impulse_grid_lens50_production_n():
    _grid_case(760, 1000, 67, 4.0, 2.0, lens="50")


# --------------------------------------------------------------------------
# G-C6: uncovered configurations
# --------------------------------------------------------------------------

# K -> forced FAST N and a frame holding >= 3x3 tiles (light density kept at or
# below about 1 per 3500 px, as in Tier B).
FAST_K = {17: (40, (200, 190)), 33: (64, (220, 200)), 67: (96, (240, 220)),
          131: (192, (190, 200))}


@pytest.mark.parametrize("K", [17, 33, 67, 131])
def test_downres_k_forced_tiles(K):
    N, (h, w) = FAST_K[K]
    _, tiles = ts.plan_tiles(h, w, K, N)
    assert len({t[0] for t in tiles}) >= 3 and len({t[2] for t in tiles}) >= 3
    ref5, got5, lights, _ = _chart_case(h, w, K, "hdr16_l1g2", tile_n=N,
                                        n_lights=6 if K < 67 else 12)
    _assert_tier_c(ref5, got5, f"downres K={K}")
    print(f"downres K={K} N={N} {h}x{w}: "
          f"{tol.summary(tol.evaluate(_total(ref5), _total(got5), 'C'))}")


@pytest.mark.slow
@pytest.mark.parametrize("K,hw", [(17, (640, 700)), (33, (640, 700)),
                                  (67, (700, 640)), (131, (820, 780))])
def test_downres_k_production_n(K, hw):
    ref5, got5, lights, _ = _chart_case(*hw, K, "hdr16_l1g2", tile_n=None, n_lights=16)
    _, tiles = ts.plan_tiles(*hw, K, ts.tile_n_for_ksize(K))
    assert len(tiles) >= 4
    judge_c(ref5, got5, lights, K, f"downres K={K} {hw}")


def test_slab_batch_size_invariance_gpu_array_equal():
    """slab_batch_size 1 / 3 / D / default: the gpu output is byte-equal (same
    configuration), and inside Tier C of the reference."""
    img, depth, _ = build_chart(240, 220, 16.0, n_lights=12, tensor=_tensor("35"))
    outs = {}
    for sbs in (1, 3, None, 11):
        kw = _gather_kwargs(img, depth, K=31, thresh=0.5, bright_stamp_gain=2.0)
        kw["slab_batch_size"] = sbs
        outs[sbs], _ = _run_gpu(kw, _tile_n=64)
    for sbs in (1, 3, 11):
        _same5(outs[None], outs[sbs], f"slab_batch_size={sbs}")
        for i in range(5):
            assert np.array_equal(outs[None][i], outs[sbs][i]), (sbs, i)
    kw = _gather_kwargs(img, depth, K=31, thresh=0.5, bright_stamp_gain=2.0)
    kw["slab_batch_size"] = 3
    ref5, _ = _run_ref(kw)
    _assert_tier_c(ref5, outs[None], "slab_batch_size vs reference")


def _staircase(h, w, od):
    ramp = np.linspace(od[0], od[-1], w, dtype=np.float64)
    ramp = np.round(ramp / 1.25) * 1.25 + 0.3
    yy = np.linspace(0.0, 2.0, h)[:, None]
    return np.clip(ramp[None, :] + yy, od[0], od[-1]).astype(np.float32)


def _monocular(h, w, od, seed=17):
    rng = np.random.default_rng(seed)
    f = rng.normal(size=(h // 8 + 2, w // 8 + 2))
    f = np.kron(f, np.ones((8, 8)))[:h, :w]
    f = (f - f.min()) / (np.ptp(f) + 1e-9)
    f = np.clip(f + rng.normal(0, 0.02, (h, w)), 0.0, 1.0)
    return np.exp(np.log(od[0]) + f * (np.log(od[-1]) - np.log(od[0]))).astype(np.float32)


@pytest.mark.parametrize("name,fn", [("linear-staircase", _staircase),
                                     ("monocular", _monocular)])
@pytest.mark.parametrize("hw", [(260, 220), (220, 260)])
def test_depth_field_families(name, fn, hw):
    ref5, got5, lights, _ = _chart_case(*hw, 67, "hdr16_l1g2", tile_n=96, n_lights=16,
                                        depth_fn=fn)
    judge_c(ref5, got5, lights, 67, f"{name} {hw}")


@pytest.mark.parametrize("hw", [(360, 320), (320, 360)])
def test_lens50_small_frame_hdr16(hw):
    t = dict(_tensor("50"))
    ref5, got5, lights, _ = _chart_case(*hw, 67, "hdr16_l1g2", tile_n=128, n_lights=16,
                                        lens="50", tensor=t)
    ris, azs = chart_coverage(lights)
    assert ris == set(range(12)) and len(azs) >= 16
    assert float(ref5[1].max()) > 0.01
    judge_c(ref5, got5, lights, 67, f"lens50 {hw} hdr16_l1g2")


@pytest.mark.slow
def test_lens50_k183_native_against_the_reference_engine():
    """Lens 50 at native-like K=183 (N=512), judged against the REFERENCE
    gather (not tiled), as the spec requires."""
    t = dict(_tensor("50"))
    t.pop("corpus_ref_long_edge", None)
    ref5, got5, lights, _ = _chart_case(900, 820, 183, "hdr16_l1g2", n_lights=20,
                                        lens="50", tensor=t)
    judge_c(ref5, got5, lights, 183, "lens50 K=183 N=512")


def test_custom_tensor_k_over_255_is_not_available():
    t = make_synthetic_tensor(12, 5, 257, 257)
    h, w = 40, 40
    img = np.full((h, w, 3), 0.3, np.float32)
    depth = np.full((h, w), float(t["object_distances_m"][2]), np.float32)
    kw = _gather_kwargs(img, depth, K=257, tensor=t)
    assert kw["scaled_ksize"] == 257
    with pytest.raises(ValueError, match="not available"):
        _run_gpu(kw)


def test_pad_mismatch_and_single_radial_bin_are_not_available():
    """The non-square-kernel refusal belongs to render() (WP3, locked-spec 3.2
    step 4); the engine itself refuses a pad other than K // 2 and Nr < 2."""
    img = np.full((40, 40, 3), 0.3, np.float32)
    od = _tensor("35")["object_distances_m"]
    kw = _gather_kwargs(img, np.full((40, 40), od[4], np.float32), K=17)
    with pytest.raises(ValueError, match="not available"):
        _run_gpu(dict(kw, pad=5), _tile_n=40)
    with pytest.raises(ValueError, match="not available"):
        _run_gpu(dict(kw, r_centers=np.array([0.5], np.float32)), _tile_n=40)
