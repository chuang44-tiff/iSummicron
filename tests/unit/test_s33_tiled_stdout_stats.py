"""S33 Tier B WP2, gate G-B4: stdout and stats through the engine directly.

``render()`` is run twice on the same synthetic scene: once with its own
reference gather (spied) and once with ``convolve._layered_gather_diffuse``
monkeypatched to the tiled gather (``_tile_n`` forced so the small frame holds
several tiles) and ``_build_image_fft_state`` swapped for a stub (the tiled
engine builds no image FFT state). Nothing here needs ``runs/`` or IMG_6915.
"""
from __future__ import annotations

import warnings
from functools import lru_cache

import numpy as np
import pytest

import summicron.render.convolve as cv
from summicron.render import _engine_tiled as eng
from summicron.render._artifacts import default_tensor_path
from tests.fixtures import tolerance as tol


@lru_cache(maxsize=None)
def _tensor(lens: str = "35") -> dict:
    return cv.load_tensor(default_tensor_path(lens))


def _scene(h=150, w=170, seed=3):
    rng = np.random.default_rng(seed)
    img = rng.uniform(0.0, 0.7, (h, w, 3)).astype(np.float32)
    for _ in range(6):
        y, x = int(rng.integers(3, h - 4)), int(rng.integers(3, w - 4))
        img[y:y + 3, x:x + 3] = 1.0
    od = _tensor()["object_distances_m"]
    ramp = np.geomspace(od[0], od[-1], w, dtype=np.float32)
    depth = np.broadcast_to(ramp[None, :], (h, w)).copy()
    depth[h // 4: h // 2, w // 5: w // 2] = np.float32(od[3])
    return img, depth


def _render(engine, img, depth, *, tile_n=96, cap=None, **kw):
    """Run render() with the gather chosen by ``engine``. Returns a dict with
    the linear output, stats, progress lines, warnings and the gather 5-tuple."""
    # corpus_ref_long_edge 464 on a 170-px long edge gives pix_scale 0.366, so
    # K = 67 (real field-dependent kernels, not the near-delta K of a tiny frame)
    tensor = dict(_tensor(), corpus_ref_long_edge=464)
    lines: list[str] = []
    stats: dict = {}
    got: dict = {}
    real_gather = cv._layered_gather_diffuse
    real_state = cv._build_image_fft_state
    state_calls = {"n": 0}

    def gather(**k):
        if cap is not None:
            k["active_slab_cap"] = cap
        if engine == "tiled":
            k["fft_state"] = None
            out = eng.tiled_layered_gather(**k, _tile_n=tile_n)
        else:
            out = real_gather(**k)
        got["five"] = out
        return out

    def state(*a, **k):
        state_calls["n"] += 1
        return None if engine == "tiled" else real_state(*a, **k)

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
    return _render("reference", img, depth, **kw), _render("tiled", img, depth, **kw)


def test_five_tuple_accumulators_array_equal():
    ref, got = _pair()
    for i, name in ((2, "weight_sum"), (3, "delta_weight_sum"), (4, "acc_alpha")):
        assert np.array_equal(ref["five"][i], got["five"][i]), name
    assert np.array_equal(ref["stats"]["acc_alpha"], got["stats"]["acc_alpha"])
    # diffuse and bright images are inside the Tier B ceilings
    res = tol.evaluate(ref["five"][0] + ref["five"][1], got["five"][0] + got["five"][1], "B")
    assert res["ok"], tol.summary(res)
    res = tol.evaluate(ref["lin"], got["lin"], "B")
    assert res["ok"], tol.summary(res)
    # the swap really took effect
    assert got["state_calls"] == 1


def test_progress_sequence_and_stdout_lines_identical():
    ref, got = _pair()
    assert ref["lines"] == got["lines"]
    joined = "\n".join(got["lines"])
    assert "layered_gather: active_slabs=" in joined
    assert "bright-pass: thresh=" in joined
    slab_lines = [x for x in got["lines"] if x.startswith("  Slab ")]
    assert slab_lines == [f"  Slab {i}/11" for i in range(1, 12)]


def test_bright_energy_ratio_digits_unchanged():
    """A changed digit in bright_energy_ratio is a STOP (R-B3), not a fix."""
    ref, got = _pair()
    pick = lambda r: [x for x in r["lines"] if "bright_energy_ratio=" in x]   # noqa: E731
    assert len(pick(ref)) == 1
    assert pick(ref) == pick(got)
    ratio = float(pick(ref)[0].rsplit("bright_energy_ratio=", 1)[1])
    assert ratio > 0.0, "vacuous: the scene must put energy in the bright image"


def test_stats_dict_keys_and_scalars_equal():
    ref, got = _pair()
    assert set(ref["stats"]) == set(got["stats"])
    for k, v in ref["stats"].items():
        g = got["stats"][k]
        if isinstance(v, np.ndarray):
            if k == "acc_alpha":
                assert np.array_equal(v, g), k
            else:
                assert np.allclose(v, g, rtol=0, atol=1e-5), k
        elif isinstance(v, float):
            assert g == pytest.approx(v, abs=1e-6), k
        else:
            assert g == v, k


def test_max_active_slabs_warning_text_identical_through_render():
    ref, got = _pair(cap=2)
    assert len(ref["warns"]) == 1
    assert ref["warns"] == got["warns"]
    assert ref["lines"] == got["lines"]


def test_progress_sequence_with_empty_slabs():
    """Slabs the reference skips (full-frame max < _EMPTY_SKIP_THRESH) still
    print their ``Slab i/D`` line and are not counted active."""
    img, _ = _scene()
    od = _tensor()["object_distances_m"]
    depth = np.full(img.shape[:2], od[4], np.float32)
    ref = _render("reference", img, depth)
    got = _render("tiled", img, depth)
    assert ref["lines"] == got["lines"]
    active = [x for x in got["lines"] if "layered_gather:" in x][0]
    assert "active_slabs=" in active and "/11" in active
    assert int(active.split("active_slabs=")[1].split("/")[0]) < 11
    for i in (3, 4):
        assert np.array_equal(ref["five"][i], got["five"][i])
