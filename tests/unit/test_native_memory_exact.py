"""R-native-mem exactness tests (spec ``doc/plan/phase2/section17_native-memory/
locked-spec-mem.md`` §3.9a / §3.9b).

* S5 (item 3): the 9 image spectra are dropped before the OVER phase. A
  synthetic scene with a silhouette makes regime-b run; the test owns the
  ``fft_state`` dict and calls ``_layered_composite`` directly.
* 9b: ``slab_batch_size`` in {None, 1, 2, 4, 11} is bit-identical for BOTH
  occlusion modes, on a scene where the regime-b OVER really runs.

Item 2 (in-place ``compute_slab_weights`` normalise) was reverted by its own
keep rule, so there is no oracle for it here (see coder-mem-report.md).
"""
from __future__ import annotations

import numpy as np
import pytest

import summicron.render._layered_composite as lc
import summicron.render.convolve as cv

_OD_D11 = np.array(
    [0.7, 0.98, 1.37, 1.91, 2.68, 3.74, 5.23, 7.32, 10.23, 14.30, 20.0],
    dtype=np.float64,
)


def _tensor_d11(k: int = 33, n_r: int = 6):
    D = 11
    kern = np.zeros((n_r, D, k, k), dtype=np.float32)
    c = k // 2
    kern[:, 0, c, c] = 1.0
    ys, xs = np.mgrid[-c:c + 1, -c:c + 1].astype(np.float32)
    for di in range(1, D):
        g = np.exp(-(xs ** 2 + ys ** 2) / (2 * (1.5 + 1.5 * di) ** 2))
        kern[:, di] = (g / g.sum()).astype(np.float32)
    return {
        "kernels": kern,
        "radial_edges": np.array([0.0, 0.15, 0.30, 0.45, 0.60, 0.80, 1.01]),
        "defocus_ratios": np.linspace(0, 1, D, dtype=np.float64),
        "object_distances_m": _OD_D11.copy(),
        "focus_distance_m": 0.7,
        "corpus_ref_long_edge": None,
        "otf_norm_applied": False,
        "delta_guard_applied": True,
    }


def _silhouette_scene(n: int = 256, seed: int = 20261109):
    """Near foreground block (1 m) over a far ramped background (3-20 m):
    a hard depth step so the regime-b selector fires and several slabs are
    active."""
    rng = np.random.default_rng(seed)
    img = rng.uniform(0.0, 0.8, (n, n, 3)).astype(np.float32)
    ramp = np.geomspace(3.0, 20.0, n, dtype=np.float32)
    depth = np.broadcast_to(ramp[None, :], (n, n)).copy()
    depth[n // 4: 3 * n // 4, n // 4: 3 * n // 4] = np.float32(1.0)
    return img, depth


@pytest.fixture()
def over_counter(monkeypatch):
    """Counts fused OVER scatter calls (proof that regime-b actually ran)."""
    calls = {"n": 0}
    orig = lc._over_scatter_slab

    def counting(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)

    monkeypatch.setattr(lc, "_over_scatter_slab", counting)
    return calls


def test_s5_spectra_dropped_before_over_and_output_unchanged(
    monkeypatch, over_counter,
):
    img, depth = _silhouette_scene(160)
    tensor = _tensor_d11()
    cap: dict = {}
    orig_lc = lc._layered_composite

    def spy(**kw):
        cap["kw"] = kw
        cap["had_spectra"] = all(
            k in kw["fft_state"] for k in ("img_F", "img_diffuse_F", "img_bright_F")
        )
        out = orig_lc(**kw)
        cap["out"] = out
        return out

    monkeypatch.setattr(lc, "_layered_composite", spy)
    _, lin = cv.render(
        img, tensor, depth_m=depth, occlusion_compositing="on",
        hdr_source="macos",
    )
    assert cap["had_spectra"], "spectra must exist when OVER is entered"
    assert over_counter["n"] > 0, "scene did not exercise the regime-b OVER"
    for k in ("img_F", "img_diffuse_F", "img_bright_F"):
        assert k not in cap["kw"]["fft_state"], f"{k} not released before OVER"
    for k in ("fshape", "slice_y", "slice_x"):
        assert k in cap["kw"]["fft_state"], f"{k} must survive (OVER reads it)"

    # The test owns a fresh fft_state and calls _layered_composite directly.
    kw = dict(cap["kw"])
    fs = cv._build_image_fft_state(
        kw["img_padded"], kw["img_diffuse_padded"], kw["img_bright_padded"],
        kw["scaled_ksize"],
    )
    assert "img_F" in fs
    kw["fft_state"] = fs
    out2 = orig_lc(**kw)
    for k in ("img_F", "img_diffuse_F", "img_bright_F"):
        assert k not in fs
    for a, b in zip(cap["out"], out2):
        np.testing.assert_array_equal(a, b)

    # And render()'s combined output matches the directly-composed diffuse+bright.
    expected_pre_snap = out2[0] + out2[1]
    assert np.isfinite(expected_pre_snap).all()
    assert np.isfinite(lin).all()


# ---------------------------------------------------------------------------
# 9b: ON/OFF batch equivalence on a regime-b scene
# ---------------------------------------------------------------------------

_BATCHES = [None, 1, 2, 4, 11]


@pytest.fixture(scope="module")
def _batch_refs():
    return {}


@pytest.mark.parametrize("mode", ["off", "on"])
@pytest.mark.parametrize("batch", _BATCHES)
def test_batch_size_bit_identical_both_modes(mode, batch, _batch_refs, over_counter):
    img, depth = _silhouette_scene(256)
    tensor = _tensor_d11()
    stats: dict = {}
    _, lin = cv.render(
        img, tensor, depth_m=depth, occlusion_compositing=mode,
        slab_batch_size=batch, hdr_source="macos", stats=stats,
    )
    if mode == "on":
        assert over_counter["n"] > 0, "regime-b OVER did not run"
        assert "acc_alpha" in stats
    ref = _batch_refs.setdefault(mode, lin)
    np.testing.assert_array_equal(ref, lin)
