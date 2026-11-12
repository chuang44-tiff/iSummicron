"""Adversarial: Q-b-legacy-removal (``--kernel-centring`` deleted).

Restores coverage lost with test_q_b_adversarial.py / test_kernel_centring_q_b.py
/ test_q_b_flip_adversarial.py that test_infocus_delta.py does not carry:

* energy + locality of the delta branch vs the old cubic-zoom path. The
  legacy path is no longer selectable, so it is emulated by a 1e-30
  off-centre neighbour on the d0 slice: the strict detector rejects it
  (nonzero count == 2) and the ordinary zoom runs, which is the old legacy
  arithmetic to within 1e-30.
* clean_runs PINS consistency with local pin files; README pin hygiene.
* failure modes of stale ``kernel_centring`` carried by namespaces/overrides.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy.ndimage import binary_dilation

import summicron.psf_model.downscale_tensor as dt
from summicron.render import _bench
from summicron.render.api import RenderOptions
from summicron.render.convolve import render

pytestmark = pytest.mark.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
NEAR = 0.9  # d0/d1 blend: the d0 kernel output is actually read (not snapped)


@pytest.fixture
def spy(monkeypatch):
    calls = []
    orig = dt._make_delta

    def _w(size):
        calls.append(size)
        return orig(size)

    monkeypatch.setattr(dt, "_make_delta", _w)
    return calls


def _synth_tensor(legacy_emulation=False):
    R, D, k = 6, 11, 33
    kern = np.zeros((R, D, k, k), dtype=np.float32)
    c = k // 2
    kern[:, 0, c, c] = 1.0
    if legacy_emulation:
        kern[:, 0, c, c + 1] = 1e-30  # defeats the strict detector only
    ys, xs = np.mgrid[-c:c + 1, -c:c + 1].astype(np.float32)
    for di in range(1, D):
        g = np.exp(-(xs ** 2 + ys ** 2) / (2 * (1.5 + 1.5 * di) ** 2))
        kern[:, di] = (g / g.sum()).astype(np.float32)
    return {
        "kernels": kern,
        "radial_edges": np.array([0, .15, .3, .45, .6, .8, 1.01]),
        "defocus_ratios": np.linspace(0, 1, 11),
        "object_distances_m": np.array(
            [0.7, 0.98, 1.37, 1.91, 2.68, 3.74, 5.23, 7.32, 10.23, 14.3, 20.0]),
        "focus_distance_m": 0.7,
        "corpus_ref_long_edge": 200,  # 96/200 -> pix_scale 0.48, sk=17
        "otf_norm_applied": False,
        "delta_guard_applied": True,
    }


def _scene(n=96):
    rng = np.random.default_rng(7)
    img = (0.2 + 0.6 * rng.random((n, n, 3))).astype(np.float32)
    depth = np.full((n, n), 20.0, dtype=np.float32)
    depth[:, :48] = NEAR
    return img, depth


def _render(legacy, occ, spy):
    img, depth = _scene()
    spy.clear()
    out, lin = render(img, _synth_tensor(legacy), depth_m=depth,
                      occlusion_compositing=occ)
    fired = len(spy)
    return np.asarray(lin if lin is not None else out, np.float64), fired


@pytest.mark.parametrize("occ", ["off", "on"])
def test_delta_energy_and_locality_vs_emulated_legacy(spy, occ):
    a, fired_a = _render(True, occ, spy)
    b, fired_b = _render(False, occ, spy)
    assert fired_a == 0, "legacy emulation must not take the delta branch"
    assert fired_b > 0, "default path must take the delta branch"
    assert np.isfinite(b).all()
    assert abs(b.sum() - a.sum()) / a.sum() < 2e-3
    d = np.abs(a - b).max(axis=-1)
    assert d.max() > 1e-3, "delta branch changes nothing in the d0 blend"
    if occ == "off":
        _, depth = _scene()
        changed = binary_dilation(depth < 20.0, iterations=17 + 8)
        assert (~changed).any()
        assert d[~changed].max() <= 1e-6, float(d[~changed].max())


def _md5(p):
    return hashlib.md5(p.read_bytes()).hexdigest()


def _clean_runs():
    spec = importlib.util.spec_from_file_location("cr", ROOT / "scripts/clean_runs.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_clean_runs_pins_consistent_with_local_files():
    cr = _clean_runs()
    assert cr.G1_MD5_PRE_QB == "38c75dd253571ece9c9a0e3223fdb580"
    assert cr.ON_MD5_PRE_QB == "07f5b91bc2a6ff46f1e93ddd96d6fc2e"
    assert {cr.G1_MD5_PRE_QB, cr.ON_MD5_PRE_QB}.isdisjoint({_bench.G1_MD5, _bench.ON_MD5})
    assert cr.PINS["vibe_bench/IMG_6915_occ_ON_pin.jpg"] == cr.ON_MD5_PRE_QB
    assert cr.PINS["vibe_bench/IMG_6915_G1_baseline_default.jpg"] == cr.G1_MD5_PRE_QB
    runs = ROOT / "runs"
    present = {r: e for r, e in cr.PINS.items() if (runs / r).is_file()}
    if not present:
        pytest.skip("no local pin files")
    bad = {r: _md5(runs / r) for r, e in present.items() if _md5(runs / r) != e}
    assert not bad, bad


def test_readme_pins_match_bench():
    p = ROOT / "runs/vibe_bench/README.md"
    if not p.exists():
        pytest.skip("README absent")
    txt = p.read_text()
    for h in (_bench.G1_MD5, _bench.ON_MD5, _bench.TENSOR_MD5):
        assert h in txt
    assert "--kernel-centring legacy" not in txt


def test_stale_namespace_attr_is_ignored_not_honoured():
    # A GUI/partial namespace still carrying kernel_centring must not crash
    # and cannot select the removed path (field loop ignores unknown attrs).
    ns = argparse.Namespace(image="x.jpg", kernel_centring="legacy")
    opts = RenderOptions.from_argparse(ns)
    assert not hasattr(opts, "kernel_centring")


def test_canonical_bench_override_rejected():
    with pytest.raises(TypeError):
        RenderOptions.canonical_bench("x.jpg", (0.4, 0.45), kernel_centring="delta")


def test_q_paired_probe_dump_rejects_flag():
    spec = importlib.util.spec_from_file_location("qpp", ROOT / "scripts/q_paired_probe.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    with pytest.raises(SystemExit) as ei:
        m.main(["dump", "--tag", "x", "--kernel-centring", "legacy"])
    assert ei.value.code == 2
