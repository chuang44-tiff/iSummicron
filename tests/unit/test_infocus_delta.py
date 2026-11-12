"""Q-b R-infocus-delta (flag removed by Q-b-legacy-removal): the strict d0
delta detector is unconditional.

A STRICT delta (exactly one nonzero, finite and > 0, at (kH//2, kW//2))
maps to an exact centred ``_make_delta(scaled_ksize)`` when pix_scale < 0.95.
Everything else (off-centre, near-delta, NaN/negative, d>=1, blended
d_t>0, pix_scale >= 0.95) takes the ordinary cubic zoom.

Wrong-polarity discipline: every "does NOT fire" probe runs in a spy harness
with a positive control that DOES fire.
Spec: doc/plan/phase2/section14_infocus-delta/locked-spec-Q-b.md.
"""
from __future__ import annotations

import inspect

import numpy as np
import pytest

import summicron.psf_model.downscale_tensor as dt
import summicron.render.convolve as cv
from summicron.render._artifacts import default_tensor_path
from summicron.render._layered_composite import _layered_composite
from summicron.render.api import RenderOptions
from summicron.render.convolve import _build_basis_kernel, load_tensor, render

pytestmark = pytest.mark.filterwarnings("ignore")

NR = 12


@pytest.fixture(scope="module")
def tensor():
    return load_tensor(default_tensor_path())


@pytest.fixture
def spy(monkeypatch):
    calls = []
    orig = dt._make_delta

    def _wrapped(size):
        calls.append(size)
        return orig(size)

    monkeypatch.setattr(dt, "_make_delta", _wrapped)
    return calls


def _geom(kH, long_edge, ref=5712):
    ps = long_edge / ref
    sk = max(3, int(round(kH * ps)) | 1) if ps < 0.95 else kH
    return ps, sk


def _build(kern, ri, d, ps, sk, use_idx=True, norm="l1", dr=None):
    kH = kern.shape[2]
    if dr is None:
        dr = np.linspace(0.0, 1.0, kern.shape[1])
    return _build_basis_kernel(
        ri, d, use_idx, dr, kern, kH, kern.shape[3], norm, ps, sk,
    )


def _synth(slice0, D=2):
    kH, kW = slice0.shape
    kern = np.zeros((1, D, kH, kW), dtype=np.float32)
    kern[0, 0] = slice0
    c = kH // 2
    ys, xs = np.mgrid[:kH, :kW].astype(np.float32)
    g = np.exp(-((ys - c) ** 2 + (xs - kW // 2) ** 2) / 50.0)
    kern[0, 1] = g / g.sum()
    return kern


def _centre(kH=183, val=1.0):
    k = np.zeros((kH, kH), dtype=np.float32)
    k[kH // 2, kH // 2] = val
    return k


# --- fires on strict deltas ---------------------------------------------------

def test_positive_control_synthetic_delta_fires(spy):
    ps, sk = _geom(183, 2048)
    k = _build(_synth(_centre()), 0, 0, ps, sk)
    assert spy == [sk]
    assert np.argwhere(k != 0).tolist() == [[sk // 2, sk // 2]]


@pytest.mark.parametrize("use_idx", [True, False])
def test_d0_exact_centred_delta_at_2048(tensor, spy, use_idx):
    K, dr = tensor["kernels"], tensor["defocus_ratios"]
    ps, sk = _geom(183, 2048)
    for ri in range(NR):
        k = _build(K, ri, 0 if use_idx else float(dr[0]), ps, sk,
                   use_idx=use_idx, dr=dr)
        assert k.shape == (sk, sk)
        assert k.max() == 1.0 and k.sum() == 1.0
        assert np.argwhere(k == 1.0).tolist() == [[sk // 2, sk // 2]]
    assert spy == [sk] * NR


@pytest.mark.parametrize("size", [512, 1024, 4096])
def test_fires_on_all_shipped_d0_slices(tensor, spy, size):
    ps, sk = _geom(183, size)
    for ri in range(NR):
        k = _build(tensor["kernels"], ri, 0, ps, sk)
        assert np.argwhere(k != 0).tolist() == [[sk // 2, sk // 2]]
        assert k.sum() == 1.0
    assert spy == [sk] * NR


@pytest.mark.parametrize("use_idx", [True, False])
def test_peak_norm_delta_is_still_delta(tensor, spy, use_idx):
    K, dr = tensor["kernels"], tensor["defocus_ratios"]
    ps, sk = _geom(183, 2048)
    for ri in range(NR):
        k = _build(K, ri, 0 if use_idx else float(dr[0]), ps, sk,
                   use_idx=use_idx, norm="peak", dr=dr)
        assert k.max() == 1.0 and k.sum() == 1.0
    assert len(spy) == NR
    k = _build(_synth(_centre(val=0.25)), 0, 0, ps, sk, norm="peak")
    assert k.max() == 1.0 and k[sk // 2, sk // 2] == 1.0


def test_even_kernel_fires_at_floor_centre(spy):
    kH = 184
    ps, sk = _geom(kH, 2048)
    k = _build(_synth(_centre(kH)), 0, 0, ps, sk)
    assert spy == [sk]
    assert np.argwhere(k != 0).tolist() == [[sk // 2, sk // 2]]
    k2 = np.zeros((kH, kH), np.float32)
    k2[kH // 2 - 1, kH // 2 - 1] = 1.0
    _build(_synth(k2), 0, 0, ps, sk)
    assert len(spy) == 1


# --- does NOT fire ------------------------------------------------------------

def _offcentre():
    k = np.zeros((183, 183), np.float32)
    k[91, 92] = 1.0
    return k


def _tiny_neighbour():
    k = _centre()
    k[91, 92] = 1e-12
    return k


def _near_delta():
    k = _centre(val=0.999)
    k[90, 91] = 0.001
    return k


def _nan_elsewhere():
    k = _centre()
    k[0, 0] = np.nan
    return k


@pytest.mark.parametrize("name,slice0", [
    ("offcentre", _offcentre()),
    ("tiny_1e-12_neighbour", _tiny_neighbour()),
    ("near_delta_0.999", _near_delta()),
    ("nan_elsewhere", _nan_elsewhere()),
])
def test_non_delta_does_not_fire(spy, name, slice0):
    ps, sk = _geom(183, 2048)
    k = _build(_synth(slice0), 0, 0, ps, sk)
    assert spy == [], f"{name}: delta branch fired"
    assert k.shape == (sk, sk)
    _build(_synth(_centre()), 0, 0, ps, sk)  # positive control
    assert len(spy) == 1


@pytest.mark.parametrize("norm", ["l1", "peak"])
def test_negative_centre_does_not_fire(spy, norm):
    ps, sk = _geom(183, 2048)
    _build(_synth(_centre(val=-1.0)), 0, 0, ps, sk, norm=norm)
    assert spy == []


def test_nan_only_centre_does_not_fire(spy):
    ps, sk = _geom(183, 2048)
    k = _build(_synth(_centre(val=np.nan)), 0, 0, ps, sk)
    assert spy == []
    assert np.isnan(k).any()


@pytest.mark.parametrize("d_t", [1e-6, 1e-3, 0.5, 0.999])
def test_blended_d_t_gt0_does_not_fire(tensor, spy, d_t):
    K, dr = tensor["kernels"], tensor["defocus_ratios"]
    ps, sk = _geom(183, 2048)
    d = float(dr[0] + d_t * (dr[1] - dr[0]))
    for ri in range(NR):
        _build(K, ri, d, ps, sk, use_idx=False, dr=dr)
    assert spy == []
    for ri in range(NR):  # positive control: d_t == 0
        _build(K, ri, float(dr[0]), ps, sk, use_idx=False, dr=dr)
    assert len(spy) == NR


@pytest.mark.parametrize("size", [512, 2048, 2856, 4096])
def test_d_ge1_untouched(tensor, spy, size):
    K = tensor["kernels"]
    ps, sk = _geom(183, size)
    for ri in range(NR):
        for d in range(1, K.shape[1]):
            k = _build(K, ri, d, ps, sk)
            assert (k > 0).sum() > 1, (ri, d)
    assert spy == []


@pytest.mark.parametrize("size", [5712, 6000])
def test_never_fires_at_pix_scale_ge_095(tensor, spy, size):
    K = tensor["kernels"]
    ps, sk = _geom(183, size)
    assert ps >= 0.95
    for ri in range(NR):
        k = _build(K, ri, 0, ps, sk)
        assert np.array_equal(k, K[ri, 0])
    assert spy == []


# --- synthetic render smoke (both compositors) ---------------------------------

def _synth_tensor():
    R, D, k = 6, 11, 33
    kern = np.zeros((R, D, k, k), dtype=np.float32)
    c = k // 2
    kern[:, 0, c, c] = 1.0
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


@pytest.mark.parametrize("occ", ["off", "on"])
def test_synthetic_render_uses_delta_branch(spy, occ):
    n = 96
    rng = np.random.default_rng(7)
    img = (0.2 + 0.6 * rng.random((n, n, 3))).astype(np.float32)
    depth = np.full((n, n), 20.0, dtype=np.float32)
    depth[:, :48] = 0.9
    out, lin = render(img, _synth_tensor(), depth_m=depth,
                      occlusion_compositing=occ)
    res = np.asarray(lin if lin is not None else out, dtype=np.float64)
    assert np.isfinite(res).all()
    assert len(spy) > 0


# --- proof of removal (Q-b-legacy-removal) ------------------------------------

def test_flag_rejected_by_convolve_parser(capsys):
    with pytest.raises(SystemExit) as ei:
        cv.build_parser().parse_args(
            ["--image", "x.heic", "--kernel-centring", "delta"])
    assert ei.value.code == 2
    assert "--kernel-centring" in capsys.readouterr().err


def test_flag_rejected_by_isummicron_parser(capsys):
    from summicron.cli import build_parser

    with pytest.raises(SystemExit) as ei:
        build_parser().parse_args(
            ["render", "--image", "x.heic", "--kernel-centring", "legacy"])
    assert ei.value.code == 2
    assert "--kernel-centring" in capsys.readouterr().err


def test_render_kwarg_removed():
    img = np.zeros((8, 8, 3), np.float32)
    with pytest.raises(TypeError):
        render(img, {}, depth_m=np.ones((8, 8), np.float32),
               kernel_centring="delta")
    for fn in (render, _build_basis_kernel, cv._scatter_one_slab_fixed_d_basis,
               cv._layered_gather_diffuse, _layered_composite):
        assert "kernel_centring" not in inspect.signature(fn).parameters


def test_render_options_field_removed():
    import dataclasses

    assert "kernel_centring" not in {
        f.name for f in dataclasses.fields(RenderOptions)}
    with pytest.raises(TypeError):
        RenderOptions(image_path="x.jpg", kernel_centring="delta")
    assert not hasattr(RenderOptions(image_path="x.jpg"), "kernel_centring")


def test_legacy_pin_constants_removed():
    from summicron.render import _bench

    assert not hasattr(_bench, "G1_MD5_LEGACY")
    assert not hasattr(_bench, "ON_MD5_LEGACY")
    assert "--kernel-centring" not in _bench.CANONICAL_FLAGS
