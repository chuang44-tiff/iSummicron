"""Unit tests for summicron.depth.diopter (Q-d S0, spec section 7 items 1-9)."""
from __future__ import annotations

import subprocess
import sys
import warnings

import numpy as np
import pytest

from summicron.depth.diopter import auto_gain, diopter_equivalent_depth

OD = np.array([0.70, 0.98, 1.37, 1.91, 2.68, 3.74, 5.23, 7.32, 10.23, 14.30, 20.00])
RATIOS = np.linspace(0.0, 1.0, len(OD))
ZF, ZFAR = 0.7, 20.0
CAP = 1 / ZF - 1 / ZFAR


def _map(depth, anchor=0.9626, gain=1.25, **kw):
    kw.setdefault("z_focus", ZF)
    kw.setdefault("z_far", ZFAR)
    return diopter_equivalent_depth(depth, anchor, gain=gain, **kw)


def test_anchor_maps_to_focus():
    d = np.full((4, 4), 0.9626)
    assert np.allclose(_map(d, 0.9626), ZF, atol=1e-6)
    d2 = np.array([[0.7, 0.7]])
    assert np.allclose(_map(d2, 0.7, gain=3.0), ZF, atol=1e-6)


def test_range_dtype_shape_no_mutation():
    rng = np.random.default_rng(0)
    d = rng.uniform(0.1, 1000.0, (16, 24))
    ref = d.copy()
    out = _map(d)
    assert out.dtype == np.float32 and out.shape == d.shape
    assert np.isfinite(out).all()
    assert out.min() >= ZF - 1e-6 and out.max() <= ZFAR + 1e-4
    assert np.array_equal(d, ref)
    ro = d.astype(np.float32)
    ro.setflags(write=False)
    assert _map(ro).dtype == np.float32


def test_nonfinite_passthrough_to_focus():
    d = np.array([[np.nan, np.inf, -np.inf, 5.0]])
    out = _map(d)
    assert np.all(out[0, :3] == np.float32(ZF))
    assert out[0, 3] > ZF


def test_monotone_in_dp_and_gain_and_limits():
    anchor = 1.0
    dp = np.linspace(0.0, 2.0, 50)
    z = 1.0 / (1.0 / anchor + dp)  # far side, positive
    out = _map(z, anchor, gain=1.0)
    assert np.all(np.diff(out) >= -1e-6)
    d = np.array([0.5, 1.5, 3.0, 8.0])
    prev = None
    for g in (0.1, 0.5, 1.0, 1.5, 2.5, 5.0):
        o = _map(d, 0.9626, gain=g)
        if prev is not None:
            assert np.all(o >= prev - 1e-6)
        prev = o
    assert np.allclose(_map(d, 0.9626, gain=1e-9), ZF, atol=1e-5)
    assert np.all(_map(d, 0.9626, gain=1e6) == np.float32(ZFAR))


def test_symmetry_near_far():
    a = 1.0
    dpv = 0.3
    z_far_side = 1.0 / (1 / a - dpv)
    z_near_side = 1.0 / (1 / a + dpv)
    out = _map(np.array([z_far_side, z_near_side]), a, gain=1.0)
    assert out[0] == pytest.approx(out[1], rel=1e-6)


def test_node_reproduction_anchor_eq_focus():
    dn = 1 / ZF - 1 / OD
    z = 1.0 / (1 / ZF - dn)
    out = _map(z, anchor=ZF, gain=1.0)
    assert np.allclose(out, OD, rtol=1e-5)


@pytest.mark.parametrize("anchor", [0.7, 0.9626, 0.618])
def test_node_reproduction_anchor_general(anchor):
    dn = 1 / ZF - 1 / OD
    # Far-side pixel at 1/z = 1/anchor - dn[d]; only nodes with 1/z > 0 exist
    # (anchor 0.9626 -> 1/a = 1.039 D < dn[10] = 1.379 D, so the far nodes
    # are unreachable on that side and are skipped; 4 of 11 remain).
    ok = (1 / anchor - dn) > 1e-3
    assert ok.sum() >= 4
    z = 1.0 / (1 / anchor - dn[ok])
    out = _map(z, anchor=anchor, gain=1.0)
    ratios = np.interp(out, OD, RATIOS)
    assert np.allclose(ratios, RATIOS[ok], atol=1e-4)


def test_cap_and_sentinel_coincide():
    d = np.array([20.0, 1000.0, 999.0])
    out = _map(d, anchor=0.7, gain=5.0)
    assert np.all(out == np.float32(ZFAR))
    # e never exceeds cap: implied e from output
    rng = np.random.default_rng(1)
    dd = rng.uniform(0.05, 1000.0, 2000)
    o = _map(dd, 0.9626, gain=10.0).astype(np.float64)
    e = 1 / ZF - 1 / o
    assert e.max() <= CAP + 1e-6


def test_synthetic_axis_not_hardcoded():
    zf, zfar = 0.5, 30.0
    od = np.geomspace(zf, zfar, 11)
    out = diopter_equivalent_depth(
        np.array([1000.0, 0.5]), 0.5, gain=10.0, z_focus=zf, z_far=zfar
    )
    assert out[0] == pytest.approx(zfar, rel=1e-5)
    assert out[1] == pytest.approx(zf, rel=1e-5)
    # auto_gain target uses od[-3]
    anchor = 0.5
    depth = np.full((10, 10), 1.0)  # dp = 1.0 everywhere
    expected = (1 / zf - 1 / od[-3]) / 1.0
    got = auto_gain(depth, anchor, z_focus=zf, od=od, lo=0.0, hi=100.0)
    assert got == pytest.approx(expected, rel=1e-9)
    assert abs(expected - (1 / ZF - 1 / OD[-3])) > 1e-3  # differs from shipped


def test_auto_gain_clip_and_flat():
    anchor = 1.0
    depth = np.full((8, 8), 1.0 / (1.0 + 0.5))  # dp = 0.5
    tgt = 1 / ZF - 1 / OD[-3]
    assert auto_gain(depth, anchor, z_focus=ZF, od=OD, lo=0.1, hi=100) == pytest.approx(
        tgt / 0.5
    )
    assert auto_gain(depth, anchor, z_focus=ZF, od=OD, lo=3.0, hi=4.0) == 3.0
    assert auto_gain(depth, anchor, z_focus=ZF, od=OD, lo=0.1, hi=1.0) == 1.0
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        flat = np.full((8, 8), 1.0)
        assert auto_gain(flat, 1.0, z_focus=ZF, od=OD) == 1.0
        assert auto_gain(np.full((4, 4), np.nan), 1.0, z_focus=ZF, od=OD) == 1.0


def test_convolve_does_not_import_diopter_at_module_level():
    code = (
        "import sys, summicron.render.convolve;"
        "print('summicron.depth.diopter' in sys.modules)"
    )
    r = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert r.stdout.strip() == "False"
