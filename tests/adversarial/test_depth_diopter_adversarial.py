"""Adversarial tests for summicron.depth.diopter (Q-d S0).

The SLOW meters-path-isolation / harness-wrapper polarity tests of spec
section 7 need scripts/probe_s16_diopter.py (Round 2) and are added then.
"""
from __future__ import annotations

import numpy as np
import pytest

from summicron.depth.diopter import diopter_equivalent_depth

OD = np.array([0.70, 0.98, 1.37, 1.91, 2.68, 3.74, 5.23, 7.32, 10.23, 14.30, 20.00])
RATIOS = np.linspace(0.0, 1.0, len(OD))
ZF, ZFAR = 0.7, 20.0


def _map(depth, anchor=0.9626, gain=1.25):
    return diopter_equivalent_depth(depth, anchor, gain=gain, z_focus=ZF, z_far=ZFAR)


def _signed_ref(depth, anchor, gain):
    """Wrong-polarity reference: signed (1/a - 1/z), clipped at 0 below."""
    dp = 1 / anchor - 1 / np.asarray(depth, float)
    cap = 1 / ZF - 1 / ZFAR
    e = np.clip(gain * dp, 0.0, cap)
    return 1 / (1 / ZF - e)


def _linear_metres_ref(depth, anchor, gain):
    """Wrong mapping: linear in metres."""
    return np.clip(ZF + gain * np.abs(np.asarray(depth, float) - anchor), ZF, ZFAR)


def test_polarity_signed_fails_symmetry_real_passes():
    a, dpv = 1.0, 0.3
    zs = np.array([1 / (1 / a - dpv), 1 / (1 / a + dpv)])
    real = _map(zs, a, 1.0)
    assert real[0] == pytest.approx(real[1], rel=1e-6)
    bad = _signed_ref(zs, a, 1.0)
    assert not bad[0] == pytest.approx(bad[1], rel=1e-3)


def test_linear_metres_fails_node_reproduction():
    a = 0.9626
    dn = 1 / ZF - 1 / OD
    ok = (1 / a - dn) > 1e-3
    z = 1 / (1 / a - dn[ok])
    ratios = np.interp(_map(z, a, 1.0), OD, RATIOS)
    assert np.allclose(ratios, RATIOS[ok], atol=1e-4)
    bad = np.interp(_linear_metres_ref(z, a, 1.0), OD, RATIOS)
    assert not np.allclose(bad, RATIOS[ok], atol=1e-2)


@pytest.mark.parametrize("anchor", [1000.0, 0.1])
def test_sentinel_or_clipped_anchor(anchor):
    d = np.array([[0.5, 1.0, 5.0, 1000.0]])
    out = _map(d, anchor, 1.25)
    assert np.isfinite(out).all()
    assert out.min() >= ZF - 1e-6 and out.max() <= ZFAR + 1e-4
    assert out.max() == np.float32(ZFAR) or anchor == 1000.0 or (out == ZFAR).any()


def test_clipped_anchor_reaches_cap():
    out = _map(np.array([1000.0]), 0.1, 1.25)
    assert out[0] == np.float32(ZFAR)


def test_zero_and_negative_depth():
    d = np.array([0.0, -1.0, -1e9, 1e-12])
    out = _map(d)
    assert np.isfinite(out).all()
    assert (out == np.float32(ZFAR)).all()  # 1/eps is huge -> capped
    assert np.isfinite(_map(d, anchor=0.0)).all()
    assert np.isfinite(_map(d, anchor=-3.0)).all()


def test_constant_frame_at_anchor_no_warnings(recwarn):
    out = _map(np.full((5, 7), 0.9626), 0.9626, 1.0)
    assert np.allclose(out, ZF, atol=1e-6)
    assert len(recwarn) == 0


@pytest.mark.parametrize("shape", [(1, 1), (2, 2), (3, 17), (17, 3)])
def test_small_and_nonsquare(shape):
    rng = np.random.default_rng(2)
    d = rng.uniform(0.3, 30.0, shape)
    out = _map(d)
    assert out.shape == shape and np.isfinite(out).all()
    d[0, 0] = 0.9626  # tap at corner
    assert _map(d, anchor=float(d[0, 0]))[0, 0] == pytest.approx(ZF, abs=1e-6)


def test_determinism():
    d = np.random.default_rng(3).uniform(0.3, 50.0, (32, 32))
    assert _map(d).tobytes() == _map(d).tobytes()


def test_plateau_stays_plateau_without_banding():
    d = np.full((10, 20), 2.0)
    d[:, 10:] = 6.0
    out = _map(d)
    assert np.unique(out[:, :10]).size == 1
    assert np.unique(out[:, 10:]).size == 1
    assert np.unique(out).size == 2


def test_r90_proxy_monotone_in_dp():
    r90 = np.array([0.7, 3.8, 5.1, 6.1, 6.7, 7.1, 7.5, 7.7, 7.9, 7.9, 7.9])
    anchor = 1.0
    dp = np.linspace(0.0, 3.0, 400)
    z = 1.0 / (1.0 / anchor + dp)
    zt = _map(z, anchor, 1.3)
    eff = np.interp(np.interp(zt, OD, RATIOS), RATIOS, r90)
    assert np.all(np.diff(eff) >= -1e-9)
