"""Unit tests for summicron.psf_model.ray_kernel (Section 24)."""
from __future__ import annotations

import numpy as np
import pytest

from summicron.psf_model import ray_kernel as rk
from summicron.render.convolve import load_tensor

SIZE = 183
C = SIZE // 2


def test_l1_and_determinism():
    rng = np.random.default_rng(1)
    a, b = rng.normal(0, 0.02, 5000), rng.normal(0, 0.02, 5000)
    k1 = rk.rays_to_kernel(a, b, 0.002)
    k2 = rk.rays_to_kernel(a, b, 0.002)
    assert k1.shape == (SIZE, SIZE) and k1.dtype == np.float64
    assert abs(k1.sum() - 1.0) < 1e-12
    assert k1.min() >= 0
    assert np.array_equal(k1, k2)


def test_single_ray_subpixel_splat():
    pitch = 0.01
    # +0.25 px outward, +0.5 px tangential
    k = rk.rays_to_kernel([0.25 * pitch], [0.5 * pitch], pitch)
    row = C + rk.OUTWARD_ROW_SIGN * 0.25
    col = C + 0.5
    r0, c0 = int(np.floor(row)), int(np.floor(col))
    fr, fc = row - r0, col - c0
    assert k[r0, c0] == pytest.approx((1 - fr) * (1 - fc))
    assert k[r0 + 1, c0] == pytest.approx(fr * (1 - fc))
    assert k[r0, c0 + 1] == pytest.approx((1 - fr) * fc)
    assert k[r0 + 1, c0 + 1] == pytest.approx(fr * fc)
    assert np.count_nonzero(k) <= 4


def test_outside_fraction():
    pitch = 0.01
    d_out = np.array([0.0] * 8 + [1000.0] * 2)  # 2 of 10 far outside the box
    k, out = rk.rays_to_kernel(d_out, np.zeros(10), pitch, return_outside=True)
    assert out == pytest.approx(0.2)
    assert abs(k.sum() - 1.0) < 1e-12
    _, out0 = rk.rays_to_kernel([0.0], [0.0], pitch, return_outside=True)
    assert out0 == 0.0
    with pytest.raises(ValueError):
        rk.rays_to_kernel([1000.0], [0.0], pitch)


def test_assemble_delta_and_load_roundtrip(tmp_path):
    rng = np.random.default_rng(2)
    R, D = 12, 11
    ker = rng.random((R, D, SIZE, SIZE)).astype(np.float32) ** 8
    t = rk.assemble_tensor(ker, {"fu": 0.4, "lens_citation": "test",
                                 "prescription_variant": "x", "transmission": np.ones((R, D))})
    d0 = t["kernels"][:, 0]
    assert np.all(d0[:, C, C] == 1.0) and np.count_nonzero(d0) == R
    assert bool(t["delta_guard_applied"]) is True
    p = tmp_path / "t.npz"
    rk.save_tensor(p, t)
    with np.load(p, allow_pickle=False) as z:
        assert str(z["schema_version"]) == "1.0"
        assert int(z["corpus_ref_long_edge"]) == 5712
        assert float(z["focus_distance_m"]) == 0.7
        assert not bool(z["otf_norm_applied"])
        assert str(z["lens_citation"]) == "test"
    lt = load_tensor(p)  # raises on schema failure; delta-guard warning would be an error
    assert (lt["R"], lt["D"]) == (R, D)
    sums = lt["kernels"].sum(axis=(2, 3), dtype=np.float64)
    assert np.allclose(sums, 1.0, atol=1e-6)
    assert lt["object_distances_m"][0] == pytest.approx(0.7)
    assert lt["delta_guard_applied"] is True


def test_assemble_rejects_bad_input():
    with pytest.raises(ValueError):
        rk.assemble_tensor(np.zeros((12, 11, 4, 4)), {})
    bad = np.zeros((12, 11, SIZE, SIZE))
    bad[0, 1, 0, 0] = -1
    with pytest.raises(ValueError):
        rk.assemble_tensor(bad, {})


def _gauss(sr, st, shift=0.0):
    y, x = np.indices((SIZE, SIZE), dtype=np.float64)
    k = np.exp(-0.5 * (((y - C - shift) / sr) ** 2 + ((x - C) / st) ** 2))
    return k / k.sum()


def test_shape_metrics_ellipses():
    # The circular 90%-radius truncation compresses the aspect of a 2:1
    # Gaussian (measured about 1.64), so check bounds and reciprocity.
    a, _ = rk.shape_metrics(_gauss(6.0, 3.0))
    b, _ = rk.shape_metrics(_gauss(3.0, 6.0))
    assert 1.4 < a < 2.0
    assert a * b == pytest.approx(1.0, rel=1e-3)
    a, t = rk.shape_metrics(_gauss(5.0, 5.0))
    assert a == pytest.approx(1.0, rel=0.02) and abs(t) < 1e-6


def test_shape_metrics_tail_sign_is_outward():
    # core plus a tail toward OUTWARD (row = C + SIGN * d)
    k = _gauss(3.0, 3.0)
    tail = np.zeros_like(k)
    tail[C + rk.OUTWARD_ROW_SIGN * 20, C] = 1.0
    out = 0.8 * k + 0.2 * tail
    _, t_out = rk.shape_metrics(out)
    inn = np.zeros_like(k)
    inn[C - rk.OUTWARD_ROW_SIGN * 20, C] = 1.0
    _, t_in = rk.shape_metrics(0.8 * k + 0.2 * inn)
    assert t_out > 0.5 > -0.5 > t_in


def test_size_341_roundtrip(tmp_path):
    n = 341
    c = n // 2
    rng = np.random.default_rng(3)
    ker = np.zeros((12, 11, n, n))
    for r in range(12):
        for d in range(1, 11):
            ker[r, d] = rk.rays_to_kernel(rng.normal(0, 20, 300), rng.normal(0, 20, 300), 1.0, size=n)
    t = rk.assemble_tensor(ker, {})
    assert t["kernels"].shape == (12, 11, n, n)
    assert np.all(t["kernels"][:, 0, c, c] == 1.0)
    p = tmp_path / "t341.npz"
    rk.save_tensor(p, t)
    lt = load_tensor(p)
    assert lt["kernels"].shape[2:] == (n, n)
    assert np.allclose(lt["kernels"].sum(axis=(2, 3), dtype=np.float64), 1.0, atol=1e-6)
    a, tl = rk.shape_metrics(lt["kernels"][5, 5])
    assert np.isfinite(a) and np.isfinite(tl)
    d = rk.make_delta(n)
    assert d[c, c] == 1.0 and d.sum() == 1.0
