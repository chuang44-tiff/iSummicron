"""Adversarial probes for section15 Track A (matte correction + staircase smoothing)."""
from __future__ import annotations

import warnings

import numpy as np
import pytest
from PIL import Image

from summicron.depth.estimate import apply_tap_focus_shift
from summicron.depth.guided import smooth_lidar_staircase
from summicron.depth.matte import (
    correct_depth_with_matte,
    load_portrait_matte,
)

URN = "urn:com:apple:photo:2018:aux:portraiteffectsmatte"
H, W = 96, 128
RK = {"r_px_2048": 64}  # r = 4 on 128-wide frames


class _Aux:
    def __init__(self, im):
        self._im = im

    def to_pillow(self):
        return self._im


class _Heif:
    def __init__(self, im):
        self.info = {"aux": {URN: [1]}}
        self._im = im

    def get_aux_image(self, _i):
        return _Aux(self._im)


def _patch(monkeypatch, alpha_img):
    import pillow_heif

    monkeypatch.setattr(pillow_heif, "open_heif", lambda *a, **k: _Heif(alpha_img))


def _l(alpha):
    return Image.fromarray(np.round(alpha * 255).astype(np.uint8), "L")


def _step(h=H, w=W, near=1.0, far=4.0):
    x = np.arange(w)[None, :].repeat(h, 0)
    depth = np.where(x < w // 2, near, far).astype(np.float32)
    alpha = (np.round(np.clip((w // 2 + 8 - x) / 16.0, 0, 1) * 255) / 255).astype(np.float32)
    return depth, alpha


def _caught(fn, *a, **k):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        out = fn(*a, **k)
    return out, [str(r.message) for r in rec]


# --- wrong polarity ---------------------------------------------------------

def test_inverted_moves_soft_pixels_the_wrong_way():
    d, alpha = _step()
    norm, rn = correct_depth_with_matte(d, alpha, **RK)
    inv, ri = correct_depth_with_matte(d, alpha, _polarity="inverted", **RK)
    assert np.array_equal(ri["changed"], rn["changed"])
    soft = rn["band"] & (alpha > 0.02) & (alpha < 0.98)
    hi_a, lo_a = soft & (alpha >= 0.5), soft & (alpha < 0.5)
    assert np.mean(inv[hi_a] > norm[hi_a]) >= 0.99  # alpha >= .5: farther
    assert np.mean(inv[lo_a] < norm[lo_a]) >= 0.99  # alpha < .5: nearer


def test_one_minus_alpha_recipe_is_degenerate():
    d, alpha = _step()
    alpha = np.where(alpha == 0.5, 0.49, alpha).astype(np.float32)
    for curve in ("linear", "hard"):
        a, _ = correct_depth_with_matte(d, alpha, curve=curve, **RK)
        b, _ = correct_depth_with_matte(d, 1.0 - alpha, curve=curve, **RK)
        if curve == "linear":
            assert np.allclose(a, b, rtol=1e-5)
        else:
            assert np.array_equal(a, b)


# --- registration -----------------------------------------------------------

def test_misregistered_by_8px_is_bounded():
    d, alpha = _step()
    rolled = np.roll(alpha, 8, axis=1)
    out, rep = correct_depth_with_matte(d, rolled, **RK)
    assert rep["changed_px"] / d.size < 0.30  # bounded by the band, not the frame
    assert np.array_equal(out[~rep["band"]], d[~rep["band"]])


def test_rotated_matte_aspect_guard(monkeypatch):
    _, alpha = _step()
    _patch(monkeypatch, _l(np.rot90(alpha)))  # 3:4 against a 4:3 target
    out, msgs = _caught(load_portrait_matte, "x", (H, W))
    assert out is None and len(msgs) == 1 and msgs[0].startswith("depth-matte:")


def _blob_scene(cx):
    yy, xx = np.mgrid[:H, :W]
    alpha = np.clip(1.0 - (np.hypot(yy - 48, xx - cx) - 30) / 6.0, 0, 1).astype(np.float32)
    depth = np.where(alpha > 0.5, 1.0, 8.0).astype(np.float32)
    return depth, alpha


def test_mirrored_matte_offcentre_fails_agreement(monkeypatch):
    depth, alpha = _blob_scene(cx=54)
    _patch(monkeypatch, _l(alpha[:, ::-1]))
    out, msgs = _caught(load_portrait_matte, "x", (H, W), depth_m=depth)
    assert out is None and len(msgs) == 1 and msgs[0].startswith("depth-matte:")


def test_mirrored_matte_centred_passes_but_damage_bounded(monkeypatch):
    depth, alpha = _blob_scene(cx=64)
    _patch(monkeypatch, _l(alpha[:, ::-1]))
    got, msgs = _caught(load_portrait_matte, "x", (H, W), depth_m=depth)
    assert got is not None and msgs == []  # documents the limit of the agreement check
    out, rep = correct_depth_with_matte(depth, got, **RK)
    assert not rep["changed"][~rep["band"]].any()
    assert np.array_equal(out[~rep["band"]], depth[~rep["band"]])


# --- scene hazards ----------------------------------------------------------

def test_sky_background_stays_finite_and_monotone():
    d, alpha = _step(far=1000.0)
    out, rep = correct_depth_with_matte(d, alpha, **RK)
    assert np.all(np.isfinite(out)) and out.min() >= 0.1 and out.max() <= 1e3
    row = out[5]
    soft = np.where((alpha[5] > 0.02) & (alpha[5] < 0.98))[0]
    assert np.all(np.diff(row[soft[0] : soft[-1] + 1]) >= -1e-6)


def test_8bit_plateaus_monotone_disparity():
    d, alpha = _step()
    out, rep = correct_depth_with_matte(d, alpha, **RK)
    soft = np.where(rep["band"][5] & (alpha[5] > 0.02) & (alpha[5] < 0.98))[0]
    disp = 1.0 / out[5, soft]
    assert np.all(np.diff(disp) <= 1e-6)  # disparity falls as alpha falls


def test_second_person_and_glass_untouched():
    d, alpha = _step()
    d = d.copy()
    d[:, 100:120] = 1.0  # alpha-0 near blob far from the band (second person)
    d[10:20, 5:15] = 4.0  # inside the subject, matte says subject, LiDAR far (glass-like) is
    #                       far from the soft zone: left untouched
    out, rep = correct_depth_with_matte(d, alpha, **RK)
    assert np.array_equal(out[:, 100:120], d[:, 100:120])
    assert np.array_equal(out[10:20, 5:15], d[10:20, 5:15])
    glass = d.copy()
    glass[30:40, 5:15] = 1.0
    o2, _ = correct_depth_with_matte(glass, alpha, **RK)
    assert np.array_equal(o2[30:40, 5:15], glass[30:40, 5:15])


@pytest.mark.parametrize("hw", [(1, 1), (2, 2), (5, 7), (8, 8)])
def test_tiny_and_odd_frames_do_not_raise(hw):
    d = np.full(hw, 2.0, np.float32)
    alpha = np.random.default_rng(0).uniform(0, 1, hw).astype(np.float32)
    (out, rep), _ = _caught(correct_depth_with_matte, d, alpha)
    assert out.shape == hw and np.all(np.isfinite(out))
    sm = smooth_lidar_staircase(d, np.zeros(hw, np.float32))
    assert sm.shape == hw and np.all(np.isfinite(sm))


def test_tap_on_fringe_gives_finite_anchor():
    d, alpha = _step()
    out, _ = correct_depth_with_matte(d, alpha, **RK)
    res = apply_tap_focus_shift(out * 10.0, (64 / W, 48 / H))
    assert np.all(np.isfinite(np.asarray(res if not isinstance(res, tuple) else res[0])))


def test_determinism():
    d, alpha = _step()
    a, ra = correct_depth_with_matte(d, alpha, **RK)
    b, rb = correct_depth_with_matte(d, alpha, **RK)
    assert a.tobytes() == b.tobytes()
    for k, v in ra.items():
        if isinstance(v, np.ndarray):
            assert np.array_equal(v, rb[k])
        else:
            assert v == rb[k]


# --- guided step ------------------------------------------------------------

def test_flat_guide_leaves_staircase_near_identity():
    # Many tiny steps (a LiDAR-like staircase on a gentle ramp), flat guide: nothing to
    # snap to, so the filter only box-averages sub-0.1% steps.
    x = np.arange(W)[None, :].repeat(H, 0)
    disp = 1.0 + 0.0005 * (x // 8)
    d = (1.0 / disp).astype(np.float32)
    out = smooth_lidar_staircase(d, np.full((H, W), 0.5, np.float32))
    assert np.abs(np.log(1.0 / out) - np.log(1.0 / d)).max() < 1e-3


def test_guide_edge_far_from_depth_step_creates_no_step():
    d = np.full((H, W), 2.0, np.float32)
    guide = np.where(np.arange(W)[None, :].repeat(H, 0) < 30, 0.1, 0.9).astype(np.float32)
    out = smooth_lidar_staircase(d, guide)
    assert np.abs(np.log(1.0 / out) - np.log(1.0 / d)).max() < 1e-3


def test_sky_scene_no_new_extreme():
    d, _ = _step(far=1000.0)
    guide = np.random.default_rng(0).uniform(0, 1, (H, W)).astype(np.float32)
    out = smooth_lidar_staircase(d, guide)
    assert out.min() >= 0.1 and out.max() <= 1e3
    disp_in, disp_out = 1.0 / d, 1.0 / out
    assert disp_out.min() >= disp_in.min() - 1e-6 and disp_out.max() <= disp_in.max() + 1e-6


def test_guided_determinism():
    d, _ = _step()
    g = np.random.default_rng(1).uniform(0, 1, (H, W)).astype(np.float32)
    assert smooth_lidar_staircase(d, g).tobytes() == smooth_lidar_staircase(d, g).tobytes()
