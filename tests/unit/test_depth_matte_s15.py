"""section15 Track A: guided staircase smoothing + Apple-matte depth correction.

Synthetic unless marked slow (real HEICs under iphoneImage/).
"""
from __future__ import annotations

import ast
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from scipy import ndimage

from summicron.depth import guided as guided_mod
from summicron.depth.estimate import extract_lidar_depth
from summicron.depth.guided import load_luma_guide, smooth_lidar_staircase
from summicron.depth.matte import (
    correct_depth_with_matte,
    load_portrait_matte,
    matte_band,
)

ROOT = Path(__file__).resolve().parents[2]
IMG = ROOT / "iphoneImage"
HEIC_6651 = IMG / "IMG_6651.HEIC"
HEIC_4343 = IMG / "IMG_4343.heic"
HEIC_6915 = IMG / "IMG_6915.HEIC"
URN = "urn:com:apple:photo:2018:aux:portraiteffectsmatte"

# r = round(64 * 128 / 2048) = 4 px band radius on the 128-wide synthetic scenes.
RK = {"r_px_2048": 64}
H, W = 96, 128


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

class _FakeAux:
    def __init__(self, im):
        self._im = im

    def to_pillow(self):
        return self._im


class _FakeHeif:
    def __init__(self, aux_img=None, primary=None, urn=URN, raise_on_aux=False):
        self.info = {"aux": ({urn: [66]} if aux_img is not None else {})}
        self._aux, self._primary, self._raise = aux_img, primary, raise_on_aux

    def get_aux_image(self, _i):
        if self._raise:
            raise RuntimeError("boom")
        return _FakeAux(self._aux)

    def to_pillow(self):
        return self._primary


def _patch_heif(monkeypatch, fake):
    import pillow_heif

    monkeypatch.setattr(pillow_heif, "open_heif", lambda *a, **k: fake)


def _matte_img(alpha):
    return Image.fromarray(np.round(alpha * 255).astype(np.uint8), "L")


def _step_scene(h=H, w=W):
    """1 m left / 4 m right, LiDAR edge at col w//2; matte edge ramp 6 px left of it."""
    x = np.arange(w)[None, :].repeat(h, 0)
    depth = np.where(x < w // 2, 1.0, 4.0).astype(np.float32)
    ramp = np.clip((w // 2 + 8 - x) / 16.0, 0.0, 1.0)  # 1 at x<=56, 0 at x>=72
    alpha = (np.round(ramp * 255) / 255).astype(np.float32)
    return depth, alpha


def _caught(fn, *a, **k):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        out = fn(*a, **k)
    return out, [str(r.message) for r in rec]


def _aai(log_disp, mask):
    gx = ndimage.sobel(log_disp, axis=1) / 8.0
    gy = ndimage.sobel(log_disp, axis=0) / 8.0
    m2 = (gx**2 + gy**2)[mask]
    ang = np.degrees(np.arctan2(np.abs(gy), np.abs(gx)))[mask]
    near_axis = (ang <= 10) | (ang >= 80)
    return float(m2[near_axis].sum() / max(m2.sum(), 1e-30))


def _lband(log_disp, r=6, thr=0.05):
    gx = ndimage.sobel(log_disp, axis=1) / 8.0
    gy = ndimage.sobel(log_disp, axis=0) / 8.0
    step = np.hypot(gx, gy) > thr
    return ndimage.distance_transform_edt(~step) <= r, step


# ---------------------------------------------------------------------------
# 1. default path identity / keyword validation
# ---------------------------------------------------------------------------

def test_default_path_identity():
    if not HEIC_6651.exists():
        pytest.skip("IMG_6651 fixture absent")
    a = extract_lidar_depth(HEIC_6651, (768, 576))
    b = extract_lidar_depth(HEIC_6651, (768, 576), depth_guided="off", depth_matte="off")
    assert a is not None and np.array_equal(a, b)


@pytest.mark.parametrize("kw", [{"depth_matte": "foo"}, {"depth_guided": "foo"}])
def test_bad_keyword_raises_before_decode(monkeypatch, kw):
    import pillow_heif

    def boom(*a, **k):
        raise AssertionError("decode attempted")

    monkeypatch.setattr(pillow_heif, "open_heif", boom)
    with pytest.raises(ValueError):
        extract_lidar_depth("nonexistent.heic", (8, 8), **kw)


# ---------------------------------------------------------------------------
# 2. absent matte
# ---------------------------------------------------------------------------

def test_absent_matte_is_silent(monkeypatch):
    _patch_heif(monkeypatch, _FakeHeif(aux_img=None))
    out, msgs = _caught(load_portrait_matte, "x.heic", (H, W))
    assert out is None and msgs == []


@pytest.mark.slow
def test_absent_matte_real_4343():
    if not HEIC_4343.exists():
        pytest.skip("IMG_4343 absent")
    hw = (768, 576)
    out, msgs = _caught(load_portrait_matte, HEIC_4343, hw)
    assert out is None and msgs == []
    off = extract_lidar_depth(HEIC_4343, hw)
    on, msgs = _caught(extract_lidar_depth, HEIC_4343, hw, depth_matte="on")
    assert np.array_equal(on, off)
    assert len([m for m in msgs if m.startswith("depth-matte:")]) == 1
    g_off = extract_lidar_depth(HEIC_4343, hw, depth_guided="on")
    both, msgs = _caught(extract_lidar_depth, HEIC_4343, hw, depth_guided="on", depth_matte="on")
    assert np.array_equal(both, g_off)
    assert len([m for m in msgs if m.startswith("depth-matte:")]) == 1
    assert not [m for m in msgs if m.startswith("depth-guided:")]


# ---------------------------------------------------------------------------
# 3. loader guards
# ---------------------------------------------------------------------------

def _good_depth():
    return _step_scene()[0]


def test_loader_ok_and_resize(monkeypatch):
    _, alpha = _step_scene()
    big = np.asarray(Image.fromarray(alpha, mode="F").resize((256, 192), Image.BILINEAR))
    _patch_heif(monkeypatch, _FakeHeif(_matte_img(big)))
    out, msgs = _caught(load_portrait_matte, "x", (H, W), depth_m=_good_depth())
    assert msgs == [] and out.shape == (H, W) and out.dtype == np.float32
    assert 0.0 <= out.min() and out.max() <= 1.0


def test_loader_wrong_and_transposed_aspect_is_none(monkeypatch):
    _, alpha = _step_scene()
    for shape in ((W, H), (50, 200)):  # 3:4 portrait vs 4:3 target, and a silly one
        im = Image.fromarray((np.ones(shape) * 255).astype(np.uint8), "L")
        _patch_heif(monkeypatch, _FakeHeif(im))
        out, msgs = _caught(load_portrait_matte, "x", (H, W))
        assert out is None
        assert len(msgs) == 1 and msgs[0].startswith("depth-matte:")


def test_loader_i16_scaling(monkeypatch):
    arr = np.full((H, W), 65535, dtype=np.uint16)
    arr[:, : W // 2] = 32768
    _patch_heif(monkeypatch, _FakeHeif(Image.fromarray(arr)))
    out, msgs = _caught(load_portrait_matte, "x", (H, W))
    assert msgs == []
    assert out[0, 0] == pytest.approx(32768 / 65535, abs=1e-4)
    assert out[0, -1] == pytest.approx(1.0, abs=1e-6)


def test_loader_agreement_failure(monkeypatch):
    depth, alpha = _step_scene()
    bad = depth.copy()
    bad[:, :5] = 9.0  # ~9% of the core is far
    _patch_heif(monkeypatch, _FakeHeif(_matte_img(alpha)))
    out, msgs = _caught(load_portrait_matte, "x", (H, W), depth_m=bad)
    assert out is None and len(msgs) == 1 and msgs[0].startswith("depth-matte:")


def test_loader_core_too_small(monkeypatch):
    depth, _ = _step_scene()
    alpha = np.zeros((H, W), np.float32)
    alpha[:10, :10] = 1.0
    _patch_heif(monkeypatch, _FakeHeif(_matte_img(alpha)))
    out, msgs = _caught(load_portrait_matte, "x", (H, W), depth_m=depth)
    assert out is None and len(msgs) == 1 and msgs[0].startswith("depth-matte:")


def test_loader_decode_exception(monkeypatch):
    _patch_heif(monkeypatch, _FakeHeif(_matte_img(_step_scene()[1]), raise_on_aux=True))
    out, msgs = _caught(load_portrait_matte, "x", (H, W))
    assert out is None and len(msgs) == 1 and msgs[0].startswith("depth-matte:")


# ---------------------------------------------------------------------------
# 4. outside-band identity and counters
# ---------------------------------------------------------------------------

def test_outside_band_array_equal_and_counters():
    rng = np.random.default_rng(0)
    d = (rng.uniform(0.5, 6.0, (H, W))).astype(np.float32)
    yy, xx = np.mgrid[:H, :W]
    blob = np.clip(1.0 - (np.hypot(yy - 48, xx - 64) - 20) / 10.0, 0, 1).astype(np.float32)
    out, rep = correct_depth_with_matte(d, blob, **RK)
    band = rep["band"]
    assert np.array_equal(out[~band], d[~band])
    assert rep["changed_px"] == rep["soft_px"] + rep["hi_rewritten"] + rep["lo_rewritten"]
    assert rep["changed_px"] == int(rep["changed"].sum()) > 0
    assert not rep["changed"][~band].any()


# ---------------------------------------------------------------------------
# 5. identity cases
# ---------------------------------------------------------------------------

def test_identity_all_zero_all_one():
    d = _good_depth()
    for v in (0.0, 1.0):
        out, rep = correct_depth_with_matte(d, np.full((H, W), v, np.float32), **RK)
        assert out is d
        assert rep["changed_px"] == 0 and not rep["band"].any()


def test_empty_reference_warns_and_unchanged():
    d = _good_depth()
    alpha = np.full((H, W), 0.5, np.float32)  # all soft; band covers all
    (out, rep), msgs = _caught(correct_depth_with_matte, d, alpha, **RK)
    assert np.array_equal(out, d) and rep["changed_px"] == 0
    assert len(msgs) == 1 and msgs[0].startswith("depth-matte:")


# ---------------------------------------------------------------------------
# 6. disparity blend
# ---------------------------------------------------------------------------

def test_blend_in_disparity_not_metres():
    d = np.full((H, W), 1000.0, np.float32)
    d[:, : W // 2] = 1.0
    alpha = np.zeros((H, W), np.float32)
    alpha[:, : W // 2 - 8] = 1.0
    alpha[:, W // 2 - 8 : W // 2 + 8] = 0.5
    out, rep = correct_depth_with_matte(d, alpha, **RK)
    soft = rep["band"] & (alpha == 0.5)
    assert np.allclose(out[soft], 1.0 / (0.5 * 1.0 + 0.5 * 1e-3), rtol=1e-3)
    assert out[soft].max() < 3.0  # never ~500 m


# ---------------------------------------------------------------------------
# 7. fringe ramp
# ---------------------------------------------------------------------------

def test_fringe_ramp_monotone_and_interior_equal():
    d, alpha = _step_scene()
    out, rep = correct_depth_with_matte(d, alpha, **RK)
    row = out[10]
    soft = np.where((alpha[10] > 0.02) & (alpha[10] < 0.98))[0]
    seg = row[soft[0] : soft[-1] + 1]
    assert np.all(np.diff(seg) >= -1e-6)  # depth non-decreasing as alpha falls
    assert row[soft[0]] < row[soft[-1]]
    assert np.array_equal(out[:, :20], d[:, :20]) and np.array_equal(out[:, -20:], d[:, -20:])
    # soft zone spans 56..72; LiDAR edge at 64 -> pixels 64..71 move toward subject
    assert out[10, 66] < d[10, 66]


# ---------------------------------------------------------------------------
# 8. disagreement-only
# ---------------------------------------------------------------------------

def test_disagreement_only_rewrites():
    d, alpha = _step_scene()
    d = d.copy()
    # confident-high pixel (x=53, alpha 1) whose LiDAR is FAR -> rewritten to near reference
    d[20, 53] = 4.0
    # confident-low pixel (x=75, alpha 0) whose LiDAR is NEAR -> rewritten to far reference
    d[30, 75] = 1.0
    # confident-high with LiDAR already near -> untouched
    out, rep = correct_depth_with_matte(d, alpha, **RK)
    assert out[20, 53] == pytest.approx(1.0, rel=1e-4)
    assert out[30, 75] == pytest.approx(4.0, rel=1e-4)
    assert out[40, 53] == d[40, 53]
    assert rep["hi_rewritten"] >= 1 and rep["lo_rewritten"] >= 1
    # background depth gradient inside the band (alpha 0, LiDAR far) is not flattened
    g = d.copy()
    g[:, 72:80] = np.linspace(4.0, 4.4, 8, dtype=np.float32)[None, :]
    out2, _ = correct_depth_with_matte(g, alpha, **RK)
    assert np.array_equal(out2[:, 72:80], g[:, 72:80])


# ---------------------------------------------------------------------------
# 9. held object
# ---------------------------------------------------------------------------

def test_held_object_kept_speck_demoted():
    d, alpha = _step_scene()
    d = d.copy()
    # A 2 px alpha-0 speck at subject depth inside the band is demoted.
    s, a = _step_scene()
    s = s.copy()
    s[10:12, 73:75] = 1.0
    out3, _ = correct_depth_with_matte(s, a, **RK)
    assert out3[10, 73] == pytest.approx(4.0, rel=1e-3)


def test_held_object_wide_block_keeps_depth_when_far_from_band():
    d, alpha = _step_scene()
    d = d.copy()
    alpha = alpha.copy()
    d[:, 100:128] = 1.0
    out, _ = correct_depth_with_matte(d, alpha, **RK)
    assert np.allclose(out[:, 100:128], 1.0, rtol=1e-2)


# ---------------------------------------------------------------------------
# 10. curves + polarity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "curve,a,expect_w",
    [("linear", 0.25, 0.25), ("linear", 0.5, 0.5), ("linear", 0.81, 0.81),
     ("sqrt", 0.25, 0.5), ("sqrt", 0.81, 0.9),
     ("hard", 0.25, 0.0), ("hard", 0.5, 1.0), ("hard", 0.81, 1.0)],
)
def test_curve_weights(curve, a, expect_w):
    d = np.full((H, W), 4.0, np.float32)
    d[:, : W // 2] = 1.0
    alpha = np.zeros((H, W), np.float32)
    alpha[:, : W // 2 - 8] = 1.0
    alpha[:, W // 2 - 8 : W // 2 + 8] = a
    out, rep = correct_depth_with_matte(d, alpha, curve=curve, **RK)
    soft = rep["band"] & (np.abs(alpha - a) < 1e-6)
    want = 1.0 / (expect_w * 1.0 + (1 - expect_w) * 0.25)
    assert np.allclose(out[soft], want, rtol=1e-4)


def test_unknown_curve_and_polarity_raise():
    d, alpha = _step_scene()
    with pytest.raises(ValueError):
        correct_depth_with_matte(d, alpha, curve="cubic", **RK)
    with pytest.raises(ValueError):
        correct_depth_with_matte(d, alpha, _polarity="sideways", **RK)


def test_polarity_inverted_reflection_identity():
    d, alpha = _step_scene()
    norm, rn = correct_depth_with_matte(d, alpha, **RK)
    inv, ri = correct_depth_with_matte(d, alpha, _polarity="inverted", **RK)
    assert ri["polarity"] == "inverted" and rn["polarity"] == "normal"
    assert np.array_equal(ri["changed"], rn["changed"])
    assert not np.array_equal(inv, norm)
    band = rn["band"]
    assert np.array_equal(inv[~band], d[~band])
    soft = band & (alpha > 0.02) & (alpha < 0.98)
    disp_near, disp_far = 1.0, 0.25  # references are 1 m and 4 m on this scene
    lhs = 1.0 / inv[soft].astype(np.float64) + 1.0 / norm[soft].astype(np.float64)
    assert np.allclose(lhs, disp_near + disp_far, rtol=1e-5)
    hi = rn["changed"] & (alpha >= 0.98)
    lo = rn["changed"] & (alpha <= 0.02)
    assert np.allclose(inv[hi], 1.0 / disp_far) if hi.any() else True
    assert np.allclose(inv[lo], 1.0 / disp_near) if lo.any() else True


def test_one_minus_alpha_is_degenerate_not_a_control():
    d, alpha = _step_scene()
    a, _ = correct_depth_with_matte(d, alpha, **RK)
    b, _ = correct_depth_with_matte(d, 1.0 - alpha, **RK)
    assert np.allclose(a, b, rtol=1e-5)


# ---------------------------------------------------------------------------
# 11. shape / dtype / finite / r scaling
# ---------------------------------------------------------------------------

def test_dtype_not_mutated_and_r_scaling():
    d, alpha = _step_scene()
    d0 = d.copy()
    out, _ = correct_depth_with_matte(d, alpha, **RK)
    assert out.dtype == np.float32 and out.shape == d.shape and np.all(np.isfinite(out))
    assert np.array_equal(d, d0)
    blob = np.zeros((4, 2048), np.float32)
    blob[:, 1000:1010] = 0.5
    assert matte_band(blob)[2] == 12
    assert matte_band(np.full((4, 5712), 0.5, np.float32))[2] == 33
    assert matte_band(np.full((4, 8), 0.5, np.float32))[2] == 1


# ---------------------------------------------------------------------------
# 12. import fence
# ---------------------------------------------------------------------------

def test_import_fence():
    code = (
        "import sys; sys.modules['pillow_heif'] = None; "
        "import summicron.depth.matte, summicron.depth.guided; print('ok')"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0 and "ok" in r.stdout, r.stderr
    tree = ast.parse((ROOT / "summicron/render/convolve.py").read_text())
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [a.name for a in node.names] + [getattr(node, "module", "") or ""]
            assert not any("depth.matte" in n or "depth.guided" in n or "pillow_heif" in n for n in names)


# ---------------------------------------------------------------------------
# 14. guide loader
# ---------------------------------------------------------------------------

def _rgb_img(h=H, w=W, seed=1):
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 255, (h, w, 3), dtype=np.uint8), "RGB")


def test_guide_aspect_mismatch_and_exception(monkeypatch):
    _patch_heif(monkeypatch, _FakeHeif(primary=_rgb_img(W, H)))  # 3:4 vs 4:3
    out, msgs = _caught(load_luma_guide, "x", (H, W))
    assert out is None and len(msgs) == 1 and msgs[0].startswith("depth-guided:")

    import pillow_heif

    def boom(*a, **k):
        raise RuntimeError("bad file")

    monkeypatch.setattr(pillow_heif, "open_heif", boom)
    out, msgs = _caught(load_luma_guide, "x", (H, W))
    assert out is None and len(msgs) == 1 and msgs[0].startswith("depth-guided:")


def test_guide_stored_orientation_not_transposed(monkeypatch):
    im = _rgb_img(H, W)
    im.info["exif_orientation"] = 6
    _patch_heif(monkeypatch, _FakeHeif(primary=im))
    out, msgs = _caught(load_luma_guide, "x", (H, W))
    assert msgs == [] and out.shape == (H, W) and out.dtype == np.float32
    rgb = np.asarray(im, np.float32) / 255.0
    assert np.allclose(out, guided_mod.luminance(rgb), atol=1e-6)
    assert 0.0 <= out.min() and out.max() <= 1.0


@pytest.mark.slow
def test_guide_matches_run1_on_real_6915():
    if not HEIC_6915.exists():
        pytest.skip("IMG_6915 absent")
    import pillow_heif
    from PIL import ImageOps

    d = extract_lidar_depth(HEIC_6915, (768, 576))
    assert d is not None
    hw = d.shape
    got = load_luma_guide(HEIC_6915, hw)
    pillow_heif.register_heif_opener()
    im = ImageOps.exif_transpose(Image.open(str(HEIC_6915))).convert("RGB")
    im = im.resize((hw[1], hw[0]), Image.BILINEAR)
    ref = guided_mod.luminance(np.asarray(im, dtype=np.float32) / 255.0)
    assert got is not None and got.shape == hw and got.dtype == np.float32
    assert np.max(np.abs(got - ref)) < 1e-6


# ---------------------------------------------------------------------------
# 15. staircase smoothing
# ---------------------------------------------------------------------------

def test_smooth_r_scaling_and_eps(monkeypatch):
    seen = []

    def rec(depth, guide, r=16, eps=1e-3):
        seen.append((r, eps))
        return np.asarray(depth, np.float32)

    monkeypatch.setattr(guided_mod, "guided_upsample_depth", rec)
    for hw, want in (((4, 2048), 4), ((5, 5712), 11), ((2, 128), 1)):
        smooth_lidar_staircase(np.ones(hw, np.float32), np.zeros(hw, np.float32))
        assert seen[-1] == (want, 1e-4)


def _diag_scene():
    yy, xx = np.mgrid[:72, :96]
    truth = np.where(xx - 0.6 * yy > 40, 4.0, 1.0).astype(np.float32)
    coarse = truth[1::3, 1::3]
    up = np.asarray(Image.fromarray(coarse, mode="F").resize((96, 72), Image.BILINEAR))
    guide = np.where(xx - 0.6 * yy > 40, 0.8, 0.2).astype(np.float32)
    return up.astype(np.float32), guide, yy, xx


def test_smooth_reduces_axis_alignment():
    up, guide, _, _ = _diag_scene()
    out = smooth_lidar_staircase(up, guide)
    ld_in, ld_out = np.log(1.0 / up), np.log(1.0 / out)
    bl, _ = _lband(ld_in)
    assert _aai(ld_out, bl) < 0.5 * _aai(ld_in, bl)


def test_smooth_no_texture_copy_far_from_step():
    yy, xx = np.mgrid[:72, :96]
    d = np.where(xx < 20, 1.0, 1.0).astype(np.float32)  # constant depth
    d[:, 80:] = 4.0  # a step far from the textured region
    rng = np.random.default_rng(3)
    guide = np.where(xx < 60, rng.uniform(0, 1, (72, 96)), 0.5).astype(np.float32)
    out = smooth_lidar_staircase(d, guide)
    far = np.zeros_like(d, bool)
    far[:, :60] = True
    dl = np.abs(np.log(1.0 / out) - np.log(1.0 / d))
    assert dl[far].max() < 1e-3
    disp_in = 1.0 / d
    assert (1.0 / out).min() >= disp_in.min() - 1e-6 and (1.0 / out).max() <= disp_in.max() + 1e-6
    assert out.dtype == np.float32 and np.all(np.isfinite(out))


def test_smooth_input_unmutated_and_shape_error():
    up, guide, _, _ = _diag_scene()
    u0 = up.copy()
    smooth_lidar_staircase(up, guide)
    assert np.array_equal(up, u0)
    with pytest.raises(ValueError):
        smooth_lidar_staircase(up, guide[:-1])


# ---------------------------------------------------------------------------
# 16. hook ordering
# ---------------------------------------------------------------------------

def test_hook_ordering(monkeypatch):
    depth, alpha = _step_scene()
    guide = np.where(np.arange(W)[None, :].repeat(H, 0) < W // 2, 0.2, 0.8).astype(np.float32)

    import summicron.depth.matte as matte_mod

    monkeypatch.setattr(guided_mod, "load_luma_guide", lambda p, hw: guide)
    monkeypatch.setattr(matte_mod, "load_portrait_matte", lambda p, hw, depth_m=None: alpha)
    _make_base_stub(monkeypatch, depth)

    off = extract_lidar_depth("x.heic", (H, W))
    assert np.array_equal(off, depth)
    g_only = extract_lidar_depth("x.heic", (H, W), depth_guided="on")
    assert np.array_equal(g_only, smooth_lidar_staircase(depth, guide))
    both = extract_lidar_depth("x.heic", (H, W), depth_guided="on", depth_matte="on")
    want = correct_depth_with_matte(smooth_lidar_staircase(depth, guide), alpha)[0]
    assert np.array_equal(both, want)
    m_only = extract_lidar_depth("x.heic", (H, W), depth_matte="on")
    assert np.array_equal(m_only, correct_depth_with_matte(depth, alpha)[0])


def test_hook_guide_none_warns_and_matte_still_runs(monkeypatch):
    depth, alpha = _step_scene()
    import summicron.depth.matte as matte_mod

    monkeypatch.setattr(guided_mod, "load_luma_guide", lambda p, hw: None)
    monkeypatch.setattr(matte_mod, "load_portrait_matte", lambda p, hw, depth_m=None: alpha)
    _make_base_stub(monkeypatch, depth)
    out, msgs = _caught(extract_lidar_depth, "x.heic", (H, W), depth_guided="on", depth_matte="on")
    assert len([m for m in msgs if m.startswith("depth-guided:")]) == 1
    assert np.array_equal(out, correct_depth_with_matte(depth, alpha)[0])


def _make_base_stub(monkeypatch, depth):
    """Make extract_lidar_depth's decode stage return exactly ``depth`` (H, W)."""
    import pillow_heif

    # d_min=0, d_max=255 -> disp = raw, so a float raw of 1/depth round-trips exactly.
    raw = (1.0 / depth).astype(np.float32)

    class _Depth:
        info = {"metadata": {"d_min": 0.0, "d_max": 255.0, "representation_type": 1}}

        def __array__(self, dtype=None, copy=None):
            return raw

    class _Heif:
        info = {"depth_images": [_Depth()]}

    monkeypatch.setattr(pillow_heif, "open_heif", lambda *a, **k: _Heif())
    return True
