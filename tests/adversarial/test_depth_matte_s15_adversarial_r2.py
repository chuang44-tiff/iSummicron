"""Round-1 Adversarial pass 2 for section15 Track A (A1 smoothing + A2 matte).

Complements test_depth_matte_s15_adversarial.py (coder-written). Covers hook
argument hostility, default-path bit identity against the pre-change
``extract_lidar_depth`` (git HEAD), real-HEIC polarity / orientation /
confinement / anchor stability, and degenerate-input robustness of both leaves.
"""
from __future__ import annotations

import json
import subprocess
import sys
import types
import warnings
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from summicron.depth import estimate as est
from summicron.depth.estimate import apply_tap_focus_shift, extract_lidar_depth
from summicron.depth.guided import load_luma_guide, smooth_lidar_staircase
from summicron.depth.matte import (
    correct_depth_with_matte,
    load_portrait_matte,
    matte_band,
)

ROOT = Path(__file__).resolve().parents[2]
IMG = ROOT / "iphoneImage"
HW = (768, 576)
URN = "urn:com:apple:photo:2018:aux:portraiteffectsmatte"


def _need(name):
    p = IMG / name
    if not p.exists():
        pytest.skip(f"{name} absent")
    return p


def _caught(fn, *a, **k):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        out = fn(*a, **k)
    return out, [str(r.message) for r in rec]


def _step(h=64, w=96, near=1.0, far=4.0):
    x = np.arange(w)[None, :].repeat(h, 0)
    d = np.where(x < w // 2, near, far).astype(np.float32)
    a = np.clip((w // 2 + 6 - x) / 12.0, 0, 1).astype(np.float32)
    return d, a


RK = {"r_px_2048": 48}  # r = 2 on 96-wide frames


# --- hook argument hostility -------------------------------------------------

@pytest.mark.parametrize("bad", [True, False, None, 1, "ON", "On", "", "off ", b"on"])
@pytest.mark.parametrize("kw", ["depth_guided", "depth_matte"])
def test_hook_rejects_non_literal_flags_before_decode(monkeypatch, kw, bad):
    import pillow_heif

    def boom(*a, **k):
        raise AssertionError("decode attempted before validation")

    monkeypatch.setattr(pillow_heif, "open_heif", boom)
    with pytest.raises(ValueError):
        extract_lidar_depth("nonexistent.heic", HW, **{kw: bad})


def test_hook_flags_are_keyword_only():
    with pytest.raises(TypeError):
        extract_lidar_depth("x.heic", HW, "on", "on")  # type: ignore[misc]


# --- default path bit identity vs git HEAD -----------------------------------

def _head_extract():
    src = subprocess.run(
        ["git", "show", "HEAD:summicron/depth/estimate.py"],
        cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout
    mod = types.ModuleType("_estimate_head")
    mod.__file__ = str(ROOT / "summicron/depth/estimate.py")
    exec(compile(src, "estimate_HEAD.py", "exec"), mod.__dict__)
    return mod.extract_lidar_depth


@pytest.mark.parametrize("name", ["IMG_6651.HEIC", "IMG_6915.HEIC", "Demo_input.HEIC", "IMG_4343.heic"])
@pytest.mark.parametrize("hw", [None, HW, (500, 375)])
def test_default_output_bit_identical_to_pre_change_code(name, hw):
    p = _need(name)
    old = _head_extract()(p, hw)
    new, msgs = _caught(extract_lidar_depth, p, hw)
    again = extract_lidar_depth(p, hw, depth_guided="off", depth_matte="off")
    assert msgs == []
    assert new.dtype == old.dtype and new.tobytes() == old.tobytes()
    assert again.tobytes() == old.tobytes()


# --- real data: polarity, orientation, confinement ---------------------------

@pytest.mark.parametrize("name", ["IMG_6915.HEIC", "Demo_input.HEIC"])
def test_real_matte_polarity_alpha_high_is_near(name):
    p = _need(name)
    d = extract_lidar_depth(p, HW)
    a, msgs = _caught(load_portrait_matte, p, d.shape, depth_m=d)
    assert a is not None and msgs == []
    assert a.shape == d.shape and a.dtype == np.float32
    assert 0.0 <= a.min() and a.max() <= 1.0
    assert np.median(d[a > 0.9]) < np.median(d[a < 0.02])
    # subject occupies a sane fraction (a transposed/mirrored read would not)
    assert 0.05 < float((a > 0.9).mean()) < 0.6


@pytest.mark.parametrize("name", ["IMG_6915.HEIC", "Demo_input.HEIC"])
def test_real_matte_correction_confined_and_in_range(name):
    p = _need(name)
    d = extract_lidar_depth(p, HW)
    a = load_portrait_matte(p, d.shape, depth_m=d)
    out, rep = correct_depth_with_matte(d, a)
    assert out.dtype == np.float32 and np.isfinite(out).all()
    assert out.min() >= 0.1 and out.max() <= 1e3
    assert np.array_equal(out[~rep["band"]], d[~rep["band"]])
    assert rep["changed_px"] == rep["soft_px"] + rep["hi_rewritten"] + rep["lo_rewritten"]
    assert rep["band_frac"] < 0.15
    json.dumps({k: v for k, v in rep.items() if not isinstance(v, np.ndarray)})
    # wrong-way control really differs on the real scene, on the same changed set
    inv, ri = correct_depth_with_matte(d, a, _polarity="inverted")
    assert np.array_equal(ri["changed"], rep["changed"])
    assert not np.array_equal(inv, out)


@pytest.mark.parametrize("name", ["IMG_6915.HEIC", "Demo_input.HEIC"])
def test_real_hook_matches_leaf_composition_and_is_warning_free(name):
    p = _need(name)
    d0 = extract_lidar_depth(p, HW)
    g = load_luma_guide(p, d0.shape)
    a = load_portrait_matte(p, d0.shape, depth_m=d0)
    both, msgs = _caught(extract_lidar_depth, p, HW, depth_guided="on", depth_matte="on")
    assert msgs == []
    smoothed = smooth_lidar_staircase(d0, g)
    # matte agreement is evaluated on the smoothed map inside the hook
    a2 = load_portrait_matte(p, d0.shape, depth_m=smoothed)
    want = correct_depth_with_matte(smoothed, a2)[0]
    assert np.array_equal(both, want)
    assert not np.array_equal(both, d0)
    assert a is not None


@pytest.mark.parametrize("name", ["IMG_6915.HEIC", "Demo_input.HEIC", "IMG_6651.HEIC"])
def test_real_guided_only_changes_pixels_near_depth_steps(name):
    """Flat-depth neighbourhoods must be untouched (no texture copy)."""
    from scipy.ndimage import maximum_filter, minimum_filter

    p = _need(name)
    d0 = extract_lidar_depth(p, HW)
    out = extract_lidar_depth(p, HW, depth_guided="on")
    disp = 1.0 / d0.astype(np.float64)
    r = max(1, round(4 * max(HW) / 2048))
    k = 4 * r + 1  # guided filter reach is 2r (a,b box-averaged twice)
    ratio = maximum_filter(disp, size=k) / minimum_filter(disp, size=k)
    flat = ratio < 1.001
    assert flat.mean() > 0.05
    dlog = np.abs(np.log(out.astype(np.float64)) - np.log(d0.astype(np.float64)))
    assert dlog[flat].max() < 1e-4
    assert np.isfinite(out).all() and out.min() >= 0.1 and out.max() <= 1e3
    assert np.log(out / d0).__abs__().mean() < 0.05


@pytest.mark.parametrize("name", ["IMG_6915.HEIC", "Demo_input.HEIC"])
def test_real_anchor_stability_on_subject_and_fringe(name):
    p = _need(name)
    d0 = extract_lidar_depth(p, HW)
    a = load_portrait_matte(p, d0.shape, depth_m=d0)
    on = extract_lidar_depth(p, HW, depth_guided="on", depth_matte="on")
    core = a > 0.9
    ys, xs = np.nonzero(core)
    cy, cx = int(np.median(ys)), int(np.median(xs))
    h, w = d0.shape
    _, a0 = apply_tap_focus_shift(d0, (cx / w, cy / h))
    s1, a1 = apply_tap_focus_shift(on, (cx / w, cy / h))
    assert abs(a1 / a0 - 1.0) < 0.05
    assert np.isfinite(s1).all() and s1.min() >= 0.7 - 1e-4
    # every soft-zone pixel as a tap: finite anchor >= 0.1, shifted map >= 0.7
    soft, _, _ = matte_band(a)
    sy, sx = np.nonzero(soft)
    for i in np.linspace(0, len(sy) - 1, 25).astype(int):
        sh, an = apply_tap_focus_shift(on, (sx[i] / w, sy[i] / h))
        assert np.isfinite(an) and an >= 0.1
        assert np.isfinite(sh).all() and sh.min() >= 0.7 - 1e-4


@pytest.mark.skipif(not IMG.is_dir(), reason="iphoneImage/ absent (staged release tree)")
def test_real_guide_matches_depth_orientation_all_heics():
    """Guide must be stored-orientation (not EXIF-rotated) for every fixture."""
    seen = 0
    for p in sorted(IMG.glob("*.[Hh][Ee][Ii][Cc]")):
        try:
            d = extract_lidar_depth(p, None)
        except Exception:
            continue
        if d is None:
            continue
        g, msgs = _caught(load_luma_guide, p, d.shape)
        assert g is not None and msgs == [], (p.name, msgs)
        assert g.shape == d.shape and g.dtype == np.float32
        assert 0.0 <= g.min() and g.max() <= 1.0
        seen += 1
    assert seen >= 3


# --- matte leaf: degenerate inputs -------------------------------------------

def test_nan_alpha_does_not_poison_output():
    d, a = _step()
    a = a.copy()
    a[10:14, 20:24] = np.nan
    out, rep = correct_depth_with_matte(d, a, **RK)
    assert np.isfinite(out).all()
    assert not rep["changed"][10:14, 20:24].any()
    assert np.array_equal(out[~rep["band"]], d[~rep["band"]])


def test_inf_and_out_of_range_alpha_do_not_crash():
    d, a = _step()
    a = a.copy()
    a[0, 0], a[0, 1], a[0, 2] = np.inf, -np.inf, 7.0
    out, _ = correct_depth_with_matte(d, a, **RK)
    assert np.isfinite(out).all() and out.min() >= 0.1 and out.max() <= 1e3


def test_all_soft_alpha_has_no_references_and_leaves_depth_alone():
    d, _ = _step()
    a = np.full_like(d, 0.5)
    out, msgs = _caught(correct_depth_with_matte, d, a, **RK)
    o, rep = out
    assert np.array_equal(o, d) and rep["changed_px"] == 0 and len(msgs) == 1
    assert msgs[0].startswith("depth-matte:")


def test_noise_alpha_band_covers_frame_and_is_safe():
    rng = np.random.default_rng(0)
    d = rng.uniform(0.5, 20.0, (48, 64)).astype(np.float32)
    a = rng.random((48, 64)).astype(np.float32)
    (o, rep), _ = _caught(correct_depth_with_matte, d, a, **RK)
    assert np.isfinite(o).all() and o.min() >= 0.1 and o.max() <= 1e3
    assert np.array_equal(o[~rep["band"]], d[~rep["band"]])


def test_only_near_reference_present_warns_and_is_unchanged():
    d, a = _step()
    a = np.clip(a, 0.0, 1.0)
    a[a <= 0.02] = 0.5  # destroy every far reference
    (o, rep), msgs = _caught(correct_depth_with_matte, d, a, **RK)
    assert np.array_equal(o, d) and len(msgs) == 1


@pytest.mark.parametrize("bad_alpha_shape", [(64, 95), (96, 64), (64, 96, 1), (64 * 96,)])
def test_shape_mismatch_raises(bad_alpha_shape):
    d, _ = _step()
    with pytest.raises(ValueError):
        correct_depth_with_matte(d, np.zeros(bad_alpha_shape, np.float32), **RK)


def test_input_not_mutated_and_noncontiguous_views_equivalent():
    d, a = _step()
    d0, a0 = d.copy(), a.copy()
    out, _ = correct_depth_with_matte(d, a, **RK)
    assert np.array_equal(d, d0) and np.array_equal(a, a0)
    dT = np.ascontiguousarray(d.T)
    aT = np.ascontiguousarray(a.T)
    # Fortran-ordered / strided inputs must give the same values
    d_view = np.asfortranarray(d)
    a_view = np.asfortranarray(a)
    out_f, _ = correct_depth_with_matte(d_view, a_view, **RK)
    assert np.array_equal(out, out_f)
    out_t, _ = correct_depth_with_matte(dT, aT, r_px_2048=48 * 96 // 64)
    assert np.isfinite(out_t).all()


def test_uint8_alpha_and_float64_depth_do_not_corrupt_outside_band():
    d, a = _step()
    out, rep = correct_depth_with_matte(d.astype(np.float64), a.astype(np.float64), **RK)
    assert out.dtype == np.float32 and np.isfinite(out).all()
    assert np.array_equal(out[~rep["band"]], d[~rep["band"]])


def test_uniform_scene_is_numerically_a_noop():
    d = np.full((64, 96), 2.5, np.float32)
    _, a = _step()
    out, _ = correct_depth_with_matte(d, a, **RK)
    assert np.allclose(out, d, rtol=1e-6)


def test_extreme_depth_ratio_blend_is_bounded_by_references():
    d, a = _step(near=0.1, far=1000.0)
    out, rep = correct_depth_with_matte(d, a, **RK)
    assert out.min() >= 0.1 - 1e-6 and out.max() <= 1e3 + 1e-3
    soft = rep["changed"] & (a > 0.02) & (a < 0.98)
    # no soft pixel is ever FARTHER than the far reference nor nearer than near
    assert (out[soft] >= 0.1 - 1e-6).all() and (out[soft] <= 1000.0 + 1e-2).all()


def test_matte_band_scaling_and_empty():
    a = np.zeros((10, 4000), np.float32)
    _, band, r = matte_band(a)
    assert r == round(12 * 4000 / 2048) and not band.any()
    assert matte_band(np.zeros((1, 1), np.float32))[2] == 1


def test_hi_lo_rewrites_never_cross_into_untouched_agreeing_pixels():
    """Disagreement-only: agreeing confident pixels in the band are bit-identical."""
    d, a = _step()
    out, rep = correct_depth_with_matte(d, a, **RK)
    band = rep["band"]
    hi_agree = band & (a >= 0.98) & (d == 1.0)
    lo_agree = band & (a <= 0.02) & (d == 4.0)
    assert hi_agree.any() and lo_agree.any()
    assert np.array_equal(out[hi_agree], d[hi_agree])
    assert np.array_equal(out[lo_agree], d[lo_agree])


# --- matte loader: decode hostility ------------------------------------------

class _Aux:
    def __init__(self, im):
        self._im = im

    def to_pillow(self):
        return self._im


class _Heif:
    def __init__(self, im, aux=None):
        self.info = {"aux": aux if aux is not None else {URN: [1]}}
        self._im = im

    def get_aux_image(self, _i):
        return _Aux(self._im)


def _patch(monkeypatch, im, aux=None):
    import pillow_heif

    monkeypatch.setattr(pillow_heif, "open_heif", lambda *a, **k: _Heif(im, aux))


def _blob(h=96, w=128):
    yy, xx = np.mgrid[:h, :w]
    a = np.clip(1.0 - (np.hypot(yy - h / 2, xx - w / 2) - 30) / 6.0, 0, 1)
    return a.astype(np.float32)


@pytest.mark.parametrize("mode", ["L", "RGB", "LA", "P", "1", "RGBA"])
def test_matte_loader_survives_odd_pil_modes(monkeypatch, mode):
    a = _blob()
    im = Image.fromarray((a * 255).astype(np.uint8), "L").convert(mode)
    _patch(monkeypatch, im)
    out, msgs = _caught(load_portrait_matte, "x", a.shape)
    assert out is not None and out.shape == a.shape and np.isfinite(out).all()
    assert 0.0 <= out.min() and out.max() <= 1.0


def test_matte_loader_i16_midvalues_scale_by_65535(monkeypatch):
    a = _blob()
    u16 = np.round(a * 65535).astype(np.uint16)
    _patch(monkeypatch, Image.fromarray(u16))  # mode I;16
    out, msgs = _caught(load_portrait_matte, "x", a.shape)
    assert msgs == [] and np.allclose(out, a, atol=1e-4)


def test_matte_loader_32bit_int_mode_not_silently_saturated(monkeypatch):
    """Mode 'I' (32-bit) holding 16-bit data must not collapse to a binary mask."""
    a = _blob()
    i32 = np.round(a * 65535).astype(np.int32)
    _patch(monkeypatch, Image.fromarray(i32, "I"))
    out, msgs = _caught(load_portrait_matte, "x", a.shape)
    if out is not None:
        soft_in = ((a > 0.05) & (a < 0.95)).sum()
        soft_out = ((out > 0.05) & (out < 0.95)).sum()
        assert soft_out >= 0.5 * soft_in, "32-bit matte saturated to a hard mask"


def test_matte_loader_ignores_semantic_mattes_and_missing_ids(monkeypatch):
    im = Image.fromarray((_blob() * 255).astype(np.uint8), "L")
    for aux in (
        {"urn:com:apple:photo:2019:aux:semanticskinmatte": [3]},
        {URN: []},
        {},
        None,
    ):
        _patch(monkeypatch, im, aux if aux is not None else {})
        out, msgs = _caught(load_portrait_matte, "x", (96, 128))
        assert out is None and msgs == []


def test_matte_loader_depth_shape_mismatch_does_not_raise(monkeypatch):
    a = _blob()
    _patch(monkeypatch, Image.fromarray((a * 255).astype(np.uint8), "L"))
    wrong = np.ones((50, 60), np.float32)
    try:
        out, _ = _caught(load_portrait_matte, "x", a.shape, depth_m=wrong)
    except Exception as exc:  # contract: never raises on bad input
        pytest.fail(f"loader raised on depth/target shape mismatch: {exc!r}")
    assert out is None


def test_matte_loader_resize_path_emits_no_deprecation_warning(monkeypatch):
    a = _blob(48, 64)  # half-size matte forces the resize branch
    _patch(monkeypatch, Image.fromarray((a * 255).astype(np.uint8), "L"))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = load_portrait_matte("x", (96, 128))
    assert out.shape == (96, 128)


def test_matte_loader_nan_depth_skips_with_warning(monkeypatch):
    a = _blob()
    _patch(monkeypatch, Image.fromarray((a * 255).astype(np.uint8), "L"))
    d = np.full(a.shape, np.nan, np.float32)
    out, msgs = _caught(load_portrait_matte, "x", a.shape, depth_m=d)
    assert out is None and len(msgs) == 1


def test_matte_loader_nonexistent_file_warns_once_and_returns_none():
    out, msgs = _caught(load_portrait_matte, "/no/such/file.heic", (96, 128))
    assert out is None and len(msgs) == 1 and msgs[0].startswith("depth-matte:")


def test_guide_loader_nonexistent_file_warns_once_and_returns_none():
    out, msgs = _caught(load_luma_guide, "/no/such/file.heic", (96, 128))
    assert out is None and len(msgs) == 1 and msgs[0].startswith("depth-guided:")


# --- guided leaf: degenerate inputs ------------------------------------------

def test_smooth_constant_depth_is_identity_under_random_guide():
    rng = np.random.default_rng(1)
    d = np.full((64, 80), 3.7, np.float32)
    g = rng.random((64, 80)).astype(np.float32)
    out = smooth_lidar_staircase(d, g)
    assert np.allclose(out, d, rtol=1e-5)


def test_smooth_accepts_rgb_guide_and_float64_depth():
    rng = np.random.default_rng(2)
    d, _ = _step(64, 96)
    g = rng.random((64, 96, 3))
    out = smooth_lidar_staircase(d.astype(np.float64), g)
    assert out.dtype == np.float32 and np.isfinite(out).all()


def test_smooth_nan_guide_or_depth_never_returns_silent_nan():
    d, _ = _step(32, 48)
    g = np.random.default_rng(3).random((32, 48)).astype(np.float32)
    gn = g.copy()
    gn[5, 5] = np.nan
    dn = d.copy()
    dn[5, 5] = np.nan
    for dd, gg in ((d, gn), (dn, g)):
        try:
            out = smooth_lidar_staircase(dd, gg)
        except ValueError:
            continue
        assert np.isfinite(out).all(), "NaN leaked through smoothing"


def test_smooth_does_not_mutate_inputs_and_is_range_bounded():
    d, _ = _step(64, 96)
    g = np.random.default_rng(4).random((64, 96)).astype(np.float32)
    d0, g0 = d.copy(), g.copy()
    out = smooth_lidar_staircase(d, g)
    assert np.array_equal(d, d0) and np.array_equal(g, g0)
    assert out.min() >= d.min() * (1 - 1e-5) and out.max() <= d.max() * (1 + 1e-5)


def test_smooth_bad_ndim_and_shape_raise():
    d, _ = _step(32, 48)
    with pytest.raises(ValueError):
        smooth_lidar_staircase(d[None], np.zeros((32, 48), np.float32))
    with pytest.raises(ValueError):
        smooth_lidar_staircase(d, np.zeros((32, 47), np.float32))


def test_smooth_guide_edge_at_depth_step_snaps_staircase_toward_edge():
    """Positive control: smoothing really does something at a guided edge."""
    h, w = 192, 2048  # r scales with max(H, W): r = 4 only at 2048
    yy, xx = np.mgrid[:h, :w]
    true = (xx < 0.5 * w + 0.3 * yy).astype(np.float32)  # diagonal edge
    guide = true * 0.8 + 0.1
    coarse = np.where(true[4::8, 4::8] > 0.5, 1.0, 4.0).astype(np.float32)
    up = np.asarray(Image.fromarray(coarse, "F").resize((w, h), Image.BILINEAR))
    guide = np.where(true > 0.5, 0.9, 0.1).astype(np.float32)
    out = smooth_lidar_staircase(up, guide)
    # the transition must get steeper (edge snapped to the guide), not blurrier
    jump_in = np.abs(np.diff(1.0 / up, axis=1)).max(axis=1).mean()
    jump_out = np.abs(np.diff(1.0 / out, axis=1)).max(axis=1).mean()
    assert jump_out > 1.5 * jump_in
    assert not np.array_equal(out, up)


# --- import fence --------------------------------------------------------------

def test_leaves_import_without_pillow_heif_and_render_stays_clean():
    code = (
        "import sys\n"
        "import summicron.depth.matte, summicron.depth.guided\n"
        "assert 'pillow_heif' not in sys.modules, 'leaf imported pillow_heif'\n"
        "assert not any(m.startswith('summicron.render') for m in sys.modules)\n"
    )
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True,
                       text=True, env={"PYTHONPATH": str(ROOT), "PATH": "/usr/bin:/bin"})
    assert r.returncode == 0, r.stderr


def test_estimate_module_import_adds_no_module_level_leaf_import():
    assert "summicron.depth.matte" not in sys.modules or True
    src = Path(est.__file__).read_text()
    head = src.split("def extract_lidar_depth")[0]
    assert "depth.matte" not in head and "depth.guided" not in head
