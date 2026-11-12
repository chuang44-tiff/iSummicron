"""Second adversarial pass on summicron.depth.diopter (Q-d S0).

Covers: non-finite inputs, degenerate gain, anchor stability, auto_gain
edge inputs, input-type/shape variety, default-path (meters) isolation and
bit-identity guards.  Tests that document a leaf-level gap that S1 must
close by validation are marked in their docstring (GAP).
"""
from __future__ import annotations

import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

from summicron.depth.diopter import auto_gain, diopter_equivalent_depth

OD = np.array([0.70, 0.98, 1.37, 1.91, 2.68, 3.74, 5.23, 7.32, 10.23, 14.30, 20.00])
ZF, ZFAR = 0.7, 20.0
ROOT = Path(__file__).resolve().parents[2]


def _map(depth, anchor=0.9626, gain=1.25, **kw):
    return diopter_equivalent_depth(depth, anchor, gain=gain, z_focus=ZF, z_far=ZFAR, **kw)


# ---------------- non-finite inputs ----------------

def test_inf_and_nan_depth_map_to_focus_without_warnings():
    d = np.array([np.nan, np.inf, -np.inf, 1.0, 5.0])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = _map(d)
    assert np.isfinite(out).all()
    assert out[0] == out[1] == out[2] == np.float32(ZF)
    assert out[3] > ZF and out[4] > out[3]


def test_all_nan_frame_is_all_focus():
    out = _map(np.full((4, 4), np.nan))
    assert (out == np.float32(ZF)).all()


def test_nan_anchor_does_not_crash_and_stays_in_range():
    """Spec says NaN anchor is impossible; leaf must still not emit NaN/out-of-range."""
    out = _map(np.array([0.5, 1.0, 50.0]), anchor=float("nan"))
    assert np.isfinite(out).all()
    assert out.min() >= ZF - 1e-6 and out.max() <= ZFAR + 1e-4


def test_inf_anchor_in_range():
    out = _map(np.array([0.5, 1.0, 50.0]), anchor=float("inf"))
    assert np.isfinite(out).all()
    assert out.min() >= ZF - 1e-6 and out.max() <= ZFAR + 1e-4


# ---------------- degenerate gain (S1 validates; leaf should not blow up) ----------------

def test_zero_gain_all_focus():
    assert (_map(np.array([0.3, 1.0, 30.0]), gain=0.0) == np.float32(ZF)).all()


@pytest.mark.xfail(strict=True, reason="leaf does not validate gain; S1 RenderOptions must (spec 2.4)")
def test_negative_gain_stays_in_range():
    """GAP: negative gain pushes e<0 so z_t < z_focus (outside tensor axis)."""
    out = _map(np.array([0.3, 1.0, 30.0]), gain=-1.0)
    assert out.min() >= ZF - 1e-6


@pytest.mark.xfail(strict=True, reason="leaf does not validate gain; S1 RenderOptions must (spec 2.4)")
def test_inf_gain_no_nan_at_anchor_pixel():
    """GAP: inf * dp(=0) = NaN at the anchor pixel."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = _map(np.array([0.9626, 1.0, 30.0]), gain=float("inf"))
    assert np.isfinite(out).all()


@pytest.mark.xfail(strict=True, reason="leaf does not validate gain; S1 RenderOptions must (spec 2.4)")
def test_nan_gain_no_nan_output():
    """GAP: NaN gain propagates NaN into the dmap."""
    out = _map(np.array([0.9626, 1.0, 30.0]), gain=float("nan"))
    assert np.isfinite(out).all()


# ---------------- types / shapes ----------------

@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64, np.int32, np.int64])
def test_dtype_variety(dtype):
    d = np.array([[1, 2, 3], [4, 5, 6]]).astype(dtype)
    out = _map(d)
    assert out.dtype == np.float32 and out.shape == d.shape
    assert np.isfinite(out).all()


def test_list_and_scalar_input():
    assert _map([[1.0, 2.0]]).shape == (1, 2)
    out = _map(1.0)
    assert np.ndim(out) == 0 and np.isfinite(out)


def test_empty_array():
    out = _map(np.zeros((0, 5)))
    assert out.shape == (0, 5)


def test_3d_and_noncontiguous():
    d = np.random.default_rng(0).uniform(0.4, 20, (6, 8, 3))[:, ::2, :]
    out = _map(d)
    assert out.shape == d.shape and np.isfinite(out).all()
    assert np.array_equal(out, _map(np.ascontiguousarray(d)))


def test_masked_array_does_not_silently_use_mask_garbage():
    d = np.ma.masked_array([1.0, 1e9, 3.0], mask=[False, True, False])
    out = _map(d)
    assert np.isfinite(out).all()


def test_readonly_input_ok():
    d = np.linspace(0.5, 20, 50)
    d.setflags(write=False)
    _map(d)


def test_nan_in_input_not_modified():
    d = np.array([np.nan, 1.0])
    _map(d)
    assert np.isnan(d[0])


# ---------------- invariants ----------------

def test_output_only_depends_on_abs_dp_over_random_scene():
    rng = np.random.default_rng(5)
    a = 1.1
    dp = rng.uniform(0, 0.8, 200)
    zn = 1 / (1 / a + dp)
    zf_ = 1 / (1 / a - dp)  # may be negative when dp>1/a: filtered
    ok = (1 / a - dp) > 1e-3
    assert np.allclose(_map(zn[ok], a), _map(zf_[ok], a), rtol=1e-5)


def test_invalid_axis_far_le_focus_does_not_crash_silently_out_of_range():
    """z_far <= z_focus makes cap<=0; output must still be finite."""
    out = diopter_equivalent_depth(np.array([1.0, 2.0]), 1.0, gain=1.0, z_focus=2.0, z_far=1.0)
    assert np.isfinite(out).all()


def test_eps_floor_extreme_depth_matches_eps_depth():
    assert _map(np.array([1e-9]))[0] == _map(np.array([1e-3]))[0]


# ---------------- anchor stability ----------------

@pytest.mark.parametrize("anchor", [0.618, 0.7, 0.9626])
def test_anchor_half_percent_shift_small_effect(anchor):
    rng = np.random.default_rng(1)
    d = rng.uniform(0.5, 6.0, 2000)
    r0 = np.interp(_map(d, anchor), OD, np.linspace(0, 1, 11))
    r1 = np.interp(_map(d, anchor * 1.005), OD, np.linspace(0, 1, 11))
    assert np.abs(r0 - r1).max() < 0.10  # inherent slope: ~0.05 observed


def test_anchor_continuity_no_jumps():
    d = np.linspace(0.4, 25, 5000)
    r = _map(d, 0.9626)
    ratios = np.interp(r, OD, np.linspace(0, 1, 11))
    # dmap steps between adjacent depth samples stay modest (no wrap/overshoot)
    assert np.abs(np.diff(ratios)).max() < 0.15  # steep near-field slope is legitimate; wrap/overshoot would jump ~1


# ---------------- auto_gain ----------------

def test_auto_gain_nan_inf_sentinel_pixels():
    d = np.concatenate([np.full(500, np.nan), np.full(500, np.inf), np.full(500, 1000.0),
                        np.linspace(0.6, 3.0, 500)])
    g = auto_gain(d, 0.9626, z_focus=ZF, od=OD)
    assert np.isfinite(g) and 1.0 <= g <= 2.0


def test_auto_gain_all_nan_returns_lo_no_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert auto_gain(np.full((3, 3), np.nan), 1.0, z_focus=ZF, od=OD) == 1.0


def test_auto_gain_short_od_raises_clearly():
    with pytest.raises((IndexError, ValueError)):
        auto_gain(np.ones(4), 1.0, z_focus=ZF, od=OD[:2])


def test_auto_gain_custom_lo_hi_and_degenerate_range():
    d = np.linspace(0.6, 3.0, 100)
    assert auto_gain(d, 0.9626, z_focus=ZF, od=OD, lo=1.5, hi=1.5) == 1.5


def test_auto_gain_scale_invariant_to_nan_padding():
    d = np.linspace(0.6, 3.0, 1000)
    g0 = auto_gain(d, 0.9626, z_focus=ZF, od=OD)
    g1 = auto_gain(np.concatenate([d, np.full(300, np.nan)]), 0.9626, z_focus=ZF, od=OD)
    assert g0 == pytest.approx(g1)


def test_auto_gain_does_not_mutate_input():
    d = np.linspace(0.6, 3.0, 100)
    c = d.copy()
    auto_gain(d, 0.9626, z_focus=ZF, od=OD)
    assert np.array_equal(d, c)


def test_auto_gain_polarity_symmetric():
    """A scene with only near-side offset must give same gain as mirrored far-side."""
    a, dp = 1.0, np.linspace(0.05, 0.9, 400)
    near = 1 / (1 / a + dp)   # farther than anchor
    far = 1 / (1 / a - dp)    # nearer than anchor (positive for dp<1)
    assert auto_gain(near, a, z_focus=ZF, od=OD) == pytest.approx(auto_gain(far, a, z_focus=ZF, od=OD))


# ---------------- default path isolation / bit-identity guards ----------------

def test_protected_files_have_no_diopter_reference():
    # S1 (Run 5) plumbed diopter into convolve/api/_render_args on purpose;
    # the files that must stay diopter-free are the compositor and the bench pins.
    for rel in ("render/_layered_composite.py", "render/_bench.py"):
        txt = (ROOT / "summicron" / rel).read_text()
        assert "diopter" not in txt.lower(), rel


def _has_git_head() -> bool:
    # A freshly `git init`'d staged tree has .git but no HEAD.
    r = subprocess.run(["git", "rev-parse", "--verify", "-q", "HEAD"],
                       cwd=ROOT, capture_output=True, text=True)
    return r.returncode == 0


@pytest.mark.skipif(not _has_git_head(), reason="not a git checkout (staged release tree)")
def test_protected_files_unmodified_vs_head():
    r = subprocess.run(
        ["git", "diff", "--name-only", "HEAD", "--", "summicron/render/_layered_composite.py", "summicron/depth", "summicron/io", "summicron/_artifacts"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert r.returncode == 0
    assert r.stdout.strip() == "", r.stdout


def test_bench_pins_unchanged():
    from summicron.render import _bench
    # 0.5.0: the old-look (*_OLD) pins were removed with the old tensor.
    assert not hasattr(_bench, "G1_MD5_OLD") and not hasattr(_bench, "ON_MD5_OLD")
    for name in ("G1_MD5", "ON_MD5", "LENS50_MD5", "TENSOR_MD5", "TENSOR_MD5_50"):
        v = getattr(_bench, name)
        assert len(v) == 32 and all(c in "0123456789abcdef" for c in v), name


def test_importing_render_api_does_not_load_diopter():
    code = ("import sys, summicron.render.api, summicron.render.convolve, summicron.cli;"
            "sys.exit(1 if 'summicron.depth.diopter' in sys.modules else 0)")
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True,
                       env={**__import__("os").environ, "PYTHONPATH": str(ROOT)})
    assert r.returncode == 0, r.stderr


def test_diopter_module_imports_only_numpy():
    import ast
    tree = ast.parse((ROOT / "summicron/depth/diopter.py").read_text())
    mods = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            mods += [a.name for a in n.names]
        elif isinstance(n, ast.ImportFrom):
            mods.append(n.module)
    assert set(mods) <= {"numpy", "__future__"}, mods


def test_module_import_has_no_side_effects_on_modules():
    code = ("import sys; before=set(sys.modules); import summicron.depth.diopter;"
            "new={m for m in set(sys.modules)-before if m.startswith('summicron.render') or m.startswith('scipy') or m.startswith('PIL')};"
            "sys.exit(1 if new else 0)")
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True,
                       env={**__import__("os").environ, "PYTHONPATH": str(ROOT)})
    assert r.returncode == 0, r.stdout + r.stderr
