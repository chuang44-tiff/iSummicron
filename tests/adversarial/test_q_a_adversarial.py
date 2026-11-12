"""Adversarial probes for Q-a R-eval-gates (``summicron.render._quality``).

FAST, synthetic arrays only. Post-mortem 1 wrong-polarity rule: every gate is
checked to FIRE on its bad case and to stay quiet on its good case.
"""
from __future__ import annotations

import dataclasses
import inspect

import numpy as np
import pytest
from scipy import ndimage

pytest.importorskip("skimage")

from summicron.render import _quality as Q  # noqa: E402

N = 160


# ---------------------------------------------------------------------------
# Synthetic scene
# ---------------------------------------------------------------------------


def _scene(seed: int = 0):
    """ref: blobby piecewise-constant texture; alpha: radial ramp (1 centre ->
    0 outside, wide band); base: ref blurred outside the snap."""
    rng = np.random.default_rng(seed)
    f = ndimage.gaussian_filter(rng.standard_normal((N, N)), 3.0)
    f = f / f.std()
    ref1 = 0.25 + 0.15 * (f > 0) + 0.15 * (f > 1.0) + 0.01 * rng.standard_normal((N, N))
    ref = np.repeat(ref1[..., None], 3, axis=2).astype(np.float32)
    ref[..., 0] *= 1.1
    ref[..., 2] *= 0.9
    yy, xx = np.mgrid[:N, :N]
    r = np.hypot(yy - N / 2, xx - N / 2)
    alpha = (1.0 - np.clip((r - 15.0) / 45.0, 0.0, 1.0)).astype(np.float32)
    blurred = _blur(ref, 2.0)
    base = (alpha[..., None] * ref + (1 - alpha[..., None]) * blurred).astype(np.float32)
    return ref, alpha, base


def _blur(img, sigma, chans=(0, 1, 2)):
    out = img.copy()
    for c in chans:
        out[..., c] = ndimage.gaussian_filter(img[..., c], sigma)
    return out


def _win():
    return Q.focus_window((N, N), (0.5, 0.5), side_2048=2048)


def _band(alpha):
    return (alpha > Q.ALPHA_LO) & (alpha < Q.ALPHA_HI)


def _pair(base, var, ref, alpha, lever):
    mb = Q.window_metrics(base, ref, alpha, _win())
    mv = Q.window_metrics(var, ref, alpha, _win(), mask_out=base)
    return Q.paired(mb, mv, lever)


# ---------------------------------------------------------------------------
# Wrong-polarity: each gate fires on bad, stays quiet on good
# ---------------------------------------------------------------------------


def test_identity_all_deltas_exactly_zero():
    ref, alpha, base = _scene()
    for lever in ("band", "nonband"):
        p = _pair(base, base.copy(), ref, alpha, lever)
        assert p["valid"]
        for k in ("d_ger", "d_ssim", "d_ovfrac", "d_bg", "d_ch"):
            assert p[k] == 0.0, (lever, k, p[k])
        assert p["gates"]["OV"] and p["gates"]["BG"] and not p["ch_flag"]
        assert p["gates"]["CEIL"]
    # Identity is NOT an improvement: the band lever's strict '>' must not pass.
    assert not _pair(base, base.copy(), ref, alpha, "band")["gates"]["Q1-B"]
    assert _pair(base, base.copy(), ref, alpha, "nonband")["gates"]["Q1-N"]


def test_band_blur_fires_q1_but_not_bg():
    ref, alpha, base = _scene()
    band = _band(alpha)
    bad = np.where(band[..., None], _blur(base, 1.0), base).astype(np.float32)
    p = _pair(base, bad, ref, alpha, "band")
    assert p["d_ger"] < 0 and p["d_ssim"] < 0
    assert not p["gates"]["Q1-B"]
    assert not _pair(base, bad, ref, alpha, "nonband")["gates"]["Q1-N"]
    # Band-only change: bg stencil is eroded away from the band -> exactly 0.
    assert p["d_bg"] == 0.0 and p["gates"]["BG"]
    # Good polarity: pull band toward ref -> both metrics rise, Q1-B passes.
    good = np.where(band[..., None], base + 0.5 * (ref - base), base).astype(np.float32)
    g = _pair(base, good, ref, alpha, "band")
    assert g["d_ger"] > 0 and g["d_ssim"] > 0 and g["gates"]["Q1-B"]
    assert g["gates"]["OV"] and not g["ch_flag"] and g["gates"]["BG"]


def test_unsharp_ringing_fires_ov_and_ceil_blur_does_not():
    ref, alpha, base = _scene()
    band = _band(alpha)
    usm = base + 3.0 * (base - _blur(base, 1.5))
    bad = np.where(band[..., None], usm, base).astype(np.float32)
    p = _pair(base, bad, ref, alpha, "band")
    assert p["d_ovfrac"] > Q.GATE_OV and not p["gates"]["OV"]
    assert p["ger_var"] > Q.GATE_CEIL and not p["gates"]["CEIL"]
    # Gaussian blur must NOT trip OV / CEIL (it is a different failure).
    blur = np.where(band[..., None], _blur(base, 1.0), base).astype(np.float32)
    q = _pair(base, blur, ref, alpha, "band")
    assert q["gates"]["OV"] and q["gates"]["CEIL"]


def test_background_only_change_fires_bg_gate_only():
    ref, alpha, base = _scene()
    bg = alpha <= Q.ALPHA_LO
    bad = np.where(bg[..., None], base + 0.5 * (ref - base), base).astype(np.float32)
    p = _pair(base, bad, ref, alpha, "nonband")
    assert p["d_bg"] > Q.GATE_BG and not p["gates"]["BG"]
    # Band metrics barely move (only the band-edge stencil reads bg pixels).
    assert p["gates"]["Q1-N"] and not p["ch_flag"] and p["gates"]["OV"]
    # Sub-threshold bg change stays quiet.
    tiny = np.where(bg[..., None], base + 0.002 * (ref - base), base).astype(np.float32)
    assert _pair(base, tiny, ref, alpha, "nonband")["gates"]["BG"]


def test_per_channel_tint_in_band_fires_ch_neutral_gain_does_not():
    ref, alpha, base = _scene()
    band = _band(alpha)
    tint = base.copy()
    tint[..., 0] = np.where(band, base[..., 0] * 1.3, base[..., 0])
    p = _pair(base, tint, ref, alpha, "nonband")
    assert p["d_ch"] > Q.GATE_CH and p["ch_flag"]
    neutral = np.where(band[..., None], base * 1.1, base).astype(np.float32)
    q = _pair(base, neutral, ref, alpha, "nonband")
    assert q["d_ch"] <= Q.GATE_CH and not q["ch_flag"]


def test_ch_is_report_only_not_a_gate():
    rng = np.random.default_rng(0)
    chans = []
    for s in (1.0, 3.0, 6.0):  # decorrelated, different spectrum per channel
        g = ndimage.gaussian_filter(rng.standard_normal((N, N)), s)
        chans.append(0.3 + 0.15 * np.tanh(2 * g / g.std()))
    ref = np.stack(chans, axis=2).astype(np.float32)
    _, alpha, _ = _scene()
    base = (alpha[..., None] * ref + (1 - alpha[..., None]) * _blur(ref, 2.0)).astype(np.float32)
    var = np.where(_band(alpha)[..., None], base + 0.25 * (ref - base), base).astype(np.float32)
    p = _pair(base, var, ref, alpha, "band")
    assert p["gates"]["Q1-B"]  # it IS a genuine (neutral) improvement
    # CH cannot separate neutral levers from tints (ruling): reported, never gated.
    assert "CH" not in p["gates"]
    assert p["d_ch"] > Q.GATE_CH and p["ch_flag"]
    assert all(p["gates"].values()), p["gates"]


def test_bg_gate_fails_when_bg_invalid():
    """Full-frame BG with no measurable bg pixels -> FAIL (invalid), not PASS."""
    ref, alpha, base = _scene()
    alpha_nobg = np.maximum(alpha, np.float32(0.5))  # no alpha<=0.02 anywhere
    bad = (base + 0.5 * (ref - base)).astype(np.float32)
    r = Q.bg_fullframe(base, bad, ref, alpha_nobg)
    assert r["valid"] is False and r["gate"] is False and r["d_bg"] is None
    # footprint covering all of the bg also empties it -> FAIL
    r2 = Q.bg_fullframe(base, bad, ref, alpha, np.ones(alpha.shape, bool))
    assert r2["valid"] is False and r2["gate"] is False
    # windowed bg_ger None is still surfaced by paired() as invalid (informational)
    mb = Q.window_metrics(base, ref, alpha_nobg, _win())
    mv = Q.window_metrics(bad, ref, alpha_nobg, _win(), mask_out=base)
    p = Q.paired(mb, mv, "nonband")
    assert p["bg_invalid"] is True


def test_bg_fullframe_detects_bg_change_and_footprint_exclusion():
    ref, alpha, base = _scene()
    bg = alpha <= 0.02
    var = np.where(bg[..., None], base + 0.5 * (ref - base), base).astype(np.float32)
    assert Q.bg_fullframe(base, var, ref, alpha)["gate"] is False
    assert Q.bg_fullframe(base, base, ref, alpha)["gate"] is True


# ---------------------------------------------------------------------------
# Float32 alpha trap + exact boundaries
# ---------------------------------------------------------------------------


def test_float32_cos2_alpha_trap_and_exact_boundaries():
    d_knee = np.float32(0.10)
    dmap = np.full((40, 40), 0.5, np.float32)  # all beyond knee
    alpha = (np.cos(0.5 * np.pi * np.minimum(dmap / d_knee, 1.0)) ** 2).astype(np.float32)
    assert (alpha > 0).all() and alpha.max() < 1e-12  # the trap is real
    y = np.full((40, 40), 0.3)
    assert not Q.band_mask(alpha, y, y, border=0).any()
    assert Q.bg_mask(alpha, y, y, border=0)[1:-1, 1:-1].all()

    for dt in (np.float32, np.float64):
        a = np.full((20, 20), 0.5, dt)
        a[5, 5] = dt(Q.ALPHA_LO)
        a[6, 6] = dt(Q.ALPHA_HI)
        a[7, 7] = np.nextafter(dt(Q.ALPHA_LO), dt(1))
        a[8, 8] = np.nextafter(dt(Q.ALPHA_HI), dt(0))
        y = np.full((20, 20), 0.3)
        m = Q.band_mask(a, y, y, border=0)
        assert not m[5, 5] and not m[6, 6], dt
        assert m[7, 7] and m[8, 8], dt


# ---------------------------------------------------------------------------
# Bloom exclusion
# ---------------------------------------------------------------------------


def test_bloom_dilation_radius_and_all_hdr_window():
    a = np.full((41, 41), 0.5, np.float32)
    ref_y = np.full((41, 41), 0.3)
    ref_y[20, 20] = 4.0
    m = Q.band_mask(a, ref_y, ref_y, border=0)
    assert not m[20, 20] and not m[20, 23] and not m[17, 20]  # within 3 px
    assert m[20, 24] and m[24, 20]  # 4 px out survives
    # out>1 alone also excluded.
    out_y = np.full((41, 41), 0.3)
    out_y[5, 5] = 1.5
    assert not Q.band_mask(a, np.full((41, 41), 0.3), out_y, border=0)[5, 5]

    # All-HDR window: every metric None, paired invalid, no exception.
    ref, alpha, base = _scene()
    hdr_ref = ref + 2.0
    hdr_out = base + 2.0
    m = Q.window_metrics(hdr_out, hdr_ref, alpha, _win())
    assert m["n_px"] == 0 and m["ger_band"] is None and m["bg_ger"] is None
    assert Q.paired(m, m, "band") == {"valid": False}
    ff = Q.focus_fidelity(hdr_out, hdr_ref, alpha, (0.5, 0.5))
    assert ff is not None and ff["n_px"] == 0 and ff["ssim_band"] is None


# ---------------------------------------------------------------------------
# Degenerate windows
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fxy", [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0),
                                 (-3.0, 7.0)])
@pytest.mark.parametrize("off", [(0, 0), (500, -500)])
def test_corner_and_offframe_windows_are_defined(fxy, off):
    ref, alpha, base = _scene()
    shape = (N, 2 * N, 3)
    ref2 = np.concatenate([ref, ref], axis=1)
    a2 = np.concatenate([alpha, alpha], axis=1)
    b2 = np.concatenate([base, base], axis=1)
    win = Q.focus_window(shape, fxy, side_2048=1024, offset_px=off)
    ys, xs = win
    assert 0 <= ys.start < ys.stop <= N and 0 <= xs.start < xs.stop <= 2 * N
    assert ys.stop - ys.start == xs.stop - xs.start
    m = Q.window_metrics(b2, ref2, a2, win)
    assert m["n_px"] >= 0
    p = Q.paired(m, m, "nonband")
    assert p == {"valid": False} or p["d_ger"] == 0.0
    Q.focus_fidelity(b2, ref2, a2, (min(max(fxy[0], 0), 1), min(max(fxy[1], 0), 1)))


@pytest.mark.parametrize("fill", [0.0, 1.0])
def test_empty_band_returns_none_not_crash(fill):
    ref, _, base = _scene()
    alpha = np.full((N, N), fill, np.float32)
    m = Q.window_metrics(base, ref, alpha, _win())
    assert m["n_px"] == 0
    assert all(m[k] is None for k in ("ger_band", "ssim_band", "ovfrac", "ger_rgb"))
    assert Q.paired(m, m, "band") == {"valid": False}
    ff = Q.focus_fidelity(base, ref, alpha, (0.5, 0.5))
    assert ff is not None and ff["n_px"] == 0


def test_tiny_image_focus_fidelity_returns_none():
    ref, alpha, base = _scene()
    assert Q.focus_fidelity(base[:5, :5], ref[:5, :5], alpha[:5, :5], (0.5, 0.5)) is None


def test_tiny_window_at_origin_mask_stays_inside_window():
    rr = np.random.default_rng(1).uniform(0.1, 0.9, (200, 200, 3)).astype(np.float32)
    a = np.full((200, 200), 0.5, np.float32)
    win = Q.focus_window((200, 200), (0.0, 0.0), side_2048=10)
    assert win == (slice(0, 2), slice(0, 2))
    m = Q.window_metrics(rr * np.float32(0.9), rr, a, win)
    # 2x2 window minus a 4-px border trim -> nothing can survive.
    assert m["n_px"] == 0


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_metrics_bit_deterministic():
    ref, alpha, base = _scene(seed=3)
    var = np.where(_band(alpha)[..., None], _blur(base, 0.7), base).astype(np.float32)
    runs = []
    for _ in range(3):
        mb = Q.window_metrics(base.copy(), ref.copy(), alpha.copy(), _win())
        mv = Q.window_metrics(var.copy(), ref.copy(), alpha.copy(), _win(), mask_out=base)
        runs.append((mb, mv, Q.paired(mb, mv, "band"),
                     Q.focus_fidelity(var, ref, alpha, (0.5, 0.5))))
    assert runs[0] == runs[1] == runs[2]


# ---------------------------------------------------------------------------
# Pipeline wiring: private _eval_sink, diagnostics contract
# ---------------------------------------------------------------------------

_OD = np.array([0.70, 0.98, 1.37, 1.91, 2.68, 3.74, 5.23, 7.32, 10.23, 14.30, 20.0])


def _tensor():
    R, D, k = 4, 11, 17
    kern = np.zeros((R, D, k, k), dtype=np.float32)
    c = k // 2
    kern[:, 0, c, c] = 1.0
    ys, xs = np.mgrid[-c:c + 1, -c:c + 1].astype(np.float32)
    for di in range(1, D):
        g = np.exp(-(xs ** 2 + ys ** 2) / (2 * (0.8 + 0.6 * di) ** 2))
        kern[:, di] = g / g.sum()
    return {
        "kernels": kern,
        "radial_edges": np.array([0.0, 0.3, 0.6, 0.85, 1.01]),
        "defocus_ratios": np.linspace(0, 1, D),
        "object_distances_m": _OD.copy(),
        "focus_distance_m": 0.7,
        "corpus_ref_long_edge": None,
        "otf_norm_applied": False,
        "delta_guard_applied": True,
    }


@pytest.fixture
def fake_pipeline(monkeypatch):
    from summicron.render import convolve as conv
    import summicron.depth.estimate as est

    h = w = 256  # focus window side = 32 px -> valid band metrics
    rng = np.random.default_rng(7)
    img = rng.uniform(0, 0.8, (h, w, 3)).astype(np.float32)
    yy, xx = np.mgrid[:h, :w]
    depth = (0.7 + 0.004 * np.hypot(yy - h / 2, xx - w / 2)).astype(np.float32)
    monkeypatch.setattr(conv, "load_image", lambda p: img.copy())
    monkeypatch.setattr(conv, "load_tensor", lambda p: _tensor())
    monkeypatch.setattr(est, "extract_lidar_depth",
                        lambda p, target_hw=None, **kw: depth.copy())
    return img


def _opts(tmp_path, **kw):
    from summicron.render.api import RenderOptions
    return RenderOptions(image_path=tmp_path / "img.heic",
                         tensor_path=tmp_path / "t.npz", save=False,
                         return_linear=True, **kw)


@pytest.mark.filterwarnings("ignore:Center SSIM:UserWarning")
def test_eval_sink_none_vs_dict_identical_result(fake_pipeline, tmp_path):
    from summicron.render.api import RenderDiagnostics, render_image

    opts = _opts(tmp_path, focus_xy=(0.5, 0.5))
    r0 = render_image(opts, on_progress=lambda m: None)
    sink: dict = {"keep": 1}
    r1 = render_image(opts, on_progress=lambda m: None, _eval_sink=sink)
    assert np.array_equal(r0.output_u8, r1.output_u8)
    assert np.array_equal(r0.output_linear, r1.output_linear)
    assert r0.diagnostics.focus_fidelity == r1.diagnostics.focus_fidelity
    assert set(sink) == {"keep", "alpha", "dmap", "ref_lin", "defocus_ratios", "bright_thresh",
                       "knee_ratio", "dp"}  # S22 S1 additions
    # Q1-I on the exported arrays: alpha==1 pixels are the reference exactly.
    a1 = sink["alpha"] == 1.0
    assert a1.any()
    assert float(np.abs(r1.output_linear - sink["ref_lin"])[a1].max()) == 0.0
    # Diagnostics: focus_fidelity then resolved (Run 6) are the last fields.
    names = [f.name for f in dataclasses.fields(RenderDiagnostics)]
    assert names[-2:] == ["focus_fidelity", "resolved"]
    ff = r1.diagnostics.focus_fidelity
    assert ff is not None and ff["n_px"] >= Q.MIN_PX and ff["ssim_band"] is not None
    assert r1.diagnostics.center_ssim == ff["ssim_band"]
    for v in ff.values():
        assert v is None or isinstance(v, (int, float, tuple)), type(v)
    # No focus -> None (and center_ssim None).
    r2 = render_image(_opts(tmp_path), on_progress=lambda m: None)
    assert r2.diagnostics.focus_fidelity is None
    assert r2.diagnostics.center_ssim is None


def test_eval_sink_is_private_keyword_only_and_undocumented():
    from summicron.render.api import render_image

    sig = inspect.signature(render_image)
    p = sig.parameters["_eval_sink"]
    assert p.kind is inspect.Parameter.KEYWORD_ONLY and p.default is None
    doc = inspect.getdoc(render_image) or ""
    assert "_eval_sink" not in doc and "eval_sink" not in doc
