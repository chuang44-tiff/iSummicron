"""Adversarial probes for R-native-mem (doc/plan/phase2/section17_native-memory/
locked-spec-mem.md rev 2).

Targets: the ``_memory_guard`` / ``MemoryBudgetError`` contract (env edge
cases, suggestion re-plug, render()/CLI wiring), the structural estimator
PER PHASE against tracemalloc on shapes outside the coder's grid (D=1, D=22,
batch=D, odd / thin / tiny sizes, bright pass active, Nr=24,
k > image/2), the slab-batch default, and the HDR-loader buffer lifetime.
All renders are small synthetics; the only HEIC touches are a decode + LiDAR
read at --downres 256 (no native render).
"""
from __future__ import annotations

import re
import sys
import tracemalloc
import warnings
from pathlib import Path

import numpy as np

import pytest

import summicron.render.convolve as cv
from summicron.render.convolve import (
    PLATFORM_BASE,
    MemoryBudgetError,
    _estimate_peak_breakdown,
    _estimate_peak_bytes,
    _memory_guard,
)

_REPO = Path(__file__).resolve().parents[2]
_HEIC = _REPO / "iphoneImage" / "IMG_6915.HEIC"
_OD11 = np.array([0.7, 0.98, 1.37, 1.91, 2.68, 3.74, 5.23, 7.32, 10.23, 14.30, 20.0])


def _tensor(d: int, k: int = 31, n_r: int = 12, corpus_ref=None) -> dict:
    od = (_OD11 if d <= 11 else np.geomspace(0.7, 20.0, d))[:d].astype(np.float64)
    kern = np.zeros((n_r, d, k, k), np.float32)
    c = k // 2
    kern[:, 0, c, c] = 1.0
    ys, xs = np.mgrid[-c:c + 1, -c:c + 1].astype(np.float32)
    for di in range(1, d):
        g = np.exp(-(xs ** 2 + ys ** 2) / (2 * (1.0 + 0.5 * di) ** 2))
        kern[:, di] = g / g.sum()
    return {
        "kernels": kern,
        "radial_edges": np.sqrt(np.linspace(0.0, 1.0, n_r + 1)),
        "defocus_ratios": np.linspace(0.0, 1.0, d),
        "object_distances_m": od,
        "focus_distance_m": 0.7,
        "corpus_ref_long_edge": corpus_ref,
        "otf_norm_applied": False,
        "delta_guard_applied": True,
    }


def _scene(h, w, d, hdr=False):
    rng = np.random.default_rng(0)
    img = rng.uniform(0.0, 0.8, (h, w, 3)).astype(np.float32)
    if hdr:
        img[np.random.default_rng(1).random((h, w)) < 0.05] = 4.0
    od = _tensor(d)["object_distances_m"]
    ramp = np.geomspace(od[0], od[-1], w, dtype=np.float32) if d > 1 else np.full(w, 0.7, np.float32)
    depth = np.broadcast_to(ramp[None, :], (h, w)).copy()
    depth[h // 3: 2 * h // 3, w // 3: 2 * w // 3] = np.float32(od[min(2, d - 1)])
    return img, depth


# ---------------------------------------------------------------------------
# (1) budget env edge cases
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", ["1e9", "12GiB", "12G", "0x10", "nan", "inf",
                                 "-0", " ", "9" * 5000, "1.0"])
def test_env_rejects_non_integer_byte_strings(monkeypatch, bad):
    monkeypatch.setenv("ISUMMICRON_MAX_BYTES", bad)
    with pytest.raises(ValueError, match="ISUMMICRON_MAX_BYTES"):
        _memory_guard(256, 256, 11)


@pytest.mark.parametrize("ok, val", [(" 12884901888 ", 12884901888),
                                      ("+12884901888", 12884901888),
                                      ("12_884_901_888", 12884901888)])
def test_env_lenient_int_forms_accepted(monkeypatch, ok, val):
    """Python int() leniency (whitespace, sign, underscores) is accepted."""
    monkeypatch.setenv("ISUMMICRON_MAX_BYTES", ok)
    assert cv._budget_bytes() == val


def test_env_huge_budget_admits_everything_under_pixel_gate(monkeypatch):
    monkeypatch.setenv("ISUMMICRON_MAX_BYTES", "9" * 40)
    _memory_guard(5712, 4284, 22, occlusion=True, slab_batch_size=22)
    _memory_guard(6000, 6000, 22, occlusion=True, slab_batch_size=22)


def test_48mp_hits_pixel_gate_with_numeric_suggestion():
    """Documents: 48 MP (8064x6048) is stopped by the MAX_PIXELS fast gate,
    a plain RuntimeError whose hint names a fitted numeric --downres."""
    for h, w in ((6048, 8064), (8064, 6048)):
        with pytest.raises(RuntimeError) as exc:
            _memory_guard(h, w, 11, occlusion=True)
        assert not isinstance(exc.value, MemoryBudgetError)
        assert "Image too large" in str(exc.value)
        assert "--downres N" not in str(exc.value)
        assert re.search(r"--downres \d+", str(exc.value))


# ---------------------------------------------------------------------------
# (1) suggestion re-plug: fits, maximal, and also fits with the real
#     (shrunk) scaled_ksize at that downres
# ---------------------------------------------------------------------------

def _downres_dims(h, w, le):
    long_edge = max(h, w)
    if long_edge <= le:
        return h, w
    s = le / long_edge
    return max(1, int(round(h * s))), max(1, int(round(w * s)))


def _scaled_k(h, w, kh=183, ref=5712):
    ps = max(h, w) / ref
    return max(3, int(round(kh * ps)) | 1) if ps < 0.95 else kh


_SUG_CASES = [
    (4284, 5712, 11, False, None), (5712, 4284, 11, True, None),
    (4031, 3023, 22, True, 1), (3023, 4031, 22, False, 22),
    (6000, 1000, 11, True, None), (1000, 6000, 1, False, None),
    (5712, 4284, 11, True, 11), (2048, 2048, 2, True, 2),
]


@pytest.mark.parametrize("h, w, d, occ, batch", _SUG_CASES)
@pytest.mark.parametrize("frac", [0.35, 0.6, 0.9])
def test_suggested_downres_fits_and_is_maximal(monkeypatch, h, w, d, occ, batch, frac):
    k = _scaled_k(h, w)
    kw = dict(occlusion=occ, kernel_size=k, n_radial=12, slab_batch_size=batch)
    est = _estimate_peak_bytes(h, w, d, **kw)
    budget = max(PLATFORM_BASE + 1, int(PLATFORM_BASE + frac * (est - PLATFORM_BASE)))
    monkeypatch.setenv("ISUMMICRON_MAX_BYTES", str(budget))
    with pytest.raises(MemoryBudgetError) as exc:
        _memory_guard(h, w, d, **kw)
    err = exc.value
    assert isinstance(err, RuntimeError)
    assert err.estimated_bytes == est and err.budget_bytes == budget
    sug = err.suggested_downres
    if sug is None:
        hh, ww = _downres_dims(h, w, 256)
        assert _estimate_peak_bytes(hh, ww, d, **kw) > budget
        return
    assert f"--downres {sug}" in str(err)
    assert sug % 64 == 0 and 256 <= sug < max(h, w)
    hh, ww = _downres_dims(h, w, sug)  # exactly what downres_array produces
    assert _estimate_peak_bytes(hh, ww, d, **kw) <= budget
    # realistic: at that downres render() shrinks the kernel -> still fits
    kw_real = dict(kw, kernel_size=_scaled_k(hh, ww))
    assert _estimate_peak_bytes(hh, ww, d, **kw_real) <= budget
    _memory_guard(hh, ww, d, **kw_real)  # and the guard itself admits it
    if sug + 64 < max(h, w):
        h2, w2 = _downres_dims(h, w, sug + 64)
        assert _estimate_peak_bytes(h2, w2, d, **kw) > budget


# ---------------------------------------------------------------------------
# (1) render() wiring: scaled_ksize / n_radial / batch reach the guard, the
#     boundary is exact, and rejection precedes the big allocations
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("occ, batch", [("off", None), ("on", 3)])
def test_render_guard_wiring_and_exact_boundary(monkeypatch, occ, batch):
    h, w, d, nr = 120, 160, 11, 6
    tensor = _tensor(d, k=61, n_r=nr, corpus_ref=320)  # pix_scale 0.5 -> k=31
    k_exp = max(3, int(round(61 * 160 / 320)) | 1)
    assert k_exp == 31
    est = _estimate_peak_bytes(h, w, d, occlusion=(occ == "on"), kernel_size=k_exp,
                               n_radial=nr, slab_batch_size=batch)
    img, depth = _scene(h, w, d)
    monkeypatch.setenv("ISUMMICRON_MAX_BYTES", str(est - 1))
    P = h * w * 4
    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        b0 = tracemalloc.get_traced_memory()[0]
        with pytest.raises(MemoryBudgetError) as exc:
            cv.render(img, tensor, depth_m=depth, occlusion_compositing=occ,
                      slab_batch_size=batch)
        peak = tracemalloc.get_traced_memory()[1] - b0
    finally:
        tracemalloc.stop()
    assert exc.value.estimated_bytes == est
    # nothing padded / no spectra / no slab_w before the guard
    assert peak < 12 * P + (1 << 20), f"guard fired late: {peak} B traced"
    monkeypatch.setenv("ISUMMICRON_MAX_BYTES", str(est))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cv.render(img, tensor, depth_m=depth, occlusion_compositing=occ,
                  slab_batch_size=batch)


# ---------------------------------------------------------------------------
# (2) estimator PER PHASE vs tracemalloc (outside the coder's 36-row grid)
# ---------------------------------------------------------------------------

def _phase_measure(h, w, d, occ, batch, k=31, nr=12, hdr=False):
    img, depth = _scene(h, w, d, hdr=hdr)
    tensor = _tensor(d, k=k, n_r=nr)
    orig = cv._layered_gather_diffuse
    rec: dict = {}

    def wrap(*a, **kw):
        rec["prep"] = tracemalloc.get_traced_memory()[1] - rec["b"]
        tracemalloc.reset_peak()
        r = orig(*a, **kw)
        rec["gather"] = tracemalloc.get_traced_memory()[1] - rec["b"]
        tracemalloc.reset_peak()
        return r

    cv._layered_gather_diffuse = wrap
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tracemalloc.start()
            tracemalloc.reset_peak()
            rec["b"] = tracemalloc.get_traced_memory()[0]
            cv.render(img, tensor, depth_m=depth, hdr_source="macos",
                      bright_thresh=1.0 if hdr else 1e9,
                      occlusion_compositing="on" if occ else "off",
                      slab_batch_size=batch)
            rec["post"] = tracemalloc.get_traced_memory()[1] - rec["b"]
            tracemalloc.stop()
    finally:
        cv._layered_gather_diffuse = orig
        if tracemalloc.is_tracing():
            tracemalloc.stop()
    est = _estimate_peak_breakdown(h, w, d, occlusion=occ, kernel_size=k,
                                   n_radial=nr, slab_batch_size=batch)
    return rec, est


@pytest.fixture(scope="module")
def _warm():
    """First render in a process carries ~2 MB of one-off lazy-import /
    cache allocations (not per-pixel; PLATFORM_BASE covers them)."""
    _phase_measure(64, 48, 3, True, 1)


_PHASE_CASES = [
    # h, w, d, occ, batch, k, nr, hdr
    (256, 192, 1, True, 1, 31, 12, False),
    (256, 192, 22, True, 1, 31, 12, False),
    (256, 192, 22, True, 22, 31, 12, False),
    (192, 256, 2, True, 2, 31, 12, False),
    (97, 131, 11, True, 1, 31, 12, False),
    (48, 300, 11, True, 1, 31, 12, False),
    (300, 48, 11, False, 4, 31, 12, False),
    (64, 48, 11, True, 1, 65, 12, False),
    (256, 192, 11, True, 11, 31, 12, True),
    (256, 192, 11, True, 1, 31, 12, True),
    (256, 192, 11, False, 1, 31, 24, False),
    (384, 512, 11, True, 1, 31, 12, True),
]


@pytest.mark.parametrize("h, w, d, occ, batch, k, nr, hdr", _PHASE_CASES)
def test_estimator_never_under_per_phase(_warm, h, w, d, occ, batch, k, nr, hdr):
    rec, est = _phase_measure(h, w, d, occ, batch, k=k, nr=nr, hdr=hdr)
    slack = 256 * 1024
    tag = f"{h}x{w} D={d} b={batch} occ={occ} k={k} nr={nr} hdr={hdr}"
    assert est["prep"] + slack >= rec["prep"], f"prep under: {tag} {rec} {est}"
    assert est["gather"] + slack >= rec["gather"], f"gather under: {tag} {rec} {est}"
    post_est = max(est["over"], est["gather"])  # render tail is in ``gather``
    assert post_est + slack >= rec["post"], f"over/tail under: {tag} {rec} {est}"


# ---------------------------------------------------------------------------
# (3) slab batch default 1 everywhere
# ---------------------------------------------------------------------------

def test_batch_default_is_1_on_every_surface():
    from summicron.render import _render_args
    from summicron.render.api import RenderOptions
    import argparse
    assert cv._SLAB_BATCH_SIZE == 1
    p = _render_args.add_render_arguments(argparse.ArgumentParser())
    act = next(a for a in p._actions if "--slab-batch-size" in a.option_strings)
    assert act.default is None
    assert "None → 1" in act.help and "4" not in act.help
    assert RenderOptions(image_path=Path("x.heic")).slab_batch_size is None
    # estimator default == explicit 1
    assert _estimate_peak_bytes(512, 384, 11) == _estimate_peak_bytes(512, 384, 11, slab_batch_size=1)


# ---------------------------------------------------------------------------
# (5) spectra drop: ON render still fills stats; caller-owned fft_state is
#     only stripped of the 3 spectra keys
# ---------------------------------------------------------------------------

def test_on_render_stats_survive_spectra_drop():
    img, depth = _scene(128, 96, 11)
    stats: dict = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        u8_on, lin_on = cv.render(img, _tensor(11), depth_m=depth,
                                  occlusion_compositing="on", stats=stats)
        u8_off, _ = cv.render(img, _tensor(11), depth_m=depth)
    assert stats, "stats sink empty on the ON path"
    assert np.isfinite(lin_on).all()
    assert u8_on.shape == u8_off.shape


# ---------------------------------------------------------------------------
# (4) HDR loader: returned array owns its data (no view onto the bytearray)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(sys.platform != "darwin" or not _HEIC.exists(),
                    reason="macOS + iphoneImage HEIC required")
def test_hdr_loader_array_owns_memory():
    pytest.importorskip("Quartz")
    from summicron.io.hdr_load_macos import load_hdr_heic
    a = load_hdr_heic(_HEIC)
    assert a.dtype == np.float32 and a.ndim == 3 and a.shape[2] == 3
    assert a.flags["WRITEABLE"]
    b = a
    while b is not None:
        assert not isinstance(b, (bytes, bytearray, memoryview)), "view onto CI buffer"
        b = getattr(b, "base", None)
    assert float(a.min()) >= 0.0


# ---------------------------------------------------------------------------
# (1) CLI end to end: real HEIC decode + LiDAR at --downres 256, budget
#     tripped in render() -> exit 2, message on stderr, nothing written
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _HEIC.exists(), reason="iphoneImage HEIC required")
def test_cli_budget_error_exit_2(monkeypatch, tmp_path, capsys):
    pytest.importorskip("pillow_heif")
    from summicron import cli
    monkeypatch.setenv("ISUMMICRON_MAX_BYTES", str(PLATFORM_BASE + 1))
    rc = cli.main(["render", "--image", str(_HEIC), "--downres", "256",
                   "--out-dir", str(tmp_path)])
    err = capsys.readouterr().err
    assert rc == 2
    assert "render: Render would exceed" in err and "--downres" in err
    assert "Traceback" not in err
    assert not any(tmp_path.iterdir())
