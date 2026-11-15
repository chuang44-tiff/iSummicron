"""S33 Tier A (exact) adversarial: try to break bit-identity of WP1 to WP3.

Oracle: ``tests/fixtures/s33_base_gather.py``, a VERBATIM frozen copy of the
base hot-path functions from ``git show 2a741a8:summicron/render/convolve.py``.
Every check compares the CURRENT ``summicron.render.convolve`` against that
frozen copy with ``tobytes()`` (so -0.0 vs +0.0 and NaN payloads count), never
against the in-tree default-argument path (which shares the staged FFTs).

Self-contained: synthetic images plus the two wheel-shipped tensors
(``summicron/_artifacts``); nothing under ``runs/``, no HEIC, no other checkout.

Covered: odd / tiny / portrait / landscape frames, both lenses, larger
kernels (``corpus_ref_long_edge`` override), the ON path, the field-edge
highlight chart, all-in-focus (delta only), no bright pixels, empty and
near-empty slabs, holes with and without a donor, ``slab_batch_size``
variants, the ``slab_support_skip`` x ``basis_hoist`` x ``_STAGED_FFT``
matrix (the non-staged bbox branch is what CI and every non-guarded build
run), repeat-run determinism, cancel at every poll point class, a
pre-set cancel with no FFT primitive called, NaN / out-of-range depth,
non-finite input, a > 256 KB spectrum case (numpy temporary elision only
kicks in above that size and silently turns ``F*K`` into ``K*F``), a random
geometry sweep of the staged FFT helpers, and the gather's traced peak.
"""
from __future__ import annotations

import functools
import re
import threading
import tracemalloc
import warnings

import numpy as np
import pytest
import scipy.fft as sp_fft

import summicron.render.convolve as cv
from summicron.render._artifacts import default_tensor_path
from tests.fixtures import s33_base_gather as base
from tests.fixtures.defocus import render_with_defocus

# Staged FFT modes this build may run: the guarded build runs both (the staged
# path is the default there; False is what CI / other builds run). Never force
# True on a build the WP2 proof does not cover.
_STAGED_MODES = [True, False] if cv._STAGED_FFT else [False]
_FLAG_MATRIX = [(True, True), (True, False), (False, True), (False, False)]
_SLOW = pytest.mark.slow


def _staged_params(fast_default_only: bool = True):
    """Staged modes; the build's default mode stays FAST, the other is slow."""
    return [pytest.param(m, marks=() if (m == cv._STAGED_FFT or not fast_default_only) else _SLOW)
            for m in _STAGED_MODES]


def _matrix_params():
    out = []
    for geom in (("35", 72, 53), ("50", 53, 72)):
        for flags in _FLAG_MATRIX:
            for st in _STAGED_MODES:
                fast = flags == (True, True) and (st == cv._STAGED_FFT or geom[0] == "35")
                out.append(pytest.param(geom, flags, st, marks=() if fast else _SLOW,
                                        id=f"{geom[0]}-{geom[1]}x{geom[2]}-skip{int(flags[0])}"
                                           f"-hoist{int(flags[1])}-staged{int(st)}"))
    return out


# --------------------------------------------------------------------- setup

@functools.lru_cache(maxsize=None)
def _shipped(lens: str) -> dict:
    return cv.load_tensor(default_tensor_path(lens))


def _tensor(lens: str, corpus_ref: float | None = None) -> dict:
    """Shipped tensor; ``corpus_ref`` shrinks the zoom so small frames get
    real (non-delta, multi-wedge) kernels instead of 3 px blobs."""
    t = dict(_shipped(lens))
    if corpus_ref is not None:
        t["corpus_ref_long_edge"] = float(corpus_ref)
    return t


def _synthetic_tensor(n_r: int, d: int, k: int = 21) -> dict:
    od = np.geomspace(0.7, 20.0, d) if d > 1 else np.array([0.7])
    kern = np.zeros((n_r, d, k, k), np.float32)
    c = k // 2
    kern[:, 0, c, c] = 1.0
    ys, xs = np.mgrid[-c:c + 1, -c:c + 1].astype(np.float32)
    for ri in range(n_r):
        for di in range(1, d):
            # anisotropic, off-centre lobes: rotation must matter per wedge
            g = np.exp(-((xs - 0.3 * ri) ** 2 / (2 * (1.0 + di) ** 2)
                         + ys ** 2 / (2 * (0.6 + 0.4 * di) ** 2)))
            kern[ri, di] = g / g.sum()
    return {
        "kernels": kern,
        "radial_edges": np.sqrt(np.linspace(0.0, 1.0, n_r + 1)),
        "defocus_ratios": np.linspace(0.0, 1.0, d),
        "object_distances_m": od.astype(np.float64),
        "corpus_ref_long_edge": None,
    }


_RNG_SEED = 20261112


def _image(h: int, w: int, kind: str = "noise", seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(_RNG_SEED + seed)
    if kind == "noise":
        return (rng.random((h, w, 3)) * 255).astype(np.uint8)
    if kind == "black":
        return np.zeros((h, w, 3), np.uint8)
    if kind == "dark":  # no pixel reaches any bright threshold used here
        return (rng.random((h, w, 3)) * 40).astype(np.uint8)
    raise ValueError(kind)


def _depth(h: int, w: int, kind: str) -> np.ndarray:
    xx = np.broadcast_to(np.arange(w)[None, :], (h, w))
    if kind == "ramp":  # step inside a ramp: slab supports end mid-frame
        d = np.broadcast_to(np.geomspace(0.7, 20.0, w, dtype=np.float32)[None, :], (h, w)).copy()
        d[h // 3: 2 * h // 3 + 1, w // 3: 2 * w // 3 + 1] = np.float32(1.37)
        return d
    if kind == "focus":  # all in focus: slab 0 only, delta kernels only
        return np.full((h, w), 0.7, np.float32)
    if kind == "two":  # two slabs; the other nine are empty
        return np.where(xx < max(1, w // 2), 0.7, 14.3).astype(np.float32)
    if kind == "wild":  # out of range both ways + a NaN pixel
        d = np.where(xx % 3 == 0, 0.05, 80.0).astype(np.float32)
        d[0, 0] = np.nan
        return d
    raise ValueError(kind)


def _render(img, tensor, depth, *, frozen: bool, **kw):
    """render() with the current gather, or with the frozen base gather
    substituted at the module attribute (``render`` and
    ``_layered_composite`` both look it up at call time)."""
    lines: list[str] = []
    stats: dict = {}
    orig = cv._layered_gather_diffuse
    if frozen:
        cv._layered_gather_diffuse = base._layered_gather_diffuse
    try:
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            try:
                u8, lin = cv.render(img, tensor, depth_m=depth,
                                    on_progress=lines.append, stats=stats, **kw)
                err = None
            except Exception as e:  # compared, not swallowed
                u8 = lin = None
                err = f"{type(e).__name__}: {e}"
    finally:
        cv._layered_gather_diffuse = orig
    return dict(u8=u8, lin=lin, err=err, lines=lines,
                warns=[f"{w.category.__name__}: {w.message}" for w in rec],
                acc=stats.get("acc_alpha"))


def _assert_render_same(img, tensor, depth, **kw):
    a = _render(img, tensor, depth, frozen=False, **kw)
    b = _render(img, tensor, depth, frozen=True, **kw)
    assert a["err"] == b["err"]
    assert a["lines"] == b["lines"]
    assert a["warns"] == b["warns"]
    if a["err"] is None:
        assert a["u8"].tobytes() == b["u8"].tobytes(), "u8 bytes differ"
        assert a["lin"].dtype == b["lin"].dtype
        assert a["lin"].tobytes() == b["lin"].tobytes(), "linear bytes differ"
        assert np.isfinite(a["lin"]).all()
        assert a["acc"].tobytes() == b["acc"].tobytes()
    return a


def _capture(img, tensor, depth, **kw) -> dict:
    """Gather kwargs exactly as render() builds them (OFF path)."""
    seen: dict = {}
    real = cv._layered_gather_diffuse

    def spy(**k):
        seen["kw"] = k
        return real(**k)

    cv._layered_gather_diffuse = spy
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cv.render(img, tensor, depth_m=depth, on_progress=lambda m: None, **kw)
    finally:
        cv._layered_gather_diffuse = real
    return dict(seen["kw"])


def _gather(fn, kw, **over):
    k = dict(kw)
    k.update(over)
    lines: list[str] = []
    k["on_progress"] = lines.append
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        try:
            out = fn(**k)
            err = None
        except Exception as e:
            out, err = None, f"{type(e).__name__}: {e}"
    return out, err, lines, [f"{w.category.__name__}: {w.message}" for w in rec]


def _assert_5tuple_same(ref, got):
    assert len(ref) == len(got) == 5
    for i, (a, b) in enumerate(zip(ref, got)):
        a, b = np.asarray(a), np.asarray(b)
        assert a.dtype == b.dtype and a.shape == b.shape, i
        assert a.tobytes() == b.tobytes(), f"5-tuple element {i} bytes differ"


def _assert_gather_same(kw, *, staged, monkeypatch, **over):
    monkeypatch.setattr(cv, "_STAGED_FFT", staged)
    ref = _gather(base._layered_gather_diffuse, kw)
    got = _gather(cv._layered_gather_diffuse, kw, **over)
    assert ref[1] == got[1]
    assert ref[2] == got[2], "progress / log lines differ"
    assert ref[3] == got[3], "warnings differ"
    if ref[1] is None:
        _assert_5tuple_same(ref[0], got[0])
    return got


class _Coverage:
    """Counts which WP1 branches the CURRENT gather actually took, so a green
    equality is not vacuous."""

    def __init__(self, monkeypatch):
        self.c: dict[str, int] = {}
        real_bb, real_ak = cv._support_bbox, cv._apply_kernel
        state = {"delta": False}

        def ak(kernel_base, *a, **k):
            state["delta"] = bool(kernel_base.max() > 0.5 * kernel_base.sum())
            self._inc("delta_unit" if state["delta"] else "wedge_ri")
            return real_ak(kernel_base, *a, **k)

        def bb(wt, support):
            r = real_bb(wt, support)
            br = "delta" if state["delta"] else "wedge"
            if r is None:
                self._inc(br + "_skip")
            else:
                ys, xs = np.nonzero(wt > 0)
                full = (int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1)
                if full != r:
                    self._inc(br + "_cut")
                if r[0] > 0 or r[1] < wt.shape[0]:
                    self._inc("partial_rows")
            return r

        monkeypatch.setattr(cv, "_support_bbox", bb)
        monkeypatch.setattr(cv, "_apply_kernel", ak)

    def _inc(self, k):
        self.c[k] = self.c.get(k, 0) + 1


_RAMP_KW: dict = {}


def _ramp_kw(lens: str, h: int, w: int, cref: float, kind: str = "ramp", **rkw):
    key = (lens, h, w, cref, kind, tuple(sorted(rkw.items())))
    if key not in _RAMP_KW:
        _RAMP_KW[key] = _capture(_image(h, w), _tensor(lens, cref), _depth(h, w, kind),
                                 bright_thresh=0.6, **rkw)
    return _RAMP_KW[key]


# ------------------------------------------------- render level (tiny kernels)

def _tiny_params():
    out = []
    for lens in ("35", "50"):
        for shape in ((64, 64), (65, 47), (47, 65), (1, 1), (2, 9), (3, 5)):
            for kind in ("ramp", "focus", "two"):
                fast = lens == "35" or shape == (65, 47)
                out.append(pytest.param(lens, shape, kind, marks=() if fast else _SLOW,
                                        id=f"{lens}-{shape[0]}x{shape[1]}-{kind}"))
    return out


@pytest.mark.parametrize("lens,shape,kind", _tiny_params())
def test_render_shipped_tensor_tiny_frames(lens, shape, kind):
    h, w = shape
    _assert_render_same(_image(h, w), _tensor(lens), _depth(h, w, kind), bright_thresh=0.8)


# -------------------------------------- gather level, real multi-wedge kernels

@pytest.mark.parametrize("geom,flags,staged", _matrix_params())
def test_gather_matrix_bytes_equal_frozen(geom, flags, staged, monkeypatch):
    lens, h, w = geom
    kw = _ramp_kw(lens, h, w, 250.0)
    cov = _Coverage(monkeypatch)
    _assert_gather_same(kw, staged=staged, monkeypatch=monkeypatch,
                        slab_support_skip=flags[0], basis_hoist=flags[1])
    c = cov.c
    assert c.get("wedge_ri", 0) > 0 and c.get("delta_unit", 0) > 0, c
    if flags[0]:  # support skip on: every branch class must have been hit
        for k in ("delta_skip", "wedge_skip", "wedge_cut", "partial_rows"):
            assert c.get(k, 0) > 0, (k, c)


@pytest.mark.parametrize("staged", _staged_params())
def test_gather_spectrum_over_256kb(staged, monkeypatch):
    """numpy elides temporaries above 256 KB (in-place, so ``F*K`` becomes
    ``K*F``, bitwise different). Keep one case past that size."""
    kw = _ramp_kw("35", 200, 150, 700.0)
    assert kw["fft_state"]["img_F"][0].nbytes > 256 * 1024
    _assert_gather_same(kw, staged=staged, monkeypatch=monkeypatch)


@pytest.mark.parametrize("staged", _staged_params())
@pytest.mark.parametrize("sbs", [pytest.param(None, marks=_SLOW), pytest.param(1, marks=_SLOW), 2,
                                 pytest.param(3, marks=_SLOW), pytest.param(11, marks=_SLOW)])
def test_gather_slab_batch_sizes(sbs, staged, monkeypatch):
    kw = dict(_ramp_kw("35", 60, 44, 250.0))
    kw["slab_batch_size"] = sbs
    _assert_gather_same(kw, staged=staged, monkeypatch=monkeypatch)


@pytest.mark.parametrize("staged", _staged_params(False))
def test_gather_holes_and_near_empty_slabs(staged, monkeypatch):
    """Hand-built non-PoU slab_w: holes with a donor (0.01 < sw < 0.02),
    holes without one, a slab that is exactly 0 (empty-skip), one whose max
    is below _EMPTY_SKIP_THRESH but != 0 (empty-skip with support), and a
    tiny-weight fringe that is != 0 (support) but far below the threshold."""
    kw = dict(_ramp_kw("35", 60, 44, 250.0))
    sw = np.array(kw["slab_w"], dtype=np.float32, copy=True)
    D, H, W = sw.shape
    sw[:, :8, :8] = 0.0                     # hole, no donor
    sw[:, 8:16, 30:40] = 0.0
    sw[3, 8:16, 30:40] = 0.015              # hole, donor = slab 3
    sw[10] = 0.0                            # exactly empty
    sw[9] = np.minimum(sw[9], 0.005)        # max < thresh, still != 0
    sw[5, 40:, :] = np.where(sw[5, 40:, :] == 0, np.float32(1e-6), sw[5, 40:, :])
    kw["slab_w"] = sw
    acc = sw.sum(axis=0)
    holes = acc < 0.02
    donor = (sw > cv._EMPTY_SKIP_THRESH).any(axis=0)
    assert (holes & donor).any() and (holes & ~donor).any()
    got = _assert_gather_same(kw, staged=staged, monkeypatch=monkeypatch)
    log = [ln for ln in got[2] if "layered_gather:" in ln]
    assert log and int(re.search(r"holes=(\d+)", log[0]).group(1)) > 0
    assert "active_slabs=9/11" in log[0], log


@pytest.mark.parametrize("staged", _staged_params())
@pytest.mark.parametrize("img_kind", ["black", pytest.param("dark", marks=_SLOW)])
def test_gather_black_and_no_bright(img_kind, staged, monkeypatch):
    """All-zero bright buffer (and an all-zero frame): signed zeros from the
    spectra must not reach the accumulators."""
    h, w = 54, 70
    kw = _capture(_image(h, w, img_kind), _tensor("35", 250.0), _depth(h, w, "ramp"),
                  bright_thresh=0.9)
    assert not np.any(kw["img_bright_padded"])
    got = _assert_gather_same(kw, staged=staged, monkeypatch=monkeypatch)
    assert not np.signbit(got[0][1]).any()  # output_bright stays +0.0


@pytest.mark.parametrize("staged", _staged_params(False))
def test_gather_all_in_focus_delta_only(staged, monkeypatch):
    kw = _capture(_image(57, 43), _tensor("50", 250.0), _depth(57, 43, "focus"),
                  bright_thresh=0.6)
    cov = _Coverage(monkeypatch)
    _assert_gather_same(kw, staged=staged, monkeypatch=monkeypatch)
    assert cov.c.get("delta_unit", 0) > 0 and cov.c.get("wedge_ri", 0) == 0, cov.c


@pytest.mark.parametrize("staged", _staged_params())
def test_gather_repeat_runs_deterministic(staged, monkeypatch):
    kw = _ramp_kw("35", 72, 53, 250.0)
    monkeypatch.setattr(cv, "_STAGED_FFT", staged)
    outs = [_gather(cv._layered_gather_diffuse, kw)[0] for _ in range(3)]
    for o in outs[1:]:
        _assert_5tuple_same(outs[0], o)


# ------------------------------------------------- render level, larger kernels

@pytest.mark.parametrize("lens,occ", [("35", "off"), ("50", "on"),
                                      pytest.param("35", "on", marks=_SLOW),
                                      pytest.param("50", "off", marks=_SLOW)])
def test_render_on_and_off_paths(lens, occ):
    h, w = 61, 83
    _assert_render_same(_image(h, w, seed=1), _tensor(lens, 300.0), _depth(h, w, "ramp"),
                        bright_thresh=0.6, occlusion_compositing=occ)


@pytest.mark.parametrize("variant", ["l1_stamp", pytest.param("no_bright", marks=_SLOW),
                                     pytest.param("batch3_on", marks=_SLOW)])
def test_render_variants(variant):
    h, w = 59, 47
    kw = dict(bright_thresh=0.6)
    if variant == "l1_stamp":
        kw["bright_stamp_gain"] = 2.0
    elif variant == "no_bright":
        kw = {}
    else:
        kw.update(slab_batch_size=3, occlusion_compositing="on")
    _assert_render_same(_image(h, w, seed=2), _tensor("35", 250.0), _depth(h, w, "ramp"), **kw)


def _edge_chart(h, w):
    """Point lights at radial_frac 0.6..0.98 on 8 azimuths, plus one light
    sitting exactly where a slab's support ends."""
    img = np.full((h, w, 3), 20, np.uint8)
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    half = float(np.hypot(cy, cx))
    for rf in (0.6, 0.7, 0.8, 0.9, 0.98):
        for k in range(8):
            t = np.radians(22.5 + 45.0 * k)
            y = int(round(cy + rf * half * np.sin(t)))
            x = int(round(cx + rf * half * np.cos(t)))
            if 0 <= y < h and 0 <= x < w:
                img[y, x] = 255
    return img


@pytest.mark.parametrize("lens", ["35", pytest.param("50", marks=_SLOW)])
def test_render_field_edge_chart(lens):
    h, w = 80, 120
    _assert_render_same(_edge_chart(h, w), _tensor(lens, 300.0), _depth(h, w, "ramp"),
                        bright_thresh=0.8)
    # through the shared defocus fixture as the gate's chart does
    t = _tensor(lens, 300.0)
    img = _edge_chart(h, w)
    a = b = None
    orig = cv._layered_gather_diffuse
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            a = render_with_defocus(img, t, hdr_source="none", bright_thresh=1.0,
                                    on_progress=lambda m: None)
            cv._layered_gather_diffuse = base._layered_gather_diffuse
            b = render_with_defocus(img, t, hdr_source="none", bright_thresh=1.0,
                                    on_progress=lambda m: None)
    finally:
        cv._layered_gather_diffuse = orig
    assert a[1].tobytes() == b[1].tobytes() and a[0].tobytes() == b[0].tobytes()


def test_render_highlight_on_support_edge():
    """A highlight exactly on the column where slab 0's support ends (sw goes
    from tiny to 0 within the kernel reach)."""
    h, w = 48, 66
    depth = _depth(h, w, "two")
    img = _image(h, w, "dark")
    img[h // 2, w // 2 - 1: w // 2 + 1] = 255
    img[5, w // 2] = 255
    _assert_render_same(img, _tensor("35", 250.0), depth, bright_thresh=0.8)
    _assert_render_same(img, _tensor("35", 250.0), depth, bright_thresh=0.8,
                        occlusion_compositing="on")


@pytest.mark.parametrize("nd", [(1, 1), (1, 3), (2, 2), (12, 1), (5, 11)])
def test_render_degenerate_tensor_axes(nd):
    """R or D of 1 or 2 (Nr == 1 makes the basis all-ones; D == 1 is one slab)."""
    n_r, d = nd
    h, w = 41, 33
    t = _synthetic_tensor(n_r, d)
    depth = np.broadcast_to(np.geomspace(0.7, 20.0, w, dtype=np.float32)[None, :], (h, w)).copy()
    _assert_render_same(_image(h, w, seed=3), t, depth, bright_thresh=0.7)


def test_render_wild_depth_nan_and_out_of_range():
    h, w = 40, 52
    _assert_render_same(_image(h, w), _tensor("35", 250.0), _depth(h, w, "wild"),
                        bright_thresh=0.7)


def test_render_nonfinite_input_same_outcome():
    """R-A3: one NaN input pixel. The gather's 5-tuple may differ under WP1 at
    mix == 0 pixels, but render() must end the same way (today: both raise
    the NaN output gate with the same text)."""
    h, w = 45, 60
    img = np.random.default_rng(5).random((h, w, 3)).astype(np.float32)
    img[7, 50, 0] = np.nan
    a = _assert_render_same(img, _tensor("35", 250.0), _depth(h, w, "two"),
                            bright_thresh=0.7)
    assert a["err"] is not None and "NaN" in a["err"]


# ------------------------------------------------------------------- cancel

class _CountingEvent(threading.Event):
    """is_set() turns True from the n-th poll on (0 = pre-set)."""

    def __init__(self, n: int):
        super().__init__()
        self.n, self.calls = n, 0

    def is_set(self):
        self.calls += 1
        return self.calls > self.n


@pytest.mark.parametrize("staged,full_sweep", [
    pytest.param(m, fs, marks=_SLOW if (fs or m != cv._STAGED_FFT) else ())
    for m in _STAGED_MODES for fs in (False, True)])
def test_cancel_texts_and_progress_at_every_poll_class(staged, full_sweep, monkeypatch):
    kw = _ramp_kw("35", 48, 40, 250.0)
    monkeypatch.setattr(cv, "_STAGED_FFT", staged)
    full = _CountingEvent(10 ** 9)
    _gather(cv._layered_gather_diffuse, kw, cancel=full)
    n_polls = full.calls
    seen_at, seen_inside = False, False
    ns = ({0, 1, 2, 3, 4, 7, 13, 14, 25, n_polls // 2, n_polls - 1, n_polls} if full_sweep
          else {0, 2, 3, n_polls})
    for n in sorted(ns):
        ref = _gather(base._layered_gather_diffuse, kw, cancel=_CountingEvent(n))
        got = _gather(cv._layered_gather_diffuse, kw, cancel=_CountingEvent(n))
        assert ref[1] == got[1], n
        assert ref[2] == got[2], n
        if ref[1] is None:
            _assert_5tuple_same(ref[0], got[0])
        else:
            seen_at |= "cancelled at slab" in ref[1]
            seen_inside |= "inside slab" in ref[1]
    assert seen_at and seen_inside


@pytest.mark.parametrize("staged", _staged_params())
def test_cancel_set_from_progress_callback(staged, monkeypatch):
    kw = _ramp_kw("35", 48, 40, 250.0)
    monkeypatch.setattr(cv, "_STAGED_FFT", staged)
    for target in ("  Slab 1/11", "  Slab 4/11", "  Slab 11/11"):
        res = []
        for fn in (base._layered_gather_diffuse, cv._layered_gather_diffuse):
            ev = threading.Event()
            lines: list[str] = []

            def prog(m, ev=ev, lines=lines):
                lines.append(m)
                if m == target:
                    ev.set()

            k = dict(kw, cancel=ev, on_progress=prog)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with pytest.raises(Exception) as ei:
                    fn(**k)
            res.append((type(ei.value).__name__, str(ei.value), lines))
        assert res[0] == res[1], target


def test_preset_cancel_calls_no_fft_primitive(monkeypatch):
    kw = _ramp_kw("35", 48, 40, 250.0)
    calls: list[str] = []
    for name in ("rfft", "irfft", "fft", "ifft", "rfft2", "irfft2", "rfftn", "irfftn",
                 "fftn", "ifftn"):
        real = getattr(sp_fft, name)

        def spy(*a, _n=name, _r=real, **k):
            calls.append(_n)
            return _r(*a, **k)

        monkeypatch.setattr(sp_fft, name, spy)
    monkeypatch.setattr(cv, "fftconvolve", lambda *a, **k: calls.append("fftconvolve"))
    ev = threading.Event()
    ev.set()
    got = _gather(cv._layered_gather_diffuse, kw, cancel=ev)
    ref = _gather(base._layered_gather_diffuse, kw, cancel=ev)
    assert got[1] == ref[1] == "CancelledError: render cancelled at slab 0/11"
    assert got[2] == ref[2] == []
    assert calls == []


# ------------------------------------------------------ staged FFT helpers

def _kernel(kind: str, k: int, rng) -> np.ndarray:
    pad = k // 2
    if kind == "dense":
        ker = rng.random((k, k), dtype=np.float32)
    elif kind == "delta":
        ker = cv._make_delta(k)
    elif kind == "sparse":
        ker = rng.random((k, k), dtype=np.float32)
        ker[rng.random((k, k)) < 0.7] = 0
    else:
        yy, xx = np.mgrid[-pad:pad + 1, -pad:pad + 1]
        ker = ((yy ** 2 + xx ** 2) <= pad ** 2).astype(np.float32)
    return ker / np.float32(max(float(ker.sum()), 1e-12))


@pytest.mark.skipif(not cv._STAGED_FFT, reason="staged FFT path not active on this build")
@pytest.mark.parametrize("seed", [0, 1, pytest.param(2, marks=_SLOW), pytest.param(3, marks=_SLOW)])
def test_staged_helpers_random_geometry_sweep(seed):
    """Random odd/even fshapes (from _fft_geometry), kernels 1..183 px of four
    kinds incl. exact deltas and sparse/disc kernels, random row windows, and
    workers 1/2/3/-1. Outputs must be byte-equal; kernel spectra value-equal
    (their signed zeros may differ, see the xfail below)."""
    rng = np.random.default_rng(1000 + seed)
    for _ in range(12):
        h, w = (int(v) for v in rng.integers(1, 360, 2))
        k = int(rng.choice([1, 3, 5, 9, 17, 31, 67, 99, 183]))
        kind = str(rng.choice(["dense", "delta", "sparse", "disc"]))
        hp, wp = h + 2 * (k // 2), w + 2 * (k // 2)
        fshape = cv._fft_geometry(hp, wp, k, padded=True)["fshape"]
        img = rng.random((hp, wp), dtype=np.float32)
        ker = _kernel(kind, k, rng)
        with sp_fft.set_workers(int(rng.choice([1, 2, 3, -1]))):
            img_F = sp_fft.rfft2(img, s=fshape)
            kref = sp_fft.rfft2(ker, s=fshape)
            kst = cv._kernel_rfft2_staged(ker, fshape)
            assert np.array_equal(kref, kst), (h, w, k, kind)
            X = img_F * kref  # named operands: no temporary elision
            full = sp_fft.irfft2(X, s=fshape)
            for _ in range(3):
                r0 = int(rng.integers(0, fshape[0]))
                r1 = int(rng.integers(r0 + 1, fshape[0] + 1))
                assert full[r0:r1].tobytes() == cv._irfft2_rows(X, fshape, r0, r1).tobytes()
            sy = slice((k - 1) // 2, (k - 1) // 2 + hp)
            sx = slice((k - 1) // 2, (k - 1) // 2 + wp)
            got = cv._fftconvolve_hoisted(img_F, ker, fshape, sy, sx)
            ref = base._fftconvolve_hoisted(img_F, ker, fshape, sy, sx)
            assert ref.tobytes() == got.tobytes(), (h, w, k, kind)


@pytest.mark.skipif(not cv._STAGED_FFT, reason="staged FFT path not active on this build")
@pytest.mark.xfail(strict=True, reason=(
    "FINDING (helper contract, not an output bug): _kernel_rfft2_staged is "
    "value-equal but NOT tobytes-equal to sp_fft.rfft2 for exact-delta, sparse "
    "and disc kernels: some spectrum entries are +0.0 where rfft2 gives -0.0 "
    "(e.g. _make_delta(1), fshape (8, 8)). The WP2 unit test only uses dense "
    "random kernels. The difference never reaches the gather 5-tuple "
    "(+0-seeded accumulators), so Tier A output stays byte-exact."))
def test_staged_kernel_spectrum_bytes_for_delta():
    ker = cv._make_delta(1)
    assert sp_fft.rfft2(ker, s=(8, 8)).tobytes() == cv._kernel_rfft2_staged(ker, (8, 8)).tobytes()


@pytest.mark.parametrize("staged", _staged_params())
@pytest.mark.parametrize("support", ["none", "partial", "empty"])
def test_apply_kernel_unit_vs_frozen(support, staged, monkeypatch):
    """One _apply_kernel unit at a large-ish fshape. With no support the
    accumulators are byte-equal everywhere; with support they are byte-equal
    on the support and the weight sums are byte-equal everywhere."""
    monkeypatch.setattr(cv, "_STAGED_FFT", staged)
    kw = _ramp_kw("35", 200, 150, 700.0)
    H, W, pad = kw["H"], kw["W"], kw["pad"]
    ri, d = 9, 6
    kb = base._build_basis_kernel(ri, d, True, kw["defocus_ratios"], kw["kernels"],
                                  kw["kH"], kw["kW"], kw["pix_scale"], kw["scaled_ksize"])
    w = base._radial_basis_weights(kw["radial_frac_map"], kw["r_centers"])[ri]
    sup = {"none": None, "partial": kw["slab_w"][d] != 0,
           "empty": np.zeros((H, W), bool)}[support]

    def acc():
        return (np.zeros((H, W, 3), np.float32), np.zeros((H, W, 3), np.float32),
                np.zeros((H, W), np.float32), np.zeros((H, W), np.float32))

    ra, ga = acc(), acc()
    args = (w, kw["img_padded"], kw["img_diffuse_padded"], kw["img_bright_padded"],
            kw["wedge_idx"], pad, H, W)
    with sp_fft.set_workers(-1):
        base._apply_kernel(kb, *args, *ra, fft_state=kw["fft_state"])
        cv._apply_kernel(kb, *args, *ga, fft_state=kw["fft_state"], slab_support=sup)
    assert ra[2].tobytes() == ga[2].tobytes() and ra[3].tobytes() == ga[3].tobytes()
    m = np.ones((H, W), bool) if sup is None else sup
    for i in (0, 1):
        assert ra[i][m].tobytes() == ga[i][m].tobytes()
    if support == "partial":
        assert 0 < m.sum() < m.size


# ------------------------------------------------------------------- memory

@_SLOW
@pytest.mark.parametrize("staged", _STAGED_MODES)
def test_gather_traced_peak_not_above_frozen(staged, monkeypatch):
    """The memory estimator is unchanged in Tier A, so the gather must not
    allocate more than the base gather did (numpy allocations, traced)."""
    kw = _ramp_kw("35", 200, 150, 700.0)
    monkeypatch.setattr(cv, "_STAGED_FFT", staged)
    peaks = []
    for fn in (base._layered_gather_diffuse, cv._layered_gather_diffuse):
        tracemalloc.start()
        try:
            _gather(fn, kw)
            peaks.append(tracemalloc.get_traced_memory()[1])
        finally:
            tracemalloc.stop()
    assert peaks[1] <= peaks[0], peaks
