"""S33 Tier B adversarial (field lens): prove that field dependence survived
the tiled engine.

Self-contained: no ``runs/``, no goldens, no IMG_6915, no other checkout. The
oracle is the in-tree reference gather (``convolve._layered_gather_diffuse``),
run in-test, and kernels rebuilt here from the reference's own primitives
(``_build_basis_kernel``, ``ndimage_rotate``, ``_bright_stamp``). Tolerances are
the Tier B ceilings from ``tests/fixtures/tolerance.py``; none is loosened.

What is attacked:

1. Kernel bank completeness and identity per (slab, radial bin, wedge): with
   every slab non-empty everywhere, the tiled engine banks all 11 slabs x 12
   bins x 8 wedges (diffuse and bright), each label's kernel is sha256-equal to
   the reference's rotated / renormalised / stamped kernel for THAT label, and
   the bank equals the reference's capture in content and order.
2. Per-unit impulse parity: every banked unit's spectrum, pushed through the
   engine's own tile primitives (``_tile_spec`` + ``_convolve_chunk``) with a
   single impulse, returns that unit's kernel (no crop, no resample, no
   mirror, no transposition).
3. Impulse grid: single-pixel lights at 12 radii x 16 azimuths (8 on wedge
   boundaries, 8 mid-wedge) plus corner lights at radial_frac > 0.9, depths
   cycling over every slab node and between-node depths, square / landscape /
   portrait, K=17 and K=67. Per-light PSF crops must meet the pointwise and
   energy ceilings; the full frame and the field-edge annulus must meet all of
   Tier B.
4. Sensitivity: the per-light judge FAILS on mutant engines that lose field
   dependence (mirrored rotation sign, one wedge, one radial bin, one slab, a
   single global kernel, a cropped PSF, a mirrored PSF, a wedge off by one), so
   (3) cannot go green on an engine without field character.
5. Static: no code path in the tiled engine or its schedule with fewer wedges,
   bins or slabs, a global kernel, or a resampled / cropped / flipped PSF.
"""
from __future__ import annotations

import ast
import hashlib
import inspect
import warnings
from functools import lru_cache

import numpy as np
import pytest

import summicron.render.convolve as cv
from summicron.render import _engine_tiled as eng
from summicron.render import _tile_schedule as ts
from summicron.render._artifacts import default_tensor_path
from tests.fixtures import tolerance as tol

TIER_B = tol.TIERS["B"]


# --------------------------------------------------------------------------
# builders (self-contained copies of what render() hands to the gather)
# --------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _tensor(lens: str) -> dict:
    return cv.load_tensor(default_tensor_path(lens))


def _geometry(h, w, radial_edges):
    ys = np.arange(h, dtype=np.float32) - np.float32((h - 1) / 2.0)
    xs = np.arange(w, dtype=np.float32) - np.float32((w - 1) / 2.0)
    max_radial = float(np.hypot((h - 1) / 2.0, (w - 1) / 2.0))
    rf = (np.hypot(ys[:, None], xs[None, :]) / np.float32(max_radial)).astype(np.float32)
    theta = np.arctan2(ys[:, None], xs[None, :])
    wedge = ((theta + np.pi) / (2 * np.pi) * cv.N_ANG).astype(np.int32) % cv.N_ANG
    rc = 0.5 * (radial_edges[:-1] + radial_edges[1:])
    return rf, wedge, rc


def _kw(img, depth, *, lens="35", K=17, thresh=1.0, gain=2.0, slab_w=None):
    t = _tensor(lens)
    kernels = t["kernels"]
    kH, kW = kernels.shape[2], kernels.shape[3]
    H, W = img.shape[:2]
    pix_scale = 1.0 if K == kH else K / kH
    pad = K // 2
    rf, wedge, rc = _geometry(H, W, np.asarray(t["radial_edges"]))
    ip = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
    lum = 0.2126 * ip[..., 0] + 0.7152 * ip[..., 1] + 0.0722 * ip[..., 2]
    bright = ip * (lum > thresh).astype(np.float32)[..., None]
    if slab_w is None:
        slab_w = cv.compute_slab_weights(depth, t["object_distances_m"])
    return dict(
        depth_m=depth, slab_w=slab_w, img_lin=img,
        img_padded=ip, img_diffuse_padded=ip - bright, img_bright_padded=bright,
        kernels=kernels, defocus_ratios=t["defocus_ratios"],
        radial_frac_map=rf, r_centers=rc, wedge_idx=wedge,
        kH=kH, kW=kW, pad=pad, pix_scale=pix_scale, scaled_ksize=K, H=H, W=W,
        slab_batch_size=None, cancel=None, bright_stamp_gain=gain,
    )


def _ref(kw):
    k = dict(kw)
    k["fft_state"] = cv._build_image_fft_state(
        kw["img_padded"], kw["img_diffuse_padded"], kw["img_bright_padded"],
        kw["scaled_ksize"])
    k["on_progress"] = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return cv._layered_gather_diffuse(**k)


def _tiled(kw, **extra):
    k = dict(kw)
    k["fft_state"] = None
    k["on_progress"] = None
    k.update(extra)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return eng.tiled_layered_gather(**k)


def _total(t5):
    return (t5[0] + t5[1]).astype(np.float32)


def _sha(a):
    return hashlib.sha256(np.ascontiguousarray(a, dtype=np.float32).tobytes()).hexdigest()


def _expected_unit_kernels(kw, d, ri, a):
    """The reference's kernels for label (d, ri, a | None), rebuilt here from
    the reference primitives (never through ``_tile_schedule``)."""
    base = cv._build_basis_kernel(
        ri, d, True, kw["defocus_ratios"], kw["kernels"], kw["kH"], kw["kW"],
        kw["pix_scale"], kw["scaled_ksize"])
    if a is None:
        return base, None
    theta_a = (a + 0.5) * (2 * np.pi / cv.N_ANG) - np.pi
    rot_deg = -(theta_a * 180.0 / np.pi + 90.0)
    rotated = cv.ndimage_rotate(base, rot_deg, reshape=False, order=1)
    rotated = np.maximum(rotated, 0.0)
    rotated /= rotated.sum() + 1e-12
    return rotated, cv._bright_stamp(rotated, kw["bright_stamp_gain"])


def _is_delta(base):
    return bool(base.max() > 0.5 * base.sum())


# --------------------------------------------------------------------------
# 1 + 2: full bank, label identity, per-unit impulse parity
# --------------------------------------------------------------------------

IMPULSE_REL = 1e-5       # per-unit impulse: max |conv - kernel| / kernel peak


def _impulse_check(spec, ker, K, N):
    """Push one impulse through the engine's own tile primitives
    (``_tile_spec`` + ``_convolve_chunk``) with ``spec``: an impulse at window
    (K-1, K-1) returns the kernel on a K x K output tile. Returns
    ``(max |conv - ker| / ker.max() over the 3 channels, not-a-mirror)``."""
    imp = np.zeros((K + K - 1, K + K - 1, 3), np.float32)
    imp[K - 1, K - 1, :] = 1.0
    S = eng._tile_spec(imp, (0, K, 0, K), K, N)
    conv = eng._convolve_chunk([S], [spec], K, K, K, N)[0]
    peak = float(ker.max())
    rel = max(float(np.abs(conv[c].astype(np.float64) - ker).max()) / peak
              for c in range(3))
    ok = True
    for alt in (ker[::-1, :], ker[:, ::-1], ker.T):
        gap = float(np.abs(alt - ker).max())
        if gap > 1e-3 * peak:
            ok &= float(np.abs(conv[0] - alt).max()) > 0.5 * gap
    return rel, ok


def _all_slabs_everywhere(h, w, D):
    """Every slab non-empty at every pixel (sums to 1 per pixel)."""
    return np.full((D, h, w), np.float32(1.0 / D), dtype=np.float32)


def _capture_bank_units(monkeypatch):
    """Spy on ``ts.slab_bank``: keep each returned unit with copies of its
    spatial kernels (the engine frees them after building the spectra)."""
    seen = []
    real = ts.slab_bank

    def spy(d, **k):
        units = real(d, **k)
        for u in units:
            seen.append(dict(
                d=d, unit=u,
                kernel=None if u.kernel is None else u.kernel.copy(),
                kernel_bright=None if u.kernel_bright is None else u.kernel_bright.copy(),
            ))
        return units

    monkeypatch.setattr(ts, "slab_bank", spy)
    return seen


def _capture_reference_hashes(kw, monkeypatch):
    """sha256 of every kernel the reference hands to a kernel FFT, in order,
    with the triple per-channel transform of is_delta units collapsed."""
    seen = []
    K = kw["scaled_ksize"]
    if cv._STAGED_FFT:
        real = cv._kernel_rfft2_staged

        def spy(k, fshape):
            seen.append((_sha(k), _is_delta(k)))
            return real(k, fshape)

        monkeypatch.setattr(cv, "_kernel_rfft2_staged", spy)
    else:
        real = cv.sp_fft.rfft2

        def spy(x, *a, **k):
            if getattr(x, "ndim", 0) == 2 and x.shape == (K, K):
                seen.append((_sha(x), _is_delta(x)))
            return real(x, *a, **k)

        monkeypatch.setattr(cv.sp_fft, "rfft2", spy)
    _ref(kw)
    monkeypatch.undo()
    out, run = [], 0
    for h, delta in seen:
        if delta:
            run += 1
            if run % 3 != 1:
                continue
        else:
            run = 0
        out.append(h)
    return out


def _bank_scene(h, w, lens, K):
    t = _tensor(lens)
    D = len(t["object_distances_m"])
    rng = np.random.default_rng(42)
    img = rng.uniform(0.0, 0.6, (h, w, 3)).astype(np.float32)
    img[h // 3, w // 3] = 9.0                  # one bright pixel: bright bank
    depth = np.full((h, w), np.float32(t["object_distances_m"][5]))
    return _kw(img, depth, lens=lens, K=K, slab_w=_all_slabs_everywhere(h, w, D)), D


@pytest.mark.parametrize("lens,K,n_tile,hw", [
    ("35", 17, 40, (121, 121)),
    ("50", 17, 40, (121, 121)),
    ("35", 67, 144, (181, 181)),
])
def test_full_bank_every_slab_bin_wedge_label_identity_and_impulse(
        monkeypatch, lens, K, n_tile, hw):
    kw, D = _bank_scene(*hw, lens, K)
    ref_hashes = _capture_reference_hashes(kw, monkeypatch)

    seen = _capture_bank_units(monkeypatch)
    _tiled(kw, _tile_n=n_tile)
    monkeypatch.undo()

    nr = len(kw["r_centers"])
    assert nr == 12 and cv.N_ANG == 8 and D == 11
    # every slab went through slab_bank, ascending
    assert sorted({e["d"] for e in seen}) == list(range(D))

    keys, tiled_hashes = [], []
    n_impulse = 0
    worst_rel = 0.0
    N = n_tile
    assert N >= 2 * K - 1                      # one tile holds a whole kernel
    for e in seen:
        u, d = e["unit"], e["d"]
        assert u.bank, f"unit d={d} ri={u.ri} a={u.a} not banked with full support"
        want, want_b = _expected_unit_kernels(kw, d, u.ri, u.a)
        assert e["kernel"].shape == (K, K)
        assert _sha(e["kernel"]) == _sha(want), (d, u.ri, u.a)
        keys.append((d, u.ri, u.a))
        tiled_hashes.append(_sha(e["kernel"]))
        if u.a is not None:
            assert _sha(e["kernel_bright"]) == _sha(want_b), (d, u.ri, u.a, "bright")
            tiled_hashes.append(_sha(e["kernel_bright"]))
        # per-unit impulse through the engine's own tile primitives
        pairs = [(u.spec, want)]
        if u.a is not None:
            pairs.append((u.spec_bright, want_b))
        for spec, ker in pairs:
            assert spec is not None and spec.shape == (N, N // 2 + 1)
            rel, mirror_ok = _impulse_check(spec, ker, K, N)
            worst_rel = max(worst_rel, rel)
            assert rel <= IMPULSE_REL, (d, u.ri, u.a, rel)
            assert mirror_ok, (d, u.ri, u.a, "matches a mirror/transpose")
            n_impulse += 1

    # bank is complete: every (d, ri) has its delta unit or all 8 wedges
    by_dri: dict = {}
    for d, ri, a in keys:
        by_dri.setdefault((d, ri), set()).add(a)
    assert set(by_dri) == {(d, ri) for d in range(D) for ri in range(nr)}
    for (d, ri), wedges in by_dri.items():
        base = cv._build_basis_kernel(
            ri, d, True, kw["defocus_ratios"], kw["kernels"], kw["kH"], kw["kW"],
            kw["pix_scale"], kw["scaled_ksize"])
        if _is_delta(base):
            assert wedges == {None}, (d, ri, wedges)
        else:
            assert wedges == set(range(cv.N_ANG)), (d, ri, wedges)
    n_rot = sum(1 for _, _, a in keys if a is not None)
    # rotated (wedge) units in at least 9 of the 11 slabs (d0 is the delta)
    assert len({d for d, _, a in keys if a is not None}) >= D - 2, keys
    # same bank content and order as the reference actually transformed
    assert tiled_hashes == ref_hashes
    print(f"lens {lens} K={K}: {len(keys)} units ({n_rot} rotated), "
          f"{n_impulse} impulse checks (worst err/peak {worst_rel:.3g}), bank == reference ({len(ref_hashes)} kernels)")


def test_per_unit_impulse_check_catches_a_spectrum_on_the_wrong_unit(monkeypatch):
    """Sensitivity of the per-unit impulse check: shift the engine's spectra
    by one unit (owners/spectra misaligned) and the check must fire on most
    units."""
    K, N = 17, 40
    kw, _ = _bank_scene(121, 121, "35", K)
    real = eng._kernel_spectra

    def shifted(arrays, n, labels=None, workers=1):
        out = real(arrays, n, labels, workers)
        return np.roll(out, 1, axis=0)

    seen = _capture_bank_units(monkeypatch)
    monkeypatch.setattr(eng, "_kernel_spectra", shifted)
    _tiled(kw, _tile_n=N)
    monkeypatch.undo()
    n, bad = 0, 0
    for e in seen:
        u, d = e["unit"], e["d"]
        want, want_b = _expected_unit_kernels(kw, d, u.ri, u.a)
        pairs = [(u.spec, want)] + ([(u.spec_bright, want_b)] if u.a is not None else [])
        for spec, ker in pairs:
            rel, ok = _impulse_check(spec, ker, K, N)
            n += 1
            bad += (rel > IMPULSE_REL) or (not ok)
    print(f"spec shifted by one: {bad}/{n} units flagged")
    assert bad >= 0.9 * n, (bad, n)


# --------------------------------------------------------------------------
# 3: impulse grid, per-light crops
# --------------------------------------------------------------------------

def _depth_cycle(tensor):
    od = np.asarray(tensor["object_distances_m"], np.float64)
    nodes = [float(od[i]) for i in range(1, len(od))]
    mids = [float(np.sqrt(od[i] * od[i + 1])) for i in range(len(od) - 1)]
    return [x for p in zip(nodes, mids) for x in p]


def impulse_grid(h, w, amp, *, lens="35", n_az=16):
    """Single-pixel lights at 12 radii x ``n_az`` azimuths (even index on a
    wedge boundary, odd mid-wedge) plus 3 corner lights per corner
    (radial_frac 0.92 / 0.95 / 0.98). Depth: nearest-light cells cycling over
    every slab node d1..d10 and between-node depths."""
    t = _tensor(lens)
    edges = np.asarray(t["radial_edges"], np.float64)
    rc = 0.5 * (edges[:-1] + edges[1:])
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    maxr = float(np.hypot(cy, cx))
    depths = _depth_cycle(t)
    img = np.zeros((h, w, 3), np.float32)
    lights = []
    pos = set()

    def put(y, x, kind, az):
        if (y, x) in pos:
            return
        pos.add((y, x))
        img[y, x] = amp
        lights.append(dict(y=y, x=x, kind=kind, az=az,
                           depth=depths[len(lights) % len(depths)]))

    for ai in range(n_az):
        th = -np.pi + ai * (2 * np.pi / n_az)
        s, c = np.sin(th), np.cos(th)
        lim = min((cy - 1) / abs(s) if abs(s) > 1e-9 else np.inf,
                  (cx - 1) / abs(c) if abs(c) > 1e-9 else np.inf)
        for ri in range(len(rc)):
            R = rc[ri] * maxr
            if R > lim:                    # outer bins along a short ray:
                R = lim * (0.55 + 0.45 * ri / (len(rc) - 1))
            put(int(round(cy + R * s)), int(round(cx + R * c)),
                "boundary" if ai % 2 == 0 else "mid", ai)
    for sy, sx in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
        for f in (0.92, 0.95, 0.98):
            put(int(round(cy + sy * f * cy)), int(round(cx + sx * f * cx)),
                "corner", None)
    ly = np.array([lt["y"] for lt in lights], np.float32)
    lx = np.array([lt["x"] for lt in lights], np.float32)
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    best = np.full((h, w), np.inf, np.float32)
    depth = np.empty((h, w), np.float32)
    for i, lt in enumerate(lights):
        dd = (ys - ly[i]) ** 2 + (xs - lx[i]) ** 2
        m = dd < best
        best[m] = dd[m]
        depth[m] = np.float32(lt["depth"])
    return img, depth, lights


def judge_lights(ref_tot, got_tot, lights, K):
    """Per-light PSF crop (side K + 2, centred on the light): pointwise max e
    and crop energy ratio against the Tier B ceilings. Returns the failing
    lights and the per-light (max e, p99.9 e, energy) rows."""
    h, w = ref_tot.shape[:2]
    half = K // 2 + 1
    fails, rows = [], []
    for lt in lights:
        y0, y1 = max(0, lt["y"] - half), min(h, lt["y"] + half + 1)
        x0, x1 = max(0, lt["x"] - half), min(w, lt["x"] + half + 1)
        r = ref_tot[y0:y1, x0:x1].astype(np.float64)
        g = got_tot[y0:y1, x0:x1].astype(np.float64)
        e = np.abs(g - r) / np.maximum(r, 1.0)
        se = float(r.sum())
        en = abs(float(g.sum()) / se - 1.0) if se > 0 else float("inf")
        row = (float(e.max()), float(np.percentile(e, 99.9)), en)
        rows.append(row)
        if not (row[0] <= TIER_B["max_e"] and row[2] <= TIER_B["energy"]
                and np.isfinite(g).all()):
            fails.append((lt, row))
    return fails, rows


def _grid_case(h, w, K, n_tile, amp, gain, lens="35"):
    img, depth, lights = impulse_grid(h, w, amp, lens=lens)
    kw = _kw(img, depth, lens=lens, K=K, thresh=1.0, gain=gain)
    rf = kw["radial_frac_map"]
    edges = np.asarray(_tensor(lens)["radial_edges"])
    for lt in lights:
        lt["rf"] = float(rf[lt["y"], lt["x"]])
        lt["ri"] = int(np.clip(np.searchsorted(edges, lt["rf"], side="right") - 1, 0, 11))
        lt["wedge"] = int(kw["wedge_idx"][lt["y"], lt["x"]])
    return kw, lights


def _assert_grid_coverage(kw, lights):
    D = kw["slab_w"].shape[0]
    assert {lt["ri"] for lt in lights} == set(range(12))
    assert len({lt["az"] for lt in lights if lt["az"] is not None}) >= 16
    assert sum(lt["rf"] > 0.9 for lt in lights) >= 8
    assert sum(lt["kind"] == "boundary" for lt in lights) >= 8 * 12 - 8
    assert {lt["wedge"] for lt in lights} == set(range(cv.N_ANG))
    # every slab d1..d10 is non-empty somewhere
    sw_max = kw["slab_w"].reshape(D, -1).max(axis=1)
    assert all(sw_max[d] >= cv._EMPTY_SKIP_THRESH for d in range(1, D)), sw_max


GRID_CASES = [
    # (h, w, K, tile_n, amp, gain)
    (301, 301, 17, 40, 1.0, None),        # LDR, peak stamp (diffuse path)
    (301, 301, 17, 40, 16.0, 2.0),        # HDR 16x, L1 gain 2 (bright path)
    (241, 321, 17, 44, 16.0, 2.0),        # landscape, non-multiple tiles
    (321, 241, 17, 44, 1.0, None),        # portrait
    (261, 261, 67, 96, 16.0, 2.0),        # K = 67 (the 2048 kernel side)
    (231, 289, 67, 96, 1.0, None),        # K = 67 landscape
]


@pytest.mark.parametrize("h,w,K,n_tile,amp,gain", GRID_CASES)
def test_impulse_grid_per_light_crops_match_reference(h, w, K, n_tile, amp, gain):
    kw, lights = _grid_case(h, w, K, n_tile, amp, gain)
    _assert_grid_coverage(kw, lights)
    assert len(ts.plan_tiles(h, w, K, n_tile)[1]) >= 9
    ref5 = _ref(kw)
    got5 = _tiled(kw, _tile_n=n_tile)
    ref_tot, got_tot = _total(ref5), _total(got5)
    if amp > 1.0:
        assert float(ref5[1].max()) > 0.01          # bright path really ran
    fails, rows = judge_lights(ref_tot, got_tot, lights, K)
    assert not fails, (
        f"{len(fails)} lights over the Tier B pointwise/energy ceilings; first "
        f"{[(f[0]['kind'], round(f[0]['rf'], 3), f[0]['wedge'], f[1]) for f in fails[:5]]}")
    res = tol.evaluate(ref_tot, got_tot, "B")
    assert res["ok"], tol.summary(res)
    for i, name in ((2, "weight_sum"), (3, "delta_weight_sum"), (4, "acc_alpha")):
        assert np.array_equal(ref5[i], got5[i]), name
    corner = [r for r, lt in zip(rows, lights) if lt["rf"] > 0.9]
    bnd = [r for r, lt in zip(rows, lights) if lt["kind"] == "boundary"]
    print(f"{h}x{w} K={K} amp={amp}: {len(lights)} lights; per-light max e "
          f"{max(r[0] for r in rows):.3g}, p99.9 max {max(r[1] for r in rows):.3g} "
          f"(report-only), energy max {max(r[2] for r in rows):.3g}; corner "
          f"({len(corner)}) max e {max(r[0] for r in corner):.3g}; wedge-boundary "
          f"({len(bnd)}) max e {max(r[0] for r in bnd):.3g}; {tol.summary(res)}")


def test_production_tile_n_k67_with_corner_and_boundary_lights():
    """Every other FAST tiled check forces ``_tile_n``; this one runs the
    production table (K=67 -> N=324, T=258) on a 3 x 2 tile frame."""
    h, w, K = 600, 460, 67
    assert ts.tile_n_for_ksize(K) == 324
    assert len(ts.plan_tiles(h, w, K, 324)[1]) == 6
    kw, lights = _grid_case(h, w, K, None, 16.0, 2.0)
    _assert_grid_coverage(kw, lights)
    ref5 = _ref(kw)
    got5 = _tiled(kw)
    fails, _ = judge_lights(_total(ref5), _total(got5), lights, K)
    assert not fails, [(f[0]["kind"], f[0]["rf"], f[1]) for f in fails[:5]]
    res = tol.evaluate(_total(ref5), _total(got5), "B")
    assert res["ok"], tol.summary(res)


# --------------------------------------------------------------------------
# 4: the per-light judge catches engines without field character
# --------------------------------------------------------------------------

def _mutant_bank(kind):
    real = ts.slab_bank

    def mutant(d, **k):
        units = real(d, **k)
        kw = dict(k, kernels=k["kernels"])
        by = {(u.ri, u.a): u for u in units if u.bank and u.a is not None}

        def base(ri, dd=d):
            return cv._build_basis_kernel(
                ri, dd, True, kw["defocus_ratios"], kw["kernels"], kw["kH"],
                kw["kW"], kw["pix_scale"], kw["scaled_ksize"])

        def finish(u, rot):
            rot = np.maximum(rot, 0.0).astype(np.float32)
            rot /= rot.sum() + 1e-12
            u.kernel = rot
            u.kernel_bright = cv._bright_stamp(rot, k["bright_stamp_gain"])

        snapshot = {key: u.kernel.copy() for key, u in by.items()}
        for (ri, a), u in by.items():
            theta = (a + 0.5) * (2 * np.pi / cv.N_ANG) - np.pi
            if kind == "rot_sign":
                finish(u, cv.ndimage_rotate(base(ri), +(theta * 180 / np.pi + 90.0),
                                            reshape=False, order=1))
            elif kind == "one_wedge":
                src = snapshot.get((ri, 0))
                if src is not None:
                    finish(u, src.copy())
            elif kind == "wedge_off_by_one":
                src = snapshot.get((ri, (a + 1) % cv.N_ANG))
                if src is not None:
                    finish(u, src.copy())
            elif kind == "one_bin":
                finish(u, cv.ndimage_rotate(base(0), -(theta * 180 / np.pi + 90.0),
                                            reshape=False, order=1))
            elif kind == "one_slab":
                finish(u, cv.ndimage_rotate(base(ri, 5), -(theta * 180 / np.pi + 90.0),
                                            reshape=False, order=1))
            elif kind == "global_kernel":
                finish(u, base(6, 5))
            elif kind == "crop_psf":
                kk = u.kernel.copy()
                K = kk.shape[0]
                q = K // 4
                m = np.zeros_like(kk)
                m[q:K - q, q:K - q] = 1.0
                finish(u, kk * m)
            elif kind == "mirror_psf":
                finish(u, u.kernel[:, ::-1].copy())
            else:
                raise AssertionError(kind)
        return units

    return mutant


MUTANTS = ["rot_sign", "one_wedge", "wedge_off_by_one", "one_bin", "one_slab",
           "global_kernel", "crop_psf", "mirror_psf"]


@lru_cache(maxsize=None)
def _mutant_reference():
    kw, lights = _grid_case(301, 301, 17, 40, 16.0, 2.0)
    return kw, lights, _total(_ref(kw))


@pytest.mark.parametrize("kind", MUTANTS)
def test_per_light_judge_rejects_engines_without_field_character(monkeypatch, kind):
    kw, lights, ref_tot = _mutant_reference()
    monkeypatch.setattr(ts, "slab_bank", _mutant_bank(kind))
    got_tot = _total(_tiled(kw, _tile_n=40))
    monkeypatch.undo()
    fails, rows = judge_lights(ref_tot, got_tot, lights, 17)
    worst = max(r[0] for r in rows)
    print(f"mutant {kind}: {len(fails)}/{len(lights)} lights fail, worst max e {worst:.3g}")
    assert fails, f"mutant {kind} slipped through the per-light judge"
    # a field-character mutant is caught by orders of magnitude, not marginally
    floor = 10 if kind == "crop_psf" else 100       # crop: measured ~86x at K=17
    assert worst > floor * TIER_B["max_e"], (kind, worst)
    if kind in ("rot_sign", "one_wedge", "wedge_off_by_one", "one_bin",
                "mirror_psf"):
        # field-dependent damage reaches the corners and the wedge boundaries
        assert any(f[0]["rf"] > 0.9 for f in fails), kind
        assert any(f[0]["kind"] == "boundary" for f in fails), kind


# --------------------------------------------------------------------------
# 5: static and structural guarantees
# --------------------------------------------------------------------------

_FORBIDDEN_CALLS = {
    "zoom", "ndimage_zoom", "resize", "imresize", "rescale", "downscale_local_mean",
    "rot90", "flip", "fliplr", "flipud", "svd", "eig", "eigh", "gaussian_filter",
    "uniform_filter", "median_filter", "clip", "trim_zeros", "shift",
}


def _calls(mod):
    tree = ast.parse(inspect.getsource(mod))
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            names.append(f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", ""))
    return tree, names


@pytest.mark.parametrize("mod", [eng, ts], ids=["_engine_tiled", "_tile_schedule"])
def test_no_resample_crop_flip_or_low_rank_call_in_the_tiled_engine(mod):
    tree, names = _calls(mod)
    bad = sorted(set(names) & _FORBIDDEN_CALLS)
    assert not bad, f"{mod.__name__} calls {bad}"
    # no negative-step slicing (mirror-half storage / flips)
    for node in ast.walk(tree):
        if isinstance(node, ast.Slice) and node.step is not None:
            step = node.step
            neg = (isinstance(step, ast.UnaryOp) and isinstance(step.op, ast.USub))
            assert not neg, f"{mod.__name__}: negative-step slice at line {node.lineno}"
    # the angular count is the reference's N_ANG, never a literal
    src = inspect.getsource(mod)
    if mod is ts:
        assert "_cv.N_ANG" in src
    else:
        assert "N_ANG" not in src.replace("_cv.N_ANG", "")
    for lit in ("range(8)", "range(4)", "range(2)", "N_ANG // 2", "N_ANG//2"):
        assert lit not in src, f"{mod.__name__} contains {lit!r}"


def test_rotation_is_only_in_slab_bank_and_matches_reference_source():
    """The rot_deg / renormalise / stamp lines exist once in the tiled stack
    (``slab_bank``) and are the reference ``_apply_kernel`` lines verbatim."""
    eng_src = inspect.getsource(eng)
    ts_src = inspect.getsource(ts)
    assert "ndimage_rotate" not in eng_src
    assert ts_src.count("ndimage_rotate(") == 1
    ref_src = inspect.getsource(cv._apply_kernel)
    bank_src = inspect.getsource(ts.slab_bank)
    for line in (
        "theta_a = (a + 0.5) * (2 * np.pi / _cv.N_ANG) - np.pi",
        "rot_deg = -(theta_a * 180.0 / np.pi + 90.0)",
        "rotated = np.maximum(rotated, 0.0)",
        "rotated /= rotated.sum() + 1e-12",
        "rotated_bright = _cv._bright_stamp(rotated, bright_stamp_gain)",
        "is_delta = kernel_base.max() > 0.5 * kernel_base.sum()",
    ):
        assert line in bank_src, line
        assert line.replace("_cv.", "") in ref_src, line


def test_slab_bank_yields_every_bin_and_wedge_and_skips_only_by_global_maxima():
    """Structure of the bank on the shipped tensor: with full support every
    non-delta (d, ri) has N_ANG wedge units, every ri is visited, and a unit
    is dropped only by the reference's global 0.01 maxima."""
    t = _tensor("35")
    K, kH = 17, t["kernels"].shape[2]
    nr = t["kernels"].shape[0]
    common = dict(nr=nr, kernels=t["kernels"], defocus_ratios=t["defocus_ratios"],
                  kH=kH, kW=kH, pix_scale=K / kH, scaled_ksize=K,
                  bright_stamp_gain=2.0)
    full = np.ones((nr, cv.N_ANG + 1), bool)
    g1 = np.ones(nr, np.float32)
    g2 = np.ones((nr, cv.N_ANG), np.float32)
    for d in range(t["kernels"].shape[1]):
        units = ts.slab_bank(d, gmax_ri=g1, gmax_ria=g2, bank_flags=full, **common)
        assert sorted({u.ri for u in units}) == list(range(nr)), d
        for ri in range(nr):
            us = [u for u in units if u.ri == ri]
            if us[0].a is None:
                assert len(us) == 1
                assert us[0].kernel.shape == (K, K)
            else:
                assert [u.a for u in us] == list(range(cv.N_ANG)), (d, ri)
                ks = [u.kernel for u in us]
                assert all(k.shape == (K, K) for k in ks)
                assert all(abs(float(k.sum()) - 1.0) < 1e-5 for k in ks)
    # global maxima below 0.01 drop exactly those units, nothing more
    g1b = g1.copy()
    g1b[4] = 0.009
    g2b = g2.copy()
    g2b[7, 3] = 0.009
    units = ts.slab_bank(6, gmax_ri=g1b, gmax_ria=g2b, bank_flags=full, **common)
    keys = {(u.ri, u.a) for u in units}
    assert all(ri != 4 for ri, _ in keys)
    assert (7, 3) not in keys
    assert len(keys) == (nr - 1) * cv.N_ANG - 1
    # a non-bank unit still exists as a weight unit (weight sums intact)
    nb = full.copy()
    nb[2, 5] = False
    units = ts.slab_bank(6, gmax_ri=g1, gmax_ria=g2, bank_flags=nb, **common)
    u25 = [u for u in units if u.ri == 2 and u.a == 5]
    assert len(u25) == 1 and not u25[0].bank and u25[0].kernel is None
