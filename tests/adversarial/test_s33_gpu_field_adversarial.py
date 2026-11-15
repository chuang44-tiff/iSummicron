"""S33 Tier C adversarial (field lens): prove that field dependence survived
the MLX gpu engine.

Self-contained: no ``runs/``, no goldens, no IMG_6915, no other checkout. The
oracle is the in-tree reference gather (``convolve._layered_gather_diffuse``),
run in-test, and kernels rebuilt here from the reference's own primitives
(``_build_basis_kernel``, ``ndimage_rotate``, ``_bright_stamp``). Ceilings are
the Tier C ceilings from ``tests/fixtures/tolerance.py`` plus the R-C12
per-impulse metric (e_imp <= 1e-4 on the K x K box, L1 within 1e-5, centroid
within 0.01 px, tail sign equal). None is loosened.

This file attacks the GPU's OWN plumbing (the shared CPU planner and bank are
already attacked by ``test_s33_tiled_field_adversarial.py``):

1. Bank identity and pair -> kernel mapping: with every slab non-empty
   everywhere, the kernels the gpu engine transforms are sha256-equal to the
   reference's for every (slab, radial bin, wedge, diffuse|bright) label and in
   the reference's transform order; every executed GPU pair row points at the
   spectrum with ITS label, at the right image stack (img_padded for the delta
   unit, diffuse for wedges, bright for bright) and at its own tile.
2. Per-unit impulse THROUGH ``gpu_layered_gather``: a mosaic frame where each
   patch is pinned to one (radial bin, wedge) by a constant radial_frac /
   wedge map, one slab at a time, one impulse per patch. Each patch's output
   must be the reference's AND the rebuilt kernel of that label (diffuse and
   bright), under the production tile, forced multi-tile, host accumulation
   and split waves.
3. Real-geometry impulse grid: 12 radii x 16 azimuths, adjacent pixel pairs
   straddling every wedge boundary (the +-pi seam included), the centre pixel,
   the four true frame corners and corner lights at radial_frac > 0.9; odd and
   even frames, landscape and portrait, forced and production tiles.
4. Sensitivity: GPU-path mutants (wedge a+1 / a+4, radial bin shift in the
   pair weights, correlation instead of convolution, spectrum index off by
   one, rot_deg sign, a single global kernel) must FAIL the judge of (3).
5. Static: no code path in the gpu engine with fewer wedges, bins or slabs, a
   global kernel, or a resampled / cropped / flipped PSF; the gpu engine never
   builds or rotates a kernel itself.

The MLX tests are marked ``gpu`` and skip where mlx/Metal is absent, EXCEPT in
the dedicated mlx env, where they run (and fail) instead of skipping. The
static tests carry no marker and run in both envs.
"""
from __future__ import annotations

import ast
import hashlib
import inspect
import sys
import warnings
from functools import lru_cache

import numpy as np
import pytest

import summicron.render.convolve as cv
from summicron.render import _engine_gpu as eg
from summicron.render import _engine_tiled as et
from summicron.render import _tile_schedule as ts
from summicron.render._artifacts import default_tensor_path
from tests.fixtures import tolerance as tol

_GPU_REASON = eg.availability()
_IN_GPU_ENV = sys.prefix.rstrip("/").endswith("isummicron-gpu")
_C = tol.TIERS["C"]

E_IMP = 1e-4        # R-C12 per-impulse ceilings (spec G-C5a); never loosened
L1_REL = 1e-5
CENTROID_PX = 0.01


def _gpu_test(fn):
    fn = pytest.mark.skipif(_GPU_REASON is not None and not _IN_GPU_ENV,
                            reason=f"gpu engine unavailable: {_GPU_REASON}")(fn)
    return pytest.mark.gpu(fn)


# --------------------------------------------------------------------------
# builders (self-contained copies of what render() hands to the gather)
# --------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _tensor(lens: str) -> dict:
    return cv.load_tensor(default_tensor_path(lens))


def _rc(lens="35"):
    e = np.asarray(_tensor(lens)["radial_edges"])
    return 0.5 * (e[:-1] + e[1:])


def _geometry(h, w, radial_edges):
    ys = np.arange(h, dtype=np.float32) - np.float32((h - 1) / 2.0)
    xs = np.arange(w, dtype=np.float32) - np.float32((w - 1) / 2.0)
    max_radial = float(np.hypot((h - 1) / 2.0, (w - 1) / 2.0))
    rf = (np.hypot(ys[:, None], xs[None, :]) / np.float32(max_radial)).astype(np.float32)
    theta = np.arctan2(ys[:, None], xs[None, :])
    wedge = ((theta + np.pi) / (2 * np.pi) * cv.N_ANG).astype(np.int32) % cv.N_ANG
    rc = 0.5 * (radial_edges[:-1] + radial_edges[1:])
    return rf, wedge, rc


def _kw(img, depth, *, lens="35", K=17, thresh=1.0, gain=2.0, slab_w=None,
        rf=None, wedge=None):
    t = _tensor(lens)
    kernels = t["kernels"]
    kH, kW = kernels.shape[2], kernels.shape[3]
    H, W = img.shape[:2]
    pix_scale = 1.0 if K == kH else K / kH
    pad = K // 2
    rf0, wedge0, rc = _geometry(H, W, np.asarray(t["radial_edges"]))
    ip = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
    lum = 0.2126 * ip[..., 0] + 0.7152 * ip[..., 1] + 0.0722 * ip[..., 2]
    bright = ip * (lum > thresh).astype(np.float32)[..., None]
    if slab_w is None:
        slab_w = cv.compute_slab_weights(depth, t["object_distances_m"])
    return dict(
        depth_m=depth, slab_w=slab_w, img_lin=img,
        img_padded=ip, img_diffuse_padded=ip - bright, img_bright_padded=bright,
        kernels=kernels, defocus_ratios=t["defocus_ratios"],
        radial_frac_map=rf0 if rf is None else rf, r_centers=rc,
        wedge_idx=wedge0 if wedge is None else wedge,
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


def _gpu(kw, **extra):
    k = dict(kw)
    k["fft_state"] = None
    k["on_progress"] = None
    k.update(extra)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return eg.gpu_layered_gather(**k)


def _total(t5):
    return (t5[0] + t5[1]).astype(np.float32)


def _sha(a):
    return hashlib.sha256(np.ascontiguousarray(a, dtype=np.float32).tobytes()).hexdigest()


def _is_delta(base):
    return bool(base.max() > 0.5 * base.sum())


def _base(kw, d, ri):
    return cv._build_basis_kernel(
        ri, d, True, kw["defocus_ratios"], kw["kernels"], kw["kH"], kw["kW"],
        kw["pix_scale"], kw["scaled_ksize"])


def _expected_unit_kernels(kw, d, ri, a):
    """The reference's kernels for label (d, ri, a | None), rebuilt from the
    reference primitives (never through ``_tile_schedule``)."""
    base = _base(kw, d, ri)
    if a is None:
        return base, None
    theta_a = (a + 0.5) * (2 * np.pi / cv.N_ANG) - np.pi
    rot_deg = -(theta_a * 180.0 / np.pi + 90.0)
    rotated = cv.ndimage_rotate(base, rot_deg, reshape=False, order=1)
    rotated = np.maximum(rotated, 0.0)
    rotated /= rotated.sum() + 1e-12
    return rotated, cv._bright_stamp(rotated, kw["bright_stamp_gain"])


def _exact_parts(ref5, got5, what):
    for i, name in ((2, "weight_sum"), (3, "delta_weight_sum"), (4, "acc_alpha")):
        assert np.array_equal(ref5[i], got5[i]), f"{what}: {name} differs"


# --------------------------------------------------------------------------
# per-impulse judge (R-C12)
# --------------------------------------------------------------------------

def imp_metrics(ref, got, centers, K):
    """(e_imp, relative L1, centroid shift, tail sign equal) on each impulse's
    K x K box (clipped to the frame)."""
    h, w = ref.shape[:2]
    half = K // 2
    out = []
    for (y, x) in centers:
        ys, ye = max(0, y - half), min(h, y + half + 1)
        xs, xe = max(0, x - half), min(w, x + half + 1)
        r = ref[ys:ye, xs:xe].astype(np.float64)
        g = got[ys:ye, xs:xe].astype(np.float64)
        e_imp = float(np.abs(g - r).max() / max(float(np.abs(r).max()), 1e-30))
        l1 = abs(float(g.sum()) / float(r.sum()) - 1.0) if r.sum() else 0.0
        yy, xx = np.mgrid[ys:ye, xs:xe]
        rl, gl = r.sum(axis=2), g.sum(axis=2)
        if rl.sum() <= 0 or gl.sum() <= 0:
            out.append((e_imp, l1, float("inf"), False))
            continue
        cr = np.array([(yy * rl).sum() / rl.sum() - y, (xx * rl).sum() / rl.sum() - x])
        cg = np.array([(yy * gl).sum() / gl.sum() - y, (xx * gl).sum() / gl.sum() - x])
        shift = float(np.abs(cg - cr).max())
        sign_ok = all(np.sign(a) == np.sign(b) for a, b in zip(cr, cg) if abs(a) > 0.05)
        out.append((e_imp, l1, shift, sign_ok))
    return out


def judge(ref5, got5, centers, K):
    """Returns (failing indices, metric rows, Tier C result)."""
    ms = imp_metrics(_total(ref5), _total(got5), centers, K)
    fails = [i for i, m in enumerate(ms)
             if not (m[0] <= E_IMP and m[1] <= L1_REL and m[2] <= CENTROID_PX and m[3])]
    res = tol.evaluate(_total(ref5), _total(got5), "C")
    return fails, ms, res


def _worst(ms):
    return (max(m[0] for m in ms), max(m[1] for m in ms), max(m[2] for m in ms))


# --------------------------------------------------------------------------
# 1: bank identity and pair -> kernel mapping on the GPU
# --------------------------------------------------------------------------

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
    rng = np.random.default_rng(7)
    img = rng.uniform(0.0, 0.6, (h, w, 3)).astype(np.float32)
    img[h // 3, w // 3] = 4.0                  # one bright pixel: bright bank
    img[h - 3, w - 4] = 4.0                    # and one in the corner
    depth = np.full((h, w), np.float32(t["object_distances_m"][5]))
    slab_w = np.full((D, h, w), np.float32(1.0 / D), dtype=np.float32)
    return _kw(img, depth, lens=lens, K=K, slab_w=slab_w), D


@_gpu_test
@pytest.mark.parametrize("lens,K,n_tile,hw,extra", [
    ("35", 17, 40, (121, 121), {}),
    ("50", 17, 45, (121, 133), {"_tile_chunk": 1, "_pairs": 5, "_host_acc": True}),
    ("35", 67, None, (181, 181), {}),
])
def test_gpu_bank_identity_and_every_pair_points_at_its_own_label(
        monkeypatch, lens, K, n_tile, hw, extra):
    kw, D = _bank_scene(*hw, lens, K)
    ref_hashes = _capture_reference_hashes(kw, monkeypatch)
    ref5 = _ref(kw)

    events = []
    real_bank, real_spec, real_cols = ts.slab_bank, eg._gpu_kernel_spectra, eg._chunk_columns

    def spy_bank(d, **k):
        units = real_bank(d, **k)
        events.append(("bank", d, [(u.ri, u.a, u.bank) for u in units]))
        return units

    def spy_spec(arrays, n, labels=None):
        for a in arrays:                      # bank kernels enter unchanged: K x K float32
            assert a.ndim == 2 and a.dtype == np.float32
        events.append(("spec", [_sha(a) for a in arrays], list(labels)))
        return real_spec(arrays, n, labels)

    def spy_cols(units, chunk, dpairs, bpairs, srow, kidx_d, kidx_b, is_b):
        cols = real_cols(units, chunk, dpairs, bpairs, srow, kidx_d, kidx_b, is_b)
        events.append(("cols", list(chunk), dict(srow), is_b, cols))
        return cols

    monkeypatch.setattr(ts, "slab_bank", spy_bank)
    monkeypatch.setattr(eg, "_gpu_kernel_spectra", spy_spec)
    monkeypatch.setattr(eg, "_chunk_columns", spy_cols)
    got5 = _gpu(kw, **({"_tile_n": n_tile} if n_tile else {}), **extra)
    monkeypatch.undo()

    nr = len(kw["r_centers"])
    assert nr == 12 and cv.N_ANG == 8 and D == 11
    assert [e[1] for e in events if e[0] == "bank"] == list(range(D))

    gpu_hashes = []
    d = labels = None
    executed: dict = {}
    n_rows = 0
    for e in events:
        if e[0] == "bank":
            d = e[1]
            labels = None
            # full support: every (ri) is a delta unit or all 8 wedges, all banked
            by_ri: dict = {}
            for ri, a, bank in e[2]:
                assert bank, (d, ri, a)
                by_ri.setdefault(ri, []).append(a)
            assert sorted(by_ri) == list(range(nr)), d
            for ri, aa in by_ri.items():
                if _is_delta(_base(kw, d, ri)):
                    assert aa == [None], (d, ri, aa)
                else:
                    assert aa == list(range(cv.N_ANG)), (d, ri, aa)
        elif e[0] == "spec":
            hashes, labels = e[1], e[2]
            gpu_hashes += hashes
            for h, lab in zip(hashes, labels):
                ld, ri, a, kind = lab
                assert ld == d, (lab, d)
                want, want_b = _expected_unit_kernels(kw, d, ri, None if a == "delta" else a)
                assert h == _sha(want_b if kind == "bright" else want), lab
            # every non-delta (ri, a) has a diffuse AND a bright spectrum
            lab_set = set(map(tuple, labels))
            for ri in range(nr):
                if (d, ri, "delta", "diffuse") in lab_set:
                    continue
                for a in range(cv.N_ANG):
                    assert (d, ri, a, "diffuse") in lab_set and (d, ri, a, "bright") in lab_set
        else:
            _, chunk, srow, is_b, cols = e
            if cols is None:
                continue
            inv = {v: k for k, v in srow.items()}
            loc, tid, sr, kr, ri, a, rank = (np.asarray(c) for c in cols)
            for j in range(len(loc)):
                lab = labels[int(kr[j])]
                ld, lri, la, lkind = lab
                assert ld == d and lri == int(ri[j]), (lab, d, int(ri[j]))
                if int(a[j]) < 0:
                    assert la == "delta" and lkind == "diffuse" and not is_b, lab
                    want_kind = "f"
                else:
                    assert la == int(a[j]), (lab, int(a[j]))
                    assert lkind == ("bright" if is_b else "diffuse"), (lab, is_b)
                    want_kind = "b" if is_b else "d"
                skind, sti = inv[int(sr[j])]
                assert skind == want_kind and sti == int(tid[j]), (skind, sti, want_kind, int(tid[j]))
                assert chunk[int(loc[j])] == int(tid[j])
                executed.setdefault(d, set()).add((lri, la, lkind))
                n_rows += 1
    # same bank content and order as the reference actually transformed
    assert gpu_hashes == ref_hashes
    # every diffuse label that was transformed was also executed somewhere (the
    # noise covers the frame). Bright labels run only on tiles whose window
    # holds a bright pixel, exactly where the reference's bright product for
    # that unit is non-zero; at least the units at those lights must run.
    n_bright = 0
    for e in events:
        if e[0] == "spec":
            for ld, ri, a, kind in e[2]:
                if kind == "diffuse":
                    assert (ri, a, kind) in executed.get(ld, set()), (ld, ri, a, kind)
    for dd, s in executed.items():
        nb = sum(1 for _, _, k in s if k == "bright")
        if any(a != "delta" for _, a, _ in s):
            assert nb >= 1, dd
        n_bright += nb

    assert np.isfinite(got5[0]).all() and np.isfinite(got5[1]).all()
    res = tol.evaluate(_total(ref5), _total(got5), "C")
    assert res["ok"], tol.summary(res)
    _exact_parts(ref5, got5, "bank scene")
    print(f"lens {lens} K={K}: {len(gpu_hashes)} kernels == reference, "
          f"{n_rows} pair rows mapped to their own label; {tol.summary(res)}")


# --------------------------------------------------------------------------
# 2: per-unit impulse THROUGH the gpu gather (synthetic field maps)
# --------------------------------------------------------------------------

def mosaic_case(d, K, amp, *, lens="35", gain=2.0, thresh=1.0):
    """12 x 8 patches, patch (ri, a) pinned to radial bin ri and wedge a by
    constant maps, one single-pixel impulse at each patch centre, slab d only
    (one-hot slab weights). Patch side K + 7 so no kernel crosses a patch."""
    t = _tensor(lens)
    D = len(t["object_distances_m"])
    rc = _rc(lens).astype(np.float32)
    P = K + 7
    H, W = 12 * P, cv.N_ANG * P
    rf = np.repeat(np.repeat(rc[:, None], cv.N_ANG, 1), P, 0).repeat(P, 1).astype(np.float32)
    wedge = np.repeat(np.repeat(np.arange(cv.N_ANG, dtype=np.int32)[None, :], 12, 0),
                      P, 0).repeat(P, 1)
    img = np.zeros((H, W, 3), np.float32)
    centers = {}
    for ri in range(12):
        for a in range(cv.N_ANG):
            y, x = ri * P + P // 2, a * P + P // 2
            img[y, x] = amp
            centers[(ri, a)] = (y, x)
    slab_w = np.zeros((D, H, W), np.float32)
    slab_w[d] = 1.0
    depth = np.full((H, W), np.float32(t["object_distances_m"][d]))
    kw = _kw(img, depth, lens=lens, K=K, thresh=thresh, gain=gain, slab_w=slab_w,
             rf=rf, wedge=wedge)
    return kw, centers, P


def _mosaic_expected(kw, d, ri, a, amp, bright):
    base = _base(kw, d, ri)
    if _is_delta(base):
        return amp * base, True
    k, kb = _expected_unit_kernels(kw, d, ri, a)
    return amp * (kb if bright else k), False


def _run_mosaic(d, K, amp, gpu_extra, lens="35"):
    bright = amp > 1.0
    kw, centers, P = mosaic_case(d, K, amp, lens=lens)
    ref5 = _ref(kw)
    got5 = _gpu(kw, **gpu_extra)
    _exact_parts(ref5, got5, f"mosaic d={d}")
    ref_t, got_t = _total(ref5), _total(got5)
    assert np.isfinite(got_t).all()
    half = K // 2
    worst_ref = worst_exp = worst_leak = 0.0
    n_rot = n_disc = 0
    for (ri, a), (y, x) in centers.items():
        exp, is_delta = _mosaic_expected(kw, d, ri, a, amp, bright)
        g = got_t[y - half:y + half + 1, x - half:x + half + 1].astype(np.float64)
        r = ref_t[y - half:y + half + 1, x - half:x + half + 1].astype(np.float64)
        peak = float(exp.max())
        for c in range(3):
            e_ref = float(np.abs(g[..., c] - r[..., c]).max()) / max(float(r[..., c].max()), 1e-30)
            e_exp = float(np.abs(g[..., c] - exp).max()) / peak
            worst_ref, worst_exp = max(worst_ref, e_ref), max(worst_exp, e_exp)
            assert e_ref <= E_IMP, (d, ri, a, c, "vs reference", e_ref)
            assert e_exp <= E_IMP, (d, ri, a, c, "vs rebuilt kernel", e_exp)
        # nothing outside the K x K box inside the patch (no shifted crop)
        py0, px0 = ri * kw["H"] // 12, a * kw["W"] // cv.N_ANG
        patch = got_t[py0:py0 + P, px0:px0 + P].astype(np.float64).copy()
        patch[y - py0 - half:y - py0 + half + 1, x - px0 - half:x - px0 + half + 1] = 0.0
        leak = float(np.abs(patch).max()) / peak
        worst_leak = max(worst_leak, leak)
        assert leak <= E_IMP, (d, ri, a, "energy outside the kernel box", leak)
        # discrimination: the neighbour wedge's kernel would NOT pass
        if not is_delta:
            n_rot += 1
            alt, _ = _mosaic_expected(kw, d, ri, (a + 1) % cv.N_ANG, amp, bright)
            if float(np.abs(alt - exp).max()) > 10 * E_IMP * peak:
                n_disc += 1
                assert float(np.abs(g[..., 0] - alt).max()) / peak > E_IMP, (d, ri, a)
    res = tol.evaluate(ref_t, got_t, "C")
    assert res["ok"], tol.summary(res)
    return worst_ref, worst_exp, worst_leak, n_rot, n_disc


MOSAIC_CONFIGS = {
    "production_n": {},
    "forced_tiles_t29": {"_tile_n": 45},
    "host_acc_split_waves": {"_tile_n": 41, "_host_acc": True, "_pairs": 3, "_tile_chunk": 1},
}


@_gpu_test
@pytest.mark.parametrize("cfg", list(MOSAIC_CONFIGS))
@pytest.mark.parametrize("amp", [0.9, 4.0], ids=["diffuse", "bright"])
def test_per_unit_impulse_every_slab_bin_wedge_through_the_gpu_gather(cfg, amp):
    """All 11 slabs x 12 radial bins x 8 wedges at K=17: each unit's output
    through ``gpu_layered_gather`` is the reference's and the rebuilt kernel's
    for THAT label (diffuse below the threshold, the L1 x 2 stamp above)."""
    rows = []
    tot_rot = tot_disc = 0
    for d in range(11):
        wr, we, wl, n_rot, n_disc = _run_mosaic(d, 17, amp, MOSAIC_CONFIGS[cfg])
        rows.append((d, wr, we, wl))
        tot_rot += n_rot
        tot_disc += n_disc
    # rotated units really exist in >= 9 slabs and most are discriminable
    assert tot_rot >= 9 * 8 * 6, tot_rot
    assert tot_disc >= 0.5 * tot_rot, (tot_disc, tot_rot)
    print(f"mosaic K=17 {cfg} amp={amp}: worst vs ref {max(r[1] for r in rows):.3g}, "
          f"vs rebuilt kernel {max(r[2] for r in rows):.3g}, leak {max(r[3] for r in rows):.3g}; "
          f"{tot_rot} rotated units, {tot_disc} wedge-discriminable")


@_gpu_test
@pytest.mark.parametrize("lens,d", [("35", 10), ("35", 6), ("50", 10)])
def test_per_unit_impulse_k33_far_slabs(lens, d):
    """Larger kernels (K=33, production N) on far slabs where the PSFs carry
    the most structure, both lenses, bright stamp."""
    wr, we, wl, n_rot, n_disc = _run_mosaic(d, 33, 4.0, {}, lens=lens)
    assert n_rot == 12 * 8 or n_rot >= 10 * 8, n_rot
    print(f"mosaic K=33 lens {lens} d={d}: vs ref {wr:.3g}, vs kernel {we:.3g}, "
          f"leak {wl:.3g}, {n_disc}/{n_rot} discriminable")


# --------------------------------------------------------------------------
# 3: real-geometry impulse grid with boundary pairs, centre and corners
# --------------------------------------------------------------------------

def _depth_cycle(tensor):
    od = np.asarray(tensor["object_distances_m"], np.float64)
    nodes = [float(od[i]) for i in range(1, len(od))]
    mids = [float(np.sqrt(od[i] * od[i + 1])) for i in range(len(od) - 1)]
    return [x for p in zip(nodes, mids) for x in p]


def field_grid(h, w, *, lens="35", n_az=16):
    """Lights: 12 radii (the bin centres) x ``n_az`` azimuths; adjacent pixel
    pairs straddling each of the 8 wedge boundaries at 3 radii (the -pi/+pi
    seam on the left horizontal axis included); the centre pixel (or the 4
    pixels around the centre); the 4 true frame corners; 3 corner lights per
    corner at radial_frac 0.88 / 0.92 / 0.95 / 0.98 (4 per corner). Depth: nearest-light cells
    cycling over every slab node d1..d10 and between-node depths."""
    t = _tensor(lens)
    rc = _rc(lens)
    rf, wedge, _ = _geometry(h, w, np.asarray(t["radial_edges"]))
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    maxr = float(np.hypot(cy, cx))
    depths = _depth_cycle(t)
    lights, pos = [], set()

    def put(y, x, kind):
        y, x = int(y), int(x)
        if not (0 <= y < h and 0 <= x < w) or (y, x) in pos:
            return
        pos.add((y, x))
        lights.append(dict(y=y, x=x, kind=kind,
                           depth=depths[len(lights) % len(depths)]))

    for ai in range(n_az):
        th = -np.pi + (ai + 0.5) * (2 * np.pi / n_az)
        s, c = np.sin(th), np.cos(th)
        lim = min((cy - 1) / abs(s) if abs(s) > 1e-9 else np.inf,
                  (cx - 1) / abs(c) if abs(c) > 1e-9 else np.inf)
        for ri in range(12):
            R = rc[ri] * maxr
            if R > lim:
                R = lim * (0.55 + 0.45 * ri / 11)
            put(round(cy + R * s), round(cx + R * c), "ray")
    # boundary-straddling adjacent pairs: walk outward along each boundary ray
    for k in range(cv.N_ANG):
        th = -np.pi + k * (2 * np.pi / cv.N_ANG)
        s, c = np.sin(th), np.cos(th)
        for frac in (0.3, 0.6, 0.85):
            y0, x0 = int(round(cy + frac * min(cy, cx) * s)), int(round(cx + frac * min(cy, cx) * c))
            done = False
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    y, x = y0 + dy, x0 + dx
                    for ny, nx in ((y + 1, x), (y, x + 1)):
                        if (0 <= y < h and 0 <= x < w and 0 <= ny < h and 0 <= nx < w
                                and wedge[y, x] != wedge[ny, nx] and not done):
                            put(y, x, "boundary")
                            put(ny, nx, "boundary")
                            done = True
    for y in {int(np.floor(cy)), int(np.ceil(cy))}:
        for x in {int(np.floor(cx)), int(np.ceil(cx))}:
            put(y, x, "centre")
    for y, x in ((0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)):
        put(y, x, "corner_px")
    for sy, sx in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
        for f in (0.88, 0.92, 0.95, 0.98):
            put(round(cy + sy * f * cy), round(cx + sx * f * cx), "corner")
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
    edges = np.asarray(t["radial_edges"])
    for lt in lights:
        lt["rf"] = float(rf[lt["y"], lt["x"]])
        lt["ri"] = int(np.clip(np.searchsorted(edges, lt["rf"], side="right") - 1, 0, 11))
        lt["wedge"] = int(wedge[lt["y"], lt["x"]])
    return depth, lights


def _grid_kw(h, w, K, amp, gain, lens="35"):
    depth, lights = field_grid(h, w, lens=lens)
    img = np.zeros((h, w, 3), np.float32)
    for lt in lights:
        img[lt["y"], lt["x"]] = amp
    kw = _kw(img, depth, lens=lens, K=K, thresh=1.0, gain=gain)
    return kw, lights


def _assert_grid_coverage(kw, lights, h, w):
    assert {lt["ri"] for lt in lights} == set(range(12))
    assert {lt["wedge"] for lt in lights} == set(range(cv.N_ANG))
    assert sum(lt["rf"] > 0.9 for lt in lights) >= 12
    assert sum(lt["kind"] == "corner_px" for lt in lights) == 4
    bnd = [lt for lt in lights if lt["kind"] == "boundary"]
    # every one of the 8 boundaries (incl. the +-pi seam: wedge 7 | 0) straddled
    pairs = set()
    for i in range(0, len(bnd) - 1, 2):
        pairs.add(frozenset((bnd[i]["wedge"], bnd[i + 1]["wedge"])))
    assert len(pairs) == cv.N_ANG, pairs
    assert frozenset((7, 0)) in pairs
    D = kw["slab_w"].shape[0]
    sw_max = kw["slab_w"].reshape(D, -1).max(axis=1)
    assert all(sw_max[d] >= cv._EMPTY_SKIP_THRESH for d in range(1, D)), sw_max


GRID_CASES = [
    # (h, w, K, tile_n, amp, gain)  -- hard Tier C: lights <= 6.8x, gain <= 4
    (301, 301, 17, 40, 4.0, 2.0),      # odd square, centre pixel, forced seams
    (240, 320, 17, 44, 4.0, 4.0),      # even landscape, gain 4
    (321, 240, 17, None, 1.0, None),   # portrait LDR peak stamp, production N
    (300, 301, 17, 41, 6.0, 2.0),      # mixed parity, 6x lights
    (261, 261, 67, 96, 4.0, 2.0),      # K = 67 (the 2048 kernel side)
]


@_gpu_test
@pytest.mark.parametrize("h,w,K,n_tile,amp,gain", GRID_CASES)
def test_field_grid_per_light_crops_match_reference(h, w, K, n_tile, amp, gain):
    kw, lights = _grid_kw(h, w, K, amp, gain)
    _assert_grid_coverage(kw, lights, h, w)
    ref5 = _ref(kw)
    got5 = _gpu(kw, **({"_tile_n": n_tile} if n_tile else {}))
    if amp > 1.0:
        assert float(ref5[1].max()) > 0.01          # bright path really ran
    centers = [(lt["y"], lt["x"]) for lt in lights]
    fails, ms, res = judge(ref5, got5, centers, K)
    assert not fails, (
        f"{len(fails)} lights miss the per-impulse ceilings; first "
        f"{[(lights[i]['kind'], round(lights[i]['rf'], 3), lights[i]['wedge'], ms[i]) for i in fails[:5]]}")
    assert res["ok"], tol.summary(res)
    _exact_parts(ref5, got5, f"grid {h}x{w}")
    by_kind = {}
    for lt, m in zip(lights, ms):
        by_kind.setdefault(lt["kind"], []).append(m)
    print(f"{h}x{w} K={K} N={n_tile} amp={amp}: {len(lights)} lights; "
          + "; ".join(f"{k} ({len(v)}) e_imp {_worst(v)[0]:.3g} L1 {_worst(v)[1]:.3g} "
                      f"c {_worst(v)[2]:.3g}" for k, v in sorted(by_kind.items()))
          + f"; {tol.summary(res)}")


# --------------------------------------------------------------------------
# 4: GPU-path mutants must fail the judge
# --------------------------------------------------------------------------

def _mut_pair_weights(kind):
    real = eg._pair_weights

    def mutant(rilo, wlo, whi, wedge, tid, ri, a):
        import mlx.core as mx
        if kind.startswith("wedge+"):
            k = int(kind[6:])
            a = mx.where(a < 0, a, mx.remainder(a + k, cv.N_ANG)).astype(mx.int32)
        elif kind == "ri+1":
            ri = mx.minimum(ri + 1, 11).astype(mx.int32)
        return real(rilo, wlo, whi, wedge, tid, ri, a)

    return mutant


def _mut_spectra(kind):
    real = eg._gpu_kernel_spectra

    def mutant(arrays, n, labels=None):
        import mlx.core as mx
        out = real(arrays, n, labels)
        if kind == "conj":
            return mx.conjugate(out)
        return mx.roll(out, 1, axis=0)

    return mutant


def _mut_bank(kind):
    real = ts.slab_bank

    def mutant(d, **k):
        units = real(d, **k)
        for u in units:
            if not u.bank or u.a is None:
                continue
            base = cv._build_basis_kernel(
                u.ri, d, True, k["defocus_ratios"], k["kernels"], k["kH"], k["kW"],
                k["pix_scale"], k["scaled_ksize"])
            theta = (u.a + 0.5) * (2 * np.pi / cv.N_ANG) - np.pi
            if kind == "rot_sign":
                rot = cv.ndimage_rotate(base, +(theta * 180 / np.pi + 90.0),
                                        reshape=False, order=1)
            else:   # global kernel
                rot = cv._build_basis_kernel(
                    6, 5, True, k["defocus_ratios"], k["kernels"], k["kH"], k["kW"],
                    k["pix_scale"], k["scaled_ksize"])
            rot = np.maximum(rot, 0.0).astype(np.float32)
            rot /= rot.sum() + 1e-12
            u.kernel = rot
            u.kernel_bright = cv._bright_stamp(rot, k["bright_stamp_gain"])
        return units

    return mutant


MUTANTS = {
    "pair_wedge+1": (eg, "_pair_weights", lambda: _mut_pair_weights("wedge+1")),
    "pair_wedge+4": (eg, "_pair_weights", lambda: _mut_pair_weights("wedge+4")),
    "pair_ri+1": (eg, "_pair_weights", lambda: _mut_pair_weights("ri+1")),
    "spec_correlation": (eg, "_gpu_kernel_spectra", lambda: _mut_spectra("conj")),
    "spec_index_off_by_one": (eg, "_gpu_kernel_spectra", lambda: _mut_spectra("roll")),
    "bank_rot_sign": (ts, "slab_bank", lambda: _mut_bank("rot_sign")),
    "bank_global_kernel": (ts, "slab_bank", lambda: _mut_bank("global")),
}


@lru_cache(maxsize=None)
def _mutant_reference():
    kw, lights = _grid_kw(301, 301, 17, 4.0, 2.0)
    return kw, lights, _ref(kw)


@_gpu_test
@pytest.mark.parametrize("name", list(MUTANTS))
def test_gpu_mutants_without_field_character_fail_the_judge(monkeypatch, name):
    kw, lights, ref5 = _mutant_reference()
    mod, attr, make = MUTANTS[name]
    monkeypatch.setattr(mod, attr, make())
    got5 = _gpu(kw, _tile_n=40)
    monkeypatch.undo()
    centers = [(lt["y"], lt["x"]) for lt in lights]
    fails, ms, res = judge(ref5, got5, centers, 17)
    worst = max(m[0] for m in ms)
    print(f"mutant {name}: {len(fails)}/{len(lights)} lights fail, worst e_imp "
          f"{worst:.3g}; Tier C ok={res['ok']}")
    assert fails, f"mutant {name} slipped through the per-impulse judge"
    assert worst > 100 * E_IMP, (name, worst)
    if name in ("pair_wedge+1", "pair_wedge+4", "bank_rot_sign", "spec_correlation"):
        # field-dependent damage reaches the corners and the wedge boundaries
        assert any(lights[i]["rf"] > 0.9 for i in fails), name
        assert any(lights[i]["kind"] == "boundary" for i in fails), name


# --------------------------------------------------------------------------
# 5: static and structural guarantees (no mlx needed)
# --------------------------------------------------------------------------

_FORBIDDEN_CALLS = {
    "zoom", "ndimage_zoom", "resize", "imresize", "rescale", "downscale_local_mean",
    "rot90", "flip", "fliplr", "flipud", "svd", "eig", "eigh", "gaussian_filter",
    "uniform_filter", "median_filter", "clip", "trim_zeros", "shift", "roll",
    "ndimage_rotate", "rotate", "_build_basis_kernel", "_bright_stamp",
    "interpolate", "map_coordinates", "affine_transform",
}

_EG_TREE = ast.parse(inspect.getsource(eg))


def _call_name(node):
    f = node.func
    return f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")


def _func(name):
    for node in ast.walk(_EG_TREE):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(name)


def test_gpu_engine_has_no_resample_crop_flip_rotate_or_low_rank_call():
    """The only forbidden-looking call is the radial bracket's index clamp in
    ``_compact_basis``, which is the reference ``_radial_basis_weights`` line
    verbatim (it clamps a bin INDEX, not PSF values)."""
    src = inspect.getsource(eg)
    bad = []
    for node in ast.walk(_EG_TREE):
        if isinstance(node, ast.Call) and _call_name(node) in _FORBIDDEN_CALLS:
            bad.append((_call_name(node), node.lineno))
    clips = [b for b in bad if b[0] == "clip"]
    assert [b for b in bad if b[0] != "clip"] == [], bad
    assert len(clips) == 1, clips
    cb = _func("_compact_basis")
    assert cb.lineno <= clips[0][1] <= cb.end_lineno
    line = src.splitlines()[clips[0][1]]          # the line after "np.clip("
    assert line.strip() == 'np.searchsorted(rc, rf, side="right") - 1, 0, nr - 2'
    ref_src = inspect.getsource(cv._radial_basis_weights)
    assert 'np.searchsorted(rc, rf, side="right") - 1, 0, Nr - 2' in ref_src
    # no negative-step slicing (mirror-half storage / flips)
    for node in ast.walk(_EG_TREE):
        if isinstance(node, ast.Slice) and node.step is not None:
            raise AssertionError(f"stepped slice at line {node.lineno}")


def test_gpu_engine_never_builds_a_kernel_and_has_one_bank_source():
    """Kernels come only from ``ts.slab_bank`` (the shared bank) and spectra
    only from the GPU helper ``_gpu_kernel_spectra`` (never the CPU
    ``et._kernel_spectra``); the engine never rotates, stamps,
    resamples or transforms a kernel itself."""
    src = inspect.getsource(eg)
    for name in ("ndimage_rotate", "_build_basis_kernel", "_bright_stamp", "rot_deg",
                 "theta_a", "ndimage", "zoom"):
        assert name not in src, name
    calls = [(_call_name(n), n.lineno) for n in ast.walk(_EG_TREE) if isinstance(n, ast.Call)]
    assert sum(c == "slab_bank" for c, _ in calls) == 1
    assert not [c for c, _ in calls if c == "_kernel_spectra"]   # CPU float64 path
    spec_calls = [ln for c, ln in calls if c == "_gpu_kernel_spectra"]
    gb, tc = _func("_gather_body"), _func("gpu_tile_convolve_full")
    assert len(spec_calls) == 2
    assert sum(gb.lineno <= ln <= gb.end_lineno for ln in spec_calls) == 1
    assert sum(tc.lineno <= ln <= tc.end_lineno for ln in spec_calls) == 1
    # inside the gather, kernel spectra are built from u.kernel / u.kernel_bright only
    seg = ast.get_source_segment(inspect.getsource(eg), gb)
    assert "arrays.append(u.kernel)" in seg and "arrays.append(u.kernel_bright)" in seg
    assert seg.count("arrays.append(") == 3
    # the slab_bank call passes the reference geometry straight through
    for n in ast.walk(gb):
        if isinstance(n, ast.Call) and _call_name(n) == "slab_bank":
            kws = {k.arg: ast.unparse(k.value) for k in n.keywords}
            assert ast.unparse(n.args[0]) == "d"
            assert kws["nr"] == "nr" and kws["kernels"] == "kernels"
            assert kws["pix_scale"] == "pix_scale" and kws["scaled_ksize"] == "scaled_ksize"
            assert kws["bright_stamp_gain"] == "bright_stamp_gain"


def test_gpu_engine_counts_are_the_references():
    """Wedges come from ``_cv.N_ANG``, bins from ``len(r_centers)``, slabs from
    ``slab_w.shape[0]``; no literal smaller count anywhere."""
    src = inspect.getsource(eg)
    assert "NA = _cv.N_ANG" in src
    assert "N_ANG" not in src.replace("_cv.N_ANG", "")
    assert "nr = int(len(r_centers))" in src
    assert "D_gate = slab_w.shape[0]" in src
    assert "for d in range(D_gate):" in src
    for lit in ("range(8)", "range(4)", "range(2)", "N_ANG // 2", "N_ANG//2",
                "% 8", "% 4", "NA // 2", "nr // 2", "D_gate // 2 +", "[::2]"):
        assert lit not in src, lit
    # every skip is one the reference makes (empty slab, zero slab weight in
    # the tile, non-bank unit, unused image stack, unit without support in the
    # tile); none drops a slab, bin or wedge by count
    guards = sorted(
        ast.unparse(n.test) for n in ast.walk(_EG_TREE)
        if isinstance(n, ast.If) and len(n.body) == 1 and isinstance(n.body[0], ast.Continue))
    assert guards == sorted([
        "float(sw_d.max()) < _cv._EMPTY_SKIP_THRESH", "not tile_has_sw[ti]",
        "not u.bank", "not u.bank", "not need", "not stats.bpos[ti, u.ri]",
        "not stats.wpos[ti, u.ri, u.a]"]), guards
    breaks = [n for n in ast.walk(_EG_TREE) if isinstance(n, ast.Break)]
    assert not breaks, [b.lineno for b in breaks]


def test_gpu_gather_accepts_every_reference_argument_and_render_passes_them():
    ref = inspect.signature(cv._layered_gather_diffuse).parameters
    gpu = inspect.signature(eg.gpu_layered_gather).parameters
    missing = [p for p in ref if p not in gpu]
    assert not missing, missing
    extra = [p for p in gpu if p not in ref]
    assert all(p.startswith("_") or p == "threads" for p in extra), extra
    # render() calls the selected gather with ONE argument list for all engines
    rsrc = inspect.getsource(cv.render)
    assert rsrc.count("_gather_fn(") == 1
    assert "_gather_fn = gpu_layered_gather" in rsrc
