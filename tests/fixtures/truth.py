"""Float64 truth and the ruled stress ratio (S33 Tier B, user ruling R-B2 revised).

The hard Tier B ceilings apply to real scenes and the standard charts at L1
gain <= 4. The STRESS family (L1 gain >= 8, HDR 16x on dense light fields,
50x / 100x HDR lights, the peak stamp on HDR, LDR peak with large patches, lens
50 LDR peak on dense small frames) is gated by

    (tiled error vs float64 truth) / (reference error vs the same truth) <= 1.5

on p99.9 of e, over the full frame AND the field-edge annulus. Only this p99.9
ratio is hard (final rule, ruling 2026-11-13 r3). The per-pixel absolute max and
the single-pixel max ratio are printed, not gated; the ``abs_max=True`` gate is
opt-in only and unused by the committed stress tests.
(same ``e = |d| / max(truth, 1)`` arithmetic as the Tier B metric). The raw Tier B
numbers (tiled vs reference) are printed alongside. The ratio is hard.
"""
from __future__ import annotations

import numpy as np
import scipy.fft as sp_fft

import summicron.render.convolve as cv
from summicron.render import _tile_schedule as ts
from tests.fixtures import tolerance as tol

RATIO_CEILING = 1.5
ABS_MAX_CEILING = tol.TIERS["B"]["max_e"]   # 1e-5
_FLOOR = 1e-9            # reference error floor so a perfect reference cannot divide by 0


def truth_gather(kw):
    """Float64 re-computation of the OFF gather for hole-free scenes: same
    kernel bank (``ts.slab_bank`` arithmetic), same weights and skips, but all
    convolutions in float64 through one-shot FFTs. Returns (diffuse, bright)."""
    H, W, K = kw["H"], kw["W"], int(kw["scaled_ksize"])
    ip = kw["img_padded"].astype(np.float64)
    idf = kw["img_diffuse_padded"].astype(np.float64)
    ibr = kw["img_bright_padded"].astype(np.float64)
    hp, wp = ip.shape[:2]
    sz = (sp_fft.next_fast_len(hp + K - 1), sp_fft.next_fast_len(wp + K - 1))
    Sf = sp_fft.rfft2(ip, s=sz, axes=(0, 1))
    Sd = sp_fft.rfft2(idf, s=sz, axes=(0, 1))
    Sb = sp_fft.rfft2(ibr, s=sz, axes=(0, 1))

    def conv(S, ker):
        Kf = sp_fft.rfft2(ker.astype(np.float64), s=sz)
        full = sp_fft.irfft2(S * Kf[..., None], s=sz, axes=(0, 1))
        return full[K - 1:K - 1 + H, K - 1:K - 1 + W]

    basis = cv._radial_basis_weights(kw["radial_frac_map"], kw["r_centers"])
    wedge = kw["wedge_idx"]
    nr = basis.shape[0]
    out = np.zeros((H, W, 3))
    out_b = np.zeros((H, W, 3))
    for d in range(kw["slab_w"].shape[0]):
        sw = kw["slab_w"][d]
        if float(sw.max()) < cv._EMPTY_SKIP_THRESH:
            continue
        acc = np.zeros((3, H, W))
        acc_b = np.zeros((3, H, W))
        wsum = np.zeros((H, W), np.float32)
        for ri in range(nr):
            if basis[ri].max() < 0.01:
                continue
            kb = cv._build_basis_kernel(
                ri, d, True, kw["defocus_ratios"], kw["kernels"],
                kw["kH"], kw["kW"], kw["pix_scale"], kw["scaled_ksize"])
            if kb.max() > 0.5 * kb.sum():
                w = basis[ri].astype(np.float32)
                wsum += w
                acc += np.moveaxis(conv(Sf, kb), -1, 0) * w
                continue
            for a in range(cv.N_ANG):
                w = ts.wedge_weight(basis[ri], wedge, a)
                if w.max() < 0.01:
                    continue
                wsum += w
                theta_a = (a + 0.5) * (2 * np.pi / cv.N_ANG) - np.pi
                rot = cv.ndimage_rotate(kb, -(theta_a * 180.0 / np.pi + 90.0),
                                        reshape=False, order=1)
                rot = np.maximum(rot, 0.0)
                rot /= rot.sum() + 1e-12
                rb = cv._bright_stamp(rot, kw["bright_stamp_gain"])
                acc += np.moveaxis(conv(Sd, rot), -1, 0) * w
                acc_b += np.moveaxis(conv(Sb, rb), -1, 0) * w
        wsafe = np.maximum(wsum, 1e-8).astype(np.float64)
        al = sw.astype(np.float64)
        out += np.moveaxis(acc / wsafe, 0, -1) * al[..., None]
        out_b += np.moveaxis(acc_b / wsafe, 0, -1) * al[..., None]
    return out, out_b


def _err(total, truth, mask):
    t = truth if mask is None else truth[mask]
    x = total if mask is None else total[mask]
    e = np.abs(x.astype(np.float64) - t) / np.maximum(t, 1.0)
    return float(e.max()), float(np.percentile(e, 99.9))


def stress_ratio(kw, ref5, got5):
    """Ratios of (tiled vs truth) / (reference vs truth) for max e and p99.9,
    full frame and annulus. Returns ``{region: {'ref': (max, p999), 'til':
    (max, p999), 'ratio': (max, p999)}}``."""
    td, tb = truth_gather(kw)
    truth = td + tb
    ref = (ref5[0] + ref5[1]).astype(np.float32)
    til = (got5[0] + got5[1]).astype(np.float32)
    h, w = truth.shape[:2]
    out = {}
    for reg, mask in (("full", None), ("edge", tol.annulus_mask(h, w))):
        r = _err(ref, truth, mask)
        t = _err(til, truth, mask)
        out[reg] = dict(ref=r, til=t, ratio=tuple(t[i] / max(r[i], _FLOOR) for i in (0, 1)))
    return out


def format_ratio(res):
    return "; ".join(
        f"{reg}: ref(max,p99.9)=({v['ref'][0]:.3g},{v['ref'][1]:.3g}) "
        f"til=({v['til'][0]:.3g},{v['til'][1]:.3g}) "
        f"ratio=({v['ratio'][0]:.3g},{v['ratio'][1]:.3g})"
        for reg, v in res.items())


def assert_stress(kw, ref5, got5, what, abs_max=False):
    """The hard stress gate. Also prints the raw Tier B numbers. The exact
    accumulators (weight_sum, delta_weight_sum, acc_alpha) must be bit-equal."""
    for i, name in ((2, "weight_sum"), (3, "delta_weight_sum"), (4, "acc_alpha")):
        assert np.array_equal(ref5[i], got5[i]), f"{what}: {name} differs"
    tb = tol.summary(tol.evaluate((ref5[0] + ref5[1]).astype(np.float32),
                                  (got5[0] + got5[1]).astype(np.float32), "B"))
    res = stress_ratio(kw, ref5, got5)
    print(f"[stress] {what}\n  raw Tier B: {tb}\n  vs truth: {format_ratio(res)}")
    for reg, v in res.items():
        # HARD (user ruling, 2026-11-13 r3, final): ONLY the p99.9 ratio <= 1.5.
        # The per-pixel max (absolute e and max ratio) is printed above, not
        # asserted. abs_max=True is opt-in for callers that want the old gate.
        assert v["ratio"][1] <= RATIO_CEILING, (
            f"{what}: {reg} p99.9 ratio {v['ratio'][1]:.3g} > {RATIO_CEILING}: "
            f"{format_ratio(res)}")
        assert (not abs_max) or v["til"][0] <= ABS_MAX_CEILING, (
            f"{what}: {reg} absolute max e {v['til'][0]:.3g} > {ABS_MAX_CEILING}: "
            f"{format_ratio(res)}")
    return res
