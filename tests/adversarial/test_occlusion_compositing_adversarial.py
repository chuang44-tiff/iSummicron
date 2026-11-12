"""R-occlusion-layering (HALO-ONLY) — ADVERSARIAL hardening tests.

These tests attack the surfaces the Coder's ``test_occlusion_compositing.py``
suite does NOT cover. They target the premultiplied-OVER compositor in
``summicron.render._layered_composite`` (the entire ON branch):

  * wrong-polarity matte (MANDATORY per Day-1 post-mortem) — the inverted-
    opacity bug class;
  * selector behaviour on noisy / in-focus depth (the load-bearing real-LiDAR
    correctness risk);
  * connected hair-region near-zero-α (beyond the Coder's single-pixel probe);
  * OVER fold ordering proven via a swap fixture (not order-blind);
  * EDT disocclusion degeneracy (empty / full / single-valid mattes);
  * 3-slab junction coverage partition;
  * validator / cross-flag edges (occlusion × shadow-match × hdr);
  * OFF bit-id robustness under unrelated flag toggles.

Genuine bugs are marked ``@pytest.mark.xfail(reason="BUG: ...")``.

ALL tests run against the dev tree — verified to import
``summicron.render._layered_composite`` (the NEW module). Run with:
``PYTHONPATH=<repo> conda run -n isummicron pytest <this file> -p no:xdist``.
"""
from __future__ import annotations

import argparse
import hashlib

import numpy as np
import pytest

import summicron.render.convolve as _CV
import summicron.render._layered_composite as _LC
from summicron.render.convolve import render

# Guard: this whole file is meaningless if it silently imports a stale mirror.
assert _LC.__file__.endswith("summicron/render/_layered_composite.py"), (
    f"_layered_composite imported from unexpected path: {_LC.__file__}"
)
assert hasattr(_LC, "_layered_composite"), "NEW module missing _layered_composite"
assert hasattr(_LC, "compute_edge_selector"), "NEW module missing compute_edge_selector"


# --- Shared fixtures --------------------------------------------------------

_RE = np.array([0.0, 0.15, 0.30, 0.45, 0.60, 0.80, 1.01], dtype=np.float64)
_DR_D11 = np.linspace(0, 1, 11, dtype=np.float64)
_OD_D11 = np.array(
    [0.7, 0.98, 1.37, 1.91, 2.68, 3.74, 5.23, 7.32, 10.23, 14.30, 20.0],
    dtype=np.float64,
)


def _tensor(ksize: int = 33) -> dict:
    R, D = 6, 11
    kern = np.zeros((R, D, ksize, ksize), dtype=np.float32)
    c = ksize // 2
    kern[:, 0, c, c] = 1.0
    ys, xs = np.mgrid[-c:c + 1, -c:c + 1].astype(np.float32)
    for di in range(1, D):
        sigma = 1.5 + 1.5 * di
        g = np.exp(-(xs ** 2 + ys ** 2) / (2 * sigma ** 2))
        kern[:, di] = (g / g.sum()).astype(np.float32)
    return {
        "kernels": kern,
        "radial_edges": _RE.copy(),
        "defocus_ratios": _DR_D11.copy(),
        "object_distances_m": _OD_D11.copy(),
        "focus_distance_m": 0.7,
        "corpus_ref_long_edge": None,
        "otf_norm_applied": False,
        "delta_guard_applied": True,
    }


def _img(h=96, w=96, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.random((h, w, 3)) * 0.6 + 0.1).astype(np.float32)


def _over_unpremult(layers):
    """Replicate the helper's back-to-front premult OVER + Kosloff-Barsky
    un-premult on a list of ``(alpha (H,W), color (H,W,3))`` ordered near→far.

    Mirrors `_layered_composite` lines 430-439 EXACTLY:
        one_minus_acc = (1 - acc_a)[..., None]
        acc_c += one_minus_acc * (alpha * color)   # premultiplied layer
        acc_a += (1 - acc_a) * alpha
        out = acc_c / max(acc_a, 1e-3)
    """
    h, w = layers[0][0].shape
    acc_c = np.zeros((h, w, 3), np.float32)
    acc_a = np.zeros((h, w), np.float32)
    for la, lc in layers:
        oma = (1.0 - acc_a)[..., None]
        acc_c = acc_c + oma * (la[..., None] * lc)
        acc_a = acc_a + (1.0 - acc_a) * la
    out = acc_c / np.maximum(acc_a, np.float32(1e-3))[..., None]
    return out, acc_a


# ===========================================================================
# SURFACE 1 (MANDATORY) — wrong-polarity matte
# ===========================================================================


class TestWrongPolarityMatte:
    def test_correct_polarity_is_foreground(self):
        """Sanity anchor: with CORRECT polarity (foreground opaque) the OVER
        composite shows the foreground color, not the background."""
        h, w = 16, 16
        fg = np.zeros((h, w, 3), np.float32); fg[..., 0] = 1.0   # red FG
        bg = np.zeros((h, w, 3), np.float32); bg[..., 2] = 1.0   # blue BG
        out, acc_a = _over_unpremult(
            [(np.ones((h, w), np.float32), fg),   # near = foreground opaque
             (np.ones((h, w), np.float32), bg)]   # far  = background opaque
        )
        assert np.allclose(out[..., 0], 1.0) and np.allclose(out[..., 2], 0.0), (
            "correct-polarity composite must show the foreground (red), not BG"
        )
        assert np.allclose(acc_a, 1.0)

    def test_matte_partition_guard_fires_on_flipped_sense(self, monkeypatch):
        """R-occlusion round-2 (xfail removed — now GUARDED): the compositor
        gained an explicit matte-polarity/partition guard. ``_layered_composite``
        raises ``RuntimeError`` when the hard nearest-slab interior matte fails
        to partition the frame (Σ_d H_d ≢ 1) — exactly what a flipped matte
        sense (``argmin`` for ``argmax``, ``1 − H_d``, wrong polarity) produces.
        So a future refactor that inverts the matte fails LOUDLY instead of
        silently bleeding background through the subject (the Day-1 post-mortem
        bug class). We simulate the flip by corrupting the module's ``argmax``
        to a degenerate out-of-range map (Σ_d H_d collapses to 0) and assert
        the guard fires."""
        img = _img(96, 96, seed=3)
        h, w = img.shape[:2]
        # Genuine occlusion frame so regime-b (where the guard lives) is entered.
        depth = np.full((h, w), 12.0, np.float32)
        depth[h // 3:2 * h // 3, w // 3:2 * w // 3] = 0.7  # near subject block

        # WITHOUT corruption the guard does NOT fire (correct matte sense).
        on_u8, on_lin = render(
            img, _tensor(), depth_m=depth.copy(),
            occlusion_compositing="on",
        )
        assert np.isfinite(on_lin).all()

        # Corrupt the matte construction: force argmax→an out-of-range constant
        # so Σ_d (argmax==d) ≡ 0 (coverage holes everywhere). A real polarity
        # flip / wrong reduction breaks the partition the same way.
        import summicron.render._layered_composite as _LCmod
        orig_argmax = _LCmod.np.argmax

        def _broken_argmax(a, axis=None, **kw):
            r = orig_argmax(a, axis=axis, **kw)
            if axis == 0 and getattr(r, "shape", None) == (h, w):
                return np.full_like(r, a.shape[0])  # == D, out of range → holes
            return r

        monkeypatch.setattr(_LCmod.np, "argmax", _broken_argmax)
        with pytest.raises(RuntimeError, match="polarity/partition guard"):
            render(
                img, _tensor(), depth_m=depth.copy(),
                occlusion_compositing="on",
            )


# ===========================================================================
# SURFACE 2 — selector on noisy / in-focus depth (real-LiDAR risk)
# ===========================================================================


class TestSelectorNoisyDepth:
    def test_far_noisy_flat_not_silhouette(self):
        """A FAR flat surface (~3 m) with LiDAR-grade noise must NOT read as a
        silhouette — depth_ref normalisation keeps the ratio small. (Holds.)"""
        rng = np.random.default_rng(0)
        h, w = 96, 96
        depth = np.full((h, w), 3.0, np.float32) + rng.normal(0, 0.05, (h, w)).astype(np.float32)
        dmap = np.full((h, w), 0.4, np.float32)
        s = _LC.compute_edge_selector(depth, dmap, pad=16)
        assert float((s > 0.5).mean()) < 0.01, (
            f"far noisy-flat surface misclassified as silhouette: "
            f"frac(s>0.5)={float((s > 0.5).mean()):.3f}"
        )

    def test_close_infocus_noisy_flat_not_silhouette(self):
        """A FLAT in-focus 0.5 m subject with REALISTIC LiDAR noise must stay
        regime-a (s≈0).

        round-2 RE-SCOPE (xfail removed): the round-1 version of this test used
        ``rng.normal(0, 0.05)`` — 5 % pixel-to-pixel depth noise on a 0.5 m
        subject, ~10× higher than real iPhone LiDAR. The real-data probe
        (probe-round-1.md) measured the selector on the ACTUAL scaled LiDAR
        depth ``render()`` uses and found regime-b at 0.33 % / 0.0077 % — the
        selector does NOT over-fire on real depth. The 35 % over-fire was a
        synthetic-fixture artifact of the inflated noise σ (the ratio
        |∇depth|/depth_ref is scale-invariant, so ``depth_scale`` does not
        change it — the driver is the unrealistic relative noise). The locked
        spec's SECONDARY-2 instruction was explicit: do NOT retune the selector
        (real-data behaviour is correct); rewrite the fixture to realistic
        scale. With LiDAR-grade σ≈0.5 % the selector correctly stays quiet."""
        rng = np.random.default_rng(7)
        h, w = 96, 96
        # σ = 0.0025 m (0.5 % of 0.5 m) — representative iPhone LiDAR noise,
        # vs the round-1 0.05 m (10×-inflated) synthetic artifact.
        depth = np.full((h, w), 0.5, np.float32) + rng.normal(0, 0.0025, (h, w)).astype(np.float32)
        dmap = np.full((h, w), 0.02, np.float32)  # essentially in focus
        s = _LC.compute_edge_selector(depth, dmap, pad=16)
        assert float((s > 0.5).mean()) < 0.02, (
            f"flat in-focus realistic-noise subject mis-classified as "
            f"silhouette over {float((s > 0.5).mean()) * 100:.1f}% of pixels "
            f"(s.max={s.max():.3f})"
        )

    def test_real_silhouette_does_fire(self):
        """Control: a genuine 0.7→8 m depth step MUST drive s→1 at the edge,
        else the selector is dead and never composites occlusion."""
        h, w = 96, 96
        depth = np.full((h, w), 0.7, np.float32)
        depth[:, w // 2:] = 8.0
        dmap = np.full((h, w), 0.1, np.float32)
        s = _LC.compute_edge_selector(depth, dmap, pad=16)
        assert float(s.max()) > 0.9, (
            f"real silhouette failed to fire the selector (s.max={s.max():.3f})"
        )


# ===========================================================================
# SURFACE 3 — connected hair-region near-zero-α (beyond single pixel)
# ===========================================================================


class TestConnectedHairAlpha:
    def test_unpremult_connected_thin_region_finite(self):
        """A CONNECTED hair-like region (1-px column, full height) where the
        blurred α collapses toward 0 must not speckle/NaN/Inf-amplify through
        acc_c / max(acc_a, 1e-3)."""
        h, w = 64, 64
        alpha = np.zeros((h, w), np.float32)
        alpha[:, w // 2] = 1e-4  # whole connected column at near-zero coverage
        color = np.full((h, w, 3), 0.5, np.float32)
        out, acc_a = _over_unpremult([(alpha, color)])
        assert np.isfinite(out).all()
        # Clamp caps amplification at 1/1e-3; never explodes to inf.
        assert float(out.max()) <= 0.5 / 1e-3 + 1.0

    def test_full_render_connected_strand_finite(self):
        """End-to-end ON render with a connected 1-px near strand on a far
        background: no NaN/Inf, output bounded (no clamp blow-up)."""
        h, w = 96, 96
        img = _img(h, w, seed=2)
        depth = np.full((h, w), 18.0, np.float32)
        depth[:, w // 2:w // 2 + 1] = 0.7
        on_u8, on_lin = render(
            img, _tensor(), depth_m=depth.copy(),
            occlusion_compositing="on",
        )
        assert np.isfinite(on_lin).all()
        # No pixel should exceed input max by anything near the 1000× clamp.
        assert float(on_lin.max()) <= float(img.max()) + 0.2, (
            f"connected-strand un-premult amplified beyond input "
            f"(out.max={on_lin.max():.3f} vs img.max={img.max():.3f})"
        )


# ===========================================================================
# SURFACE 4 — OVER fold ordering (swap fixture proves not order-blind)
# ===========================================================================


class TestOverFoldOrdering:
    def test_near_opaque_occludes_far(self):
        """Near opaque layer fully occludes a far bright layer where α=1."""
        h, w = 16, 16
        near = np.zeros((h, w, 3), np.float32); near[..., 1] = 1.0   # green near
        far = np.full((h, w, 3), 0.9, np.float32)                    # bright far
        out, _ = _over_unpremult(
            [(np.ones((h, w), np.float32), near),   # near first (front-to-back)
             (np.ones((h, w), np.float32), far)]
        )
        assert np.allclose(out[..., 1], 1.0) and np.allclose(out[..., 0], 0.0), (
            "near opaque layer did not fully occlude the far bright layer"
        )

    def test_swap_order_changes_result(self):
        """Swap fixture: if the order is reversed (far in front), the result
        MUST differ — proves the OVER test above is not order-blind. A test
        that passed under arbitrary pixel rearrangement would be worthless."""
        h, w = 16, 16
        a = np.zeros((h, w, 3), np.float32); a[..., 1] = 1.0
        b = np.full((h, w, 3), 0.9, np.float32)
        near_first, _ = _over_unpremult(
            [(np.ones((h, w), np.float32), a), (np.ones((h, w), np.float32), b)]
        )
        far_first, _ = _over_unpremult(
            [(np.ones((h, w), np.float32), b), (np.ones((h, w), np.float32), a)]
        )
        assert not np.allclose(near_first, far_first), (
            "fold is order-blind — occlusion ordering is not respected"
        )
        # And the winner is whichever was composited first (front layer).
        assert np.allclose(near_first[..., 1], 1.0)
        assert np.allclose(far_first, 0.9)


# ===========================================================================
# SURFACE 5 — EDT disocclusion degeneracy
# ===========================================================================


class TestEdtDisocclusionDegeneracy:
    def test_all_foreground_no_hole(self):
        """matte all-1 (no background hole): fill is a no-op, finite."""
        img = _img(16, 16)
        m = np.ones((16, 16), np.float32)
        out = _LC._disocclusion_fill(m[..., None] * img, m)
        assert np.isfinite(out).all()
        assert np.array_equal(out, m[..., None] * img)

    def test_all_background_no_occluder(self):
        """matte all-0 (no occluder at all): no valid pixel to pull from →
        must not crash or fill garbage; returns the (all-zero) premult."""
        img = _img(16, 16)
        m = np.zeros((16, 16), np.float32)
        out = _LC._disocclusion_fill(m[..., None] * img, m)
        assert np.isfinite(out).all()
        assert np.allclose(out, 0.0)

    def test_single_valid_pixel(self):
        """Minimal support (one valid pixel): EDT must fill the whole frame
        from it without NaN/Inf."""
        img = _img(16, 16)
        m = np.zeros((16, 16), np.float32); m[8, 8] = 1.0
        out = _LC._disocclusion_fill(m[..., None] * img, m)
        assert np.isfinite(out).all()
        assert (out != 0).all(), "single-valid EDT left unfilled holes"


# ===========================================================================
# SURFACE 6 — 3-slab junction coverage partition
# ===========================================================================


class TestThreeSlabJunction:
    def test_hard_matte_partitions_at_junction(self):
        """Where three depth slabs meet at a point, the hard nearest-slab matte
        Σ_d H_d must still be exactly 1 everywhere (no pinhole coverage hole)."""
        h, w = 60, 60
        depth = np.full((h, w), 0.7, np.float32)
        depth[:h // 2, w // 2:] = 2.68
        depth[h // 2:, :] = 10.23
        slab_w = _CV.compute_slab_weights(depth, _OD_D11)
        argmax_d = np.argmax(slab_w, axis=0)
        interior = sum(
            (argmax_d == d).astype(np.float32) for d in range(slab_w.shape[0])
        )
        assert np.allclose(interior, 1.0, atol=1e-4), (
            "hard nearest-slab matte left a coverage hole at the 3-slab junction"
        )
        assert len(set(argmax_d.ravel().tolist())) >= 3, (
            "fixture failed to create a genuine 3-slab junction"
        )

    def test_pou_partition_at_junction(self):
        """PoU slab_w must also sum to 1 at the junction (regime-a coverage)."""
        h, w = 60, 60
        depth = np.full((h, w), 0.7, np.float32)
        depth[:h // 2, w // 2:] = 2.68
        depth[h // 2:, :] = 10.23
        slab_w = _CV.compute_slab_weights(depth, _OD_D11)
        assert np.allclose(slab_w.sum(0), 1.0, atol=1e-4)


# ===========================================================================
# SURFACE 7 — validator / cross-flag edges
# ===========================================================================


def _ns(**kw):
    ns = argparse.Namespace(
        focus_xy=None, focus_target_m=None,
        shadow_band_hi=0.7, shadow_match_max_gain=4.0, shadow_match_strength=1.0,
        slab_batch_size=None, occlusion_compositing="off",
        hdr_source="none", shadow_match_mode="off",
    )
    for k, v in kw.items():
        setattr(ns, k, v)
    return ns


class TestValidatorEdges:
    def test_invalid_occlusion_value_errors(self):
        p = argparse.ArgumentParser()
        with pytest.raises(SystemExit):
            _CV._validate_args(_ns(occlusion_compositing="maybe"), p)

    def test_occlusion_on_does_not_block_valid_config(self):
        """occlusion=on with valid hdr/shadow config must NOT error."""
        p = argparse.ArgumentParser()
        _CV._validate_args(_ns(occlusion_compositing="on"), p)  # no raise

    def test_render_invalid_mode_raises_valueerror(self):
        """render() itself (not just argparse) rejects an invalid mode —
        direct API callers can't slip past argparse choices."""
        img = _img()
        depth = np.broadcast_to(
            np.linspace(1, 18, 96)[None, :], (96, 96)
        ).astype(np.float32).copy()
        with pytest.raises(ValueError, match="occlusion_compositing"):
            render(img, _tensor(), depth_m=depth,
                   occlusion_compositing="ON")  # case-sensitive: not a choice

    def test_shadow_match_requirement_still_fires_with_occlusion_on(self):
        """occlusion=on must not mask the K11 shadow-match-mode/hdr invariant."""
        p = argparse.ArgumentParser()
        with pytest.raises(SystemExit):
            _CV._validate_args(
                _ns(occlusion_compositing="on",
                    shadow_match_mode="on", hdr_source="none"), p
            )


# ===========================================================================
# SURFACE 8 — OFF bit-id robustness under unrelated flag toggles
# ===========================================================================


class TestOffBitIdRobustness:
    def _md5(self, a):
        return hashlib.md5(np.ascontiguousarray(a).tobytes()).hexdigest()

    def test_off_md5_stable_across_slab_batch_size(self):
        """Toggling an UNRELATED flag (slab_batch_size) while occlusion is OFF
        must not perturb the OFF output bytes."""
        img = _img(seed=0)
        depth = np.broadcast_to(
            np.linspace(1, 18, 96)[None, :], (96, 96)
        ).astype(np.float32).copy()
        a, _ = render(img, _tensor(), depth_m=depth.copy(),
                      occlusion_compositing="off")
        b, _ = render(img, _tensor(), depth_m=depth.copy(),
                      occlusion_compositing="off", slab_batch_size=2)
        assert self._md5(a) == self._md5(b), "OFF bytes perturbed by unrelated flag"

    def test_off_equals_no_kwarg(self):
        """occlusion='off' is byte-identical to omitting the kwarg."""
        img = _img(seed=1)
        depth = np.broadcast_to(
            np.linspace(1, 18, 96)[None, :], (96, 96)
        ).astype(np.float32).copy()
        base, base_lin = render(img, _tensor(), depth_m=depth.copy())
        off, off_lin = render(img, _tensor(), depth_m=depth.copy(),
                              occlusion_compositing="off")
        assert np.array_equal(base, off)
        assert np.array_equal(base_lin, off_lin)


# ===========================================================================
# SURFACE 9 — fixture self-checks (would a pixel-shuffler still pass?)
# ===========================================================================


class TestEnergyCoverageInvariants:
    def test_on_uniform_input_mean_matches_off(self):
        """Real invariant (not pixel-shuffle-passable): on a uniform-luminance
        input + depth step, the ON composite's MEAN must track the OFF mean to
        within a thin selector-band tolerance. Catches an energy/coverage leak
        in the OVER path that a per-pixel-equality test would miss but a
        rearranging compositor would trip."""
        h, w = 96, 96
        img = np.full((h, w, 3), 0.5, np.float32)
        depth = np.full((h, w), 0.7, np.float32)
        depth[:, w // 2:] = 15.0
        off_u8, off_lin = render(img, _tensor(), depth_m=depth.copy(), occlusion_compositing="off")
        on_u8, on_lin = render(img, _tensor(), depth_m=depth.copy(), occlusion_compositing="on")
        assert abs(float(on_lin.mean()) - float(off_lin.mean())) < 0.02, (
            f"ON mean {on_lin.mean():.4f} drifted from OFF mean "
            f"{off_lin.mean():.4f} — energy/coverage leak in the OVER path"
        )
        assert np.isfinite(on_lin).all()
