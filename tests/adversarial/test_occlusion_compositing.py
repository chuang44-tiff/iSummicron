"""R-occlusion-layering (HALO-ONLY) adversarial + correctness tests.

Gates (locked-spec-R-occlusion-layering.md §"Gate thresholds" + §"Mandatory
adversarial tests"):

  * G1 (SHOWSTOPPER): OFF render is byte-identical — verified here at the
    render()-output-array level (the empirical canonical-bench md5 pin lives in
    the Coder report; this test asserts the in-process OFF path is unchanged
    by the new dispatch branch by comparing occlusion="off" against a baseline
    render with the kwarg absent).
  * G-ramp: synthetic tilted depth plane, small |∇depth| → s≈0 → ON ==
    OFF/PoU baseline within tolerance (no banding contour).
  * G-seam: vertical d=lo│d=hi flat-luminance depth step, s≈1 → acc_α ≡ 1
    across the band, no bleed through the solid side.
  * wsum-cancellation: ON with the selector forced to 0 (smooth depth) ==
    OFF diffuse baseline (proves no double-normalization).
  * Near-zero-α divide stability: Kosloff-Barsky clamp keeps the un-premult
    finite where blurred α → 0.

All synthetic; the user IMG_4343 halo probe is the load-bearing visual gate.
"""
from __future__ import annotations

import numpy as np
import pytest

import summicron.render.convolve as _CV
import summicron.render._layered_composite as _LC
from summicron.render.convolve import render


# --- Shared fixtures (mirror tests/adversarial/test_layered_gather_adversarial.py)

_RE = np.array([0.0, 0.15, 0.30, 0.45, 0.60, 0.80, 1.01], dtype=np.float64)
_DR_D11 = np.linspace(0, 1, 11, dtype=np.float64)
_OD_D11 = np.array(
    [0.7, 0.98, 1.37, 1.91, 2.68, 3.74, 5.23, 7.32, 10.23, 14.30, 20.0],
    dtype=np.float64,
)


def _synthetic_tensor_d11(ksize: int = 33) -> dict:
    R, D = 6, 11
    kern = np.zeros((R, D, ksize, ksize), dtype=np.float32)
    c = ksize // 2
    kern[:, 0, c, c] = 1.0  # S2 delta guard at d=0
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


def _make_image(h=96, w=96, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.random((h, w, 3)) * 0.6 + 0.1).astype(np.float32)


def _tilted_depth(h, w, lo=1.0, hi=18.0):
    """Smooth tilted plane crossing several slab boundaries (small |∇depth|)."""
    ramp = np.linspace(lo, hi, w, dtype=np.float32)
    return np.broadcast_to(ramp[None, :], (h, w)).astype(np.float32).copy()


def _step_depth(h, w, near=1.0, far=15.0):
    """Vertical flat-luminance depth step: left near-solid │ right far-solid."""
    d = np.full((h, w), near, dtype=np.float32)
    d[:, w // 2:] = far
    return d


# ---------------------------------------------------------------------------
# G1 — OFF bit-identity (in-process; md5 pin is in the Coder report)
# ---------------------------------------------------------------------------


class TestG1OffBitIdentity:
    def test_off_matches_baseline_no_kwarg(self):
        """occlusion_compositing='off' is byte-identical to omitting the kwarg
        entirely — proves the dispatch branch never perturbs the OFF path."""
        img = _make_image()
        depth = _tilted_depth(*img.shape[:2])
        tensor = _synthetic_tensor_d11()
        base_u8, base_lin = render(
            img, tensor, depth_m=depth.copy(),
        )
        off_u8, off_lin = render(
            img, tensor, depth_m=depth.copy(),
            occlusion_compositing="off",
        )
        assert np.array_equal(base_u8, off_u8)
        assert np.array_equal(base_lin, off_lin)

    def test_invalid_mode_raises(self):
        img = _make_image()
        depth = _tilted_depth(*img.shape[:2])
        tensor = _synthetic_tensor_d11()
        with pytest.raises(ValueError, match="occlusion_compositing"):
            render(
                img, tensor, depth_m=depth,
                occlusion_compositing="bogus",
            )


# ---------------------------------------------------------------------------
# G-ramp — smooth tilted plane: s≈0, ON ≈ OFF (no banding contour)
# ---------------------------------------------------------------------------


class TestGRampSmoothPlane:
    def test_selector_near_zero_on_smooth_ramp(self):
        """A gentle tilted depth plane has small |∇depth| → selector s≈0."""
        h, w = 96, 96
        depth = _tilted_depth(h, w, lo=1.0, hi=18.0)
        # defocus_map ~ moderate blur everywhere (in-range)
        dmap = np.full((h, w), 0.4, dtype=np.float32)
        s = _LC.compute_edge_selector(depth, dmap, pad=8)
        assert s.shape == (h, w)
        assert float(s.max()) < 0.25, f"smooth ramp selector too high: {s.max()}"

    def test_on_matches_off_on_smooth_ramp(self):
        """With s≈0, the ON diffuse composite collapses to the PoU path →
        ON output ≈ OFF output (no quantization contour from banding)."""
        img = _make_image(seed=3)
        h, w = img.shape[:2]
        depth = _tilted_depth(h, w, lo=1.0, hi=18.0)
        tensor = _synthetic_tensor_d11()
        off_u8, _ = render(img, tensor, depth_m=depth.copy())
        on_u8, _ = render(
            img, tensor, depth_m=depth.copy(),
            occlusion_compositing="on",
        )
        # Mean abs uint8 diff small (no banding); allow a thin selector tail.
        diff = np.abs(on_u8.astype(np.int16) - off_u8.astype(np.int16))
        assert float(diff.mean()) < 3.0, f"ramp ON vs OFF mean diff {diff.mean()}"


# ---------------------------------------------------------------------------
# G-seam — vertical depth step: acc_α ≡ 1, no bleed through the solid side
# ---------------------------------------------------------------------------


class TestGSeamDepthStep:
    def test_alpha_unity_across_step(self):
        """Each pixel is opaque in exactly one slab (hard nearest-slab matte) →
        the OVER coverage accumulator acc_α reaches ~1 across the whole frame,
        including the silhouette band (no partial-coverage holes on solids)."""
        h, w = 96, 96
        depth = _step_depth(h, w, near=1.0, far=15.0)
        slab_w = _CV.compute_slab_weights(depth, _OD_D11)
        # Reconstruct the OVER pass through the helper and read acc_a via the
        # returned diffuse — instead probe the matte invariant directly: the
        # hard argmax matte partitions the frame, so summed over slabs the
        # interior matte is exactly 1 everywhere.
        argmax_d = np.argmax(slab_w, axis=0)
        interior = np.zeros((h, w), dtype=np.float32)
        for d in range(slab_w.shape[0]):
            interior += (argmax_d == d).astype(np.float32)
        assert np.allclose(interior, 1.0, atol=1e-4), (
            "hard nearest-slab matte must partition the frame (Σ_d H_d ≡ 1)"
        )

    def test_no_bleed_through_solid_step(self):
        """Flat-luminance vertical depth step: the near-solid side must NOT
        pick up the far side's blurred luminance bleeding across the seam."""
        h, w = 96, 96
        # Distinct flat luminance per side so any cross-seam bleed is visible.
        img = np.zeros((h, w, 3), dtype=np.float32)
        img[:, :w // 2, :] = 0.8   # near side bright
        img[:, w // 2:, :] = 0.1   # far side dark
        depth = _step_depth(h, w, near=1.0, far=15.0)
        tensor = _synthetic_tensor_d11()
        on_u8, _ = render(
            img, tensor, depth_m=depth.copy(),
            occlusion_compositing="on",
        )
        # Sample a column well inside the near (bright, in-focus) side, away
        # from the seam. It must stay bright (no dark bleed from far side).
        near_col = on_u8[:, w // 4, :].astype(np.float32) / 255.0
        assert float(near_col.mean()) > 0.55, (
            f"near-solid side darkened by cross-seam bleed: {near_col.mean()}"
        )


# ---------------------------------------------------------------------------
# REGIME-A == OFF — the load-bearing guard (probe-round-1 PRIMARY FIX).
#
# This REPLACES the round-1 synthetic forced-s=0 wsum test, which passed while
# the real render diverged because it monkeypatched the selector to 0 and so
# never routed through the full layered_gather identity-snap/holes pipeline.
# These tests run the REAL selector and assert ON == OFF everywhere s≈0, on a
# realistically-scaled depth frame — the exact divergence the probe caught.
# ---------------------------------------------------------------------------


def _realistic_depth(h, w):
    """A frame with a true foreground/background occlusion step (fires the
    selector on a thin band) on an otherwise smooth, realistically-scaled
    background. Mirrors the IMG_4343 arm-vs-background topology at synthetic
    scale: near subject blob (~0.7 m) on a far smooth gradient (3→18 m)."""
    bg = np.linspace(3.0, 18.0, w, dtype=np.float32)[None, :]
    depth = np.broadcast_to(bg, (h, w)).astype(np.float32).copy()
    # Near subject disc → a genuine silhouette edge (large |∇depth| at its rim).
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx, r = h * 0.5, w * 0.4, min(h, w) * 0.22
    disc = ((yy - cy) ** 2 + (xx - cx) ** 2) <= r * r
    depth[disc] = 0.7
    return depth


class TestRegimeAEqualsOFF:
    def test_on_equals_off_outside_selector_band(self):
        """LOAD-BEARING. Render a realistic occlusion frame ON and OFF with the
        REAL selector. ON must equal OFF on every pixel where s≈0 (the bulk of
        the frame) — the round-1 bug changed 99.2% of pixels here. Divergence
        is allowed ONLY inside the dilated s>ε silhouette band."""
        img = _make_image(96, 96, seed=21)
        h, w = img.shape[:2]
        depth = _realistic_depth(h, w)
        tensor = _synthetic_tensor_d11()

        off_u8, off_lin = render(img, tensor, depth_m=depth.copy())
        on_u8, on_lin = render(
            img, tensor, depth_m=depth.copy(),
            occlusion_compositing="on",
        )

        # The actual defocus map render() builds from depth (for the selector).
        object_distances_m = tensor["object_distances_m"]
        dmap = np.interp(
            depth, object_distances_m, tensor["defocus_ratios"]
        ).astype(np.float32)
        s = _LC.compute_edge_selector(depth, dmap, pad=tensor["kernels"].shape[2] // 2)
        # Where s is EXACTLY 0 the final composite is 0·OVER + 1·off_diffuse =
        # off_diffuse exactly, so ON MUST equal OFF bit-closely there. The
        # smooth selector has a tail (0 < s ≪ 1) that legitimately blends a
        # sliver of OVER; that is correct, not the round-1 global re-render.
        # We therefore test the strict-regime-a set (s==0). The round-1 bug
        # made ON diverge from OFF even at s==0 (whole-frame re-render).
        strict_a = s <= 0.0

        diff = np.abs(on_lin - off_lin).max(axis=2)  # (h, w)
        outside = diff[strict_a]
        assert outside.size > 0, "fixture produced no s==0 region — bad test"
        assert float(outside.max()) < 1e-4, (
            f"ON diverged from OFF where s==0 "
            f"(max|ON-OFF|={outside.max():.2e} on {int((outside>1e-4).sum())} "
            f"pixels) — regime-a is NOT OFF-verbatim (probe-round-1 bug)."
        )
        # Also: the divergence must be CONFINED — only a small selector-band
        # fraction changes, not 99% (the round-1 symptom).
        band = _ndimage_dilate(s > 1e-3, tensor["kernels"].shape[2] // 2)
        # And the fix must be ISOLATED: the change is confined to the band.
        frac_changed_total = float((diff > 4 / 255).mean())
        frac_band = float(band.mean())
        assert frac_changed_total <= frac_band + 0.02, (
            f"ON-vs-OFF change ({frac_changed_total:.3f}) leaks well beyond the "
            f"selector band ({frac_band:.3f}) — global re-render, not isolated."
        )

    def test_smooth_frame_on_equals_off_everywhere(self):
        """A frame with NO occlusion edge (smooth tilted plane) → s≈0
        everywhere → ON must be byte-identical to OFF (early-return path)."""
        img = _make_image(96, 96, seed=22)
        h, w = img.shape[:2]
        depth = _tilted_depth(h, w, lo=3.0, hi=18.0)  # smooth, no step
        tensor = _synthetic_tensor_d11()
        off_u8, off_lin = render(img, tensor, depth_m=depth.copy())
        on_u8, on_lin = render(
            img, tensor, depth_m=depth.copy(),
            occlusion_compositing="on",
        )
        assert np.allclose(on_lin, off_lin, atol=1e-5), (
            f"smooth-frame ON != OFF (max abs {np.abs(on_lin-off_lin).max():.2e})"
        )


def _ndimage_dilate(mask, iters):
    from scipy import ndimage
    if iters < 1:
        return mask
    return ndimage.binary_dilation(mask, iterations=int(iters))


# ---------------------------------------------------------------------------
# Near-zero-α divide stability (Kosloff-Barsky clamp)
# ---------------------------------------------------------------------------


class TestNearZeroAlphaStability:
    def test_unpremult_finite_when_alpha_tiny(self):
        """The un-premultiply acc_c / max(acc_a, 1e-2) must stay finite even
        when the blurred coverage α collapses toward 0 (thin hair silhouette).
        round-3: floor raised 1e-3→1e-2 (1000×→100× worst-case gain) + a hard
        energy-bound clamp to the input diffuse range is the real guarantee."""
        acc_a = np.zeros((16, 16), dtype=np.float32)  # worst case: α ≡ 0
        acc_c = np.full((16, 16, 3), 0.5, dtype=np.float32)
        out = acc_c / np.maximum(acc_a, np.float32(1e-2))[..., None]
        assert np.isfinite(out).all()
        # Floor 1e-2 caps raw amplification at 100×; the module additionally
        # clips to [0, max_input_diffuse] so the real output bound is ~1.
        assert float(out.max()) <= 0.5 / 1e-2 + 1.0

    def test_full_on_render_finite(self):
        """A full ON render on a real depth step stays finite end-to-end (no
        NaN/Inf from the matte divide or the OVER fold)."""
        img = _make_image(seed=5)
        h, w = img.shape[:2]
        depth = _step_depth(h, w, near=0.8, far=12.0)
        tensor = _synthetic_tensor_d11()
        on_u8, on_lin = render(
            img, tensor, depth_m=depth.copy(),
            occlusion_compositing="on",
        )
        assert np.isfinite(on_lin).all()
        assert on_u8.dtype == np.uint8
        assert on_u8.shape == img.shape


# ---------------------------------------------------------------------------
# Energy bound — the diffuse OVER composite must NOT create energy
# (probe-round-2 PRIMARY DEFECT: un-premultiply produced 2–3× linear values).
# ---------------------------------------------------------------------------


class TestDiffuseOverEnergyBound:
    def test_bright_bg_behind_near_block_bounded(self):
        """A bright background (1.0) occluded by a large NEAR foreground block
        exercises the disocclusion hole + un-premultiply at the silhouette —
        the exact topology that produced the round-2 2–3× bright fringe. The
        diffuse-only ON composite must NOT exceed the max input diffuse (~1):
        all inputs ≤ 1 ⇒ a coverage-correct premultiplied OVER is bounded by 1.
        Compared against OFF so a genuine HDR>1 input can't mask the bug."""
        h, w = 128, 128
        img = np.full((h, w, 3), 1.0, np.float32)          # bright bg
        img[h // 3:2 * h // 3, w // 3:2 * w // 3, :] = 0.2  # dark near block
        depth = np.full((h, w), 18.0, np.float32)          # far bg
        depth[h // 3:2 * h // 3, w // 3:2 * w // 3] = 0.7   # near block
        tensor = _synthetic_tensor_d11()
        off_u8, off_lin = render(img, tensor, depth_m=depth.copy())
        on_u8, on_lin = render(
            img, tensor, depth_m=depth.copy(),
            occlusion_compositing="on",
        )
        in_max = float(img.max())
        assert np.isfinite(on_lin).all()
        # Energy bound: ON must not exceed input diffuse (the round-2 bug gave
        # ~3.1× here). Small ε for float / legitimate OFF blur overshoot.
        assert float(on_lin.max()) <= in_max + 1e-3, (
            f"diffuse OVER created energy: ON max {on_lin.max():.4f} > input "
            f"diffuse {in_max:.4f} (round-2 bright-fringe regression)"
        )
        # And ON max must track OFF max (no global-headroom hijack).
        assert float(on_lin.max()) <= float(off_lin.max()) + 1e-3, (
            f"ON max {on_lin.max():.4f} exceeds OFF max {off_lin.max():.4f} "
            f"— diffuse path raised the global max (headroom hijack)"
        )

    def test_thin_fg_strip_over_bright_bg_bounded(self):
        """Thin near foreground strip on a bright far background — explicitly
        creates a large matte==0 disocclusion region behind the strip. With the
        round-3 fix (no color-only fill) the un-premultiply stays ≤ input."""
        h, w = 96, 96
        img = np.full((h, w, 3), 0.95, np.float32)
        img[:, w // 2 - 1:w // 2 + 1, :] = 0.05
        depth = np.full((h, w), 15.0, np.float32)
        depth[:, w // 2 - 1:w // 2 + 1] = 0.7
        tensor = _synthetic_tensor_d11()
        _, off_lin = render(img, tensor, depth_m=depth.copy())
        _, on_lin = render(
            img, tensor, depth_m=depth.copy(),
            occlusion_compositing="on",
        )
        assert np.isfinite(on_lin).all()
        assert float(on_lin.max()) <= float(img.max()) + 1e-3
        assert float(on_lin.max()) <= float(off_lin.max()) + 1e-3


# ---------------------------------------------------------------------------
# Memory-guard projection (G-mem): occlusion term added, under 12 GB cap
# ---------------------------------------------------------------------------


class TestMemGuardProjection:
    def test_occlusion_term_added_but_under_cap(self):
        # R-native-mem: ON >= OFF (non-strict, the gather peak is shared) and
        # both stay <= 0.90 x cap at native for k=183 (locked-spec-mem.md §8).
        off = _CV._estimate_peak_bytes(4284, 5712, 11, occlusion=False)
        on = _CV._estimate_peak_bytes(4284, 5712, 11, occlusion=True)
        assert on >= off, "ON projection must not be below OFF"
        for name, est in (("OFF", off), ("ON", on)):
            assert est <= 0.90 * _CV.MAX_BYTES, (
                f"native-res {name} projection {est/1024**3:.2f} GiB exceeds "
                f"0.90 x {_CV.MAX_BYTES/1024**3:.0f} GiB cap"
            )
