"""P0.2 — radial/depth decoupling in ``summicron.render.convolve``.

Each (defocus_bin, radial_slab) pair now consumes its own kernel and the
renderer selects that kernel by the pixel's SPATIAL position (via a
precomputed ``radial_frac_map``) rather than by the defocus bin index.

LOCKED SPEC: ``doc/plan/phase0/section10/locked-spec-radial-decouple.md``.
"""
from __future__ import annotations

import numpy as np
import pytest

from summicron.render._artifacts import default_tensor_path

from summicron.render.convolve import _bilinear_weights
from tests.fixtures.defocus import render_with_defocus




# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

V3_TENSOR_PATH = default_tensor_path()


def _build_radial_frac_map(H: int, W: int) -> np.ndarray:
    """Mirror of the production computation, for direct bound tests."""
    cy = (H - 1) / 2.0
    cx = (W - 1) / 2.0
    max_r = float(np.hypot(cy, cx))
    if max_r < 1e-6:
        return np.zeros((H, W), dtype=np.float32)
    ys = np.arange(H, dtype=np.float32) - np.float32(cy)
    xs = np.arange(W, dtype=np.float32) - np.float32(cx)
    return (np.hypot(ys[:, None], xs[None, :]) / np.float32(max_r)).astype(np.float32)


def _delta_impulse_image(H: int, W: int, y: int, x: int) -> np.ndarray:
    """sRGB [0,1] image with a single 1.0 pixel at (y, x), zeros elsewhere."""
    img = np.zeros((H, W, 3), dtype=np.float32)
    img[y, x, :] = 1.0
    return img


def _effective_kernel_at(img_lin_out: np.ndarray, y: int, x: int, half: int) -> np.ndarray:
    """Extract a (2*half+1)² patch from a linear-space impulse response."""
    patch = img_lin_out[y - half:y + half + 1, x - half:x + half + 1, 1]  # green channel
    return patch.astype(np.float32)


def _load_v3_tensor_or_skip() -> dict:
    if not V3_TENSOR_PATH.exists():
        pytest.skip(f"v3 tensor not present at {V3_TENSOR_PATH}")
    from summicron.render.convolve import load_tensor
    return load_tensor(V3_TENSOR_PATH)


# ---------------------------------------------------------------------------
# T1 — radial_frac_map center and corners, square + odd-dim
# ---------------------------------------------------------------------------

class TestRadialFracMapCenterAndCorners:
    @pytest.mark.parametrize("dim", [1024, 1023])
    def test_corners_are_one(self, dim: int):
        m = _build_radial_frac_map(dim, dim)
        cy = (dim - 1) / 2.0
        # center pixel (rounded to integer index closest to (cy, cy))
        ci = dim // 2
        if dim % 2 == 1:
            # odd -> exact-zero center
            assert m[ci, ci] < 2e-3
        else:
            # even -> half-pixel offset; still tiny
            assert m[ci, ci] < 2e-3
        for (yy, xx) in [(0, 0), (0, dim - 1), (dim - 1, 0), (dim - 1, dim - 1)]:
            assert m[yy, xx] == pytest.approx(1.0, abs=1e-5), (
                f"corner ({yy},{xx}) = {m[yy,xx]}"
            )


# ---------------------------------------------------------------------------
# T2 — effective kernel at (512,512) is near-delta on v3 + zero defocus
# ---------------------------------------------------------------------------

class TestEffectiveKernelCenterIsNearDelta:
    def test_center_peak_geq_0p80(self):
        tensor = _load_v3_tensor_or_skip()
        H = W = 1024
        img = _delta_impulse_image(H, W, 512, 512)
        defocus_map = np.zeros((H, W), dtype=np.float32)
        _u8, out_lin = render_with_defocus(img, tensor, dmap=defocus_map)
        # Extract effective kernel (~183 px span) around (512, 512)
        half = 91
        kern = _effective_kernel_at(out_lin, 512, 512, half)
        peak = float(kern.max())
        nnz = int(np.sum(kern > 1e-4))
        assert peak >= 0.80, f"center peak {peak:.4f} < 0.80"
        assert nnz <= 5000, f"center nnz {nnz} > 5000"


# ---------------------------------------------------------------------------
# T3 — effective kernel at corner differs from center
# ---------------------------------------------------------------------------

class TestEffectiveKernelCornerDiffersFromCenter:
    def test_l1_distance_gt_0p05(self):
        """Physical invariant (P0.2 decoupling): center pixel and corner pixel
        receive DIFFERENT effective kernels.

        History: pre-fftshift fix in ``summicron/psf_model/otf_norm.py``, this
        L1 distance measured ~2.0 — but that was an artifact. The corner
        kernel peaked at (0, 0) (FFT origin) while the center kernel peaked
        at (K//2, K//2) (geometric center), so the two patches lived in
        DISJOINT supports and the L1 reflected the centering bug, not a
        physical kernel difference.

        Post-fix: ``normalize_otf`` applies ``np.fft.fftshift`` so both
        effective kernels are centered at (K//2, K//2), and the L1 now
        reflects the GENUINE physical difference in kernel shape between
        the two radial slabs (~0.12 on the current v3 tensor). Threshold
        is 0.05 — decisively above the ~0.01 numerical-round-off floor
        and robust to tensor regeneration within the same spec.
        """
        tensor = _load_v3_tensor_or_skip()
        H = W = 1024
        # P1.9.1: dmap must be >= D_KNEE (0.10) so α=0 and convolution runs.
        # At dmap=0 the α blend returns the sharp input bit-exactly, collapsing
        # both center and corner to identity responses.
        defocus_map = np.full((H, W), 0.5, dtype=np.float32)

        # Center impulse
        img_c = _delta_impulse_image(H, W, 512, 512)
        _u8, out_c = render_with_defocus(img_c, tensor, dmap=defocus_map)
        half = 91
        kern_c = _effective_kernel_at(out_c, 512, 512, half)

        # Off-center impulse at (100, 100)
        img_o = _delta_impulse_image(H, W, 100, 100)
        _u8, out_o = render_with_defocus(img_o, tensor, dmap=defocus_map)
        kern_o = _effective_kernel_at(out_o, 100, 100, half)

        l1 = float(np.abs(kern_c - kern_o).sum())
        assert l1 > 0.05, f"center-vs-corner kernel L1 distance {l1:.4f} <= 0.05"


# ---------------------------------------------------------------------------
# T4 — selection uses spatial radial, not defocus
# ---------------------------------------------------------------------------

def _make_distinguishable_tensor(ksize: int = 129) -> dict:
    """Synthetic (R=4, D=2) tensor where kernels[r, d] are centered Gaussians
    with distinct widths so the effective kernel varies by (r, d).

    Each kernel is L1-normalized (sum=1) to preserve energy.  Sigma grows
    with both r and d so that center and corner spatial slabs produce
    visibly different effective kernels (the test checks L1 distance of
    the extracted impulse response).

    P1.3 note: the old version used offset-delta kernels which broke
    the renderer's energy gate after the convolve-then-blend fix.
    Centered Gaussians are physically realistic and compatible with
    post-convolution weighting.
    """
    R, D = 4, 2
    kernels = np.zeros((R, D, ksize, ksize), dtype=np.float32)
    c = ksize // 2
    yy, xx = np.mgrid[:ksize, :ksize] - c
    for r in range(R):
        for d in range(D):
            sigma = max(0.5, 1.0 + 2.0 * r + 1.5 * d)
            g = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma * sigma)).astype(np.float32)
            g /= g.sum()
            kernels[r, d] = g
    radial_edges = np.array(
        [0.0, 0.15, 0.4, 0.7, 1.01], dtype=np.float64
    )  # R+1 = 5
    defocus_ratios = np.array([0.0, 1.0], dtype=np.float64)
    return {
        "kernels": kernels,
        "radial_edges": radial_edges,
        "defocus_ratios": defocus_ratios,
    }


class TestSelectionUsesSpatialRadialNotDefocus:
    def test_center_and_corner_select_different_kernels(self):
        tensor = _make_distinguishable_tensor()
        H = W = 256
        # P1.9.1: dmap must be >= D_KNEE (0.10) so α=0 and convolution runs.
        # At dmap=0 the α blend returns the sharp input, making center and
        # corner effective kernels both identity — defeats the test premise.
        dmap = np.full((H, W), 0.5, dtype=np.float32)

        img_c = _delta_impulse_image(H, W, 128, 128)
        _u8, out_c = render_with_defocus(img_c, tensor, dmap=dmap)
        img_o = _delta_impulse_image(H, W, 20, 20)
        _u8, out_o = render_with_defocus(img_o, tensor, dmap=dmap)

        # Extract a patch around each impulse and compare.
        half = 20
        kc = out_c[128 - half:128 + half + 1, 128 - half:128 + half + 1, 1]
        ko = out_o[20 - half:20 + half + 1, 20 - half:20 + half + 1, 1]
        assert not np.allclose(kc, ko, atol=1e-4), (
            "center and corner effective kernels are byte-identical; "
            "radial selection is NOT spatial"
        )

    def test_wrong_polarity_defocus_map_still_spatial(self):
        tensor = _make_distinguishable_tensor()
        H = W = 256
        # Wrong-polarity: all-ones dmap. Selection must STILL depend on
        # spatial position only (defocus picks the bin, radial picks the slab).
        dmap = np.ones((H, W), dtype=np.float32)

        img_c = _delta_impulse_image(H, W, 128, 128)
        _u8, out_c = render_with_defocus(img_c, tensor, dmap=dmap)
        img_o = _delta_impulse_image(H, W, 20, 20)
        _u8, out_o = render_with_defocus(img_o, tensor, dmap=dmap)

        half = 20
        kc = out_c[128 - half:128 + half + 1, 128 - half:128 + half + 1, 1]
        ko = out_o[20 - half:20 + half + 1, 20 - half:20 + half + 1, 1]
        assert not np.allclose(kc, ko, atol=1e-4), (
            "radial selection depends on defocus polarity, not spatial position"
        )


# ---------------------------------------------------------------------------
# T5 — radial_frac_map bounds and dtype
# ---------------------------------------------------------------------------

class TestRadialFracMapBoundsAndDtype:
    @pytest.mark.parametrize("shape", [(1024, 1024), (1024, 1536)])
    def test_bounds_and_dtype(self, shape: tuple[int, int]):
        H, W = shape
        m = _build_radial_frac_map(H, W)
        assert m.dtype == np.float32
        assert m.min() >= 0.0
        assert m.max() <= 1.001


# ---------------------------------------------------------------------------
# T6 — tap-to-focus wrap preserves spatial radial
# ---------------------------------------------------------------------------

class TestTapFocusWrapPreservesSpatialRadial:
    def test_wrap_changes_d_not_radial(self):
        """The spatial radial_frac map is a function of (H, W) and the corner
        pixel's radial_frac is the same before/after a depth-map wrap. Only
        d_center (i.e. the defocus bin selection) changes."""
        from summicron.depth.estimate import apply_tap_focus_shift

        H, W = 128, 128
        # Synthetic monotonic depth, 0.4m -> 1.2m along rows
        depth = np.tile(
            np.linspace(0.4, 1.2, W, dtype=np.float32)[None, :], (H, 1)
        )

        # Pre-wrap radial_frac at corner.
        m_pre = _build_radial_frac_map(H, W)
        rf_pre = float(m_pre[0, 0])

        # Wrap via tap-to-focus at a chosen pixel.
        shifted, _anchor = apply_tap_focus_shift(
            depth, tap_xy_norm=(0.25, 0.25), focus_target_m=0.7
        )

        # Post-wrap radial_frac at the same corner: function of H, W only.
        m_post = _build_radial_frac_map(H, W)
        rf_post = float(m_post[0, 0])

        assert rf_pre == pytest.approx(rf_post, abs=1e-6), (
            "spatial radial_frac changed across the tap-focus wrap"
        )
        # And the depth DID change (otherwise the test proves nothing).
        assert not np.allclose(shifted, depth)


# ---------------------------------------------------------------------------
# T7 — xfail witness for no kernel rotation
# ---------------------------------------------------------------------------

# CLI-1a fix round: the earlier version passed trivially (dmap=0 hits the
# layered identity snap -> output == input, and the window was off-centre).
# With real defocus (dmap=1.0) and a window centred on the impulse, the
# measured principal angle at top-centre is ~44deg (see reason).
@pytest.mark.xfail(
    strict=True,
    reason="CONVENTION MISMATCH (not a missing rotation): stored-kernel "
    "reference axis is vertical, so the stored 45deg tilt is preserved at "
    "top-centre (~44deg); this test assumes the stored long axis = radial "
    "(~90deg). Per-pixel rotation verified by "
    "test_astigmatism_relative_rotation.",
)
def test_astigmatism_direction_is_reference_axis_xfail():
    """Construct a tensor where `kernels[r>0, d=0]` is an elongated
    Gaussian tilted 45° in the stored array. Render an impulse at the
    TOP-center pixel (radial direction = vertical). With correct
    per-pixel rotation, the rendered impulse's elongation would follow
    the radial direction (vertical). Without rotation, the impulse
    inherits the stored 45° tilt. Assert that the rendered patch's
    principal axis is the RADIAL direction (vertical), i.e. strict
    correctness. Today this FAILS (the rendered elongation is 45°),
    hence xfail=strict."""
    ksize = 65
    R, D = 3, 2
    c = ksize // 2
    yy, xx = np.mgrid[-c:c + 1, -c:c + 1].astype(np.float32)
    # 45-degree elongated Gaussian: long axis along (1,1)
    u = (yy + xx) / np.sqrt(2.0)
    v = (yy - xx) / np.sqrt(2.0)
    sigma_long, sigma_short = 8.0, 2.0
    g = np.exp(-(u ** 2) / (2 * sigma_long ** 2) - (v ** 2) / (2 * sigma_short ** 2))
    g /= g.sum()
    kernels = np.zeros((R, D, ksize, ksize), dtype=np.float32)
    kernels[0, 0, c, c] = 1.0
    kernels[1, 0] = g.astype(np.float32)
    kernels[2, 0] = g.astype(np.float32)
    kernels[:, 1] = kernels[:, 0]  # copy to d=1
    radial_edges = np.array([0.0, 0.3, 0.7, 1.01], dtype=np.float64)
    defocus_ratios = np.array([0.0, 1.0], dtype=np.float64)
    tensor = {
        "kernels": kernels,
        "radial_edges": radial_edges,
        "defocus_ratios": defocus_ratios,
    }
    H = W = 256
    row = 40
    # Impulse at top-center -> radial direction is vertical (from frame center)
    img = _delta_impulse_image(H, W, row, W // 2)
    # dmap=1.0 -> fully defocused (dmap=0 would hit the identity snap)
    dmap = np.full((H, W), 1.0, dtype=np.float32)
    _u8, out_lin = render_with_defocus(img, tensor, dmap=dmap)
    # Extract patch CENTRED on the impulse (covers the full 65px kernel)
    half = 32
    patch = out_lin[row - half:row + half + 1, W // 2 - half:W // 2 + half + 1, 1]
    # Principal axis via second moments
    ys, xs = np.mgrid[0:patch.shape[0], 0:patch.shape[1]].astype(np.float32)
    w = patch + 1e-12
    wsum = w.sum()
    my = (ys * w).sum() / wsum
    mx = (xs * w).sum() / wsum
    cyy = ((ys - my) ** 2 * w).sum() / wsum
    cxx = ((xs - mx) ** 2 * w).sum() / wsum
    cxy = ((ys - my) * (xs - mx) * w).sum() / wsum
    # Principal angle
    angle = 0.5 * np.arctan2(2.0 * cxy, cxx - cyy)  # radians, measured from x-axis
    # Radial direction here is vertical -> principal angle should be ~pi/2.
    assert abs(abs(angle) - np.pi / 2.0) < np.pi / 6.0, (
        f"principal axis angle {np.degrees(angle):.1f}° not near 90° "
        "(rotation not implemented)"
    )


def test_astigmatism_relative_rotation():
    """Per-pixel rotation DOES happen: the principal-axis angle of the
    rendered impulse changes with position by the radial-direction change.
    top vs left differ by ~90deg; top vs top-left by ~45deg (mod 180).
    Anisotropy guard prevents passing on an isotropic blob."""
    ksize = 65
    R, D = 3, 2
    c = ksize // 2
    yy, xx = np.mgrid[-c:c + 1, -c:c + 1].astype(np.float32)
    u = (yy + xx) / np.sqrt(2.0)
    v = (yy - xx) / np.sqrt(2.0)
    g = np.exp(-(u ** 2) / (2 * 8.0 ** 2) - (v ** 2) / (2 * 2.0 ** 2))
    g /= g.sum()
    kernels = np.zeros((R, D, ksize, ksize), dtype=np.float32)
    kernels[0, 0, c, c] = 1.0
    kernels[1, 0] = g.astype(np.float32)
    kernels[2, 0] = g.astype(np.float32)
    kernels[:, 1] = kernels[:, 0]
    tensor = {
        "kernels": kernels,
        "radial_edges": np.array([0.0, 0.3, 0.7, 1.01], dtype=np.float64),
        "defocus_ratios": np.array([0.0, 1.0], dtype=np.float64),
    }
    H = W = 256
    dmap = np.full((H, W), 1.0, dtype=np.float32)

    def measure(row: int, col: int) -> tuple[float, float]:
        img = _delta_impulse_image(H, W, row, col)
        _u8, out_lin = render_with_defocus(img, tensor, dmap=dmap)
        half = 32
        patch = out_lin[row - half:row + half + 1, col - half:col + half + 1, 1]
        ys, xs = np.mgrid[0:patch.shape[0], 0:patch.shape[1]].astype(np.float64)
        w = patch.astype(np.float64) + 1e-12
        wsum = w.sum()
        my = (ys * w).sum() / wsum
        mx = (xs * w).sum() / wsum
        cyy = ((ys - my) ** 2 * w).sum() / wsum
        cxx = ((xs - mx) ** 2 * w).sum() / wsum
        cxy = ((ys - my) * (xs - mx) * w).sum() / wsum
        angle = np.degrees(0.5 * np.arctan2(2.0 * cxy, cxx - cyy))
        tr = cxx + cyy
        det = cxx * cyy - cxy ** 2
        disc = np.sqrt(max(tr * tr / 4.0 - det, 0.0))
        l1, l2 = tr / 2.0 + disc, tr / 2.0 - disc
        return float(angle), float(l1 / max(l2, 1e-12))

    a_top, r_top = measure(40, 128)
    a_left, r_left = measure(128, 40)
    a_tl, r_tl = measure(40, 40)
    a_bot, r_bot = measure(216, 128)
    print(f"angles top={a_top:.1f} left={a_left:.1f} "
          f"top-left={a_tl:.1f} bottom={a_bot:.1f}")

    def mod180_diff(a: float, b: float) -> float:
        d = (a - b) % 180.0
        return min(d, 180.0 - d)

    for name, ratio in (("top", r_top), ("left", r_left),
                        ("top-left", r_tl), ("bottom", r_bot)):
        assert ratio > 2.0, f"{name} anisotropy {ratio:.2f} <= 2"
    assert abs(mod180_diff(a_top, a_left) - 90.0) < 15.0, (a_top, a_left)
    assert abs(mod180_diff(a_top, a_tl) - 45.0) < 15.0, (a_top, a_tl)


# ---------------------------------------------------------------------------
# T8 — off-center optical axis effective kernel on tiny frame
# ---------------------------------------------------------------------------
# P1.10.4 C4b: `test_tiny_frame_center_vs_corner_differ` deleted —
# slab-tuned diffuse-energy gate envelope was structurally obsolete after
# slab body deletion (C3). G2 / RPS fixtures + basis-mode regression armor
# preserve coverage of "per-pixel radial kernel selection produces
# center≠corner outputs."
