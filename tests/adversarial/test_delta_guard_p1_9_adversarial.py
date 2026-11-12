"""P1.9 adversarial probes for the build-time S2 delta guard.

Attack surface covered:
- B1: is_delta fast-path flips at a d-index boundary produced by the
      bilinear blend. With d=0 a pure delta + d=1 a raw coma PSF, the
      blended kernel at render-time *first* defocus bin takes the
      fast-path (no angular rotation), then at the second bin takes the
      full 8-wedge rotated path. Quality discontinuity across the bin
      boundary. XFAIL-strict — documents the seam.
- B2: Warning predicate with numpy scalar True for only one flag.
      Guards against `bool(np.bool_)` quirks when one flag is numpy
      scalar and the other is Python bool.
- B3: Inverse focus (tree-sharp) sanity — pin the observed stuffed-
      animal-still-sharper pathology for carry-forward. XFAIL-strict on
      the real IMG_6651.HEIC fixture, guarded by file existence.
- B4: downscale_tensor silently accepts its own output as input
      (scale=1, all gates still pass). Flag as an idempotence smell.
- B5: L1 preservation check across the guard at float32 exactness —
      pre-guard kernel L1 is float32-exact 1.0 only via assertion
      tolerance; `_make_delta` is float32 with exact sum 1.0. Pin.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from summicron.render._artifacts import default_tensor_path

from summicron.psf_model.downscale_tensor import _make_delta, downscale_tensor
from summicron.render.convolve import _bilinear_weights


REAL_CODEV = Path(__file__).resolve().parents[2] / "runs" / "ag_dblgauss" / "ag_dblgauss_f2.0_render.npz"
REAL_IPHONE = default_tensor_path()


# ----------------------------------------------------------------------
# B1: is_delta fast-path flips at d-index boundary
# ----------------------------------------------------------------------

@pytest.mark.xfail(
    strict=True,
    reason="P1.9 pre-existing bilinear-blend seam: bin_c=0.0625 blends "
           "0.375*delta + 0.625*raw[r,1] -> peak/sum ~0.62 (is_delta True) but "
           "bin_c=0.1875 blends 0.125*raw[r,1] + 0.875*raw[r,2] -> is_delta "
           "False. Angular rotation is applied in bin 1 but not bin 0, "
           "producing a visible kernel discontinuity across the boundary. "
           "Fix TBD: lower is_delta threshold, or rotate the blended kernel "
           "unconditionally.",
)
@pytest.mark.skipif(not REAL_IPHONE.exists(), reason="iphone tensor not regen'd")
def test_b1_bilinear_blend_d0_boundary_no_is_delta_flip():
    """With delta at d=0 + raw PSF at d=1, the is_delta predicate
    (peak > 0.5*sum) must be CONSISTENT across the whole d-index range
    [d_center_bin_0, d_center_bin_1]; otherwise the renderer switches
    code paths mid-blend and produces a visible seam.
    """
    d = np.load(str(REAL_IPHONE))
    re = d["radial_edges"]
    dr = d["defocus_ratios"]
    K = d["kernels"]
    # Probe: for every r, check bin_c=0.0625 vs bin_c=0.1875. Build the
    # blended kernel. Must produce the SAME is_delta verdict.
    for r in range(K.shape[0]):
        # use the slab midpoint radial value for r=1..Nr-1
        r_mid = 0.5 * (re[r] + re[r + 1])
        for bin_c in (0.0625, 0.1875):
            idx, w = _bilinear_weights(r_mid, bin_c, re, dr)
            kb = np.zeros_like(K[0, 0])
            for (ri, di), bw in zip(idx, w):
                kb = kb + bw * K[ri, di]
            is_delta = kb.max() > 0.5 * kb.sum()
            # We pin the EXPECTED invariant: adjacent bins should agree.
            # Today they DO NOT at bin_c=0.0625 vs 0.1875 for r=0 —
            # hence xfail. When this starts passing, either threshold
            # was changed or the blend was reshaped.
            if bin_c == 0.0625:
                bin0_is_delta = is_delta
            else:
                bin1_is_delta = is_delta
        assert bin0_is_delta == bin1_is_delta, (
            f"r={r}: bin_0.0625 is_delta={bin0_is_delta} != "
            f"bin_0.1875 is_delta={bin1_is_delta} — quality discontinuity"
        )


# ----------------------------------------------------------------------
# B2: Warning predicate numpy-scalar mixed types
# ----------------------------------------------------------------------

def test_b2_warning_silent_when_only_delta_guard_true(tmp_path: Path):
    """Warning MUST NOT fire when delta_guard_applied=True even if
    otf_norm_applied is absent. Covers the legacy-npz-regenerated-with-
    guard path.
    """
    import warnings
    from summicron.render.convolve import load_tensor

    npz = tmp_path / "guard_only.npz"
    np.savez(
        str(npz),
        kernels=np.zeros((1, 1, 3, 3), dtype=np.float32),
        radial_edges=np.array([0.0, 1.0]),
        defocus_ratios=np.array([0.5]),
        corpus_ref_long_edge=np.int64(5712),
        delta_guard_applied=np.bool_(True),
        # otf_norm_applied intentionally absent
    )
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        r = load_tensor(npz)
        smear = [
            x for x in w if "smear" in str(x.message) or "delta_guard_applied" in str(x.message)
        ]
        assert len(smear) == 0, (
            f"False warning when delta_guard_applied=True: {[str(x.message) for x in smear]}"
        )
    assert r["delta_guard_applied"] is True


def test_b2b_warning_silent_when_both_true(tmp_path: Path):
    """Defensive: BOTH True should not warn (redundant but guards
    predicate refactors that flip polarity)."""
    import warnings
    from summicron.render.convolve import load_tensor

    npz = tmp_path / "both_true.npz"
    np.savez(
        str(npz),
        kernels=np.zeros((1, 1, 3, 3), dtype=np.float32),
        radial_edges=np.array([0.0, 1.0]),
        defocus_ratios=np.array([0.5]),
        corpus_ref_long_edge=np.int64(5712),
        otf_norm_applied=np.bool_(True),
        delta_guard_applied=np.bool_(True),
    )
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        load_tensor(npz)
        smear = [x for x in w if "smear" in str(x.message)]
        assert len(smear) == 0


# ----------------------------------------------------------------------
# B3: Inverse-focus sanity (tree-sharp when tapped on tree)
# ----------------------------------------------------------------------

REAL_IMAGE = Path(__file__).resolve().parents[2] / "iphoneImage" / "IMG_6651.HEIC"


@pytest.mark.slow
@pytest.mark.skipif(
    not (REAL_IMAGE.exists() and REAL_IPHONE.exists()),
    reason="bench image or tensor missing",
)
def test_b3_inverse_focus_tree_is_sharper():
    """Render with tap on tree; tree should be sharper than stuffed animal."""
    import subprocess
    import tempfile
    from scipy.ndimage import laplace

    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "bench_tree.jpg"
        subprocess.run(
            [
                sys.executable, "-m",
                "summicron.render.convolve",
                "--image", str(REAL_IMAGE),
                "--tensor", str(REAL_IPHONE),
                "--depth-source", "lidar",
                "--depth-scale", "10",
                "--downres", "2048",
                "--focus-xy", "0.5", "0.15",
                "--psf-norm", "l1",
                "--hdr-source", "macos",
                "--output", str(out),
            ],
            check=True,
            capture_output=True,
        )
        from PIL import Image
        im = np.asarray(Image.open(out))
        H, W, _ = im.shape
        sy = int(0.75 * H)
        ty = int(0.15 * H)
        tx = W // 2
        stuffed = im[sy - 30:sy + 30, tx - 40:tx + 40].mean(-1).astype(float)
        tree = im[max(0, ty - 30):ty + 30, tx - 40:tx + 40].mean(-1).astype(float)
        s_sharp = float(np.abs(laplace(stuffed)).mean())
        t_sharp = float(np.abs(laplace(tree)).mean())
        assert t_sharp > s_sharp, (
            f"tap on tree but tree_sharp={t_sharp:.3f} <= "
            f"stuffed_sharp={s_sharp:.3f}"
        )


# ----------------------------------------------------------------------
# B4: Idempotence on own output
# ----------------------------------------------------------------------

@pytest.mark.skipif(not REAL_IPHONE.exists(), reason="iphone tensor not regen'd")
def test_b4_downscale_on_own_output_is_near_idempotent(tmp_path: Path):
    """Running downscale_tensor on its own output (already-downscaled)
    must not corrupt the tensor. Observed: succeeds but introduces
    ~1e-7 float32 jitter from cubic zoom at scale=1 (NOT bit-identical).
    Pins the design flaw: no regen-detection guard. If downscale ever
    gains a hard guard against double-downscale, flip this to expect an
    error. If it ever becomes bit-identical (scale=1 short-circuit),
    tighten to assert_array_equal."""
    out = tmp_path / "second_pass.npz"
    downscale_tensor(REAL_IPHONE, out, target_long_edge=5712)
    d1 = np.load(str(REAL_IPHONE))
    d2 = np.load(str(out))
    # Within float32 cubic-zoom jitter (~1e-7) — the delta slices at d=0
    # are exact though.
    np.testing.assert_allclose(d1["kernels"], d2["kernels"], atol=1e-6)
    # d=0 delta slices must remain EXACT across passes.
    np.testing.assert_array_equal(d1["kernels"][:, 0], d2["kernels"][:, 0])
    assert bool(d2["delta_guard_applied"]) is True


# ----------------------------------------------------------------------
# B5: L1 preservation exactness
# ----------------------------------------------------------------------

def test_b5_make_delta_l1_is_float32_exact_one():
    """`_make_delta` must return dtype=float32 with L1 EXACTLY 1.0
    (not 1.0 ± eps). This is what the build-time G1 gate asserts, and a
    refactor returning float64 or a sum!=1 delta would silently cast
    float32→float32 and break G1 at runtime."""
    for size in (3, 17, 183, 257):
        d = _make_delta(size)
        assert d.dtype == np.float32
        assert float(d.sum()) == 1.0
        assert float(d.max()) == 1.0
        assert int((d != 0).sum()) == 1
        # Geometric center, not FFT origin
        assert d[size // 2, size // 2] == 1.0
