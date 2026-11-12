"""P1.10.4 C4a — Basis-mode ring-artifact suppression coverage.

Ports G9 envelope check from killswitch file (deleted in C4b) so
ring-artifact suppression coverage survives slab path deletion.

Locked spec: doc/plan/phase1/section18/locked-spec-P1_10_4.md § 9 + § 11.

Source ported (verbatim envelope assertion):
  tests/unit/test_radial_blend_mode_killswitch.py:211-258
  (TestG9XfailFlip.test_gradient_second_derivative_bounded_basis_mode)

The original sibling was an `xfail-strict` that flipped to xpass once the
per-pixel basis blend removed the slab-boundary discontinuity at its source.
With slab now the deleted foil and basis the SOLE render kernel (P1.10.4
C3a, SHA `75a1f51`), this test is XFAIL-strict (regression-tracking marker —
see xfail reason for details).
"""
from __future__ import annotations

import numpy as np
import pytest

from summicron.render.convolve import render
from tests.fixtures.defocus import render_with_defocus


def _make_gaussian_tensor(R=4, D=4, ksize=33, sigma=4.0):
    """Tiny tensor for FAST smoke. Mirrors the legacy fixture used in
    `tests/adversarial/test_fft_artifact_adversarial.py` and the killswitch
    sibling at `tests/unit/test_radial_blend_mode_killswitch.py`."""
    cy, cx = ksize // 2, ksize // 2
    yy, xx = np.mgrid[:ksize, :ksize]
    g = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * sigma * sigma))
    g = g.astype(np.float32) / g.sum()
    kernels = np.tile(g[None, None, :, :], (R, D, 1, 1))
    # Pure delta at d=0 to mirror S2 guard semantics.
    delta = np.zeros((ksize, ksize), dtype=np.float32)
    delta[cy, cx] = 1.0
    kernels[:, 0] = delta
    return {
        "kernels": kernels,
        "radial_edges": np.linspace(0.0, 1.0, R + 1).astype(np.float64),
        "defocus_ratios": np.linspace(0.0, 1.0, D).astype(np.float64),
        "corpus_ref_long_edge": None,
        "otf_norm_applied": False,
        "delta_guard_applied": True,
        "object_distances_m": None,
        "focus_distance_m": None,
    }


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Basis-mode current 2nd-derivative ratio ~34.6 > 15.0 envelope on this "
        "synthetic delta-impulse fixture (radial gradient, R=4 D=4 sigma=4 tensor). "
        "Preserved as regression-tracking marker post-killswitch-delete (P1.10.4 C4b): "
        "if a future round (basis kernel refinement, R-bin bump, etc.) drops the ratio "
        "below 15.0, this xpass-strict will fire and alert us. Original sibling at "
        "test_radial_blend_mode_killswitch.py:209-256 (deleted in C4b)."
    ),
)
def test_gradient_second_derivative_bounded_basis_mode():
    """Basis-mode 2nd-derivative envelope on central column < 15.0.

    Load-bearing ring-artifact suppression assertion: per-pixel basis blend
    must remove slab-boundary discontinuity at its source. Verbatim envelope
    from the original G9 sibling (killswitch line 253).
    """
    sz = 256
    cy, cx = sz // 2, sz // 2
    yy, xx = np.mgrid[:sz, :sz]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).astype(np.float32)
    r /= r.max() + 1e-8
    img = np.zeros((sz, sz, 3), dtype=np.float32)
    img[:, :, :] = (0.1 + 0.8 * r)[:, :, None]

    tensor = _make_gaussian_tensor(R=4, D=4, ksize=65, sigma=4.0)
    dmap = r.copy()
    _, out_lin = render_with_defocus(
        img, tensor, dmap=dmap,
    )

    profile = out_lin[:, cx, 0]
    d2 = np.abs(np.diff(profile, n=2))
    median_d2 = float(np.median(d2)) + 1e-8
    max_d2 = float(d2.max())
    ratio = max_d2 / median_d2
    assert ratio < 15.0, (
        f"Seam artifact (basis mode): ratio {ratio:.1f} > 15 "
        f"(max={max_d2:.6f}, median={median_d2:.6f})"
    )
