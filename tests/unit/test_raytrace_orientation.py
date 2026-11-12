"""Gate G-O (Section 24): the renderer must carry ``OUTWARD_ROW_SIGN`` mass outward.

A hand-made 12x11 tensor has every non-d0 kernel's mass at
``row = centre + 10 * OUTWARD_ROW_SIGN``. Eight isolated highlights, one per
renderer angular wedge centre, are rendered at constant defocus; each blob's
displacement must point radially outward (cosine >= 0.9) in 8/8 wedges.
If this fails, the fix is ONLY flipping ``ray_kernel.OUTWARD_ROW_SIGN``.
"""
from __future__ import annotations

import numpy as np

from summicron.psf_model.ray_kernel import (
    OUTWARD_ROW_SIGN,
    assemble_tensor,
    save_tensor,
)
from summicron.render.convolve import N_ANG, load_tensor
from tests.fixtures.defocus import render_with_defocus

H, W = 1536, 2048
SIZE = 183


def _tensor(tmp_path):
    k = np.zeros((12, 11, SIZE, SIZE), np.float64)
    k[:, 1:, SIZE // 2 + 10 * OUTWARD_ROW_SIGN, SIZE // 2] = 1.0
    p = tmp_path / "hand.npz"
    save_tensor(p, assemble_tensor(k, {}))
    return load_tensor(p)


def test_outward_mass_moves_outward_in_all_wedges(tmp_path):
    tensor = _tensor(tmp_path)
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    max_r = float(np.hypot(cy, cx))
    img = np.zeros((H, W, 3), np.float32)
    spots = []
    for a in range(N_ANG):
        theta = (a + 0.5) * (2 * np.pi / N_ANG) - np.pi
        u = np.array([np.cos(theta), np.sin(theta)])  # (x, y-down)
        # r ~ 0.9 of the half-diagonal, capped to stay inside the frame.
        edge = min(cx / max(abs(u[0]), 1e-9), cy / max(abs(u[1]), 1e-9))
        r = min(0.9 * max_r, 0.92 * edge)
        x = int(round(cx + r * u[0]))
        y = int(round(cy + r * u[1]))
        img[y, x] = 1.0
        spots.append((x, y, u))

    out = render_with_defocus(img, tensor, dmap=np.full((H, W), 0.5, np.float32))[0]
    out = np.asarray(out, np.float64)
    lum = out.mean(axis=2) if out.ndim == 3 else out
    win = 24
    cosines = []
    for x, y, u in spots:
        sub = lum[y - win:y + win + 1, x - win:x + win + 1]
        assert sub.sum() > 0
        ys, xs = np.indices(sub.shape)
        dy = (sub * ys).sum() / sub.sum() - win
        dx = (sub * xs).sum() / sub.sum() - win
        d = np.array([dx, dy])
        n = np.linalg.norm(d)
        assert n > 0.5, f"no displacement at {(x, y)}: {d}"
        cosines.append(float(d @ u / n))
    assert all(c >= 0.9 for c in cosines), f"outward cosines {np.round(cosines, 3)}"
