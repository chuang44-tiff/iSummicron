"""Shared defocus-map → depth_m migration fixture (CLI-1a A1).

``render()`` is moving to a single distance-indexed contract
(``depth_m`` + ``object_distances_m``). Tests historically drove the
renderer with a normalized ``defocus_map`` in ``[0, 1]`` (or with no map
at all, relying on the synthetic radial map). This module is the single
migration point: it converts such a map to ``depth_m`` via the tensor's
own ``(defocus_ratios, object_distances_m)`` axes and calls ``render``.

Import as ``from tests.fixtures.defocus import render_with_defocus, radial_dmap``.
"""
from __future__ import annotations

import numpy as np


def radial_dmap(h: int, w: int) -> np.ndarray:
    """Quadratic radial defocus: 0 at center, 1 at corners. Shape (H, W) float32.

    Verbatim copy of the retired ``convolve.build_defocus_map``.
    """
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    max_dist = cy ** 2 + cx ** 2
    if max_dist < 1e-6:
        # Degenerate case (e.g. 1x1 image): no defocus possible
        return np.zeros((h, w), dtype=np.float32)
    ys = np.arange(h, dtype=np.float32)
    xs = np.arange(w, dtype=np.float32)
    dy = ys - cy
    dx = xs - cx
    # Broadcast: (H, 1) + (1, W)
    d = (dy[:, None] ** 2 + dx[None, :] ** 2) / max_dist
    return np.clip(d, 0.0, 1.0).astype(np.float32)


def render_with_defocus(img, tensor, dmap=None, **kw):
    """Render ``img`` with a normalized defocus map translated to ``depth_m``.

    - ``tensor`` is shallow-copied; if it lacks ``object_distances_m`` a
      geometric axis ``np.geomspace(0.7, 20.0, D)`` is injected
      (``D == 1`` → ``[0.7]``).
    - ``dmap=None`` → :func:`radial_dmap` of the image's ``(H, W)``.
    - ``depth_m = interp(dmap, defocus_ratios, object_distances_m)``.
    """
    from summicron.render import convolve as _cv

    t = dict(tensor)
    D = int(np.asarray(t["kernels"]).shape[1])
    ods = t.get("object_distances_m")
    if ods is None:
        ods = np.array([0.7]) if D == 1 else np.geomspace(0.7, 20.0, D)
        t["object_distances_m"] = ods
    ods = np.asarray(ods, dtype=np.float64)

    if dmap is None:
        H, W = np.asarray(img).shape[:2]
        dmap = radial_dmap(H, W)
    dmap = np.asarray(dmap)

    if D == 1:
        depth_m = np.full(dmap.shape, ods[0], dtype=np.float32)
    else:
        ratios = np.asarray(t["defocus_ratios"], dtype=np.float64)
        depth_m = np.interp(dmap, ratios, ods).astype(np.float32)
    return _cv.render(img, t, depth_m=depth_m, **kw)
