"""Shared optiland helpers for the PSF tensor builder and the optional optimiser.

Run under the ``optiland`` conda env only. Never imported by the ``summicron``
package. Lens tables are lists of ``(radius | "STOP", thickness, (nd, vd) | None)``
in patent units (f = 100); ``S`` scales them to mm.
"""
from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
import optiland.backend as be  # noqa: E402
from optiland import optic  # noqa: E402
from optiland.materials import AbbeMaterial  # noqa: E402
from optiland.physical_apertures.radial import RadialAperture  # noqa: E402

WL = 0.55
# Defaults = the 35/2 build (US 3,006,249 Example II; bit-identical). Other focal lengths: set ISUMMICRON_BUILD_PSF_EFL_MM,
# ISUMMICRON_BUILD_PSF_EPD_MM and ISUMMICRON_BUILD_PSF_HALF_FIELD_DEG in the environment (spawned workers inherit them).
HALF_FIELD_DEG = float(os.environ.get("ISUMMICRON_BUILD_PSF_HALF_FIELD_DEG", 31.7))
TAN_MAX = float(np.tan(np.radians(HALF_FIELD_DEG)))
EFL_TARGET = float(os.environ.get("ISUMMICRON_BUILD_PSF_EFL_MM", 35.0))
EPD_TARGET = float(os.environ.get("ISUMMICRON_BUILD_PSF_EPD_MM", 17.5))
MIN_AIR_GAP_MM = 0.05
LENS_DIR = Path(__file__).resolve().parent / "lenses"
PATENT_JSON = LENS_DIR / "us3006249_ex2.json"


def load_lens(path) -> dict:
    return json.loads(Path(path).read_text())


def to_table(lens: dict) -> list:
    tab = []
    for s in lens["surfaces"]:
        r = "STOP" if s.get("stop") else float(s["radius"])
        g = (float(s["nd"]), float(s["vd"])) if s.get("nd") else None
        tab.append((r, float(s["thickness"]), g))
    return tab


def mk(tab, S, epd, fld, aiming=None):
    """Optic from a patent-unit table scaled by S; object at infinity."""
    o = optic.Optic()
    o.add_surface(index=0, radius=be.inf, thickness=be.inf)
    for i, (r, t, g) in enumerate(tab, 1):
        kw = dict(index=i, radius=be.inf if r == "STOP" else r * S, thickness=t * S)
        if r == "STOP":
            kw["is_stop"] = True
        if g:
            kw["material"] = AbbeMaterial(n=g[0], abbe=g[1])
        o.add_surface(**kw)
    o.add_surface(index=len(tab) + 1)
    o.set_aperture("EPD", epd)
    o.fields.set_type("angle")
    o.fields.add(y=0)
    o.fields.add(y=fld)
    o.wavelengths.add(value=WL, is_primary=True)
    if aiming:
        o.ray_tracer.set_aiming(aiming)
    return o


def scale_factor(tab) -> float:
    """Global scale S that holds EFL = EFL_TARGET (35.000 mm by default)."""
    f = float(mk(tab, 1.0, 50.0, HALF_FIELD_DEG).paraxial.f2())
    return EFL_TARGET / f


def scaled_optic(tab, aiming=None):
    s = scale_factor(tab)
    return mk(tab, s, EPD_TARGET, HALF_FIELD_DEG, aiming), s


def set_object(o, s_ep_mm: float, epl: float) -> None:
    """Finite object at ``s_ep_mm`` from the ENTRANCE PUPIL (EP datum).

    ``epl`` = signed distance vertex(surface 1) -> EP, optiland convention
    (EP lies at +epl along the axis). Max field is tan(31.7 deg) about the EP;
    normalised Hy scales it linearly.
    """
    o.set_thickness(s_ep_mm - epl, 0)
    o.fields.fields.clear()
    o.fields.set_type("object_height")
    o.fields.add(y=0)
    o.fields.add(y=s_ep_mm * TAN_MAX)


def square_pupil(n: int):
    g = np.linspace(-1, 1, n)
    X, Y = np.meshgrid(g, g)
    m = X ** 2 + Y ** 2 <= 1
    return X[m], Y[m]


def trace_xy(o, hy, px, py):
    r = o.trace_generic(np.zeros(len(px)), np.full(len(px), hy), px, py, WL)
    x, y, i = (be.to_numpy(v) for v in (r.x, r.y, r.i))
    return x, y, np.isfinite(x) & np.isfinite(y) & (i > 0)


def derive_sd(o35, fu: float, s_ep_mm: float = 700.0, n: int = 281) -> dict:
    """Fu-rule clear semi-diameters on a 35 mm optic (call on the PATENT lens).

    Stop SD = on-axis f/2 marginal height; other surfaces = max bundle height at
    fields {0, Fu} (focus 0.7 m), capped at 0.95|R|. Returns {"sd": [ns floats,
    index = surface number, 0 unused], "stop": int, "capped": [...]}.
    """
    ni = o35.surface_group.num_surfaces - 2
    o35.set_thickness(1e10, 0)
    epl = float(o35.paraxial.EPL())
    set_object(o35, s_ep_mm, epl)
    stop = [k for k in range(1, ni + 1) if o35.surfaces[k].is_stop][0]
    px, py = square_pupil(n)
    ns = ni + 1
    sd = np.zeros(ns)

    def heights(hy):
        o35.trace_generic(np.zeros(len(px)), np.full(len(px), hy), px, py, WL)
        return np.nanmax(np.hypot(be.to_numpy(o35.surface_group.x),
                                  be.to_numpy(o35.surface_group.y)), axis=1)[:ns]

    h0 = heights(0.0)
    hstop = float(h0[stop])
    sd = np.maximum(h0, heights(fu))
    sd[stop] = hstop
    capped = []
    for k in range(1, ni + 1):
        R = float(be.to_numpy(o35.surfaces[k].geometry.radius))
        if np.isfinite(R) and sd[k] > 0.95 * abs(R):
            capped.append(k)
            sd[k] = 0.95 * abs(R)
    return {"sd": [float(v) for v in sd], "stop": stop, "capped": capped, "epl": epl,
            "h_axis": [float(v) for v in h0]}


def apply_sd(o, sd, cap_check=True) -> list:
    """Set RadialAperture on surfaces 1..ni; returns surfaces where the 0.95|R| cap clamps."""
    ni = o.surface_group.num_surfaces - 2
    clamped = []
    for k in range(1, ni + 1):
        v = float(sd[k])
        R = float(be.to_numpy(o.surfaces[k].geometry.radius))
        if cap_check and np.isfinite(R) and v > 0.95 * abs(R):
            clamped.append((k, v, 0.95 * abs(R)))
            v = 0.95 * abs(R)
        o.surfaces[k].aperture = RadialAperture(r_max=v)
    return clamped


def _sag(R: float, h):
    if not np.isfinite(R):
        return np.zeros_like(h)
    return R - np.sign(R) * np.sqrt(R * R - h * h)


MIN_EDGE_GLASS_MM = 0.3
MIN_AIR_OVERLAP_MM = 0.01


def _pair_gap(tab, S, sd, a, b, n=2000):
    (ia, ra, _), (ib, rb, _) = a, b
    t = sum(tab[j][1] for j in range(ia, ib)) * S
    hmax = min(max(sd[ia + 1], sd[ib + 1]), 0.999 * abs(ra * S), 0.999 * abs(rb * S))
    h = np.linspace(0.0, hmax, n)
    return t, h, t + _sag(rb * S, h) - _sag(ra * S, h)


def enforce_mechanical(tab, S, sd, min_edge_mm=MIN_EDGE_GLASS_MM, air_margin_mm=MIN_AIR_OVERLAP_MM):
    """Mechanical feasibility of the Fu-rule semi-diameters (spec item 5 fix).

    Every glass element must keep an edge thickness >= min_edge_mm and every air
    gap zero sag overlap (>= air_margin_mm) for all heights up to the larger of
    its two semi-diameters. Offending semi-diameters are shrunk to the largest
    value that satisfies it. Returns (new_sd list, reductions) with
    reductions = [(surface, before, after, pair, reason)].
    """
    sd = np.array(sd, dtype=float)
    before = sd.copy()
    real = [(i, r, t) for i, (r, t, g) in enumerate(tab) if r != "STOP"]
    why = {}
    for _ in range(100):
        changed = False
        for a, b in zip(real[:-1], real[1:]):
            glass = tab[a[0]][2] is not None
            thr = min_edge_mm if glass else air_margin_mm
            _, h, g = _pair_gap(tab, S, sd, a, b)
            bad = np.where(g < thr)[0]
            if bad.size == 0:
                continue
            k = int(bad[0])
            H = float(h[k - 1]) if k > 0 else 0.0
            for idx in (a[0] + 1, b[0] + 1):
                if sd[idx] > H + 1e-9:
                    sd[idx] = H
                    why[idx] = f"{a[0] + 1}->{b[0] + 1} {'glass edge<%.2f' % thr if glass else 'air overlap'}"
                    changed = True
        if not changed:
            break
    else:
        raise RuntimeError("enforce_mechanical did not converge")
    red = [(k, float(before[k]), float(sd[k]), why.get(k, "")) for k in range(1, len(sd))
           if sd[k] < before[k] - 1e-9]
    return [float(v) for v in sd], red


def check_gaps(tab, S, sd, min_air_mm=MIN_AIR_GAP_MM, edge_glass=0.0, air_margin=0.0):
    """A2: air gaps >= min_air_mm axially; zero sag overlap for air, edge >= edge_glass
    for glass, over the clear aperture (up to the larger semi-diameter).
    sd is indexed by optiland surface number (table index + 1, stop included).

    Returns (ok, rows); rows = per real-surface-pair dict.
    """
    real = [(i, r, t) for i, (r, t, g) in enumerate(tab) if r != "STOP"]
    rows, ok = [], True
    for a, b in zip(real[:-1], real[1:]):
        t, h, gap_h = _pair_gap(tab, S, sd, a, b)
        glass = tab[a[0]][2] is not None
        mn = float(gap_h.min())
        thr = edge_glass if glass else air_margin
        bad = (mn < thr) or (mn <= 0.0) or (not glass and t < min_air_mm)
        ok &= not bad
        rows.append(dict(surf=f"{a[0] + 1}->{b[0] + 1}", kind="glass" if glass else "air",
                         axial_mm=float(t), min_over_clear_mm=mn, ok=not bad))
    return ok, rows
