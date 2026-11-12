"""Pure (numpy-only) helpers for the PSF tensor build gates. No optiland import."""
from __future__ import annotations

G_S_TOL = 0.03          # traced 99.9th-pct disc vs real-ray meridional-scan disc


def ref3_diameter_um(delta_mm: float, xpd_mm: float, xpl_mm: float) -> float:
    """Paraxial (no spherical aberration) full blur diameter, in um, at a sensor plane
    displaced ``delta_mm`` from the paraxial image: the exit-pupil cone converges to the
    paraxial image at distance |XPL|, so radius = delta * (XPD/2) / |XPL| (NOT the
    exit-pupil-to-sensor distance, which under-reads by |XPL|/(|XPL|+delta))."""
    return 1e3 * abs(delta_mm) * xpd_mm / abs(xpl_mm)


def gs_ratio_ok(traced_um: float, marginal_um: float, tol: float = G_S_TOL) -> bool:
    """G-S: the traced disc must match the independent real-ray meridional-scan disc (includes the stop-edge marginal ray) within tol."""
    return bool(marginal_um > 0 and abs(traced_um / marginal_um - 1.0) <= tol)


def t_scale(h1_nominal_mm: float, epd_collimated_mm: float) -> float:
    """Factor turning T (passing / nominal launch disc) into passing / real collimated-EPD
    area. h1_nominal_mm = surface-1 height of the nominal launch-disc edge ray."""
    return (h1_nominal_mm / (0.5 * epd_collimated_mm)) ** 2
