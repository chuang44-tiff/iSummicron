"""Constrained light optimisation of the US 3,006,249 Example II prescription (expired patent; public domain).

Optional tool: it produces a modified prescription JSON for experiments; the
shipped tensors use the unmodified prescription and do not need it.

Run: conda run -n optiland python scripts/build_psf/optimise_lens.py [--out JSON]

Variables : 14 radii (+-3 %) and air gaps a1, a2, a4, a5 (+-0.5 patent units).
Fixed     : stop (0.6 a3), glass thicknesses, glasses. EFL held at 35.000 by a
            global rescale of the whole table.
Merit     : RMS spot, common best focus at infinity, 550 nm. On axis at f/2
            (weight 3); fields 0.5 / 0.7 / 1.0 of 31.7 deg at pupil 0.714 (f/2.8)
            with weights 1 / 1 / 0.25.
Operands  : LSA(1.0) and LSA(0.7) held within +-10 % of the patent values.
Optimiser : scipy Powell, maxfev 600, deterministic.
A2        : air gaps >= 0.05 mm at 35 mm scale; no element overlap at the clear
            semi-diameters (Fu 0.4 rule, derived on the patent lens). Newly
            introduced violations FAIL LOUDLY; violations already present in the
            patent prescription itself are reported (use --strict-overlap to fail
            on those too).
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
from scipy.optimize import minimize, minimize_scalar  # noqa: E402

import _lens_common as lc  # noqa: E402
from _lens_common import be  # noqa: E402

PS = 0.714          # f/2.8 pupil scale for field terms
WAX = 3.0
FIELD_W = [1.0, 1.0, 0.25]
FIELDS = (0.5, 0.7, 1.0)
AIR_IDX = [1, 4, 9, 12]
AIR_NAMES = ["a1", "a2", "a4", "a5"]
RADIUS_BOUND = 0.03
GAP_BOUND = 0.5
RIM_LO, RIM_HI = 0.75 * 2.42, 1.25 * 2.42   # G-L character gate (1.8 - 3.0)
PX, PY = lc.square_pupil(25)
PX61, PY61 = lc.square_pupil(61)

rng = np.random.default_rng(1)
_rr = np.sqrt(rng.random(120000))
_ph = 2 * np.pi * rng.random(120000)
JX, JY = _rr * np.cos(_ph), _rr * np.sin(_ph)


def rms(o, hy, ps):
    r = o.trace_generic(np.zeros(len(PX)), np.full(len(PX), hy), PX * ps, PY * ps, lc.WL)
    x, y = be.to_numpy(r.x), be.to_numpy(r.y)
    ok = np.isfinite(x)
    return 1e3 * np.sqrt(np.var(x[ok]) + np.var(y[ok])) if ok.mean() > 0.9 else 1e3


def lsa(o, p):
    r = o.trace_generic(np.zeros(1), np.zeros(1), np.zeros(1), np.array([p]), lc.WL)
    y, M, N = (float(be.to_numpy(v)[0]) for v in (r.y, r.M, r.N))
    return -y * N / M


def lsa_pair(o):
    return [lsa(o, 1.0) - lsa(o, 0.01), lsa(o, 0.7) - lsa(o, 0.01)]


def rms_set(o, ps_field=PS, ps_axis=1.0):
    return [rms(o, 0, ps_axis)] + [rms(o, h, ps_field) for h in FIELDS]


def best_focus_rms(o, ni, ps_axis=1.0, ps_field=PS, weights=None):
    """7-point common-focus scan around the paraxial image (as the merit does)."""
    weights = weights or [WAX] + FIELD_W
    b0 = float(be.to_numpy(o.surfaces[ni].thickness))
    best, rr, bb = 1e9, None, b0
    for db in np.linspace(-0.4, 0.2, 7):
        o.set_thickness(b0 + db, ni)
        r = rms_set(o, ps_field, ps_axis)
        v = np.sqrt(np.average(np.square(r), weights=weights))
        if v < best:
            best, rr, bb = v, r, b0 + db
    o.set_thickness(b0, ni)
    return best, rr, bb


def character(tab, s, tag, verbose=True):
    """Bokeh character: 20 m on-axis rim/core areal density at fixed R = 0.3825 mm
    (focus 0.7 m, sensor refocused on axis), plus in-focus RMS and field moments."""
    o = lc.mk(tab, s, lc.EPD_TARGET, lc.HALF_FIELD_DEG)
    ni = len(tab)
    o.set_thickness(700.0, 0)
    o.fields.fields.clear()
    o.fields.set_type("object_height")
    o.fields.add(y=0)
    o.fields.add(y=700 * np.tan(np.radians(31.7)))
    o.updater.image_solve()
    b0 = float(be.to_numpy(o.surfaces[ni].thickness))

    def tr(hy, px=PX61, py=PY61):
        x, y, ok = lc.trace_xy(o, hy, px, py)
        return x[ok], y[ok]

    def r0(b):
        o.set_thickness(b, ni)
        x, y = tr(0)
        return 1e3 * np.sqrt(x.var() + y.var())

    b = minimize_scalar(r0, bounds=(b0 - 2, b0 + 1), method="bounded").x
    o.set_thickness(b, ni)
    infoc = []
    for h in (0, 0.5, 0.7):
        x, y = tr(h)
        infoc.append(1e3 * np.sqrt(x.var() + y.var()))
    o.set_thickness(20000.0, 0)
    o.fields.fields.clear()
    o.fields.set_type("object_height")
    o.fields.add(y=0)
    o.fields.add(y=20000 * np.tan(np.radians(31.7)))
    x, y = tr(0, JX, JY)
    r = np.hypot(x - x.mean(), y - y.mean())
    R = 0.3825
    rim = ((r > 0.85 * R) & (r <= R)).mean() / (1 - 0.85 ** 2)
    core = (r < 0.5 * R).mean() / 0.25
    rc = float(rim / core)
    if verbose:
        print(f"  [{tag}] BFD(0.7m)={b:.3f} in-focus RMS um (0, .5, .7) = "
              f"{np.round(infoc, 1).tolist()}  20m on-axis rim/core @R=0.3825 = {rc:.3f}")
    return rc, infoc


def make_table(base, p, curv):
    tab = [list(r) for r in base]
    for k, i in enumerate(curv):
        tab[i][0] = base[i][0] * (1 + p[k])
    for k, i in enumerate(AIR_IDX):
        tab[i][1] = base[i][1] + p[len(curv) + k]
    return [tuple(r) for r in tab]


def build(base, p, curv):
    tab = make_table(base, p, curv)
    s = lc.scale_factor(tab)
    o = lc.mk(tab, s, lc.EPD_TARGET, lc.HALF_FIELD_DEG)
    o.updater.image_solve()
    return o, tab, s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patent", default=str(lc.PATENT_JSON))
    ap.add_argument("--out", default="us3006249_ex2_mod1.json")
    ap.add_argument("--fu", type=float, default=0.4)
    ap.add_argument("--maxfev", type=int, default=600)
    ap.add_argument("--strict-overlap", action="store_true")
    ap.add_argument("--no-gap-penalty", action="store_true", help="debug: do not penalise A2 inside the search")
    a = ap.parse_args()

    lens = lc.load_lens(a.patent)
    base = lc.to_table(lens)
    curv = [i for i, r in enumerate(base) if r[0] != "STOP"]
    nvar = len(curv) + len(AIR_IDX)
    p0 = np.zeros(nvar)
    ni = len(base)

    # patent-lens EFL gate (raw, scale 0.35)
    efl_raw = float(lc.mk(base, 0.35, lc.EPD_TARGET, lc.HALF_FIELD_DEG).paraxial.f2())
    print(f"patent lens: EFL x0.35 = {efl_raw:.3f} mm (gate 35.21 +- 0.1)")

    # A2 reference apertures: Fu rule on the PATENT lens
    o_pat, s_pat = lc.scaled_optic(base)
    sdinfo = lc.derive_sd(o_pat, a.fu)
    sd, sd_red = lc.enforce_mechanical(base, s_pat, sdinfo["sd"])
    for k, b4, af, why in sd_red:
        print(f"A2 aperture rule: surface {k} semi-diameter {b4:.3f} -> {af:.3f} mm ({why})")
    MECH = dict(edge_glass=lc.MIN_EDGE_GLASS_MM, air_margin=lc.MIN_AIR_OVERLAP_MM)
    ok0, rows0 = lc.check_gaps(base, s_pat, sd, **MECH)
    base_bad = {r["surf"] for r in rows0 if not r["ok"]}
    print(f"A2 patent baseline (Fu {a.fu} semi-diameters, scale {s_pat:.5f}): "
          f"{'OK' if ok0 else 'PRE-EXISTING VIOLATIONS at ' + ', '.join(sorted(base_bad))}")
    for r in rows0:
        if not r["ok"]:
            print(f"    patent {r['surf']} {r['kind']}: axial {r['axial_mm']:.3f} mm, "
                  f"min over clear aperture {r['min_over_clear_mm']:.3f} mm")
    base_min = {r["surf"]: r["min_over_clear_mm"] for r in rows0}

    def gap_violation(tab, s):
        """Smooth A2 violation in mm (0 when feasible): shortfall of every pair's minimum
        gap/edge against its threshold, plus air gaps below 0.05 mm axially."""
        _, rows = lc.check_gaps(tab, s, sd, **MECH)
        v = 0.0
        for r in rows:
            thr = lc.MIN_EDGE_GLASS_MM if r["kind"] == "glass" else lc.MIN_AIR_OVERLAP_MM
            v += max(0.0, thr - r["min_over_clear_mm"])
            if r["kind"] == "air":
                v += max(0.0, lc.MIN_AIR_GAP_MM - r["axial_mm"])
        return v

    L0 = None

    def merit(p, verbose=False):
        o, tab, s = build(base, p, curv)
        viol = 0.0 if a.no_gap_penalty else gap_violation(tab, s)
        best, rr, _ = best_focus_rms(o, ni)
        L = lsa_pair(o)
        pen = (sum(((L[i] - L0[i]) / (0.1 * abs(L0[i]) + 1e-6)) ** 2 for i in range(2))
               if L0 else 0.0)
        if verbose:
            print("  rms(ax f/2; .5,.7,1.0 at pupil %.3f) um =" % PS, np.round(rr, 1),
                  "LSA", np.round(L, 4), "merit", round(best, 1))
        return best * (1 + 0.1 * pen) * (1 + 100.0 * viol)

    o, tab, s = build(base, p0, curv)
    L0 = lsa_pair(o)
    print("RAW (patent lens, EFL 35):")
    merit(p0, True)
    rc_raw, _ = character(tab, s, "patent")
    f2raw = {}
    best, rr, bf = best_focus_rms(o, ni, 1.0, 1.0)      # all fields at f/2
    f2raw["f/2"] = rr
    best, rr, bf = best_focus_rms(o, ni, 1.0, PS)
    f2raw["f/2.8"] = rr

    lo = np.r_[np.full(len(curv), -RADIUS_BOUND), np.full(len(AIR_IDX), -GAP_BOUND)]
    hi = -lo
    t0 = time.time()
    res = minimize(lambda p: merit(np.clip(p, lo, hi)), p0, method="Powell",
                   bounds=list(zip(lo, hi)),
                   options=dict(maxfev=a.maxfev, xtol=1e-3, ftol=1e-3))
    p = np.clip(res.x, lo, hi)
    print(f"OPT: Powell nfev={res.nfev} {time.time() - t0:.0f}s")

    # ---- result checks ------------------------------------------------------
    o, tab, s = build(base, p, curv)
    print("OPT (mod1):")
    merit(p, True)
    rc_opt, _ = character(tab, s, "mod1")
    f2opt = {}
    _, rr, _ = best_focus_rms(o, ni, 1.0, 1.0)
    f2opt["f/2"] = rr
    _, rr, _ = best_focus_rms(o, ni, 1.0, PS)
    f2opt["f/2.8"] = rr
    L1 = lsa_pair(o)
    lsa_rel = [L1[i] / L0[i] - 1 for i in range(2)]

    efl = float(o.paraxial.f2())
    fno = float(o.paraxial.FNO())
    r = o.trace_generic(np.zeros(1), np.ones(1), np.zeros(1), np.zeros(1), lc.WL)
    ychief = abs(float(be.to_numpy(r.y)[0]))
    dist = 100 * (ychief - efl * np.tan(np.radians(31.7))) / (efl * np.tan(np.radians(31.7)))

    print("\nBefore/after RMS spot (um), common best focus at infinity, 550 nm")
    print("  pupil     case    on-axis   f0.5    f0.7    f1.0")
    for key in ("f/2", "f/2.8"):
        print(f"  {key:6s}  patent  " + "  ".join(f"{v:7.1f}" for v in f2raw[key]))
        print(f"  {key:6s}  mod1    " + "  ".join(f"{v:7.1f}" for v in f2opt[key]))
    print(f"  rim/core (20 m on axis, R=0.3825 mm): patent {rc_raw:.3f} -> mod1 {rc_opt:.3f}")
    print(f"  LSA(1.0), LSA(0.7) mm: patent {np.round(L0, 4).tolist()} -> mod1 {np.round(L1, 4).tolist()}"
          f"  (rel {100 * lsa_rel[0]:+.1f}%, {100 * lsa_rel[1]:+.1f}%)")

    # delta table
    delta = []
    for k, i in enumerate(curv):
        delta.append(dict(param=f"R{k + 1}", kind="radius", patent=base[i][0],
                          mod1=tab[i][0], delta=tab[i][0] - base[i][0], pct=100 * p[k]))
    for k, i in enumerate(AIR_IDX):
        v = p[len(curv) + k]
        delta.append(dict(param=AIR_NAMES[k], kind="air_gap", patent=base[i][1],
                          mod1=tab[i][1], delta=v, pct=100 * v / base[i][1]))
    print("\nDelta table (patent units, f = 100)")
    print("  param  kind      patent      mod1        delta     pct")
    for d in delta:
        print(f"  {d['param']:5s}  {d['kind']:8s} {d['patent']:9.4f} {d['mod1']:10.4f} "
              f"{d['delta']:+9.4f} {d['pct']:+7.2f}%")
    max_dr = float(np.abs(p[:len(curv)]).max())

    # A2
    ok1, rows1 = lc.check_gaps(tab, s, sd, **MECH)
    print("\nA2 gap check (mod1, Fu semi-diameters of the patent lens)")
    print("  pair    kind  axial_mm  min_over_clear_mm  patent_min  status")
    fail = []
    for r_ in rows1:
        pre = r_["surf"] in base_bad
        status = "ok" if r_["ok"] else ("PRE-EXISTING" if pre else "FAIL")
        if r_["kind"] == "air" and r_["axial_mm"] < lc.MIN_AIR_GAP_MM:
            status = "FAIL(air<0.05mm)"
        if status.startswith("FAIL") or (a.strict_overlap and not r_["ok"]):
            fail.append(r_["surf"])
        if pre:
            status = "PRE-EXISTING(d=%+.3f)" % (r_["min_over_clear_mm"] - base_min[r_["surf"]])
        print(f"  {r_['surf']:7s} {r_['kind']:5s} {r_['axial_mm']:8.4f} {r_['min_over_clear_mm']:15.4f}"
              f"   {base_min[r_['surf']]:9.4f}  {status}")

    # G-L
    gates = {
        "raw EFLx0.35 = 35.21+-0.1": abs(efl_raw - 35.21) <= 0.1,
        "EFL 35.000+-0.01": abs(efl - 35.0) <= 0.01,
        "f/2.00+-0.01": abs(fno - 2.0) <= 0.01,
        "|distortion|<2%": abs(dist) < 2.0,
        "max dR/R<=3%": max_dr <= RADIUS_BOUND + 1e-9,
        f"rim/core in [{RIM_LO:.2f},{RIM_HI:.2f}]": RIM_LO <= rc_opt <= RIM_HI,
        "LSA operands within 10%": all(abs(v) <= 0.10 + 1e-9 for v in lsa_rel),
        "A2 no new overlap / air>=0.05mm": not fail,
    }
    print(f"\nG-L: EFL={efl:.4f} f/{fno:.4f} distortion@1.0={dist:+.3f}% max|dR/R|={100 * max_dr:.2f}%")
    for k, v in gates.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    if fail:
        print(f"A2 FAIL LOUDLY: {sorted(set(fail))}", file=sys.stderr)

    out = copy.deepcopy(lens)
    out["name"] = lens["name"] + " (mod1: light constrained optimisation)"
    out["modified_from_patent"] = True
    out["source_lens"] = Path(a.patent).name
    for k, i in enumerate(curv):
        out["surfaces"][i]["radius"] = float(tab[i][0])
    for k, i in enumerate(AIR_IDX):
        out["surfaces"][i]["thickness"] = float(tab[i][1])
    out["delta_table"] = delta
    out["optimisation"] = dict(
        method="scipy Powell, maxfev %d, deterministic" % a.maxfev, nfev=int(res.nfev),
        rms_um_patent=f2raw, rms_um_mod1=f2opt, rim_core_patent=rc_raw, rim_core_mod1=rc_opt,
        lsa_patent=L0, lsa_mod1=L1, efl_mm=efl, fno=fno, distortion_pct=float(dist),
        scale_to_35mm=float(s), gates={k: bool(v) for k, v in gates.items()},
        a2_rows=rows1, a2_patent_preexisting=sorted(base_bad))
    if not all(gates.values()):
        print("G-L FAILED: " + ", ".join(k for k, v in gates.items() if not v)
              + f"; NOT writing {a.out}", file=sys.stderr)
        sys.exit(2)
    Path(a.out).write_text(json.dumps(out, indent=2))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
