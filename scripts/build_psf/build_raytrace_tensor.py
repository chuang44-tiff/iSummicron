"""Ray-traced PSF tensor builder for the iSummicron render pipeline (optiland).

Builds the ``(R=12, D=11, n, n)`` kernel tensor consumed by ``isummicron render``
(``--tensor PATH``) from a lens prescription JSON, by real-ray tracing with
optiland (MIT licence; PyPI ``optiland``). Run it in the optiland environment
(``environment-optiland.yml``); the ``summicron`` package itself never imports
optiland. A second interpreter that can import ``summicron`` (``--gt-python``)
is used for the load-tensor gate.

Lens prescriptions (``scripts/build_psf/lenses/``):
  * ``us3006249_ex2.json``: 35 mm f/2, ray-traced from the expired US patent
    3,006,249, Example II (public domain);
  * ``optiland_doublegauss_50f2.json``: 50 mm f/2 double-Gauss from the optiland
    sample objective DoubleGauss (MIT).

Exact commands for the two tensors shipped with the package (run from the
repository root; ``<DIR>`` is any output directory, ~4.5 min each, ``--procs``
defaults to 12). The output ``kernels`` array is deterministic (fixed seed):

  35 mm f/2 (EFL 35.0, EPD 17.5, half field 31.7 deg are the defaults):
    conda run -n optiland python scripts/build_psf/build_raytrace_tensor.py \
        --patent-lens scripts/build_psf/lenses/us3006249_ex2.json \
        --fu 0.4 --rays-grid 1600 --seed 24 \
        --gt-python "conda run -n isummicron python" \
        --out <DIR>/psf_lens35_us3006249_fu0.4.npz

  50 mm f/2 (the focal length, pupil and field are set through the environment):
    ISUMMICRON_BUILD_PSF_EFL_MM=50 ISUMMICRON_BUILD_PSF_EPD_MM=25 ISUMMICRON_BUILD_PSF_HALF_FIELD_DEG=23.3962047 \
    conda run -n optiland python scripts/build_psf/build_raytrace_tensor.py \
        --patent-lens scripts/build_psf/lenses/optiland_doublegauss_50f2.json \
        --fu 0.4 --rays-grid 1600 --seed 24 \
        --gt-python "conda run -n isummicron python" \
        --out <DIR>/psf_lens50_doublegauss_fu0.4.npz

Design:
  * lens scaled to the target EFL/EPD, robust ray aiming, 550 nm;
  * Fu-rule clear semi-diameters are derived ONCE on the prescription given by
    ``--patent-lens`` and reused verbatim for ``--lens`` (0.95|R| cap re-checked);
  * focus at 0.7 m from the entrance pupil (on-axis RMS minimum), sensor frozen;
  * fields tan(theta) = r_center * tan(half field); objects at the schema
    object distances measured from the entrance pupil;
  * ~2e6 stratified-jittered pupil cells per node (``--rays-grid`` squared cells
    over the launch pupil); cells that cannot contain survivors are skipped
    (same cell density, verified against the survivor bounding box);
  * rays centred on the chief ray; d_out = outward displacement, d_tan = dx;
  * the PSF is never cropped, clipped or smoothed: the kernel box is sized to
    the PSF (one odd size for all nodes) and the in-box energy is reported.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import multiprocessing as mp
import shlex
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402

import _gates as gates_mod  # noqa: E402
import _lens_common as lc  # noqa: E402
from _lens_common import be  # noqa: E402

OVER = 1.6            # launch pupil = OVER x nominal EPD, so the stop SD does the clipping
SEED = 24             # default; overridden by --seed (the shipped tensors use 24)
PITCH_FACTOR = 1.4422  # pitch_um = EFL_um * 1.4422 / 5712 (A17.5 rule)
CORPUS_REF = 5712
CHUNK_ROWS = 100
MIN_BOX = 183


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def git_sha() -> str:
    """HEAD at build time; suffixed ``-dirty`` if scripts/build_psf or summicron/psf_model are modified."""
    try:
        sha = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
        st = subprocess.check_output(
            ["git", "-C", str(ROOT), "status", "--porcelain", "--", "scripts/build_psf", "summicron/psf_model"],
            text=True).strip()
        return sha + ("-dirty" if st else "")
    except Exception:
        return "unknown"


def check_lens_gates(lens, path):
    """Refuse a lens JSON that recorded gate failures or lacks all-true optimiser gates."""
    fails = lens.get("gate_failures")
    if fails:
        raise SystemExit(f"refusing lens {path}: gate_failures={fails}")
    gates = (lens.get("optimisation") or {}).get("gates")
    if gates is not None and not all(gates.values()):
        bad = [k for k, v in gates.items() if not v]
        raise SystemExit(f"refusing lens {path}: optimiser gates not all true: {bad}")


def shipped_axes():
    """Schema axes (``ray_kernel`` constants): edges, bin centres, object distances (m)."""
    rk = import_ray_kernel()
    edges = rk.RADIAL_EDGES.astype(np.float64)
    dist = rk.OBJECT_DISTANCES_M.astype(np.float64)
    return edges, 0.5 * (edges[:-1] + edges[1:]), dist


# --------------------------------------------------------------------------- optic
def make_traced_optic(lens_json, sd, bfd=None):
    """Optic at EFL 35 / EPD 17.5 with Fu-rule apertures, robust aiming, EPD x OVER."""
    tab = lc.to_table(lc.load_lens(lens_json))
    o, s = lc.scaled_optic(tab, aiming="robust")
    o.set_thickness(1e10, 0)
    epl = float(o.paraxial.EPL())
    clamped = lc.apply_sd(o, sd)
    o.set_aperture("EPD", lc.EPD_TARGET * OVER)
    if bfd is not None:
        o.set_thickness(bfd, len(tab))
    return o, tab, s, epl, clamped


def find_focus(o, ni, epl, s_ep_mm=700.0):
    from scipy.optimize import minimize_scalar
    lc.set_object(o, s_ep_mm, epl)
    o.updater.image_solve()
    b0 = float(be.to_numpy(o.surfaces[ni].thickness))
    px, py = lc.square_pupil(281)

    def f(b):
        o.set_thickness(b, ni)
        x, y, ok = lc.trace_xy(o, 0.0, px, py)
        return float(np.sqrt(np.var(x[ok]) + np.var(y[ok])))

    b = float(minimize_scalar(f, bounds=(b0 - 2, b0 + 2), method="bounded",
                              options=dict(xatol=1e-6)).x)
    o.set_thickness(b, ni)
    return b, f(b)


# --------------------------------------------------------------------------- tracing a node
def _alive_chunk(o, hy, px, py, ni, sd=None):
    r = o.trace_generic(np.zeros(len(px)), np.full(len(px), hy), px, py, lc.WL)
    x, y, i = (be.to_numpy(v) for v in (r.x, r.y, r.i))
    ok = np.isfinite(x) & np.isfinite(y) & (i > 0)
    Z = be.to_numpy(o.surface_group.z)[1:ni + 2]
    back = np.zeros(len(px), bool)
    with np.errstate(invalid="ignore"):
        back = np.any(np.diff(Z, axis=0) < -1e-9, axis=0)
    exceed = np.zeros(len(px), bool)
    if sd is not None:
        H = np.hypot(be.to_numpy(o.surface_group.x), be.to_numpy(o.surface_group.y))
        lim = np.asarray(sd[:ni + 1], dtype=float)[:, None] * (1 + 1e-6) + 1e-9
        with np.errstate(invalid="ignore"):
            exceed = np.any(H[1:ni + 1] > lim[1:ni + 1], axis=0)
    return x, y, ok, back, exceed


def trace_node(o, hy, rng_key, grid, ni, sd):
    """Trace one node. Returns (x, y, stats) for survivors (mm image coordinates)."""
    # coarse pass -> survivor bounding box in launch-pupil coordinates
    cpx, cpy = lc.square_pupil(281)
    _, _, cok, _, _ = _alive_chunk(o, hy, cpx, cpy, ni)
    if cok.sum() < 20:
        raise RuntimeError("coarse pass: <20 surviving rays (node dead)")
    cell = 2.0 / grid
    margin = 0.03 + 4 * cell
    x0, x1 = cpx[cok].min() - margin, cpx[cok].max() + margin
    y0, y1 = cpy[cok].min() - margin, cpy[cok].max() + margin
    ia0, ia1 = max(0, int(np.floor((x0 + 1) / cell))), min(grid, int(np.ceil((x1 + 1) / cell)))
    ib0, ib1 = max(0, int(np.floor((y0 + 1) / cell))), min(grid, int(np.ceil((y1 + 1) / cell)))
    rng = np.random.default_rng(rng_key)
    xs, ys, pxs, pys = [], [], [], []
    n_launched = n_back = n_exc = 0
    for rs in range(ib0, ib1, CHUNK_ROWS):
        re = min(ib1, rs + CHUNK_ROWS)
        rows = np.arange(rs, re)
        cols = np.arange(ia0, ia1)
        CC, RR = np.meshgrid(cols, rows)
        u = rng.random(CC.shape)
        v = rng.random(CC.shape)
        px = (CC + u) * cell - 1.0
        py = (RR + v) * cell - 1.0
        m = px ** 2 + py ** 2 <= 1.0
        px, py = px[m], py[m]
        n_launched += px.size
        x, y, ok, back, exc = _alive_chunk(o, hy, px, py, ni, sd)
        n_back += int((back & ok).sum())
        n_exc += int((exc & ok).sum())
        xs.append(x[ok])
        ys.append(y[ok])
        pxs.append(px[ok])
        pys.append(py[ok])
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    px = np.concatenate(pxs)
    py = np.concatenate(pys)
    if x.size < 1000:
        raise RuntimeError(f"only {x.size} surviving rays")
    # survivors must not touch the box edge (bbox adequacy) nor the launch rim
    edge = 2 * cell
    if (px.min() < x0 - 1e-12 + edge and x0 > -1 + edge) or (px.max() > x1 - edge and x1 < 1 - edge) \
            or (py.min() < y0 + edge and y0 > -1 + edge) or (py.max() > y1 - edge and y1 < 1 - edge):
        raise RuntimeError("survivors reach the traced-region boundary; widen margin")
    rim = float((np.hypot(px, py) > 0.95).mean())
    if rim > 0.0:
        raise RuntimeError(f"survivors reach the launch-pupil rim (frac {rim:.4f}); raise OVER")
    if n_back or n_exc:
        raise RuntimeError(f"mechanically infeasible rays survive: {n_back} backward z-step, "
                           f"{n_exc} beyond a semi-diameter (of {x.size})")
    # centring: chief ray, else nominal-EPD centroid
    cx, cy, cok1, _, _ = _alive_chunk(o, hy, np.zeros(1), np.zeros(1), ni)
    if cok1[0]:
        xc, yc, ref = float(cx[0]), float(cy[0]), "chief"
    else:
        nom = np.hypot(px, py) <= 1.0 / OVER
        xc, yc, ref = float(x[nom].mean()), float(y[nom].mean()), "centroid(nominal EPD; chief vignetted)"
    sgn = np.sign(yc) if (hy != 0 and yc != 0) else 1.0
    d_out = (y - yc) * sgn        # +y field: image y negative -> d_out = -dy
    d_tan = x - xc
    area_nom = np.pi / OVER ** 2
    T = x.size * 4.0 / (grid ** 2) / area_nom        # density grid^2/4 per unit area
    stats = dict(n_launched=int(n_launched), n_alive=int(x.size), T=float(T), ref=ref,
                 xc=xc, yc=yc, unphysical_frac=float(n_back / x.size),
                 cells=[ia0, ia1, ib0, ib1])
    return d_out.astype(np.float64), d_tan.astype(np.float64), stats


def worker(args):
    (lens_json, sd, ridx, rc, bfd, dists, grid, cache_dir, fu, seed) = args
    out = []
    for d, sm in enumerate(dists):
        node = f"r{ridx:02d}_d{d:02d}"
        try:
            t0 = time.time()
            o, tab, s, epl, _ = make_traced_optic(lens_json, sd, bfd)   # fresh optic: nodes independent
            ni = len(tab)
            lc.set_object(o, sm * 1000.0, epl)
            o.set_thickness(bfd, ni)
            dout, dtan, st = trace_node(o, rc, [seed, ridx, d], grid, ni, sd)
            st.update(node=node, r=ridx, d=d, seconds=time.time() - t0)
            np.savez(Path(cache_dir) / f"{node}.npz", d_out=dout, d_tan=dtan)
            out.append(st)
        except Exception as e:  # fail loudly with the node id
            raise RuntimeError(f"node {node} (r_center={rc:.4f}, s={sm} m): {type(e).__name__}: {e}") from e
    return out


# --------------------------------------------------------------------------- helpers
def import_ray_kernel():
    return importlib.import_module("summicron.psf_model.ray_kernel")


def need_half_px(dout, dtan, pitch_mm):
    """99.5th percentile of Chebyshev displacement (px) -> half box needed."""
    cheb = np.maximum(np.abs(dout), np.abs(dtan)) / pitch_mm
    return float(np.percentile(cheb, 99.5)), float(np.percentile(cheb, 99.9)), float(cheb.max())


def ring_energy(k, w=2):
    m = np.ones_like(k, bool)
    m[w:-w, w:-w] = False
    return float(k[m].sum())


def load_node(cache_dir, r, d):
    with np.load(Path(cache_dir) / f"r{r:02d}_d{d:02d}.npz") as z:
        return z["d_out"], z["d_tan"]


def collimated_epd_and_tscale(o, ni):
    """Real collimated EPD (largest passing surface-1 height x2) and the factor that
    re-normalises T from the nominal launch disc to that EPD. Metadata only."""
    from optiland.rays import RealRays
    o.set_thickness(1e10, 0)
    n = 6001
    h = np.linspace(0.0, 1.6 * lc.EPD_TARGET / 2, n)
    r = RealRays(np.zeros(n), h, np.full(n, -1.0), np.zeros(n), np.zeros(n), np.ones(n),
                 np.ones(n), np.full(n, lc.WL))
    o.surface_group.trace(r)
    ok = (be.to_numpy(r.i) > 0) & np.isfinite(be.to_numpy(r.y))
    hmax = float(h[ok].max())
    lc.set_object(o, 1e7, float(o.paraxial.EPL()))
    o.trace_generic(np.zeros(1), np.zeros(1), np.zeros(1), np.array([1.0 / OVER]), lc.WL)
    h1_nom = float(be.to_numpy(o.surface_group.y)[1, 0])
    return 2 * hmax, h1_nom, gates_mod.t_scale(h1_nom, 2 * hmax)


def marginal_disc_um(o, ni):
    """Independent real-ray reference for G-S (on axis, current object/sensor of ``o``).

    Dense 1-D meridional scan of the whole launch pupil (0..1, 8001 rays, bisect-refined
    stop edge): returns (full disc diameter = 2*max|y| at the sensor in um, stop-edge
    marginal-ray diameter in um). The scan max is the gate reference: it follows real
    spherical aberration (zonal rays can exceed the marginal ray) but shares no sampling
    or percentile logic with the 2-D tensor trace. The edge value is information only."""
    ps = np.linspace(0.0, 1.0, 8001)
    x, y, ok = lc.trace_xy(o, 0.0, np.zeros_like(ps), ps)
    idx = np.where(ok)[0]
    lo = ps[idx.max()]
    hi = ps[min(idx.max() + 1, len(ps) - 1)]
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        _, _, okm = lc.trace_xy(o, 0.0, np.zeros(1), np.array([mid]))
        if okm[0]:
            lo = mid
        else:
            hi = mid
    _, ye, _ = lc.trace_xy(o, 0.0, np.zeros(1), np.array([lo]))
    return 2e3 * float(np.abs(y[ok]).max()), 2e3 * abs(float(ye[0]))


def thin_lens_diameter(A, f, s0, s):
    return A * f * abs(s - s0) / (s * (s0 - f))


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lens", default=None,
                    help="lens JSON to trace (default: the --patent-lens file)")
    ap.add_argument("--fu", type=float, required=True)
    ap.add_argument("--rays-grid", type=int, default=1600)
    ap.add_argument("--out", required=True, help="output tensor path (.npz, compressed)")
    ap.add_argument("--patent-lens", default=str(lc.PATENT_JSON),
                    help="reference prescription for the Fu-rule apertures")
    ap.add_argument("--seed", type=int, default=SEED, help="ray-jitter seed (shipped tensors: 24)")
    ap.add_argument("--gt-python", default=None,
                    help="interpreter (shell-split) that can import summicron, used for the "
                         "load_tensor gate; default: this interpreter")
    ap.add_argument("--procs", type=int, default=12)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--reuse-cache", action="store_true", help="skip tracing; reuse cached rays")
    ap.add_argument("--verify-nodes", type=int, default=6,
                    help="G-D: re-trace this many random nodes and compare ray hashes")
    a = ap.parse_args()
    a.lens = a.lens or a.patent_lens
    seed = a.seed

    t_all = time.time()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    name = out.stem
    cache = Path(a.cache_dir) if a.cache_dir else out.parent / f"cache_{name}"
    cache.mkdir(parents=True, exist_ok=True)
    rk = import_ray_kernel()

    lens = lc.load_lens(a.lens)
    check_lens_gates(lens, a.lens)
    variant = lens.get("prescription_variant") or ("mod1" if lens.get("modified_from_patent") else "patent")
    edges, rcent, dists = shipped_axes()
    R, D = len(rcent), len(dists)
    timings = {}

    # ---- apertures: derived ONCE on the PATENT lens
    t0 = time.time()
    pat_tab = lc.to_table(lc.load_lens(a.patent_lens))
    o_pat, _ = lc.scaled_optic(pat_tab)
    sdinfo = lc.derive_sd(o_pat, a.fu)
    sd_fu = sdinfo["sd"]
    s_pat = lc.scale_factor(pat_tab)
    sd, sd_red = lc.enforce_mechanical(pat_tab, s_pat, sd_fu)
    print(f"[{name}] variant={variant} fu={a.fu}; Fu-rule sd_mm = {np.round(sd_fu[1:], 3).tolist()} "
          f"(stop surface {sdinfo['stop']}, capped {sdinfo['capped']})")
    print(f"[{name}] mechanical feasibility (glass edge >= {lc.MIN_EDGE_GLASS_MM} mm, air overlap 0), "
          f"derived on the patent lens; reduced semi-diameters:")
    for k, b4, af, why in sd_red:
        print(f"[{name}]    surface {k:2d}: {b4:.3f} -> {af:.3f} mm  ({why})")
    if not sd_red:
        print(f"[{name}]    (none)")
    for k, h in enumerate(sdinfo["h_axis"]):
        if k >= 1 and sd[k] < h - 1e-6:
            raise SystemExit(f"FAIL: feasible sd[{k}]={sd[k]:.3f} < on-axis f/2 marginal height {h:.3f}; "
                             "the mechanical limit would stop down the on-axis beam")
    print(f"[{name}] final sd_mm = {np.round(sd[1:], 3).tolist()}")

    # ---- focus
    o, tab, s, epl, clamped = make_traced_optic(a.lens, sd)
    ni = len(tab)
    if clamped:
        print(f"[{name}] 0.95|R| cap re-check: CLAMPED {clamped}")
    else:
        print(f"[{name}] 0.95|R| cap re-check: no clamping needed")
    # target-lens check: glass edge >= 0.3 mm, air gaps strictly non-overlapping (> 0). The 0.01 mm
    # margin is only used when deriving sd on the patent lens; a shared sd that leaves a modified
    # lens at 0.005 mm clearance is still overlap-free (the runtime backward-step check is the guard).
    ok_gap, gap_rows = lc.check_gaps(tab, s, sd, edge_glass=lc.MIN_EDGE_GLASS_MM, air_margin=0.0)
    if not ok_gap:
        for r_ in gap_rows:
            if not r_["ok"]:
                print(f"[{name}] INFEASIBLE {r_['surf']} {r_['kind']}: axial {r_['axial_mm']:.3f} mm, "
                      f"min over clear aperture {r_['min_over_clear_mm']:.3f} mm")
        raise SystemExit("FAIL: shared patent-derived semi-diameters are mechanically infeasible on this lens")
    for r_ in gap_rows:
        if not r_["ok"]:
            print(f"[{name}] gap note {r_['surf']} {r_['kind']}: axial {r_['axial_mm']:.3f} mm, "
                  f"min over clear aperture {r_['min_over_clear_mm']:.3f} mm")
    bfd, rms_ax = find_focus(o, ni, epl)
    efl = float(o.paraxial.f2())
    epd = float(o.paraxial.EPD()) / OVER
    print(f"[{name}] EFL={efl:.4f} EPD={epd:.4f} EPL={epl:.4f} scale={s:.6f} "
          f"BFD(0.7 m EP-datum)={bfd:.5f} on-axis RMS={rms_ax * 1e3:.2f} um")
    timings["setup_s"] = time.time() - t0

    pitch_mm = efl * PITCH_FACTOR / CORPUS_REF          # mm (EFL_um*1.4422/5712 um -> mm)
    pitch2048_mm = pitch_mm * CORPUS_REF / 2048.0
    print(f"[{name}] pitch@5712 = {pitch_mm * 1e3:.4f} um; @2048 = {pitch2048_mm * 1e3:.4f} um")

    # ---- trace
    t0 = time.time()
    stats = {}
    if not a.reuse_cache:
        jobs = [(a.lens, sd, r, float(rcent[r]), bfd, [float(x) for x in dists], a.rays_grid,
                 str(cache), a.fu, seed) for r in range(R)]
        ctx = mp.get_context("spawn")
        with ctx.Pool(min(a.procs, R)) as pool:
            for res in pool.imap_unordered(worker, jobs):
                for st in res:
                    stats[(st["r"], st["d"])] = st
                print(f"[{name}] r-bin {res[0]['r']:2d} done "
                      f"({sum(x['seconds'] for x in res):.0f} s node-time)", flush=True)
        json.dump({f"{k[0]},{k[1]}": v for k, v in stats.items()}, open(cache / "stats.json", "w"))
    else:
        raw = json.load(open(cache / "stats.json"))
        stats = {tuple(int(i) for i in k.split(",")): v for k, v in raw.items()}
    timings["trace_s"] = time.time() - t0

    # ---- G-D (cheap path): re-trace a few random nodes and compare ray hashes
    gd_ok, gd_msgs = True, []
    if a.verify_nodes and not a.reuse_cache:
        rs = np.random.default_rng(7)
        picks = [(int(rs.integers(R)), int(rs.integers(1, D))) for _ in range(a.verify_nodes)]
        for (r, d) in picks:
            o2, tab2, _, epl2, _ = make_traced_optic(a.lens, sd, bfd)
            lc.set_object(o2, float(dists[d]) * 1000.0, epl2)
            o2.set_thickness(bfd, len(tab2))
            do2, dt2, _ = trace_node(o2, float(rcent[r]), [seed, r, d], a.rays_grid, len(tab2), sd)
            d0, t0_ = load_node(cache, r, d)
            same = np.array_equal(do2, d0) and np.array_equal(dt2, t0_)
            gd_ok &= same
            gd_msgs.append(f"r{r:02d}_d{d:02d}:{'identical' if same else 'DIFFERENT'}")

    # ---- kernel box: sized to the PSF (never clip the PSF to the box)
    t0 = time.time()
    half_need = 0.0
    p995 = np.zeros((R, D))
    p999 = np.zeros((R, D))
    for r in range(R):
        for d in range(1, D):
            dout, dtan = load_node(cache, r, d)
            h995, h999, hmax = need_half_px(dout, dtan, pitch_mm)
            p995[r, d], p999[r, d] = h995, h999
            half_need = max(half_need, h995)
    n_box = max(MIN_BOX, 2 * int(np.ceil(half_need) + 3) + 1)
    for _ in range(40):                       # also keep the outer 2 px ring < 1e-3
        worst_ring = 0.0
        for r in range(R):
            for d in range(1, D):
                dout, dtan = load_node(cache, r, d)
                k = rk.rays_to_kernel(dout, dtan, pitch_mm, size=n_box)
                worst_ring = max(worst_ring, ring_energy(k))
        if worst_ring < 1e-3:
            break
        n_box += 20 if n_box % 2 == 1 else 21
        n_box = n_box if n_box % 2 == 1 else n_box + 1
    print(f"[{name}] kernel box chosen: {n_box} x {n_box} (needed half-extent for 99.5% = "
          f"{half_need:.1f} px at 5712; worst outer-2px ring {worst_ring:.2e})")

    # ---- final kernels
    kernels = np.zeros((R, D, n_box, n_box))
    outside = np.zeros((R, D))
    r99 = np.zeros((R, D))
    off_px = np.zeros((R, D))
    Tm = np.zeros((R, D))
    Tm_nominal = np.zeros((R, D))
    o_t, tab_t, _, _, _ = make_traced_optic(a.lens, sd, bfd)
    epd_coll, h1_nom, t_fac = collimated_epd_and_tscale(o_t, len(tab_t))
    print(f"[{name}] collimated EPD {epd_coll:.4f} mm; nominal-launch edge at h1 {h1_nom:.4f} mm; "
          f"T rescale x{t_fac:.4f}")
    for r in range(R):
        for d in range(D):
            st = stats[(r, d)]
            Tm_nominal[r, d] = st["T"]
            Tm[r, d] = st["T"] * t_fac
            dout, dtan = load_node(cache, r, d)
            rr = np.hypot(dout, dtan) / pitch_mm
            r99[r, d] = np.percentile(rr, 99)
            off_px[r, d] = np.hypot(st["xc"], st["yc"]) / pitch_mm   # chief height only (info)
            if d == 0:
                kernels[r, d] = rk.make_delta(n_box)
                continue
            k, outf = rk.rays_to_kernel(dout, dtan, pitch_mm, size=n_box, return_outside=True)
            kernels[r, d] = k
            outside[r, d] = outf
    pixel_pitch_um = np.full((R, D), pitch_mm * 1e3)
    meta = dict(
        transmission=Tm, transmission_nominal_disc=Tm_nominal,
        epd_collimated_mm=np.array(epd_coll), t_norm_note=np.array(
            "transmission = passing-bundle area / real collimated-EPD area (on axis ~1); "
            "transmission_nominal_disc = old normalisation vs the nominal launch disc"), fu=np.array(a.fu), apertures_assumed=np.array(True),
        aperture_note=np.array("Fu-rule stylised clear semi-diameters derived on the reference prescription given by --patent-lens; "
                               "Fu is a stylisation, not a measured aperture"),
        lens_citation=np.array(lens.get("citation", "")), prescription_variant=np.array(variant),
        modified_from_patent=np.array(bool(lens.get("modified_from_patent", False))),
        sd_mm=np.array(sd),
        sd_mm_index_note=np.array("sd_mm and sd_mm_fu_rule are indexed by optic surface; index 0 is the "
                                  "object surface (unused placeholder), apertures start at index 1"),
        sd_mm_fu_rule=np.array(sd_fu),
        sd_reductions=np.array(json.dumps([[k, b4, af, why] for k, b4, af, why in sd_red])),
        min_edge_glass_mm=np.array(lc.MIN_EDGE_GLASS_MM), pixel_pitch_um=pixel_pitch_um, efl_mm=np.array(efl), epd_mm=np.array(epd),
        framing_equiv_focal_mm=np.array(24.0), kernel_box=np.array(n_box, dtype=np.int64),
        rays_grid=np.array(a.rays_grid, dtype=np.int64), rng_seed=np.array(seed, dtype=np.int64),
        sensor_bfd_mm=np.array(bfd),
    )
    tdict = rk.assemble_tensor(kernels, meta)
    rk.save_tensor(out, tdict)
    kernels32 = tdict["kernels"]
    ksha = hashlib.sha256(kernels32.tobytes()).hexdigest()
    fsha = sha256_file(out)
    timings["splat_s"] = time.time() - t0

    # ---- gates ---------------------------------------------------------------
    gates = {}
    # G-S: exact on-axis spot diameters vs frozen-sensor thin lens (EP datum)
    o3, tab3, _, epl3, _ = make_traced_optic(a.lens, sd, bfd)
    ni3 = len(tab3)
    s0 = 700.0
    gs_rows = []
    gpx, gpy = lc.square_pupil(801)
    oc, _ = lc.scaled_optic(tab3)           # nominal-EPD optic for the thick-lens reference
    oc.set_thickness(1e10, 0)
    epl_c = float(oc.paraxial.EPL())
    for d in (1, 3, 5, 10):
        lc.set_object(o3, float(dists[d]) * 1000.0, epl3)
        o3.set_thickness(bfd, ni3)
        x, y, ok = lc.trace_xy(o3, 0.0, gpx, gpy)
        rr = np.hypot(x[ok] - x[ok].mean(), y[ok] - y[ok].mean())
        dia = 2e3 * np.percentile(rr, 99.9)
        thin = 1e3 * thin_lens_diameter(epd, efl, s0, float(dists[d]) * 1000.0)
        marg, marg_edge = marginal_disc_um(o3, ni3)
        # informational: the lens's own paraxial prediction (image shift / working F#)
        o3.updater.image_solve()
        b_s = float(be.to_numpy(o3.surfaces[ni3].thickness))
        par = 1e3 * abs(b_s - bfd) / (2.0 * (1.0 + float(abs(o3.paraxial.magnification()))))
        o3.set_thickness(bfd, ni3)
        # thick-lens reference: defocus at the frozen sensor x exit-pupil diameter / (XP -> sensor distance)
        lc.set_object(oc, float(dists[d]) * 1000.0, epl_c)
        oc.updater.image_solve()
        b_c = float(be.to_numpy(oc.surfaces[ni3].thickness))
        xpd = float(oc.paraxial.XPD())
        # ref3 (report-only, paraxial, excludes spherical aberration): cone apex = paraxial image
        xp = gates_mod.ref3_diameter_um(b_c - bfd, xpd, float(oc.paraxial.XPL()))
        gs_rows.append((d, float(dists[d]), dia, thin, dia / thin - 1, par, 2e3 * rr.max(), xp, marg, marg_edge))
    # Binding G-S (Team Lead ruling): traced 99.9th-pct disc within 3% of the real stop-edge
    # marginal-ray disc at d3/d5/d10. ref1/ref2/ref3 and d1 are report-only.
    gs_ref = lens.get("gs_reference", "scan")
    gs_col = {"scan": 8, "ref1": 3, "ref3": 7}.get(gs_ref)
    if gs_col is None:
        raise SystemExit(f"lens {a.lens}: unknown gs_reference {gs_ref!r} (scan|ref1|ref3)")
    gates["G-S"] = all(gates_mod.gs_ratio_ok(r_[2], r_[gs_col]) for r_ in gs_rows if r_[0] in (3, 5, 10))

    # G-E (box sized to the PSF): in-box, outer 2 px ring, r99 + |offset|
    nonzero = [(r, d) for r in range(R) for d in range(1, D)]
    inbox_min = float(1 - outside[:, 1:].max())
    ring_max = max(ring_energy(kernels32[r, d].astype(np.float64)) for r, d in nonzero)
    ge_reach = max(r99[r, d] for r, d in nonzero)
    gates["G-E"] = inbox_min >= 0.995 and ring_max < 1e-3

    # G-T
    kk = kernels32.astype(np.float64)
    l1err = float(np.abs(kk.sum(axis=(2, 3)) - 1).max())
    delta = rk.make_delta(n_box)
    d0_exact = bool(np.all(kernels32[:, 0] == delta.astype(np.float32)))
    gt = (l1err <= 1e-6 and not np.isnan(kernels32).any() and kernels32.min() >= 0 and d0_exact)
    load_msg = "not run"
    try:
        code = ("import sys;sys.path.insert(0,%r);from summicron.render.convolve import load_tensor;"
                "t=load_tensor(%r);print(t['kernels'].shape)" % (str(ROOT), str(out)))
        gt_cmd = shlex.split(a.gt_python) if a.gt_python else [sys.executable]
        pr = subprocess.run([*gt_cmd, "-c", code], capture_output=True, text=True, timeout=300)
        load_ok = pr.returncode == 0 and "(" in pr.stdout
        load_msg = pr.stdout.strip().splitlines()[-1] if load_ok else (pr.stderr or pr.stdout)[-300:]
        gt = gt and load_ok
    except Exception as e:  # noqa: BLE001
        load_msg = f"load_tensor subprocess failed: {e}"
        gt = False
    gates["G-T"] = bool(gt)

    # G-N: split-half relative L2 at 2048
    half2048 = int(np.ceil(half_need * 2048 / CORPUS_REF)) + 3
    n2048 = max(31, 2 * half2048 + 1)
    gn = np.zeros((R, D))
    thumbs = {}
    for r in range(R):
        for d in range(1, D):
            dout, dtan = load_node(cache, r, d)
            kA = rk.rays_to_kernel(dout[0::2], dtan[0::2], pitch2048_mm, size=n2048)
            kB = rk.rays_to_kernel(dout[1::2], dtan[1::2], pitch2048_mm, size=n2048)
            kf = 0.5 * (kA + kB)
            gn[r, d] = np.linalg.norm(kA - kB) / np.linalg.norm(kf)
            thumbs[(r, d)] = kf
    gates["G-N"] = bool(gn[:, 1:].max() <= 0.03)

    # G-D
    gates["G-D"] = bool(gd_ok)

    # A4: in-focus r99 vs 249 um
    a4 = [(r, float(rcent[r]), r99[r, 0] * pitch_mm * 1e3) for r in range(R) if rcent[r] <= 0.7]

    # shape metrics (5712 kernels)
    aspects = np.zeros((R, D))
    tails = np.zeros((R, D))
    for r in range(R):
        for d in range(1, D):
            aspects[r, d], tails[r, d] = rk.shape_metrics(kernels32[r, d])

    # thumbnails (2048 scale)
    thumb_path = None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ds = [1, 3, 5, 7, 10]
        rsel = [0, 3, 6, 9, 11]
        fig, ax = plt.subplots(len(rsel), len(ds), figsize=(2.2 * len(ds), 2.2 * len(rsel)))
        for i, r in enumerate(rsel):
            for j, d in enumerate(ds):
                k = thumbs[(r, d)]
                ax[i, j].imshow(k ** 0.5, cmap="magma")
                ax[i, j].set_xticks([])
                ax[i, j].set_yticks([])
                if i == 0:
                    ax[i, j].set_title(f"d{d} {dists[d]:.2f} m", fontsize=7)
                if j == 0:
                    ax[i, j].set_ylabel(f"r={rcent[r]:.2f}", fontsize=7)
        fig.suptitle(f"{name}: kernels at 2048 scale (sqrt; row 0 = outward/up)", fontsize=8)
        thumb_path = out.parent / f"thumbs_{name}.png"
        fig.savefig(thumb_path, dpi=110)
        plt.close(fig)
    except Exception as e:  # noqa: BLE001
        thumb_path = f"thumbnails failed: {e}"

    # ---- report --------------------------------------------------------------
    print("\n" + "=" * 78)
    print(f"GATE TABLE  {name}")
    print("=" * 78)
    print(f"box size chosen: {n_box} x {n_box} (one size for all nodes)")
    print("G-S on-axis full diameter (2*r99.9, frozen sensor) vs thin lens (EP datum):")
    print("    d    dist   traced     ref1 thin-lens(EP) ratio | ref2 paraxial/F# ratio | "
          "ref3 paraxial XP (report-only, no SA) ratio | real-ray meridional scan ratio")
    for d, s_m, dia, thin, rel, par, dmax, xp, marg, medge in gs_rows:
        print(f"    d{d:<2d} {s_m:6.2f} m {dia:8.1f}  {thin:8.1f} {dia / thin:6.3f} ({100 * rel:+5.1f}%) | "
              f"{par:8.1f} {dia / par:6.3f} | {xp:8.1f} {dia / xp:6.3f} | {marg:8.1f} {dia / marg:6.3f}   [2*rmax {dmax:.1f}; stop-edge marginal ray {medge:.1f}]")
    print(f"  G-S {'PASS' if gates['G-S'] else 'FAIL'} vs real-ray meridional-scan disc (incl. stop-edge marginal ray) at d3/d5/d10 "
          f"(|dev| <= {100 * gates_mod.G_S_TOL:.0f}%; d1, ref1, ref2, ref3 report-only)")
    print(f"G-E in-box min {100 * inbox_min:.3f}% (>=99.5%), outer-2px ring max {ring_max:.2e} (<1e-3), "
          f"max r99 {ge_reach:.1f} px (half box {n_box // 2}), max r99.9 {p999.max():.1f} px, "
          f"99.5%-half-extent {half_need:.1f} px  -> {'PASS' if gates['G-E'] else 'FAIL'}")
    print(f"G-T load_tensor: {load_msg}; L1 max err {l1err:.2e}; NaN {bool(np.isnan(kernels32).any())}; "
          f"min {kernels32.min():.2e}; d0 exact {d0_exact} -> {'PASS' if gates['G-T'] else 'FAIL'}")
    print(f"G-N split-half rel L2 @2048 (box {n2048}): max {gn[:, 1:].max():.4f} at "
          f"r{np.unravel_index(gn[:, 1:].argmax(), gn[:, 1:].shape)[0]}_d"
          f"{np.unravel_index(gn[:, 1:].argmax(), gn[:, 1:].shape)[1] + 1} -> "
          f"{'PASS' if gates['G-N'] else 'FAIL'} (<=3%)")
    print(f"G-D identical ray hashes on {a.verify_nodes} re-traced nodes: {gd_msgs} -> "
          f"{'PASS' if gates['G-D'] else 'FAIL'}; kernels sha256 {ksha[:16]}")
    d1_ax = gs_rows[0][2]
    print(f"A4 in-focus r99 (d0) at fields <= 0.7 vs on-axis d1 traced disc {d1_ax:.0f} um "
          f"(spec reference 249 um): thin-lens d1 = {gs_rows[0][3]:.0f} um")
    for r, rc, v in a4:
        print(f"    r{r:02d} ({rc:.3f}) r99 {v:7.1f} um = {v / (pitch2048_mm * 1e3):6.1f} px@2048 "
              f"= {v / (pitch_mm * 1e3):6.1f} px@5712")
    print("transmission T (rows r, cols d1,d5,d10): " +
          "; ".join(f"r{r}:" + "/".join(f"{Tm[r, d]:.2f}" for d in (1, 5, 10)) for r in range(R)))
    maxback = max(stats[(r, d)]["unphysical_frac"] for r in range(R) for d in range(D))
    refs = sorted({stats[(r, d)]["ref"] for r in range(R) for d in range(D)})
    vig = [(r, d) for r in range(R) for d in range(D) if stats[(r, d)]["ref"] != "chief"]
    print(f"centring reference: {refs}; vignetted-chief nodes: {vig}")
    print(f"diagnostic: max fraction of surviving rays with a backward z-step between surfaces "
          f"(sd-overlap rays): {maxback:.4f}")
    print("shape_metrics (core_aspect / tail[+=outward]) at 5712; rows r, cols d=1,3,5,7,10")
    for r in range(R):
        print(f"  r{r:02d} {rcent[r]:.3f}  " + "  ".join(f"{aspects[r, d]:5.2f}/{tails[r, d]:+5.2f}"
                                                         for d in (1, 3, 5, 7, 10)))
    print(f"tail>0 (outward) in {(tails[:, 1:] > 0).sum()}/{tails[:, 1:].size} nodes; "
          f"thumbnails: {thumb_path}")
    print(f"npz {out}  sha256 {fsha}\nkernels sha256 {ksha}")
    timings["total_s"] = time.time() - t_all
    print("timings:", {k: round(v, 1) for k, v in timings.items()})

    manifest_path = out.parent / "manifest.json"
    man = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"tensors": {}}
    man["tensors"][name] = dict(
        npz=str(out), sha256=fsha, kernels_sha256=ksha, git_sha=git_sha(),
        optiland=importlib.import_module("optiland").__version__ if hasattr(
            importlib.import_module("optiland"), "__version__") else "0.6.3",
        grid=a.rays_grid, seed=seed, fu=a.fu, variant=variant, lens=str(a.lens),
        sd_mm=sd, sd_mm_fu_rule=sd_fu, sd_reductions=sd_red, sd_capped=sdinfo["capped"], sd_clamped_on_target=[list(c) for c in clamped],
        efl_mm=efl, epd_mm=epd, bfd_mm=bfd, box=n_box, pitch_um_5712=pitch_mm * 1e3,
        gates={k: bool(v) for k, v in gates.items()}, timings=timings,
        gs_rows=[list(map(float, r_)) for r_ in gs_rows], epd_collimated_mm=epd_coll, t_rescale=t_fac,
        gap_rows=gap_rows, g_d=gd_msgs, unphysical_frac_max=maxback,
        node_stats={f"{k[0]},{k[1]}": v for k, v in stats.items()},
    )
    manifest_path.write_text(json.dumps(man, indent=1))
    print("manifest:", manifest_path)
    sys.exit(0 if all(gates.values()) else 3)


if __name__ == "__main__":
    main()
