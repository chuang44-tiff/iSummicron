"""Q-a paired focus-fidelity probe (local QA tool, not CI).

Subcommands
-----------
``dump --tag T [--downres N] [--scenes ...]``
    Render the canonical-bench scenes via ``render_image(..., _eval_sink=...)``
    and write ``<root>/T/<scene>_<long_edge>.npz`` (out, ref, alpha, dmap,
    jpg_md5). Variant generation (flag/monkeypatch) is the caller's job.

``degrade --base T --tag T2 [--sigma 1.0]``
    Build a synthetic band-only degradation of a dump: gaussian blur
    (per channel) applied only where ``0.02 < alpha < 0.98``. alpha == 1 and
    out-of-band pixels stay bit-identical (validation pair (a)).

``compare BASE VAR --lever {band,nonband} [--expect {better,worse}] [--footprint {none,slab0}]``
    10 placements x {w128, w256}; per-scene table; footprint check; exit
    code != 0 on any gate FAIL. Gated scenes: IMG_6651 and Demo_input
    (IMG_6915 is report-only). The bloom mask is built from BASE output so
    base and variant share one mask. CH (inter-channel) is report-only.
    The BG gate is always FULL FRAME (|d bgGER| <= 0.01; per-placement bg is
    report-only; no measurable bg -> FAIL). ``--footprint slab0`` excludes
    pixels with ``dmap < defocus_ratios[2]`` (the lever's own footprint)
    from it, and reports the signed dGER inside
    footprint & bg separately ("footprint d, for G11"; not gated).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from summicron.render import _quality as Q  # noqa: E402

D_KNEE = 0.10  # convolve.D_KNEE_LAYERED
SCENES = {
    "IMG_6915": ("iphoneImage/IMG_6915.HEIC", (0.40, 0.45)),
    "IMG_6651": ("iphoneImage/IMG_6651.HEIC", (0.5, 0.75)),
    "Demo_input": ("iphoneImage/Demo_input.HEIC", (0.40, 0.39)),
}
GATED = ("IMG_6651", "Demo_input")
OFFSETS_2048 = [(0, 0), (24, 0), (-24, 0), (0, 24), (0, -24), (40, 40),
                (-40, -40), (64, 0), (-64, 0), (0, 64)]
WIDTHS = (128, 256)
DEFAULT_ROOT = ROOT / "runs" / "q_probe"


def _scene_of(name: str) -> str:
    return name.rsplit("_", 1)[0]


def cmd_dump(a) -> int:
    import hashlib

    from summicron.render.api import RenderOptions, render_image

    outdir = Path(a.root) / a.tag
    outdir.mkdir(parents=True, exist_ok=True)
    for scene in a.scenes or list(SCENES):
        rel, fxy = SCENES[scene]
        over = dict(save=True, return_linear=True,
                    out_path=outdir / f"{scene}.jpg")
        opts = RenderOptions.canonical_bench(ROOT / rel, fxy, **over)
        if a.downres is not None:
            from dataclasses import replace
            opts = replace(opts, downres=a.downres)
        sink: dict = {}
        res = render_image(opts, _eval_sink=sink, on_progress=lambda m: None)
        out = np.asarray(res.output_linear, dtype=np.float32)
        long_edge = max(out.shape[:2])
        md5 = hashlib.md5((outdir / f"{scene}.jpg").read_bytes()).hexdigest()
        np.savez(outdir / f"{scene}_{long_edge}.npz", out=out,
                 ref=np.asarray(sink["ref_lin"], dtype=np.float32),
                 alpha=np.asarray(sink["alpha"], dtype=np.float32),
                 dmap=np.asarray(sink["dmap"], dtype=np.float32),
                 defocus_ratios=np.asarray(sink["defocus_ratios"], dtype=np.float64),
                 jpg_md5=md5)
        print(f"dumped {scene}_{long_edge}.npz  md5={md5}  "
              f"t={res.elapsed_s:.1f}s")
    return 0


def cmd_degrade(a) -> int:
    src = Path(a.root) / a.base
    dst = Path(a.root) / a.tag
    dst.mkdir(parents=True, exist_ok=True)
    for f in sorted(src.glob("*.npz")):
        d = dict(np.load(f))
        out, alpha = d["out"], d["alpha"]
        band = (alpha > Q.ALPHA_LO) & (alpha < Q.ALPHA_HI)
        blur = np.stack([ndimage.gaussian_filter(out[..., c], a.sigma)
                         for c in range(3)], axis=2)
        d["out"] = np.where(band[..., None], blur, out).astype(np.float32)
        np.savez(dst / f.name, **d)
        print(f"degraded {f.name}: band px = {int(band.sum())}")
    return 0


def _fmt(x, p=4):
    return "  n/a " if x is None else f"{x:+.{p}f}"


def _knee_ger(base, var, ref, alpha, dmap, win):
    crop, inner = Q._crop(win, base.shape, Q._PAD)
    k = Q.knee_mask(alpha[crop], dmap[crop], D_KNEE) & inner
    k &= ~Q._dilate((Q.luma(ref[crop]) > 1) | (Q.luma(base[crop]) > 1), 3)
    mb = Q.q1_metrics(base[crop], ref[crop], alpha[crop], k)
    mv = Q.q1_metrics(var[crop], ref[crop], alpha[crop], k)
    if mb["ger_band"] is None or mv["ger_band"] is None:
        return None
    return mv["ger_band"] - mb["ger_band"]


def _slab0_threshold(b) -> float:
    """``defocus_ratios[2]`` from the dump (fallback: shipped tensor)."""
    if "defocus_ratios" in b.files:
        return float(b["defocus_ratios"][2])
    from summicron.render._artifacts import default_tensor_path

    with np.load(default_tensor_path(), allow_pickle=False) as t:
        return float(t["defocus_ratios"][2])


def _fullframe_bg(base, var, ref, alpha, dmap, footprint, b, fail):
    """ALWAYS-full-frame BG gate (bg minus footprint) + footprint report."""
    fp = None
    if footprint == "slab0":
        thr = _slab0_threshold(b)
        fp = dmap < thr
    r = Q.bg_fullframe(base, var, ref, alpha, fp)
    label = "BG(full-frame" + (", slab0 footprint excluded)" if fp is not None else ")")
    if not r["valid"]:
        print(f"  {label}: invalid: no background pixels (n={r['n_px']})  FAIL")
        fail("BG invalid: no background pixels")
    else:
        print(f"  {label}: n={r['n_px']}  dbgGER {r['d_bg']:+.4f}"
              f" (|.|<={Q.GATE_BG}) {'PASS' if r['gate'] else 'FAIL'}")
        if not r["gate"]:
            fail(f"full-frame BG |dbgGER|={abs(r['d_bg']):.4f}")
    if fp is not None:
        yr, yb, yv = Q.luma(ref), Q.luma(base), Q.luma(var)
        inside = Q.bg_mask(alpha, yr, yb) & fp
        n_in = int(inside.sum())
        gb = Q._ger(yb, yr, inside) if n_in >= Q.MIN_PX else None
        gv = Q._ger(yv, yr, inside) if n_in >= Q.MIN_PX else None
        msg = "n/a" if gb is None or gv is None else f"{gv - gb:+.4f}"
        print(f"  footprint px={int(fp.sum())} footprint&bg px={n_in}"
              f"  footprint dGER (signed, for G11; report-only): {msg}")


def _compare_scene(name, bfile, vfile, lever, expect, fails, footprint="none"):
    scene = _scene_of(name)
    gated = scene in GATED
    b = np.load(bfile)
    v = np.load(vfile)
    ref, alpha, dmap = b["ref"], b["alpha"], b["dmap"]
    base, var = b["out"], v["out"]
    # VAR must share BASE's ref/alpha/dmap: the gates use BASE's, so a lever
    # that legitimately changes them cannot be judged here (fail loudly).
    for key, arr in (("ref", ref), ("alpha", alpha), ("dmap", dmap)):
        if key not in v.files or v[key].shape != arr.shape \
                or not np.array_equal(v[key], arr):
            msg = (f"VAR '{key}' differs from BASE '{key}' (or missing); this "
                   f"lever changes {key} and cannot be gated with BASE's")
            print(f"\n=== {name} ===\n  FAIL: {msg}")
            fails.append(f"{name}: {msg}")
            return False
    h, w = base.shape[:2]
    s = max(h, w) / 2048.0
    fxy = SCENES[scene][1]
    tag = "GATED" if gated else "report-only"
    print(f"\n=== {name}  ({tag}; lever={lever}, expect={expect}) ===")

    def fail(msg):
        print(f"  FAIL: {msg}")
        if gated:
            fails.append(f"{name}: {msg}")
        else:
            print("        (report-only scene; not counted)")

    for width in WIDTHS:
        rows, ker = [], []
        for dx, dy in OFFSETS_2048:
            win = Q.focus_window(base.shape, fxy, width,
                                 (int(round(dx * s)), int(round(dy * s))))
            mb = Q.window_metrics(base, ref, alpha, win)
            mv = Q.window_metrics(var, ref, alpha, win, mask_out=base)
            p = Q.paired(mb, mv, lever)
            if p.get("valid"):
                p["cov"] = mb["coverage"]
                rows.append(p)
                ker.append(_knee_ger(base, var, ref, alpha, dmap, win))
        if not rows:
            print(f"  w{width}: no valid placements (band too small)")
            fail(f"w{width}: no valid placements")
            continue
        dg = np.array([r["d_ger"] for r in rows])
        ds = np.array([r["d_ssim"] for r in rows])
        ker = [k for k in ker if k is not None]
        bg = [abs(r["d_bg"]) for r in rows if r["d_bg"] is not None]
        print(f"  w{width}: n={len(rows)}/10 cov={np.mean([r['cov'] for r in rows]):.3f}"
              f"  dGER {dg.mean():+.4f}±{dg.std():.4f} #neg={int((dg < 0).sum())}"
              f"  dSSIM {ds.mean():+.4f}±{ds.std():.4f} #neg={int((ds < 0).sum())}")
        print(f"        dOvfrac max {max(r['d_ovfrac'] for r in rows):+.4f}"
              f"  GER ceil(max var) {max(r['ger_var'] for r in rows):.3f}"
              f"  |dbgGER| max {(max(bg) if bg else float('nan')):.4f}"
              f"  inter-ch max {max(r['d_ch'] for r in rows):.4f}"
              f"  knee dGER mean {(np.mean(ker) if ker else float('nan')):+.4f}")
        dch_ = [r["d_chroma"] for r in rows if r.get("d_chroma") is not None]
        print(f"        d_chroma (report-only) mean "
              f"{(np.mean(dch_) if dch_ else float('nan')):+.5f}")
        if lever == "band":
            ok = (dg.mean() > 0 and ds.mean() > 0) if expect == "better" \
                else (dg.mean() < 0 and ds.mean() < 0)
            print(f"        sign check ({expect}): {'PASS' if ok else 'FAIL'}")
            if not ok:
                fail(f"w{width} mean dGER/dSSIM not '{expect}'")
        n_ch = sum(1 for r in rows if r["ch_flag"])
        if n_ch:
            print(f"        CH (report-only): {n_ch} placement(s) above "
                  f"{Q.GATE_CH} (not a gate)")
        # Per-placement BG is a report-only diagnostic (|dbgGER| max above);
        # the BG gate is the full-frame one below.
        for g in ("OV", "CEIL", "Q1-N"):
            if any(g in r["gates"] and not r["gates"][g] for r in rows):
                # Degradations legitimately move OV/CEIL/CH; enforce only BG
                # (and Q1-N) under --expect worse.
                enforced = expect == "better" or g == "Q1-N"
                n = sum(1 for r in rows if g in r["gates"] and not r["gates"][g])
                print(f"        gate {g}: {n} placement(s) FAIL"
                      f"{'' if enforced else ' (not enforced for --expect worse)'}")
                if enforced:
                    fail(f"w{width} gate {g}")

    _fullframe_bg(base, var, ref, alpha, dmap, footprint, b, fail)

    # Footprint check (full frame).
    diff = np.abs(var.astype(np.float64) - base.astype(np.float64)).max(axis=2)
    a1 = alpha == 1.0
    band = (alpha > Q.ALPHA_LO) & (alpha < Q.ALPHA_HI)
    knee = Q.knee_mask(alpha, dmap, D_KNEE)
    outside_band = ~band
    outside_knee = ~knee
    ch = diff > 0
    n_ch = max(int(ch.sum()), 1)
    m_a1 = float(diff[a1].max()) if a1.any() else 0.0
    ref_err = max(
        float(np.abs(base - ref)[a1].max()) if a1.any() else 0.0,
        float(np.abs(var - ref)[a1].max()) if a1.any() else 0.0,
    )
    print(f"  footprint: alpha==1 px={int(a1.sum())}  max|d| on alpha==1 = {m_a1:.3e}"
          f"  max|out-ref| on alpha==1 (base,var) = {ref_err:.3e}")
    print(f"             max|d| outside band = {float(diff[outside_band].max()):.3e}"
          f"  outside knee = {float(diff[outside_knee].max()):.3e}"
          f"  | changed px: {int(ch.sum())}  band {int((ch & band).sum()) / n_ch:.1%}"
          f"  knee\\band {int((ch & knee & ~band).sum()) / n_ch:.1%}"
          f"  outside-knee {int((ch & outside_knee).sum()) / n_ch:.1%}")
    if m_a1 != 0.0:
        fail("Q1-I: variant changes alpha==1 pixels")
    if ref_err != 0.0:
        fail("Q1-I: out != ref on alpha==1")
    if expect == "worse" and float(diff[outside_band].max()) != 0.0:
        fail("footprint: change outside band under --expect worse")
    return True


def cmd_compare(a) -> int:
    bdir, vdir = Path(a.root) / a.base, Path(a.root) / a.var
    fails: list[str] = []
    files = sorted(bdir.glob("*.npz"))
    if not files:
        print(f"no npz in {bdir}")
        return 2
    compared: set[str] = set()
    for bf in files:
        vf = vdir / bf.name
        if not vf.exists():
            print(f"skip {bf.name}: missing in {vdir}")
            continue
        if _compare_scene(bf.stem, bf, vf, a.lever, a.expect, fails,
                          a.footprint) is not False:
            compared.add(_scene_of(bf.stem))
    # A gated scene that was not compared (missing BASE or VAR) is a FAIL.
    for g in GATED:
        if g not in compared and not any(
                f.startswith(g + "_") for f in fails):
            msg = f"gated scene {g} not compared (missing BASE or VAR npz)"
            print(f"  FAIL: {msg}")
            fails.append(msg)
    if not compared:
        print("no scenes compared")
        fails.append("no scenes compared")
    print("\n" + ("ALL GATES PASS" if not fails else
                  "GATE FAILURES:\n  " + "\n  ".join(fails)))
    return 0 if not fails else 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--root", default=str(DEFAULT_ROOT))
    sp = p.add_subparsers(dest="cmd", required=True)
    d = sp.add_parser("dump")
    d.add_argument("--tag", required=True)
    d.add_argument("--downres", type=int, default=None)
    d.add_argument("--scenes", nargs="*", choices=list(SCENES))
    g = sp.add_parser("degrade")
    g.add_argument("--base", required=True)
    g.add_argument("--tag", required=True)
    g.add_argument("--sigma", type=float, default=1.0)
    c = sp.add_parser("compare")
    c.add_argument("base")
    c.add_argument("var")
    c.add_argument("--lever", choices=("band", "nonband"), required=True)
    c.add_argument("--expect", choices=("better", "worse"), default="better")
    c.add_argument("--footprint", choices=("none", "slab0"), default="none")
    a = p.parse_args(argv)
    return {"dump": cmd_dump, "degrade": cmd_degrade, "compare": cmd_compare}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
