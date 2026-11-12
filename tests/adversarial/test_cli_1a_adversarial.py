"""CLI-1a Adversarial round — try to BREAK the CLI-1a changes.

Targets (spec: doc/plan/phase2/section9_cli-1a/locked-spec-CLI-1a.md):
- render() entry kernel-L1 guard (A3.1b), incl. WRONG-POLARITY probes
  (under-energy, negative sums, NaN, boundary).
- render() keyword-only ``depth_m`` contract (A1/A3).
- ``tests/fixtures/defocus.py::render_with_defocus`` fidelity.
- C3 single argparse source: convolve vs ``isummicron render`` parity,
  ``RenderOptions.from_argparse``, import-footprint of ``--help``.
- ``_layered_composite`` lazy imports from convolve still resolve.

Genuine bugs are ``xfail(strict=True, reason="BUG: ...")``.
"""
from __future__ import annotations

import argparse
import random
import subprocess
import sys

import numpy as np
import pytest

from summicron.render import convolve as cv
from summicron.render.api import RenderOptions
from tests.fixtures.defocus import render_with_defocus

pytestmark = pytest.mark.filterwarnings("ignore")

_RE = np.array([0.0, 0.15, 0.30, 0.45, 0.60, 0.80, 1.01], dtype=np.float64)
_DR = np.linspace(0, 1, 11, dtype=np.float64)
_OD = np.array(
    [0.7, 0.98, 1.37, 1.91, 2.68, 3.74, 5.23, 7.32, 10.23, 14.30, 20.0],
    dtype=np.float64,
)
H = W = 40


def _tensor(ksize: int = 17, D: int = 11) -> dict:
    R = 6
    kern = np.zeros((R, D, ksize, ksize), dtype=np.float64)
    c = ksize // 2
    kern[:, 0, c, c] = 1.0
    ys, xs = np.mgrid[-c:c + 1, -c:c + 1].astype(np.float64)
    for di in range(1, D):
        sigma = 1.0 + 0.6 * di
        g = np.exp(-(xs ** 2 + ys ** 2) / (2 * sigma ** 2))
        kern[:, di] = g / g.sum()
    t = {
        "kernels": kern,
        "radial_edges": _RE.copy(),
        "defocus_ratios": np.linspace(0, 1, D) if D > 1 else np.array([0.0]),
        "focus_distance_m": 0.7,
        "corpus_ref_long_edge": None,
        "otf_norm_applied": False,
        "delta_guard_applied": True,
    }
    if D == 11:
        t["object_distances_m"] = _OD.copy()
    return t


def _img(seed=0):
    rng = np.random.default_rng(seed)
    return (rng.random((H, W, 3)) * 0.6 + 0.1).astype(np.float32)


def _depth(v=5.0):
    return np.full((H, W), v, dtype=np.float32)


# Depth 5.0 m interpolates between D-slices 5 (3.74 m) and 6 (5.23 m).
_UNUSED_D = 9  # 14.30 m slice — not touched at depth 5.0 m


# =========================================================================
# 1. Kernel-L1 guard — wrong polarity + boundary
# =========================================================================
class TestKernelL1Guard:
    @pytest.mark.parametrize("scale", [0.5, 0.94, 1.06, 2.0])
    def test_under_and_over_energy_both_trip(self, scale):
        """Wrong polarity: under-energy (0.5, 0.94) must trip as well as
        over-energy — the guard is |sum-1|, not sum>1."""
        t = _tensor()
        t["kernels"][3, 4] *= scale
        with pytest.raises(RuntimeError, match="energy ratio"):
            cv.render(_img(), t, depth_m=_depth())

    def test_single_bad_slice_in_unused_position_trips(self):
        """One bad slice among 66, never sampled by the depth map, still trips
        (guard is tensor-wide, not depth-gated)."""
        t = _tensor()
        t["kernels"][5, _UNUSED_D] *= 0.5
        with pytest.raises(RuntimeError, match=r"energy ratio 0\.5000"):
            cv.render(_img(), t, depth_m=_depth())

    def test_boundary_exactly_1_05_and_0_95_pass(self):
        """|sum-1| == 0.05 is NOT > 0.05 → passes (float64, exact-ish)."""
        for s in (1.05, 0.9500001):
            t = _tensor()
            t["kernels"][2, 3] *= s
            cv.render(_img(), t, depth_m=_depth())

    def test_boundary_just_outside_trips(self):
        for s in (1.0501, 0.9499):
            t = _tensor()
            t["kernels"][2, 3] *= s
            with pytest.raises(RuntimeError, match="energy ratio"):
                cv.render(_img(), t, depth_m=_depth())

    def test_reported_ratio_is_max_bad_sum(self):
        """Message reports max of BAD sums (2.0), not the 0.5 one."""
        t = _tensor()
        t["kernels"][1, 2] *= 0.5
        t["kernels"][4, 7] *= 2.0
        with pytest.raises(RuntimeError, match=r"energy ratio 2\.0000"):
            cv.render(_img(), t, depth_m=_depth())

    def test_inf_slice_trips(self):
        t = _tensor()
        t["kernels"][1, 4, 0, 0] = np.inf
        with pytest.raises(RuntimeError, match="energy ratio inf"):
            cv.render(_img(), t, depth_m=_depth())

    def test_all_zero_slice_exempt(self):
        t = _tensor()
        t["kernels"][5, _UNUSED_D] = 0.0
        cv.render(_img(), t, depth_m=_depth())

    def test_all_zero_tensor_caught_downstream(self):
        """All-zero tensor passes the entry guard (every slice exempt) but the
        layered total-energy gate catches it — no silent black render."""
        t = _tensor()
        t["kernels"][:] = 0.0
        with pytest.raises(RuntimeError, match="energy ratio"):
            cv.render(_img(), t, depth_m=_depth())

    def test_negative_lobed_kernel_summing_to_one_passes_documented(self):
        """DOCUMENTED LIMITATION (not a bug): the guard is a SUM check, so a
        kernel with a large negative lobe whose sum is 1 passes. Pinned so a
        future positivity gate is a conscious change."""
        t = _tensor()
        k = t["kernels"][2, 5]
        c = k.shape[0] // 2
        k *= 2.0
        k[c, c] -= k.sum() - 1.0
        assert k.min() < -0.5 and abs(k.sum() - 1.0) < 1e-9
        out, lin = cv.render(_img(), t, depth_m=_depth())
        assert np.isfinite(lin).all()

    def test_nan_slice_trips_guard(self):
        t = _tensor()
        t["kernels"][5, _UNUSED_D, 0, 0] = np.nan
        with pytest.raises(RuntimeError, match="energy ratio"):
            cv.render(_img(), t, depth_m=_depth())

    def test_negative_sum_slice_trips_guard(self):
        t = _tensor()
        t["kernels"][5, _UNUSED_D] *= -1.0
        with pytest.raises(RuntimeError, match="energy ratio"):
            cv.render(_img(), t, depth_m=_depth())

    def test_nan_in_used_slice_still_fails_somewhere(self):
        """Even with the guard gap, a NaN in a SAMPLED slice must not yield
        a silent NaN image."""
        t = _tensor()
        t["kernels"][:, 5:7] = np.nan
        with pytest.raises(RuntimeError, match="NaN"):
            cv.render(_img(), t, depth_m=_depth())

    def test_peak_mode_skips_guard_intended(self):
        """psf_norm='peak' renormalizes per-kernel → sums are irrelevant;
        guard intentionally l1-only."""
        t = _tensor()
        t["kernels"][3, 4] *= 3.0
        cv.render(_img(), t, depth_m=_depth(), psf_norm="peak")

    def test_guard_does_not_mutate_tensor(self):
        t = _tensor()
        before = t["kernels"].copy()
        cv.render(_img(), t, depth_m=_depth())
        np.testing.assert_array_equal(t["kernels"], before)


# =========================================================================
# 2. render() API contract
# =========================================================================
class TestRenderSignature:
    def test_positional_depth_rejected(self):
        with pytest.raises(TypeError):
            cv.render(_img(), _tensor(), _depth())

    def test_defocus_map_kwarg_rejected(self):
        with pytest.raises(TypeError):
            cv.render(_img(), _tensor(), defocus_map=np.zeros((H, W)))

    def test_missing_depth_rejected(self):
        with pytest.raises(TypeError):
            cv.render(_img(), _tensor())

    def test_none_depth_documented(self):
        """depth_m=None slips past the TypeError; np.asarray(None).shape==()
        → ValueError on shape. Loud, acceptable."""
        with pytest.raises(ValueError, match="shape"):
            cv.render(_img(), _tensor(), depth_m=None)

    def test_wrong_shape_rejected(self):
        with pytest.raises(ValueError, match="shape"):
            cv.render(_img(), _tensor(), depth_m=_depth()[None])
        with pytest.raises(ValueError, match="shape"):
            cv.render(_img(), _tensor(), depth_m=_depth().T[:-1])

    def test_missing_ods_rejected(self):
        t = _tensor()
        del t["object_distances_m"]
        with pytest.raises(ValueError, match="object_distances_m"):
            cv.render(_img(), t, depth_m=_depth())

    def test_int_depth_equals_float_depth(self):
        a, _ = cv.render(_img(), _tensor(), depth_m=np.full((H, W), 5, np.int64))
        b, _ = cv.render(_img(), _tensor(), depth_m=_depth(5.0))
        np.testing.assert_array_equal(a, b)

    def test_nan_depth_fails_loud_late(self):
        """C4 follow-up: NaN depth is not validated early but still raises
        (late NaN gate) — never a silent NaN image."""
        d = _depth()
        d[10, 10] = np.nan
        with pytest.raises(RuntimeError, match="NaN"):
            cv.render(_img(), _tensor(), depth_m=d)

    def test_inf_and_negative_depth_clamp_documented(self):
        """C4 follow-up: +Inf clamps to the far slice, negative to focus
        (np.interp edge clamp). Negative depth == focus distance render."""
        t = _tensor()
        d_inf = _depth()
        d_inf[5, 5] = np.inf
        _, lin = cv.render(_img(), t, depth_m=d_inf)
        assert np.isfinite(lin).all()
        a, _ = cv.render(_img(), t, depth_m=_depth(-3.0))
        b, _ = cv.render(_img(), t, depth_m=_depth(0.7))
        np.testing.assert_array_equal(a, b)


# =========================================================================
# 3. render_with_defocus fixture fidelity
# =========================================================================
class TestDefocusFixture:
    def test_does_not_mutate_caller_tensor(self):
        t = _tensor()
        del t["object_distances_m"]
        keys = set(t)
        k_id = id(t["kernels"])
        render_with_defocus(_img(), t, np.full((H, W), 0.5, np.float32))
        assert set(t) == keys, "fixture injected object_distances_m into caller dict"
        assert id(t["kernels"]) == k_id

    @pytest.mark.parametrize("v,od", [(0.0, 0.7), (1.0, 20.0)])
    def test_dmap_endpoints_map_to_ods_endpoints(self, v, od):
        t = _tensor()
        a, _ = render_with_defocus(_img(), t, np.full((H, W), v, np.float32))
        b, _ = cv.render(_img(), t, depth_m=_depth(od))
        np.testing.assert_array_equal(a, b)

    def test_dmap_roundtrip_through_render_interp(self):
        """dmap → depth_m (fixture) → dmap (render) is identity on a grid."""
        dm = np.linspace(0, 1, H * W, dtype=np.float32).reshape(H, W)
        depth = np.interp(dm, _DR, _OD)
        back = np.interp(depth, _OD, _DR)
        np.testing.assert_allclose(back, dm, atol=1e-6)

    def test_d1_tensor_renders(self):
        t = _tensor(D=1)
        assert "object_distances_m" not in t
        out, lin = render_with_defocus(_img(), t, None)
        assert out.shape == (H, W, 3) and np.isfinite(lin).all()

    @pytest.mark.parametrize("D", [2, 3, 7, 23])
    def test_injected_ods_strictly_monotonic(self, D, monkeypatch):
        """Injected axis must be strictly increasing (np.interp silently
        garbages on non-monotonic xp) and depth_m must stay within it."""
        seen = {}

        def spy(img, t, *, depth_m, **kw):
            seen["ods"] = np.asarray(t["object_distances_m"])
            seen["depth"] = depth_m
            return None, None

        monkeypatch.setattr(cv, "render", spy)
        t = _tensor(D=D)
        render_with_defocus(_img(), t, None)
        ods = seen["ods"]
        assert ods.shape == (D,) and np.all(np.diff(ods) > 0)
        assert seen["depth"].min() >= ods[0] - 1e-5
        assert seen["depth"].max() <= ods[-1] + 1e-4
        assert "object_distances_m" not in t


# =========================================================================
# 4. C3 — single argparse source / RenderOptions
# =========================================================================
def _cli_render_parser():
    from summicron.cli import build_parser

    p = build_parser()
    sub = next(a for a in p._actions if isinstance(a, argparse._SubParsersAction))
    return sub.choices["render"]


_CHOICES = {
    "--psf-norm": ["l1", "peak"],
    "--depth-source": ["lidar", "monocular"],
    "--hdr-source": ["none", "macos"],
    "--shadow-match-mode": ["off", "on"],
    "--occlusion-compositing": ["off", "on"],
}
_NUMERIC = {
    "--depth-scale": lambda r: f"{r.uniform(0.1, 10):.3f}",
    "--downres": lambda r: str(r.randint(256, 4096)),
    "--focus-target-m": lambda r: f"{r.uniform(0.7, 20):.2f}",
    "--bright-pass-thresh": lambda r: f"{r.uniform(0, 5):.3f}",
    "--bright-pass-region-size": lambda r: str(r.randint(0, 255)),
    "--bright-pass-floor": lambda r: f"{r.random():.3f}",
    "--slab-batch-size": lambda r: str(r.randint(1, 11)),
    "--shadow-band-hi": lambda r: f"{r.random():.3f}",
}


def _random_argv(r: random.Random) -> list[str]:
    argv = ["--image", "x.heic"]
    picked = {}
    for flag, ch in _CHOICES.items():
        if r.random() < 0.5:
            picked[flag] = r.choice(ch)
    # CLI-1b C4: RenderOptions now rejects shadow-match-mode on + hdr none.
    if picked.get("--shadow-match-mode") == "on" and picked.get("--hdr-source", "none") == "none":
        picked["--hdr-source"] = "macos"
    for flag, val in picked.items():
        argv += [flag, val]
    for flag, gen in _NUMERIC.items():
        if r.random() < 0.5:
            argv += [flag, gen(r)]
    if r.random() < 0.5:
        argv += ["--focus-xy", f"{r.random():.3f}", f"{r.random():.3f}"]
    if r.random() < 0.3:
        argv += ["--output", "runs/x.jpg", "--tensor", "t.npz"]
    return argv


class TestArgparseParity:
    @pytest.mark.parametrize("seed", range(25))
    def test_random_argv_identical_options(self, seed):
        argv = _random_argv(random.Random(seed))
        a = RenderOptions.from_argparse(cv.build_parser().parse_args(argv))
        b = RenderOptions.from_argparse(_cli_render_parser().parse_args(argv))
        assert a == b
        assert hash(a) == hash(b)

    def test_occlusion_on_accepted_and_propagated(self):
        argv = ["--image", "x", "--occlusion-compositing", "on"]
        for p in (cv.build_parser(), _cli_render_parser()):
            assert RenderOptions.from_argparse(
                p.parse_args(argv)).occlusion_compositing == "on"

    def test_dof_m_rejected_by_both(self):
        for p in (cv.build_parser(), _cli_render_parser()):
            with pytest.raises(SystemExit):
                p.parse_args(["--image", "x", "--dof-m", "2.0"])
        from summicron.cli import main

        with pytest.raises(SystemExit):
            main(["render", "--image", "x", "--dof-m", "2.0"])

    def test_partial_namespace_keeps_defaults(self):
        opts = RenderOptions.from_argparse(argparse.Namespace(psf_norm="peak"))
        default = RenderOptions(image_path=opts.image_path)
        assert opts.psf_norm == "peak"
        for f in ("occlusion_compositing", "bright_pass_region_size",
                  "hdr_source", "focus_xy", "out_path", "tensor_path"):
            assert getattr(opts, f) == getattr(default, f), f

    def test_focus_xy_tuple_hashable(self):
        ns = cv.build_parser().parse_args(
            ["--image", "x", "--focus-xy", "0.4", "0.45"])
        assert isinstance(ns.focus_xy, list)
        opts = RenderOptions.from_argparse(ns)
        assert opts.focus_xy == (0.4, 0.45)
        assert isinstance(opts.focus_xy, tuple)
        hash(opts)
        {opts: 1}

    def test_from_argparse_unknown_attr_ignored(self):
        ns = argparse.Namespace(image="x", func=print, command="render",
                                bogus=1)
        opts = RenderOptions.from_argparse(ns)
        assert str(opts.image_path) == "x"

    def test_from_argparse_does_not_mutate_namespace(self):
        ns = cv.build_parser().parse_args(
            ["--image", "x", "--focus-xy", "0.1", "0.2"])
        before = vars(ns).copy()
        RenderOptions.from_argparse(ns)
        assert vars(ns) == before


def _footprint(argv: list[str]) -> tuple[bool, bool]:
    code = (
        "import sys\n"
        "from summicron import cli\n"
        "try:\n"
        f"    cli.main({argv!r})\n"
        "except SystemExit:\n"
        "    pass\n"
        "print('FOOT', 'scipy' in sys.modules, "
        "'summicron.render.convolve' in sys.modules)\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                       text=True, timeout=120)
    line = next(ln for ln in r.stdout.splitlines() if ln.startswith("FOOT"))
    _, sp, conv = line.split()
    return sp == "True", conv == "True"


class TestImportFootprint:
    @pytest.mark.parametrize("argv", [["--help"], ["render", "--help"]])
    def test_help_does_not_import_renderer(self, argv):
        sp, conv = _footprint(argv)
        assert not sp, "scipy imported by --help (_render_args not stdlib-only?)"
        assert not conv

    def test_eval_help_does_not_import_convolve(self):
        """eval --help imports scipy via summicron.eval (pre-existing; not a
        CLI-1a surface) but must not drag in the renderer."""
        _, conv = _footprint(["eval", "--help"])
        assert not conv

    def test_render_args_module_stdlib_only(self):
        import ast
        import inspect

        from summicron.render import _render_args

        tree = ast.parse(inspect.getsource(_render_args))
        mods = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                mods |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module:
                mods.add(node.module.split(".")[0])
        assert mods <= {"__future__", "argparse", "pathlib"}, mods

    def test_eval_intercept_before_parse(self, monkeypatch):
        """Flags before any positional still reach eval's main."""
        import summicron.eval.__main__ as em

        seen = {}

        def fake(argv=None):
            seen["argv"] = argv
            return 0

        monkeypatch.setattr(em, "main", fake)
        from summicron.cli import main

        assert main(["eval", "--input", "a", "--rendered", "b"]) == 0
        assert seen["argv"] == ["--input", "a", "--rendered", "b"]


# =========================================================================
# 5. _layered_composite lazy imports
# =========================================================================
def test_layered_composite_lazy_symbols_resolve():
    from summicron.render.convolve import (  # noqa: F401
        _EMPTY_SKIP_THRESH,
        _layered_gather_diffuse,
        _scatter_one_slab_fixed_d_basis,
    )

    assert callable(_layered_gather_diffuse)
    assert callable(_scatter_one_slab_fixed_d_basis)


def test_occlusion_on_renders_through_new_contract():
    t = _tensor()
    d = _depth(0.7)
    d[:, W // 2:] = 10.0
    out, lin = cv.render(_img(), t, depth_m=d, occlusion_compositing="on")
    assert out.shape == (H, W, 3) and np.isfinite(lin).all()
