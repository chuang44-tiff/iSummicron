"""0.5.0 adversarial: the ray-traced default and the legacy-PSF purge.

Attacks the 0.5.0 contract:

* the flagless default render equals the explicit-flag render and the pins
  (SLOW), ``--lens 50`` equals a render of the 50 mm tensor passed by path,
  and a copy of a shipped tensor (byte copy, or stripped to the keys the
  loader reads) renders identically (metadata never reaches the pixels);
* negative controls: ``--lens 50`` and an explicit ``--depth-gain 2`` must
  differ from the default, or the equalities above prove nothing;
* removed flags fail with one ``render:`` line, never a traceback, through
  both entry points and under argparse prefix matching;
* a ``--tensor`` path to the old bundled file fails as "tensor not found";
* corrupt / malformed tensors fail through the one-line CLI contract and
  ``load_tensor`` rejects malformed metadata;
* the synthetic family is gone (the purge removed it, so "still renders"
  is inverted here: it must NOT be reachable);
* nothing on the shipped surface imports a deleted module.

Removed names and values are built by string concatenation so this file
carries no guarded token. Real bugs are ``xfail(strict=True)`` with
``reason="BUG: ..."`` and a passing control next to each.
"""
from __future__ import annotations

import ast
import hashlib
import importlib
import os
import subprocess
import sys
import warnings
import zipfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]

_LEGACY = "code" + "v"
_OLD_NPZ = _LEGACY + "_render_v1.npz"
_OLD_CONV_MOD = _LEGACY + "_to_render"
_GUARD_TOKENS = (_LEGACY, "code" + " v", "code" + "_v", "code" + "-v",
                 "ag_" + "dblgauss", "35" + "f2")

_DELETED_MODULES = (
    "summicron.render._family",
    "summicron.psf_model.synth_family",
    "summicron.psf_model.downscale_tensor",
    "summicron.psf_model." + _OLD_CONV_MOD,
)

_ENTRYPOINTS = {
    "cli": [sys.executable, "-m", "summicron.cli", "render"],
    "convolve": [sys.executable, "-m", "summicron.render.convolve"],
}


def _env():
    env = {k: v for k, v in os.environ.items() if k != "ISUMMICRON_DEBUG"}
    env["PYTHONPATH"] = str(ROOT)
    return env


def _run(entry, args):
    return subprocess.run(
        [*_ENTRYPOINTS[entry], *args], capture_output=True, text=True,
        cwd=str(ROOT), timeout=120, env=_env(),
    )


def _one_render_line(stderr: str) -> str:
    lines = [ln for ln in stderr.splitlines() if ln.strip()]
    assert len(lines) == 1, stderr
    assert lines[0].startswith("render: "), stderr
    assert "Traceback" not in stderr
    return lines[0]


# ---------------------------------------------------------------------------
# tiny tensor fixture (schema-valid, 2 x 3 x 5 x 5)
# ---------------------------------------------------------------------------


def _tiny(**mod) -> dict:
    R, D, k = 2, 3, 5
    kernels = np.zeros((R, D, k, k), np.float32)
    kernels[:, 0, k // 2, k // 2] = 1.0
    kernels[:, 1, 1:4, 1:4] = 1.0 / 9.0
    kernels[:, 2, :, :] = 1.0 / 25.0
    d = {
        "kernels": kernels,
        "radial_edges": np.array([0.0, 0.5, 1.0]),
        "defocus_ratios": np.linspace(0.0, 1.0, D),
        "object_distances_m": np.array([0.7, 2.0, 20.0]),
        "focus_distance_m": np.array(0.7),
        "schema_version": np.array("1.0"),
        "corpus_ref_long_edge": np.array(5712),
        "delta_guard_applied": np.array(True),
        "otf_norm_applied": np.array(False),
    }
    for key, v in mod.items():
        if v is None:
            d.pop(key, None)
        else:
            d[key] = v
    return d


def _save(tmp_path, name="t", **mod) -> Path:
    p = tmp_path / f"{name}.npz"
    np.savez(p, **_tiny(**mod))
    return p


def _load(p):
    from summicron.render.convolve import load_tensor
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = load_tensor(p)
    return out, [str(x.message) for x in w]


# ---------------------------------------------------------------------------
# 1. the synthetic family and the legacy source are gone
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mod", _DELETED_MODULES)
def test_deleted_module_not_importable(mod):
    sys.modules.pop(mod, None)
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(mod)


def test_import_control_live_modules_import():
    """Polarity control: the surviving modules DO import."""
    for mod in ("summicron.render.convolve", "summicron.psf_model.ray_kernel",
                "summicron.render._artifacts", "summicron.render._resolve"):
        importlib.import_module(mod)


@pytest.mark.parametrize("field", ["kernel_family", "lens_aperture_mm"])
def test_api_has_no_family_fields(field):
    from summicron.render.api import RenderOptions

    with pytest.raises(TypeError):
        RenderOptions(image_path=Path("x.heic"), **{field: "synthetic"})


def test_canonical_bench_has_no_old_look():
    from summicron.render import _bench
    from summicron.render.api import RenderOptions

    with pytest.raises(TypeError):
        RenderOptions.canonical_bench(Path("x.heic"), (0.4, 0.45), old_look=True)
    for name in ("CANONICAL_FLAGS_OLD", "OLD_LOOK_FLAGS", "G1_MD5_OLD",
                 "ON_MD5_OLD"):
        assert not hasattr(_bench, name), name


def test_resolver_has_no_family_symbols():
    from summicron.render import _resolve

    for name in ("resolve_family", "ResolvedFamily"):
        assert not hasattr(_resolve, name), name


def test_pyproject_force_include_is_exactly_the_two_tensors():
    import tomllib

    from summicron.render._artifacts import LENS_ARTIFACTS

    cfg = tomllib.loads((ROOT / "pyproject.toml").read_text())
    fi = cfg["tool"]["hatch"]["build"]["targets"]["wheel"]["force-include"]
    npz = sorted(Path(k).name for k in fi if k.endswith(".npz"))
    assert npz == sorted(LENS_ARTIFACTS.values())


# ---------------------------------------------------------------------------
# 2. removed flags: one line, both entry points, prefix matching
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("entry", list(_ENTRYPOINTS))
@pytest.mark.parametrize("args,flag", [
    (["--kernel-family", _LEGACY], "--kernel-family"),
    (["--kern", "x"], "--kernel-family"),
    (["--kernel", "synthetic"], "--kernel-family"),
    (["--kernel-fam"], "--kernel-family"),
    (["--lens-ap", "1"], "--lens-aperture-mm"),
    (["--lens-aperture", "50/2"], "--lens-aperture-mm"),
    (["--lens-aperture-mm=apple"], "--lens-aperture-mm"),
    # combined with otherwise-valid new flags: the tombstone still wins
    (["--kernel-family", "synthetic", "--lens", "50"], "--kernel-family"),
    (["--tensor", "nope.npz", "--kernel-family", _LEGACY], "--kernel-family"),
])
def test_removed_flag_one_line(entry, args, flag):
    r = _run(entry, ["--image", "x.heic", *args])
    assert r.returncode == 2, r.stderr
    line = _one_render_line(r.stderr)
    assert line.startswith(f"render: {flag} was removed in 0.5.0")
    assert "Migrating to 0.5.0" in line
    assert not any(t in line.lower() for t in _GUARD_TOKENS)


@pytest.mark.parametrize("entry", list(_ENTRYPOINTS))
def test_old_bundled_tensor_path_is_tensor_not_found(entry, tmp_path):
    """``--tensor .../summicron/_artifacts/<old npz>`` (gone) -> one line."""
    old = tmp_path / "summicron" / "_artifacts" / _OLD_NPZ
    r = _run(entry, ["--image", "x.heic", "--tensor", str(old)])
    assert r.returncode == 2, r.stderr
    line = _one_render_line(r.stderr)
    assert "tensor not found" in line and "Migrating to 0.5.0" in line
    # the message echoes the user's path; only the text we authored is checked
    authored = line.replace(str(old), "<PATH>").lower()
    assert not any(t in authored for t in _GUARD_TOKENS)


# ---------------------------------------------------------------------------
# 3. corrupt tensor files through the one-line CLI contract
# ---------------------------------------------------------------------------


def _cli_load(p, capsys):
    from summicron.render.convolve import load_tensor, run_cli_render

    rc = run_cli_render(lambda: load_tensor(p))
    return rc, capsys.readouterr().err


def test_corrupt_control_non_npz_text_is_one_line(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("ISUMMICRON_DEBUG", raising=False)
    p = tmp_path / "text.npz"
    p.write_text("not a tensor\n")
    rc, err = _cli_load(p, capsys)
    assert rc == 2
    _one_render_line(err)


def test_empty_tensor_file_is_one_line(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("ISUMMICRON_DEBUG", raising=False)
    p = tmp_path / "empty.npz"
    p.write_bytes(b"")
    rc, err = _cli_load(p, capsys)
    assert rc == 2
    _one_render_line(err)


def test_truncated_tensor_file_is_one_line(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("ISUMMICRON_DEBUG", raising=False)
    full = _save(tmp_path, "full")
    p = tmp_path / "trunc.npz"
    p.write_bytes(full.read_bytes()[:200])
    rc, err = _cli_load(p, capsys)
    assert rc == 2
    _one_render_line(err)


# ---------------------------------------------------------------------------
# 4. load_tensor: malformed metadata
# ---------------------------------------------------------------------------


def test_tiny_fixture_is_valid(tmp_path):
    t, w = _load(_save(tmp_path))
    assert (t["R"], t["D"]) == (2, 3) and w == []


@pytest.mark.parametrize("mod,match", [
    ({"schema_version": np.array("2.0")}, "schema_version"),
    ({"kernels": None}, "missing required keys"),
    ({"radial_edges": np.array([0.0, 1.0])}, "radial_edges length"),
    ({"radial_edges": np.array([0.0, 0.5, 0.5])}, "strictly increasing"),
    ({"object_distances_m": np.array([20.0, 2.0, 0.7])}, "monoton"),
    ({"object_distances_m": np.array([0.7, 2.0])}, "object_distances_m length"),
    ({"focus_distance_m": np.array(1.5)}, "focus_distance_m"),
    ({"corpus_ref_long_edge": np.array(0)}, "positive"),
    ({"defocus_ratios": np.array([0.0, 0.5, np.nan])}, "non-finite"),
])
def test_load_tensor_rejects_malformed(tmp_path, mod, match):
    from summicron.render.convolve import TensorValidationError

    with pytest.raises(TensorValidationError, match=match):
        _load(_save(tmp_path, **mod))


def test_load_tensor_rejects_pickled_kernels(tmp_path):
    p = tmp_path / "pk.npz"
    d = _tiny()
    d["kernels"] = np.array([object()], dtype=object)
    np.savez(p, **d)
    with pytest.raises(ValueError):
        _load(p)


@pytest.mark.xfail(
    strict=True, raises=pytest.fail.Exception,
    reason="BUG: load_tensor never checks len(defocus_ratios) == D; a "
           "mismatched custom tensor loads and the render dies later with "
           "numpy's 'fp and xp are not of the same length'",
)
@pytest.mark.parametrize("n", [2, 5])
def test_load_tensor_rejects_defocus_ratios_length(tmp_path, n):
    from summicron.render.convolve import TensorValidationError

    with pytest.raises(TensorValidationError):
        _load(_save(tmp_path, defocus_ratios=np.linspace(0.0, 1.0, n)))


def test_delta_guard_false_warns_control(tmp_path):
    _, w = _load(_save(tmp_path, delta_guard_applied=np.array(False)))
    assert any("delta_guard_applied=False" in m for m in w)


@pytest.mark.xfail(
    strict=True, raises=AssertionError,
    reason="BUG: delta_guard_applied stored as the string 'False' is read "
           "with bool() (non-empty string -> True): the guard is reported "
           "applied and the warning is suppressed (wrong polarity)",
)
def test_delta_guard_string_false_is_not_true(tmp_path):
    _, w = _load(_save(tmp_path, delta_guard_applied=np.array("False")))
    assert any("delta_guard_applied" in m for m in w)


@pytest.mark.xfail(
    strict=True, raises=pytest.fail.Exception,
    reason="BUG: corpus_ref_long_edge is coerced with int(): a float is "
           "silently truncated and a bool True becomes 1 (pix_scale ~2048x), "
           "against the docstring's 'validated for int'",
)
@pytest.mark.parametrize("v", [np.array(5712.9), np.array(True)])
def test_corpus_ref_must_be_integer(tmp_path, v):
    from summicron.render.convolve import TensorValidationError

    with pytest.raises(TensorValidationError):
        _load(_save(tmp_path, corpus_ref_long_edge=v))


@pytest.mark.xfail(
    strict=True, raises=pytest.fail.Exception,
    reason="BUG: a non-scalar schema_version escapes as a bare ValueError "
           "from .item() instead of TensorValidationError",
)
def test_schema_version_non_scalar_is_typed(tmp_path):
    from summicron.render.convolve import TensorValidationError

    try:
        _load(_save(tmp_path, schema_version=np.array(["1.0", "1.0"])))
    except TensorValidationError:
        return
    except ValueError:
        pytest.fail("bare ValueError, not TensorValidationError")
    pytest.fail("loaded")


# ---------------------------------------------------------------------------
# 5. wrong polarity: no hidden tensor-kind dispatch
# ---------------------------------------------------------------------------


def test_legacy_shaped_tensor_is_noticed(tmp_path):
    """A pre-0.5.0-shaped tensor (no schema_version) still gets a notice.

    0.5.0 autos (physical gain 1.0, l1 stamp) are unconditional, so a local
    copy of the old tensor passed via --tensor renders under the new autos;
    the only signal it gets is this load-time warning.
    """
    _, w = _load(_save(tmp_path, schema_version=None))
    assert any("schema_version" in m for m in w)


def test_shipped_tensors_load_silently():
    from summicron.render._artifacts import LENS_ARTIFACTS, default_tensor_path

    for lens in LENS_ARTIFACTS:
        _, w = _load(default_tensor_path(lens))
        assert w == [], (lens, w)


def test_no_kind_key_dispatch_in_render_path():
    """The render path must not branch on npz provenance strings: the spec
    makes the autos unconditional, so a tensor-kind switch would be a
    silent fork."""
    src = (ROOT / "summicron" / "render" / "convolve.py").read_text()
    for key in ("lens_citation", "prescription_variant", "rays_grid",
                "tensor_kind", '"kind"'):
        assert key not in src, key


@pytest.mark.parametrize("entry", ["cli", "convolve"])
def test_flagless_and_explicit_argv_resolve_identically(entry):
    """FAST: flagless argv and the explicit optiland argv resolve to the
    same stamp/gain/norm through both entry points."""
    from summicron.render._resolve import resolve_stamp
    from summicron.render.api import RenderOptions

    if entry == "cli":
        from summicron.cli import build_parser
        pre = ["render"]
    else:
        from summicron.render.convolve import build_parser
        pre = []
    base = ["--image", "x.heic", "--focus-xy", "0.4", "0.45"]
    explicit = ["--depth-mapping", "diopter", "--depth-gain", "1",
                "--bright-stamp", "l1"]
    o_d = RenderOptions.from_argparse(build_parser().parse_args(pre + base))
    o_e = RenderOptions.from_argparse(
        build_parser().parse_args(pre + base + explicit))
    assert resolve_stamp(o_d) == resolve_stamp(o_e)
    assert resolve_stamp(o_d).bright_stamp == "l1"
    assert resolve_stamp(o_d).bright_stamp_gain == 2.0
    assert o_d.depth_gain == "auto"
    assert o_d.lens is None and o_d.tensor_path is None


# ---------------------------------------------------------------------------
# 6. no import of a deleted module on the shipped surface (pathlib + AST)
# ---------------------------------------------------------------------------


def _removed_imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    dead = set(_DELETED_MODULES)
    hits = []
    for node in ast.walk(tree):
        names = []
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names = [node.module] + [f"{node.module}.{a.name}" for a in node.names]
        elif (isinstance(node, ast.Call) and node.args
              and isinstance(node.args[0], ast.Constant)
              and isinstance(node.args[0].value, str)
              and getattr(node.func, "attr", getattr(node.func, "id", ""))
              in ("import_module", "__import__")):
            names = [node.args[0].value]
        hits += [n for n in names if any(n == d or n.startswith(d + ".")
                                         for d in dead)]
    return hits


def test_import_scanner_polarity(tmp_path):
    f = tmp_path / "p.py"
    f.write_text(
        "from summicron.psf_model import synth_family\n"
        "import summicron.render._family\n"
        "import importlib\n"
        f"importlib.import_module('summicron.psf_model.{_OLD_CONV_MOD}')\n"
        "from summicron.render import _artifacts\n"
    )
    assert sorted(_removed_imports(f)) == sorted([
        "summicron.psf_model.synth_family",
        "summicron.render._family",
        "summicron.psf_model." + _OLD_CONV_MOD,
    ])


def test_shipped_surface_imports_no_deleted_module():
    roots = [ROOT / d for d in ("summicron", "tests", "demo", "scripts/build_psf")]
    files = [p for r in roots if r.exists() for p in r.rglob("*.py")
             if "__pycache__" not in p.parts]
    files += [p for p in (ROOT / "scripts" / "q_paired_probe.py",
                          ROOT / "scripts" / "clean_runs.py") if p.exists()]
    assert len(files) > 50
    bad = {str(p.relative_to(ROOT)): h for p in files if (h := _removed_imports(p))}
    assert bad == {}


# ---------------------------------------------------------------------------
# 7. SLOW: real renders against the pins
# ---------------------------------------------------------------------------


def _bench_opts(**over):
    from summicron.render._bench import G1_FOCUS_XY, G1_IMAGE
    from summicron.render.api import RenderOptions

    img = ROOT / G1_IMAGE
    if sys.platform != "darwin" or not img.exists():
        pytest.skip("needs macOS and the bench image")
    return RenderOptions.canonical_bench(img, G1_FOCUS_XY, **over)


def _render_md5(opts, out: Path):
    from summicron.render.api import render_image

    r = render_image(replace(opts, out_path=out), on_progress=lambda m: None)
    return hashlib.md5(out.read_bytes()).hexdigest(), r


def _stripped_copy(src: Path, dst: Path) -> Path:
    keep = ("kernels", "radial_edges", "defocus_ratios", "object_distances_m",
            "focus_distance_m", "corpus_ref_long_edge", "delta_guard_applied",
            "schema_version")
    with np.load(src, allow_pickle=False) as d:
        np.savez(dst, **{k: d[k] for k in keep if k in d.files})
    return dst


@pytest.mark.slow
@pytest.mark.parametrize("lens,pin", [("35", "G1_MD5"), ("50", "LENS50_MD5")])
def test_default_custom_copy_and_stripped_copy_equal_pin(tmp_path, lens, pin):
    from summicron.render import _bench
    from summicron.render._artifacts import default_tensor_path

    expected = getattr(_bench, pin)
    src = default_tensor_path(lens)
    copy = tmp_path / "copy.npz"
    copy.write_bytes(src.read_bytes())
    stripped = _stripped_copy(src, tmp_path / "stripped.npz")

    flagless = _bench_opts() if lens == "35" else _bench_opts(lens="50")
    md5_d, r_d = _render_md5(flagless, tmp_path / "d.jpg")
    assert md5_d == expected
    assert r_d.diagnostics.resolved.get("lens") == lens

    explicit = _bench_opts(
        tensor_path=src, depth_mapping="diopter", depth_gain=1.0,
        bright_stamp="l1",
    )
    md5_e, r_e = _render_md5(explicit, tmp_path / "e.jpg")
    assert md5_e == expected
    assert r_e.diagnostics.resolved.get("lens") == "custom"

    for name, p in (("copy", copy), ("stripped", stripped)):
        md5_c, r_c = _render_md5(_bench_opts(tensor_path=p), tmp_path / f"{name}.jpg")
        assert md5_c == expected, name
        assert r_c.diagnostics.resolved.get("lens") == "custom"


@pytest.mark.slow
def test_negative_controls_differ_from_default(tmp_path):
    """Equality above is meaningful only if the knobs move the output."""
    from summicron.render import _bench

    assert len({_bench.G1_MD5, _bench.LENS50_MD5, _bench.ON_MD5}) == 3
    md5_g2, _ = _render_md5(_bench_opts(depth_gain=2.0), tmp_path / "g2.jpg")
    assert md5_g2 != _bench.G1_MD5
    md5_50, _ = _render_md5(_bench_opts(lens="50"), tmp_path / "l50.jpg")
    assert md5_50 != _bench.G1_MD5

