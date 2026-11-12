"""CLI-1b Adversarial — try to BREAK the RenderOptions / render(stats=) /
_emit / _is_default_tensor / save + out_dir contract.

Wrong-polarity focus (Post-mortem 1): the validator assumes "valid" means
"in range and finite"; every numeric field is fed NaN / +-Inf / negative /
zero, every enum gets case variants. Accepted-but-wrong values are pinned as
``xfail(strict=True, reason="BUG: ...")`` so a future fix flips them loudly.

Spec: doc/plan/phase2/section10_cli-1b/locked-spec-CLI-1b.md
"""
from __future__ import annotations

import argparse
import dataclasses
import shlex
import shutil
import warnings
from pathlib import Path

import numpy as np
import pytest

from summicron.render import convolve as conv
from summicron.render.api import RenderOptions, render_image
from summicron.render._artifacts import default_tensor_path

IMG = Path("dummy.heic")
pytestmark = [
    pytest.mark.filterwarnings("ignore:Center SSIM.*:UserWarning"),
    pytest.mark.filterwarnings("ignore:.*delta_guard_applied=False.*:UserWarning"),
]

NAN = float("nan")
INF = float("inf")


def _ro(**kw):
    return RenderOptions(image_path=IMG, **kw)


# ---------------------------------------------------------------------------
# 1. Wrong polarity — every validated numeric field rejects NaN/Inf/out-of-range
# ---------------------------------------------------------------------------

REJECT = [
    # focus_xy components
    {"focus_xy": (NAN, 0.5)}, {"focus_xy": (0.5, NAN)},
    {"focus_xy": (INF, 0.5)}, {"focus_xy": (0.5, -INF)},
    {"focus_xy": (-1e-12, 0.5)}, {"focus_xy": (0.5, 1.0 + 1e-12)},
    # focus_target_m
    {"focus_target_m": NAN}, {"focus_target_m": INF}, {"focus_target_m": -INF},
    {"focus_target_m": 0.0}, {"focus_target_m": -0.0}, {"focus_target_m": -1.0},
    # shadow-match range gates
    {"shadow_band_hi": NAN}, {"shadow_band_hi": INF}, {"shadow_band_hi": -0.01},
    {"shadow_match_max_gain": NAN}, {"shadow_match_max_gain": INF},
    {"shadow_match_max_gain": 0.0}, {"shadow_match_max_gain": 16.0001},
    {"shadow_match_strength": NAN}, {"shadow_match_strength": -INF},
    {"shadow_match_strength": -0.1}, {"shadow_match_strength": 4.0001},
    # slab_batch_size
    {"slab_batch_size": 0}, {"slab_batch_size": -1}, {"slab_batch_size": False},
    {"slab_batch_size": 0.5}, {"slab_batch_size": NAN},
]


@pytest.mark.parametrize("kw", REJECT, ids=lambda k: repr(k))
def test_wrong_polarity_numeric_rejected(kw):
    # Any ValueError is acceptable (NaN slab_batch_size → int() ValueError).
    with pytest.raises(ValueError):
        _ro(**kw)


def test_slab_batch_size_inf_raises_valueerror():
    with pytest.raises(ValueError):
        _ro(slab_batch_size=INF)


@pytest.mark.parametrize(
    "kw",
    [
        {"occlusion_compositing": "ON"}, {"occlusion_compositing": "Off"},
        {"occlusion_compositing": " on"}, {"occlusion_compositing": None},
        {"occlusion_compositing": True},
        {"hdr_source": "Macos"}, {"hdr_source": "MACOS"}, {"hdr_source": "None"},
        {"hdr_source": None},
        {"depth_source": "LiDAR"}, {"depth_source": ""},
        {"shadow_match_mode": "ON", "hdr_source": "macos"},
        {"shadow_match_mode": False},
    ],
    ids=lambda k: repr(k),
)
def test_wrong_enum_case_variants_rejected(kw):
    with pytest.raises(ValueError):
        _ro(**kw)


def test_valid_boundaries_accepted():
    """Range ends are inclusive; 1-slab batches and tiny-positive focus OK."""
    for kw in [
        {"focus_xy": (0.0, 0.0)}, {"focus_xy": (1.0, 1.0)},
        {"focus_target_m": 5e-324},
        {"shadow_band_hi": 0.0}, {"shadow_band_hi": 1.0},
        {"shadow_match_max_gain": 1.0}, {"shadow_match_max_gain": 16.0},
        {"shadow_match_strength": 0.0}, {"shadow_match_strength": 4.0},
        {"slab_batch_size": 1}, {"slab_batch_size": None},
        {"shadow_match_mode": "on", "hdr_source": "macos"},
        {"depth_source": "monocular"},
    ]:
        _ro(**kw)  # must not raise


def test_slab_batch_size_bool_true_accepted_by_design():
    """Spec ruling: 'No bool rejection for slab_batch_size'. True == 1."""
    assert _ro(slab_batch_size=True).slab_batch_size is True


# Numeric fields that RenderOptions.__post_init__ must reject (GUI-MVP binds
# RenderOptions directly, bypassing argparse). Ranges mirror render()'s own
# checks (single source of truth).
UNVALIDATED = [
    {"depth_scale": NAN}, {"depth_scale": 0.0}, {"depth_scale": -10.0},
    {"depth_scale": INF},
    {"downres": 0}, {"downres": -2048},
    {"bright_pass_thresh": NAN},
    {"bright_pass_region_size": -31}, {"bright_pass_region_size": 256},
    {"bright_pass_floor": NAN}, {"bright_pass_floor": -1.0},
    {"bright_pass_norm_pct": NAN}, {"bright_pass_norm_pct": 150.0},
    {"bright_pass_norm_pct": 49.9}, {"bright_pass_norm_pct": 0.0},
]


@pytest.mark.parametrize("kw", UNVALIDATED, ids=lambda k: repr(k))
def test_unvalidated_fields_should_reject(kw):
    with pytest.raises(ValueError):
        _ro(**kw)


@pytest.mark.parametrize("kw", [
    {"bright_pass_norm_pct": 50.0}, {"bright_pass_norm_pct": 100.0},
    {"bright_pass_region_size": 0}, {"bright_pass_region_size": 255},
], ids=lambda k: repr(k))
def test_render_range_boundaries_accepted(kw):
    _ro(**kw)


# ---------------------------------------------------------------------------
# 2. replace / canonical_bench
# ---------------------------------------------------------------------------

def test_replace_revalidates_wrong_polarity():
    base = _ro()
    for kw in [{"focus_target_m": NAN}, {"hdr_source": "Macos"},
               {"slab_batch_size": -1}, {"focus_xy": (NAN, NAN)}]:
        with pytest.raises(ValueError):
            dataclasses.replace(base, **kw)


def test_canonical_bench_bogus_field_typeerror():
    with pytest.raises(TypeError):
        RenderOptions.canonical_bench(IMG, (0.4, 0.45), bogus_field=1)


@pytest.mark.parametrize(
    "kw",
    [{"occlusion_compositing": "ON"}, {"shadow_band_hi": NAN},
     {"hdr_source": "none", "shadow_match_mode": "on"},
     {"focus_target_m": -1.0}],
    ids=lambda k: repr(k),
)
def test_canonical_bench_invalid_override_valueerror(kw):
    with pytest.raises(ValueError):
        RenderOptions.canonical_bench(IMG, (0.4, 0.45), **kw)


def test_canonical_bench_invalid_focus_xy_valueerror():
    with pytest.raises(ValueError):
        RenderOptions.canonical_bench(IMG, (NAN, 0.45))
    with pytest.raises(ValueError):
        RenderOptions.canonical_bench(IMG, (1.5, 0.45))


def _readme_bench_argv() -> list[str] | None:
    readme = Path(__file__).resolve().parents[2] / "runs/vibe_bench/README.md"
    if not readme.is_file():
        return None
    lines = readme.read_text().splitlines()
    for i, ln in enumerate(lines):
        if "python -m summicron.render.convolve" in ln:
            block = []
            for nxt in lines[i + 1:]:
                block.append(nxt.rstrip().rstrip("\\"))
                if not nxt.rstrip().endswith("\\"):
                    break
            return shlex.split(" ".join(block))
    return None


def test_canonical_bench_equals_readme_command():
    argv = _readme_bench_argv()
    if argv is None:
        pytest.skip("runs/vibe_bench/README.md absent (runs/ is gitignored)")
    p = argparse.ArgumentParser()
    from summicron.render import _render_args

    _render_args.add_render_arguments(p)
    ns = p.parse_args(argv)
    from_readme = RenderOptions.from_argparse(ns)
    cb = RenderOptions.canonical_bench(
        ns.image, tuple(ns.focus_xy), out_path=Path(ns.output)
    )
    assert cb == from_readme
    assert cb.hdr_source == "macos" and cb.depth_scale == 1.0
    assert cb.downres == 2048 and cb.focus_xy == (0.40, 0.45)


def test_canonical_bench_macos_constructible_on_linux_but_render_refuses(
    monkeypatch,
):
    monkeypatch.setattr("sys.platform", "linux")
    opts = RenderOptions.canonical_bench(IMG, (0.4, 0.45))
    assert opts.hdr_source == "macos"
    called = []
    monkeypatch.setattr(
        conv, "_run_render_pipeline", lambda *a, **k: called.append(1)
    )
    with pytest.raises(ValueError, match="requires macOS; got linux"):
        render_image(opts)
    assert called == []  # refused BEFORE the pipeline (and any Quartz import)


# ---------------------------------------------------------------------------
# 3. _is_default_tensor + schema_version warning polarity
# ---------------------------------------------------------------------------

def test_is_default_tensor_path_variants(tmp_path):
    dflt = default_tensor_path()
    assert conv._is_default_tensor(dflt) is True
    assert conv._is_default_tensor(str(dflt)) is True
    dotted = dflt.parent / ".." / dflt.parent.name / dflt.name
    assert conv._is_default_tensor(dotted) is True
    link = tmp_path / "link.npz"
    link.symlink_to(dflt)
    assert conv._is_default_tensor(link) is True  # resolve() follows symlinks
    # A byte-identical COPY elsewhere is NOT the default (path-scoped by spec).
    cp = tmp_path / "copy.npz"
    shutil.copyfile(dflt, cp)
    assert conv._is_default_tensor(cp) is False
    assert conv._is_default_tensor(tmp_path / "nope" / "x.npz") is False
    assert conv._is_default_tensor("bad\0path") is False


def test_is_default_tensor_survives_missing_artifact(monkeypatch, tmp_path):
    import summicron.render._artifacts as art

    def boom():
        raise FileNotFoundError("artifact missing")

    monkeypatch.setattr(art, "default_tensor_path", boom)
    assert conv._is_default_tensor(tmp_path / "x.npz") is False


def test_is_default_tensor_none_never_raises():
    assert conv._is_default_tensor(None) is False


def _write_tensor_no_schema(path: Path) -> None:
    k = np.zeros((2, 2, 3, 3), dtype=np.float32)
    k[:, :, 1, 1] = 1.0
    np.savez(path, kernels=k, radial_edges=np.array([0.0, 0.5, 1.01]),
             defocus_ratios=np.array([0.0, 1.0]))


def test_schema_warning_still_fires_for_non_default(tmp_path):
    p = tmp_path / "foreign.npz"
    _write_tensor_no_schema(p)
    with pytest.warns(UserWarning, match="schema_version"):
        conv.load_tensor(p)


def test_schema_warning_suppressed_for_default_and_symlink(tmp_path):
    link = tmp_path / "link.npz"
    link.symlink_to(default_tensor_path())
    for p in (default_tensor_path(), link):
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            conv.load_tensor(p)
        assert not [w for w in rec if "schema_version" in str(w.message)]


# ---------------------------------------------------------------------------
# 4. render(stats=...) + _emit sink
# ---------------------------------------------------------------------------

_OD = np.array([0.7, 0.98, 1.37, 1.91, 2.68, 3.74, 5.23, 7.32, 10.23, 14.3,
                20.0])


def _tensor():
    R, D, k = 4, 11, 17
    kern = np.zeros((R, D, k, k), dtype=np.float32)
    c = k // 2
    kern[:, 0, c, c] = 1.0
    ys, xs = np.mgrid[-c:c + 1, -c:c + 1].astype(np.float32)
    for di in range(1, D):
        g = np.exp(-(xs ** 2 + ys ** 2) / (2 * (0.8 + 0.6 * di) ** 2))
        kern[:, di] = g / g.sum()
    return {
        "kernels": kern,
        "radial_edges": np.array([0.0, 0.3, 0.6, 0.85, 1.01]),
        "defocus_ratios": np.linspace(0, 1, D),
        "object_distances_m": _OD.copy(),
        "focus_distance_m": 0.7,
        "corpus_ref_long_edge": None,
        "otf_norm_applied": False,
        "delta_guard_applied": True,
    }


def _scene(h=40, w=40):
    rng = np.random.default_rng(7)
    img = rng.uniform(0, 0.8, (h, w, 3)).astype(np.float32)
    img[5:8, 5:8] = 1.0  # a few bright pixels for the bright pass
    depth = np.full((h, w), 0.7, np.float32)
    depth[:, w // 2:] = 8.0  # silhouette step
    return img, depth


@pytest.mark.parametrize("occ", ["off", "on"])
def test_stats_acc_alpha_contract_and_image_unchanged(occ):
    img, depth = _scene()
    kw = dict(depth_m=depth, bright_thresh=0.9, occlusion_compositing=occ,
              on_progress=lambda m: None)
    out_a, lin_a = conv.render(img, _tensor(), **kw)
    stats = {"junk": 123, "acc_alpha": "stale"}
    out_b, lin_b = conv.render(img, _tensor(), stats=stats, **kw)
    assert np.array_equal(out_a, out_b)
    assert np.array_equal(lin_a, lin_b)
    assert stats["junk"] == 123  # unrelated caller keys preserved
    acc = stats["acc_alpha"]  # stale value overwritten
    assert isinstance(acc, np.ndarray) and acc.shape == img.shape[:2]
    assert np.issubdtype(acc.dtype, np.floating)
    assert np.all(np.isfinite(acc))
    assert acc.min() >= 0.0 and acc.max() <= 1.0 + 1e-5
    assert isinstance(stats["bright_pixels_pct"], float)
    assert 0.0 < stats["bright_pixels_pct"] <= 100.0
    assert isinstance(stats["alpha_mean"], float)
    assert 0.0 <= stats["alpha_mean"] <= 1.0


def test_stats_non_dict_raises():
    """A list sink is rejected with TypeError (render() validates the sink type up front)."""
    img, depth = _scene(24, 24)
    with pytest.raises(TypeError):
        conv.render(img, _tensor(), depth_m=depth, stats=[],
                    on_progress=lambda m: None)


def test_global_acc_alpha_removed():
    img, depth = _scene(24, 24)
    conv.render(img, _tensor(), depth_m=depth, on_progress=lambda m: None)
    assert not hasattr(conv, "_LAST_LAYERED_ACC_ALPHA")


@pytest.mark.parametrize("occ", ["off", "on"])
def test_emit_sink_no_stdout_all_messages(capsys, occ):
    img, depth = _scene(32, 32)
    got: list[str] = []
    conv.render(img, _tensor(), depth_m=depth, bright_thresh=0.9,
                occlusion_compositing=occ, on_progress=got.append)
    out = capsys.readouterr().out
    assert out == "", f"stdout leaked with a sink: {out!r}"
    joined = "\n".join(got)
    assert "bright-pass:" in joined and "α-blend:" in joined
    marker = "occlusion-compositing: ON" if occ == "on" else "layered_gather:"
    assert marker in joined


def test_emit_none_prints(capsys):
    img, depth = _scene(24, 24)
    conv.render(img, _tensor(), depth_m=depth)
    out = capsys.readouterr().out
    assert "bright-pass:" in out and "α-blend:" in out


def test_emit_sink_exception_propagates():
    """Documented behavior: a raising sink aborts the render (not swallowed)."""
    img, depth = _scene(24, 24)

    def bad(msg):
        raise RuntimeError("sink broke")

    with pytest.raises(RuntimeError, match="sink broke"):
        conv.render(img, _tensor(), depth_m=depth, on_progress=bad)


# ---------------------------------------------------------------------------
# 5. save / out_dir through render_image (synthetic pipeline, no HEIC)
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_pipeline(monkeypatch):
    img, depth = _scene(32, 32)
    monkeypatch.setattr(conv, "load_image", lambda p: img.copy())
    monkeypatch.setattr(conv, "load_tensor", lambda p: _tensor())
    import summicron.depth.estimate as est

    monkeypatch.setattr(est, "extract_lidar_depth",
                        lambda p, target_hw=None, **kw: depth.copy())
    return img


def _list_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file())


def test_save_false_writes_nothing(fake_pipeline, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for extra in ({}, {"out_path": tmp_path / "o" / "x.jpg"},
                  {"out_dir": tmp_path / "od"}):
        res = render_image(
            RenderOptions(image_path=tmp_path / "img.heic",
                          save=False, **extra),
            on_progress=lambda m: None,
        )
        assert res.output_path is None
        assert res.output_u8.shape == (32, 32, 3)
    assert _list_files(tmp_path) == []
    assert not (tmp_path / "o").exists() and not (tmp_path / "od").exists()


def test_default_out_dir_resolves_under_cwd(fake_pipeline, tmp_path,
                                            monkeypatch):
    monkeypatch.chdir(tmp_path)
    render_image(
        RenderOptions(image_path=tmp_path / "img.heic",
                      ),
        on_progress=lambda m: None,
    )
    files = _list_files(tmp_path)
    assert files, "default out_dir wrote nothing"
    assert all(f.parent == tmp_path / "runs" / "render" for f in files)
    assert {f.suffix for f in files} == {".jpg", ".png"}


def test_diagnostics_center_ssim_none_without_focus(fake_pipeline, tmp_path):
    res = render_image(
        RenderOptions(image_path=tmp_path / "img.heic",
                      save=False),
        on_progress=lambda m: None,
    )
    d = res.diagnostics
    assert d.center_ssim is None
    assert d.focus_fidelity is None  # Q-a: focus_xy=None -> no metrics
    assert d.bright_pixels_pct > 0.0 and 0.0 < d.alpha_mean <= 1.0


def test_out_dir_save_reports_output_path(fake_pipeline, tmp_path,
                                          monkeypatch):
    monkeypatch.chdir(tmp_path)
    res = render_image(
        RenderOptions(image_path=tmp_path / "img.heic",
                      ),
        on_progress=lambda m: None,
    )
    assert res.output_path is not None and res.output_path.is_file()
