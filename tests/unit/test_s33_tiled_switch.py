"""S33 Tier B WP3: the --engine switch (names, refusals, ON note, CLI contract).

Small synthetic frames only; no runs/ and no IMG_6915.
"""
from __future__ import annotations

import io
import contextlib

import numpy as np
import pytest

from summicron.render import convolve as cv
from tests.fixtures.synthetic_tensor import make_synthetic_tensor

H = W = 40


def _tensor(ksize=17):
    return make_synthetic_tensor(12, 11, ksize, ksize, corpus_ref_long_edge=H)


def _scene(t):
    rng = np.random.default_rng(3)
    img = (rng.random((H, W, 3)) * 0.4).astype(np.float32)
    img[10:13, 25:28] = 0.95
    od = np.asarray(t["object_distances_m"])
    depth = np.full((H, W), float(od[5]), np.float32)
    depth[:, : W // 2] = float(od[8])
    return img, depth


def _render(engine=None, ksize=17, **kw):
    t = _tensor(ksize)
    img, depth = _scene(t)
    lines: list[str] = []
    args = dict(depth_m=depth, bright_thresh=0.6, on_progress=lines.append,
                bright_stamp_gain=2.0)
    args.update(kw)
    if engine is not None:
        args["engine"] = engine
    out = cv.render(img, t, **args)
    return out, lines


def test_unknown_engine_name():
    with pytest.raises(ValueError, match=r"engine must be one of \('reference', 'tiled', 'gpu'\), got 'cuda'"):
        _render("cuda")


@pytest.fixture
def no_mlx(monkeypatch):
    """Simulate an environment without mlx (works in both envs)."""
    import sys

    monkeypatch.setitem(sys.modules, "mlx", None)
    monkeypatch.setitem(sys.modules, "mlx.core", None)


@pytest.mark.parametrize("occ", ["off", "on"])
def test_gpu_not_available_even_with_occlusion_on(occ, no_mlx):
    with pytest.raises(ValueError, match="engine 'gpu' is not available"):
        _render("gpu", occlusion_compositing=occ)


def test_k_over_255_not_available_for_tiled():
    t = make_synthetic_tensor(12, 11, 257, 257, corpus_ref_long_edge=H)
    img, depth = _scene(t)
    with pytest.raises(ValueError, match="engine 'tiled' is not available for .*; use --engine reference"):
        cv.render(img, t, depth_m=depth, engine="tiled")


def test_explicit_reference_equals_default_bytes():
    (u8a, la), _ = _render(None)
    (u8b, lb), _ = _render("reference")
    assert la.tobytes() == lb.tobytes()
    assert u8a.tobytes() == u8b.tobytes()


def test_tiled_runs_and_engine_trace(monkeypatch):
    from summicron.render import _engine_tiled as et

    calls = {"spec": 0, "ref": 0}
    orig = et._kernel_spectra

    def spy(*a, **k):
        calls["spec"] += 1
        return orig(*a, **k)

    monkeypatch.setattr(et, "_kernel_spectra", spy)
    ref_apply = cv._apply_kernel

    def ref_spy(*a, **k):
        calls["ref"] += 1
        return ref_apply(*a, **k)

    monkeypatch.setattr(cv, "_apply_kernel", ref_spy)
    (_, lin_t), _ = _render("tiled")
    assert calls["spec"] > 0 and calls["ref"] == 0
    (_, lin_r), _ = _render("reference")
    assert calls["ref"] > 0
    e = np.abs(lin_t - lin_r) / np.maximum(lin_r, 1.0)
    assert float(e.max()) <= 1e-5


def test_on_note_and_bit_identity():
    (u8r, lr), lines_r = _render("reference", occlusion_compositing="on")
    (u8t, lt), lines_t = _render("tiled", occlusion_compositing="on")
    assert lr.tobytes() == lt.tobytes() and u8r.tobytes() == u8t.tobytes()
    note = ("note: engine tiled -> reference (--occlusion-compositing on runs "
            "on the reference engine in 0.5.2)")
    assert lines_t.count(note) == 1
    assert [ln for ln in lines_t if ln != note] == lines_r


def test_no_note_with_default_engine_on():
    _, lines = _render(None, occlusion_compositing="on")
    assert not [ln for ln in lines if ln.startswith("note: engine")]
    _, lines = _render("reference", occlusion_compositing="on")
    assert not [ln for ln in lines if ln.startswith("note: engine")]


def test_resolve_engine_pure():
    from summicron.render._resolve import ENGINE_ON_NOTE, resolve_engine

    assert resolve_engine("reference", "on") == ("reference", None)
    assert resolve_engine("reference", "off") == ("reference", None)
    assert resolve_engine("tiled", "off") == ("tiled", None)
    assert resolve_engine("tiled", "on") == ("reference", ENGINE_ON_NOTE)


def test_cli_flag_parses_and_defaults():
    p = cv.build_parser()
    assert p.parse_args(["--image", "x.heic"]).engine == "reference"
    assert p.parse_args(["--image", "x.heic", "--engine", "tiled"]).engine == "tiled"
    with pytest.raises(SystemExit):
        with contextlib.redirect_stderr(io.StringIO()):
            p.parse_args(["--image", "x.heic", "--engine", "cuda"])


def test_cli_gpu_one_line_exit2(tmp_path, monkeypatch, capsys):
    """run_cli_render turns the gpu refusal into one 'render:' line, exit 2."""
    import subprocess
    import sys
    from pathlib import Path

    img = Path(__file__).resolve().parents[2] / "iphoneImage" / "Demo_input.HEIC"
    if not img.exists():
        pytest.skip("kitchen HEIC absent")
    r = subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.modules['mlx'] = None; sys.modules['mlx.core'] = None\n"
         "import runpy; sys.argv = ['convolve', '--image', %r, '--engine', "
         "'gpu', '--out-dir', %r, '--downres', '256']\n"
         "runpy.run_module('summicron.render.convolve', run_name='__main__')"
         % (str(img), str(tmp_path))],
        capture_output=True, text=True,
        env={**__import__("os").environ,
             "PYTHONPATH": str(Path(__file__).resolve().parents[2])},
    )
    assert r.returncode == 2, (r.returncode, r.stderr[-500:])
    err = [ln for ln in r.stderr.splitlines() if ln.strip()]
    assert any(ln.startswith("render: ") and "not available" in ln for ln in err)


def test_import_is_lazy_for_reference():
    import subprocess
    import sys
    from pathlib import Path

    code = (
        "import sys, numpy as np\n"
        "from summicron.render import convolve\n"
        "assert 'summicron.render._engine_tiled' not in sys.modules\n"
        "assert 'mlx' not in sys.modules\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       env={**__import__("os").environ,
                            "PYTHONPATH": str(Path(__file__).resolve().parents[2])})
    assert r.returncode == 0, r.stderr
