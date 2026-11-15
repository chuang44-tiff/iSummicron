"""S33 Tier C WP3: the --engine gpu switch (locked-spec-S33-gpu 3.2, 3.3).

Runs in both envs. Tests that need a usable gpu engine skip when mlx or Metal
is absent; the absent-mlx cases simulate it with ``sys.modules`` so they mean
the same in both envs. Small synthetic frames only; the kitchen case is slow
and skips without the HEIC.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from summicron.render import _resolve
from summicron.render import convolve as cv
from tests.fixtures.synthetic_tensor import make_synthetic_tensor

H = W = 40
REPO = Path(__file__).resolve().parents[2]


def _gpu_reason():
    from summicron.render._engine_gpu import availability

    return availability()


needs_gpu = pytest.mark.skipif(
    _gpu_reason() is not None, reason="gpu engine unavailable (mlx or Metal absent)"
)


def _tensor(k=17):
    return make_synthetic_tensor(12, 11, k, k, corpus_ref_long_edge=H)


def _scene(t):
    rng = np.random.default_rng(3)
    img = (rng.random((H, W, 3)) * 0.4).astype(np.float32)
    img[10:13, 25:28] = 0.95
    od = np.asarray(t["object_distances_m"])
    depth = np.full((H, W), float(od[5]), np.float32)
    depth[:, : W // 2] = float(od[8])
    return img, depth


def _render(engine, t=None, **kw):
    t = t if t is not None else _tensor()
    img, depth = _scene(t)
    lines: list[str] = []
    args = dict(depth_m=depth, bright_thresh=0.6, on_progress=lines.append,
                bright_stamp_gain=2.0, engine=engine)
    args.update(kw)
    return cv.render(img, t, **args), lines


@pytest.fixture
def no_mlx(monkeypatch):
    monkeypatch.setitem(sys.modules, "mlx", None)
    monkeypatch.setitem(sys.modules, "mlx.core", None)


# ---------------------------------------------------------------- resolution

def test_resolve_engine_table():
    r = _resolve.resolve_engine
    assert r("tiled", "on") == ("reference", _resolve.ENGINE_ON_NOTE)
    assert r("gpu", "on") == ("reference", _resolve.ENGINE_ON_NOTE_GPU)
    for eng in ("reference", "tiled", "gpu"):
        assert r(eng, "off") == (eng, None)
    assert r("reference", "on") == ("reference", None)


def test_notes_text():
    assert _resolve.ENGINE_ON_NOTE_GPU == (
        "note: engine gpu -> reference (--occlusion-compositing on runs on the "
        "reference engine in 0.5.2)"
    )
    assert _resolve.ENGINE_ON_NOTE == (
        "note: engine tiled -> reference (--occlusion-compositing on runs on "
        "the reference engine in 0.5.2)"
    )


def test_bad_name_raises_before_availability(no_mlx):
    with pytest.raises(ValueError, match="engine must be one of"):
        _render("cuda")


@pytest.mark.parametrize("occ", ["off", "on"])
def test_unavailable_before_on_rule(no_mlx, occ):
    with pytest.raises(ValueError) as ei:
        _render("gpu", occlusion_compositing=occ)
    msg = str(ei.value)
    assert msg.startswith("engine 'gpu' is not available: mlx is not importable")
    assert "[gpu] extra" in msg


def test_no_metal_is_not_available(monkeypatch):
    mx = pytest.importorskip("mlx.core")
    monkeypatch.setattr(mx.metal, "is_available", lambda: False)
    with pytest.raises(ValueError, match="engine 'gpu' is not available: no Metal GPU device"):
        _render("gpu")


# ------------------------------------------------------------------ refusals

@needs_gpu
def test_k_over_255_refused():
    t = make_synthetic_tensor(12, 11, 257, 257, corpus_ref_long_edge=H)
    with pytest.raises(ValueError, match="engine 'gpu' is not available for a kernel side 257 > 255; use --engine reference"):
        _render("gpu", t=t)


@needs_gpu
def test_non_square_refused():
    t = make_synthetic_tensor(12, 11, 17, 19, corpus_ref_long_edge=H)
    with pytest.raises(ValueError, match="engine 'gpu' is not available for a non-square kernel"):
        _render("gpu", t=t)


@needs_gpu
def test_refusal_runs_no_engine(monkeypatch):
    from summicron.render import _engine_gpu as eg

    def boom(*a, **k):
        raise AssertionError("engine ran")

    monkeypatch.setattr(eg, "gpu_layered_gather", boom)
    t = make_synthetic_tensor(12, 11, 257, 257, corpus_ref_long_edge=H)
    with pytest.raises(ValueError, match="not available for"):
        _render("gpu", t=t)


# ----------------------------------------------------------------- ON note

@needs_gpu
def test_gpu_on_notes_and_equals_reference():
    (ref, ref_lines) = _render("reference", occlusion_compositing="on")
    (out, lines) = _render("gpu", occlusion_compositing="on")
    notes = [ln for ln in lines if ln.startswith("note: engine")]
    assert notes == [_resolve.ENGINE_ON_NOTE_GPU]
    assert not [ln for ln in ref_lines if ln.startswith("note: engine")]
    for a, b in zip(ref, out):
        assert np.array_equal(a, b)
    assert [ln for ln in lines if ln != _resolve.ENGINE_ON_NOTE_GPU] == ref_lines


@needs_gpu
def test_gpu_on_never_enters_gpu_engine(monkeypatch):
    from summicron.render import _engine_gpu as eg

    def boom(*a, **k):
        raise AssertionError("gpu gather ran on the ON path")

    monkeypatch.setattr(eg, "gpu_layered_gather", boom)
    _render("gpu", occlusion_compositing="on")


# ------------------------------------------------------------ OFF dispatch

@needs_gpu
def test_gpu_off_runs_gpu_gather_not_cpu(monkeypatch):
    from summicron.render import _engine_gpu as eg
    from summicron.render import _engine_tiled as et

    calls = {"waves": 0, "apply": 0, "ref": 0, "tiled": 0, "state": 0}
    orig = eg._run_waves

    def w(*a, **k):
        calls["waves"] += 1
        return orig(*a, **k)

    def count(key, fn):
        def f(*a, **k):
            calls[key] += 1
            return fn(*a, **k)
        return f

    monkeypatch.setattr(eg, "_run_waves", w)
    monkeypatch.setattr(cv, "_apply_kernel", count("apply", cv._apply_kernel))
    monkeypatch.setattr(cv, "_layered_gather_diffuse", count("ref", cv._layered_gather_diffuse))
    monkeypatch.setattr(et, "tiled_layered_gather", count("tiled", et.tiled_layered_gather))
    monkeypatch.setattr(cv, "_build_image_fft_state", count("state", cv._build_image_fft_state))
    out, _ = _render("gpu")
    assert calls["waves"] > 0
    assert calls["apply"] == calls["ref"] == calls["tiled"] == calls["state"] == 0
    ref, _ = _render("reference")
    assert np.isfinite(out[0]).all()
    assert np.allclose(out[0], ref[0], rtol=1e-3, atol=1e-4)


@needs_gpu
def test_execution_proof_cpu_executors_raise(monkeypatch):
    """Positive proof that MLX ran: every CPU executor raises, gpu renders."""
    import mlx.core as mx

    from summicron.render import _engine_gpu as eg
    from summicron.render import _engine_tiled as et

    def boom(*a, **k):
        raise AssertionError("CPU executor ran inside a gpu render")

    for mod, name in ((et, "tiled_layered_gather"), (et, "_convolve_chunk"),
                      (et, "tile_convolve_full"), (cv, "_apply_kernel")):
        monkeypatch.setattr(mod, name, boom)
    dev = mx.default_device()
    eg._LAST_RUN.clear()
    out, _ = _render("gpu")
    assert np.isfinite(out[0]).all()
    assert eg._LAST_RUN["device"] == "gpu"
    assert eg._LAST_RUN["waves"] > 0
    assert eg._LAST_RUN["peak_bytes"] > 0
    assert mx.default_device() == dev


# --------------------------------------------------------------- leak (a)

@needs_gpu
def test_gpu_then_reference_equals_fresh_process_reference(tmp_path):
    ref_here_before, _ = _render("reference")
    _render("gpu")
    ref_after, _ = _render("reference")
    for a, b in zip(ref_here_before, ref_after):
        assert np.array_equal(a, b)
    code = (
        "import sys, numpy as np\n"
        "sys.path.insert(0, %r)\n"
        "import tests.unit.test_s33_gpu_switch as m\n"
        "out, _ = m._render('reference')\n"
        "np.savez(%r, a=out[0], b=out[1])\n" % (str(REPO), str(tmp_path / "f.npz"))
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       env={**os.environ, "PYTHONPATH": str(REPO)})
    assert r.returncode == 0, r.stderr[-800:]
    z = np.load(tmp_path / "f.npz")
    assert np.array_equal(z["a"], ref_after[0])
    assert np.array_equal(z["b"], ref_after[1])


# --------------------------------------------------------------------- CLI

def _cli(engine_args, tmp_path, block_mlx):
    img = REPO / "iphoneImage" / "Demo_input.HEIC"
    if not img.exists():
        pytest.skip("kitchen HEIC absent")
    pre = "import sys; sys.modules['mlx'] = None; sys.modules['mlx.core'] = None\n" if block_mlx else ""
    code = (pre + "import runpy, sys\n"
            "sys.argv = ['convolve', '--image', %r, '--out-dir', %r, '--downres', '256'] + %r\n"
            "runpy.run_module('summicron.render.convolve', run_name='__main__')"
            % (str(img), str(tmp_path), list(engine_args)))
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                          env={**os.environ, "PYTHONPATH": str(REPO)})


def test_cli_unavailable_one_line_exit2(tmp_path):
    r = _cli(["--engine", "gpu"], tmp_path, True)
    assert r.returncode == 2, r.stderr[-500:]
    lines = [ln for ln in r.stderr.splitlines() if ln.startswith("render:")]
    assert len(lines) == 1 and "not available" in lines[0] and "mlx is not importable" in lines[0]


@pytest.mark.slow
@needs_gpu
def test_cli_gpu_on_note_and_md5_equals_reference(tmp_path):
    outs = {}
    for name, extra in (("gpu", ["--engine", "gpu"]), ("reference", ["--engine", "reference"])):
        d = tmp_path / name
        d.mkdir()
        r = _cli(extra + ["--occlusion-compositing", "on"], d, False)
        assert r.returncode == 0, r.stderr[-800:]
        if name == "gpu":
            assert r.stdout.count(_resolve.ENGINE_ON_NOTE_GPU) == 1
        else:
            assert "note: engine" not in r.stdout
        files = sorted(p for p in d.rglob("*") if p.is_file() and p.suffix.lower() in (".png", ".tif", ".tiff", ".jpg", ".heic"))
        assert files, list(d.rglob("*"))
        outs[name] = hashlib.md5(files[0].read_bytes()).hexdigest()
    assert outs["gpu"] == outs["reference"]
