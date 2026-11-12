"""Follow-ups sweep (2026-11-10): CLI error parity, hints, env validation."""
from __future__ import annotations

import warnings

import numpy as np
import pytest
from PIL import Image

from summicron import cli
from summicron.render import convolve as cv
from tests.unit.test_over_scatter_fused import _step_scene, _tensor


@pytest.fixture
def png(tmp_path):
    p = tmp_path / "in.png"
    Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(p)
    return p


def _force_budget_error(monkeypatch):
    def boom(opts, **kw):
        cv._memory_guard(512, 512, 11)

    import sys

    # render_image resolves convolve via sys.modules at call time; patch that
    # object (robust to reloads elsewhere in the suite) and this binding.
    live = sys.modules["summicron.render.convolve"]
    monkeypatch.setattr(live, "_run_render_pipeline", boom)
    monkeypatch.setattr(cv, "_run_render_pipeline", boom)


def test_both_entrypoints_exit2_identical_message(monkeypatch, capsys, png):
    monkeypatch.setenv("ISUMMICRON_MAX_BYTES", "1048576")
    _force_budget_error(monkeypatch)
    rc1 = cv.main(["--image", str(png), "--out-dir", str(png.parent)])
    e1 = capsys.readouterr().err
    rc2 = cli.main(["render", "--image", str(png), "--out-dir", str(png.parent)])
    e2 = capsys.readouterr().err
    assert rc1 == rc2 == 2
    assert e1 == e2 and e1.startswith("render: ")
    assert len(e1.strip().splitlines()) == 1
    assert "budget below the fixed platform overhead" in e1


@pytest.mark.parametrize("entry", ["convolve", "cli"])
def test_bad_env_exits2_before_image_load(monkeypatch, capsys, png, entry):
    monkeypatch.setenv("ISUMMICRON_MAX_BYTES", "banana")
    calls = []
    monkeypatch.setattr(cv, "load_image", lambda *a, **k: calls.append(1))
    argv = ["--image", str(png), "--out-dir", str(png.parent)]
    rc = cv.main(argv) if entry == "convolve" else cli.main(["render", *argv])
    err = capsys.readouterr().err
    assert rc == 2 and calls == []
    assert "ISUMMICRON_MAX_BYTES must be an integer > 0" in err


def test_pixel_guard_names_passing_number():
    h, w = 8064, 6048  # 48 MP
    with pytest.raises(RuntimeError, match=r"--downres (\d+)") as exc:
        cv._pixel_guard(h, w)
    n = int(__import__("re").search(r"--downres (\d+)", str(exc.value)).group(1))
    assert n % 64 == 0
    sc = n / max(h, w)
    cv._pixel_guard(round(h * sc), round(w * sc))  # must not raise
    assert round(h * (n + 64) / max(h, w)) * round(w * (n + 64) / max(h, w)) > cv.MAX_PIXELS


def test_budget_below_base_message(monkeypatch):
    monkeypatch.setenv("ISUMMICRON_MAX_BYTES", str(cv.PLATFORM_BASE // 2))
    with pytest.raises(cv.MemoryBudgetError, match="raise ISUMMICRON_MAX_BYTES") as exc:
        cv._memory_guard(512, 512, 11)
    assert "smaller than 256" not in str(exc.value)


# ---------------------------------------------------------------------------
# Fix round: single-step downres suggestion, run_cli_render re-raise, debug
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("h,w", [(8064, 6048), (6000, 6001)])
def test_pixel_guard_with_d_suggestion_fits_both_guards(h, w):
    import re
    with pytest.raises(RuntimeError) as exc:
        cv._pixel_guard(h, w, 11)
    n = int(re.search(r"--downres (\d+)", str(exc.value)).group(1))
    sc = n / max(h, w)
    nh, nw = max(1, round(h * sc)), max(1, round(w * sc))
    cv._pixel_guard(nh, nw)
    cv._memory_guard(nh, nw, 11)


@pytest.mark.parametrize("exc_t", [NotImplementedError, RecursionError])
def test_run_cli_render_reraises_programming_errors(exc_t, monkeypatch):
    monkeypatch.delenv("ISUMMICRON_DEBUG", raising=False)

    def boom():
        raise exc_t("x")
    with pytest.raises(exc_t):
        cv.run_cli_render(boom)


def test_run_cli_render_debug_env(monkeypatch, capsys):
    def boom():
        raise RuntimeError("kaput")
    monkeypatch.setenv("ISUMMICRON_DEBUG", "1")
    with pytest.raises(RuntimeError, match="kaput"):
        cv.run_cli_render(boom)
    for v in ("", "0"):
        monkeypatch.setenv("ISUMMICRON_DEBUG", v)
        assert cv.run_cli_render(boom) == 2
    assert "render: kaput" in capsys.readouterr().err
