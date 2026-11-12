"""Adversarial probes for the follow-ups sweep (section19_followups).

Entrypoint parity (``convolve.main`` vs ``isummicron render``), the shared
``run_cli_render`` contract, the 48 MP numeric suggestion.
Small synthetic inputs only.
"""
from __future__ import annotations

import re
import sys
import warnings

import numpy as np
import pytest
from PIL import Image

from summicron import cli
from summicron.render import convolve as cv
from summicron.render.api import RenderOptions, render_image


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def png(tmp_path):
    p = tmp_path / "in.png"
    rng = np.random.default_rng(0)
    Image.fromarray(rng.integers(0, 255, (32, 40, 3), dtype=np.uint8)).save(p)
    return p


@pytest.fixture
def fake_lidar(monkeypatch):
    """Synthetic depth: near left half, far right half (no HEIC needed)."""
    import summicron.depth.estimate as de

    def fake(path, target_hw, **kw):
        h, w = target_hw
        d = np.full((h, w), 0.7, np.float32)
        d[:, w // 2:] = 8.0
        return d

    monkeypatch.setattr(de, "extract_lidar_depth", fake)


def _run_both(argv, capsys):
    """Run both entrypoints; return [(rc|exc, out, err), ...]."""
    res = []
    for entry in ("convolve", "cli"):
        try:
            rc = cv.main(list(argv)) if entry == "convolve" else cli.main(
                ["render", *argv])
        except BaseException as e:  # noqa: BLE001 - recording parity
            rc = type(e).__name__
        cap = capsys.readouterr()
        res.append((rc, cap.out, cap.err))
    return res


# ---------------------------------------------------------------------------
# (1) entrypoint parity matrix
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("env", ["0", "-5", "1.5e9", "", " 12 "])
def test_parity_bad_or_odd_env(monkeypatch, capsys, png, env):
    monkeypatch.setenv("ISUMMICRON_MAX_BYTES", env)
    calls = []
    monkeypatch.setattr(cv, "load_image", lambda *a, **k: calls.append(1))
    argv = ["--image", str(png), "--out-dir", str(png.parent)]
    (rc1, o1, e1), (rc2, o2, e2) = _run_both(argv, capsys)
    assert (rc1, e1) == (rc2, e2)
    if env.strip().lstrip("-").isdigit() and int(env) > 0:
        return  # " 12 " is a valid int; reaches load (spy returns None)
    assert rc1 == 2 and calls == []
    assert e1.startswith("render: ISUMMICRON_MAX_BYTES") and e1.count("\n") == 1


def test_parity_real_tiny_budget_inside_render(monkeypatch, capsys, png,
                                               fake_lidar):
    """Budget passes the pre-load parse but the real _memory_guard rejects."""
    monkeypatch.setenv("ISUMMICRON_MAX_BYTES", "1048576")
    argv = ["--image", str(png), "--out-dir", str(png.parent / "o")]
    (rc1, o1, e1), (rc2, o2, e2) = _run_both(argv, capsys)
    assert rc1 == rc2 == 2
    assert e1 == e2 and e1.startswith("render: Render would exceed")
    assert "raise ISUMMICRON_MAX_BYTES" in e1
    assert o1 == o2  # identical progress lines before the failure
    assert not (png.parent / "o").exists() or not any(
        (png.parent / "o").iterdir())


def test_parity_missing_image(capsys, tmp_path):
    argv = ["--image", str(tmp_path / "nope.png"), "--out-dir", str(tmp_path)]
    (rc1, o1, e1), (rc2, o2, e2) = _run_both(argv, capsys)
    assert rc1 == rc2 == 2 and e1 == e2 and e1.startswith("render: ")


def test_parity_bad_tensor_wrong_keys(capsys, png, tmp_path, fake_lidar):
    bad = tmp_path / "bad.npz"
    np.savez(bad, foo=np.zeros(3))
    argv = ["--image", str(png), "--tensor", str(bad), "--out-dir",
            str(tmp_path)]
    (rc1, o1, e1), (rc2, o2, e2) = _run_both(argv, capsys)
    assert rc1 == rc2 == 2 and e1 == e2 and e1.startswith("render: ")


def test_parity_corrupt_tensor_not_a_zip(capsys, png, tmp_path, fake_lidar):
    """Documents: a non-zip tensor file escapes the contract on BOTH
    entrypoints identically (parity holds; whatever type it is)."""
    bad = tmp_path / "bad.npz"
    bad.write_bytes(b"not a zip file at all")
    argv = ["--image", str(png), "--tensor", str(bad), "--out-dir",
            str(tmp_path)]
    (rc1, _, e1), (rc2, _, e2) = _run_both(argv, capsys)
    assert rc1 == rc2
    assert e1 == e2


def test_parity_success_occ_on(monkeypatch, capsys, png, tmp_path,
                                    fake_lidar):
    outs = []
    for entry in ("convolve", "cli"):
        od = tmp_path / entry
        argv = ["--image", str(png), "--out-dir", str(od),
                "--occlusion-compositing", "on"]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            rc = cv.main(argv) if entry == "convolve" else cli.main(
                ["render", *argv])
        cap = capsys.readouterr()
        assert rc == 0
        files = sorted(p.name for p in od.iterdir())
        assert files, entry
        outs.append((cap.out.replace(str(od), "<OD>"), files,
                     [np.asarray(Image.open(od / f)) for f in files
                      if f.endswith((".png", ".jpg"))]))
    assert outs[0][0] == outs[1][0]
    assert outs[0][1] == outs[1][1]
    for a, b in zip(outs[0][2], outs[1][2]):
        assert np.array_equal(a, b)


def test_hdr_platform_preflight_parity(monkeypatch, capsys, png):
    """render_image runs _check_platform; convolve.main bypasses it."""
    monkeypatch.setattr(sys, "platform", "linux")
    import summicron.io.hdr_load_macos as hl

    def no_pyobjc(path):
        raise ImportError("No module named 'Quartz'")

    monkeypatch.setattr(hl, "load_hdr_heic", no_pyobjc)
    argv = ["--image", str(png), "--out-dir", str(png.parent),
            "--hdr-source", "macos"]
    (rc1, _, e1), (rc2, _, e2) = _run_both(argv, capsys)
    # Recorded, not asserted equal: see report.
    print(f"PLATFORM convolve={rc1!r} {e1.strip()[:80]!r} | "
          f"cli={rc2!r} {e2.strip()[:80]!r}", file=sys.__stdout__)
    assert rc2 == 2 and "requires macOS" in e2


# ---------------------------------------------------------------------------
# run_cli_render: what propagates, what is swallowed
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("exc", [KeyboardInterrupt, SystemExit, TypeError,
                                 MemoryError, AttributeError])
def test_run_cli_render_propagates(exc):
    def boom():
        raise exc("x")

    with pytest.raises(exc):
        cv.run_cli_render(boom)


@pytest.mark.parametrize("exc", [NotImplementedError, RecursionError])
def test_run_cli_render_reraises_bug_class_subclasses(monkeypatch, exc):
    """Bug-class RuntimeError subclasses propagate with a traceback."""
    monkeypatch.delenv("ISUMMICRON_DEBUG", raising=False)

    def boom():
        raise exc("x")

    with pytest.raises(exc):
        cv.run_cli_render(boom)


def test_nan_gate_reaches_user_as_one_line(capsys):
    def boom():
        raise RuntimeError("Gate FAIL: NaN in output")

    assert cv.run_cli_render(boom) == 2
    assert capsys.readouterr().err == "render: Gate FAIL: NaN in output\n"


def test_budget_parsed_before_run_and_once_per_entry(monkeypatch):
    seen = []
    real = cv._budget_bytes
    monkeypatch.setattr(cv, "_budget_bytes",
                        lambda: seen.append("b") or real())
    assert cv.run_cli_render(lambda: seen.append("run")) == 0
    assert seen == ["b", "run"]


# ---------------------------------------------------------------------------
# (3) 48 MP suggestion
# ---------------------------------------------------------------------------

_SHAPES = [
    (8064, 6048), (6048, 8064), (6000, 6001), (6001, 6000), (6000, 6000),
    (1, 36_000_001), (36_000_001, 1), (9000, 4001), (12000, 9000),
    (6001, 5999), (8000, 4500), (7000, 7000),
]


@pytest.mark.parametrize("h,w", _SHAPES)
def test_pixel_suggestion_passes_and_is_maximal(h, w):
    if h * w <= cv.MAX_PIXELS:
        cv._pixel_guard(h, w)
        return
    with pytest.raises(RuntimeError) as exc:
        cv._pixel_guard(h, w)
    n = int(re.search(r"--downres (\d+)", str(exc.value)).group(1))
    assert n % 64 == 0 and n < max(h, w)
    img = np.zeros((1, 1, 3), np.float32)  # shape math only
    sc = n / max(h, w)
    nh, nw = max(1, int(round(h * sc))), max(1, int(round(w * sc)))
    cv._pixel_guard(nh, nw)
    # aspect preserved to rounding
    assert abs(nh / nw - h / w) <= (1 / min(nh, nw)) * (h / w) + 1e-9
    sc2 = (n + 64) / max(h, w)
    assert max(1, round(h * sc2)) * max(1, round(w * sc2)) > cv.MAX_PIXELS
    del img


@pytest.mark.parametrize("h,w", [(8064, 6048), (6048, 8064), (6000, 6001)])
def test_pixel_suggestion_vs_memory_budget(h, w):
    """With D known (as at the render call site) the single --downres
    suggestion passes BOTH the pixel guard and the memory guard."""
    with pytest.raises(RuntimeError) as exc:
        cv._pixel_guard(h, w, 11)
    n = int(re.search(r"--downres (\d+)", str(exc.value)).group(1))
    sc = n / max(h, w)
    nh, nw = int(round(h * sc)), int(round(w * sc))
    cv._memory_guard(nh, nw, 11)  # must not raise (includes pixel guard)


def test_budget_below_base_hint_wording():
    """Budget below the platform base: sentence ends 'raise ISUMMICRON_MAX_BYTES.'"""
    import os
    old = os.environ.get("ISUMMICRON_MAX_BYTES")
    os.environ["ISUMMICRON_MAX_BYTES"] = str(cv.PLATFORM_BASE - 1)
    try:
        with pytest.raises(cv.MemoryBudgetError) as exc:
            cv._memory_guard(64, 64, 11)
    finally:
        if old is None:
            del os.environ["ISUMMICRON_MAX_BYTES"]
        else:
            os.environ["ISUMMICRON_MAX_BYTES"] = old
    print(f"BELOW_BASE: {exc.value}", file=sys.__stdout__)
    assert str(exc.value).endswith("raise ISUMMICRON_MAX_BYTES.")
