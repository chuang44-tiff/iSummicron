"""G1 — canonical bench back-compat SHOWSTOPPER (HYG-0 form).

The md5 constant ``summicron.render._bench.G1_MD5`` IS the contract; the
on-disk baseline ``runs/vibe_bench/IMG_6915_R2_layered.jpg`` is optional
(gitignored) and only used for a max-abs diagnostic on failure.

FAST tests (always run):
- Baseline file, if present, has md5 == ``G1_MD5`` and decodes as uint8 RGB
  (skips with the exact ``cp`` restore command if absent).
- Wheel-shipped tensor bytes md5 == ``TENSOR_MD5`` / ``TENSOR_MD5_50``.
- ``--layered-gather`` flag is REMOVED from argparse (P1.10.4 C5).

SLOW test (``-m slow``, 5 cases):
- Re-run the canonical IMG_6915 bench (argv built from ``CANONICAL_FLAGS``,
  default 35 mm tensor) via both entrypoints x {OFF, ``--occlusion-compositing
  on``} and assert output md5 == ``G1_MD5`` / ``ON_MD5``; plus ``--lens 50``
  (convolve only) == ``LENS50_MD5``. On failure the message carries macOS
  build, pyobjc version, tensor md5 and max-abs vs the baseline (if present)
  so OS drift explains itself.
"""
from __future__ import annotations

import hashlib
import os
import platform
import sys
import subprocess
from pathlib import Path

import numpy as np
import pytest

from summicron.render._bench import (
    CANONICAL_FLAGS,
    G1_FOCUS_XY,
    G1_IMAGE,
    G1_MD5,
    LENS50_MD5,
    ON_MD5,
    TENSOR_MD5,
    TENSOR_MD5_50,
)

ROOT = Path(__file__).resolve().parents[2]
BASELINE_JPG = ROOT / "runs" / "vibe_bench" / "IMG_6915_R2_layered.jpg"
CANONICAL_IMAGE = ROOT / G1_IMAGE
RESTORE_CMD = (
    "cp runs/vibe_bench/IMG_6915_G1_baseline_s20.jpg "
    "runs/vibe_bench/IMG_6915_R2_layered.jpg"
)


def _md5(path: Path) -> str:
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def _skip_if_no_baseline() -> None:
    if not BASELINE_JPG.exists():
        pytest.skip(
            f"G1 baseline absent at {BASELINE_JPG} (gitignored, optional; the "
            f"md5 constant is the contract). Restore with: {RESTORE_CMD}"
        )


def _env_diagnostics(out_path: Path | None = None) -> str:
    """macOS build, pyobjc version, tensor md5, max-abs vs baseline."""
    lines = []
    mac_ver = platform.mac_ver()[0] or "n/a"
    try:
        build = subprocess.run(
            ["sw_vers", "-buildVersion"], capture_output=True, text=True,
            timeout=10,
        ).stdout.strip() or "n/a"
    except (OSError, subprocess.SubprocessError):
        build = "n/a"
    lines.append(f"macOS: {mac_ver} build {build}")
    try:
        import importlib.metadata as md

        pyobjc = md.version("pyobjc-core")
    except Exception:  # noqa: BLE001 — diagnostic only
        pyobjc = "not installed"
    lines.append(f"pyobjc-core: {pyobjc}")
    try:
        from summicron.render._artifacts import default_tensor_path

        tp = default_tensor_path()
        lines.append(f"tensor md5: {_md5(tp)} (expected {TENSOR_MD5}) @ {tp}")
    except Exception as exc:  # noqa: BLE001 — diagnostic only
        lines.append(f"tensor md5: unavailable ({exc!r})")
    if out_path is not None and out_path.exists() and BASELINE_JPG.exists():
        from PIL import Image

        with Image.open(str(BASELINE_JPG)) as im:
            pre = np.asarray(im.convert("RGB"), dtype=np.int16)
        with Image.open(str(out_path)) as im:
            post = np.asarray(im.convert("RGB"), dtype=np.int16)
        if pre.shape != post.shape:
            lines.append(f"max-abs vs baseline: shape drift {pre.shape} vs {post.shape}")
        else:
            d = np.abs(pre - post)
            lines.append(
                f"max-abs vs baseline: {int(d.max())} "
                f"(differing px: {int(np.count_nonzero(d.max(axis=2)))})"
            )
    else:
        lines.append("max-abs vs baseline: n/a (baseline or output absent)")
    return "\n".join(lines)


# --- FAST tests --------------------------------------------------------------


def test_backcompat_baseline_exists():
    """Baseline JPG, if present, must be the G1-pinned bytes."""
    _skip_if_no_baseline()
    got = _md5(BASELINE_JPG)
    assert got == G1_MD5, (
        f"baseline {BASELINE_JPG} md5 {got} != G1_MD5 {G1_MD5}; restore "
        f"with: {RESTORE_CMD}"
    )


def test_backcompat_baseline_readable():
    """Baseline file is a valid JPG decodable as a uint8 RGB array."""
    _skip_if_no_baseline()
    from PIL import Image

    assert _md5(BASELINE_JPG) == G1_MD5
    with Image.open(str(BASELINE_JPG)) as im:
        arr = np.asarray(im.convert("RGB"), dtype=np.uint8)
    assert arr.ndim == 3 and arr.shape[2] == 3
    assert arr.dtype == np.uint8


def test_default_tensor_md5_matches_pin():
    """HYG-0 7b — wheel-shipped tensor bytes are the pinned artifact."""
    from summicron.render._artifacts import default_tensor_path

    for lens, pin in (("35", TENSOR_MD5), ("50", TENSOR_MD5_50)):
        tp = default_tensor_path(lens)
        got = _md5(tp)
        assert got == pin, f"lens {lens} tensor {tp} md5 {got} != pin {pin}"


def test_layered_gather_flag_removed():
    """P1.10.4 C5 — --layered-gather flag must be GONE from argparse;
    --target-R / --target-D must remain absent.

    CLI-1a C3: the flags now live in ``_render_args.add_render_arguments``
    (shared by both entrypoints), so introspect the built parsers plus the
    sources of both modules instead of grepping convolve.py alone.
    """
    import summicron.render._render_args as render_args
    import summicron.render.convolve as convolve
    from summicron.cli import build_parser as cli_build_parser

    for mod in (convolve, render_args):
        src = Path(mod.__file__).read_text(encoding="utf-8")
        assert '"--layered-gather"' not in src, (
            f"P1.10.4 C5 FAIL: --layered-gather flag still wired in {mod.__name__}"
        )
        assert "--target-R" not in src
        assert "--target-D" not in src
    cli_parser = cli_build_parser()
    for parser in (convolve.build_parser(), cli_parser):
        opts = {o for a in parser._actions for o in a.option_strings}
        assert "--layered-gather" not in opts
        assert "--no-layered-gather" not in opts
    cli_render = cli_parser.parse_args(["render", "--image", "x.heic"])
    assert not hasattr(cli_render, "layered_gather")


# --- SLOW test ---------------------------------------------------------------


_ENTRYPOINTS = {
    "convolve": [sys.executable, "-m", "summicron.render.convolve"],
    "isummicron": [str(Path(sys.executable).parent / "isummicron"), "render"],
}
_ON = ("--occlusion-compositing", "on")
# mode -> (extra argv, expected md5, canonical flags)
_MODES = {
    "off": ((), G1_MD5, CANONICAL_FLAGS),
    "on": (_ON, ON_MD5, CANONICAL_FLAGS),
    "lens50": (("--lens", "50"), LENS50_MD5, CANONICAL_FLAGS),
}
_LABEL = {"off": "G1", "on": "ON", "lens50": "LENS50"}
_CASES = [
    pytest.param(ep, mode, id=f"{ep}-{mode}")
    for mode in ("off", "on")
    for ep in _ENTRYPOINTS
] + [
    pytest.param("convolve", "lens50", id="convolve-lens50"),
]


def test_backcompat_canonical_render_matches_baseline(
    tmp_path, entrypoint="convolve", mode="off"
):
    """G1 SHOWSTOPPER — canonical render md5 must equal the pin exactly.

    argv == runs/vibe_bench/README.md primary command (both built from
    ``CANONICAL_FLAGS``); default wheel tensor (no ``--tensor``).
    CLI-1a B2.4: parametrized over both entrypoints
    (``python -m summicron.render.convolve`` and the ``isummicron render``
    console script) x {OFF -> ``G1_MD5``, ``--occlusion-compositing on`` ->
    ``ON_MD5``}.
    """
    if not CANONICAL_IMAGE.exists():
        pytest.skip(f"canonical image missing: {CANONICAL_IMAGE}")

    if entrypoint == "isummicron" and not Path(_ENTRYPOINTS["isummicron"][0]).exists():
        pytest.skip("isummicron console script not installed next to sys.executable")

    extra, expected, flags = _MODES[mode]
    out_path = tmp_path / f"IMG_6915_{entrypoint}_{mode}.jpg"
    cmd = [
        *_ENTRYPOINTS[entrypoint],
        "--image", G1_IMAGE,
        *flags,
        "--focus-xy", *(str(v) for v in G1_FOCUS_XY),
        "--output", str(out_path),
        *extra,
    ]
    # PYTHONPATH pins the checkout under test: the ``isummicron`` console
    # script of an editable install may resolve ``summicron`` elsewhere.
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    result = subprocess.run(
        cmd, capture_output=True, text=True, cwd=str(ROOT), timeout=900,
        env=env,
    )
    if result.returncode != 0:
        pytest.fail(
            f"canonical render failed (rc={result.returncode}):\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}\n"
            f"{_env_diagnostics()}"
        )
    assert out_path.exists(), f"render produced no output at {out_path}"
    got = _md5(out_path)
    assert got == expected, (
        f"{_LABEL[mode]} FAIL via {entrypoint}: render md5 "
        f"{got} != {expected}\n"
        f"argv: {' '.join(cmd)}\n{_env_diagnostics(out_path)}"
    )


# Plain callable (defaults = convolve/OFF, the pre-CLI-1a G1 form) so the
# HYG-0 wrong-polarity probes in tests/adversarial/test_hyg0_adversarial.py
# can keep invoking it directly with ``(tmp_path)``; collection goes through
# the parametrized wrapper below.
test_backcompat_canonical_render_matches_baseline.__test__ = False


@pytest.mark.slow
@pytest.mark.parametrize("entrypoint,mode", _CASES)
def test_backcompat_canonical_render_md5(tmp_path, entrypoint, mode):
    """CLI-1a B2.4 — {convolve, isummicron} x {OFF, ON}, plus --lens 50."""
    test_backcompat_canonical_render_matches_baseline(tmp_path, entrypoint, mode)
