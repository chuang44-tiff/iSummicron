"""Unit tests for the GUI anchor picker (P1.10).

Covers gates G1, G3, G4, G5, G6. G2 is manual; G7 (clean exit) and G8 (ruff)
are process gates not testable here.
"""
from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from summicron.tools.anchor_picker import (
    CANONICAL_TEMPLATE,
    _normalize_click,
    emit_canonical_command,
    probe_depth,
)


# --- G5: idempotent emitter ------------------------------------------------

def test_command_emitter_idempotent():
    a = emit_canonical_command("iphoneImage/IMG_6651.HEIC", 0.5, 0.75)
    b = emit_canonical_command("iphoneImage/IMG_6651.HEIC", 0.5, 0.75)
    assert a == b and isinstance(a, str)


def test_command_emitter_substitutes_focus_xy():
    out = emit_canonical_command("iphoneImage/IMG_6651.HEIC", 0.5, 0.75)
    assert "--focus-xy 0.5000 0.7500" in out
    assert "--image iphoneImage/IMG_6651.HEIC" in out
    assert out.endswith("--output runs/vibe_bench/IMG_6651_bench.jpg")


def test_canonical_template_constant_keys():
    assert "{IMAGE}" in CANONICAL_TEMPLATE
    assert "{X:.4f}" in CANONICAL_TEMPLATE
    assert "{Y:.4f}" in CANONICAL_TEMPLATE


# --- G1: identity contract for the canonical pixel -------------------------

def test_focus_xy_identity_for_canonical_pixel():
    # IMG_6651.HEIC storage: W=4284, H=5712. Canonical anchor (0.5, 0.75).
    x_norm, y_norm = _normalize_click(2142.0, 4284.0, 4284, 5712)
    assert pytest.approx(x_norm, abs=1e-9) == 0.5
    assert pytest.approx(y_norm, abs=1e-9) == 0.75
    cmd = emit_canonical_command("iphoneImage/IMG_6651.HEIC", x_norm, y_norm)
    assert "--focus-xy 0.5000 0.7500" in cmd


# --- G3, G4: depth probe co-planar gating ----------------------------------

def _make_depth(center_val: float, ring_val: float, h: int = 256, w: int = 256,
                cx: int = 128, cy: int = 128) -> np.ndarray:
    depth = np.full((h, w), ring_val, dtype=np.float32)
    depth[cy - 10:cy + 11, cx - 10:cx + 11] = center_val
    return depth


def test_depth_probe_flat_046_warns_per_spec():
    """Flat 0.46m satisfies the co-planar rule per the LOCKED SPEC
    (|center-annulus|<0.05 AND 0<center<1.0), so the warn IS expected.
    The Coder task brief's inverse phrasing is a typo vs the spec; the spec wins.
    """
    depth = np.full((256, 256), 0.46, dtype=np.float32)
    out = probe_depth(depth, px=128, py=128)
    assert out["coplanar_warn"] is True
    assert pytest.approx(out["center_med"], abs=1e-6) == 0.46
    assert pytest.approx(out["annulus_med"], abs=1e-6) == 0.46


def test_depth_probe_flat_far_no_warn():
    depth = np.full((256, 256), 2.5, dtype=np.float32)
    out = probe_depth(depth, px=128, py=128)
    assert out["coplanar_warn"] is False


def test_depth_probe_flat_zero_no_warn_when_lidar_missing():
    depth = np.zeros((256, 256), dtype=np.float32)
    out = probe_depth(depth, px=128, py=128)
    assert out["coplanar_warn"] is False


def test_depth_probe_coplanar_warn():
    """G3: center 21px disk = 0.46m, annulus = 0.49m -> WARN."""
    depth = _make_depth(center_val=0.46, ring_val=0.49)
    out = probe_depth(depth, px=128, py=128)
    assert out["coplanar_warn"] is True
    assert pytest.approx(out["center_med"], abs=1e-6) == 0.46
    assert pytest.approx(out["annulus_med"], abs=1e-6) == 0.49


def test_depth_probe_subject_isolated_no_warn():
    """G4: center=0.46m, annulus=1.20m -> NO WARN."""
    depth = _make_depth(center_val=0.46, ring_val=1.20)
    out = probe_depth(depth, px=128, py=128)
    assert out["coplanar_warn"] is False


def test_depth_probe_iqr_nonzero():
    rng = np.random.default_rng(7)
    depth = rng.uniform(0.4, 0.6, size=(256, 256)).astype(np.float32)
    out = probe_depth(depth, px=128, py=128)
    assert out["iqr"] > 0.0


# --- G6: no module-import side effects -------------------------------------

def test_module_import_no_backend():
    for name in list(sys.modules):
        if name.startswith("matplotlib"):
            del sys.modules[name]
    sys.modules.pop("summicron.tools.anchor_picker", None)
    importlib.import_module("summicron.tools.anchor_picker")
    assert "matplotlib.pyplot" not in sys.modules
    assert "matplotlib" not in sys.modules


# --- Real-HEIC passthrough (skip if file or pillow_heif unavailable) -------

def test_load_image_passthrough_heic():
    pytest.importorskip("pillow_heif")
    heic = Path("iphoneImage/IMG_6651.HEIC")
    if not heic.exists():
        pytest.skip("IMG_6651.HEIC not present")
    from summicron.render.convolve import load_image
    arr = load_image(heic)
    assert arr.dtype == np.float32
    assert arr.ndim == 3 and arr.shape[2] == 3
    h, w = arr.shape[:2]
    x_norm, y_norm = _normalize_click(w * 0.5, h * 0.75, w, h)
    assert pytest.approx(x_norm, abs=1e-9) == 0.5
    assert pytest.approx(y_norm, abs=1e-9) == 0.75


# --- Smoke: --no-show CLI exits 0 and prints canonical command -------------

def test_no_show_smoke_exits_zero():
    heic = Path("iphoneImage/IMG_6651.HEIC")
    if not heic.exists():
        pytest.skip("IMG_6651.HEIC not present")
    proc = subprocess.run(
        [sys.executable, "-m", "summicron.tools.anchor_picker",
         "--image", str(heic), "--no-show"],
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert "--focus-xy" in proc.stdout
    assert "--image" in proc.stdout
