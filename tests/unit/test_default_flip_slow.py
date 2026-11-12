"""Run 6 default flip, SLOW: T-def bitwise equality and the non-LiDAR smoke (G6)."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from summicron.render.api import RenderOptions, render_image

ROOT = Path(__file__).resolve().parents[2]
SCENES = (
    ("iphoneImage/IMG_6915.HEIC", (0.40, 0.45)),
    ("iphoneImage/Demo_input.HEIC", (0.40, 0.39)),
)
pytestmark = pytest.mark.slow


@pytest.mark.parametrize("rel,fxy", SCENES, ids=["IMG_6915", "Demo_input"])
def test_default_render_equals_explicit_new_look(rel, fxy):
    """T-def / G3: ``auto`` resolves to exactly the explicit approved look."""
    img = ROOT / rel
    if not img.exists():
        pytest.skip(f"missing {img}")
    a = render_image(RenderOptions.canonical_bench(
        img, fxy, return_linear=True, save=False))
    b = render_image(RenderOptions.canonical_bench(
        img, fxy, return_linear=True, save=False,
        depth_guided="on", depth_mapping="diopter", tonemap="input-lut"))
    assert a.diagnostics.resolved["tonemap"] == "input-lut"
    assert a.diagnostics.resolved["depth_mapping"] == "diopter"
    assert np.array_equal(a.output_u8, b.output_u8)
    assert np.array_equal(a.output_linear, b.output_linear)


def _synthetic_inputs(tmp_path: Path) -> dict[str, Path]:
    import pillow_heif
    from PIL import Image

    from summicron.render.convolve import downres_array, load_image

    src = ROOT / "iphoneImage" / "Demo_input.HEIC"
    if not src.exists():
        pytest.skip(f"missing {src}")
    arr = downres_array(load_image(src), 512)
    u8 = np.clip(np.asarray(arr) * (255.0 if arr.max() <= 1.5 else 1.0), 0, 255).astype(np.uint8)
    im = Image.fromarray(u8, "RGB")
    out = {}
    out["png"] = tmp_path / "s.png"
    im.save(out["png"])
    out["jpg"] = tmp_path / "s.jpg"
    im.save(out["jpg"], quality=95)
    pillow_heif.register_heif_opener()
    out["heic"] = tmp_path / "s.heic"
    im.save(out["heic"], format="HEIF")
    return out


def _cli(args: list[str]) -> subprocess.CompletedProcess:
    env = dict(os.environ, HF_HUB_OFFLINE="1")
    return subprocess.run(
        [sys.executable, "-m", "summicron.render.convolve", *args],
        capture_output=True, text=True, cwd=str(ROOT), env=env, timeout=900,
    )


def test_non_lidar_defaults_render_and_equal_explicit_old_path(tmp_path):
    """G6: PNG / JPEG / depth-less HEIC render by default; the fallback is
    exactly the explicit monocular + linear + ci render."""
    try:
        from summicron.depth.estimate import load_depth_model  # noqa: F401
        os.environ["HF_HUB_OFFLINE"] = "1"
        load_depth_model(device="mps")
    except Exception as exc:  # model not cached / [depth] extra missing
        pytest.skip(f"DA2 not available offline: {exc!r}")
    inputs = _synthetic_inputs(tmp_path)
    explicit = ["--depth-source", "monocular", "--depth-guided", "off",
                "--depth-mapping", "linear", "--tonemap", "ci"]
    for kind, path in inputs.items():
        for hdr in ([], ["--hdr-source", "macos"]):
            tag = f"{kind}_{'hdr' if hdr else 'sdr'}"
            o1, o2 = tmp_path / f"{tag}_default.png", tmp_path / f"{tag}_explicit.png"
            r1 = _cli(["--image", str(path), "--downres", "512", *hdr,
                       "--output", str(o1)])
            assert r1.returncode == 0, (tag, r1.stderr[-800:])
            assert o1.exists()
            assert "depth_source auto -> monocular" in r1.stdout, tag
            # A Display-P3 SDR file loaded with --hdr-source macos can carry
            # headroom > 1.001 (spec R2.9): auto then resolves to input-lut, and
            # the explicit equivalent is that tonemap, not ci.
            ex = list(explicit)
            if "tonemap=input-lut" in r1.stdout:
                ex[ex.index("ci")] = "input-lut"
            r2 = _cli(["--image", str(path), "--downres", "512", *hdr, *ex,
                       "--output", str(o2)])
            assert r2.returncode == 0, (tag, r2.stderr[-800:])
            assert o1.read_bytes() == o2.read_bytes(), tag
