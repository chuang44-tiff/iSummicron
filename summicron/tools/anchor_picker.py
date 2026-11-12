"""GUI anchor picker (P1.10) — read-only UX wrapper around the v0 renderer.

Run: ``python -m summicron.tools.anchor_picker --image PATH [--tensor PATH]
[--depth-source lidar|monocular] [--no-show]``. Click anywhere to print a
depth report and the canonical bench command with ``--focus-xy`` substituted.
Last click wins; on window close the most recent command is reprinted.

G2 (canonical bench reproducible) is MANUAL: feed the emitted command back
through ``summicron.render.convolve`` and visually check the resulting JPG
against the canonical-bench acceptance band. No automated test exists for it.

Matplotlib imports are lazy (inside :func:`main`) so the compute helpers stay
headless-safe (G6).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

from summicron.render._bench import CANONICAL_FLAGS

# Built from summicron.render._bench.CANONICAL_FLAGS (HYG-0: single source
# of truth shared with the G1 test and runs/vibe_bench/README.md). No
# --tensor: the renderer defaults to the wheel-shipped canonical tensor.
CANONICAL_TEMPLATE = (
    "conda run -n isummicron python -m summicron.render.convolve \\\n"
    "  --image {IMAGE} \\\n"
    "  " + " ".join(CANONICAL_FLAGS) + " \\\n"
    "  --focus-xy {X:.4f} {Y:.4f} \\\n"
    "  --output runs/vibe_bench/IMG_6651_bench.jpg"
)

_YELLOW, _RED, _RESET = "\033[33m", "\033[31m", "\033[0m"


# --- pure compute helpers (no matplotlib) -----------------------------------

def _normalize_click(x_event: float, y_event: float, w: int, h: int) -> tuple[float, float]:
    """Identity map: matplotlib event coords -> normalized (x_norm, y_norm)."""
    return float(x_event) / float(w), float(y_event) / float(h)


def emit_canonical_command(image_path: str | Path, x_norm: float, y_norm: float) -> str:
    """Format the canonical bench command. Idempotent (G5)."""
    return CANONICAL_TEMPLATE.format(IMAGE=str(image_path), X=float(x_norm), Y=float(y_norm))


def probe_depth(depth: np.ndarray, px: int, py: int) -> dict[str, Any]:
    """Return ``{center_med, iqr, annulus_med, coplanar_warn}`` at click ``(px, py)``.

    21x21 disk for center+IQR; ring ``30 <= r <= 80`` for annulus. Co-planar warn:
    ``|center-annulus| < 0.05`` AND ``0 < center < 1.0`` (zero-depth excluded so
    the LiDAR-missing flat-zeros sentinel does not trip it).
    """
    h, w = depth.shape[:2]
    px, py = int(px), int(py)
    y0, y1 = max(py - 10, 0), min(py + 11, h)
    x0, x1 = max(px - 10, 0), min(px + 11, w)
    disk = depth[y0:y1, x0:x1]
    if disk.size == 0:
        return {"center_med": 0.0, "iqr": 0.0, "annulus_med": 0.0, "coplanar_warn": False}
    center_med = float(np.median(disk))
    iqr = float(np.percentile(disk, 75) - np.percentile(disk, 25))
    yy, xx = np.ogrid[:h, :w]
    rr2 = (yy - py) ** 2 + (xx - px) ** 2
    mask = (rr2 >= 30 * 30) & (rr2 <= 80 * 80)
    annulus_med = float(np.median(depth[mask])) if mask.any() else 0.0
    coplanar_warn = (
        abs(center_med - annulus_med) < 0.05 and 0.0 < center_med < 1.0
    )
    return {"center_med": center_med, "iqr": iqr,
            "annulus_med": annulus_med, "coplanar_warn": coplanar_warn}


# --- I/O wrappers — fail loud at boundaries ---------------------------------

def _load_depth(image_path: Path, depth_source: str, target_hw: tuple[int, int],
                arr: np.ndarray) -> np.ndarray:
    if depth_source == "lidar":
        from summicron.depth.estimate import extract_lidar_depth
        depth = extract_lidar_depth(image_path, target_hw=target_hw)
        if depth is None:
            print(f"{_YELLOW}WARN: no LiDAR depth aux in {image_path.name}; "
                  f"using flat zeros — co-planar check disabled.{_RESET}")
            depth = np.zeros(target_hw, dtype=np.float32)
        return depth
    if depth_source == "monocular":
        from summicron.depth.estimate import estimate_depth, load_depth_model
        bundle = load_depth_model()
        return estimate_depth(arr, bundle).astype(np.float32)  # FAR_M=1.0 per convolve.py
    raise ValueError(f"--depth-source must be 'lidar' or 'monocular', got {depth_source!r}")


def _read_exif_orientation(image_path: Path) -> int:
    """Best-effort read of EXIF Orientation. Returns 1 on any failure."""
    try:
        from PIL import ExifTags, Image
        with Image.open(str(image_path)) as im:
            exif = im.getexif() or {}
        for k, v in ExifTags.TAGS.items():
            if v == "Orientation":
                return int(exif.get(k, 1))
    except Exception:
        return 1
    return 1


# --- entry point ------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m summicron.tools.anchor_picker",
        description="Click-to-anchor GUI for canonical bench renders.",
    )
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--tensor", type=Path, default=None,
                        help="Optional tensor path (informational only).")
    parser.add_argument("--depth-source", choices=("lidar", "monocular"), default="lidar")
    parser.add_argument("--no-show", action="store_true", help="Skip plt.show() — for tests.")
    args = parser.parse_args(argv)

    if not args.image.exists():
        print(f"ERROR: image not found: {args.image}", file=sys.stderr)
        return 2

    from summicron.render.convolve import load_image
    arr = load_image(args.image)
    h, w = arr.shape[:2]
    depth = _load_depth(args.image, args.depth_source, (h, w), arr)
    orientation = _read_exif_orientation(args.image)

    is_heic = args.image.suffix.lower() in (".heic", ".heif")
    if not is_heic:
        print(f"{_YELLOW}WARN: non-HEIC input — PIL does not auto-transpose. "
              f"Coords assume EXIF Orientation=1.{_RESET}")

    if args.no_show:
        # Smoke path for tests / CI: print a placeholder command and exit cleanly.
        print(emit_canonical_command(args.image, 0.5, 0.75))
        return 0

    import matplotlib
    matplotlib.use("TkAgg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    last_command = {"value": emit_canonical_command(args.image, 0.5, 0.75)}
    overlay_artists: list[Any] = []

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(arr, aspect="equal")
    banner = (f"arr.shape=(H={h}, W={w})  EXIF Orientation={orientation}"
              + ("  [non-HEIC: orientation assumed 1]" if not is_heic else ""))
    ax.set_title(banner, fontsize=10)
    ax.set_xlabel("col (x_event)")
    ax.set_ylabel("row (y_event)")

    def on_click(event):
        if event.inaxes is not ax or event.xdata is None or event.ydata is None:
            return
        for art in overlay_artists:
            try:
                art.remove()
            except (NotImplementedError, ValueError):
                pass
        overlay_artists.clear()

        x_ev, y_ev = float(event.xdata), float(event.ydata)
        x_norm, y_norm = _normalize_click(x_ev, y_ev, w, h)
        report = probe_depth(depth, int(round(x_ev)), int(round(y_ev)))

        plus = ax.plot([x_ev], [y_ev], marker="+", color="red", markersize=14, mew=2)[0]
        overlay_artists.append(plus)
        for radius, ls in ((10.5, "-"), (30.0, "--"), (80.0, "--")):
            patch = Circle((x_ev, y_ev), radius=radius, fill=False,
                           edgecolor="white", linewidth=1.0, linestyle=ls)
            ax.add_patch(patch)
            overlay_artists.append(patch)
        fig.canvas.draw_idle()

        print(f"tap=({x_norm:.4f},{y_norm:.4f}) "
              f"depth_center={report['center_med']:.3f}m "
              f"IQR={report['iqr']:.3f}m annulus={report['annulus_med']:.3f}m")
        if report["coplanar_warn"]:
            print(f"{_RED}WARN: co-planar surface (center≈annulus); "
                  f"subject may not isolate{_RESET}")
        cmd = emit_canonical_command(args.image, x_norm, y_norm)
        last_command["value"] = cmd
        print(cmd)

    fig.canvas.mpl_connect("button_press_event", on_click)

    try:
        plt.show()
    except KeyboardInterrupt:
        print(last_command["value"])
        return 130

    print(last_command["value"])  # G7: reprint last command on close
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
