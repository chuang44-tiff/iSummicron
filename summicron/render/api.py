"""GUI/CLI-shared backend contract for the v0 render pipeline.

P1.audit Item 3: stable typed surface. The CLI shim, the GUI, and any
in-process caller construct a :class:`RenderOptions`, hand it to
:func:`render_image`, and read a fully-populated :class:`RenderResult`.

Cooperative cancellation is supported via ``cancel: threading.Event``;
checked at every slab loop iteration in the renderer (see
``convolve.py`` slab-iter checkpoints). Progress messages are emitted
through ``on_progress``; when ``None``, the underlying renderer routes
through ``print()`` for byte-identical CLI behaviour.

This module is dependency-light by design — it MUST be importable from
``summicron.cli`` without dragging in scipy/PIL/HEIC/OS-specific deps.
The actual render is invoked lazily.
"""
from __future__ import annotations

import argparse
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal

import numpy as np

# ---------------------------------------------------------------------------
# Cancellation
# ---------------------------------------------------------------------------


class CancelledError(Exception):
    """Raised by :func:`render_image` when its ``cancel`` event is set
    between expensive stages.

    The renderer checks ``cancel.is_set()`` at each slab-loop iteration
    and at the radial-bin loop inside ``_scatter_one_slab_fixed_d_basis``.
    Callers (GUI sliders, batch tools) own the ``threading.Event``; the
    renderer never mutates it.
    """


# ---------------------------------------------------------------------------
# Typed contract — frozen dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RenderOptions:
    """Immutable render request. Constructed by the CLI shim or the GUI.

    The shapes mirror the existing ``convolve.py`` argparse surface; see
    that module's parser for per-flag documentation. ``tensor_path=None``
    is reserved for the Item 7 ``default_tensor_path()`` resolver shipped
    by Coder P0.
    """

    image_path: Path
    tensor_path: Path | None = None  # None → default_tensor_path() (Item 7)
    out_path: Path | None = None
    out_dir: Path | None = None
    depth_source: Literal["lidar", "monocular"] = "lidar"
    focus_xy: tuple[float, float] | None = None
    focus_target_m: float | None = None
    depth_scale: float = 1.0
    dof_m: float = 0.0
    downres: int | None = None
    psf_norm: Literal["l1", "peak"] = "l1"
    hdr_source: Literal["none", "macos"] = "none"
    bright_pass_thresh: float = 1.0
    bright_pass_region_size: int = 31
    bright_pass_floor: float = 0.10
    bright_pass_norm_pct: float = 99.0
    shadow_match_mode: Literal["off", "on"] = "off"
    shadow_band_hi: float = 0.70
    shadow_match_max_gain: float = 4.0
    shadow_match_strength: float = 1.0
    slab_batch_size: int | None = None  # R-perf-1 I1; None → module default (4)
    return_linear: bool = False  # opt-in for LDR-mode linear array

    @classmethod
    def from_argparse(cls, args: argparse.Namespace) -> "RenderOptions":
        """Build a :class:`RenderOptions` from an argparse namespace
        as produced by ``convolve.main``'s parser.

        Keys that may be absent on the namespace (test/GUI surfaces that
        construct a partial namespace) fall back to dataclass defaults.
        """
        def _get(name: str, default):
            return getattr(args, name, default)

        image = _get("image", None)
        tensor = _get("tensor", None)
        out_path = _get("output", None)
        out_dir = _get("out_dir", None)
        focus_xy = _get("focus_xy", None)
        if focus_xy is not None:
            focus_xy = (float(focus_xy[0]), float(focus_xy[1]))
        return cls(
            image_path=Path(image) if image is not None else Path(""),
            tensor_path=Path(tensor) if tensor is not None else None,
            out_path=Path(out_path) if out_path is not None else None,
            out_dir=Path(out_dir) if out_dir is not None else None,
            depth_source=_get("depth_source", "lidar"),
            focus_xy=focus_xy,
            focus_target_m=_get("focus_target_m", None),
            depth_scale=float(_get("depth_scale", 1.0)),
            dof_m=float(_get("dof_m", 0.0)),
            downres=_get("downres", None),
            psf_norm=_get("psf_norm", "l1"),
            hdr_source=_get("hdr_source", "none"),
            bright_pass_thresh=float(_get("bright_pass_thresh", 1.0)),
            bright_pass_region_size=int(_get("bright_pass_region_size", 31)),
            bright_pass_floor=float(_get("bright_pass_floor", 0.10)),
            bright_pass_norm_pct=float(_get("bright_pass_norm_pct", 99.0)),
            shadow_match_mode=_get("shadow_match_mode", "off"),
            shadow_band_hi=float(_get("shadow_band_hi", 0.70)),
            shadow_match_max_gain=float(_get("shadow_match_max_gain", 4.0)),
            shadow_match_strength=float(_get("shadow_match_strength", 1.0)),
            slab_batch_size=(
                None
                if _get("slab_batch_size", None) is None
                else int(_get("slab_batch_size", None))
            ),
            return_linear=bool(_get("return_linear", False)),
        )


@dataclass(frozen=True)
class RenderDiagnostics:
    """Optional render-quality diagnostics surfaced for GUI display.

    ``warnings`` is a tuple of human-readable strings collected during the
    render (energy gates, missing aux, etc.); empty tuple means clean run.
    """

    center_ssim: float | None = None
    bright_pixels_pct: float = 0.0
    alpha_mean: float = 0.0
    shadow_gains_rgb: tuple[float, float, float] | None = None
    warnings: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class RenderResult:
    """Render output container.

    ``output_u8`` is ALWAYS populated — including the HDR path, where the
    renderer tonemaps the linear buffer to sRGB uint8 in-band. This is
    contract: a GUI binding can write ``output_u8`` straight to a QImage
    without branching on mode.

    ``output_linear`` is populated when ``mode == "hdr"`` OR when the
    caller requested ``return_linear=True`` on the LDR path.
    """

    output_u8: np.ndarray            # (H, W, 3) uint8 — ALWAYS populated
    output_linear: np.ndarray | None # (H, W, 3) float32 — HDR or return_linear
    mode: Literal["ldr", "hdr"]
    elapsed_s: float
    diagnostics: RenderDiagnostics
    output_path: Path | None = None  # path written to disk if any


# ---------------------------------------------------------------------------
# Backend entrypoint — single function consumed by CLI + GUI
# ---------------------------------------------------------------------------


def render_image(
    opts: RenderOptions,
    *,
    on_progress: Callable[[str], None] | None = None,
    cancel: threading.Event | None = None,
) -> RenderResult:
    """Run the v0 render pipeline against ``opts``.

    Parameters
    ----------
    opts : RenderOptions
        Immutable request (image path, tensor, focus, HDR mode, etc.).
    on_progress : Callable[[str], None] | None
        Per-stage status emitter. When ``None`` (CLI default), the
        renderer routes messages through ``print()`` so stdout stays
        byte-identical to the pre-refactor CLI. GUIs pass a closure that
        forwards each line into a status pane / progress bar.
    cancel : threading.Event | None
        Cooperative cancellation token. The renderer polls ``is_set()``
        at each slab-loop iteration and at the radial-bin loop inside
        ``_scatter_one_slab_fixed_d_basis``; on set, it raises
        :class:`CancelledError` (no partial output is written to disk).

    Returns
    -------
    RenderResult
        Always populated ``output_u8`` (uint8 sRGB); optional
        ``output_linear`` (float32 linear) on HDR or ``return_linear``;
        timings + diagnostics; ``output_path`` if the caller requested
        on-disk save (``opts.out_path`` or ``opts.out_dir``).

    Raises
    ------
    CancelledError
        If ``cancel`` was set while the render was in progress.
    ValueError, RuntimeError, FileNotFoundError, TensorValidationError
        Bad inputs / metadata / energy gate failures. The CLI shim
        catches these and exits with code 2.
    """
    # Local import: avoids dragging scipy / PIL into callers that only
    # need the dataclass surface (e.g. the cli.py import-time path).
    from summicron.render import convolve as _conv

    t0 = time.perf_counter()
    result = _conv._run_render_pipeline(
        opts, on_progress=on_progress, cancel=cancel
    )
    elapsed = time.perf_counter() - t0
    # Re-pack with measured wall time. ``_run_render_pipeline`` returns a
    # provisional RenderResult with elapsed_s=0.0; the public API stamps
    # the real wall time here so the helper stays free of timing noise.
    return RenderResult(
        output_u8=result.output_u8,
        output_linear=result.output_linear,
        mode=result.mode,
        elapsed_s=elapsed,
        diagnostics=result.diagnostics,
        output_path=result.output_path,
    )
