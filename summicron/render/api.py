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
import math
import sys
import threading
import time
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Callable, Literal, Mapping

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
    slab_batch_size: int | None = None  # R-perf-1 I1; None → module default (1)
    occlusion_compositing: Literal["off", "on"] = "off"  # R-occlusion-layering
    return_linear: bool = False  # opt-in for LDR-mode linear array
    # API-only (no CLI flag): False skips every disk write (GUI/probe use);
    # RenderResult.output_path is then None.
    save: bool = True

    def __post_init__(self) -> None:
        """CLI-1b C4: invariants formerly enforced only by ``_validate_args``.

        Messages are byte-identical to the historical ``parser.error``
        strings (CLI flag spelling retained). Raises ``ValueError``.
        Platform is deliberately NOT checked here (see
        :func:`_check_platform`); no filesystem checks; ``image_path`` is
        not validated.
        """
        if self.focus_xy is not None:
            x, y = self.focus_xy
            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                raise ValueError("--focus-xy values must be in [0, 1]")
        if self.focus_target_m is not None and not (
            math.isfinite(self.focus_target_m) and self.focus_target_m > 0
        ):
            raise ValueError("--focus-target-m must be > 0")
        # P1.13 K11 range gates
        if not (0.0 <= self.shadow_band_hi <= 1.0):
            raise ValueError("--shadow-band-hi must be in [0.0, 1.0]")
        if not (1.0 <= self.shadow_match_max_gain <= 16.0):
            raise ValueError("--shadow-match-max-gain must be in [1.0, 16.0]")
        if not (0.0 <= self.shadow_match_strength <= 4.0):
            raise ValueError("--shadow-match-strength must be in [0.0, 4.0]")
        # R-perf-1 I1
        if self.slab_batch_size is not None:
            try:
                _sb = int(self.slab_batch_size)
            except (OverflowError, ValueError, TypeError):
                raise ValueError(
                    f"--slab-batch-size must be a finite integer >= 1, "
                    f"got {self.slab_batch_size!r}"
                ) from None
            if _sb < 1:
                raise ValueError("--slab-batch-size must be >= 1")
        # CLI-1b fix round: previously-unchecked numeric params.
        if not (math.isfinite(self.depth_scale) and self.depth_scale > 0):
            raise ValueError(
                f"--depth-scale must be finite and > 0, got {self.depth_scale!r}"
            )
        if self.downres is not None and self.downres < 1:
            raise ValueError(f"--downres must be >= 1, got {self.downres!r}")
        if not math.isfinite(self.bright_pass_thresh):
            raise ValueError(
                f"--bright-pass-thresh must be finite, got "
                f"{self.bright_pass_thresh!r}"
            )
        if not (0 <= self.bright_pass_region_size <= 255):
            raise ValueError(
                f"--bright-pass-region-size must be in [0, 255], got "
                f"{self.bright_pass_region_size!r}"
            )
        if not (0.0 <= self.bright_pass_floor <= 1.0):
            raise ValueError(
                f"--bright-pass-floor must be in [0.0, 1.0], got "
                f"{self.bright_pass_floor!r}"
            )
        if not (50.0 <= self.bright_pass_norm_pct <= 100.0):
            raise ValueError(
                f"--bright-pass-norm-pct must be in [50, 100], got "
                f"{self.bright_pass_norm_pct!r}"
            )
        # R-occlusion-layering
        if self.occlusion_compositing not in ("off", "on"):
            raise ValueError("--occlusion-compositing must be 'off' or 'on'")
        for _name, _flag, _allowed in _ENUM_FIELDS:
            if getattr(self, _name) not in _allowed:
                raise ValueError(
                    f"{_flag} must be one of {', '.join(map(repr, _allowed))}"
                )
        # F3: early-error guard
        if self.shadow_match_mode == "on" and self.hdr_source == "none":
            raise ValueError(
                "--shadow-match-mode on requires --hdr-source != none "
                "(needs HDR linear buffer + CI preview pass)."
            )

    @classmethod
    def canonical_bench(
        cls,
        image_path: Path | str,
        focus_xy: tuple[float, float],
        **overrides,
    ) -> "RenderOptions":
        """Options for the canonical bench render (``_bench.CANONICAL_FLAGS``).

        Parses the bench argv through the real shared parser
        (``_render_args.add_render_arguments``; argparse+pathlib only, so
        the ``summicron.cli`` import fence holds) so the result is identical
        to what the CLI builds. ``overrides`` are applied last via
        ``dataclasses.replace`` (re-validated by ``__post_init__``), e.g.
        ``out_path=...``, ``occlusion_compositing="on"``,
        ``return_linear=True``, ``save=False``.
        """
        from summicron.render import _render_args
        from summicron.render._bench import CANONICAL_FLAGS

        parser = argparse.ArgumentParser()
        _render_args.add_render_arguments(parser)
        ns = parser.parse_args([
            "--image", str(image_path),
            "--focus-xy", str(focus_xy[0]), str(focus_xy[1]),
            *CANONICAL_FLAGS,
        ])
        opts = cls.from_argparse(ns)
        return replace(opts, **overrides) if overrides else opts

    @classmethod
    def from_argparse(cls, args: argparse.Namespace) -> "RenderOptions":
        """Build a :class:`RenderOptions` from an argparse namespace as
        produced by ``summicron.render._render_args.add_render_arguments``
        (shared by ``convolve.build_parser`` and ``isummicron render``).

        CLI-1a C3: a field loop instead of per-field literals. Namespace
        attrs are looked up by field name (or its argparse alias in
        :data:`_ARGPARSE_ALIAS`); attrs absent from the namespace (partial
        namespaces built by tests/GUI) fall back to the dataclass default.
        argparse already types the values, so only the path fields
        (``str`` -> ``Path``; ``None`` stays ``None``) and ``focus_xy``
        (list -> hashable ``tuple[float, float]``) are coerced.
        ``return_linear`` is API-only (no CLI flag) but is honoured when
        present on the namespace.
        """
        kwargs: dict = {}
        for f in fields(cls):
            name = _ARGPARSE_ALIAS.get(f.name, f.name)
            if not hasattr(args, name):
                continue
            value = getattr(args, name)
            if f.name in _PATH_FIELDS:
                value = Path(value) if value is not None else None
            elif f.name == "focus_xy" and value is not None:
                value = (float(value[0]), float(value[1]))
            kwargs[f.name] = value
        # image_path is required on the dataclass; absent/None -> Path("")
        # (pre-CLI-1a contract for partial namespaces).
        if kwargs.get("image_path") is None:
            kwargs["image_path"] = Path("")
        return cls(**kwargs)


# (field, CLI flag, allowed) for the Literal fields not already covered above.
_ENUM_FIELDS = (
    ("depth_source", "--depth-source", ("lidar", "monocular")),
    ("psf_norm", "--psf-norm", ("l1", "peak")),
    ("hdr_source", "--hdr-source", ("none", "macos")),
    ("shadow_match_mode", "--shadow-match-mode", ("off", "on")),
)


def _check_platform(hdr_source: str) -> None:
    """macOS-HDR preflight (P1.audit Item 8). Raises ``ValueError``.

    Platform is an environment fact, not an option invariant, so it lives
    outside ``RenderOptions.__post_init__``. Reads ``sys.platform`` at call
    time (tests monkeypatch ``sys.platform``).
    """
    if hdr_source == "macos" and sys.platform != "darwin":
        raise ValueError(
            f"--hdr-source macos requires macOS; got {sys.platform}. "
            f"Use --hdr-source none on this platform."
        )


# argparse dest -> RenderOptions field name differs only for these.
_ARGPARSE_ALIAS = {
    "image_path": "image",
    "tensor_path": "tensor",
    "out_path": "output",
}
_PATH_FIELDS = frozenset({"image_path", "tensor_path", "out_path", "out_dir"})


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
    # Q-a: band metrics in the focus window (None when focus_xy is None or
    # skimage is missing). center_ssim == focus_fidelity["ssim_band"].
    # hash/compare disabled: the dict value would make the frozen dataclass
    # unhashable.
    focus_fidelity: Mapping[str, Any] | None = field(
        default=None, hash=False, compare=False
    )


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
    _eval_sink: dict | None = None,  # private: Q-a probe/slow-test hook
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
    _check_platform(opts.hdr_source)
    # Local import: avoids dragging scipy / PIL into callers that only
    # need the dataclass surface (e.g. the cli.py import-time path).
    from summicron.render import convolve as _conv

    t0 = time.perf_counter()
    result = _conv._run_render_pipeline(
        opts, on_progress=on_progress, cancel=cancel, _eval_sink=_eval_sink
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
