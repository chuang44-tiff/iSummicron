"""Resolution of the ``auto`` render modes (Run 6 default flip, spec 1.2).

Pure and stdlib-only: ``resolve_auto`` maps (options, facts about the input)
to concrete modes plus the one-line notes the pipeline emits. Explicit
values are never overridden; ``auto`` is only ever the default.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ResolvedModes:
    hdr_source: str
    depth_source: str
    depth_guided: str
    depth_mapping: str
    tonemap: str
    notes: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    any_auto: bool = False

    def resolved_line(self) -> str:
        return (
            f"Resolved: hdr_source={self.hdr_source} "
            f"depth_source={self.depth_source} "
            f"depth_guided={self.depth_guided} "
            f"depth_mapping={self.depth_mapping} tonemap={self.tonemap}"
        )


@dataclass(frozen=True)
class ResolvedFamily:
    kernel_family: str
    lens_aperture_mm: float | None
    bright_stamp: str
    bright_stamp_gain: float | None


def resolve_family(opts) -> ResolvedFamily:
    """Resolve the S22 S1 family / stamp flags (pure; spec 3.1).

    Aperture: ``None`` for codev, 17.5 under synthetic unless explicit. Stamp:
    ``auto`` -> ``l1`` (synthetic) / ``peak`` (codev). Gain: ``None`` under a
    ``peak`` stamp, else the explicit value or 2.0 (``is None`` idiom, not
    truthiness).
    """
    from summicron.render._family import (
        DEFAULT_APERTURE_MM, DEFAULT_BRIGHT_STAMP_GAIN,
    )

    synth = opts.kernel_family == "synthetic"
    aperture = (
        (DEFAULT_APERTURE_MM if opts.lens_aperture_mm is None else opts.lens_aperture_mm)
        if synth else None
    )
    stamp = opts.bright_stamp
    if stamp == "auto":
        stamp = "l1" if synth else "peak"
    if stamp == "peak":
        gain = None
    else:
        gain = (
            DEFAULT_BRIGHT_STAMP_GAIN if opts.bright_stamp_gain is None
            else opts.bright_stamp_gain
        )
    return ResolvedFamily(
        kernel_family=opts.kernel_family, lens_aperture_mm=aperture,
        bright_stamp=stamp, bright_stamp_gain=gain,
    )


MONOCULAR_NOTE = (
    "note: depth_source auto -> monocular (no LiDAR depth in input); "
    "relative depth, pass --depth-scale N for stronger blur"
)


def resolve_hdr_source(
    requested: str, *, loader_available: bool, has_hdr_data: bool
) -> tuple[str, str | None]:
    """Resolve ``--hdr-source`` (Rev 3). Returns ``(value, note_or_None)``.

    Explicit ``macos`` / ``none`` are returned unchanged (``macos`` stays
    strict: the loader errors as before if unavailable). ``auto`` picks
    ``macos`` iff the loader is available AND the photo carries HDR data;
    otherwise ``none`` plus one fallback note.
    """
    if requested != "auto":
        return requested, None
    if loader_available and has_hdr_data:
        return "macos", None
    reason = (
        "HDR loader unavailable (needs macOS + pyobjc-framework-Quartz)"
        if not loader_available else "no HDR gain map in input"
    )
    return "none", f"note: hdr_source auto -> none ({reason})"


def resolve_auto(
    opts, *, has_lidar: bool, hdr_headroom: bool,
    hdr_source: str | None = None, nonfinite: bool = False,
) -> ResolvedModes:
    """Resolve every ``auto`` in ``opts``.

    ``has_lidar``: the LiDAR extraction returned an array.
    ``hdr_headroom``: the loaded (post-downres) buffer has a finite
    ``max > 1.001``.
    ``hdr_source``: the already-resolved HDR source (the pipeline passes it);
    defaults to ``opts.hdr_source`` (an unresolved ``auto`` counts as none).
    ``nonfinite``: the HDR buffer has NaN/Inf pixels (tonemap auto -> ci).
    Raises ``ValueError`` for explicit requests the input cannot satisfy.
    """
    notes: list[str] = []
    warns: list[str] = []
    hdr = hdr_source if hdr_source is not None else opts.hdr_source
    if hdr == "auto":
        hdr = "none"
    src_in, gd_in, map_in, tm_in = (
        opts.depth_source, opts.depth_guided, opts.depth_mapping, opts.tonemap,
    )
    any_auto = "auto" in (src_in, gd_in, map_in, tm_in, opts.hdr_source)

    # ---- depth source
    if src_in == "lidar":
        if not has_lidar:
            raise ValueError(
                "--depth-source=lidar but image has no embedded depth aux."
            )
        src = "lidar"
    elif src_in == "monocular":
        src = "monocular"
    else:
        src = "lidar" if has_lidar else "monocular"
        if src == "monocular":
            notes.append(MONOCULAR_NOTE)

    # ---- guided / mapping
    if gd_in == "auto":
        gd = "on" if src == "lidar" else "off"
    else:
        gd = gd_in
    if map_in == "auto":
        mp = "diopter" if src == "lidar" else "linear"
    else:
        mp = map_in
    if src != "lidar" and src_in == "auto":
        if map_in == "diopter":
            raise ValueError(
                "--depth-mapping diopter requires LiDAR depth; the image has "
                "no depth aux image (pass --depth-source monocular only with "
                "--depth-mapping linear)"
            )
        if gd_in == "on":
            raise ValueError(
                "--depth-guided on requires LiDAR depth; the image has no "
                "depth aux image"
            )
    if (
        map_in == "auto" and mp == "linear"
        and not isinstance(opts.depth_gain, str)
    ):
        w = (
            f"--depth-gain {opts.depth_gain:g} is ignored: depth mapping "
            "resolved to linear (no LiDAR depth)"
        )
        warns.append(w)

    if getattr(opts, "kernel_family", "codev") == "synthetic" and mp != "diopter":
        raise ValueError(
            "--kernel-family synthetic requires LiDAR depth "
            "(depth mapping resolved to linear)"
        )

    # ---- tonemap
    if tm_in == "auto":
        if hdr == "none":
            tm = "ci"
        elif opts.shadow_match_mode == "on":
            tm = "ci"
            notes.append(
                "note: tonemap auto -> ci (--shadow-match-mode on uses the "
                "Core Image path)"
            )
        elif nonfinite:
            tm = "ci"
            notes.append(
                "note: tonemap auto -> ci (non-finite pixels in the HDR buffer)"
            )
        elif not hdr_headroom:
            tm = "ci"
            notes.append("note: tonemap auto -> ci (no HDR headroom in input)")
        else:
            tm = "input-lut"
    else:
        tm = tm_in
    if opts.bright_pass_thresh == "auto" and tm == "ci":
        notes.append(
            "note: bright_pass_thresh auto -> 1.0 (tonemap resolved to ci)"
        )

    return ResolvedModes(
        hdr_source=hdr, depth_source=src, depth_guided=gd, depth_mapping=mp, tonemap=tm,
        notes=tuple(notes), warnings=tuple(warns), any_auto=any_auto,
    )
