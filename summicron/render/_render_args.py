"""Single argparse source for the render CLI flags (CLI-1a C3).

``add_render_arguments(parser)`` installs every render flag on ``parser``.
It is shared by ``summicron.render.convolve.build_parser()`` (the
``python -m summicron.render.convolve`` entrypoint) and the ``render``
subparser of ``summicron.cli.build_parser()`` (the ``isummicron render``
console script), so the two surfaces cannot drift.

Stdlib-only by design (``argparse`` + ``pathlib``, cf. ``_bench.py``):
``summicron.cli`` imports this at parser-build time and must stay free of
the renderer's numpy/scipy/PIL deps (``cli._cmd_render`` imports them
lazily). Every default is a literal. Dests must stay in sync with the
fields of ``summicron.render.api.RenderOptions`` (pinned by
``tests/unit/test_render_argparse_parity.py``).
"""
from __future__ import annotations

import argparse


def _depth_gain_type(value: str):
    """argparse ``type=`` for ``--depth-gain``: ``'auto'`` or a float.

    Range/finite checks live in ``RenderOptions.__post_init__`` (single source).
    """
    if value == "auto":
        return "auto"
    try:
        return float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"invalid --depth-gain {value!r}: expected 'auto' or a number"
        ) from None


def _bright_thresh_type(value: str):
    """argparse ``type=`` for ``--bright-pass-thresh``: ``'auto'`` or a float.

    Range/finite checks live in ``RenderOptions.__post_init__`` (single source).
    """
    if value == "auto":
        return "auto"
    try:
        return float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"invalid --bright-pass-thresh {value!r}: expected 'auto' or a number"
        ) from None


def _stamp_gain_type(value: str):
    """argparse ``type=`` for ``--bright-stamp-gain`` (range check in
    ``RenderOptions.__post_init__``)."""
    try:
        return float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"invalid --bright-stamp-gain {value!r}: expected a number"
        ) from None


def add_render_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Install every render flag on ``parser`` and return it."""
    parser.add_argument("--image", required=True, help="Input image path (HEIC/JPEG/PNG)")
    parser.add_argument(
        "--tensor",
        default=None,
        help="custom tensor built with scripts/build_psf; not combinable with "
             "--lens. When omitted, the wheel-shipped tensor selected by --lens "
             "is used.",
    )
    parser.add_argument(
        "--lens",
        choices=["35", "50"],
        default=None,
        help="35 = 35 mm f/2 ray-traced from the expired US patent 3,006,249, "
             "Example II (default); 50 = 50 mm f/2 double-Gauss (optiland "
             "sample). Both ray-traced with optiland.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Explicit output JPG path. Overrides --out-dir/auto-naming when set.",
    )
    parser.add_argument(
        "--depth-scale",
        type=float,
        default=1.0,
        help="Multiplier for depth_m under --depth-mapping linear (default 1.0); "
             "ignored with a warning under diopter",
    )
    parser.add_argument(
        "--depth-source",
        choices=["auto", "lidar", "monocular"],
        default="auto",
        help="Depth backend; 'auto' (default) = 'lidar' when the HEIC carries "
             "a depth aux image, else 'monocular' (with a note). 'lidar' "
             "decodes HEIC depth_images[0] (strict: errors without it), "
             "'monocular' runs Depth Anything V2 (downloads ~100 MB DA2 "
             "model on first invocation; requires the [depth] extra).",
    )
    parser.add_argument(
        "--depth-guided",
        choices=["auto", "off", "on"],
        default="auto",
        help="S1: 'on' = light luminance-guided smoothing of the LiDAR depth "
             "staircase (depth.guided.smooth_lidar_staircase, leaf defaults) "
             "inside extract_lidar_depth. 'auto' (default) = 'on' with LiDAR "
             "depth, else 'off'. Explicit 'on' requires LiDAR depth.",
    )
    parser.add_argument(
        "--depth-mapping",
        choices=["auto", "linear", "diopter"],
        default="auto",
        help="S1: depth -> defocus-axis mapping. 'auto' (default) = "
             "'diopter' with LiDAR depth, else 'linear'. 'linear' = "
             "depth_m * --depth-scale path (the pre-Run-6 look). 'diopter' = "
             "|1/z - 1/anchor| mapped onto the tensor D axis (consumes "
             "UNSCALED metres; --depth-scale != 1 is ignored with a warning). "
             "Explicit 'diopter' requires LiDAR depth.",
    )
    parser.add_argument(
        "--depth-gain",
        type=_depth_gain_type,
        default="auto",
        help="S1: diopter gain, 'auto' (default; physical, gain 1.0) or "
             "a number in (0, 10]. Only valid with --depth-mapping diopter/auto "
             "(except the inert default 'auto').",
    )
    parser.add_argument(
        "--focus-xy",
        nargs=2,
        type=float,
        default=None,
        metavar=("X", "Y"),
        help="Normalized image coords [0,1] (axis: x=col, y=row) for tap-to-focus. "
             "Omit to use closest-subject (p2 of depth_m).",
    )
    parser.add_argument(
        "--focus-target-m",
        type=float,
        default=None,
        help="Tensor focus distance in meters (default: from tensor's focus_distance_m, else 0.7).",
    )
    parser.add_argument(
        "--downres",
        type=int,
        default=None,
        help="Downsample so the long edge equals this value (e.g. 2048). "
             "Never upscales. Affects output filename suffix (_r<value>). "
             "P1.10.4: replaces former `--working-long-edge` (collapsed; behavior "
             "preserved via downres_array's HDR-aware scipy bilinear path).",
    )
    # P1.10.4 C5: --layered-gather / --no-layered-gather flag removed.
    # Layered gather is the SOLE rendering path. The render() kwarg
    # survives as a silently-accepted noop (default True) for ABI.
    parser.add_argument(
        "--bright-pass-thresh",
        type=_bright_thresh_type,
        default=None,
        help="Dual-norm bright-pass threshold in linear luminance (P1.11 R1). "
             "Pixels above T convolve with a peak-renormed kernel (bloom "
             "gain); pixels below T use the existing L1 path. Unset = 1.0 "
             "under --tonemap ci (targets HDR highlights) and "
             "max(1.0, 1.1*H(0.9)) under --tonemap input-lut; 'auto' = the "
             "latter (requires input-lut or tonemap auto). A number is used verbatim. Pass 1e9 "
             "to disable (backward-compat -- bit-identical to pre-R1). See "
             "doc/plan/phase1/section11/locked-spec-R1-brightpass.md.",
    )
    parser.add_argument(
        "--bright-pass-region-size",
        type=int,
        default=31,
        help="P1.10: morphological top-hat structuring-element size for "
             "region-aware bright-pass. Default 31 since P1.10.2 (was 0 "
             "pre-P1.10.2; pass 0 to disable, bit-identical to P1.11 "
             "R1-PATCH). Recommended: 15 (small lights only — string"
             "lights, candle), 31 (canonical), 63 (large lights). Detector "
             "fires on pixels with high local-contrast (true highlights); "
             "flat-bright midtones (face, white shirt) get suppressed.",
    )
    parser.add_argument(
        "--bright-pass-floor",
        type=float,
        default=0.10,
        help="P1.10: floor multiplier on the region-detector signal. 0.0 = "
             "strict (only true local-contrast highlights bloom). 1.0 = no "
             "region gate. Default 0.10 preserves a small bloom for "
             "large-uniform-bright safety net (LED panel, sun-disc filling "
             "frame). Only consulted when --bright-pass-region-size > 0.",
    )
    parser.add_argument(
        "--bright-pass-norm-pct",
        type=float,
        default=99.0,
        help="P1.10: percentile of detector values used as scene-scale "
             "anchor. Default 99.0 (top 1%% of pixels define bloom ceiling). "
             "Range [50.0, 100.0]. Only consulted when --bright-pass-region-size > 0.",
    )
    # P1.10.4 C4b: --radial-blend-mode flag deleted. Basis is the sole
    # radial-blend kernel; the `radial_blend_mode` kwarg survives on
    # render() as a silently-accepted noop (default "basis") to preserve
    # call-site ABI for tests not enumerated in spec § 6. See
    # doc/plan/phase1/section18/locked-spec-P1_10_4.md § 9 C4b.
    parser.add_argument(
        "--hdr-source",
        choices=["auto", "none", "macos"],
        default="auto",
        help="HDR input path. 'auto' (default) = 'macos' when the macOS HDR "
             "loader is available (macOS + pyobjc) AND the photo carries an "
             "HDR gain map, else 'none' (one note line). Explicit values "
             "are strict. 'none' = LDR sRGB via pillow/HEIF "
             "(bit-identical to pre-P1.8 render). 'macos' = extended-linear "
             "float32 via Core Image expandToHDR, then Apple's Core Image "
             "CIToneMapHeadroom HDR→SDR step after render() (P1.10.2 retired "
             "the prior Reinhard-tonemap stage). Requires macOS + the "
             "[hdr-macos] extra. See doc/plan/phase1/section8/"
             "locked-spec-hdr-access.md.",
    )
    # ---- P1.13 K11 LDR-anchored per-channel shadow match (locked-spec § 2)
    parser.add_argument(
        "--tonemap",
        choices=["auto", "ci", "input-lut"],
        default="auto",
        help="HDR final tonemap. 'auto' (default) = 'input-lut' when "
             "--hdr-source macos delivers HDR headroom and shadow-match is "
             "off, else 'ci'. 'ci' = Core Image tonemap. "
             "'input-lut' = reproduce the input's OWN SDR tone with a measured "
             "luma LUT (HDR-vs-SDR base of the input HEIC); also sets the "
             "auto bright-pass threshold. Explicit 'input-lut' requires "
             "--hdr-source macos; not combinable with --shadow-match-mode on.",
    )
    parser.add_argument(
        "--shadow-match-mode",
        choices=["off", "on"],
        default="off",
        help="P1.13 K11 LDR-anchored per-channel shadow match. PRE-tonemap "
             "only (the post-tonemap variant from K10 was rejected -- killed "
             "dynamic range). When 'on': preview-tonemap to derive per-channel "
             "shadow ratios in display-linear space, then apply on linear-HDR "
             "buffer pre-final-tonemap (full headroom). 'off' (default) = "
             "bit-identical pre-K11. Requires --hdr-source != none.",
    )
    parser.add_argument(
        "--shadow-band-hi",
        type=float,
        default=0.70,
        help="P1.13 K11 LDR-luma cutoff defining 'shadow' band for "
             "--shadow-match-mode. Default 0.70 (covers shadow + mid; "
             "G11-confirmed on Demo_input as the right coverage to fix "
             "render's mid-band warm cast). 0.30 = shadow only (corrects "
             "shadow blue cast but leaves mid uncorrected). Range "
             "[0.0, 1.0]. Inert when --shadow-match-mode=off.",
    )
    parser.add_argument(
        "--shadow-match-max-gain",
        type=float,
        default=4.0,
        help="P1.13 K11 upper clamp on per-channel gain. Prevents pathological "
             "over-lift on degenerate scenes. Default 4.0. Range [1.0, 16.0]. "
             "Inert when --shadow-match-mode=off.",
    )
    parser.add_argument(
        "--shadow-match-strength",
        type=float,
        default=1.0,
        help="P1.13 K11 strength multiplier on auto-derived per-channel "
             "shadow gain. Default 1.0 (exact LDR match; 1.2 was tried and "
             "deemed too aggressive on G11 review). >1.0 overshoots; <1.0 "
             "undershoots. Multiplied BEFORE clamp. Range [0.0, 4.0]. "
             "Inert when --shadow-match-mode=off.",
    )
    parser.add_argument(
        "--slab-batch-size",
        type=int,
        default=None,
        help="Depth-layered gather slab-batching size. Default (None → 1). "
             "Values > 1 retain N slabs and only cost memory; kept for "
             "compatibility. Output is bit-identical for every value.",
    )
    # ---- R-occlusion-layering (HALO-ONLY) -----------------------------------
    parser.add_argument(
        "--occlusion-compositing",
        choices=["off", "on"],
        default="off",
        help="R-occlusion-layering premultiplied-alpha layered OVER "
             "compositor. 'off' (default) = bit-identical kill-switch (the "
             "existing PoU linear-weighted-sum gather; canonical-bench md5 "
             "preserved). 'on' = ∇depth-gated three-regime matte: smooth "
             "depth stays PoU-linear (no banding), true silhouettes get a "
             "hard nearest-slab matte + guided-feather boundary band → "
             "premultiplied Porter-Duff OVER (kills the foreground/background "
             "occlusion-boundary milky halo by construction). The bright path "
             "is unchanged this round (snowflakes deferred to "
             "R-highlight-guard). See doc/plan/phase2/section3_occlusion-"
             "layering/locked-spec-R-occlusion-layering.md.",
    )
    # ---- removed in 0.5.0: tombstones (hidden). _validate_args turns any use
    # into a one-line error that points at the migration note.
    parser.add_argument(
        "--kernel-family", nargs="?", default=argparse.SUPPRESS,
        dest="_removed_kernel_family", help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--lens-aperture-mm", nargs="?", default=argparse.SUPPRESS,
        dest="_removed_lens_aperture_mm", help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--psf-norm", nargs="?", default=argparse.SUPPRESS,
        dest="_removed_psf_norm", help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--bright-stamp",
        choices=["auto", "peak", "l1"],
        default="auto",
        help="Bright-pass stamp. 'auto' (default) = 'l1'. 'peak' = "
             "peak-normalised kernel (area-times bloom); 'l1' = the "
             "L1-normalised kernel times --bright-stamp-gain "
             "(energy-honest).",
    )
    parser.add_argument(
        "--bright-stamp-gain",
        type=_stamp_gain_type,
        default=None,
        help="Gain of the 'l1' bright stamp, a number in (0, 16] (default "
             "2.0 under an 'l1' stamp). Requires an 'l1' stamp (--bright-stamp "
             "l1, or auto).",
    )
    return parser
