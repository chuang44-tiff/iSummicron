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


def add_render_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Install every render flag on ``parser`` and return it."""
    parser.add_argument("--image", required=True, help="Input image path (HEIC/JPEG/PNG)")
    parser.add_argument(
        "--tensor",
        default=None,
        help="4D tensor .npz path. When omitted, defaults to the wheel-shipped "
             "canonical tensor (summicron.render._artifacts.default_tensor_path()).",
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
        help="Multiplier for depth_m array before render (default 1.0; only applies with --depth-source=lidar)",
    )
    parser.add_argument(
        "--depth-source",
        choices=["lidar", "monocular"],
        default="lidar",
        help="Depth backend; 'lidar' decodes HEIC depth_images[0], "
             "'monocular' runs Depth Anything V2 (downloads ~100 MB DA2 "
             "model on first invocation; requires the [depth] extra).",
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
    parser.add_argument(
        "--psf-norm",
        choices=["l1", "peak"],
        default="l1",
        help="PSF normalization mode. 'l1' (default) keeps kernels L1-normed "
             "and averages via post-conv weight-sum divide (bit-identical to "
             "pre-P1.7 render path). 'peak' peak-normalizes kernels (stamp "
             "mode): bright pixels bloom by sum(kernel) and overflow clips at "
             "finalize. See doc/plan/phase1/section7/locked-spec-stamp-mode.md.",
    )
    # P1.10.4 C5: --layered-gather / --no-layered-gather flag removed.
    # Layered gather is the SOLE rendering path. The render() kwarg
    # survives as a silently-accepted noop (default True) for ABI.
    parser.add_argument(
        "--bright-pass-thresh",
        type=float,
        default=1.0,
        help="Dual-norm bright-pass threshold in linear luminance (P1.11 R1). "
             "Pixels above T convolve with a peak-renormed kernel (bloom "
             "gain); pixels below T use the existing L1 path. Default 1.0 "
             "targets HDR highlights. Pass 1e9 to disable (backward-compat "
             "— bit-identical to pre-R1). See "
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
        choices=["none", "macos"],
        default="none",
        help="HDR input path. 'none' (default) = LDR sRGB via pillow/HEIF "
             "(bit-identical to pre-P1.8 render). 'macos' = extended-linear "
             "float32 via Core Image expandToHDR, then Apple's Core Image "
             "CIToneMapHeadroom HDR→SDR step after render() (P1.10.2 retired "
             "the prior Reinhard-tonemap stage). Requires macOS + the "
             "[hdr-macos] extra. See doc/plan/phase1/section8/"
             "locked-spec-hdr-access.md.",
    )
    # ---- P1.13 K11 LDR-anchored per-channel shadow match (locked-spec § 2)
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
    return parser
