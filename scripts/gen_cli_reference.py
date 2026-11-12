#!/usr/bin/env python3
"""Generate docs/cli-reference.md from the real argparse definition (stdlib only).

    python scripts/gen_cli_reference.py --write [PATH]
    python scripts/gen_cli_reference.py --check [PATH]

Flags, choices and defaults come from ``summicron.render._render_args``. The prose is
curated below and must cover every flag exactly: a new flag fails this script until it
is documented. The default PATH is ``docs/cli-reference.md`` under the docs root, which
is ``release/public-overlay`` in a dev checkout and the repository root otherwise.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True  # must precede any summicron import: stray bytecode breaks the release gates

import argparse  # noqa: E402
import difflib  # noqa: E402
from pathlib import Path  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

GROUP_ORDER = ("Basic", "Depth and tone", "Highlights", "Advanced", "Shadow match (experimental)", "Bright stamp (experimental)")

DESCRIPTIONS = {
    "image": "Input image (HEIC, JPEG or PNG). LiDAR depth needs an iPhone Portrait HEIC; other images fall back to monocular depth under `--depth-source auto`.",
    "tensor": "Your own PSF tensor `.npz` (see \"Bring your own PSF tensor\" in the usage guide; `scripts/build_psf` builds one with optiland). Default: the bundled tensor chosen by `--lens`. Not combinable with `--lens`.",
    "lens": "Bundled lens. `35` (default): a 35 mm f/2 ray-traced from the expired US patent 3,006,249, Example II, using optiland. `50`: a 50 mm f/2 double-Gauss, the optiland sample prescription. Both are traced design prescriptions, not measurements of any production lens. Not combinable with `--tensor`.",
    "out_dir": "Output directory (default `./runs/render`). Writes `<stem>[_r<N>]_rendered[_ds<scale>].jpg` and `.png`.",
    "output": "Write exactly this file instead (`.png` writes PNG, anything else JPEG quality 95). Overrides `--out-dir`.",
    "depth_scale": "Multiplier on the depth map (meters) before focusing. Only the `linear` depth mapping uses it (the old look used 10); under `diopter` it is ignored with a warning.",
    "depth_source": "`auto`: `lidar` when the HEIC carries a depth image, otherwise `monocular` with a note. `lidar`: the HEIC's embedded depth (strict: errors without one). `monocular`: Depth Anything V2 Small (needs the `depth` extra; downloads about 100 MB on first use).",
    "depth_guided": "Light luminance-guided smoothing of the LiDAR depth staircase. `auto`: `on` with LiDAR depth, else `off`. An explicit `on` needs LiDAR depth.",
    "depth_mapping": "How depth maps onto the PSF table's distance axis. `auto`: `diopter` with LiDAR depth, else `linear`. `linear`: `depth * --depth-scale` (the pre-0.4.0 look). `diopter`: reciprocal-depth distance from the focus point; ignores `--depth-scale`. An explicit `diopter` needs LiDAR depth.",
    "depth_gain": "Gain of the `diopter` mapping: `auto` (a fixed 1.0, the physical value) or a number in (0, 10]. Larger means more blur and a narrower sharp zone.",
    "tonemap": "Final HDR-to-SDR tone. `auto`: `input-lut` when `--hdr-source macos` delivers HDR headroom and shadow match is off, else `ci`. `ci`: the Apple Core Image tonemap. `input-lut`: reproduce the input photo's own SDR tone with a measured luma curve. An explicit `input-lut` needs `--hdr-source macos`.",
    "focus_xy": "Focus point, normalized: X = column, Y = row, (0, 0) = top-left, each in [0, 1]. Omitted: the closest subject.",
    "focus_target_m": "Advanced. Tensor focus distance in meters (default: the tensor's own, 0.7 m for the shipped tensor). Other values warn that the subject may not be sharp.",
    "downres": "Downsample so the long edge is N pixels (never upscales). Adds `_r<N>` to output names. 2048 is the fast preview size.",
    "bright_pass_thresh": "Linear-luminance level above which pixels bloom as highlights. Unset or `auto`: chosen per image from the tonemap. A number is used as given; 1e9 disables.",
    "bright_pass_region_size": "Highlight detector size in pixels (0–255; 0 turns the region gate off). Suggested 15 small lights, 31 default, 63 large lights.",
    "bright_pass_floor": "0–1. Bloom kept outside detected highlights (0 strict, 1 no gate). Used when region size > 0.",
    "bright_pass_norm_pct": "50–100. Detector percentile that sets the bloom scale. Used when region size > 0.",
    "hdr_source": "`auto`: `macos` when the macOS loader is available and the photo carries a gain map, else `none` with a note. `none`: 8-bit SDR load. `macos`: load the HDR gain map with Core Image and tonemap back to SDR, so highlights keep headroom (macOS 15+, `hdr-macos` extra).",
    "shadow_match_mode": "Experimental. `on` matches shadow colour to the SDR input before tonemapping. Needs `--hdr-source macos`.",
    "shadow_band_hi": "0–1. Upper luma bound of the matched band.",
    "shadow_match_max_gain": "1–16. Cap on the per-channel gain.",
    "shadow_match_strength": "0–4. Strength multiplier (1.0 = exact match).",
    "slab_batch_size": "Depth slabs per batch (default 1). Larger values only use more memory; output is identical.",
    "bright_stamp": "Experimental. Bright-pass stamp: `auto` is `l1`; `peak` is the peak-normalized kernel; `l1` is the L1-normalized kernel times `--bright-stamp-gain`.",
    "bright_stamp_gain": "Experimental. Gain of the `l1` bright stamp, a number in (0, 16] (default 2.0 under an `l1` stamp).",
    "occlusion_compositing": "`on`: layered compositing that stacks the depth layers front to back, for about 1.3–1.6× render time. Off by default: with real-lens blur it only changes a hairline along the subject's outline.",
}

GROUPS = {
    "lens": "Basic", "image": "Basic", "out_dir": "Basic", "output": "Basic", "depth_source": "Basic", "depth_scale": "Basic",
    "focus_xy": "Basic", "downres": "Basic", "hdr_source": "Basic", "occlusion_compositing": "Basic",
    "bright_pass_thresh": "Highlights", "bright_pass_region_size": "Highlights",
    "bright_pass_floor": "Highlights", "bright_pass_norm_pct": "Highlights",
    "tensor": "Advanced", "focus_target_m": "Advanced", "slab_batch_size": "Advanced",
    "depth_guided": "Depth and tone", "depth_mapping": "Depth and tone", "depth_gain": "Depth and tone", "tonemap": "Depth and tone",
    "bright_stamp": "Bright stamp (experimental)", "bright_stamp_gain": "Bright stamp (experimental)",
    "shadow_match_mode": "Shadow match (experimental)", "shadow_band_hi": "Shadow match (experimental)",
    "shadow_match_max_gain": "Shadow match (experimental)", "shadow_match_strength": "Shadow match (experimental)",
}

REMOVED = {
    "--kernel-family": "the only PSF source is the ray-traced lens tensor (use --lens 35|50 or --tensor PATH)",
    "--lens-aperture-mm": "the only PSF source is the ray-traced lens tensor (use --lens 35|50 or --tensor PATH)",
    "--psf-norm": "PSFs are always L1-normalised",
}

ENV = {
    "ISUMMICRON_MAX_BYTES": "Memory budget in bytes for the pre-render guard (default 12 GiB). A render estimated above it stops with a message that names a `--downres` that fits.",
    "ISUMMICRON_DEBUG": "Set to any value other than empty or `0` to print the full traceback instead of the one-line `render: <msg>` error.",
}


def _parser() -> argparse.ArgumentParser:
    from summicron.render import _render_args

    assert Path(_render_args.__file__).resolve().is_relative_to(REPO), _render_args.__file__
    p = argparse.ArgumentParser(prog="isummicron render", add_help=False)
    return _render_args.add_render_arguments(p)


def _actions():
    return [a for a in _parser()._actions
            if a.option_strings and "--help" not in a.option_strings and not a.dest.startswith("_removed_")]


def _coverage_errors(dests: set[str]) -> list[str]:
    errs = []
    for name, table in (("DESCRIPTIONS", DESCRIPTIONS), ("GROUPS", GROUPS)):
        keys = set(table)
        if keys != dests:
            errs.append(f"{name}: missing {sorted(dests - keys)} extra {sorted(keys - dests)}")
    bad = sorted({g for g in GROUPS.values()} - set(GROUP_ORDER))
    if bad:
        errs.append(f"GROUPS: unknown group {bad}")
    return errs


def _default(a: argparse.Action) -> str:
    d = a.default
    if a.dest == "lens":
        return "35"  # argparse default is None so --lens/--tensor exclusivity can be checked
    if d is None:
        return "unset"
    return d if isinstance(d, str) else repr(d)


def _value(a: argparse.Action) -> str:
    if a.nargs == 2:
        return " X Y"
    if a.choices:
        return " {" + "|".join(a.choices) + "}"
    if a.dest in ("image", "tensor", "output"):
        return " PATH"
    if a.dest == "out_dir":
        return " DIR"
    return " N" if a.type is int else " VALUE"


def render() -> str:
    acts = _actions()
    errs = _coverage_errors({a.dest for a in acts})
    if errs:
        raise SystemExit("gen_cli_reference: " + "; ".join(errs))
    conv = (REPO / "summicron/render/convolve.py").read_text()
    for name in ENV:
        if name not in conv:
            raise SystemExit(f"gen_cli_reference: {name} not found in convolve.py")
    out = [
        "<!-- GENERATED by scripts/gen_cli_reference.py from summicron/render/_render_args.py. Do not edit. -->",
        "# CLI reference",
        "",
        "```text",
        "isummicron render --image IMAGE [options]",
        "```",
        "",
        "`python -m summicron.render.convolve` accepts the same flags. Back to the [README](../README.md).",
        "",
    ]
    for g in GROUP_ORDER:
        members = [a for a in acts if GROUPS[a.dest] == g]
        if not members:
            continue
        out += [f"## {g}", ""]
        for a in members:
            flag = a.option_strings[0]
            val = _value(a)
            out.append(f"### `{flag}{val}`")
            out.append("")
            facts = []
            if a.required:
                facts.append("Required")
            else:
                facts.append(f"Default: `{_default(a)}`")
            if a.choices:
                facts.append("Choices: " + ", ".join(f"`{c}`" for c in a.choices))
            out.append(". ".join(facts) + ".")
            out += [DESCRIPTIONS[a.dest], ""]
    out += ["## Removed in 0.5.0", "",
            "These flags are still recognised but only to stop with a one-line error. See \"Migrating to 0.5.0\" in the [CHANGELOG](../CHANGELOG.md).", ""]
    for flag, why in REMOVED.items():
        out += [f"- `{flag}`: `render: {flag} was removed in 0.5.0; {why}. See 'Migrating to 0.5.0' in CHANGELOG.md.`"]
    out += ["", "## Environment variables", ""]
    for k, v in ENV.items():
        out += [f"- `{k}`: {v}"]
    out += ["", "## Exit codes", "", "- `0`: the render finished.",
            "- `2`: an input or option error; one line `render: <message>` on stderr.", ""]
    return "\n".join(out)


def _default_path() -> Path:
    root = REPO / "release" / "public-overlay"
    return (root if root.is_dir() else REPO) / "docs" / "cli-reference.md"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--write", nargs="?", const="", metavar="PATH")
    g.add_argument("--check", nargs="?", const="", metavar="PATH")
    a = ap.parse_args(argv)
    text = render()
    if a.write is not None:
        path = Path(a.write) if a.write else _default_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        print(f"wrote {path}")
        return 0
    path = Path(a.check) if a.check else _default_path()
    have = path.read_text() if path.is_file() else ""
    if have == text:
        print(f"ok: {path} is current")
        return 0
    sys.stdout.writelines(difflib.unified_diff(have.splitlines(True), text.splitlines(True), str(path), "generated"))
    print(f"\ngen_cli_reference: {path} is stale; run --write", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
