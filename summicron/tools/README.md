# `summicron.tools` — user-facing utilities

Read-only wrappers around the renderer. Nothing in this package modifies the
pipeline, tensor format, or HEIC loader. If a `tools/` module touches
`render/`, `depth/`, or `psf_model/`, it's a bug — file an issue.

---

## `anchor_picker` — click-to-anchor GUI

A matplotlib window that lets you click on the subject of your iPhone photo,
shows the LiDAR depth at the tap with a co-planar warning, and prints the
canonical bench command with `--focus-xy` baked in — ready to copy into a
shell.

**Built 2026-04-19 (P1.10).** Motivation: every round P1.5 → P1.9.1 shipped
correct renderer improvements that were undermined by the user picking the
wrong pixel or the wrong axis. The picker kills that failure mode: what you
click is what the renderer sees.

### Quick start

```bash
conda run -n isummicron python -m summicron.tools.anchor_picker \
  --image iphoneImage/IMG_6651.HEIC
```

A window opens. Click on the subject. Each click:

1. Draws a red `+` at the click pixel
2. Outlines the 21×21 center disk (solid white) and the 30 px / 80 px
   annulus rings (dashed)
3. Prints the depth report and the canonical bench command to stdout

Press `q` or close the window. The most recent click's command is reprinted
on exit for easy copy-paste.

### What you see

```
tap=(0.5000,0.7500) depth_center=0.462m IQR=0.001m annulus=0.464m
WARN: co-planar surface (center≈annulus); subject may not isolate

conda run -n isummicron python -m summicron.render.convolve \
  --image iphoneImage/IMG_6651.HEIC \
  --depth-source lidar --depth-scale 10 --downres 2048 --hdr-source macos \
  --focus-xy 0.5000 0.7500 \
  --output runs/vibe_bench/IMG_6651_bench.jpg
```

Interpreting the fields:

- `depth_center` — median depth over a 21×21 disk around the click
- `IQR` — p75 − p25 across that disk (spread; 0 means uniform)
- `annulus` — median depth over the ring 30 ≤ r ≤ 80 px around the click
- `WARN` — fires when `|depth_center − annulus| < 0.05 m` AND
  `0 < depth_center < 1.0 m`. Meaning: the subject you clicked is
  co-planar with its surround; even a correct tap will NOT isolate the
  subject from the background at this depth resolution. Options: pick a
  different subject, add segmentation, or use a finer depth sampling.

### Flags

| Flag | Default | Purpose |
|---|---|---|
| `--image PATH` | required | HEIC / JPEG / PNG. HEIC gets EXIF auto-applied by `pillow_heif`. |
| `--depth-source` | `lidar` | `lidar` reads Portrait HEIC LiDAR aux. `monocular` runs Depth Anything V2 with `FAR_M=1.0`. (`none` choice removed in P1.10.2 Cut 5.) |
| `--tensor PATH` | canonical | Documentation only — the picker does not render; the tensor path is just substituted into the emitted command. |
| `--no-show` | off | Skip `plt.show()` and exit immediately. Used by tests; prints a placeholder command. |

### Axis convention

**Identity.** The picker shows exactly what `summicron.render.convolve.load_image`
returns and normalizes your click against that grid. `event.xdata / W` →
`--focus-xy X`, `event.ydata / H` → `--focus-xy Y`. No rotation, no EXIF
math on the hot path.

Why: `pillow_heif` auto-applies EXIF orientation on HEIC decode, so
`load_image` returns the visual-orientation grid for HEIC inputs. The
renderer consumes that same grid. The picker inherits both contracts by
passthrough — there is no "display vs storage" axis in the hot path.

For non-HEIC inputs with missing EXIF, the picker shows a yellow banner
and assumes Orientation=1. The `_orientation.py` helper module exists for
future non-HEIC callers but is NOT wired into the picker's click path
(known bugs in the 6/8 entries; fix when someone first needs it).

### Depth probe details

- **Center stat:** median of a 21×21 pixel disk centered at the click,
  clamped to image bounds
- **Spread stat:** IQR (p75 − p25) over that disk — `0` means the disk
  covers a single depth surface
- **Annulus stat:** median over the ring `30 ≤ r ≤ 80` pixels around the
  click (asymmetric near edges but still samples thousands of pixels in
  the non-corner case)
- **Co-planar threshold:** `|center − annulus| < 0.05 m` AND
  `center < 1.0 m`. The 0.05 m threshold corresponds to ~26 mm at 0.46 m,
  which is the stuffed-animal-vs-sofa case that motivated the rule

### When the picker will disappoint you

- **No subject segmentation.** You pick one pixel, get a 21×21 disk. If
  your subject is large and contiguous but occluded by a foreground object,
  the picker can't tell the difference. Segmentation overlay (SAM / GrabCut)
  is deferred to a future round — if you hit this case, file it.
- **LiDAR resolution is the real ceiling.** When subject and background
  are < 30 mm apart, no anchor math saves you. The picker will warn you,
  but the fix is LiDAR × DA2 depth fusion (a different round) or a
  finer tensor depth sampling (also a different round).
- **No preview render.** The picker prints a command; it does not render.
  Copy-paste into your shell. A "render now" button was cut as scope.
- **Static window only.** No zoom / pan / high-DPI refinement beyond
  matplotlib's default toolbar. Fine on Retina for normal-sized subjects.
- **Shell-escape not applied to `--image`.** If your image path contains
  shell metacharacters (e.g. `$(...)`, backticks), the emitted command is
  not safe for blind paste. Known issue — fix pending (tracked as
  `test_emit_canonical_shell_escapes_image_path` xfail).

### Tests

```bash
conda run -n isummicron python -m pytest tests/unit/tools/ tests/adversarial/tools/ -v
```

Expected: all tests pass in a few seconds. Any xfails are
documented tracking debt (see (dev-only planning doc, not shipped)
§ Carry-forward items).

### Architecture

- `anchor_picker.py` — entry module. `main()` → argparse → `load_image` →
  `_load_depth` → matplotlib window → `on_click` loop. Pure compute is
  factored into `probe_depth`, `emit_canonical_command`,
  `_normalize_click` so unit tests don't need a matplotlib backend.
- `_orientation.py` — 8-case EXIF Orientation table. Exported for future
  callers; not wired into the picker. Shipped with known bugs in the 6/8
  entries (xfail-strict pinned); will be fixed by whoever first needs it.

### Further reading

- **Locked spec:** (dev-only planning doc, not shipped)
- **Round TLDR:** (dev-only planning doc, not shipped)
- **Real-data probe:** (dev-only planning doc, not shipped)
- **Canonical bench command spec:** `runs/vibe_bench/README.md`
- **Why this round exists:** repeated anchor-selection failures on
  near-coplanar subjects (for example a stuffed animal in front of a sofa)
