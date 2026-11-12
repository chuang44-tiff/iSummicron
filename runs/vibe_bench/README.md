# vibe_bench — canonical visual-review bench spec

**This file is the source of truth for the bench render standard.**
If you change the standard, update this file first. If a probe / handoff / TLDR
ever disagrees with this file, **this file wins**.

Last updated: 2026-11-08 (HYG-0: IMG_6915 @ 0.40 0.45 is the primary scene; one command per scene, no `--tensor`; PINS table; argv sourced from `summicron/render/_bench.py`)
Prior: 2026-05-03 (P1.audit: backend hardening; `--tensor` now defaults to wheel-shipped `summicron/_artifacts/codev_render_v1.npz`; native iPhone resolution renders now require explicit `--downres N` per the new 8 GB byte-budget memory guard — canonical bench at `--downres 2048` unaffected)
Prior: 2026-05-02 (P1.13: K11 LDR-anchored shadow-match shipped; default `--shadow-match-mode off` keeps canonical bench bit-identical to md5 `38c75dd2…`)
Prior: 2026-04-27 (P1.10.2: top-hat region=31 + basis + layered-gather become render defaults; canonical command unchanged because flags were already canonical, just defaults absorb them)
Prior: 2026-04-23 (R0-DEBUG: post-R0 canonical IMG_6915 face anchor locked at `--focus-xy 0.40 0.45`)
Prior: 2026-04-18 (P1.9: tensor path switches from _maxlog.npz to plain .npz; S2 delta guard is now build-time)

## Change log

- **2026-11-08 HYG-0 — bench spec consolidated.** Primary scene is IMG_6915 @ `0.40 0.45` (the G1-pinned scene); IMG_6651 @ `0.5 0.75` is the alternate. `--tensor` dropped from every command (default wheel tensor). PINS table added (OFF/ON/tensor md5). `CANONICAL_FLAGS` in `summicron/render/_bench.py` drives the G1 test argv and the `anchor_picker` template. Stale `IMG_6915_ON*.jpg` (`2529c327…`) flagged.
- **2026-05-02 P1.13 — K11 LDR-anchored per-channel pre-tonemap shadow match (opt-in).** New flag `--shadow-match-mode {off, on}` (default `off` = bit-id preserved against canonical R2 baseline `38c75dd2…`). When `on`: preview-tonemap-derived per-channel shadow ratios applied on linear-HDR pre-final-tonemap, smoothstep-band-gated to LDR shadow+mid band. Auto-derived per scene (no hardcoded gamma). 4 flags: `--shadow-match-mode`, `--shadow-band-hi 0.70`, `--shadow-match-strength 1.0`, `--shadow-match-max-gain 4.0`. Cleanup deleted 5 K0-K6 tonemap modules + 9 superseded CLI flags (-2316 LOC). Canonical bench command UNCHANGED — defaults preserve bit-id. Commits: `a01a388` ship → `c0214e4` cleanup → `fab0177` adversarial fixes. Spec: (dev-only planning doc, not shipped).
- **2026-04-28 P1.10.4 — slab + scatter (layered=False) paths fully deleted.** Basis + layered are now sole render kernels. Removed flags: `--working-long-edge` (collapsed into `--downres`), `--r-blend-strength`, `--radial-blend-mode`, `--layered-gather`/`--no-layered-gather` (kwargs remain on `render()` signature as silently-accepted noops with `DeprecationWarning`). Added: `tests/adversarial/test_basis_chroma_battery.py` (5 fixtures × 3 gates, G_CHROMA_HUE ≤1% drift). Bit-identity to canonical bench md5 `38c75dd253571ece9c9a0e3223fdb580` preserved across all 8 P0 commits (C1 → C6: `8733561` → `f20e115` → `55b758b` → `75a1f51` → `f4a292a` → `7d402f8` → `748ad76` (WAYPOINT-3) → `8290a30`). Spec: (dev-only planning doc, not shipped).
- **2026-04-27 P1.10.2 — top-hat region-bright-pass + radial basis blend + layered-gather become default ON.** Canonical bench command body unchanged (no flag was added or removed from the typical invocation), but the underlying defaults flipped: `bright_pass_region_size 0→31`, `radial_blend_mode "slab"→"basis"`, `layered_gather False→True` (`argparse.BooleanOptionalAction`). Old behavior reachable via three explicit kill-switches: `--bright-pass-region-size 0 --radial-blend-mode slab --no-layered-gather`. New canonical R2 baseline `IMG_6915_R2_layered.jpg` md5 `38c75dd253571ece9c9a0e3223fdb580` (replaces the prior P1.10.1 baseline). The `--r-blend-strength` flag is now vestigial in the default `basis` path (still consumed by slab path); scheduled for removal alongside slab in P1.10.4. Also removed in P1.10.2: `--bright-pass-knee` (hardcoded 0.20 constant), `--dual-tensor`, `--depth-source none`. See (dev-only planning doc, not shipped).

## PINS (exact md5; no tolerance)

The md5 constants live in `summicron/render/_bench.py` (private, dev-only) — that
module is the single source of truth; this table mirrors it.

| Item | Value |
|---|---|
| Platform pinned on | macOS 27.2 (build 26B123), pyobjc-core 12.2 |
| Tensor (`summicron/_artifacts/codev_render_v1.npz`) | `e6f501546cb77ba40ad904dbd4ba9369` (`TENSOR_MD5`) |
| OFF — primary command below (exact d0 delta; unconditional since Q-b-legacy-removal) | `a106801a97a05ee7d5033e2b8d647f11` (`G1_MD5`) |
| ON — primary + `--occlusion-compositing on` | `6cebff5403f7f0fdcda109cb8bbff35a` (`ON_MD5`) |
| HISTORICAL pre-Q-b renders (flag removed; no longer reproducible; protected by `scripts/clean_runs.py` literals) | OFF `38c75dd253571ece9c9a0e3223fdb580` / ON `07f5b91bc2a6ff46f1e93ddd96d6fc2e` |
| ON pin file (local copy, gitignored; HISTORICAL pre-Q-b ON render, md5 `07f5b91b…`) | `runs/vibe_bench/IMG_6915_occ_ON_pin.jpg` (copied from `runs/render/current_check/IMG_6915_occ_on.jpg`, RUNS-PRUNE) |

**Stale, never cite:** `IMG_6915_ON*.jpg` in this directory (md5 `2529c327…`) are
pre-`cc880ac` intermediates — not the ON pin. Local G1 baseline
`IMG_6915_R2_layered.jpg` (gitignored, optional) is restored with
`cp runs/vibe_bench/IMG_6915_G1_baseline_delta.jpg runs/vibe_bench/IMG_6915_R2_layered.jpg`.

**Retention (RUNS-PRUNE):** archived probe dirs live under `runs/_archive/<group>/`; old→new paths are in `runs/_archive/clean_log.jsonl`.

In-process equivalent of the bench command: `RenderOptions.canonical_bench(image_path, focus_xy)` (`summicron.render.api`).

## Primary canonical bench — IMG_6915 (face / HDR string lights)

Flags after `--image` and before `--focus-xy` == `CANONICAL_FLAGS`; this is the
exact argv of the SLOW G1 test (`tests/unit/test_back_compat_canonical.py`).
No `--tensor`: the renderer defaults to the wheel-shipped tensor.

```bash
conda run -n isummicron python -m summicron.render.convolve \
  --image iphoneImage/IMG_6915.HEIC \
  --depth-source lidar --depth-scale 10 --downres 2048 --psf-norm l1 --hdr-source macos \
  --focus-xy 0.40 0.45 \
  --output runs/vibe_bench/IMG_6915_bench.jpg
```

ON variant (occlusion compositor; ~150 s @2048):

```bash
conda run -n isummicron python -m summicron.render.convolve \
  --image iphoneImage/IMG_6915.HEIC \
  --depth-source lidar --depth-scale 10 --downres 2048 --psf-norm l1 --hdr-source macos \
  --focus-xy 0.40 0.45 --occlusion-compositing on \
  --output runs/vibe_bench/IMG_6915_bench_on.jpg
```

**Visual success criterion:** face in focus; palm leaves / room behind it blurred.
The scene has 12.25% HDR headroom (peak 3.5× above SDR), so it exercises the
bright-pass path. Output is portrait `(2048, 1536, 3)` uint8.

**Anchor history:** `--focus-xy 0.34 0.60` (pre-R0 landscape-storage anchor) lands
on the shoulder in post-R0 portrait storage — **do not use**. `0.40 0.45` lands on
the eyes. See (dev-only planning doc, not shipped).

## Alternate scene — IMG_6651 (stuffed animal / tree)

```bash
conda run -n isummicron python -m summicron.render.convolve \
  --image iphoneImage/IMG_6651.HEIC \
  --depth-source lidar --depth-scale 10 --downres 2048 --psf-norm l1 --hdr-source macos \
  --focus-xy 0.5 0.75 \
  --output runs/vibe_bench/IMG_6651_bench.jpg
```

**Visual success criterion:** the stuffed animal CLEAR (in focus), the tree in the
background BLURRED. Either inverted = bench broken. No md5 pin for this scene.
Expected stats: `(1536, 2048, 3)` uint8, mean ∈ [115, 140], `frac(==255) < 0.5%`.

## Why each flag is the way it is

| Flag | Value | Why |
|---|---|---|
| `--tensor` | omitted | Defaults to wheel-shipped `summicron/_artifacts/codev_render_v1.npz` (raw tensor + build-time S2 delta guard, P1.9; OTF-norm deprecated — do not use it in any bench). |
| `--depth-source lidar` | — | iPhone HEIC carries a LiDAR disparity aux; used instead of monocular depth. |
| `--depth-scale 10` | 10× | Expands the compressed near-field LiDAR depth into the tensor's distance range. Scale is applied BEFORE the anchor shift (P1.8.2). |
| `--downres 2048` | 2048 | Working resolution; native 5712 is slow and needs the byte-budget memory guard headroom. |
| `--focus-xy` | per scene | Tap target that sets `anchor_m`; the depth map is shifted so the anchor lands in the tensor's 0.7 m focus slot. |
| `--psf-norm l1` | L1 | Per-kernel sum-to-1, energy-preserving. |
| `--hdr-source macos` | Core Image | HDR gain-map load + linear-HDR convolve + `CIToneMapHeadroom` HDR→SDR. macOS only. |

## Known quality concerns (carry-forward, not bench-blockers)

- **HDR downres uses bilinear, not Lanczos** — small quality regression vs LDR's Lanczos path.

## Comparison benches (for spec-change experiments only)

If a round needs to compare the canonical bench against a flag variation,
save the variant under a descriptive name alongside, e.g.:

- `IMG_6915_bench.jpg` — canonical (this spec)
- `IMG_6915_bench_psf_peak.jpg` — `--psf-norm peak` variant
- `IMG_6915_bench_depth5.jpg` — `--depth-scale 5` variant

Do NOT change the canonical name. Do NOT use the OTF tensor in any new bench.

## Change log

- **2026-04-23 R0-DEBUG** — Added IMG_6915 alt-canonical section. Anchor corrected from `--focus-xy 0.34 0.60` (pre-R0 landscape-storage, lands on shoulder post-R0) to `--focus-xy 0.40 0.45` (portrait-storage equivalent, lands on eyes). Root cause: the R0 HDR orientation fix rotated storage from landscape to portrait, but `--focus-xy` coords baked in before R0 continued to be used verbatim afterward, landing 0.4m past the face in shifted-depth. Eye Lap variance 4.49 (wrong) vs 60.78 (correct). Doc-only fix — no code change. See (dev-only planning doc, not shipped).
- **2026-04-18 P1.8.2 (late)** — Fixed depth pipeline order: scale-then-shift instead of shift-then-scale. Stuffed animal now correctly lands at 0.7m (tensor focus slice) post-transform. Visual criterion added: stuffed animal clear, tree blurred. User directive: *"first scale all the depth x 10 then put the anchor on the stuffed animal"* + *"stuff animal is clear and tree is blur."*
- **2026-04-18 P1.8.2** — Locked standard: raw tensor + `--focus-xy 0.5 0.75` + 10× depth + HDR CI tonemap. User directive: *"all bench should put the stuffed animal as anchor: 0.7m ... use raw tensor (OTF is bad) ... 10X LiDAR distance."*
- **2026-04-18 P1.8.1** — HDR CI tonemap landed (replaces extended Reinhard).
- **2026-04-17 P1.8** — HDR loader via Core Image `expandToHDR` landed.
- **2026-04-xx P1.6** — First confirmation that raw tensor beats OTF on stuffed-animal subject.
