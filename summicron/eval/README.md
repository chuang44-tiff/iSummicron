# summicron/eval/ — PSF-QC eval harness

Synthetic-fixture sanity checks for the render pipeline. **Numbers, not
gates.** Six metrics (M1–M5, M7; M6 optional) computed on three fixture
families. No pass/fail verdicts in v0 — the report is a flat dict of
scalars + flags that the Auditor inspects by hand.

See (dev-only planning doc, not shipped) for the full
spec. This README is the operator-level summary.

## What's here

```
summicron/eval/
├── __main__.py           # CLI: python -m summicron.eval ...
├── score.py              # score() dispatcher (Python API)
├── fixtures/             # fixture generators (Coder A)
└── metrics/
    ├── common.py         # fit_ellipse_guarded, gradient_energy, radial projection
    ├── energy.py         # M1
    ├── bubble.py         # M2 + M3 + M4
    ├── usaf.py           # M5 (+ M6 optional)
    └── wrap.py           # M7 wrap-probe sanity
```

## Fixtures

Three scenes (details in the spec §2; generators in `summicron/eval/fixtures/`):

| scene          | content                                                     |
|----------------|-------------------------------------------------------------|
| `bubble_grid`  | 12 discs on deterministic grid. 11 PSF-QC + 1 wrap probe.   |
| `bubble_random`| 12 discs, seed=42, 96 px rejection. No wrap probe.          |
| `usaf_depth`   | 5 USAF-style cross tiles at the five tensor-native depths.  |

All fixtures: 1024×1024, monochrome (R=G=B), float32 in [0, 1] sRGB
linear-intensity, saved as `.npz` + sidecar `.json`. Optical center at
(512, 512).

Generate them:

```bash
python -m summicron.eval.fixtures --scene bubble_grid --out runs/eval/bubble_grid_input.npz
python -m summicron.eval.fixtures --scene bubble_random --seed 42 --out runs/eval/bubble_random_input.npz
python -m summicron.eval.fixtures --scene usaf_depth --out runs/eval/usaf_depth_input.npz
```

### The wrap probe disc

`bubble_grid` includes disc #11 at (720, 512) with `depth_m = 0.40` and
`wrap_applied = true`. With `focus_distance_m = 0.70`, the tap-to-focus
wrap path (`summicron/depth/estimate.py:apply_tap_focus_shift`) sends this
pixel to effective depth `2·0.70 − 0.40 = 1.00 m`. M7 asserts the render
came out non-degenerate on this disc. This is **not** a "did we pick the
right PSF for 0.4m?" check — no PSF-for-0.4m exists. It's a liveness
check on the wrap code path.

## Metrics

| ID | What                                           | Input needed  | Applicable scenes           |
|----|------------------------------------------------|---------------|-----------------------------|
| M1 | energy ratio = rendered.sum() / input.sum()    | input + rend. | all                         |
| M2 | on-axis disc roundness (minor/major)           | rendered      | bubble_grid, bubble_random  |
| M3 | off-axis sagittal/tangential **signed** ratio  | rendered      | bubble_grid, bubble_random  |
| M4 | Spearman ρ of diameter vs \|depth−focus\|, per-column | rendered | bubble_grid, bubble_random |
| M5 | USAF center-tile gradient-energy ratio         | input + rend. | usaf_depth                  |
| M6 | per-tile gradient-energy ratio (optional)      | input + rend. | usaf_depth                  |
| M7 | wrap-probe sanity (finite_ok, support_px, diameter_ratio) | rendered | bubble_grid |

M2/M3/M4 **exclude** discs with `wrap_applied=true` from their scopes.

M3 uses **radial-vector projection** (project the fitted ellipse's
covariance onto the disc→optical-center unit vector and its perpendicular)
and returns a **signed** ratio `sigma_sag / sigma_tan`. This is the P0.2
regression witness — direction matters.

M4 uses `scipy.stats.spearmanr(nan_policy="raise")` — a column with any NaN
diameter returns NaN with flag `"upstream_nan"` rather than silently
poisoning the correlation.

Degeneracy guards across all disc metrics: minimum support 20 px, minimum
minor axis 0.5 px. Failures return NaN with a typed flag instead of
zero-axis fits.

## Python API

```python
from summicron.eval import score  # re-exported from summicron/eval/__init__.py

report = score(
    rendered_rgb,      # (H, W, 3) float32
    depth_m,           # (H, W) float32 — from the fixture, not the renderer
    annotation,        # dict loaded from fixture sidecar .json
    input_rgb=input_rgb,                     # required for M1 + M5
    metrics=("M1", "M2", "M3", "M4", "M5", "M7"),
)
# report is a flat JSON-serializable dict
# { "M1": 1.0, "M1__flag": null, "M2": 0.98, ..., "_meta": {...} }
```

## CLI

```bash
python -m summicron.eval \
  --input    runs/eval/bubble_grid_input.npz \
  --rendered runs/eval/bubble_grid_rendered.npz \
  --output   runs/eval/bubble_grid_report.json \
  --metrics  M1,M2,M3,M4,M7
```

The rendered `.npz` must carry the same `annotation` (scene mismatch is a
hard error). The caller is responsible for running the renderer; this
harness only scores.

## Report schema (flat)

```jsonc
{
  "M1": 1.002,         "M1__flag": null,
  "M2": 0.98,          "M2__flag": null,
  "M3": [              // per-disc, only for bubble scenes
    {"idx": 3, "sag_tan": 1.42, "theta_deg": -2.1, "flag": null},
    ...
  ],
  "M4": {"center": 0.95, "mid": 0.98, "corner": 0.99},
  "M4__flag_center": null, "M4__flag_mid": null, "M4__flag_corner": null,
  "M5": 0.97,          "M5__flag": null,
  "M7": {"finite_ok": true, "support_px": 132, "diameter_ratio": 1.04},
  "M7__flag": null,
  "_meta": {"scene": "bubble_grid", "n_discs": 12, "frame_hw": [1024, 1024], ...}
}
```

No pass/fail verdicts. Downstream code reads numbers and records baselines.
