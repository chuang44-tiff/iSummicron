# tests/ — pytest suite

**Empty in pre-v0 phase.** Four layers of tests.

## Layout

```
tests/
├── README.md              this file
├── unit/                  pure Python logic (planned)
├── adversarial/           edge cases and hostile inputs (planned)
├── correctness/           numerical invariants (planned)
└── [no integration/]      deliberately none — no external system
```

## Test layers

| Layer | What it tests | When run |
|---|---|---|
| `unit/` | Bin math, `.npz` schema, coordinate conventions, highlight expansion math, tone curve math, EXIF helpers, annulus photometry helpers, CLI arg parsing | FAST (every round) |
| `adversarial/` | Corrupt DNG/HEIC, empty bins, single-sample bins, saturation-clipped blobs, `.npz` schema mismatches, missing EXIF, missing face, rotation drift, bin-boundary seams | FAST (every round) |
| `correctness/` | **Numerical invariants** — synthetic-scene round-trip, energy conservation (0.3% with rotation renorm), rotation invariance at 0°/15°/30°/45°, bin-boundary seam absence, canonicalization idempotence, optical-center symmetry score | FAST every round; `slow`-marked subset (full depth model load, real corpus subset) only in FULL mode |
| **Visual output review** | Bokeh quality vs reference Summicron crops. Not code: a manual look at rendered output at the end of a development round | Round end, run manually |

## FAST vs FULL modes

**FAST** (default, every coding round):
```bash
pytest tests/unit/ tests/adversarial/ tests/correctness/ -v -m "not slow"
```

**FULL** (section end or Team Lead override):
```bash
pytest tests/ -v
```

FULL includes `slow`-marked tests that load Depth Anything 3 Base onto MPS or touch the real NAS corpus subset. Slow tests must skip gracefully if weights are not cached (first run on a cold machine should not be a false failure).

## xfail lifecycle

Adversarial agent writes failing tests as `@pytest.mark.xfail(reason="BUG: ...")`. Coder fixes the underlying bug and removes the xfail marker in the next round. Team Lead tracks open vs resolved count in the TLDR report. Section cannot PASS with unresolved xfails unless Team Lead explicitly defers with a documented reason.

## "Tests that only fit the synthetic fixture" rule

Every test using a hand-built kernel / synthetic blob / hand-written `.npz` must also assert an invariant that would catch a real algorithm bug (energy conservation, rotation invariance, bin addressing). Tests that only prove "my synthetic fixture round-trips through my synthetic pipeline" are rejected by the Auditor.

## Related

- `../summicron/` — code under test
