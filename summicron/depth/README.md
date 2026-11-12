# summicron/depth/ -- Monocular Depth & Defocus Mapping

Estimate scene depth from a single image and convert it to a defocus map
for the Summicron PSF renderer.

## What it does

1. **`load_depth_model()`** -- loads Depth Anything V2 Small via the
   `transformers` depth-estimation pipeline. MPS (Apple Silicon) with
   automatic CPU fallback. Weights cached in `~/.cache/summicron/depth/`.

2. **`estimate_depth()`** -- resize input to 1024px long edge, run
   inference, bicubic upsample back to original resolution. Returns a
   float32 depth map (higher = farther).

3. **`find_anchor_depth()`** -- 2nd percentile of the depth map, used as
   the closest-subject anchor (the plane that stays sharp).

4. **`build_depth_defocus_map()`** -- converts depth to defocus using a
   reciprocal-difference formula relative to the anchor depth. Normalized
   by the 98th percentile so the map spans [0, 1].

5. **`estimate_and_build_defocus()`** -- convenience wrapper that chains
   steps 1-4 in a single call. Legacy: the renderer no longer accepts a
   normalized defocus map; it takes metric `depth_m` (see below).

## Usage

```python
from summicron.depth.estimate import load_depth_model, estimate_and_build_defocus

model = load_depth_model()                        # one-time load
defocus = estimate_and_build_defocus(image, model) # float32 (H, W) in [0, 1]

# The renderer consumes metric depth, not a defocus map:
#   render(image, tensor, depth_m=depth_in_meters)
# (e.g. LiDAR via extract_lidar_depth). A normalized defocus map from the
# helpers above must be converted to meters before use.
```

## Model choice

DA-v2 Small (`depth-anything/Depth-Anything-V2-Small-hf`) is the current
default. DA3-Base is the intended target but is not yet available on
HuggingFace as of 2026-04. When it ships, update `_MODEL_ID` in
`estimate.py`.

Both DA-v2 Small and DA3-Base are Apache-2.0 licensed.

## iPhone LiDAR depth extraction (experimental)

Portrait Mode HEIC files from iPhones with LiDAR contain an embedded
disparity map (576x768, uint8). This can be extracted via `pillow_heif`
and used as an alternative to monocular estimation:

```python
import pillow_heif
heif = pillow_heif.open_heif("IMG_portrait.HEIC")
# The auxiliary depth/disparity image is in heif[1] or similar
```

This was validated in Day 6: rendering a regular shot with LiDAR depth
from a matched Portrait shot produced results visually close to Apple's
Portrait Mode output. This path is experimental and not yet integrated
into the CLI.

## Known limitations

- **DA-v2 Small, not DA3-Base.** Smaller model = noisier depth on fine
  structures. Upgrade path is a one-line constant change.
- **Monocular depth is relative, not metric.** The reciprocal-difference
  formula and percentile normalization handle this, but extreme scenes
  (all objects at similar depth) may produce flat defocus maps.
- **No temporal consistency.** Single-frame estimation only.
