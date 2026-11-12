# Canonical bench

The bench is the fixed render used to check that a change does not alter the output by accident. It is a
regression guard, not a quality judgement: the same command on the same input must give the same bytes.

The bench photo (`IMG_6915.HEIC`, an iPhone Portrait photo of a person) is a private test image and is not
distributed. The commands below show the exact flags; to try the same look on the public sample, use the
quick start in the main README.

## PINS

Exact md5 sums of the output JPEG and of the two shipped tensors. They were measured on macOS 27.2 with
pyobjc-core 12.2; the HDR path uses Core Image, so other macOS versions can differ.

| Item | Value |
|---|---|
| Tensor 35 (`summicron/_artifacts/psf_lens35_us3006249_fu0.4.npz`) | `894f858c54e0cad45eedcddab47be6d7` (`TENSOR_MD5`) |
| Tensor 50 (`summicron/_artifacts/psf_lens50_doublegauss_fu0.4.npz`) | `10824ae374e7ab57e651d3d770a76b6a` (`TENSOR_MD5_50`) |
| Default look, 35 mm f/2, command below | `a15314d23dfe09070667a15f66a362d9` (`G1_MD5`) |
| Default look + `--occlusion-compositing on` | `3659beff1f331b6d041ca50d90421718` (`ON_MD5`) |
| Default look + `--lens 50` (50 mm f/2) | `b2eebd9d0605cb1a1e1e617b12f41d0a` (`LENS50_MD5`) |

The constants live in `summicron/render/_bench.py`; this table mirrors them. In Python the same render is
`RenderOptions.canonical_bench(image_path, focus_xy)` from `summicron.render.api`.

## Primary canonical bench

Flags after `--image` and before `--focus-xy` are `CANONICAL_FLAGS`. There is no `--tensor` and no `--lens`: the
renderer uses the default 35 mm tensor shipped in the wheel.

```bash
python -m summicron.render.convolve \
  --image iphoneImage/IMG_6915.HEIC \
  --depth-source lidar --downres 2048 --hdr-source macos \
  --focus-xy 0.40 0.45 \
  --output runs/vibe_bench/IMG_6915_bench.jpg
```

The default look comes from the `auto` defaults: with LiDAR depth the depth is refined with a guided filter and
mapped through a reciprocal (diopter) curve, and with HDR headroom the tone follows the input photo's own SDR
tone (`--tonemap input-lut`). The `Resolved:` line on stdout shows what ran.

## Variants

The 50 mm f/2 lens:

```bash
python -m summicron.render.convolve \
  --image iphoneImage/IMG_6915.HEIC \
  --depth-source lidar --downres 2048 --hdr-source macos --lens 50 \
  --focus-xy 0.40 0.45 \
  --output runs/vibe_bench/IMG_6915_bench_lens50.jpg
```

Occlusion compositing on (about 1.3 to 1.6 times the render time):

```bash
python -m summicron.render.convolve \
  --image iphoneImage/IMG_6915.HEIC \
  --depth-source lidar --downres 2048 --hdr-source macos \
  --focus-xy 0.40 0.45 --occlusion-compositing on \
  --output runs/vibe_bench/IMG_6915_bench_on.jpg
```
