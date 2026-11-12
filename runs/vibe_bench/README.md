# Canonical bench

The bench is the fixed render used to check that a change does not alter the output by accident. It is a
regression guard, not a quality judgement: the same command on the same input must give the same bytes.

The bench photo (`IMG_6915.HEIC`, an iPhone Portrait photo of a person) is a private test image and is not
distributed. The commands below show the exact flags; to try the same look on the public sample, use the
quick start in the main README.

## PINS

Exact md5 sums of the output JPEG and of the shipped tensor. They were measured on macOS 27.2 with pyobjc-core
12.2; the HDR path uses Core Image, so other macOS versions can differ.

| Item | Value |
|---|---|
| Tensor (`summicron/_artifacts/codev_render_v1.npz`) | `e6f501546cb77ba40ad904dbd4ba9369` (`TENSOR_MD5`) |
| Default look, command below | `f5d86ac424b612770d79a0d1694736b7` (`G1_MD5`) |
| Default look + `--occlusion-compositing on` | `0def67a51677aa1394307aa974fcb3fd` (`ON_MD5`) |
| Earlier look (before 0.4.0), command at the end | `a106801a97a05ee7d5033e2b8d647f11` (`G1_MD5_OLD`) |
| Earlier look + `--occlusion-compositing on` | `6cebff5403f7f0fdcda109cb8bbff35a` (`ON_MD5_OLD`) |

The constants live in `summicron/render/_bench.py`; this table mirrors them. In Python the same render is
`RenderOptions.canonical_bench(image_path, focus_xy)` from `summicron.render.api`.

## Primary canonical bench

Flags after `--image` and before `--focus-xy` are `CANONICAL_FLAGS`. There is no `--tensor`: the renderer uses
the tensor shipped in the wheel.

```bash
python -m summicron.render.convolve \
  --image iphoneImage/IMG_6915.HEIC \
  --depth-source lidar --downres 2048 --psf-norm l1 --hdr-source macos \
  --focus-xy 0.40 0.45 \
  --output runs/vibe_bench/IMG_6915_bench.jpg
```

The default look comes from the `auto` defaults: with LiDAR depth the depth is refined with a guided filter and
mapped through a reciprocal (diopter) curve, and with HDR headroom the tone follows the input photo's own SDR
tone (`--tonemap input-lut`). The `Resolved:` line on stdout shows what ran.

## Occlusion compositing variant

```bash
python -m summicron.render.convolve \
  --image iphoneImage/IMG_6915.HEIC \
  --depth-source lidar --downres 2048 --psf-norm l1 --hdr-source macos \
  --focus-xy 0.40 0.45 --occlusion-compositing on \
  --output runs/vibe_bench/IMG_6915_bench_on.jpg
```

## Earlier look

The look before 0.4.0 stays reachable with explicit flags (`CANONICAL_FLAGS_OLD`):

```bash
python -m summicron.render.convolve \
  --image iphoneImage/IMG_6915.HEIC \
  --depth-source lidar --depth-scale 10 --downres 2048 --psf-norm l1 --hdr-source macos \
  --depth-guided off --depth-mapping linear --tonemap ci --bright-pass-thresh 1.0 \
  --focus-xy 0.40 0.45 \
  --output runs/vibe_bench/IMG_6915_bench_old.jpg
```
