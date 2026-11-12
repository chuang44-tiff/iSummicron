# Troubleshooting

Back to the [README](../README.md). Every CLI error prints one line starting with `render: ` on stderr and exits with code 2. Messages below are shown without that prefix, with paths and numbers replaced by `…`. To see the full traceback, run with `ISUMMICRON_DEBUG=1`.

```bash
export ISUMMICRON_DEBUG=1
```

## The render stops before it starts: memory budget

The estimated peak memory is above the budget (12 GiB by default).

```text
Render would exceed … GiB budget (estimated … GiB at H=… W=… D=…); …
pass --downres … to reduce
```

Fix: pass the `--downres` value the message names, or raise the budget with `ISUMMICRON_MAX_BYTES=<bytes>`. If the budget is tiny, the message says so instead:

```text
budget below the fixed platform overhead; raise ISUMMICRON_MAX_BYTES
```

A value that is not a positive integer is rejected up front:

```text
ISUMMICRON_MAX_BYTES must be an integer > 0 (bytes); got …
```

## Image too large

```text
Image too large: …x… = … pixels > …; pass --downres … to reduce.
```

Fix: use the `--downres` value the message names.

## No depth in the HEIC

```text
--depth-source=lidar but image has no embedded depth aux.
```

The file is a HEIC without a depth image and you passed the strict `--depth-source lidar`. Use an iPhone Portrait HEIC, or drop the flag (the default `auto` falls back to monocular depth) or pass `--depth-source monocular`. For a JPEG or PNG, `--depth-source lidar` fails earlier with a reader message:

<!-- external -->
```text
Invalid input: No 'ftyp' box: File does not start with 'ftyp' box.
```

## HEIC does not open

Without the `heic` extra (pillow-heif), Pillow cannot read HEIC files. With `--hdr-source macos` the failure is a one-line error:

```text
_canonicalize_orientation: PIL failed to open …
```

With `--hdr-source none` it is an uncaught exception:

<!-- external -->
```text
PIL.UnidentifiedImageError: cannot identify image file …
```

Fix: `pip install "isummicron[heic]"` (or the same extras you installed, plus `heic`).

## macOS HDR loader requires …

```text
macOS HDR loader requires `pip install isummicron[hdr-macos]` (installs pyobjc-framework-Quartz, which provides Core Image). Use --hdr-source none to fall back to LDR sRGB load.
```

Fix: install the `hdr-macos` extra, or use `--hdr-source none`. `--hdr-source macos` needs macOS.

## Monocular depth needs the depth extra

```text
DA2 monocular depth requires `pip install isummicron[depth]` (installs torch + transformers; ~2 GB on first invocation). Use --depth-source lidar to avoid this dependency.
```

On the first monocular run the Depth Anything V2 Small weights are downloaded, so it needs a network connection. After the first download you can set `HF_HUB_OFFLINE=1` to stay offline.

## OMP: Error #15

If your own script imports torch before numpy in the conda environment, it aborts:

<!-- external -->
```text
OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib already initialized.
```

Fix: import numpy before torch.

## Center SSIM warning

```text
Center SSIM … < 0.80
```

This `UserWarning` is printed on normal renders and is harmless.

## Exit codes

`0` means the render finished; `2` means an input or option error. Any other exit, with a traceback, is a bug: please report it with the output of `ISUMMICRON_DEBUG=1`.
