# Python API

Back to the [README](../README.md). The CLI is a thin wrapper around two names: `RenderOptions` and `render_image`.

```python
from pathlib import Path
from summicron.render.api import RenderOptions, render_image

opts = RenderOptions(
    image_path=Path("iphoneImage/Demo_input.HEIC"),
    focus_xy=(0.40, 0.39), downres=2048,
    hdr_source="macos", out_dir=Path("runs/render"),
)
result = render_image(opts, on_progress=print)
print(result.output_path, result.mode, round(result.elapsed_s, 1))
```

Notes:

- The fields of `RenderOptions` mirror the [CLI flags](cli-reference.md) (`--depth-scale` is `depth_scale`, and so on). `image_path`, `tensor_path` and `out_path` are the names for `--image`, `--tensor` and `--output`.
- `lens` is `"35"`, `"50"` or `None` (the default, which means the 35 mm lens). It cannot be combined with `tensor_path`.
- `engine` is `"reference"` (the default), `"tiled"` or `"gpu"`, the same as `--engine`. The `"gpu"` value is accepted when the options are built and refused when the render starts if mlx or a Metal GPU is missing (`ValueError`, with the message the CLI prints). With `occlusion_compositing="on"` a fast engine runs the reference engine and reports a `note: engine ... -> reference` line through `on_progress`. See [Render engines](usage.md#render-engines).
- `kernel_family`, `lens_aperture_mm` and `psf_norm` were removed in 0.5.0 and raise `TypeError`; use `lens`. Kernels are always L1-normalized.
- Invalid values raise `ValueError` when the options are constructed, with the same message the CLI prints.
- `render_image` returns a result with `output_u8` (always set, an `(H, W, 3)` uint8 array), `output_linear` (set for HDR renders, or when `RenderOptions(return_linear=True)`), `mode`, `elapsed_s` and `output_path`. Set `RenderOptions(save=False)` to skip every disk write.
- Pass `cancel=threading.Event()` and set it from another thread to stop a render; the call then raises `CancelledError`.
- `on_progress` receives one line of text per step.
