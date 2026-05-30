"""Monocular depth estimation and depth-based defocus mapping.

Uses Depth Anything V2 Small (fallback from DA3-Base which is not yet on
HuggingFace) via the transformers depth-estimation pipeline. Float32 throughout.
No OpenCV dependency.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"
# NOTE: DA3-Base-hf does not yet exist on HuggingFace as of 2026-04.
# When it ships, update _MODEL_ID to "depth-anything/DA3-Base-hf".

_CACHE_DIR = Path.home() / ".cache" / "summicron" / "depth"
_INFERENCE_LONG_EDGE = 1024


# ---------------------------------------------------------------------------
# S1: Load depth model
# ---------------------------------------------------------------------------

def load_depth_model(device: str = "mps") -> dict:
    """Load depth-estimation pipeline, with MPS -> CPU fallback.

    Parameters
    ----------
    device : str
        Target device ("mps", "cuda", "cpu").

    Returns
    -------
    dict with keys ``model`` (pipeline) and ``device`` (str, actual device used).
    """
    # P1.audit Item 1: actionable error if the `[depth]` extra is missing.
    # transformers also pulls torch transitively; both are gated under the
    # same extra so a single message covers either failure.
    try:
        from transformers import pipeline as hf_pipeline
    except ImportError as e:
        raise ImportError(
            "DA2 monocular depth requires `pip install isummicron[depth]` "
            "(installs torch + transformers; ~2 GB on first invocation). "
            "Use --depth-source lidar to avoid this dependency."
        ) from e

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)

    actual_device = device
    try:
        model = hf_pipeline(
            "depth-estimation",
            model=_MODEL_ID,
            device=actual_device,
            model_kwargs={"cache_dir": str(_CACHE_DIR)},
        )
        # Quick smoke test on MPS to trigger runtime errors early
        if actual_device == "mps":
            _smoke = model(Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)))
    except RuntimeError as exc:
        warnings.warn(
            f"Depth model failed on device={actual_device!r}: {exc}. "
            f"Falling back to CPU."
        )
        actual_device = "cpu"
        model = hf_pipeline(
            "depth-estimation",
            model=_MODEL_ID,
            device=actual_device,
            model_kwargs={"cache_dir": str(_CACHE_DIR)},
        )

    return {"model": model, "device": actual_device}


# ---------------------------------------------------------------------------
# S2: Estimate depth
# ---------------------------------------------------------------------------

def estimate_depth(image_rgb: np.ndarray, model_bundle: dict) -> np.ndarray:
    """Run monocular depth estimation on an RGB image.

    Parameters
    ----------
    image_rgb : (H, W, 3) uint8 or float32 [0,1]
    model_bundle : dict from ``load_depth_model``

    Returns
    -------
    depth : (H, W) float32 in [0, 1] where 0 = near, 1 = far.
    """
    H, W = image_rgb.shape[:2]

    # Convert to PIL
    if image_rgb.dtype != np.uint8:
        pil_img = Image.fromarray(
            (np.clip(image_rgb, 0, 1) * 255).astype(np.uint8), mode="RGB"
        )
    else:
        pil_img = Image.fromarray(image_rgb, mode="RGB")

    # Resize longest edge to _INFERENCE_LONG_EDGE for inference speed
    long_edge = max(H, W)
    if long_edge > _INFERENCE_LONG_EDGE:
        scale = _INFERENCE_LONG_EDGE / long_edge
        new_w = max(1, int(round(W * scale)))
        new_h = max(1, int(round(H * scale)))
        pil_img = pil_img.resize((new_w, new_h), Image.BILINEAR)

    # Run model
    pipe = model_bundle["model"]
    result = pipe(pil_img)

    # Extract predicted_depth tensor -> numpy
    disp = result["predicted_depth"]  # torch.Tensor (h, w)
    disp_np = disp.detach().cpu().numpy().astype(np.float32)

    # Upsample back to original (H, W) with bicubic via PIL (no OpenCV)
    if disp_np.shape != (H, W):
        disp_pil = Image.fromarray(disp_np, mode="F")
        disp_pil = disp_pil.resize((W, H), Image.BICUBIC)
        disp_np = np.asarray(disp_pil, dtype=np.float32)

    # Invert disparity to depth: depth = 1 - norm(disp)
    d_min = disp_np.min()
    d_max = disp_np.max()
    d_range = d_max - d_min
    if d_range < 1e-8:
        # Flat disparity -> flat depth
        depth = np.full((H, W), 0.5, dtype=np.float32)
    else:
        depth = 1.0 - (disp_np - d_min) / d_range

    depth = depth.astype(np.float32)

    # Gate: no NaN/Inf (P1.audit Item 5: assert→raise; depth output is an
    # input to the renderer, classified as ValueError per spec Item 5).
    if np.any(np.isnan(depth)):
        raise ValueError("Gate FAIL: NaN in depth output")
    if np.any(np.isinf(depth)):
        raise ValueError("Gate FAIL: Inf in depth output")

    return depth


# ---------------------------------------------------------------------------
# S2b: Extract LiDAR depth from iPhone HEIC
# ---------------------------------------------------------------------------

def extract_lidar_depth(
    heic_path: str | Path,
    target_hw: tuple[int, int] | None = None,
) -> np.ndarray | None:
    """Decode iPhone HEIC embedded LiDAR/disparity to METERS.

    Returns (H, W) float32 in METERS (typically 0.43..20+ for an iPhone
    LiDAR scene), small values = near, large values = far. Returns ``None``
    if no depth aux image is present in the HEIC.

    Parameters
    ----------
    heic_path : path to .HEIC file.
    target_hw : optional ``(H, W)`` to resize to via bilinear. If None, the
        native depth-aux shape is returned.

    Notes
    -----
    Decoder follows ISO/IEC 23008-12 §6.5.5 "HEIF depth image representation":
    ``representation_type=1`` is uniform normalized disparity.
    ``disp = d_min + (raw/255) * (d_max - d_min)`` in units of 1/m.
    Far-cap: depth_m is clipped to [0.1, 1e3] m before return as defence
    against disp ≈ 0 producing extreme reciprocals downstream.
    """
    import pillow_heif

    heif = pillow_heif.open_heif(str(heic_path), convert_hdr_to_8bit=False)
    depth_imgs = heif.info.get("depth_images") or []
    if not depth_imgs:
        return None

    di = depth_imgs[0]
    raw = np.asarray(di).astype(np.float32)  # (dH, dW) uint8 L-mode -> float32

    meta = di.info.get("metadata", {}) or {}
    d_min = float(meta.get("d_min", 0.0))
    d_max = float(meta.get("d_max", 1.0))
    rep_type = int(meta.get("representation_type", 1))

    if rep_type != 1:
        warnings.warn(
            "Unsupported HEIC depth representation_type="
            f"{rep_type}; falling back to raw-disparity heuristic."
        )
        disp = np.maximum(raw, 1.0)
    else:
        disp = d_min + (raw / 255.0) * (d_max - d_min)

    # disp in 1/m; guard zero before inverting.
    depth_m = 1.0 / np.maximum(disp, 1e-6)

    # Resize to caller's target shape, if any (no OpenCV — PIL bilinear).
    if target_hw is not None:
        tH, tW = int(target_hw[0]), int(target_hw[1])
        if depth_m.shape != (tH, tW):
            pil = Image.fromarray(depth_m.astype(np.float32), mode="F")
            pil = pil.resize((tW, tH), Image.BILINEAR)
            depth_m = np.asarray(pil, dtype=np.float32)

    # Far-cap: clip to [0.1, 1e3] m before returning. Belt-and-suspenders
    # vs. disp ≈ 0 producing 1e6 m sky values.
    depth_m = np.clip(depth_m, 0.1, 1e3).astype(np.float32)

    # Gate: no NaN/Inf (P1.audit Item 5: assert→raise).
    if np.any(np.isnan(depth_m)):
        raise ValueError("Gate FAIL: NaN in LiDAR depth")
    if np.any(np.isinf(depth_m)):
        raise ValueError("Gate FAIL: Inf in LiDAR depth")
    return depth_m


# ---------------------------------------------------------------------------
# S3: Find anchor depth
# ---------------------------------------------------------------------------

def find_anchor_depth(depth_map: np.ndarray, percentile: float = 2.0) -> float:
    """Return the percentile-th value of the depth map (nearest-subject anchor).

    Parameters
    ----------
    depth_map : (H, W) float32 in METERS (or legacy [0,1] for monocular wrapper)
    percentile : float, default 2.0

    Returns
    -------
    anchor : float (meters when depth_map is metric)
    """
    anchor = float(np.percentile(depth_map, percentile))
    if anchor <= 1e-3:
        warnings.warn(
            f"Degenerate anchor depth {anchor:.4f}m <= 1e-3m; "
            f"depth map may be all-zero."
        )
    return anchor


# ---------------------------------------------------------------------------
# S4: Build depth-based defocus map
# ---------------------------------------------------------------------------

def build_depth_defocus_map(
    depth_m: np.ndarray, anchor_m: float, dof_m: float = 0.0
) -> np.ndarray:
    """Reciprocal-difference defocus from a metric depth map and anchor.

    Adapted reciprocal-difference algorithm operating in 1/m. Renderer
    contract preserved: returns (H, W) float32 in [0, 1].

    Parameters
    ----------
    depth_m : (H, W) float32 in meters (or legacy [0,1] — units cancel)
    anchor_m : float, anchor depth (same units as depth_m)
    dof_m : float, default 0.0 — DoF half-width in meters. When > 0, pixels
        within ``|depth - anchor| < dof_m`` are nulled to defocus=0 BEFORE
        p98 normalisation (Item C).

    Returns
    -------
    defocus : (H, W) float32 in [0, 1]
    """
    eps_m = 1e-3
    z = depth_m.astype(np.float32)

    # Flat-scene warning (relative-to-max test)
    z_max = float(z.max())
    z_min = float(z.min())
    ptp = z_max - z_min
    if (ptp / (z_max + 1e-8)) < 0.05:
        warnings.warn(
            f"Flat scene detected (ptp/max = {ptp / (z_max + 1e-8):.4f} < 0.05). "
            f"Returning near-zero defocus."
        )
        raw_flat = np.abs(
            1.0 / np.maximum(z, eps_m) - 1.0 / max(anchor_m, eps_m)
        )
        return (raw_flat * 0.01).astype(np.float32)

    inv_z = 1.0 / np.maximum(z, eps_m)
    inv_anchor = 1.0 / max(anchor_m, eps_m)
    raw = np.abs(inv_z - inv_anchor)

    # Item C: DoF band — null pixels within ±dof_m of anchor_m before p98.
    if dof_m > 0:
        raw = np.where(np.abs(z - anchor_m) < dof_m, 0.0, raw)

    p98 = float(np.percentile(raw, 98))
    if p98 < 1e-8:
        return np.zeros_like(z, dtype=np.float32)

    defocus = np.clip(raw / p98, 0.0, 1.0).astype(np.float32)
    return defocus


def _nan_safe_percentile_anchor(depth_m: np.ndarray, percentile: float) -> float:
    """NaN-tolerant variant of find_anchor_depth used for tap-fallback paths.

    Filters out NaN/Inf before percentile. If no finite values remain,
    returns 0.0 (caller's flat-NaN guard handles this case before us).
    """
    finite = depth_m[np.isfinite(depth_m)]
    if finite.size == 0:
        return 0.0
    anchor = float(np.percentile(finite, percentile))
    if anchor <= 1e-3:
        warnings.warn(
            f"Degenerate fallback anchor {anchor:.4f}m <= 1e-3m; "
            f"depth map may be all-zero."
        )
    return anchor


# ---------------------------------------------------------------------------
# S5: Tap-to-focus shift with wrap-around (Item B)
# ---------------------------------------------------------------------------

def apply_tap_focus_shift(
    depth_m: np.ndarray,
    tap_xy_norm: tuple[float, float] | None,
    focus_target_m: float = 0.7,
    fallback_percentile: float = 2.0,
) -> tuple[np.ndarray, float]:
    """Shift a metric depth map so the tap point lands at ``focus_target_m``,
    wrapping near pixels back to the far side.

    Axis convention
    ---------------
    ``tap_xy_norm`` is **(x, y)** in normalized image coordinates [0, 1],
    where x = column / W and y = row / H. (Standard image-xy, NOT (row, col).)

    Parameters
    ----------
    depth_m : (H, W) float32 in meters.
    tap_xy_norm : tuple ``(x, y)`` in [0, 1], or ``None`` for closest-subject
        fallback (uses ``fallback_percentile`` of depth_m).
    focus_target_m : tensor focus distance (default 0.7 m).
    fallback_percentile : percentile for fallback anchor (default 2.0).

    Returns
    -------
    shifted_depth_m : (H, W) float32 — depth shifted+wrapped, all values
        ``>= focus_target_m``.
    tapped_anchor_m : float — depth value at the tap (or fallback anchor).

    Wrap math
    ---------
    ``shifted = depth_m + (focus_target_m - anchor_m)``
    Where ``shifted < focus_target_m``, fold:
    ``shifted = 2*focus_target_m - shifted``.
    """
    H, W = int(depth_m.shape[0]), int(depth_m.shape[1])

    # All-NaN safety: warn + return zero-shift map at focus_target_m.
    if not np.any(np.isfinite(depth_m)):
        warnings.warn("All depth values non-finite; returning flat focus_target_m map.")
        out = np.full((H, W), focus_target_m, dtype=np.float32)
        return out, focus_target_m

    if tap_xy_norm is None:
        anchor_m = find_anchor_depth(depth_m, fallback_percentile)
    else:
        x_in, y_in = float(tap_xy_norm[0]), float(tap_xy_norm[1])
        # NaN tap-coord guard
        if not (np.isfinite(x_in) and np.isfinite(y_in)):
            warnings.warn(
                f"Non-finite tap coordinates; falling back to p{fallback_percentile} anchor."
            )
            anchor_m = find_anchor_depth(depth_m, fallback_percentile)
        else:
            # Clip OOB with warning.
            if not (0.0 <= x_in <= 1.0 and 0.0 <= y_in <= 1.0):
                warnings.warn(
                    "Tap coords out of [0,1]; clipping. (axis convention: x=col, y=row)"
                )
            x_clip = float(np.clip(x_in, 0.0, 1.0))
            y_clip = float(np.clip(y_in, 0.0, 1.0))
            x_pix = min(int(x_clip * W), W - 1)
            y_pix = min(int(y_clip * H), H - 1)
            # 3x3 median around tap (mirror-clamp at borders).
            y0, y1 = max(0, y_pix - 1), min(H, y_pix + 2)
            x0, x1 = max(0, x_pix - 1), min(W, x_pix + 2)
            patch = depth_m[y0:y1, x0:x1]
            finite = patch[np.isfinite(patch)]
            if finite.size == 0:
                warnings.warn(
                    "Tap landed on all-NaN 3x3 patch; falling back to percentile anchor."
                )
                anchor_m = _nan_safe_percentile_anchor(depth_m, fallback_percentile)
            else:
                anchor_m = float(np.median(finite))
                if anchor_m <= 1e-3:
                    warnings.warn(
                        f"Tap 3x3 median {anchor_m:.4f}m <= 1e-3m; "
                        f"falling back to percentile anchor."
                    )
                    anchor_m = _nan_safe_percentile_anchor(depth_m, fallback_percentile)

    shift = focus_target_m - anchor_m
    shifted = (depth_m.astype(np.float32) + shift).astype(np.float32)

    # Wrap near pixels: shifted = 2*focus_target_m - shifted where shifted < focus_target_m.
    near_mask = shifted < focus_target_m
    shifted[near_mask] = 2.0 * focus_target_m - shifted[near_mask]

    # Sanity clip: never below focus_target_m, never above 1e3.
    shifted = np.clip(shifted, focus_target_m, 1e3).astype(np.float32)
    return shifted, float(anchor_m)


# ---------------------------------------------------------------------------
# S7: Convenience wrapper
# ---------------------------------------------------------------------------

def estimate_and_build_defocus(
    image_rgb: np.ndarray, model_bundle: dict
) -> tuple[np.ndarray, dict]:
    """Chain depth estimation -> anchor -> defocus map.

    NOTE: This wrapper consumes monocular ``estimate_depth`` output, which
    is normalized [0, 1] disparity-derived depth (NOT meters). It is kept
    for backwards compatibility but is NOT on the metric path. The real-data
    metric path goes through ``extract_lidar_depth`` -> ``apply_tap_focus_shift``
    -> ``build_depth_defocus_map``.

    Parameters
    ----------
    image_rgb : (H, W, 3) uint8 or float32
    model_bundle : dict from ``load_depth_model``

    Returns
    -------
    defocus_map : (H, W) float32 in [0, 1]
    info : dict with anchor_depth, depth_range, flat_scene
    """
    depth = estimate_depth(image_rgb, model_bundle)
    anchor = find_anchor_depth(depth)

    ptp = float(depth.max() - depth.min())
    d_max = float(depth.max())
    flat_scene = (ptp / (d_max + 1e-8)) < 0.05

    defocus_map = build_depth_defocus_map(depth, anchor)

    info = {
        "anchor_depth": anchor,
        "depth_range": ptp,
        "flat_scene": flat_scene,
    }
    return defocus_map, info
