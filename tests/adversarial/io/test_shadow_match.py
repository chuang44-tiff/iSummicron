"""P1.13 K11 — adversarial probes for LDR-anchored per-channel band-gated
shadow match.

K11 mechanism (User-pinned 2026-04-30 after K10 post-tonemap "killed
dynamic range"): preview-tonemap-derived per-channel ratio gain + strength
multiplier applied PRE-tonemap on the linear-HDR buffer, smoothstep-
band-gated to LDR shadow band only.

Identity by construction:
- render_chan_mean ~ ldr_chan_mean for every channel + strength=1.0 ->
  gain_c=1.0 (no-op).
- shape mismatch / < 64 valid shadow pixels -> gain=1.0 (degeneracy).
- raw_c * strength < 1.0 -> clamp to 1.0 (one-sided lift).
- raw_c * strength > max_gain -> clamp to max_gain.
"""
from __future__ import annotations

import numpy as np

from summicron.render.convolve import _ldr_shadow_match_gain


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_uniform_rgb(h: int, w: int, gray: float) -> np.ndarray:
    return np.full((h, w, 3), gray, dtype=np.float32)


def _make_rgb(h: int, w: int, r: float, g: float, b: float) -> np.ndarray:
    out = np.empty((h, w, 3), dtype=np.float32)
    out[..., 0] = r
    out[..., 1] = g
    out[..., 2] = b
    return out


# ---------------------------------------------------------------------------
# 1. Identity (off-mode helper-level: matched render+ldr, strength=1)
# ---------------------------------------------------------------------------

def test_shadow_match_off_is_bit_identical():
    """Matched render/LDR with strength=1.0 -> gains_rgb = (1, 1, 1) and
    gain_per_pixel = ones(H, W, 3). Mirrors hdr-source=none working buffer."""
    rng = np.random.default_rng(42)
    img = rng.uniform(0.0, 0.30, size=(64, 64, 3)).astype(np.float32)
    gain_map, gains_rgb = _ldr_shadow_match_gain(
        img, img, shadow_band_hi=0.30, strength=1.0
    )
    np.testing.assert_allclose(
        gain_map, np.ones((64, 64, 3), dtype=np.float32)
    )
    np.testing.assert_allclose(gains_rgb, np.ones(3, dtype=np.float32))


# ---------------------------------------------------------------------------
# 2. Per-channel lift when render is dim in one channel only
# ---------------------------------------------------------------------------

def test_shadow_match_lifts_when_render_dim_per_channel():
    """Render is half-brightness in R only; verify gains_rgb[0] > 1 and
    gains_rgb[1], gains_rgb[2] ~ 1."""
    H, W = 64, 64
    # LDR shadow band: uniform 0.20 across all channels
    ldr = _make_uniform_rgb(H, W, 0.20)
    # Render: R is 0.10 (half brightness), G/B match LDR
    render = _make_rgb(H, W, 0.10, 0.20, 0.20)
    _, gains_rgb = _ldr_shadow_match_gain(
        render, ldr, shadow_band_hi=0.30, strength=1.0
    )
    # R channel: 0.20 / 0.10 = 2.0 -> clamped to 2.0 (within [1, max_gain])
    assert gains_rgb[0] > 1.5, (
        f"expected R-gain > 1.5, got {gains_rgb[0]}"
    )
    np.testing.assert_allclose(gains_rgb[0], 2.0, atol=0.05)
    # G/B channels: matched -> raw=1.0 -> clamped to 1.0
    np.testing.assert_allclose(gains_rgb[1], 1.0, atol=1e-3)
    np.testing.assert_allclose(gains_rgb[2], 1.0, atol=1e-3)


# ---------------------------------------------------------------------------
# 3. Color cast fix — render with blue cast in shadow band
# ---------------------------------------------------------------------------

def test_shadow_match_color_cast_fix():
    """Render has 20% blue cast (R/G dimmer than B) in shadow band; verify
    the per-channel gain corrects the imbalance — output ratios closer to
    LDR than scalar luma would be."""
    H, W = 64, 64
    # LDR: neutral gray at 0.20
    ldr = _make_uniform_rgb(H, W, 0.20)
    # Render: blue cast — R/G crushed by 20%, B at LDR level
    render = _make_rgb(H, W, 0.16, 0.16, 0.20)
    gain_map, gains_rgb = _ldr_shadow_match_gain(
        render, ldr, shadow_band_hi=0.30, strength=1.0
    )
    # R/G should be lifted (raw=0.20/0.16 = 1.25); B unchanged
    np.testing.assert_allclose(gains_rgb[0], 1.25, atol=0.02)
    np.testing.assert_allclose(gains_rgb[1], 1.25, atol=0.02)
    np.testing.assert_allclose(gains_rgb[2], 1.0, atol=1e-3)
    # Apply gain & verify resulting per-channel mean matches LDR more
    # closely than the original render did.
    corrected = render * gain_map
    # Sample shadow region (all pixels here qualify since LDR<0.30)
    for c in range(3):
        np.testing.assert_allclose(
            corrected[..., c].mean(), 0.20, atol=0.005
        )


# ---------------------------------------------------------------------------
# 4. Strength multiplier — overshoot before clamp
# ---------------------------------------------------------------------------

def test_shadow_match_strength_multiplier():
    """raw_gain ~= 1.5 with strength=1.2 produces clamped gain ~= 1.8
    (within [1, max_gain=4])."""
    H, W = 64, 64
    ldr = _make_uniform_rgb(H, W, 0.15)
    render = _make_uniform_rgb(H, W, 0.10)  # raw = 0.15/0.10 = 1.5
    _, gains_rgb = _ldr_shadow_match_gain(
        render, ldr, shadow_band_hi=0.30, max_gain=4.0, strength=1.2
    )
    # Expected: 1.5 * 1.2 = 1.8 (no clamp)
    for c in range(3):
        np.testing.assert_allclose(gains_rgb[c], 1.8, atol=0.02)


# ---------------------------------------------------------------------------
# 5. One-sided clamp — render brighter than LDR -> identity (no darken)
# ---------------------------------------------------------------------------

def test_shadow_match_one_sided_clamp_preserves_identity():
    """Render is BRIGHTER than LDR (raw_gain < 1.0 like IMG_6651 with raw
    ~0.76); verify clamped to 1.0 (no darkening)."""
    H, W = 64, 64
    ldr = _make_uniform_rgb(H, W, 0.15)
    # Render brighter: raw = 0.15 / 0.20 = 0.75 (< 1)
    render = _make_uniform_rgb(H, W, 0.20)
    gain_map, gains_rgb = _ldr_shadow_match_gain(
        render, ldr, shadow_band_hi=0.30, strength=1.0
    )
    # All channels should clamp to 1.0
    np.testing.assert_allclose(gains_rgb, np.ones(3, dtype=np.float32))
    np.testing.assert_allclose(
        gain_map, np.ones((H, W, 3), dtype=np.float32)
    )


# ---------------------------------------------------------------------------
# 6. max_gain clamp — pathological dim render
# ---------------------------------------------------------------------------

def test_shadow_match_max_gain_clamp():
    """Synthetic raw_gain = 5 (LDR 0.25, render 0.05); verify clamped to
    max_gain=4.0."""
    H, W = 64, 64
    ldr = _make_uniform_rgb(H, W, 0.25)
    render = _make_uniform_rgb(H, W, 0.05)  # raw = 5.0
    _, gains_rgb = _ldr_shadow_match_gain(
        render, ldr, shadow_band_hi=0.30, max_gain=4.0, strength=1.0
    )
    for c in range(3):
        np.testing.assert_allclose(gains_rgb[c], 4.0, atol=1e-3)


# ---------------------------------------------------------------------------
# 7. Degeneracy guard — < 64 shadow pixels -> identity
# ---------------------------------------------------------------------------

def test_shadow_match_degeneracy_guard():
    """All-bright scene (no shadow band pixels) -> gain=1.0 everywhere."""
    H, W = 32, 32
    ldr = _make_uniform_rgb(H, W, 0.95)  # all above shadow_band_hi=0.30
    render = _make_uniform_rgb(H, W, 0.10)
    gain_map, gains_rgb = _ldr_shadow_match_gain(
        render, ldr, shadow_band_hi=0.30, strength=1.0
    )
    np.testing.assert_allclose(gains_rgb, np.ones(3, dtype=np.float32))
    np.testing.assert_allclose(
        gain_map, np.ones((H, W, 3), dtype=np.float32)
    )


# ---------------------------------------------------------------------------
# 8. Shape-mismatch guard
# ---------------------------------------------------------------------------

def test_shadow_match_shape_mismatch_guard():
    """Mismatched render/LDR shapes -> identity gain (no crash)."""
    render = _make_uniform_rgb(64, 64, 0.10)
    ldr = _make_uniform_rgb(32, 32, 0.20)
    gain_map, gains_rgb = _ldr_shadow_match_gain(render, ldr)
    np.testing.assert_allclose(gains_rgb, np.ones(3, dtype=np.float32))
    np.testing.assert_allclose(
        gain_map, np.ones((64, 64, 3), dtype=np.float32)
    )
