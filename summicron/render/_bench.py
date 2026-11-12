"""Canonical bench spec — PRIVATE, DEV-ONLY (HYG-0).

Single source of truth for the canonical bench argv and its md5 pins.
Consumed by ``summicron.tools.anchor_picker`` (emitted command template),
``tests/unit/test_back_compat_canonical.py`` (G1 gate) and mirrored by
``runs/vibe_bench/README.md``. It ships in the wheel only because
``anchor_picker`` imports it; it is NOT a public API and is intentionally
not re-exported from ``summicron.render``.

Re-pinned by the Run 6 default flip: ``CANONICAL_FLAGS`` now rides the
defaults (the approved look); ``CANONICAL_FLAGS_OLD`` + ``*_OLD`` pins keep
the pre-flip look reproducible.

``CANONICAL_FLAGS`` omits ``--image``, ``--focus-xy``, ``--output`` and
``--tensor`` (the renderer defaults to the wheel-shipped tensor).
Stdlib-only: no imports.
"""

CANONICAL_FLAGS = (
    "--depth-source", "lidar",
    "--downres", "2048",
    "--psf-norm", "l1",
    "--hdr-source", "macos",
)

# Pre-Run-6 look, still reachable (and pinned) via explicit flags.
OLD_LOOK_FLAGS = (
    "--depth-guided", "off",
    "--depth-mapping", "linear",
    "--tonemap", "ci",
    "--bright-pass-thresh", "1.0",
)
CANONICAL_FLAGS_OLD = (
    "--depth-source", "lidar",
    "--depth-scale", "10",
    "--downres", "2048",
    "--psf-norm", "l1",
    "--hdr-source", "macos",
    *OLD_LOOK_FLAGS,
)
G1_MD5_OLD = "a106801a97a05ee7d5033e2b8d647f11"  # live gate (explicit old flags)
ON_MD5_OLD = "6cebff5403f7f0fdcda109cb8bbff35a"

G1_IMAGE = "iphoneImage/IMG_6915.HEIC"
G1_FOCUS_XY = (0.40, 0.45)

# OFF (default) canonical render md5 — G1 SHOWSTOPPER pin.
# Re-pinned by Run 6 (default flip: guided depth + reciprocal-depth mapping + input-lut tone;
# macOS 27.2 build 26B123, pyobjc-core 12.2). The previous pin (Q-b-flip,
# ``a106801a…``) is ``G1_MD5_OLD`` above, reproduced by ``CANONICAL_FLAGS_OLD``.
G1_MD5 = "f5d86ac424b612770d79a0d1694736b7"
# Same argv + ``--occlusion-compositing on``.
ON_MD5 = "0def67a51677aa1394307aa974fcb3fd"
# md5 of summicron/_artifacts/codev_render_v1.npz (default_tensor_path()).
TENSOR_MD5 = "e6f501546cb77ba40ad904dbd4ba9369"
