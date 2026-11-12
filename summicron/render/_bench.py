"""Canonical bench spec — PRIVATE, DEV-ONLY (HYG-0).

Single source of truth for the canonical bench argv and its md5 pins.
Consumed by ``summicron.tools.anchor_picker`` (emitted command template),
``tests/unit/test_back_compat_canonical.py`` (G1 gate) and mirrored by
``runs/vibe_bench/README.md``. It ships in the wheel only because
``anchor_picker`` imports it; it is NOT a public API and is intentionally
not re-exported from ``summicron.render``.

``CANONICAL_FLAGS`` omits ``--image``, ``--focus-xy``, ``--output`` and
``--tensor`` (the renderer defaults to the wheel-shipped tensor).
Stdlib-only: no imports.
"""

CANONICAL_FLAGS = (
    "--depth-source", "lidar",
    "--depth-scale", "10",
    "--downres", "2048",
    "--psf-norm", "l1",
    "--hdr-source", "macos",
)

G1_IMAGE = "iphoneImage/IMG_6915.HEIC"
G1_FOCUS_XY = (0.40, 0.45)

# OFF (default) canonical render md5 — G1 SHOWSTOPPER pin.
# Re-pinned by Q-b-flip (exact d0 delta became the default; Q-b-legacy-removal
# then deleted the flag, so this is the only behaviour).
G1_MD5 = "a106801a97a05ee7d5033e2b8d647f11"
# Same argv + ``--occlusion-compositing on``.
ON_MD5 = "6cebff5403f7f0fdcda109cb8bbff35a"
# md5 of summicron/_artifacts/codev_render_v1.npz (default_tensor_path()).
TENSOR_MD5 = "e6f501546cb77ba40ad904dbd4ba9369"
