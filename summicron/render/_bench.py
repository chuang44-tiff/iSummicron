"""Canonical bench spec — PRIVATE, DEV-ONLY (HYG-0).

Single source of truth for the canonical bench argv and its md5 pins.
Consumed by ``summicron.tools.anchor_picker`` (emitted command template),
``tests/unit/test_back_compat_canonical.py`` (G1 gate) and mirrored by
``runs/vibe_bench/README.md``. It ships in the wheel only because
``anchor_picker`` imports it; it is NOT a public API and is intentionally
not re-exported from ``summicron.render``.

``CANONICAL_FLAGS`` omits ``--image``, ``--focus-xy``, ``--output`` and
``--tensor`` / ``--lens`` (the renderer defaults to the wheel-shipped 35 mm
f/2 tensor). Stdlib-only: no imports.
"""

CANONICAL_FLAGS = (
    "--depth-source", "lidar",
    "--downres", "2048",
    "--hdr-source", "macos",
)

G1_IMAGE = "iphoneImage/IMG_6915.HEIC"
G1_FOCUS_XY = (0.40, 0.45)

# OFF (default) canonical render md5 — G1 SHOWSTOPPER pin (35 mm f/2 default).
G1_MD5 = "a15314d23dfe09070667a15f66a362d9"
# Same argv + ``--occlusion-compositing on``.
ON_MD5 = "3659beff1f331b6d041ca50d90421718"
# Same argv + ``--lens 50`` (OFF).
LENS50_MD5 = "b2eebd9d0605cb1a1e1e617b12f41d0a"
# md5 of the shipped tensor files (default_tensor_path("35") / ("50")).
TENSOR_MD5 = "894f858c54e0cad45eedcddab47be6d7"
TENSOR_MD5_50 = "10824ae374e7ab57e651d3d770a76b6a"
# Non-default shipped artifacts: file name -> md5 (read by the release gate).
EXTRA_ARTIFACT_MD5 = {"psf_lens50_doublegauss_fu0.4.npz": "10824ae374e7ab57e651d3d770a76b6a"}
