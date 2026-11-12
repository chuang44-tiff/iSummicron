"""Adversarial test: isolated HDR spike on LDR background must survive downres."""
import numpy as np

from summicron.render.convolve import downres_array


def test_downres_array_hdr_isolated_spike_survives():
    """A single 100x100 HDR-max=32 patch on LDR background must survive downres."""
    img = np.zeros((1024, 1536, 3), dtype=np.float32)
    img[400:500, 400:500, :] = 32.0  # isolated HDR spike
    out = downres_array(img, target_long_edge=512)
    assert out.max() > 10.0, f"HDR spike lost: max={out.max():.4f}"
