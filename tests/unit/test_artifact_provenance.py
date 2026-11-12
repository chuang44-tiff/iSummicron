"""0.5.0 provenance: the shipped npz files are exactly the audited build.

FAST, numpy-only: no conda, git, optiland or ``runs/`` (works in public CI).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from summicron.render._artifacts import (
    KERNELS_SHA256,
    LENS_ARTIFACTS,
    default_tensor_path,
)
from summicron.render._bench import TENSOR_MD5, TENSOR_MD5_50

_FILE_MD5 = {"35": TENSOR_MD5, "50": TENSOR_MD5_50}
# built by concatenation so this file carries no guarded token
_TOKENS = ("code" + "v", "code" + " v", "code" + "_v", "code" + "-v",
           "ag_" + "dblgauss", "35" + "f2", "mand" + "ler", "lei" + "tz", "http", "://")


@pytest.mark.parametrize("lens", ["35", "50"])
def test_kernels_sha256_matches_build(lens):
    with np.load(default_tensor_path(lens), allow_pickle=False) as z:
        k = z["kernels"]
    assert k.dtype == np.float32
    assert hashlib.sha256(k.tobytes()).hexdigest() == KERNELS_SHA256[lens]


@pytest.mark.parametrize("lens", ["35", "50"])
def test_file_md5_pinned(lens):
    got = hashlib.md5(default_tensor_path(lens).read_bytes()).hexdigest()
    assert got == _FILE_MD5[lens]


def test_artifact_dir_contains_exactly_two_npz():
    d = default_tensor_path("35").parent
    npz = sorted(p.name for p in d.glob("*.npz"))
    assert npz == sorted(LENS_ARTIFACTS.values())
    extra = {p.name for p in d.iterdir()} - set(npz) - {"__init__.py", "__pycache__"}
    assert not extra, extra


@pytest.mark.parametrize("lens", ["35", "50"])
def test_npz_strings_clean(lens):
    with np.load(default_tensor_path(lens), allow_pickle=False) as z:
        for key in z.files:
            low_key = key.lower()
            assert not any(t in low_key for t in _TOKENS)
            a = z[key]
            if a.dtype.kind in "US":
                text = " ".join(str(x) for x in np.ravel(a)).lower()
                assert not any(t in text for t in _TOKENS), (key, text[:120])
