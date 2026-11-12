"""S22 S1: synthetic family build + user-cache (`summicron.render._family`)."""
from __future__ import annotations

import hashlib
import os
import stat
from pathlib import Path

import numpy as np
import pytest

from summicron.psf_model import synth_family
from summicron.render import _family
from summicron.render._artifacts import default_tensor_path
from summicron.render._bench import TENSOR_MD5


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    d = tmp_path_factory.mktemp("fam")
    t, p, hit = _family.load_family_tensor(17.5, cache_dir=d)
    return d, t, p, hit


def _md5(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def test_build_then_hit(built):
    d, t, p, hit = built
    assert hit is False and p.parent == d and p.name.startswith("synth_")
    t2, p2, hit2 = _family.load_family_tensor(17.5, cache_dir=d)
    assert hit2 is True and p2 == p
    assert np.array_equal(t["kernels"], t2["kernels"])


def test_second_call_does_not_rebuild(built, monkeypatch):
    d, *_ = built

    def boom(*a, **k):
        raise AssertionError("builder called on cache hit")

    monkeypatch.setattr(synth_family, "build_synthetic_tensor", boom)
    _family.load_family_tensor(17.5, cache_dir=d)


def test_metadata_and_validation(built):
    _, t, *_ = built
    assert t["synth_family"] == "synthetic_disc_v0"
    assert t["kernels"].shape == (12, 16, 183, 183)
    assert t["D"] == 16 and abs(t["object_distances_m"][0] - 0.7) < 1e-12
    with np.load(str(built[2]), allow_pickle=False) as z:
        assert str(z["schema_version"]) == "1.0"
        assert int(z["synth_builder_version"]) == synth_family.BUILDER_VERSION


def test_no_legacy_schema_warning(built, recwarn):
    from summicron.render.convolve import load_tensor

    load_tensor(built[2])
    assert not [w for w in recwarn if "schema_version" in str(w.message)]


def test_kernels_bit_equal_two_builds(built, tmp_path):
    t2, *_ = _family.load_family_tensor(17.5, cache_dir=tmp_path)
    assert np.array_equal(built[1]["kernels"], t2["kernels"])


def test_key_sensitive_to_A_and_builder_version(monkeypatch):
    k1, k2 = _family.family_key(17.5), _family.family_key(25.0)
    assert k1 != k2
    monkeypatch.setattr(synth_family, "BUILDER_VERSION", 99)
    assert _family.family_key(17.5) != k1


def test_truncated_file_rebuilt(built, tmp_path):
    t, p, _ = built[1:]
    bad = tmp_path / p.name
    bad.write_bytes(p.read_bytes()[:1000])
    t2, p2, hit = _family.load_family_tensor(17.5, cache_dir=tmp_path)
    assert hit is False and p2 == bad
    assert np.array_equal(t["kernels"], t2["kernels"])
    assert _family.load_family_tensor(17.5, cache_dir=tmp_path)[2] is True


def test_foreign_key_and_flipped_bytes_rebuilt(built, tmp_path):
    _, t, p, _ = built
    with np.load(str(p), allow_pickle=False) as z:
        payload = {k: z[k] for k in z.files}
    # right key, flipped kernel bytes -> hash mismatch
    k = payload["kernels"].copy()
    k[3, 5, 90, 90] += np.float32(0.5)
    bad = dict(payload, kernels=k)
    np.savez(tmp_path / p.name, **bad)
    t2, _, hit = _family.load_family_tensor(17.5, cache_dir=tmp_path)
    assert hit is False and np.array_equal(t2["kernels"], t["kernels"])
    # foreign key
    payload["synth_key"] = np.array("0" * 16)
    np.savez(tmp_path / p.name, **payload)
    assert _family.load_family_tensor(17.5, cache_dir=tmp_path)[2] is False


def test_atomic_write_leaves_no_tmp(built):
    assert not list(built[0].glob("*.tmp*"))


def test_unwritable_dir_falls_back(tmp_path):
    ro = tmp_path / "ro"
    ro.mkdir()
    ro.chmod(stat.S_IRUSR | stat.S_IXUSR)
    try:
        if os.access(ro, os.W_OK):
            pytest.skip("cannot make dir read-only (root?)")
        lines = []
        t, p, hit = _family.load_family_tensor(
            17.5, cache_dir=ro / "sub", on_progress=lines.append)
        assert hit is False and t["D"] == 16
        assert any("not writable" in l for l in lines)
    finally:
        ro.chmod(stat.S_IRWXU)


def test_cache_dir_resolution(monkeypatch, tmp_path):
    monkeypatch.setenv("ISUMMICRON_CACHE_DIR", str(tmp_path / "a b"))
    assert _family.default_cache_dir() == tmp_path / "a b" / "synth"
    monkeypatch.delenv("ISUMMICRON_CACHE_DIR")
    assert _family.default_cache_dir().name == "synth"


def test_cache_dir_is_a_file_falls_back(tmp_path):
    f = tmp_path / "file"
    f.write_text("x")
    t, p, hit = _family.load_family_tensor(17.5, cache_dir=f)
    assert hit is False and t["D"] == 16


def test_shipped_artifact_untouched_and_key_none(built):
    assert _md5(default_tensor_path()) == TENSOR_MD5
    from summicron.render.convolve import load_tensor

    assert load_tensor(default_tensor_path())["synth_family"] is None
