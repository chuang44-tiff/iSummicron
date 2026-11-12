"""Section 22 S0: synthetic disc family (summicron.psf_model.synth_family)."""
from __future__ import annotations

import numpy as np
import pytest

from summicron.psf_model import synth_family as sf

SHIPPED = sf.DEFAULT_SOURCE
PS = 2048 / 5712


@pytest.fixture(scope="module")
def shipped():
    d = np.load(SHIPPED, allow_pickle=False)
    return {k: d[k] for k in d.files}


@pytest.fixture(scope="module")
def crisp():
    return sf.build_synthetic_tensor(aperture_mm=17.5, psf_scale=0.25, rim_beta=0.3)


@pytest.fixture(scope="module")
def soft():
    return sf.build_synthetic_tensor(aperture_mm=17.5, psf_scale=1.0, rim_beta=0.3)


def test_schema_parity_with_shipped(crisp, shipped, tmp_path):
    p = tmp_path / "fam.npz"
    sf.build_synthetic_tensor(out_path=p)
    d = np.load(p, allow_pickle=False)  # no pickled objects
    for k in shipped:
        assert k in d.files, k
    extra = {k for k in d.files if k not in shipped}
    assert extra and all(k.startswith("synth_") for k in extra)
    R, D = d["kernels"].shape[:2]
    assert R == 12 and D == 16
    assert d["kernels"].shape[2:] == shipped["kernels"].shape[2:]
    assert d["kernels"].dtype == np.float32
    assert d["pixel_pitch_um"].shape == (R, D)
    assert d["full_frame_dim_px"].shape == (R, D, 2)
    assert len(d["object_distances_m"]) == D == len(d["defocus_ratios"])
    assert np.all(np.diff(d["object_distances_m"]) > 0)
    assert np.all(np.diff(d["defocus_ratios"]) > 0)
    assert float(d["object_distances_m"][0]) == pytest.approx(float(d["focus_distance_m"]))
    assert int(d["corpus_ref_long_edge"]) == 5712
    assert bool(d["delta_guard_applied"]) and not bool(d["otf_norm_applied"])
    np.testing.assert_array_equal(d["radial_edges"], shipped["radial_edges"])


def test_geometric11_axis_matches_shipped_od(shipped):
    p = sf.build_synthetic_tensor(axis="geometric11")
    assert p["kernels"].shape[:2] == (12, 11)
    np.testing.assert_allclose(p["object_distances_m"], shipped["object_distances_m"])


def test_loads_through_load_tensor_with_d16(tmp_path):
    from summicron.render.convolve import load_tensor

    pth = tmp_path / "fam.npz"
    sf.build_synthetic_tensor(out_path=pth)
    with pytest.warns(UserWarning, match="schema_version"):
        t = load_tensor(pth)
    assert t["D"] == 16 and t["R"] == 12


def test_l1_and_exact_delta(crisp):
    K = crisp["kernels"]
    assert np.abs(K.astype(np.float64).sum(axis=(2, 3)) - 1.0).max() < 1e-6
    assert K.min() >= 0.0
    c = K.shape[2] // 2
    for r in range(K.shape[0]):
        assert K[r, 0, c, c] == 1.0
        assert K[r, 0].sum() == 1.0


def test_coc_monotone_in_defocus(crisp):
    dp = 1.0 / 0.7 - 1.0 / crisp["object_distances_m"]
    coc = crisp["synth_coc_px_ref"]
    assert coc[0] == 0.0 and np.all(np.diff(coc) > 0)
    # linear in dp
    np.testing.assert_allclose(coc[1:] / dp[1:], coc[-1] / dp[-1], rtol=1e-9)
    # measured r90 grows monotonically too (r-bin 0)
    r90 = [
        sf.crisp_metrics(sf.render_scale_kernel(crisp["kernels"][0, d], PS, 67))["r90"]
        for d in range(1, 16)
    ]
    assert np.all(np.diff(r90) > 0)


def test_coc_formula_values():
    # 1.365 * A * dp at 2048
    assert float(sf.coc_diameter_px(17.5, 1.0, 2048)) == pytest.approx(1.365 * 17.5, rel=2e-3)


@pytest.mark.parametrize("coc", [20.0, 45.0, 91.9])
def test_disc_radius_within_1px_of_formula(coc):
    k = sf.disc_kernel(coc, 183)
    # half-max radius from the azimuthal profile
    yy, xx = np.mgrid[0:183, 0:183]
    rr = np.hypot(yy - 91, xx - 91)
    prof = np.bincount(np.floor(rr).astype(int).ravel(), weights=k.ravel()) / np.maximum(
        np.bincount(np.floor(rr).astype(int).ravel()), 1
    )
    half = prof.max() / 2
    r_half = int(np.argmax(prof < half))
    assert abs(r_half - coc / 2) <= 1.0
    # and area-equivalent radius from the energy-weighted support
    area = (k > 0.5 * k.max()).sum()
    assert abs(np.sqrt(area / np.pi) - coc / 2) <= 1.0


def test_family_disc_radius_matches_formula_beta0():
    p = sf.build_synthetic_tensor(rim_beta=0.0, psf_scale=0.25)
    for d in (4, 9, 15):
        k = p["kernels"][0, d]
        coc = p["synth_coc_px_ref"][d]
        area = (k > 0.5 * k.max()).sum()
        assert abs(np.sqrt(area / np.pi) - coc / 2) <= 1.0


def test_crispness_passes_on_recommended_arm(crisp):
    t = sf.gate_table(crisp["kernels"])
    g = t["gated"]
    assert g.sum() >= 100
    assert t["crisp_ok"].sum() / g.sum() >= 0.95
    assert np.nanmin(t["flatness"][g]) >= 0.84  # threshold 0.85, borderline pixel-quantisation
    assert np.nanmin(t["rim"][g]) >= 0.8
    assert np.nanmax(t["haze"][g]) <= 0.003


def test_crispness_fails_on_soft_control(soft):
    """Negative control: PSF at scale 1.0 must be rejected by the same gates."""
    t = sf.gate_table(soft["kernels"])
    g = t["gated"]
    assert t["crisp_ok"].sum() == 0
    assert np.nanmin(t["flatness"][g]) < 0.6
    assert np.nanmin(t["rim"][g]) < 0.4
    assert np.nanmax(t["haze"][g]) > 0.01


def test_shipped_tensor_fails_crisp_gate(shipped):
    t = sf.gate_table(shipped["kernels"])
    assert t["crisp_ok"].sum() == 0


def test_zoom2x_control_is_hazy():
    z = sf.build_zoom_control(zoom=2.0)
    t = sf.gate_table(z["kernels"])
    assert t["crisp_ok"].sum() == 0
    assert z["kernels"].shape[:2] == (12, 11)
    assert np.abs(z["kernels"].astype(np.float64).sum(axis=(2, 3)) - 1).max() < 1e-6


def test_diopter16_adjacent_step_le_4px(crisp):
    coc_2048 = crisp["synth_coc_px_ref"] * PS
    assert np.diff(coc_2048).max() <= 4.0
    assert coc_2048[1] <= 2.5  # node 1 stays near-focus
    # measured (zoom-tail r90-equivalent) step also within the budget
    eff = [0.0] + [
        2 * sf.crisp_metrics(sf.render_scale_kernel(crisp["kernels"][0, d], PS, 67))["r90"]
        / np.sqrt(0.9)
        for d in range(1, 16)
    ]
    assert np.diff(eff).max() <= 4.0


def test_geometric11_step_exceeds_budget():
    """Why diopter16: the shipped geometric axis has ~9.7 px first step at A=17.5."""
    p = sf.build_synthetic_tensor(axis="geometric11")
    assert np.diff(p["synth_coc_px_ref"] * PS).max() > 4.0


def test_deterministic(tmp_path):
    a, b = tmp_path / "a.npz", tmp_path / "b.npz"
    sf.build_synthetic_tensor(out_path=a)
    sf.build_synthetic_tensor(out_path=b)
    assert np.array_equal(np.load(a)["kernels"], np.load(b)["kernels"])


def test_cat_eye_area_not_larger_than_round():
    r = sf.disc_kernel(40.0, 91)
    c = sf.disc_kernel(40.0, 91, cat_eye=0.5)
    assert (c > 0.5 * c.max()).sum() < (r > 0.5 * r.max()).sum()
    yy, xx = np.mgrid[0:91, 0:91]
    assert abs((c * yy).sum() - 45) < 0.5  # centroid stays central


@pytest.mark.parametrize("bad", [dict(axis="nope"), dict(aperture_mm=0.0), dict(psf_scale=-1.0)])
def test_invalid_params_raise(bad):
    with pytest.raises(ValueError):
        sf.build_synthetic_tensor(**bad)
