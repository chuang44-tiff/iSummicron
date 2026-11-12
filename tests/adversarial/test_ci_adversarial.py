"""CI round adversarial tests (doc/plan/phase2/section12_ci/locked-spec-CI.md).

Wrong-polarity requirement: the codev raw-file skip guards must SKIP when
``runs/ag_dblgauss/`` is absent AND RUN when raw files are present. Each
polarity is proven in a child pytest whose ``DEFAULT_INPUT_DIR`` is redirected
(by a tiny plugin, before the codev test modules import it) to either a
nonexistent dir or a synthetic raw-file dir. No real data needed, so these
run in a public clone too (no reads of doc/, iphoneImage/ or runs/).

Also pins: the ``hdr-macos`` extra lists only real distributions, the ruff
config is pinned + explicit, and ci.yml is dispatchable (never scheduled) running FAST.
"""
from __future__ import annotations

import os
import shlex
import subprocess
import sys
import textwrap
import tomllib
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = ROOT / "pyproject.toml"
CI_YML = ROOT / ".github" / "workflows" / "ci.yml"

UNIT = "tests/unit/test_codev_to_render.py"
ADV = "tests/adversarial/test_codev_to_render_adversarial.py"
P193 = "tests/adversarial/test_p1_9_3_adversarial.py"

# Node ids that need the raw CodeV files (skip when absent, run when present).
RAW_NODES = [
    f"{UNIT}::TestConvertShape::test_kernels_dtype",
    f"{UNIT}::TestConvertShape::test_output_shape",
    f"{ADV}::TestEnergyConservation::test_no_nan_in_output",
    f"{P193}::TestCodevToRenderFileDrift::test_convert_glob_pattern_restrictive",
]
# Synthetic tmp_path test that must run regardless (module stays importable).
SYNTH_NODE = f"{ADV}::TestMissingOrMalformedFiles::test_empty_directory_raises"

_PLUGIN = textwrap.dedent(
    """
    import os
    from pathlib import Path

    def pytest_configure(config):
        import summicron.psf_model.codev_to_render as m
        m.DEFAULT_INPUT_DIR = Path(os.environ["CI_ADV_CODEV_DIR"])
        _real = m.convert
        out_dir = Path(os.environ["CI_ADV_OUT_DIR"])

        def _convert(input_dir=None, out_path=None, **kw):
            # Never write into ROOT/runs/ (absent in CI / public clone).
            name = Path(out_path).name if out_path else "render.npz"
            return _real(input_dir=input_dir, out_path=out_dir / name, **kw)

        m.convert = _convert
    """
)


def _write_fake_raw_dir(d: Path, R: int = 12, D: int = 11, K: int = 32) -> None:
    """Synthetic raw CodeV dir (same recipe as the adversarial fake_npz_dir)."""
    d.mkdir(parents=True)
    rng = np.random.default_rng(0)
    h_max_mm = 21.6
    meta = {
        "focus_distance_m": np.float64(0.7),
        "h_max_mm": np.float64(h_max_mm),
        "wavelengths_nm": np.array([470.0, 550.0, 660.0]),
        "object_distances_m": np.geomspace(0.7, 20.0, D),
        "pixel_pitch_um": np.full((R, 3), 5.0, dtype=np.float64),
        "radial_bin_centers_mm": h_max_mm * np.sqrt(np.linspace(0.0, 0.954, R)),
    }
    for di in range(D):
        data = {
            "distance_index": np.array(di),
            "radial_bin_edges_hmax": np.linspace(0.0, 1.0, R),
            **meta,
        }
        for ri in range(R):
            for wi in range(3):
                data[f"psf_r{ri:02d}_w{wi}"] = (
                    rng.random((K, K)).astype(np.float32) + 0.01
                )
        np.savez(d / f"ag_dblgauss_f2.0_focus0.70m_d{di:02d}_fake.npz", **data)
    # Non-distance sibling the glob must exclude (glob-restrictive test).
    np.savez(d / "ag_dblgauss_f2.0_render.npz", x=np.zeros(1))


def _run_child(tmp_path: Path, codev_dir: Path, nodes: list[str]) -> dict[str, str]:
    """Run ``nodes`` in a child pytest with DEFAULT_INPUT_DIR -> codev_dir.

    Returns {test name: "passed" | "skipped" | "failed"}.
    """
    plug_dir = tmp_path / "plug"
    plug_dir.mkdir(exist_ok=True)
    (plug_dir / "_ci_adv_plug.py").write_text(_PLUGIN)
    out_dir = tmp_path / "out"
    out_dir.mkdir(exist_ok=True)
    xml = tmp_path / f"junit_{codev_dir.name}.xml"
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(plug_dir), env["PYTHONPATH"]] if env.get("PYTHONPATH") else [str(plug_dir)]
    )
    env["CI_ADV_CODEV_DIR"] = str(codev_dir)
    env["CI_ADV_OUT_DIR"] = str(out_dir)
    cmd = [
        sys.executable, "-m", "pytest", *nodes,
        "-p", "_ci_adv_plug", "-p", "no:xdist", "-p", "no:cacheprovider",
        "-q", f"--junitxml={xml}",
    ]
    proc = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True,
                          timeout=600)
    assert xml.exists(), f"child pytest produced no junit xml:\n{proc.stdout}\n{proc.stderr}"
    outcomes: dict[str, str] = {}
    for case in ET.parse(xml).getroot().iter("testcase"):
        kids = {c.tag for c in case}
        if "skipped" in kids:
            o = "skipped"
        elif kids & {"failure", "error"}:
            o = "failed"
        else:
            o = "passed"
        outcomes[f"{case.get('classname')}::{case.get('name')}"] = o
    assert len(outcomes) == len(nodes), (
        f"expected {len(nodes)} outcomes, got {outcomes}\n{proc.stdout[-3000:]}"
    )
    return outcomes


def _by_name(outcomes: dict[str, str], test_name: str) -> str:
    hits = [o for k, o in outcomes.items() if k.endswith("::" + test_name)]
    assert len(hits) == 1, (test_name, outcomes)
    return hits[0]


# ── (a) codev skip guards: wrong polarity both ways ─────────────────────────

class TestCodevSkipGuardPolarity:
    def test_raw_absent_skips_raw_tests_but_synthetic_runs(self, tmp_path):
        missing = tmp_path / "no_such_codev_dir"
        out = _run_child(tmp_path, missing, RAW_NODES + [SYNTH_NODE])
        for node in RAW_NODES:
            name = node.rsplit("::", 1)[1]
            assert _by_name(out, name) == "skipped", (node, out)
        # Module import fallback (shipped tensor R/D) keeps synthetic tests live.
        assert _by_name(out, "test_empty_directory_raises") == "passed", out

    def test_raw_present_runs_raw_tests(self, tmp_path):
        fake = tmp_path / "fake_codev"
        _write_fake_raw_dir(fake)
        out = _run_child(tmp_path, fake, RAW_NODES + [SYNTH_NODE])
        # "Not skipped" is the polarity proof; the robust checks must pass.
        for node in RAW_NODES:
            name = node.rsplit("::", 1)[1]
            assert _by_name(out, name) != "skipped", (node, out)
        assert _by_name(out, "test_kernels_dtype") == "passed", out
        assert _by_name(out, "test_output_shape") == "passed", out
        assert _by_name(out, "test_no_nan_in_output") == "passed", out
        assert _by_name(out, "test_convert_glob_pattern_restrictive") == "passed", out

    def test_fallback_RD_matches_raw_RD_when_raw_present(self):
        """The absent-path fallback (shipped tensor's (R, D)) is only correct
        while it equals the (R, D) derived from the raw files."""
        from summicron.psf_model.codev_to_render import DEFAULT_INPUT_DIR
        from summicron.render._artifacts import default_tensor_path

        files = sorted(DEFAULT_INPUT_DIR.glob("ag_dblgauss_f2.0_focus*_d*.npz"))
        if not files:
            pytest.skip(f"raw CodeV file absent: {DEFAULT_INPUT_DIR}")
        with np.load(str(files[0])) as f0:
            raw_rd = (len(f0["radial_bin_edges_hmax"]), len(files))
        with np.load(str(default_tensor_path())) as ft:
            shipped_rd = tuple(int(x) for x in ft["kernels"].shape[:2])
        assert shipped_rd == raw_rd


# ── (b) hdr-macos extra lists only real distributions ───────────────────────

def _pyproject() -> dict:
    return tomllib.loads(PYPROJECT.read_text())


def _dist_name(req: str) -> str:
    base = req.split(";", 1)[0].strip()
    for sep in ("[", "=", "<", ">", "!", "~", " "):
        base = base.split(sep, 1)[0]
    return base.strip().lower()


class TestPyprojectExtras:
    def test_hdr_macos_extra_has_quartz_not_coreimage(self):
        reqs = _pyproject()["project"]["optional-dependencies"]["hdr-macos"]
        names = {_dist_name(r) for r in reqs}
        assert "pyobjc-framework-coreimage" not in names, reqs
        assert "pyobjc-framework-quartz" in names, reqs
        # Darwin-gated so a Linux install of the extra is a no-op, not a failure.
        assert all("sys_platform == 'darwin'" in r for r in reqs), reqs

    def test_hdr_hint_does_not_name_nonexistent_package(self):
        src = (ROOT / "summicron" / "io" / "hdr_load_macos.py").read_text()
        assert "pyobjc-framework-CoreImage" not in src
        assert "Quartz + CoreImage" not in src

    def test_ruff_pinned_exactly_and_select_explicit(self):
        pp = _pyproject()
        dev = pp["project"]["optional-dependencies"]["dev"]
        ruff = [r for r in dev if _dist_name(r) == "ruff"]
        assert len(ruff) == 1 and "==" in ruff[0] and ">" not in ruff[0], dev
        select = pp["tool"]["ruff"]["lint"].get("select")
        assert select, "explicit [tool.ruff.lint] select required (default drifts)"
        assert set(select) == {"E4", "E7", "E9", "F"}, select


# ── (d) ci.yml: dispatch-only + FAST command ────────────────────────────────

def _ci() -> dict:
    yaml = pytest.importorskip("yaml")
    if not CI_YML.exists():
        pytest.skip(f"{CI_YML} absent")
    return yaml.safe_load(CI_YML.read_text())


class TestCiWorkflow:
    def test_triggers_dispatchable_never_scheduled(self):
        doc = _ci()
        on = doc.get("on", doc.get(True))  # PyYAML 1.1 loads `on:` as True
        keys = set(on) if isinstance(on, dict) else {on} if isinstance(on, str) else set(on)
        assert "workflow_dispatch" in keys, keys
        assert "schedule" not in keys, keys
        assert keys <= {"push", "pull_request", "workflow_dispatch"}, keys
        assert doc.get("permissions") == {"contents": "read"}, doc.get("permissions")

    @pytest.mark.skipif(
        not (ROOT / "release" / "public-overlay").is_dir(),
        reason="dev-only billing guard: release/public-overlay/ absent (shipped tree)",
    )
    def test_dev_ci_is_workflow_dispatch_only(self):
        """The DEV ci.yml stays manual (billed macOS minutes).

        Runs only where release/public-overlay/ exists (dev); the shipped
        public tree has no overlay dir and carries the public ci.yml.
        """
        doc = _ci()
        on = doc.get("on", doc.get(True))
        keys = set(on) if isinstance(on, dict) else {on} if isinstance(on, str) else set(on)
        assert keys == {"workflow_dispatch"}, keys

    def test_pytest_step_is_fast_suite(self):
        doc = _ci()
        runs = [s.get("run", "") for j in doc["jobs"].values() for s in j["steps"]]
        cmds = [shlex.split(r) for r in runs if "pytest" in r]
        assert len(cmds) == 1, runs
        argv = cmds[0]
        i = argv.index("pytest")
        assert argv[i - 1] == "-m" and argv[i + 1] == "tests/", argv
        mi = argv.index("-m", i)
        assert argv[mi + 1] == "not slow", argv
        pi = argv.index("-p", i)
        assert argv[pi + 1] == "no:xdist", argv
        # Nothing that would deselect/skip beyond the FAST marker expression.
        for bad in ("-k", "--deselect", "--ignore", "-x", "--lf"):
            assert bad not in argv, argv

    def test_install_covers_every_extra(self):
        doc = _ci()
        runs = " ".join(s.get("run", "") for j in doc["jobs"].values() for s in j["steps"])
        extras = set(_pyproject()["project"]["optional-dependencies"])
        for name in extras:
            assert name in runs, f"extra {name!r} not installed in CI"
