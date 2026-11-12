"""CI round adversarial tests (doc/plan/phase2/section12_ci/locked-spec-CI.md).

Pins: the ``hdr-macos`` extra lists only real distributions, the ruff
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
