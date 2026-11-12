"""Root conftest: refuse to run against a stray/shadowing `summicron` (HYG-0 item 9)."""
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent

def pytest_configure(config):
    import summicron
    f = getattr(summicron, "__file__", None)  # None => namespace fragment (stray site-packages dir)
    if f is None or Path(f).resolve().parent != _REPO_ROOT / "summicron":
        pytest.exit(f"summicron imported from {f!r}, not under repo root {_REPO_ROOT}", returncode=4)


@pytest.fixture(autouse=True)
def _clear_memory_budget_env(monkeypatch):
    """R-native-mem 9c: the memory guard reads one piece of external state,
    ``ISUMMICRON_MAX_BYTES``. Clear it so a developer's shell never changes a
    test outcome (CI determinism)."""
    monkeypatch.delenv("ISUMMICRON_MAX_BYTES", raising=False)
