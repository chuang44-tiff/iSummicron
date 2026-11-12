"""Memory guard tests (P1.audit Item 6; rewritten for R-native-mem, spec
``doc/plan/phase2/section17_native-memory/locked-spec-mem.md`` §3.7 / §3.8).

The estimator is now a structural plane sum (see
``test_memory_estimate_honesty.py`` for the tracemalloc honesty check) and the
guard runs against a fixed 12 GiB cap, overridable only through the
``ISUMMICRON_MAX_BYTES`` environment variable.
"""
from __future__ import annotations

import pytest

from summicron.render.convolve import (
    MAX_BYTES,
    MAX_PIXELS,
    MemoryBudgetError,
    _estimate_peak_bytes,
    _memory_guard,
)

# Every iphoneImage/ HEIC decodes to one of these two shapes (hard-coded so
# the test does not read the HEICs): landscape 5712x4284, portrait 4284x5712.
_BENCH_SHAPES = [(5712, 4284), (4284, 5712)]
_GIB = 1024**3
_BUDGET = int(1.5 * _GIB)  # PLATFORM_BASE is 1 GiB, so 1 GiB admits nothing


def test_max_bytes_default_is_12_gb() -> None:
    """The cap stays 12 GiB (spec §8e); it is a fixed deterministic constant."""
    assert MAX_BYTES == 12 * 1024**3


# ---------------------------------------------------------------------------
# 8(a): every real bench shape passes the guard and sits <= 0.90 x cap
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("occlusion", [False, True])
@pytest.mark.parametrize("h, w", _BENCH_SHAPES)
def test_bench_shapes_pass_guard_under_90pct_of_cap(
    h: int, w: int, occlusion: bool,
) -> None:
    """G6: never block a render that works today (native, D=11, k=183, Nr=12,
    default batch, OFF and ON, both orientations)."""
    _memory_guard(h, w, 11, occlusion=occlusion, kernel_size=183, n_radial=12)
    est = _estimate_peak_bytes(
        h, w, 11, occlusion=occlusion, kernel_size=183, n_radial=12,
    )
    assert est <= 0.90 * MAX_BYTES, (
        f"{h}x{w} occlusion={occlusion}: estimate {est / _GIB:.2f} GiB > "
        f"0.90 x cap ({0.90 * MAX_BYTES / _GIB:.2f} GiB)"
    )


def test_acceptable_downres_sizes_pass() -> None:
    for h, w in [(2048, 1536), (2048, 2048), (4096, 3072)]:
        _memory_guard(h, w, 11)


# ---------------------------------------------------------------------------
# 8(b): rejection through the byte budget alone
# ---------------------------------------------------------------------------

def test_budget_rejection_via_env_names_downres(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Tiny budget so that a 2048x1536 render (which passes the pixel guard)
    # is rejected purely by the byte budget.
    monkeypatch.setenv("ISUMMICRON_MAX_BYTES", str(_BUDGET))
    assert 2048 * 1536 < MAX_PIXELS
    with pytest.raises(MemoryBudgetError) as exc:
        _memory_guard(2048, 1536, 11)
    err = exc.value
    assert isinstance(err, RuntimeError)
    assert "--downres" in str(err)
    assert err.budget_bytes == _BUDGET
    assert err.estimated_bytes > err.budget_bytes
    sug = err.suggested_downres
    assert sug is not None and sug % 64 == 0 and sug >= 256
    # The suggestion fits (long edge = H here; same aspect ratio, kernel_size
    # held at the default) and is the largest multiple of 64 that does.
    w = max(1, round(1536 / 2048 * sug))
    assert _estimate_peak_bytes(sug, w, 11) <= _BUDGET
    w2 = max(1, round(1536 / 2048 * (sug + 64)))
    assert _estimate_peak_bytes(sug + 64, w2, 11) > _BUDGET


def test_budget_rejection_without_suggestion_when_nothing_fits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ISUMMICRON_MAX_BYTES", "1048576")  # 1 MiB < PLATFORM_BASE
    with pytest.raises(MemoryBudgetError) as exc:
        _memory_guard(512, 512, 11)
    assert exc.value.suggested_downres is None
    assert "raise ISUMMICRON_MAX_BYTES" in str(exc.value)


# ---------------------------------------------------------------------------
# 8(c): invalid env values
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", ["abc", "0", "-1", "", "1.5"])
def test_invalid_env_raises_value_error(
    monkeypatch: pytest.MonkeyPatch, bad: str,
) -> None:
    monkeypatch.setenv("ISUMMICRON_MAX_BYTES", bad)
    with pytest.raises(ValueError, match="ISUMMICRON_MAX_BYTES"):
        _memory_guard(256, 256, 11)


def test_env_override_raises_budget_for_big_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Native batch=11 is rejected under the default cap and admitted when the
    budget is raised (the dev-box use of ISUMMICRON_MAX_BYTES)."""
    with pytest.raises(MemoryBudgetError):
        _memory_guard(5712, 4284, 11, slab_batch_size=11)
    monkeypatch.setenv("ISUMMICRON_MAX_BYTES", str(64 * _GIB))
    _memory_guard(5712, 4284, 11, slab_batch_size=11)


# ---------------------------------------------------------------------------
# 8(d): GiB units in messages
# ---------------------------------------------------------------------------

def test_messages_are_in_gib(monkeypatch: pytest.MonkeyPatch) -> None:
    msgs: list[str] = []
    _memory_guard(2048, 1536, 11, on_progress=msgs.append)
    assert any("Memory budget" in m for m in msgs)
    assert any("GiB peak" in m and "cap 12.0 GiB" in m for m in msgs)
    assert not any(" GB" in m for m in msgs)

    monkeypatch.setenv("ISUMMICRON_MAX_BYTES", str(_BUDGET))
    with pytest.raises(MemoryBudgetError) as exc:
        _memory_guard(2048, 1536, 11)
    assert "1.5 GiB budget" in str(exc.value)
    assert "GiB at H=2048 W=1536 D=11" in str(exc.value)


# ---------------------------------------------------------------------------
# Fast-path early gate (MAX_PIXELS) is preserved
# ---------------------------------------------------------------------------

def test_max_pixels_fast_path_still_rejects() -> None:
    h, w, d = 6042, 6042, 1
    assert h * w > MAX_PIXELS
    with pytest.raises(RuntimeError, match="Image too large") as exc:
        _memory_guard(h, w, d)
    assert "downres" in str(exc.value)
    assert not isinstance(exc.value, MemoryBudgetError)


# ---------------------------------------------------------------------------
# CLI bit-id: progress emit is silent when no sink is registered
# ---------------------------------------------------------------------------

def test_memory_guard_silent_on_cli_path(capsys: pytest.CaptureFixture[str]) -> None:
    # No on_progress sink => no stdout (CLI path; G1 md5 protected).
    _memory_guard(2048, 2048, 11)
    captured = capsys.readouterr()
    assert captured.out == "", (
        f"_memory_guard leaked stdout on CLI path: {captured.out!r} — "
        f"breaks canonical-bench bit-id"
    )


def test_memory_guard_emits_to_progress_sink_when_provided() -> None:
    msgs: list[str] = []
    _memory_guard(2048, 2048, 11, on_progress=msgs.append)
    assert any("Memory budget" in m for m in msgs), (
        f"progress sink missed the budget line: {msgs!r}"
    )
    assert any("GiB peak" in m for m in msgs)
