"""macOS-HDR argparse-time preflight (P1.audit Item 8).

When ``--hdr-source macos`` is requested on a non-darwin platform the
validator MUST surface a clean ``SystemExit(2)`` with an actionable
message — not let the deferred Quartz/CoreImage import inside
:func:`load_hdr_heic` raise a stack trace.
"""
from __future__ import annotations

import pytest

from summicron.render.convolve import _validate_args, build_parser


def test_macos_hdr_on_linux_argparse_clean_exit(monkeypatch, capsys):
    """Validator fires before any deferred macOS import; exit 2 with clean msg."""
    monkeypatch.setattr("sys.platform", "linux")
    parser = build_parser()
    # Build a minimal-valid args namespace that only differs by --hdr-source macos.
    args = parser.parse_args([
        "--image", "x.heic",
        "--tensor", "x.npz",
        "--hdr-source", "macos",
    ])
    with pytest.raises(SystemExit) as exc:
        _validate_args(args, parser)
    assert exc.value.code == 2
    captured = capsys.readouterr()
    # parser.error() writes to stderr; assert message hints at macOS.
    assert "macOS" in captured.err or "darwin" in captured.err.lower()


def test_macos_hdr_preflight_fires_before_shadow_match_validator(monkeypatch, capsys):
    """When BOTH --hdr-source macos and --shadow-match-mode on are wrong, the
    platform error wins (it's listed first in _validate_args)."""
    monkeypatch.setattr("sys.platform", "linux")
    parser = build_parser()
    args = parser.parse_args([
        "--image", "x.heic",
        "--tensor", "x.npz",
        "--hdr-source", "macos",
        "--shadow-match-mode", "on",
    ])
    with pytest.raises(SystemExit) as exc:
        _validate_args(args, parser)
    assert exc.value.code == 2
    captured = capsys.readouterr()
    # Platform error wins — the actual error line (vs. the argparse usage
    # banner) cites macOS/darwin.
    assert "macOS" in captured.err or "darwin" in captured.err.lower()
    # Specifically, the "--shadow-match-mode on requires --hdr-source != none"
    # message must NOT be the active error line.
    assert "shadow-match-mode on requires" not in captured.err
