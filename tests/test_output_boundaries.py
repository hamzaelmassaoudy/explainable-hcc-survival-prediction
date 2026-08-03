"""Tests for local experiment-output boundaries."""

from __future__ import annotations

from pathlib import Path

import pytest

from hcc_survival.artifacts import ensure_local_output_path


def test_project_outputs_require_an_ignored_local_root() -> None:
    """Generated files cannot be redirected beside versioned source files."""

    assert ensure_local_output_path(Path("artifacts") / "example", purpose="Test output") == (
        Path("artifacts") / "example"
    )
    with pytest.raises(ValueError, match="ignored local-output directory"):
        ensure_local_output_path(Path("untracked_output"), purpose="Test output")


def test_external_output_paths_remain_available_for_local_research(tmp_path: Path) -> None:
    """A caller can still choose a local path outside the checked-out project."""

    assert ensure_local_output_path(tmp_path / "output", purpose="Test output") == (
        tmp_path / "output"
    )
