"""Tests for local experiment-output boundaries."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import hcc_survival.artifacts as artifacts
from hcc_survival.artifacts import create_run_directory, ensure_local_output_path


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


@pytest.mark.parametrize("name", ["../outside", "/outside"])
def test_run_directory_cannot_escape_its_configured_root(tmp_path: Path, name: str) -> None:
    """Artifact creation remains bounded when called without configuration loading."""

    root = tmp_path / "runs"
    config = {"experiment": {"name": name}}

    with pytest.raises(ValueError, match="configured output root"):
        create_run_directory(config, root=root)

    assert root.is_dir()
    assert not list(root.iterdir())
    assert not list(tmp_path.glob("outside-*"))


def test_run_provenance_uses_the_project_root_when_called_externally(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Provenance describes the package worktree rather than the caller's directory."""

    observed_roots: list[Path] = []

    def capture_git_state(project_root: Path) -> dict[str, str]:
        observed_roots.append(project_root)
        return {"commit_sha": "synthetic", "working_tree": "clean"}

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(artifacts, "git_state", capture_git_state)
    run_dir = create_run_directory({"experiment": {"name": "provenance"}}, tmp_path / "runs")

    assert observed_roots == [artifacts.PROJECT_ROOT]
    provenance = json.loads((run_dir / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["git"] == {"commit_sha": "synthetic", "working_tree": "clean"}
