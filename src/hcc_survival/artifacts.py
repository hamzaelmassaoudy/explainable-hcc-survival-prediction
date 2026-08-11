"""Safe, reproducible experiment artifact management."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import yaml

from hcc_survival.constants import DEFAULT_ARTIFACT_ROOT

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_OUTPUT_ROOTS = (
    PROJECT_ROOT / "artifacts",
    PROJECT_ROOT / "data" / "processed",
    PROJECT_ROOT / "local_audit",
    PROJECT_ROOT / "models",
    PROJECT_ROOT / "outputs",
    PROJECT_ROOT / "reports",
    PROJECT_ROOT / "results",
    PROJECT_ROOT / "runs",
)


def ensure_local_output_path(path: Path | str, *, purpose: str) -> Path:
    """Allow project outputs only in ignored local directories or outside the project.

    Relative paths retain their existing working-directory behavior. A path inside this
    repository must be under an ignored local-output root, which prevents generated
    patient-level derivatives and experiment artifacts from being written beside source files.
    """

    candidate = Path(path)
    resolved = candidate.resolve()
    project_root = PROJECT_ROOT.resolve()
    if resolved.is_relative_to(project_root) and not any(
        resolved.is_relative_to(root.resolve()) for root in LOCAL_OUTPUT_ROOTS
    ):
        allowed = ", ".join(root.name for root in LOCAL_OUTPUT_ROOTS)
        raise ValueError(
            f"{purpose} must be written under an ignored local-output directory "
            f"({allowed}) or outside the project directory."
        )
    return candidate


def package_versions() -> dict[str, str]:
    """Capture relevant installed package versions."""

    versions = {"python": platform.python_version()}
    for name in (
        "hcc-survival",
        "numpy",
        "pandas",
        "scikit-learn",
        "scipy",
        "joblib",
        "matplotlib",
        "seaborn",
        "streamlit",
        "ucimlrepo",
        "xgboost",
    ):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = "not installed"
    return versions


def git_state(project_root: Path) -> dict[str, str]:
    """Capture Git revision and cleanliness without requiring a repository."""

    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout
        return {"commit_sha": sha, "working_tree": "dirty" if status.strip() else "clean"}
    except (OSError, subprocess.SubprocessError):
        return {"commit_sha": "not available", "working_tree": "not a Git worktree"}


def create_run_directory(config: dict[str, Any], root: Path | str = DEFAULT_ARTIFACT_ROOT) -> Path:
    """Create a unique readable run directory without overwriting."""

    root = ensure_local_output_path(root, purpose="Experiment artifacts")
    root.mkdir(parents=True, exist_ok=True)
    resolved_root = root.resolve()
    config_text = yaml.safe_dump(config, sort_keys=True)
    digest = hashlib.sha256(config_text.encode()).hexdigest()[:8]
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = root / f"{config['experiment']['name']}-{timestamp}-{digest}"
    if not run_dir.resolve().is_relative_to(resolved_root):
        raise ValueError("Experiment artifacts must remain under the configured output root.")
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    for child in ("figures", "tables", "models"):
        (run_dir / child).mkdir(parents=True)
    (run_dir / "config.yaml").write_text(config_text, encoding="utf-8")
    write_json(run_dir / "environment.json", package_versions())
    write_json(
        run_dir / "provenance.json",
        {
            "configuration_sha256": hashlib.sha256(config_text.encode()).hexdigest(),
            "git": git_state(Path.cwd()),
        },
    )
    return run_dir


def write_json(path: Path, value: Any) -> None:
    """Write JSON with NumPy-friendly fallback conversion."""

    path.write_text(
        json.dumps(
            value,
            indent=2,
            default=lambda item: item.item() if hasattr(item, "item") else str(item),
        ),
        encoding="utf-8",
    )
