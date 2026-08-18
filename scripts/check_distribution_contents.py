"""Verify that built source and wheel archives contain only approved public files."""

from __future__ import annotations

import argparse
import stat
import sys
import tarfile
import zipfile
from collections.abc import Iterable
from pathlib import Path, PurePosixPath


class DistributionBoundaryError(ValueError):
    """Raised when a distribution archive crosses the public file boundary."""


SDIST_ALLOWED_MEMBERS = frozenset(
    {
        ".github/workflows/ci.yml",
        ".gitignore",
        ".pre-commit-config.yaml",
        "CHANGELOG.md",
        "CITATION.cff",
        "CODE_OF_CONDUCT.md",
        "CONTRIBUTING.md",
        "LICENSE",
        "PKG-INFO",
        "README.md",
        "SECURITY.md",
        "app/streamlit_app.py",
        "configs/fast.yaml",
        "configs/full.yaml",
        "data/README.md",
        "data/processed/.gitkeep",
        "data/raw/.gitkeep",
        "docs/DATA_CARD.md",
        "docs/DATA_DICTIONARY.md",
        "docs/DEVELOPMENT.md",
        "docs/IMPLEMENTATION_PLAN.md",
        "docs/INTENDED_USE.md",
        "docs/LEARNING_GUIDE.md",
        "docs/LIMITATIONS.md",
        "docs/METHODS.md",
        "docs/MODEL_CARD.md",
        "docs/PRIVACY.md",
        "docs/RELEASE_CHECKLIST.md",
        "docs/REPRODUCIBILITY.md",
        "docs/TEST_EVIDENCE.md",
        "pyproject.toml",
        "scripts/check_distribution_contents.py",
        "scripts/verify_project.py",
        "src/hcc_survival/__init__.py",
        "src/hcc_survival/__main__.py",
        "src/hcc_survival/artifacts.py",
        "src/hcc_survival/calibration.py",
        "src/hcc_survival/cli.py",
        "src/hcc_survival/config.py",
        "src/hcc_survival/constants.py",
        "src/hcc_survival/data.py",
        "src/hcc_survival/diagnostics.py",
        "src/hcc_survival/eda.py",
        "src/hcc_survival/evaluation.py",
        "src/hcc_survival/explainability.py",
        "src/hcc_survival/metrics.py",
        "src/hcc_survival/models.py",
        "src/hcc_survival/prediction.py",
        "src/hcc_survival/preprocessing.py",
        "src/hcc_survival/reporting.py",
        "src/hcc_survival/schemas.py",
        "src/hcc_survival/sensitivity.py",
        "src/hcc_survival/subgroup.py",
        "tests/conftest.py",
        "tests/test_cli.py",
        "tests/test_cli_and_streamlit_contracts.py",
        "tests/test_config.py",
        "tests/test_data.py",
        "tests/test_distribution_contents.py",
        "tests/test_metrics.py",
        "tests/test_models.py",
        "tests/test_output_boundaries.py",
        "tests/test_prediction.py",
        "tests/test_preprocessing.py",
        "tests/test_repeated_oof_contracts.py",
        "tests/test_reproducibility_and_artifacts.py",
        "tests/test_scientific_configuration_controls.py",
        "tests/test_scientific_contracts.py",
        "tests/test_selection_contracts.py",
        "tests/test_smoke_pipeline.py",
        "tests/test_subgroup.py",
    }
)

WHEEL_ALLOWED_MEMBERS = frozenset(
    {
        "hcc_survival/__init__.py",
        "hcc_survival/__main__.py",
        "hcc_survival/artifacts.py",
        "hcc_survival/calibration.py",
        "hcc_survival/cli.py",
        "hcc_survival/config.py",
        "hcc_survival/constants.py",
        "hcc_survival/data.py",
        "hcc_survival/diagnostics.py",
        "hcc_survival/eda.py",
        "hcc_survival/evaluation.py",
        "hcc_survival/explainability.py",
        "hcc_survival/metrics.py",
        "hcc_survival/models.py",
        "hcc_survival/prediction.py",
        "hcc_survival/preprocessing.py",
        "hcc_survival/reporting.py",
        "hcc_survival/schemas.py",
        "hcc_survival/sensitivity.py",
        "hcc_survival/subgroup.py",
        "hcc_survival-0.1.0.dist-info/METADATA",
        "hcc_survival-0.1.0.dist-info/RECORD",
        "hcc_survival-0.1.0.dist-info/WHEEL",
        "hcc_survival-0.1.0.dist-info/licenses/LICENSE",
    }
)


def _safe_member_path(name: str) -> PurePosixPath:
    """Return a portable archive member path or reject an unsafe path."""

    path = PurePosixPath(name)
    if (
        not name
        or "\\" in name
        or path.is_absolute()
        or not path.parts
        or any(part in {".", ".."} for part in path.parts)
        or any(":" in part for part in path.parts)
    ):
        raise DistributionBoundaryError("Distribution archive contains an unsafe member path.")
    return path


def _check_member_set(
    actual_members: set[str], allowed_members: frozenset[str], archive_label: str
) -> None:
    """Require an archive to match its reviewed public file set exactly."""

    if actual_members != allowed_members:
        raise DistributionBoundaryError(
            f"{archive_label} member set does not match the approved public distribution."
        )


def _source_member_paths(members: Iterable[tarfile.TarInfo]) -> set[str]:
    """Return normalized source-distribution file paths after boundary validation."""

    parsed_members = [(_safe_member_path(member.name), member) for member in members]
    if not parsed_members:
        raise DistributionBoundaryError("Source distribution is empty.")
    roots = {path.parts[0] for path, _ in parsed_members}
    if len(roots) != 1:
        raise DistributionBoundaryError("Source distribution must have exactly one root directory.")

    files: set[str] = set()
    for path, member in parsed_members:
        if member.issym() or member.islnk():
            raise DistributionBoundaryError("Source distribution must not contain archive links.")
        if member.isdir():
            continue
        if not member.isfile() or len(path.parts) < 2:
            raise DistributionBoundaryError("Source distribution contains an unsupported member.")
        relative = PurePosixPath(*path.parts[1:]).as_posix()
        if relative in files:
            raise DistributionBoundaryError("Source distribution contains duplicate members.")
        files.add(relative)
    return files


def validate_sdist_archive(
    archive_path: Path, allowed_members: frozenset[str] = SDIST_ALLOWED_MEMBERS
) -> None:
    """Validate a source distribution without extracting it."""

    with tarfile.open(archive_path, mode="r:gz") as archive:
        members = _source_member_paths(archive.getmembers())
    _check_member_set(members, allowed_members, "Source archive")


def _wheel_member_paths(members: Iterable[zipfile.ZipInfo]) -> set[str]:
    """Return normalized wheel file paths after boundary validation."""

    files: set[str] = set()
    names: set[str] = set()
    for member in members:
        path = _safe_member_path(member.filename)
        normalized = path.as_posix()
        if normalized in names:
            raise DistributionBoundaryError("Wheel contains duplicate members.")
        names.add(normalized)
        if stat.S_ISLNK(member.external_attr >> 16):
            raise DistributionBoundaryError("Wheel must not contain archive links.")
        if member.is_dir():
            continue
        files.add(normalized)
    if not files:
        raise DistributionBoundaryError("Wheel is empty.")
    return files


def validate_wheel_archive(
    archive_path: Path, allowed_members: frozenset[str] = WHEEL_ALLOWED_MEMBERS
) -> None:
    """Validate a wheel without extracting it."""

    with zipfile.ZipFile(archive_path) as archive:
        _check_member_set(_wheel_member_paths(archive.infolist()), allowed_members, "Wheel")


def _single_distribution(dist_dir: Path, pattern: str, label: str) -> Path:
    """Return the one expected built archive of a given type."""

    candidates = sorted(path for path in dist_dir.glob(pattern) if path.is_file())
    if len(candidates) != 1:
        raise DistributionBoundaryError(f"Expected exactly one {label} in the build directory.")
    return candidates[0]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the directory that contains freshly built distribution archives."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dist-dir",
        type=Path,
        default=Path("dist"),
        help="Directory containing one source archive and one wheel (default: dist).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Validate the built source archive and wheel against the public boundary."""

    args = parse_args(argv)
    try:
        validate_sdist_archive(_single_distribution(args.dist_dir, "*.tar.gz", "source archive"))
        validate_wheel_archive(_single_distribution(args.dist_dir, "*.whl", "wheel"))
    except (DistributionBoundaryError, OSError, tarfile.TarError, zipfile.BadZipFile) as error:
        print(f"Distribution boundary check failed: {error}", file=sys.stderr)
        return 1
    print("Distribution contents verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
