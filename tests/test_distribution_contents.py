"""Tests for source and wheel distribution boundaries."""

from __future__ import annotations

import io
import stat
import tarfile
import zipfile
from pathlib import Path

import pytest

from scripts.check_distribution_contents import (
    DistributionBoundaryError,
    validate_sdist_archive,
    validate_wheel_archive,
)

_ROOT = "hcc_survival-0.1.0"


def _write_sdist(path: Path, members: list[str]) -> None:
    with tarfile.open(path, mode="w:gz") as archive:
        for name in members:
            data = b"synthetic distribution member"
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))


def _write_wheel(path: Path, members: list[str]) -> None:
    with zipfile.ZipFile(path, mode="w") as archive:
        for name in members:
            archive.writestr(name, b"synthetic distribution member")


def test_sdist_allows_only_the_expected_members(tmp_path: Path) -> None:
    archive = tmp_path / "package.tar.gz"
    allowed = frozenset({"README.md", "data/raw/.gitkeep", "data/processed/.gitkeep"})
    _write_sdist(
        archive,
        [
            f"{_ROOT}/README.md",
            f"{_ROOT}/data/raw/.gitkeep",
            f"{_ROOT}/data/processed/.gitkeep",
        ],
    )

    validate_sdist_archive(archive, allowed)


@pytest.mark.parametrize(
    "unexpected_member",
    [
        "notes/unreviewed.md",
        "private/process_notes.md",
        "__pycache__/module.pyc",
        "artifacts/run.json",
        "data/raw/patient.csv",
        "models/model.joblib",
        "reports/table.csv",
        "result.pdf",
        "scripts/generate_pdf_report.py",
    ],
)
def test_sdist_rejects_unapproved_publication_members(
    tmp_path: Path, unexpected_member: str
) -> None:
    archive = tmp_path / "package.tar.gz"
    _write_sdist(archive, [f"{_ROOT}/README.md", f"{_ROOT}/{unexpected_member}"])

    with pytest.raises(DistributionBoundaryError, match="member set"):
        validate_sdist_archive(archive, frozenset({"README.md"}))


@pytest.mark.parametrize(
    "unsafe_member", [f"{_ROOT}/../outside", "/outside", f"{_ROOT}/C:/outside"]
)
def test_sdist_rejects_unsafe_member_paths(tmp_path: Path, unsafe_member: str) -> None:
    archive = tmp_path / "package.tar.gz"
    _write_sdist(archive, [unsafe_member])

    with pytest.raises(DistributionBoundaryError, match="unsafe member path"):
        validate_sdist_archive(archive, frozenset())


def test_sdist_rejects_duplicate_and_link_members(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.tar.gz"
    _write_sdist(duplicate, [f"{_ROOT}/README.md", f"{_ROOT}/README.md"])
    with pytest.raises(DistributionBoundaryError, match="duplicate"):
        validate_sdist_archive(duplicate, frozenset({"README.md"}))

    link = tmp_path / "link.tar.gz"
    with tarfile.open(link, mode="w:gz") as archive:
        info = tarfile.TarInfo(f"{_ROOT}/README.md")
        info.type = tarfile.SYMTYPE
        info.linkname = "target"
        archive.addfile(info)
    with pytest.raises(DistributionBoundaryError, match="archive links"):
        validate_sdist_archive(link, frozenset({"README.md"}))


def test_wheel_allows_only_the_expected_members(tmp_path: Path) -> None:
    archive = tmp_path / "package.whl"
    allowed = frozenset({"hcc_survival/module.py"})
    _write_wheel(archive, ["hcc_survival/module.py"])

    validate_wheel_archive(archive, allowed)


@pytest.mark.parametrize(
    "unexpected_member",
    [
        "docs/METHODS.md",
        "tests/test_data.py",
        "data/raw/patient.csv",
        "models/model.joblib",
        "reports/table.csv",
        "result.pdf",
    ],
)
def test_wheel_rejects_non_package_members(tmp_path: Path, unexpected_member: str) -> None:
    archive = tmp_path / "package.whl"
    _write_wheel(archive, ["hcc_survival/module.py", unexpected_member])

    with pytest.raises(DistributionBoundaryError, match="member set"):
        validate_wheel_archive(archive, frozenset({"hcc_survival/module.py"}))


def test_wheel_rejects_unsafe_duplicate_and_link_members(tmp_path: Path) -> None:
    traversal = tmp_path / "traversal.whl"
    _write_wheel(traversal, ["../outside"])
    with pytest.raises(DistributionBoundaryError, match="unsafe member path"):
        validate_wheel_archive(traversal, frozenset())

    duplicate = tmp_path / "duplicate.whl"
    with zipfile.ZipFile(duplicate, mode="w") as archive:
        archive.writestr("hcc_survival/module.py", b"first")
        with pytest.warns(UserWarning, match="Duplicate name"):
            archive.writestr("hcc_survival/module.py", b"second")
    with pytest.raises(DistributionBoundaryError, match="duplicate"):
        validate_wheel_archive(duplicate, frozenset({"hcc_survival/module.py"}))

    link = tmp_path / "link.whl"
    with zipfile.ZipFile(link, mode="w") as archive:
        info = zipfile.ZipInfo("hcc_survival/link")
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        archive.writestr(info, b"target")
    with pytest.raises(DistributionBoundaryError, match="archive links"):
        validate_wheel_archive(link, frozenset())
