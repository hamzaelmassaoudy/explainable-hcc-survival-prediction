"""Tests for the public experiment configuration loader."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from hcc_survival.config import ConfigurationError, load_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("filename", "outer_folds", "outer_repeats"),
    [("fast.yaml", 3, 1), ("full.yaml", 5, 5)],
)
def test_example_configurations_load(
    filename: str,
    outer_folds: int,
    outer_repeats: int,
) -> None:
    """Both checked-in configurations satisfy the shared validation rules."""

    config = load_config(PROJECT_ROOT / "configs" / filename)

    assert config["experiment"]["outer_folds"] == outer_folds
    assert config["experiment"]["outer_repeats"] == outer_repeats
    assert config["threshold"]["optimize_training_only"] is True
    assert config["calibration"]["selection_metric"] == "brier"


def test_missing_configuration_error_does_not_echo_a_local_path(tmp_path: Path) -> None:
    """Missing-file errors remain useful without exposing machine-specific paths."""

    missing = tmp_path / "private" / "missing.yaml"

    with pytest.raises(ConfigurationError, match="Configuration file does not exist") as error:
        load_config(missing)

    assert str(missing) not in str(error.value)


@pytest.mark.parametrize(
    ("section", "replacement", "message"),
    [
        ("experiment", [], "experiment must be a mapping"),
        ("models", {"include": []}, "models.include"),
        ("explainability", {"permutation_repeats": 0}, "permutation_repeats"),
    ],
)
def test_invalid_configuration_sections_are_rejected(
    tmp_path: Path,
    section: str,
    replacement: object,
    message: str,
) -> None:
    """Malformed sections fail with a configuration error rather than an attribute error."""

    baseline = yaml.safe_load((PROJECT_ROOT / "configs" / "fast.yaml").read_text(encoding="utf-8"))
    candidate = deepcopy(baseline)
    candidate[section] = replacement
    path = tmp_path / f"invalid-{section}.yaml"
    path.write_text(yaml.safe_dump(candidate), encoding="utf-8")

    with pytest.raises(ConfigurationError, match=message):
        load_config(path)
