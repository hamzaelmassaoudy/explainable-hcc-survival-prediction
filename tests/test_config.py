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


@pytest.mark.parametrize("name", ["../outside", "nested/run", r"nested\run", r"C:\outside"])
def test_path_like_experiment_names_are_rejected(tmp_path: Path, name: str) -> None:
    """Experiment labels cannot redirect local artifact paths."""

    baseline = yaml.safe_load((PROJECT_ROOT / "configs" / "fast.yaml").read_text(encoding="utf-8"))
    candidate = deepcopy(baseline)
    candidate["experiment"]["name"] = name
    path = tmp_path / "invalid-name.yaml"
    path.write_text(yaml.safe_dump(candidate), encoding="utf-8")

    with pytest.raises(ConfigurationError, match="without path separators or drive prefixes"):
        load_config(path)


@pytest.mark.parametrize("included", [["not-a-model"], ["dummy", "not-a-model"]])
def test_unknown_model_names_are_rejected(tmp_path: Path, included: list[str]) -> None:
    """Configuration loading rejects model names that cannot be constructed."""

    baseline = yaml.safe_load((PROJECT_ROOT / "configs" / "fast.yaml").read_text(encoding="utf-8"))
    candidate = deepcopy(baseline)
    candidate["models"]["include"] = included
    path = tmp_path / "unknown-model.yaml"
    path.write_text(yaml.safe_dump(candidate), encoding="utf-8")

    with pytest.raises(ConfigurationError, match="Unsupported model names") as error:
        load_config(path)

    assert "not-a-model" in str(error.value)
    assert str(path) not in str(error.value)


@pytest.mark.parametrize(
    ("candidate_variant", "calibration_variants"),
    [("uncalibrated", ["uncalibrated"]), ("sigmoid", ["uncalibrated", "sigmoid"])],
)
def test_configured_selection_variants_are_accepted(
    tmp_path: Path, candidate_variant: str, calibration_variants: list[str]
) -> None:
    """Configured calibration variants remain eligible selection candidates."""

    baseline = yaml.safe_load((PROJECT_ROOT / "configs" / "fast.yaml").read_text(encoding="utf-8"))
    candidate = deepcopy(baseline)
    candidate["models"]["include"] = ["dummy"]
    candidate["calibration"]["variants"] = calibration_variants
    candidate["selection"]["candidate_variant"] = candidate_variant
    path = tmp_path / "configured-selection.yaml"
    path.write_text(yaml.safe_dump(candidate), encoding="utf-8")

    loaded = load_config(path)

    assert loaded["selection"]["candidate_variant"] == candidate_variant


@pytest.mark.parametrize(
    ("candidate_variant", "calibration_variants", "missing_rank", "unexpected_rank", "message"),
    [
        (
            "sigmoid",
            ["uncalibrated"],
            None,
            None,
            "selection.candidate_variant must be one of",
        ),
        (
            "not-produced",
            None,
            None,
            None,
            "selection.candidate_variant must be one of",
        ),
        (
            "training_selected",
            None,
            "logistic_regression",
            None,
            "selection.simplicity_order is missing configured models",
        ),
        (
            "training_selected",
            None,
            None,
            "not-a-model",
            "selection.simplicity_order contains unsupported model names",
        ),
    ],
)
def test_selection_settings_must_match_configured_models_and_variants(
    tmp_path: Path,
    candidate_variant: str,
    calibration_variants: list[str] | None,
    missing_rank: str | None,
    unexpected_rank: str | None,
    message: str,
) -> None:
    """Selection settings must describe models and variants produced by the run."""

    baseline = yaml.safe_load((PROJECT_ROOT / "configs" / "fast.yaml").read_text(encoding="utf-8"))
    candidate = deepcopy(baseline)
    candidate["selection"]["candidate_variant"] = candidate_variant
    if calibration_variants is not None:
        candidate["calibration"]["variants"] = calibration_variants
    if missing_rank is not None:
        candidate["selection"]["simplicity_order"].pop(missing_rank)
    if unexpected_rank is not None:
        candidate["selection"]["simplicity_order"][unexpected_rank] = 5
    path = tmp_path / "invalid-selection.yaml"
    path.write_text(yaml.safe_dump(candidate), encoding="utf-8")

    with pytest.raises(ConfigurationError, match=message) as error:
        load_config(path)

    assert str(path) not in str(error.value)


@pytest.mark.parametrize(
    ("section", "key", "value", "expected_key"),
    [
        (None, "notes", "unreviewed", "notes"),
        ("experiment", "outer_repeat", 5, "experiment.outer_repeat"),
        ("models", "incldue", ["dummy"], "models.incldue"),
        (
            "calibration",
            "minimum_brier_improvment",
            0.005,
            "calibration.minimum_brier_improvment",
        ),
        ("threshold", "fixed_value", 0.5, "threshold.fixed_value"),
        (
            "explainability",
            "permutation_repeat",
            5,
            "explainability.permutation_repeat",
        ),
        (
            "selection",
            "roc_auc_equivalence_magin",
            0.01,
            "selection.roc_auc_equivalence_magin",
        ),
    ],
)
def test_unknown_configuration_keys_are_rejected(
    tmp_path: Path,
    section: str | None,
    key: str,
    value: object,
    expected_key: str,
) -> None:
    """Typos in configuration names fail instead of silently changing nothing."""

    baseline = yaml.safe_load((PROJECT_ROOT / "configs" / "fast.yaml").read_text(encoding="utf-8"))
    candidate = deepcopy(baseline)
    if section is None:
        candidate[key] = value
    else:
        candidate[section][key] = value
    path = tmp_path / "unknown-key.yaml"
    path.write_text(yaml.safe_dump(candidate), encoding="utf-8")

    with pytest.raises(ConfigurationError) as error:
        load_config(path)

    assert expected_key in str(error.value)
    assert str(path) not in str(error.value)
