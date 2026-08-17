"""Configuration loading and validation."""

from math import isfinite
from pathlib import Path
from typing import Any

import yaml

from hcc_survival.models import available_model_names


class ConfigurationError(ValueError):
    """Raised for invalid experiment configuration."""


_CALIBRATION_VARIANTS = frozenset({"uncalibrated", "sigmoid"})
_THRESHOLD_OBJECTIVES = frozenset(
    {
        "accuracy",
        "balanced_accuracy",
        "death_specificity",
        "f1",
        "negative_predictive_value",
        "positive_predictive_value",
        "survivor_sensitivity",
    }
)
_SELECTION_DIRECTIONS = {
    "roc_auc": "higher",
    "pr_auc": "higher",
    "brier": "lower",
}
_STABILITY_MARGINS = {
    "roc_auc_repetition_std": "roc_auc_stability_margin",
    "brier_repetition_std": "brier_stability_margin",
}
_SECTION_KEYS = {
    "experiment": frozenset(
        {
            "name",
            "random_seed",
            "outer_folds",
            "outer_repeats",
            "inner_folds",
            "bootstrap_resamples",
            "n_jobs",
        }
    ),
    "models": frozenset({"include"}),
    "calibration": frozenset({"variants", "selection_metric", "minimum_brier_improvement"}),
    "threshold": frozenset({"fixed", "optimize_training_only", "objective"}),
    "explainability": frozenset({"permutation_repeats"}),
    "selection": frozenset(
        {
            "candidate_variant",
            "primary_metric",
            "primary_direction",
            "roc_auc_equivalence_margin",
            "secondary_metrics",
            "pr_auc_equivalence_margin",
            "brier_equivalence_margin",
            "ece_equivalence_margin",
            "stability_metrics",
            "roc_auc_stability_margin",
            "brier_stability_margin",
            "simplicity_order",
        }
    ),
}
_TOP_LEVEL_KEYS = frozenset(_SECTION_KEYS)


def _non_negative_finite(value: Any, name: str) -> None:
    """Validate a non-negative finite numeric configuration value."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationError(f"{name} must be a non-negative finite number.")
    if not isfinite(value) or value < 0:
        raise ConfigurationError(f"{name} must be a non-negative finite number.")


def _validate_mapping_keys(
    mapping: dict[Any, Any], allowed: frozenset[str], section: str | None = None
) -> None:
    """Reject unrecognized keys so misspelled settings cannot be silently ignored."""

    unexpected = sorted(
        (key for key in mapping if not isinstance(key, str) or key not in allowed),
        key=repr,
    )
    if unexpected:
        labels = ", ".join(
            f"{section}.{key}" if section is not None else str(key) for key in unexpected
        )
        raise ConfigurationError(f"Unsupported configuration keys: {labels}.")


def _positive_integer(value: Any, name: str) -> None:
    """Validate a positive integer configuration value."""

    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ConfigurationError(f"{name} must be a positive integer.")


def _non_negative_integer(value: Any, name: str) -> None:
    """Validate a non-negative integer configuration value."""

    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ConfigurationError(f"{name} must be a non-negative integer.")


def _validate_experiment(experiment: Any) -> None:
    """Validate cross-validation and execution settings."""

    if not isinstance(experiment, dict):
        raise ConfigurationError("experiment must be a mapping.")
    _validate_mapping_keys(experiment, _SECTION_KEYS["experiment"], "experiment")
    name = experiment.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ConfigurationError("experiment.name must be a non-empty string.")
    if name != name.strip() or any(character in name for character in ("/", "\\", ":")):
        raise ConfigurationError(
            "experiment.name must be a non-empty label without path separators or drive prefixes."
        )
    for key in ("outer_folds", "outer_repeats", "inner_folds", "bootstrap_resamples"):
        _positive_integer(experiment.get(key), f"experiment.{key}")
    if experiment["outer_folds"] < 2 or experiment["inner_folds"] < 2:
        raise ConfigurationError("Cross-validation fold counts must be at least 2.")
    _non_negative_integer(experiment.get("random_seed"), "experiment.random_seed")
    n_jobs = experiment.get("n_jobs")
    if isinstance(n_jobs, bool) or not isinstance(n_jobs, int) or n_jobs == 0:
        raise ConfigurationError("experiment.n_jobs must be a non-zero integer.")


def _validate_models(models: Any) -> None:
    """Validate the declared model identifiers without constructing models."""

    if not isinstance(models, dict):
        raise ConfigurationError("models must be a mapping.")
    _validate_mapping_keys(models, _SECTION_KEYS["models"], "models")
    included = models.get("include")
    if not isinstance(included, list) or not included:
        raise ConfigurationError("models.include must be a non-empty list of model names.")
    if not all(isinstance(item, str) and item.strip() for item in included):
        raise ConfigurationError("models.include must contain non-empty model names.")
    if len(included) != len(set(included)):
        raise ConfigurationError("models.include must not contain duplicates.")
    supported = available_model_names()
    unsupported = sorted(set(included) - set(supported))
    if unsupported:
        raise ConfigurationError(
            f"Unsupported model names: {unsupported}. Supported model names are {list(supported)}."
        )


def _validate_explainability(explainability: Any) -> None:
    """Validate the configuration for future model-interpretation analyses."""

    if not isinstance(explainability, dict):
        raise ConfigurationError("explainability must be a mapping.")
    _validate_mapping_keys(explainability, _SECTION_KEYS["explainability"], "explainability")
    _positive_integer(
        explainability.get("permutation_repeats"),
        "explainability.permutation_repeats",
    )


def _validate_calibration(calibration: Any) -> None:
    """Validate the explicitly supported training-only calibration choices."""

    if not isinstance(calibration, dict):
        raise ConfigurationError("calibration must be a mapping.")
    _validate_mapping_keys(calibration, _SECTION_KEYS["calibration"], "calibration")
    variants = calibration.get("variants")
    if not isinstance(variants, list) or not all(isinstance(item, str) for item in variants):
        raise ConfigurationError("calibration.variants must be a list of variant names.")
    if len(variants) != len(set(variants)):
        raise ConfigurationError("calibration.variants must not contain duplicates.")
    if "uncalibrated" not in variants:
        raise ConfigurationError("calibration.variants must include 'uncalibrated'.")
    unsupported = sorted(set(variants) - _CALIBRATION_VARIANTS)
    if unsupported:
        raise ConfigurationError(
            f"Unsupported calibration variants: {unsupported}. "
            "Supported variants are uncalibrated and sigmoid."
        )
    if calibration.get("selection_metric") != "brier":
        raise ConfigurationError("calibration.selection_metric must be brier.")
    _non_negative_finite(
        calibration.get("minimum_brier_improvement"),
        "calibration.minimum_brier_improvement",
    )


def _validate_threshold(threshold: Any) -> None:
    """Validate a fixed or training-only optimized decision threshold."""

    if not isinstance(threshold, dict):
        raise ConfigurationError("threshold must be a mapping.")
    _validate_mapping_keys(threshold, _SECTION_KEYS["threshold"], "threshold")
    fixed = threshold.get("fixed")
    if isinstance(fixed, bool) or not isinstance(fixed, (int, float)):
        raise ConfigurationError("threshold.fixed must be a finite number from 0 to 1.")
    if not isfinite(fixed) or not 0 <= fixed <= 1:
        raise ConfigurationError("threshold.fixed must be a finite number from 0 to 1.")
    if not isinstance(threshold.get("optimize_training_only"), bool):
        raise ConfigurationError("threshold.optimize_training_only must be a boolean.")
    objective = threshold.get("objective")
    if objective not in _THRESHOLD_OBJECTIVES:
        raise ConfigurationError(
            f"threshold.objective must be one of {sorted(_THRESHOLD_OBJECTIVES)}."
        )


def _validate_selection(selection: Any) -> None:
    """Validate the configured sequence of model-selection gates."""

    if not isinstance(selection, dict):
        raise ConfigurationError("selection must be a mapping.")
    _validate_mapping_keys(selection, _SECTION_KEYS["selection"], "selection")
    primary_metric = selection.get("primary_metric")
    if primary_metric not in _SELECTION_DIRECTIONS:
        raise ConfigurationError(
            f"selection.primary_metric must be one of {sorted(_SELECTION_DIRECTIONS)}."
        )
    primary_direction = selection.get("primary_direction")
    if primary_direction != _SELECTION_DIRECTIONS[primary_metric]:
        raise ConfigurationError(
            "selection.primary_direction must match the declared primary metric direction."
        )
    _non_negative_finite(
        selection.get(f"{primary_metric}_equivalence_margin"),
        f"selection.{primary_metric}_equivalence_margin",
    )
    secondary = selection.get("secondary_metrics")
    if not isinstance(secondary, list) or not all(isinstance(item, str) for item in secondary):
        raise ConfigurationError("selection.secondary_metrics must be a list of metric names.")
    if len(secondary) != len(set(secondary)):
        raise ConfigurationError("selection.secondary_metrics must not contain duplicates.")
    if primary_metric in secondary:
        raise ConfigurationError("selection.secondary_metrics must not repeat the primary metric.")
    unsupported_secondary = sorted(set(secondary) - set(_SELECTION_DIRECTIONS))
    if unsupported_secondary:
        raise ConfigurationError(
            f"Unsupported secondary selection metrics: {unsupported_secondary}."
        )
    for metric in secondary:
        _non_negative_finite(
            selection.get(f"{metric}_equivalence_margin"),
            f"selection.{metric}_equivalence_margin",
        )
    if "ece_equivalence_margin" in selection and selection["ece_equivalence_margin"] is not None:
        _non_negative_finite(
            selection["ece_equivalence_margin"],
            "selection.ece_equivalence_margin",
        )
    stability = selection.get("stability_metrics")
    if not isinstance(stability, list) or not all(isinstance(item, str) for item in stability):
        raise ConfigurationError("selection.stability_metrics must be a list of metric names.")
    if len(stability) != len(set(stability)):
        raise ConfigurationError("selection.stability_metrics must not contain duplicates.")
    unsupported_stability = sorted(set(stability) - set(_STABILITY_MARGINS))
    if unsupported_stability:
        raise ConfigurationError(
            f"Unsupported stability selection metrics: {unsupported_stability}."
        )
    for metric in stability:
        margin = _STABILITY_MARGINS[metric]
        _non_negative_finite(selection.get(margin), f"selection.{margin}")
    if not isinstance(selection.get("candidate_variant"), str):
        raise ConfigurationError("selection.candidate_variant must be a string.")
    simplicity = selection.get("simplicity_order")
    if not isinstance(simplicity, dict) or not simplicity:
        raise ConfigurationError("selection.simplicity_order must be a non-empty mapping.")
    if not all(isinstance(name, str) for name in simplicity):
        raise ConfigurationError("selection.simplicity_order keys must be model names.")
    for model, rank in simplicity.items():
        if isinstance(rank, bool) or not isinstance(rank, (int, float)) or not isfinite(rank):
            raise ConfigurationError(
                f"selection.simplicity_order[{model!r}] must be a finite numeric rank."
            )


def validate_config(config: Any) -> dict[str, Any]:
    """Validate a complete in-memory experiment configuration."""

    if not isinstance(config, dict):
        raise ConfigurationError("Configuration root must be a mapping.")
    _validate_mapping_keys(config, _TOP_LEVEL_KEYS)
    missing = _TOP_LEVEL_KEYS - set(config)
    if missing:
        raise ConfigurationError(f"Missing configuration sections: {sorted(missing)}")
    _validate_experiment(config["experiment"])
    _validate_models(config["models"])
    _validate_calibration(config["calibration"])
    _validate_threshold(config["threshold"])
    _validate_explainability(config["explainability"])
    _validate_selection(config["selection"])
    return config


def load_config(path: Path | str) -> dict[str, Any]:
    """Load and validate an experiment YAML file."""

    path = Path(path)
    if not path.is_file():
        raise ConfigurationError("Configuration file does not exist.")
    try:
        content = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ConfigurationError("Configuration file could not be read.") from error
    except yaml.YAMLError as error:
        raise ConfigurationError("Configuration file is not valid YAML.") from error
    return validate_config(content)
