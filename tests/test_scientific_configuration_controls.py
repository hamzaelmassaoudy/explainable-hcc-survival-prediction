"""Regression tests for configurable scientific decision controls."""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.model_selection import RepeatedStratifiedKFold

from hcc_survival.config import ConfigurationError, load_config
from hcc_survival.evaluation import _select_threshold, run_nested_experiment
from hcc_survival.reporting import fit_final_model
from hcc_survival.sensitivity import run_missingness_sensitivity


def _small_config() -> dict[str, object]:
    config = load_config("configs/fast.yaml")
    config["experiment"]["outer_folds"] = 2
    config["experiment"]["outer_repeats"] = 1
    config["experiment"]["inner_folds"] = 2
    config["experiment"]["bootstrap_resamples"] = 3
    config["experiment"]["n_jobs"] = 1
    config["models"]["include"] = ["dummy"]
    config["explainability"]["permutation_repeats"] = 1
    return config


def _write_config(path, config: dict[str, object]) -> None:
    path.write_text(yaml.safe_dump(config), encoding="utf-8")


@pytest.mark.parametrize(
    ("calibration", "message"),
    [
        ({"variants": ["sigmoid"]}, "include 'uncalibrated'"),
        ({"selection_metric": "roc_auc"}, "selection_metric must be brier"),
    ],
)
def test_calibration_configuration_rejects_unsupported_protocols(
    tmp_path, calibration: dict[str, object], message: str
) -> None:
    config = _small_config()
    config["calibration"].update(calibration)
    path = tmp_path / "invalid.yaml"
    _write_config(path, config)

    with pytest.raises(ConfigurationError, match=message):
        load_config(path)


def test_uncalibrated_only_protocol_is_honored_by_validation_and_final_refit(
    synthetic_data, tmp_path
) -> None:
    features, target = synthetic_data
    config = _small_config()
    config["calibration"]["variants"] = ["uncalibrated"]
    config["threshold"]["fixed"] = 0.37
    config["threshold"]["optimize_training_only"] = False
    run_dir = run_nested_experiment(features, target, config, artifact_root=tmp_path / "run")

    predictions = pd.read_csv(run_dir / "oof_predictions_all.csv")
    assert set(predictions["variant"]) == {"uncalibrated", "training_selected"}
    folds = pd.read_csv(run_dir / "fold_metrics.csv")
    assert folds["selected_threshold"].eq(0.37).all()
    assert folds["threshold_source"].eq("fixed_configured").all()
    assert not folds["threshold_optimization_enabled"].any()
    assert folds["training_brier_sigmoid"].isna().all()

    aggregate = pd.read_csv(run_dir / "aggregated_metrics.csv")
    assert "training_threshold_balanced_accuracy" in aggregate
    assert "training_threshold_roc_auc" not in aggregate
    assert "training_threshold_pr_auc" not in aggregate
    assert "training_threshold_brier" not in aggregate
    assert "training_threshold_calibration_ece_5_quantile_bins" not in aggregate
    patient = pd.read_csv(run_dir / "oof_patient_dummy_training_selected.csv")
    assert "training_threshold_vote_fraction" in patient
    assert "training_threshold_vote" not in patient

    (run_dir / "selection.json").write_text(
        json.dumps({"model": "dummy", "variant": "training_selected"}),
        encoding="utf-8",
    )
    model_path = fit_final_model(features, target, run_dir, output_path=tmp_path / "final.joblib")
    bundle = joblib.load(model_path)
    assert bundle["metadata"]["final_calibration_status"] == "uncalibrated"
    assert set(bundle["metadata"]["training_only_calibration_brier"]) == {"uncalibrated"}


@pytest.mark.parametrize(
    ("selection", "message"),
    [
        (
            {"model": "reduced_clinical_logistic", "variant": "training_selected"},
            "not configured",
        ),
        (
            {"model": "dummy", "variant": "uncalibrated"},
            "does not match the configured candidate variant",
        ),
        (["dummy"], "must be a mapping"),
    ],
)
def test_final_refit_rejects_selection_records_outside_the_run_configuration(
    synthetic_data, tmp_path, selection: object, message: str
) -> None:
    """Final refits reject stale or malformed selection records before model fitting."""

    features, target = synthetic_data
    config = _small_config()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_config(run_dir / "config.yaml", config)
    (run_dir / "selection.json").write_text(json.dumps(selection), encoding="utf-8")
    output_path = tmp_path / "final.joblib"

    with pytest.raises(ValueError, match=message):
        fit_final_model(features, target, run_dir, output_path=output_path)

    assert not output_path.exists()


def test_threshold_optimization_receives_the_configured_objective(monkeypatch) -> None:
    seen: list[str] = []

    def fake_optimizer(y_true: pd.Series, probabilities: np.ndarray, *, objective: str) -> float:
        seen.append(objective)
        return 0.42

    monkeypatch.setattr("hcc_survival.evaluation._optimize_threshold_training_only", fake_optimizer)
    threshold, source = _select_threshold(
        pd.Series([0, 1, 0, 1]),
        np.array([0.1, 0.8, 0.3, 0.9]),
        {"fixed": 0.5, "optimize_training_only": True, "objective": "f1"},
    )

    assert threshold == 0.42
    assert source == "training_oof_optimized"
    assert seen == ["f1"]


def test_missingness_sensitivity_validates_in_memory_config_before_creating_output(
    synthetic_data, tmp_path
) -> None:
    """Direct sensitivity calls validate configuration before creating local output."""

    features, target = synthetic_data
    config = _small_config()
    config["experiment"]["outer_repeat"] = 1
    output_dir = tmp_path / "sensitivity"

    with pytest.raises(ConfigurationError, match=r"experiment\.outer_repeat"):
        run_missingness_sensitivity(features, target, config, output_dir)

    assert not output_dir.exists()


def test_missingness_exclusion_is_derived_per_outer_training_partition(
    synthetic_data, tmp_path
) -> None:
    features, target = synthetic_data
    features = features.copy()
    config = _small_config()
    config["experiment"]["outer_folds"] = 3
    outer = RepeatedStratifiedKFold(n_splits=3, n_repeats=1, random_state=2025)
    first_train, first_validation = next(outer.split(features, target))
    features.loc[first_validation, "Age"] = np.nan
    features.loc[first_train[:5], "Age"] = np.nan
    assert float(features["Age"].isna().mean()) > 0.40

    output = run_missingness_sensitivity(features, target, config, tmp_path / "sensitivity")
    metadata = pd.read_csv(output / "missingness_sensitivity_fold_metadata.csv")
    excluded = metadata[
        (metadata["configuration"] == "exclude_gt_40pct_with_numeric_indicators")
        & (metadata["repeat"] == 0)
        & (metadata["fold"] == 0)
    ].iloc[0]
    assert excluded["n_features_retained"] == features.shape[1]
    assert "Age" not in json.loads(excluded["excluded_features"])
    assert {"n_features_retained", "n_features_excluded", "excluded_features"} <= set(
        metadata.columns
    )

    result = pd.read_csv(output / "missingness_sensitivity_results.csv")
    selected = result.loc[
        result["configuration"] == "exclude_gt_40pct_with_numeric_indicators"
    ].iloc[0]
    counts = metadata.loc[
        metadata["configuration"] == "exclude_gt_40pct_with_numeric_indicators",
        "n_features_retained",
    ]
    assert selected["n_features_min_across_outer_training_partitions"] == counts.min()
    assert selected["n_features_max_across_outer_training_partitions"] == counts.max()
