"""Reproducibility and model-artifact contract tests."""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import pytest
from matplotlib.axes import Axes
from pandas.testing import assert_frame_equal

import hcc_survival.diagnostics as diagnostics
from hcc_survival.config import load_config
from hcc_survival.diagnostics import generate_experiment_tables
from hcc_survival.evaluation import run_nested_experiment
from hcc_survival.models import build_model
from hcc_survival.prediction import load_model_bundle, predict_survival


def _deterministic_config() -> dict[str, object]:
    config = load_config("configs/fast.yaml")
    config["experiment"]["outer_folds"] = 3
    config["experiment"]["outer_repeats"] = 1
    config["experiment"]["bootstrap_resamples"] = 5
    config["models"]["include"] = ["dummy", "logistic_regression"]
    config["calibration"]["variants"] = ["uncalibrated"]
    config["explainability"]["permutation_repeats"] = 1
    return config


def test_fixed_seed_reproduces_patient_predictions_and_metrics(synthetic_data, tmp_path) -> None:
    features, target = synthetic_data
    first_dir = run_nested_experiment(
        features, target, _deterministic_config(), artifact_root=tmp_path / "first"
    )
    second_dir = run_nested_experiment(
        features, target, _deterministic_config(), artifact_root=tmp_path / "second"
    )
    assert_frame_equal(
        pd.read_csv(first_dir / "aggregated_metrics.csv"),
        pd.read_csv(second_dir / "aggregated_metrics.csv"),
        check_dtype=False,
    )
    assert_frame_equal(
        pd.read_csv(first_dir / "oof_predictions_all.csv"),
        pd.read_csv(second_dir / "oof_predictions_all.csv"),
        check_dtype=False,
    )


def test_missing_model_artifact_has_actionable_message(tmp_path) -> None:
    missing = tmp_path / "missing.joblib"
    try:
        load_model_bundle(missing)
    except FileNotFoundError as exc:
        message = str(exc)
    else:  # pragma: no cover - assertion gives a clearer failure than pytest.raises
        raise AssertionError("missing model artifact must raise FileNotFoundError")
    assert "hcc_survival train" in message
    assert "fit-final" in message


def test_save_load_prediction_consistency(synthetic_data, tmp_path) -> None:
    features, target = synthetic_data
    model, _ = build_model("logistic_regression", seed=11)
    model.fit(features, target)
    bundle = {"model": model, "feature_names": list(features.columns)}
    path = tmp_path / "model.joblib"
    joblib.dump(bundle, path)
    loaded = load_model_bundle(path)
    records = features.iloc[:4].copy()
    expected = predict_survival(bundle, records)
    actual = predict_survival(loaded, records)
    assert_frame_equal(expected, actual, check_dtype=False)


def test_run_metadata_declares_positive_class(synthetic_data, tmp_path) -> None:
    features, target = synthetic_data
    source = tmp_path / "validated_input.csv"
    source.write_bytes(b"synthetic,validated,input\n")
    run_dir = run_nested_experiment(
        features,
        target,
        _deterministic_config(),
        artifact_root=tmp_path,
        dataset_path=source,
    )
    metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["positive_class"] == "1 = survived at one year"
    assert metadata["dataset_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()


def test_diagnostic_missingness_aligns_with_validation_row_positions(
    synthetic_data, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Diagnostic summaries use the row positions saved in validation artifacts."""

    features, _ = synthetic_data
    source_index = pd.Index([101, 205, 309, 411])
    features = features.iloc[:4].copy().fillna(0)
    features.index = source_index
    features["Age"] = [50.0, 60.0, 70.0, 80.0]
    features["Gender"] = [0, 1, 0, 1]
    features.loc[101, "Age"] = np.nan
    features.loc[205, ["Age", "Gender"]] = np.nan
    features.loc[411, ["Age", "Gender", "Symptoms"]] = np.nan
    target = pd.Series([0, 1, 0, 1], index=source_index, name="Class")

    run_dir = tmp_path / "diagnostic_run"
    tables = run_dir / "tables"
    tables.mkdir(parents=True)
    config = _deterministic_config()
    config["threshold"]["fixed"] = 0.37
    (run_dir / "config.yaml").write_text(json.dumps(config), encoding="utf-8")
    pd.DataFrame(
        {
            "patient_index": [0, 1, 2, 3],
            "model": ["dummy"] * 4,
            "variant": ["training_selected"] * 4,
            "observed": [0, 1, 0, 1],
            "probability_survived_one_year": [0.8, 0.4, 0.1, 0.2],
        }
    ).to_csv(run_dir / "oof_predictions_all.csv", index=False)
    pd.DataFrame(
        {
            "model": ["dummy"],
            "variant": ["training_selected"],
            "repeat": [0],
            "roc_auc": [0.5],
        }
    ).to_csv(tables / "repetition_metrics.csv", index=False)
    pd.DataFrame(
        {
            "model": ["dummy"],
            "variant": ["training_selected"],
            "repeat": [0],
            "fold": [0],
            "calibration_decision": ["not_requested"],
            "selected_threshold": [0.5],
            "best_params": ["{}"],
            "roc_auc": [0.5],
            "pr_auc": [0.5],
            "brier": [0.25],
            "survivor_sensitivity": [0.5],
            "death_specificity": [0.5],
        }
    ).to_csv(run_dir / "fold_metrics.csv", index=False)
    (run_dir / "selection.json").write_text(
        json.dumps({"model": "dummy", "variant": "training_selected"}), encoding="utf-8"
    )

    observed_thresholds: list[float] = []
    original_metrics = diagnostics.exploratory_subgroup_metrics

    def capture_subgroup_threshold(
        predictions: pd.DataFrame, subgroup: pd.Series, *, threshold: float = 0.5
    ) -> pd.DataFrame:
        observed_thresholds.append(threshold)
        return original_metrics(predictions, subgroup, threshold=threshold)

    monkeypatch.setattr(diagnostics, "exploratory_subgroup_metrics", capture_subgroup_threshold)
    generate_experiment_tables(run_dir, features, target)

    summary = pd.read_csv(tables / "anonymous_error_analysis.csv").set_index("anonymous_pattern")
    assert observed_thresholds == [0.37, 0.37]
    assert summary.loc["incorrect_at_0.37", "value"] == 2.0
    assert summary.loc["within_0.05_of_threshold", "value"] == 1.0
    assert summary.loc["mean_missing_features_in_incorrect_cases", "value"] == 2.0
    assert summary.loc["mean_missing_features_in_correct_cases", "value"] == 1.0


def test_diagnostic_figures_use_the_saved_fixed_threshold(tmp_path, monkeypatch) -> None:
    """Confusion matrices use the fixed decision rule recorded with the run."""

    run_dir = tmp_path / "diagnostic_run"
    run_dir.mkdir()
    config = _deterministic_config()
    config["threshold"]["fixed"] = 0.37
    (run_dir / "config.yaml").write_text(json.dumps(config), encoding="utf-8")
    pd.DataFrame(
        {
            "patient_index": [0, 1, 2, 3],
            "model": ["dummy"] * 4,
            "variant": ["training_selected"] * 4,
            "observed": [0, 1, 0, 1],
            "probability_survived_one_year": [0.36, 0.37, 0.38, 0.2],
        }
    ).to_csv(run_dir / "oof_predictions_all.csv", index=False)
    pd.DataFrame(
        {
            "model": ["dummy"],
            "variant": ["training_selected"],
            "roc_auc": [0.5],
            "pr_auc": [0.5],
            "brier": [0.25],
        }
    ).to_csv(run_dir / "fold_metrics.csv", index=False)

    decisions: list[list[int]] = []
    titles: list[str] = []
    original_set_title = Axes.set_title

    class CapturingDisplay:
        """Small stand-in that records decisions while retaining real plot objects."""

        @staticmethod
        def from_predictions(_observed, predicted, **_kwargs):
            decisions.append(pd.Series(predicted).astype(int).tolist())
            figure, axis = diagnostics.plt.subplots()
            return SimpleNamespace(ax_=axis, figure_=figure)

    def capture_title(axis, label, *args, **kwargs):
        titles.append(label)
        return original_set_title(axis, label, *args, **kwargs)

    monkeypatch.setattr(diagnostics, "ConfusionMatrixDisplay", CapturingDisplay)
    monkeypatch.setattr(Axes, "set_title", capture_title)

    diagnostics.generate_curve_figures(run_dir)

    assert decisions == [[0, 1, 1, 0]]
    assert "dummy: fixed threshold 0.37" in titles
