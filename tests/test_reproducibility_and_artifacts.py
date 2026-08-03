"""Reproducibility and model-artifact contract tests."""

from __future__ import annotations

import hashlib
import json

import joblib
import pandas as pd
from pandas.testing import assert_frame_equal

from hcc_survival.config import load_config
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
    bundle = {"model": model}
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
