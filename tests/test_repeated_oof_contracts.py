"""Tests for repeated nested-CV assignment and patient-level aggregation."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from hcc_survival.config import load_config
from hcc_survival.evaluation import run_nested_experiment


def _small_config(*, folds: int, repeats: int) -> dict[str, object]:
    config = load_config("configs/fast.yaml")
    config["experiment"]["outer_folds"] = folds
    config["experiment"]["outer_repeats"] = repeats
    config["experiment"]["inner_folds"] = 2
    config["experiment"]["bootstrap_resamples"] = 3
    config["models"]["include"] = ["dummy"]
    config["calibration"]["variants"] = ["uncalibrated"]
    config["explainability"]["permutation_repeats"] = 1
    return config


def test_repeated_oof_has_one_assignment_per_patient_per_repeat(synthetic_data, tmp_path) -> None:
    features, target = synthetic_data
    run_dir = run_nested_experiment(
        features,
        target,
        _small_config(folds=3, repeats=2),
        artifact_root=tmp_path,
    )
    raw = pd.read_csv(run_dir / "oof_predictions_all.csv")
    selected = raw[raw["variant"] == "training_selected"]
    assert selected["patient_index"].nunique() == len(features)
    counts = selected.groupby(["patient_index", "repeat"]).size()
    assert (counts == 1).all()
    assert set(selected["repeat"]) == {0, 1}
    assert len(selected) == len(features) * 2

    patient = pd.read_csv(run_dir / "oof_patient_dummy_training_selected.csv")
    assert patient["prediction_count"].eq(2).all()
    expected = (
        selected.groupby("patient_index")["probability_survived_one_year"].mean().sort_index()
    )
    np.testing.assert_allclose(
        patient.set_index("patient_index")["probability_survived_one_year"].sort_index(),
        expected,
    )


def test_full_repeated_design_gives_exactly_five_predictions_per_patient(
    synthetic_data, tmp_path
) -> None:
    features, target = synthetic_data
    run_dir = run_nested_experiment(
        features,
        target,
        _small_config(folds=5, repeats=5),
        artifact_root=tmp_path,
    )
    raw = pd.read_csv(run_dir / "oof_predictions_all.csv")
    for _, variant in raw.groupby("variant"):
        assert len(variant) == len(features) * 5
        assert variant.groupby("patient_index").size().eq(5).all()
        assert variant.groupby(["patient_index", "repeat"]).size().eq(1).all()


def test_outer_fold_train_and_validation_indices_do_not_overlap(synthetic_data, tmp_path) -> None:
    features, target = synthetic_data
    run_dir = run_nested_experiment(
        features,
        target,
        _small_config(folds=3, repeats=1),
        artifact_root=tmp_path,
    )
    folds = pd.read_csv(run_dir / "fold_metrics.csv")
    for row in folds.itertuples(index=False):
        train = set(json.loads(row.train_indices))
        validation = set(json.loads(row.validation_indices))
        assert train.isdisjoint(validation)
