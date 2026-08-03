"""Regression tests for the predeclared model-selection behavior."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hcc_survival.config import load_config
from hcc_survival.reporting import select_model


def _metrics(rows: list[dict[str, object]]) -> pd.DataFrame:
    for row in rows:
        row.setdefault("calibration_ece_5_quantile_bins", 0.05)
        row.setdefault("roc_auc_repetition_std", 0.01)
        row.setdefault("brier_repetition_std", 0.01)
    return pd.DataFrame(rows)


def _rule() -> dict[str, object]:
    return load_config("configs/fast.yaml")["selection"]


def test_clear_primary_auc_advantage_beats_simplicity() -> None:
    metrics = _metrics(
        [
            {
                "model": "logistic_regression",
                "variant": "training_selected",
                "roc_auc": 0.79,
                "pr_auc": 0.90,
                "brier": 0.20,
            },
            {
                "model": "random_forest",
                "variant": "training_selected",
                "roc_auc": 0.84,
                "pr_auc": 0.91,
                "brier": 0.18,
            },
        ]
    )
    selected, trace = select_model(metrics, _rule())
    assert selected["model"] == "random_forest"
    assert trace["metric_directions"]["roc_auc"] == "higher"
    assert trace["metric_directions"]["brier"] == "lower"


def test_simple_model_wins_after_all_equivalence_gates() -> None:
    metrics = _metrics(
        [
            {
                "model": "logistic_regression",
                "variant": "training_selected",
                "roc_auc": 0.800,
                "pr_auc": 0.800,
                "brier": 0.200,
            },
            {
                "model": "random_forest",
                "variant": "training_selected",
                "roc_auc": 0.809,
                "pr_auc": 0.809,
                "brier": 0.191,
            },
        ]
    )
    selected, trace = select_model(metrics, _rule())
    assert selected["model"] == "logistic_regression"
    assert set(trace["candidates_after_brier"]) == {
        "logistic_regression",
        "random_forest",
    }


def test_selection_is_deterministic_for_identical_input() -> None:
    metrics = _metrics(
        [
            {
                "model": "logistic_regression",
                "variant": "training_selected",
                "roc_auc": 0.80,
                "pr_auc": 0.81,
                "brier": 0.19,
            },
            {
                "model": "random_forest",
                "variant": "training_selected",
                "roc_auc": 0.81,
                "pr_auc": 0.81,
                "brier": 0.19,
            },
        ]
    )
    first, _ = select_model(metrics, _rule())
    second, _ = select_model(
        metrics.sample(frac=1, random_state=19).reset_index(drop=True), _rule()
    )
    assert (first["model"], first["variant"]) == (second["model"], second["variant"])


def test_missing_required_metric_is_rejected() -> None:
    metrics = _metrics(
        [
            {
                "model": "logistic_regression",
                "variant": "training_selected",
                "roc_auc": 0.80,
                "brier": 0.20,
            },
        ]
    )
    with pytest.raises(ValueError, match="missing"):
        select_model(metrics, _rule())


def test_non_finite_selection_metric_is_rejected() -> None:
    metrics = _metrics(
        [
            {
                "model": "logistic_regression",
                "variant": "training_selected",
                "roc_auc": np.inf,
                "pr_auc": 0.80,
                "brier": 0.20,
            },
            {
                "model": "random_forest",
                "variant": "training_selected",
                "roc_auc": 0.70,
                "pr_auc": 0.80,
                "brier": 0.20,
            },
        ]
    )
    with pytest.raises(ValueError, match="finite"):
        select_model(metrics, _rule())


def test_lower_brier_is_preferred_after_discrimination_equivalence() -> None:
    metrics = _metrics(
        [
            {
                "model": "logistic_regression",
                "variant": "training_selected",
                "roc_auc": 0.80,
                "pr_auc": 0.81,
                "brier": 0.23,
            },
            {
                "model": "random_forest",
                "variant": "training_selected",
                "roc_auc": 0.80,
                "pr_auc": 0.81,
                "brier": 0.18,
            },
        ]
    )
    selected, trace = select_model(metrics, _rule())
    assert selected["model"] == "random_forest"
    assert trace["candidates_after_brier"] == ["random_forest"]


def test_lower_calibration_error_breaks_metric_equivalence_before_simplicity() -> None:
    metrics = _metrics(
        [
            {
                "model": "logistic_regression",
                "variant": "training_selected",
                "roc_auc": 0.80,
                "pr_auc": 0.81,
                "brier": 0.20,
                "calibration_ece_5_quantile_bins": 0.10,
            },
            {
                "model": "random_forest",
                "variant": "training_selected",
                "roc_auc": 0.80,
                "pr_auc": 0.81,
                "brier": 0.20,
                "calibration_ece_5_quantile_bins": 0.05,
            },
        ]
    )
    selected, _ = select_model(metrics, _rule())
    assert selected["model"] == "random_forest"


def test_configured_primary_metric_drives_the_first_gate() -> None:
    metrics = _metrics(
        [
            {
                "model": "logistic_regression",
                "variant": "training_selected",
                "roc_auc": 0.82,
                "pr_auc": 0.80,
                "brier": 0.20,
            },
            {
                "model": "random_forest",
                "variant": "training_selected",
                "roc_auc": 0.81,
                "pr_auc": 0.84,
                "brier": 0.20,
            },
        ]
    )
    rule = _rule()
    rule["primary_metric"] = "pr_auc"
    rule["primary_direction"] = "higher"
    rule["secondary_metrics"] = ["roc_auc", "brier"]

    selected, trace = select_model(metrics, rule)

    assert selected["model"] == "random_forest"
    assert trace["primary_metric"] == "pr_auc"
    assert trace["candidates_after_pr_auc"] == ["random_forest"]


def test_non_finite_ece_is_rejected_when_ece_gate_is_configured() -> None:
    metrics = _metrics(
        [
            {
                "model": "logistic_regression",
                "variant": "training_selected",
                "roc_auc": 0.80,
                "pr_auc": 0.81,
                "brier": 0.20,
                "calibration_ece_5_quantile_bins": np.nan,
            },
        ]
    )

    with pytest.raises(ValueError, match="finite"):
        select_model(metrics, _rule())


def test_missing_configured_stability_metric_is_rejected() -> None:
    metrics = pd.DataFrame(
        [
            {
                "model": "logistic_regression",
                "variant": "training_selected",
                "roc_auc": 0.80,
                "pr_auc": 0.81,
                "brier": 0.20,
                "calibration_ece_5_quantile_bins": 0.05,
            },
        ]
    )

    with pytest.raises(ValueError, match="missing"):
        select_model(metrics, _rule())
