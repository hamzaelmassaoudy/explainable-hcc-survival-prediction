"""Discrimination, classification, and calibration metrics."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _validate_binary_outcomes(y_true: np.ndarray) -> np.ndarray:
    """Return non-empty binary one-year survival outcomes as integers."""

    try:
        outcomes = np.asarray(y_true, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Outcomes must be encoded as binary numeric values.") from exc
    if outcomes.ndim != 1 or outcomes.size == 0:
        raise ValueError("Outcomes must be a non-empty one-dimensional array.")
    if not np.isfinite(outcomes).all() or not np.isin(outcomes, (0.0, 1.0)).all():
        raise ValueError(
            "Outcomes must be encoded as 0 (died within one year) or 1 (survived at one year)."
        )
    return outcomes.astype(int)


def _validate_probability_inputs(
    y_true: np.ndarray, probabilities: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Validate aligned binary outcomes and finite one-year survival probabilities."""

    outcomes = _validate_binary_outcomes(y_true)
    try:
        values = np.asarray(probabilities, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Probabilities must be numeric values.") from exc
    if values.ndim != 1 or values.shape != outcomes.shape:
        raise ValueError("Probabilities must be a one-dimensional array aligned with outcomes.")
    if not np.isfinite(values).all():
        raise ValueError("Probabilities must be finite.")
    if (values < 0.0).any() or (values > 1.0).any():
        raise ValueError("Probabilities must be between 0 and 1.")
    return outcomes, values


def _validate_threshold(threshold: float) -> float:
    """Validate a fixed probability threshold used for decision-only metrics."""

    try:
        value = float(threshold)
    except (TypeError, ValueError) as exc:
        raise ValueError("Threshold must be a finite value between 0 and 1.") from exc
    if not np.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError("Threshold must be a finite value between 0 and 1.")
    return value


def calibration_table(
    y_true: np.ndarray, probabilities: np.ndarray, n_bins: int = 5
) -> pd.DataFrame:
    """Return equal-frequency calibration bins without discarding duplicate edges."""

    frame = pd.DataFrame({"y": y_true, "p": probabilities}).dropna()
    if frame.empty:
        return pd.DataFrame(
            columns=["bin", "n", "mean_predicted", "observed_survival", "absolute_gap"]
        )
    if frame["p"].nunique() == 1:
        frame["bin"] = 1
    else:
        frame["bin"] = (
            pd.qcut(
                frame["p"],
                q=min(n_bins, len(frame)),
                labels=False,
                duplicates="drop",
            )
            + 1
        )
    result = (
        frame.groupby("bin", observed=True)
        .agg(
            n=("y", "size"),
            mean_predicted=("p", "mean"),
            observed_survival=("y", "mean"),
        )
        .reset_index()
    )
    result["absolute_gap"] = abs(result["observed_survival"] - result["mean_predicted"])
    return result


def calibration_error(
    y_true: np.ndarray, probabilities: np.ndarray, n_bins: int = 5
) -> tuple[float, int]:
    """Compute descriptive ECE with equal-frequency bins."""

    bins = calibration_table(y_true, probabilities, n_bins=n_bins)
    if bins.empty:
        return float("nan"), 0
    ece = np.average(bins["absolute_gap"], weights=bins["n"])
    return float(ece), len(bins)


def calibration_slope_intercept(
    y_true: np.ndarray, probabilities: np.ndarray
) -> tuple[float, float]:
    """Estimate calibration intercept and slope; return NaN when not estimable."""

    if len(np.unique(y_true)) < 2 or len(y_true) < 30:
        return float("nan"), float("nan")
    clipped = np.clip(probabilities, 1e-6, 1 - 1e-6)
    try:
        model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000)
        model.fit(logit(clipped).reshape(-1, 1), y_true)
        return float(model.intercept_[0]), float(model.coef_[0, 0])
    except (ValueError, FloatingPointError):
        warnings.warn("Calibration intercept/slope could not be estimated.", stacklevel=2)
        return float("nan"), float("nan")


def decision_metrics(y_true: np.ndarray, decisions: np.ndarray) -> dict[str, float | int]:
    """Calculate decision-only metrics with survived-at-one-year as the positive class."""

    outcomes = _validate_binary_outcomes(y_true)
    predicted = np.asarray(decisions)
    if predicted.ndim != 1 or predicted.shape != outcomes.shape:
        raise ValueError("Decisions must be a one-dimensional array aligned with outcomes.")
    if not np.isin(predicted, (0, 1)).all():
        raise ValueError("Decisions must be encoded as 0 or 1.")
    predicted = predicted.astype(int)
    tn, fp, fn, tp = confusion_matrix(outcomes, predicted, labels=[0, 1]).ravel()
    death_specificity = tn / (tn + fp) if tn + fp else float("nan")
    npv = tn / (tn + fn) if tn + fn else float("nan")
    return {
        "accuracy": float(accuracy_score(outcomes, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(outcomes, predicted)),
        "survivor_sensitivity": float(recall_score(outcomes, predicted, zero_division=0)),
        "death_specificity": float(death_specificity),
        "positive_predictive_value": float(precision_score(outcomes, predicted, zero_division=0)),
        "negative_predictive_value": float(npv),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "true_survivors": int(tp),
        "false_non_survivors": int(fn),
        "true_deaths": int(tn),
        "false_survivors": int(fp),
        "predicted_survivor_count": int(tp + fp),
        "predicted_death_count": int(tn + fn),
    }


def classification_metrics(
    y_true: np.ndarray, probabilities: np.ndarray, threshold: float = 0.5
) -> dict[str, float | int]:
    """Calculate probability and fixed-threshold metrics for one-year survival."""

    outcomes, values = _validate_probability_inputs(y_true, probabilities)
    threshold_value = _validate_threshold(threshold)
    predicted = (values >= threshold_value).astype(int)
    has_both_outcomes = len(np.unique(outcomes)) == 2
    auc = roc_auc_score(outcomes, values) if has_both_outcomes else float("nan")
    pr_auc = average_precision_score(outcomes, values) if has_both_outcomes else float("nan")
    ece, usable_bins = calibration_error(outcomes, values)
    intercept, slope = calibration_slope_intercept(outcomes, values)
    return {
        "roc_auc": float(auc),
        "pr_auc": float(pr_auc),
        **decision_metrics(outcomes, predicted),
        "brier": float(brier_score_loss(outcomes, values)),
        "calibration_ece_5_quantile_bins": ece,
        "calibration_usable_bins": usable_bins,
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "threshold": threshold_value,
    }


BOOTSTRAP_METRIC_NAMES = (
    "roc_auc",
    "pr_auc",
    "brier",
    "accuracy",
    "balanced_accuracy",
    "survivor_sensitivity",
    "death_specificity",
    "positive_predictive_value",
    "negative_predictive_value",
    "f1",
    "calibration_ece_5_quantile_bins",
)


def bootstrap_confidence_intervals(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    *,
    n_resamples: int,
    seed: int,
) -> dict[str, dict[str, Any]]:
    """Patient-level bootstrap CIs for aggregated out-of-fold predictions."""

    rng = np.random.default_rng(seed)
    results: dict[str, dict[str, Any]] = {}
    indices = np.arange(len(y_true))
    sampled_values: dict[str, list[float]] = {name: [] for name in BOOTSTRAP_METRIC_NAMES}
    invalid_by_metric = {name: 0 for name in BOOTSTRAP_METRIC_NAMES}
    for _ in range(n_resamples):
        sample = rng.choice(indices, size=len(indices), replace=True)
        try:
            sampled = classification_metrics(y_true[sample], probabilities[sample])
        except ValueError:
            for name in BOOTSTRAP_METRIC_NAMES:
                invalid_by_metric[name] += 1
            continue
        for name in BOOTSTRAP_METRIC_NAMES:
            value = float(sampled[name])
            if np.isfinite(value):
                sampled_values[name].append(value)
            else:
                invalid_by_metric[name] += 1
    for name in BOOTSTRAP_METRIC_NAMES:
        values = sampled_values[name]
        invalid = invalid_by_metric[name]
        results[name] = {
            "lower_95": float(np.quantile(values, 0.025)) if values else float("nan"),
            "upper_95": float(np.quantile(values, 0.975)) if values else float("nan"),
            "requested_resamples": n_resamples,
            "valid_resamples": len(values),
            "invalid_resamples": invalid,
            "random_seed": seed,
            "method": "patient-level percentile bootstrap",
            "confidence_level": 0.95,
            "interpretation": "Internal-validation uncertainty; not external generalization.",
        }
    return results
