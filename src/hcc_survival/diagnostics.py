"""Post-experiment diagnostic tables and figures derived from saved artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    auc,
    precision_recall_curve,
    roc_curve,
)

from hcc_survival.artifacts import ensure_local_output_path
from hcc_survival.constants import DEFAULT_MODEL_PATH
from hcc_survival.metrics import calibration_table
from hcc_survival.schemas import PLAUSIBILITY_BOUNDS
from hcc_survival.subgroup import exploratory_subgroup_metrics, predefined_subgroups

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _selected_predictions(run_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    selection = json.loads((run_dir / "selection.json").read_text(encoding="utf-8"))
    predictions = pd.read_csv(run_dir / "oof_predictions_all.csv")
    selected = predictions[
        (predictions["model"] == selection["model"])
        & (predictions["variant"] == selection["variant"])
    ]
    return selected, selection


def _patient_predictions(group: pd.DataFrame) -> pd.DataFrame:
    return (
        group.groupby("patient_index", as_index=False)
        .agg(
            observed=("observed", "first"),
            probability_survived_one_year=("probability_survived_one_year", "mean"),
        )
        .sort_values("patient_index")
    )


def generate_curve_figures(run_dir: Path | str) -> None:
    """Generate ROC, PR, calibration, confusion, and stability figures."""

    run_dir = ensure_local_output_path(run_dir, purpose="Diagnostic output")
    figures = run_dir / "figures"
    figures.mkdir(exist_ok=True)
    predictions = pd.read_csv(run_dir / "oof_predictions_all.csv")
    serious = predictions[predictions["variant"] == "training_selected"]
    plt.figure(figsize=(7, 6))
    for model, group in serious.groupby("model"):
        patient = _patient_predictions(group)
        false_positive, true_positive, _ = roc_curve(
            patient["observed"], patient["probability_survived_one_year"]
        )
        plt.plot(
            false_positive,
            true_positive,
            label=f"{model} (AUC={auc(false_positive, true_positive):.3f})",
        )
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Chance")
    plt.xlabel("False-positive rate among deaths")
    plt.ylabel("Survivor sensitivity")
    plt.title("Patient-aggregated nested-CV ROC curves")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(figures / "roc_curves.png", dpi=300)
    plt.close()

    plt.figure(figsize=(7, 6))
    for model, group in serious.groupby("model"):
        patient = _patient_predictions(group)
        precision, recall, _ = precision_recall_curve(
            patient["observed"], patient["probability_survived_one_year"]
        )
        plt.plot(recall, precision, label=model)
    plt.xlabel("Survivor sensitivity")
    plt.ylabel("Positive predictive value")
    plt.title("Patient-aggregated nested-CV precision-recall curves")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(figures / "precision_recall_curves.png", dpi=300)
    plt.close()

    variants = [
        ("logistic_regression", "uncalibrated"),
        ("logistic_regression", "sigmoid"),
        ("random_forest", "uncalibrated"),
        ("random_forest", "sigmoid"),
        ("xgboost", "uncalibrated"),
        ("xgboost", "sigmoid"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), sharex=True, sharey=True)
    for axis, (model, variant) in zip(axes.flat, variants, strict=True):
        group = predictions[(predictions["model"] == model) & (predictions["variant"] == variant)]
        if group.empty:
            axis.set_visible(False)
            continue
        patient = _patient_predictions(group)
        bins = calibration_table(
            patient["observed"].to_numpy(),
            patient["probability_survived_one_year"].to_numpy(),
        )
        axis.plot([0, 1], [0, 1], "--", color="gray")
        axis.plot(bins["mean_predicted"], bins["observed_survival"], marker="o")
        axis.set_title(f"{model}\n{variant}", fontsize=9)
        axis.set_xlabel("Mean predicted survival")
        axis.set_ylabel("Observed survival")
    fig.suptitle("Five-bin equal-frequency calibration curves")
    fig.tight_layout()
    fig.savefig(figures / "calibration_curves.png", dpi=300)
    plt.close(fig)

    for model, group in serious.groupby("model"):
        patient = _patient_predictions(group)
        predicted = (patient["probability_survived_one_year"] >= 0.5).astype(int)
        display = ConfusionMatrixDisplay.from_predictions(
            patient["observed"],
            predicted,
            labels=[0, 1],
            display_labels=["Died", "Survived"],
            cmap="Blues",
            colorbar=False,
        )
        display.ax_.set_title(f"{model}: fixed threshold 0.50")
        display.ax_.set_xlabel("Predicted one-year outcome")
        display.ax_.set_ylabel("Observed one-year outcome")
        display.figure_.tight_layout()
        display.figure_.savefig(figures / f"confusion_matrix_{model}.png", dpi=300)
        plt.close(display.figure_)

    fold = pd.read_csv(run_dir / "fold_metrics.csv")
    fold = fold[fold["variant"] == "training_selected"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for axis, metric, label in zip(
        axes,
        ("roc_auc", "pr_auc", "brier"),
        ("ROC-AUC", "PR-AUC", "Brier score"),
        strict=True,
    ):
        for model, group in fold.groupby("model"):
            axis.plot(range(len(group)), group[metric], marker="o", label=model)
        axis.set_title(f"{label} by outer fold")
        axis.set_xlabel("Sequential outer fold")
        axis.set_ylabel(label)
    axes[-1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(figures / "fold_stability.png", dpi=300)
    plt.close(fig)


def generate_experiment_tables(
    run_dir: Path | str, features: pd.DataFrame, target: pd.Series
) -> None:
    """Create model ranking, subgroup, missingness, quality, and error tables."""

    run_dir = ensure_local_output_path(run_dir, purpose="Diagnostic output")
    tables = run_dir / "tables"
    tables.mkdir(exist_ok=True)
    predictions = pd.read_csv(run_dir / "oof_predictions_all.csv")
    repetition = pd.read_csv(tables / "repetition_metrics.csv")
    selected_repetition = repetition[repetition["variant"] == "training_selected"].copy()
    selected_repetition["rank_roc_auc"] = selected_repetition.groupby("repeat")["roc_auc"].rank(
        method="min", ascending=False
    )
    selected_repetition.to_csv(tables / "model_rankings_by_repetition.csv", index=False)

    fold = pd.read_csv(run_dir / "fold_metrics.csv")
    fold[fold["variant"] == "training_selected"][
        [
            "model",
            "repeat",
            "fold",
            "calibration_decision",
            "selected_threshold",
            "best_params",
            "roc_auc",
            "pr_auc",
            "brier",
            "survivor_sensitivity",
            "death_specificity",
        ]
    ].to_csv(tables / "fold_stability_and_decisions.csv", index=False)

    selected_all, selection = _selected_predictions(run_dir)
    patient = _patient_predictions(selected_all)
    for name, subgroup in predefined_subgroups(features).items():
        exploratory_subgroup_metrics(patient, subgroup).to_csv(
            tables / f"exploratory_subgroup_{name}.csv", index=False
        )

    missingness = features.isna().sum(axis=1)
    missing_by_outcome = (
        pd.DataFrame({"outcome": target, "missing_feature_count": missingness})
        .groupby("outcome")["missing_feature_count"]
        .agg(["count", "mean", "median", "std"])
        .reset_index()
    )
    missing_by_outcome.to_csv(tables / "missingness_by_outcome.csv", index=False)

    patient = patient.set_index("patient_index")
    all_models = {
        model: _patient_predictions(group).set_index("patient_index")[
            "probability_survived_one_year"
        ]
        for model, group in predictions[predictions["variant"] == "training_selected"].groupby(
            "model"
        )
    }
    probability_frame = pd.DataFrame(all_models)
    selected_probability = patient["probability_survived_one_year"]
    observed = patient["observed"]
    incorrect = (selected_probability >= 0.5).astype(int) != observed
    high_confidence_incorrect = incorrect & (
        ((selected_probability >= 0.8) & (observed == 0))
        | ((selected_probability <= 0.2) & (observed == 1))
    )
    near_threshold = (selected_probability - 0.5).abs() <= 0.05
    disagreement = probability_frame.max(axis=1) - probability_frame.min(axis=1)
    extensive_missingness = missingness >= 10
    plausibility_mask = pd.Series(False, index=features.index)
    for feature, (low, high) in PLAUSIBILITY_BOUNDS.items():
        values = pd.to_numeric(features[feature], errors="coerce")
        if low is not None:
            plausibility_mask |= values < low
        if high is not None:
            plausibility_mask |= values > high
    logistic_rf_disagreement = pd.Series(False, index=patient.index)
    if {"logistic_regression", "random_forest"} <= set(probability_frame.columns):
        logistic_rf_disagreement = (probability_frame["logistic_regression"] >= 0.5) != (
            probability_frame["random_forest"] >= 0.5
        )
    error_summary = pd.DataFrame(
        [
            ["incorrect_at_0.50", int(incorrect.sum())],
            ["high_confidence_incorrect", int(high_confidence_incorrect.sum())],
            ["within_0.05_of_threshold", int(near_threshold.sum())],
            ["model_disagreement_ge_0.30", int((disagreement >= 0.30).sum())],
            ["extensive_missingness_ge_10_features", int(extensive_missingness.sum())],
            ["plausibility_flag_cases", int(plausibility_mask.sum())],
            [
                "logistic_random_forest_classification_disagreement",
                int(logistic_rf_disagreement.sum()),
            ],
            [
                "mean_missing_features_in_incorrect_cases",
                float(missingness.reindex(incorrect.index)[incorrect].mean()),
            ],
            [
                "mean_missing_features_in_correct_cases",
                float(missingness.reindex(incorrect.index)[~incorrect].mean()),
            ],
        ],
        columns=["anonymous_pattern", "value"],
    )
    error_summary["selected_model"] = selection["model"]
    error_summary.to_csv(tables / "anonymous_error_analysis.csv", index=False)


def generate_coefficient_table(
    model_path: Path | str = DEFAULT_MODEL_PATH,
    output_path: Path | str | None = None,
) -> Path | None:
    """Export descriptive coefficients when the final fitted model is logistic."""

    bundle = joblib.load(model_path)
    if "logistic" not in bundle["metadata"]["model_name"]:
        return None
    model = bundle["model"]
    pipelines = []
    if hasattr(model, "calibrated_classifiers_"):
        pipelines = [item.estimator for item in model.calibrated_classifiers_]
    elif hasattr(model, "named_steps"):
        pipelines = [model]
    if not pipelines:
        return None
    names = pipelines[0].named_steps["preprocess"].get_feature_names_out()
    coefficients = np.vstack([pipeline.named_steps["model"].coef_[0] for pipeline in pipelines])
    frame = pd.DataFrame(
        {
            "transformed_feature": names,
            "coefficient_mean": coefficients.mean(axis=0),
            "coefficient_std_across_calibration_fits": coefficients.std(axis=0),
        }
    )
    frame["odds_ratio"] = np.exp(frame["coefficient_mean"])
    frame["interpretation"] = np.where(
        frame["transformed_feature"].str.startswith("numeric__"),
        "Effect per one-standard-deviation increase",
        "Compared with the encoded reference category; see schema",
    )
    frame["warning"] = (
        "Descriptive model association; correlated predictors can destabilize coefficients; "
        "not causal."
    )
    path = (
        Path(output_path)
        if output_path
        else Path(model_path).parent / "final_logistic_coefficients.csv"
    )
    path = ensure_local_output_path(path, purpose="Coefficient-table output")
    frame.to_csv(path, index=False)
    return path


def run_diagnostics(
    run_dir: Path | str,
    features: pd.DataFrame,
    target: pd.Series,
    *,
    model_path: Path | str | None = None,
) -> None:
    """Generate all post-experiment diagnostics from immutable artifacts."""

    run_dir = ensure_local_output_path(run_dir, purpose="Diagnostic output")
    generate_curve_figures(run_dir)
    generate_experiment_tables(run_dir, features, target)
    if model_path and Path(model_path).exists():
        generate_coefficient_table(
            model_path,
            Path(run_dir) / "tables" / "final_logistic_coefficients.csv",
        )
