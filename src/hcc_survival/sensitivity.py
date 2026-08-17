"""Predeclared missingness sensitivity analysis."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline

from hcc_survival.artifacts import ensure_local_output_path
from hcc_survival.config import validate_config
from hcc_survival.metrics import classification_metrics
from hcc_survival.preprocessing import build_preprocessor

MISSINGNESS_EXCLUSION_THRESHOLD = 0.40


def _pipeline(
    *,
    add_indicators: bool,
    feature_subset: tuple[str, ...] | None,
    seed: int,
) -> Pipeline:
    return Pipeline(
        [
            (
                "preprocess",
                build_preprocessor(
                    scale_numeric=True,
                    add_indicators=add_indicators,
                    feature_subset=feature_subset,
                ),
            ),
            (
                "model",
                LogisticRegression(solver="liblinear", max_iter=2000, random_state=seed),
            ),
        ]
    )


def _retained_features_from_training_partition(
    features: pd.DataFrame,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Derive the predeclared missingness subset from one outer-training partition."""

    retained = tuple(
        column
        for column in features.columns
        if float(features[column].isna().mean()) <= MISSINGNESS_EXCLUSION_THRESHOLD
    )
    excluded = tuple(column for column in features.columns if column not in retained)
    if not retained:
        raise ValueError("The training-partition missingness rule excluded every feature.")
    return retained, excluded


def run_missingness_sensitivity(
    features: pd.DataFrame,
    target: pd.Series,
    config: dict[str, Any],
    output_dir: Path | str,
) -> Path:
    """Compare three predeclared fold-safe missingness configurations."""

    config = validate_config(config)
    output_dir = ensure_local_output_path(output_dir, purpose="Sensitivity-analysis output")
    output_dir.mkdir(parents=True, exist_ok=True)
    seed = int(config["experiment"]["random_seed"])
    outer_folds = int(config["experiment"]["outer_folds"])
    outer_repeats = int(config["experiment"]["outer_repeats"])
    configurations = {
        "all_with_numeric_missing_indicators": {
            "add_indicators": True,
            "exclude_high_missing": False,
        },
        "all_without_missing_indicators": {
            "add_indicators": False,
            "exclude_high_missing": False,
        },
        "exclude_gt_40pct_with_numeric_indicators": {
            "add_indicators": True,
            "exclude_high_missing": True,
        },
    }
    outer = RepeatedStratifiedKFold(
        n_splits=outer_folds,
        n_repeats=outer_repeats,
        random_state=seed,
    )
    predictions: list[dict[str, Any]] = []
    fold_metadata: list[dict[str, Any]] = []
    for configuration_name, options in configurations.items():
        for fold_index, (train_idx, validation_idx) in enumerate(outer.split(features, target)):
            x_train = features.iloc[train_idx]
            repeat = fold_index // outer_folds
            fold = fold_index % outer_folds
            if bool(options["exclude_high_missing"]):
                feature_subset, excluded = _retained_features_from_training_partition(x_train)
                subset_source = "outer_training_partition_missingness"
            else:
                feature_subset = None
                excluded = ()
                subset_source = "all_validated_predictors"
            pipeline = _pipeline(
                add_indicators=bool(options["add_indicators"]),
                feature_subset=feature_subset,
                seed=seed + fold_index,
            )
            # This predeclared sensitivity comparison intentionally fixes the logistic
            # hyperparameters. That keeps missingness-derived feature selection wholly
            # inside the outer-training partition rather than leaking it into inner tuning.
            pipeline.fit(x_train, target.iloc[train_idx])
            probability = pipeline.predict_proba(features.iloc[validation_idx])[:, 1]
            fold_metadata.append(
                {
                    "configuration": configuration_name,
                    "repeat": repeat,
                    "fold": fold,
                    "feature_subset_source": subset_source,
                    "n_features_retained": len(feature_subset)
                    if feature_subset is not None
                    else features.shape[1],
                    "n_features_excluded": len(excluded),
                    "excluded_features": json.dumps(excluded),
                }
            )
            for position, patient_index in enumerate(validation_idx):
                predictions.append(
                    {
                        "configuration": configuration_name,
                        "patient_index": int(patient_index),
                        "repeat": repeat,
                        "fold": fold,
                        "observed": int(target.iloc[patient_index]),
                        "probability_survived_one_year": float(probability[position]),
                    }
                )
    prediction_frame = pd.DataFrame(predictions)
    prediction_frame.to_csv(output_dir / "missingness_sensitivity_oof.csv", index=False)
    fold_metadata_frame = pd.DataFrame(fold_metadata)
    fold_metadata_frame.to_csv(
        output_dir / "missingness_sensitivity_fold_metadata.csv", index=False
    )
    rows: list[dict[str, Any]] = []
    for name, group in prediction_frame.groupby("configuration"):
        patient = group.groupby("patient_index", as_index=False).agg(
            observed=("observed", "first"),
            probability_survived_one_year=("probability_survived_one_year", "mean"),
            prediction_count=("probability_survived_one_year", "size"),
        )
        metrics = classification_metrics(
            patient["observed"].to_numpy(),
            patient["probability_survived_one_year"].to_numpy(),
        )
        feature_counts = fold_metadata_frame.loc[
            fold_metadata_frame["configuration"] == name, "n_features_retained"
        ]
        rows.append(
            {
                "configuration": name,
                "n_features_mean_across_outer_training_partitions": float(feature_counts.mean()),
                "n_features_min_across_outer_training_partitions": int(feature_counts.min()),
                "n_features_max_across_outer_training_partitions": int(feature_counts.max()),
                **metrics,
            }
        )
    pd.DataFrame(rows).to_csv(output_dir / "missingness_sensitivity_results.csv", index=False)
    manifest = {
        "threshold": MISSINGNESS_EXCLUSION_THRESHOLD,
        "threshold_rationale": (
            "Predeclared pragmatic threshold: features missing for more than 40% of the "
            "cohort may contribute more missingness-process signal than measured biology."
        ),
        "feature_exclusion_rule": (
            "For the exclusion configuration, retained and excluded features are derived "
            "separately within each outer-training partition."
        ),
        "model_specification": (
            "Predeclared logistic regression (C=1.0, class_weight=None) without inner "
            "hyperparameter tuning, so the missingness-derived subset remains outer-fold local."
        ),
        "per_fold_metadata_file": "missingness_sensitivity_fold_metadata.csv",
        "configurations": {
            key: {
                "add_numeric_missing_indicators": bool(value["add_indicators"]),
                "feature_subset": (
                    "derived per outer-training partition"
                    if bool(value["exclude_high_missing"])
                    else "all validated predictors"
                ),
            }
            for key, value in configurations.items()
        },
        "warning": (
            "Missingness may encode test-ordering and care pathways; this analysis is "
            "predictive and does not establish a causal missingness mechanism."
        ),
    }
    (output_dir / "missingness_sensitivity_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return output_dir
