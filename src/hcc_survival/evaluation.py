"""Repeated nested cross-validation with training-only model decisions."""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.inspection import permutation_importance
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import (
    GridSearchCV,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_predict,
)

from hcc_survival.artifacts import create_run_directory, write_json
from hcc_survival.config import validate_config
from hcc_survival.constants import DATASET_DOI
from hcc_survival.metrics import (
    bootstrap_confidence_intervals,
    calibration_table,
    classification_metrics,
    decision_metrics,
)
from hcc_survival.models import build_model

LOGGER = logging.getLogger(__name__)


def _splitter(folds: int, seed: int) -> StratifiedKFold:
    return StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)


def _fit_tuned(
    pipeline: Any,
    grid: Any,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    inner_folds: int,
    seed: int,
    n_jobs: int,
) -> tuple[Any, dict[str, Any]]:
    """Tune only within the outer-training partition."""

    if not grid:
        fitted = clone(pipeline).fit(x_train, y_train)
        return fitted, {}
    search = GridSearchCV(
        clone(pipeline),
        grid,
        scoring="roc_auc",
        cv=_splitter(inner_folds, seed),
        n_jobs=n_jobs,
        refit=True,
        error_score="raise",
    )
    search.fit(x_train, y_train)
    return search.best_estimator_, search.best_params_


def _training_only_variants(
    tuned: Any,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    inner_folds: int,
    seed: int,
    n_jobs: int,
    calibration: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray], str, dict[str, float]]:
    """Fit variants and choose calibration using inner out-of-fold predictions only."""

    configured_variants = calibration.get("variants")
    if not isinstance(configured_variants, list) or "uncalibrated" not in configured_variants:
        raise ValueError("calibration.variants must include 'uncalibrated'.")
    unsupported = set(configured_variants) - {"uncalibrated", "sigmoid"}
    if unsupported:
        raise ValueError(f"Unsupported calibration variants: {sorted(unsupported)}.")
    if calibration.get("selection_metric") != "brier":
        raise ValueError("Only Brier-score calibration selection is supported.")
    minimum_brier_improvement = float(calibration["minimum_brier_improvement"])

    uncalibrated_oof = cross_val_predict(
        clone(tuned),
        x_train,
        y_train,
        cv=_splitter(inner_folds, seed + 101),
        method="predict_proba",
        n_jobs=n_jobs,
    )[:, 1]
    training_brier = {
        "uncalibrated": float(brier_score_loss(y_train, uncalibrated_oof)),
    }
    variants = {"uncalibrated": tuned}
    oof = {"uncalibrated": uncalibrated_oof}
    selected = "uncalibrated"
    if "sigmoid" in configured_variants:
        calibrated_template = CalibratedClassifierCV(
            estimator=clone(tuned),
            method="sigmoid",
            cv=_splitter(inner_folds, seed + 202),
        )
        sigmoid_oof = cross_val_predict(
            calibrated_template,
            x_train,
            y_train,
            cv=_splitter(inner_folds, seed + 303),
            method="predict_proba",
            n_jobs=n_jobs,
        )[:, 1]
        calibrated = clone(calibrated_template).fit(x_train, y_train)
        training_brier["sigmoid"] = float(brier_score_loss(y_train, sigmoid_oof))
        improvement = training_brier["uncalibrated"] - training_brier["sigmoid"]
        if improvement >= minimum_brier_improvement:
            selected = "sigmoid"
        variants["sigmoid"] = calibrated
        oof["sigmoid"] = sigmoid_oof
    return variants, oof, selected, training_brier


def _optimize_threshold_training_only(
    y_true: pd.Series,
    probabilities: np.ndarray,
    *,
    objective: str = "balanced_accuracy",
) -> float:
    """Select a configured decision objective using training OOF probabilities only."""

    candidates = np.linspace(0.1, 0.9, 81)
    scores = np.asarray(
        [
            decision_metrics(y_true.to_numpy(), (probabilities >= float(value)).astype(int)).get(
                objective, float("nan")
            )
            for value in candidates
        ],
        dtype=float,
    )
    if not np.isfinite(scores).any():
        raise ValueError(f"Threshold objective {objective!r} was not estimable.")
    return float(candidates[int(np.nanargmax(scores))])


def _select_threshold(
    y_true: pd.Series,
    probabilities: np.ndarray,
    threshold_config: dict[str, Any],
) -> tuple[float, str]:
    """Use either a fixed threshold or a training-only optimized decision threshold."""

    if bool(threshold_config["optimize_training_only"]):
        return (
            _optimize_threshold_training_only(
                y_true,
                probabilities,
                objective=str(threshold_config["objective"]),
            ),
            "training_oof_optimized",
        )
    return float(threshold_config["fixed"]), "fixed_configured"


def _verify_assignments(
    prediction_frame: pd.DataFrame,
    *,
    n_patients: int,
    repeats: int,
    model: str,
    variant: str,
) -> None:
    group = prediction_frame[
        (prediction_frame["model"] == model) & (prediction_frame["variant"] == variant)
    ]
    if group.duplicated(["patient_index", "repeat"]).any():
        raise RuntimeError(f"Duplicate patient/repetition assignment for {model}/{variant}.")
    counts = group.groupby("patient_index").size()
    if len(counts) != n_patients or not (counts == repeats).all():
        raise RuntimeError(
            f"Each patient must have exactly {repeats} predictions for {model}/{variant}."
        )
    per_repeat = group.groupby("repeat")["patient_index"].nunique()
    if len(per_repeat) != repeats or not (per_repeat == n_patients).all():
        raise RuntimeError(f"A repetition is missing patients for {model}/{variant}.")


def _repetition_summaries(prediction_frame: pd.DataFrame, fixed_threshold: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (model, variant, repeat), group in prediction_frame.groupby(["model", "variant", "repeat"]):
        metrics = classification_metrics(
            group["observed"].to_numpy(),
            group["probability_survived_one_year"].to_numpy(),
            fixed_threshold,
        )
        rows.append({"model": model, "variant": variant, "repeat": repeat, **metrics})
    return pd.DataFrame(rows)


def run_nested_experiment(
    features: pd.DataFrame,
    target: pd.Series,
    config: dict[str, Any],
    *,
    artifact_root: Path | str | None = None,
    dataset_path: Path | str | None = None,
) -> Path:
    """Run configured models and persist auditable outer-fold artifacts."""

    config = validate_config(config)
    started = datetime.now(UTC)
    experiment = config["experiment"]
    seed = int(experiment["random_seed"])
    outer_folds = int(experiment["outer_folds"])
    repeats = int(experiment["outer_repeats"])
    inner_folds = int(experiment["inner_folds"])
    n_jobs = int(experiment.get("n_jobs", 1))
    run_dir = (
        create_run_directory(config, artifact_root)
        if artifact_root
        else create_run_directory(config)
    )
    splitter = RepeatedStratifiedKFold(
        n_splits=outer_folds,
        n_repeats=repeats,
        random_state=seed,
    )
    predictions: list[dict[str, Any]] = []
    folds: list[dict[str, Any]] = []
    importance_rows: list[dict[str, Any]] = []
    model_errors: dict[str, str] = {}
    for model_name in config["models"]["include"]:
        try:
            base_pipeline, grid = build_model(model_name, seed)
        except RuntimeError as exc:
            model_errors[model_name] = str(exc)
            LOGGER.warning("%s", exc)
            continue
        for fold_index, (train_idx, validation_idx) in enumerate(splitter.split(features, target)):
            repeat = fold_index // outer_folds
            fold = fold_index % outer_folds
            fold_seed = seed + fold_index
            x_train, x_validation = features.iloc[train_idx], features.iloc[validation_idx]
            y_train, y_validation = target.iloc[train_idx], target.iloc[validation_idx]
            if set(train_idx).intersection(validation_idx):
                raise RuntimeError(
                    "Patient overlap detected between outer training and validation."
                )
            tuned, best_params = _fit_tuned(
                base_pipeline,
                grid,
                x_train,
                y_train,
                inner_folds=inner_folds,
                seed=fold_seed,
                n_jobs=n_jobs,
            )
            explicit, training_oof, selected_name, training_brier = _training_only_variants(
                tuned,
                x_train,
                y_train,
                inner_folds=inner_folds,
                seed=fold_seed,
                n_jobs=n_jobs,
                calibration=config["calibration"],
            )
            variants = {
                **explicit,
                "training_selected": explicit[selected_name],
            }
            variant_training_oof = {
                **training_oof,
                "training_selected": training_oof[selected_name],
            }
            for variant, fitted in variants.items():
                probability = fitted.predict_proba(x_validation)[:, 1]
                selected_threshold, threshold_source = _select_threshold(
                    y_train,
                    variant_training_oof[variant],
                    config["threshold"],
                )
                fixed_metrics = classification_metrics(
                    y_validation.to_numpy(),
                    probability,
                    float(config["threshold"]["fixed"]),
                )
                threshold_metrics = decision_metrics(
                    y_validation.to_numpy(),
                    (probability >= selected_threshold).astype(int),
                )
                folds.append(
                    {
                        "model": model_name,
                        "variant": variant,
                        "repeat": repeat,
                        "fold": fold,
                        "train_indices": json.dumps(train_idx.tolist()),
                        "validation_indices": json.dumps(validation_idx.tolist()),
                        "selected_threshold": selected_threshold,
                        "threshold_source": threshold_source,
                        "threshold_optimization_enabled": bool(
                            config["threshold"]["optimize_training_only"]
                        ),
                        "threshold_objective": str(config["threshold"]["objective"]),
                        "calibration_decision": selected_name,
                        "training_brier_uncalibrated": training_brier["uncalibrated"],
                        "training_brier_sigmoid": training_brier.get("sigmoid", float("nan")),
                        "best_params": json.dumps(best_params, sort_keys=True),
                        **fixed_metrics,
                        **{
                            f"training_threshold_{key}": value
                            for key, value in threshold_metrics.items()
                        },
                    }
                )
                for position, patient_index in enumerate(validation_idx):
                    predictions.append(
                        {
                            "patient_index": int(patient_index),
                            "model": model_name,
                            "variant": variant,
                            "repeat": repeat,
                            "fold": fold,
                            "observed": int(y_validation.iloc[position]),
                            "probability_survived_one_year": float(probability[position]),
                            "fixed_threshold_prediction": int(
                                probability[position] >= float(config["threshold"]["fixed"])
                            ),
                            "training_selected_threshold": selected_threshold,
                            "training_threshold_prediction": int(
                                probability[position] >= selected_threshold
                            ),
                            "calibration_decision": selected_name,
                        }
                    )
                if variant == "training_selected":
                    importance = permutation_importance(
                        fitted,
                        x_validation,
                        y_validation,
                        scoring="roc_auc",
                        n_repeats=int(config["explainability"]["permutation_repeats"]),
                        random_state=fold_seed,
                        n_jobs=1,
                    )
                    for feature, mean, std in zip(
                        features.columns,
                        importance.importances_mean,
                        importance.importances_std,
                        strict=True,
                    ):
                        importance_rows.append(
                            {
                                "model": model_name,
                                "variant": variant,
                                "repeat": repeat,
                                "fold": fold,
                                "feature": feature,
                                "importance_mean": float(mean),
                                "importance_std": float(std),
                                "scoring": "ROC-AUC; higher positive values indicate loss "
                                "after permutation",
                            }
                        )
    if not predictions:
        raise RuntimeError("No configured model completed successfully.")
    prediction_frame = pd.DataFrame(predictions)
    fold_frame = pd.DataFrame(folds)
    for model, variant in (
        prediction_frame[["model", "variant"]].drop_duplicates().itertuples(index=False, name=None)
    ):
        _verify_assignments(
            prediction_frame,
            n_patients=len(features),
            repeats=repeats,
            model=model,
            variant=variant,
        )
    prediction_frame.to_csv(run_dir / "oof_predictions_all.csv", index=False)
    fold_frame.to_csv(run_dir / "fold_metrics.csv", index=False)
    pd.DataFrame(importance_rows).to_csv(
        run_dir / "tables" / "held_out_permutation_importance.csv", index=False
    )
    repetition_frame = _repetition_summaries(prediction_frame, float(config["threshold"]["fixed"]))
    repetition_frame.to_csv(run_dir / "tables" / "repetition_metrics.csv", index=False)
    summaries: list[dict[str, Any]] = []
    confidence: dict[str, Any] = {}
    for (model, variant), group in prediction_frame.groupby(["model", "variant"]):
        patient = (
            group.groupby("patient_index", as_index=False)
            .agg(
                observed=("observed", "first"),
                probability_survived_one_year=("probability_survived_one_year", "mean"),
                prediction_count=("probability_survived_one_year", "size"),
                training_threshold_vote_fraction=("training_threshold_prediction", "mean"),
            )
            .sort_values("patient_index")
        )
        patient.to_csv(run_dir / f"oof_patient_{model}_{variant}.csv", index=False)
        y = patient["observed"].to_numpy()
        p = patient["probability_survived_one_year"].to_numpy()
        fixed_metrics = classification_metrics(y, p, float(config["threshold"]["fixed"]))
        training_threshold_metrics = decision_metrics(
            y,
            (patient["training_threshold_vote_fraction"].to_numpy() >= 0.5).astype(int),
        )
        repetition_group = repetition_frame[
            (repetition_frame["model"] == model) & (repetition_frame["variant"] == variant)
        ]
        summaries.append(
            {
                "model": model,
                "variant": variant,
                **fixed_metrics,
                **{
                    f"training_threshold_{key}": value
                    for key, value in training_threshold_metrics.items()
                },
                "training_threshold_decision_rule": "majority_vote_across_repetitions",
                "roc_auc_repetition_mean": float(repetition_group["roc_auc"].mean()),
                "roc_auc_repetition_std": float(repetition_group["roc_auc"].std(ddof=0)),
                "brier_repetition_mean": float(repetition_group["brier"].mean()),
                "brier_repetition_std": float(repetition_group["brier"].std(ddof=0)),
            }
        )
        confidence[f"{model}:{variant}"] = bootstrap_confidence_intervals(
            y,
            p,
            n_resamples=int(experiment["bootstrap_resamples"]),
            seed=seed,
        )
        bins = calibration_table(y, p)
        bins.insert(0, "variant", variant)
        bins.insert(0, "model", model)
        bins.to_csv(
            run_dir / "tables" / f"calibration_bins_{model}_{variant}.csv",
            index=False,
        )
    pd.DataFrame(summaries).to_csv(run_dir / "aggregated_metrics.csv", index=False)
    write_json(run_dir / "confidence_intervals.json", confidence)
    finished = datetime.now(UTC)
    source = Path(dataset_path) if dataset_path is not None else None
    dataset_hash = (
        hashlib.sha256(source.read_bytes()).hexdigest()
        if source is not None and source.is_file()
        else "not available"
    )
    write_json(
        run_dir / "run_metadata.json",
        {
            "dataset_id": 423,
            "dataset_doi": DATASET_DOI,
            "dataset_sha256": dataset_hash,
            "random_seed": seed,
            "model_errors": model_errors,
            "positive_class": "1 = survived at one year",
            "execution_start_utc": started.isoformat(),
            "execution_end_utc": finished.isoformat(),
            "validation_design": {
                "outer_folds": outer_folds,
                "outer_repeats": repeats,
                "inner_folds": inner_folds,
            },
            "bootstrap_resamples": int(experiment["bootstrap_resamples"]),
            "validation_note": (
                "Every OOF probability came from a model that did not train on that patient."
            ),
        },
    )
    return run_dir
