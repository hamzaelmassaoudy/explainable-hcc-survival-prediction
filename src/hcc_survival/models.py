"""Model pipelines and conservative predefined search spaces."""

from __future__ import annotations

from typing import Any

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from hcc_survival.preprocessing import build_preprocessor
from hcc_survival.schemas import REDUCED_CLINICAL_FEATURES


def available_model_names() -> tuple[str, ...]:
    """Return stable public model identifiers."""

    return (
        "dummy",
        "reduced_clinical_logistic",
        "logistic_regression",
        "random_forest",
        "xgboost",
    )


def build_model(name: str, seed: int) -> tuple[Pipeline, Any]:
    """Create a complete pipeline and its deliberately small search grid."""

    if name == "dummy":
        estimator = DummyClassifier(strategy="prior", random_state=seed)
        return Pipeline(
            [("preprocess", build_preprocessor(scale_numeric=False)), ("model", estimator)]
        ), {}
    if name == "logistic_regression":
        estimator = LogisticRegression(solver="liblinear", max_iter=2000, random_state=seed)
        grid = {"model__C": [0.1, 1.0, 10.0], "model__class_weight": [None, "balanced"]}
        return Pipeline(
            [("preprocess", build_preprocessor(scale_numeric=True)), ("model", estimator)]
        ), grid
    if name == "reduced_clinical_logistic":
        estimator = LogisticRegression(solver="liblinear", max_iter=2000, random_state=seed)
        grid = {"model__C": [0.1, 1.0, 10.0], "model__class_weight": [None, "balanced"]}
        return Pipeline(
            [
                (
                    "preprocess",
                    build_preprocessor(
                        scale_numeric=True,
                        feature_subset=REDUCED_CLINICAL_FEATURES,
                    ),
                ),
                ("model", estimator),
            ]
        ), grid
    if name == "random_forest":
        estimator = RandomForestClassifier(
            n_estimators=300, random_state=seed, n_jobs=1, class_weight=None
        )
        grid = [
            {
                "model__max_depth": [3],
                "model__min_samples_leaf": [5],
                "model__max_features": ["sqrt"],
                "model__class_weight": [None],
            },
            {
                "model__max_depth": [5],
                "model__min_samples_leaf": [10],
                "model__max_features": [0.5],
                "model__class_weight": ["balanced"],
            },
        ]
        return Pipeline(
            [("preprocess", build_preprocessor(scale_numeric=False)), ("model", estimator)]
        ), grid
    if name == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise RuntimeError(
                "XGBoost was requested but is unavailable. Install with "
                '`python -m pip install -e ".[boosting]"`.'
            ) from exc
        estimator = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            device="cpu",
            n_jobs=1,
            random_state=seed,
        )
        grid = [
            {
                "model__n_estimators": [100],
                "model__max_depth": [1],
                "model__learning_rate": [0.03],
                "model__subsample": [0.8],
                "model__colsample_bytree": [0.8],
                "model__reg_alpha": [0.0],
                "model__reg_lambda": [1.0],
            },
            {
                "model__n_estimators": [200],
                "model__max_depth": [2],
                "model__learning_rate": [0.08],
                "model__subsample": [0.8],
                "model__colsample_bytree": [0.8],
                "model__reg_alpha": [0.5],
                "model__reg_lambda": [5.0],
            },
        ]
        return Pipeline(
            [("preprocess", build_preprocessor(scale_numeric=False)), ("model", estimator)]
        ), grid
    raise ValueError(f"Unknown model {name!r}; choose from {available_model_names()}.")
