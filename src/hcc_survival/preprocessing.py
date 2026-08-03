"""Fold-local preprocessing factories for research experiments."""

from __future__ import annotations

from collections.abc import Sequence

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from hcc_survival.schemas import (
    CATEGORICAL_FEATURES,
    FEATURE_NAMES,
    FEATURE_SPECS,
    NUMERICAL_FEATURES,
    ORDINAL_FEATURES,
)


class PreprocessingError(ValueError):
    """Raised when a requested preprocessing configuration is invalid."""


_FEATURE_CATEGORY_MAP = {spec.name: spec.categories for spec in FEATURE_SPECS}
_FEATURE_NAME_SET = frozenset(FEATURE_NAMES)


def _resolve_feature_subset(feature_subset: Sequence[str] | None) -> tuple[str, ...]:
    """Return a non-empty, unique schema-backed feature subset in schema order."""

    if feature_subset is None:
        return FEATURE_NAMES
    if isinstance(feature_subset, (str, bytes)) or not isinstance(feature_subset, Sequence):
        raise PreprocessingError("feature_subset must be a sequence of documented feature names.")

    selected = tuple(feature_subset)
    if not selected:
        raise PreprocessingError("feature_subset must contain at least one documented feature.")
    if not all(isinstance(name, str) for name in selected):
        raise PreprocessingError("feature_subset must contain only feature names.")
    if len(set(selected)) != len(selected):
        raise PreprocessingError("feature_subset must not contain duplicate feature names.")

    unknown = sorted(set(selected) - _FEATURE_NAME_SET)
    if unknown:
        raise PreprocessingError(f"feature_subset contains unknown feature names: {unknown}.")
    selected_set = set(selected)
    return tuple(name for name in FEATURE_NAMES if name in selected_set)


def _categories_for(feature_names: Sequence[str]) -> list[list[int]]:
    """Return documented category values for categorical or ordinal features."""

    categories: list[list[int]] = []
    for name in feature_names:
        values = _FEATURE_CATEGORY_MAP[name]
        if values is None:
            raise PreprocessingError(f"Feature {name!r} has no documented categories.")
        categories.append(list(values))
    return categories


def _require_bool(value: bool, name: str) -> None:
    """Require an explicit boolean preprocessing option."""

    if not isinstance(value, bool):
        raise PreprocessingError(f"{name} must be a boolean.")


def build_preprocessor(
    *,
    scale_numeric: bool,
    add_indicators: bool = True,
    feature_subset: Sequence[str] | None = None,
) -> ColumnTransformer:
    """Build an unfitted transformer whose learned state comes only from fit rows.

    Place the returned transformer inside an estimator pipeline and fit that pipeline on each
    training partition. It does not choose cross-validation folds itself. Documented category
    values are fixed by the schema, while imputation and optional scaling are learned during
    ``fit``. Empty training-fold columns are retained with deterministic imputations.
    """

    _require_bool(scale_numeric, "scale_numeric")
    _require_bool(add_indicators, "add_indicators")
    selected = _resolve_feature_subset(feature_subset)
    selected_set = set(selected)
    numerical = [name for name in NUMERICAL_FEATURES if name in selected_set]
    categorical_names = [name for name in CATEGORICAL_FEATURES if name in selected_set]
    ordinal_names = [name for name in ORDINAL_FEATURES if name in selected_set]

    numeric_steps: list[tuple[str, object]] = [
        (
            "impute",
            SimpleImputer(
                strategy="median",
                add_indicator=add_indicators,
                keep_empty_features=True,
            ),
        )
    ]
    if scale_numeric:
        numeric_steps.append(("scale", StandardScaler()))

    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if numerical:
        transformers.append(("numeric", Pipeline(numeric_steps), numerical))
    if categorical_names:
        categorical = Pipeline(
            [
                ("impute", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
                (
                    "encode",
                    OneHotEncoder(
                        categories=_categories_for(categorical_names),
                        handle_unknown="ignore",
                        sparse_output=False,
                    ),
                ),
            ]
        )
        transformers.append(("categorical", categorical, categorical_names))
    if ordinal_names:
        ordinal = Pipeline(
            [
                ("impute", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
                (
                    "encode",
                    OrdinalEncoder(
                        categories=_categories_for(ordinal_names),
                        handle_unknown="use_encoded_value",
                        unknown_value=-1,
                    ),
                ),
            ]
        )
        transformers.append(("ordinal", ordinal, ordinal_names))

    return ColumnTransformer(
        transformers,
        remainder="drop",
        sparse_threshold=0.0,
        verbose_feature_names_out=True,
    )
