"""Tests for fold-local preprocessing behavior with synthetic inputs."""

from __future__ import annotations

import numpy as np
import pytest

from hcc_survival.preprocessing import PreprocessingError, build_preprocessor
from hcc_survival.schemas import CATEGORICAL_FEATURES, NUMERICAL_FEATURES, ORDINAL_FEATURES


def test_preprocessor_handles_missing_and_unknown_categories(synthetic_data) -> None:
    """Unknown categories are handled without refitting the training transformer."""

    x, _ = synthetic_data
    training = x.iloc[:30].copy()
    held_out = x.iloc[30:].copy()
    pipeline = build_preprocessor(scale_numeric=True)
    transformed = pipeline.fit_transform(training)
    unknown = held_out.iloc[[0]].copy()
    unknown["Gender"] = 99
    unknown["PS"] = 99
    unknown["Age"] = np.nan
    transformed_unknown = pipeline.transform(unknown)

    assert transformed_unknown.shape[1] == transformed.shape[1]
    assert np.isfinite(transformed_unknown).all()

    feature_names = list(pipeline.get_feature_names_out())
    gender_columns = [
        index for index, name in enumerate(feature_names) if name.startswith("categorical__Gender_")
    ]
    assert gender_columns
    assert np.all(transformed_unknown[0, gender_columns] == 0)
    ordinal_index = feature_names.index("ordinal__PS")
    assert transformed_unknown[0, ordinal_index] == -1


def test_preprocessor_learns_statistics_from_training_rows_only(synthetic_data) -> None:
    """Imputation and scaling statistics exclude held-out extreme values."""

    x, _ = synthetic_data
    training = x.iloc[:30].copy()
    held_out = x.iloc[30:].copy()
    training["Age"] = np.concatenate((np.zeros(29), np.array([100.0])))
    held_out["Age"] = 10_000.0
    training["Gender"] = np.array([0] * 16 + [1] * 14)
    held_out["Gender"] = 1
    training["PS"] = np.array([1] * 16 + [2] * 14)
    held_out["PS"] = 4

    pipeline = build_preprocessor(scale_numeric=True)
    pipeline.fit(training)

    numeric = pipeline.named_transformers_["numeric"].named_steps
    categorical = pipeline.named_transformers_["categorical"].named_steps
    ordinal = pipeline.named_transformers_["ordinal"].named_steps
    age_index = NUMERICAL_FEATURES.index("Age")
    gender_index = CATEGORICAL_FEATURES.index("Gender")
    ps_index = ORDINAL_FEATURES.index("PS")

    assert numeric["impute"].statistics_[age_index] == pytest.approx(0.0)
    assert numeric["scale"].mean_[age_index] == pytest.approx(100.0 / 30.0)
    assert categorical["impute"].statistics_[gender_index] == 0
    assert ordinal["impute"].statistics_[ps_index] == 1
    assert np.isfinite(pipeline.transform(held_out)).all()


def test_missingness_indicators_are_learned_from_training_rows_only(synthetic_data) -> None:
    """Held-out missing values do not create new fitted missingness indicators."""

    x, _ = synthetic_data
    training = x.iloc[:30].copy()
    held_out = x.iloc[30:].copy()
    training["Age"] = 50.0
    held_out["Age"] = np.nan

    with_indicators = build_preprocessor(scale_numeric=False, add_indicators=True)
    with_indicators.fit(training)
    assert "numeric__missingindicator_Age" not in with_indicators.get_feature_names_out()
    assert np.isfinite(with_indicators.transform(held_out)).all()

    without_indicators = build_preprocessor(scale_numeric=False, add_indicators=False)
    without_indicators.fit(training)
    assert not any(
        "missingindicator" in name for name in without_indicators.get_feature_names_out()
    )
    assert "scale" not in without_indicators.named_transformers_["numeric"].named_steps


def test_preprocessor_retains_all_missing_training_columns(synthetic_data) -> None:
    """An all-missing training-fold column remains transformable and finite."""

    x, _ = synthetic_data
    training = x.iloc[:30].copy()
    training[["Age", "Gender", "PS"]] = np.nan

    pipeline = build_preprocessor(scale_numeric=True)
    transformed = pipeline.fit_transform(training)
    feature_names = pipeline.get_feature_names_out()

    assert np.isfinite(transformed).all()
    assert "numeric__Age" in feature_names
    assert any(name.startswith("categorical__Gender_") for name in feature_names)
    assert "ordinal__PS" in feature_names


def test_preprocessor_supports_documented_feature_subsets(synthetic_data) -> None:
    """Subset output uses only requested schema-backed columns and requested scaling."""

    x, _ = synthetic_data
    pipeline = build_preprocessor(
        scale_numeric=False,
        add_indicators=False,
        feature_subset=("PS", "Age", "Gender"),
    )
    transformed = pipeline.fit_transform(x)

    assert transformed.shape[1] == len(pipeline.get_feature_names_out())
    assert {"numeric", "categorical", "ordinal"}.issubset(pipeline.named_transformers_)
    assert "scale" not in pipeline.named_transformers_["numeric"].named_steps


@pytest.mark.parametrize(
    "feature_subset, message",
    [
        ((), "at least one"),
        (("Age", "Age"), "duplicate"),
        (("not_a_documented_feature",), "unknown"),
    ],
)
def test_preprocessor_rejects_invalid_feature_subsets(
    feature_subset: tuple[str, ...], message: str
) -> None:
    """Invalid subsets fail before any estimator can be fitted."""

    with pytest.raises(PreprocessingError, match=message):
        build_preprocessor(scale_numeric=True, feature_subset=feature_subset)
