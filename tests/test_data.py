"""Synthetic tests for HCC dataset acquisition and schema validation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from hcc_survival.data import (
    DataValidationError,
    data_quality_report,
    download_dataset,
    load_local_dataset,
    validate_dataset,
    write_data_outputs,
)
from hcc_survival.schemas import FEATURE_NAMES


def test_documented_schema_has_49_unique_features() -> None:
    """The public schema matches UCI's documented feature count."""

    assert len(FEATURE_NAMES) == 49
    assert len(FEATURE_NAMES) == len(set(FEATURE_NAMES))


def test_documented_missing_marker_and_target_validation(
    synthetic_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """A documented missing marker is preserved while target codes remain binary."""

    features, target = synthetic_data
    frame = pd.concat([features, target], axis=1)
    frame["Age"] = frame["Age"].astype(object)
    frame.loc[0, "Age"] = "?"

    valid_features, valid_target = validate_dataset(frame)

    assert pd.isna(valid_features.loc[0, "Age"])
    assert set(valid_target.unique()) <= {0, 1}


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("Age", "not-a-number", "non-numeric"),
        ("Gender", 0.5, "integer category codes"),
    ],
)
def test_invalid_feature_values_fail_validation(
    synthetic_data: tuple[pd.DataFrame, pd.Series],
    column: str,
    value: object,
    message: str,
) -> None:
    """Unexpected text and fractional category codes are rejected rather than coerced."""

    features, target = synthetic_data
    frame = pd.concat([features, target], axis=1)
    frame[column] = frame[column].astype(object)
    frame.loc[1, column] = value

    with pytest.raises(DataValidationError, match=message):
        validate_dataset(frame)


def test_schema_mismatch_fails(
    synthetic_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Missing documented columns cannot be silently accepted."""

    features, target = synthetic_data
    frame = pd.concat([features, target], axis=1).drop(columns=[FEATURE_NAMES[0]])

    with pytest.raises(DataValidationError, match="documented HCC schema"):
        validate_dataset(frame)


def test_invalid_target_fails(
    synthetic_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """The target remains restricted to the documented survival coding."""

    features, target = synthetic_data
    target.iloc[0] = 2

    with pytest.raises(DataValidationError, match="Target"):
        validate_dataset(pd.concat([features, target], axis=1))


def test_local_data_errors_and_writes_do_not_escape_ignored_roots(
    synthetic_data: tuple[pd.DataFrame, pd.Series],
    tmp_path: Path,
) -> None:
    """Public data helpers redact missing paths and reject arbitrary output destinations."""

    features, _ = synthetic_data
    missing_cache = tmp_path / "private" / "cache.csv"

    with pytest.raises(FileNotFoundError) as error:
        load_local_dataset(missing_cache)
    assert str(missing_cache) not in str(error.value)

    with pytest.raises(ValueError, match="data/raw"):
        download_dataset(missing_cache)
    with pytest.raises(ValueError, match="data/processed"):
        write_data_outputs(features, tmp_path)


def test_quality_report_is_aggregate_only(
    synthetic_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Plausibility summaries expose counts rather than row-level observed values."""

    features, _ = synthetic_data
    features["Age"] = 60.0
    features.loc[1, "Age"] = 200.0

    report = data_quality_report(features)
    age_flag = next(item for item in report["plausibility_flags"] if item["feature"] == "Age")

    assert age_flag["flagged_value_count"] == 1
    assert "observed_value" not in age_flag
