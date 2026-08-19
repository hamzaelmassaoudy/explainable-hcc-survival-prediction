"""Tests for patient-level subgroup safeguards."""

import pandas as pd
import pytest

from hcc_survival.subgroup import exploratory_subgroup_metrics, predefined_subgroups


def _patient_predictions() -> pd.DataFrame:
    """Return a small synthetic patient-level prediction table."""

    return pd.DataFrame(
        {
            "patient_index": range(15),
            "observed": [0] * 7 + [1] * 8,
            "probability_survived_one_year": [0.2] * 7 + [0.8] * 8,
        }
    )


def test_subgroup_metrics_apply_suppression_at_patient_level() -> None:
    """A small patient-level subgroup remains suppressed."""

    predictions = _patient_predictions()
    subgroup = pd.Series("example", index=predictions["patient_index"])

    result = exploratory_subgroup_metrics(predictions, subgroup)

    assert result.loc[0, "n"] == 15
    assert bool(result.loc[0, "metrics_suppressed"])


def test_subgroup_metrics_reject_repeated_out_of_fold_predictions() -> None:
    """Repeated out-of-fold rows cannot inflate subgroup sample sizes."""

    predictions = pd.concat([_patient_predictions(), _patient_predictions()], ignore_index=True)
    subgroup = pd.Series("example", index=range(15))

    with pytest.raises(ValueError, match="patient-aggregated prediction per patient"):
        exploratory_subgroup_metrics(predictions, subgroup)


def test_subgroup_metrics_reject_missing_patient_index() -> None:
    """A missing patient identifier cannot be treated as an aggregated prediction."""

    predictions = _patient_predictions()
    predictions.loc[14, "patient_index"] = None
    subgroup = pd.Series("example", index=range(15))

    with pytest.raises(ValueError, match="patient-aggregated prediction per patient"):
        exploratory_subgroup_metrics(predictions, subgroup)


def test_subgroup_metrics_require_patient_index() -> None:
    """The patient identifier is required for aggregation-safe subgroup metrics."""

    predictions = _patient_predictions().drop(columns="patient_index")

    with pytest.raises(ValueError, match="patient_index column"):
        exploratory_subgroup_metrics(predictions, pd.Series("example", index=range(15)))


def test_predefined_subgroups_align_with_validation_row_positions() -> None:
    """Subgroup labels match positional patient identifiers from validation splits."""

    features = pd.DataFrame(
        {"Age": [55, 65, 75], "Gender": [0, 1, None]},
        index=[101, 205, 309],
    )
    predictions = pd.DataFrame(
        {
            "patient_index": [0, 1, 2],
            "observed": [0, 1, 1],
            "probability_survived_one_year": [0.2, 0.8, 0.9],
        }
    )

    subgroups = predefined_subgroups(features)

    assert subgroups["sex"].index.tolist() == [0, 1, 2]
    assert subgroups["age_group"].index.tolist() == [0, 1, 2]
    assert set(exploratory_subgroup_metrics(predictions, subgroups["sex"])["subgroup"]) == {
        "Female",
        "Male",
        "Unknown",
    }
    assert set(exploratory_subgroup_metrics(predictions, subgroups["age_group"])["subgroup"]) == {
        "<60",
        "60-69",
        "70+",
    }
