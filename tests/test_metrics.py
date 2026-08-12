import numpy as np
import pytest

from hcc_survival.metrics import (
    calibration_error,
    calibration_table,
    classification_metrics,
    decision_metrics,
)


def test_specificity_and_npv():
    result = classification_metrics(
        np.array([0, 0, 1, 1]), np.array([0.1, 0.4, 0.6, 0.9]), threshold=0.5
    )
    assert result["death_specificity"] == 1.0
    assert result["negative_predictive_value"] == 1.0
    assert result["brier"] < 0.1


@pytest.mark.parametrize("threshold", [-0.01, 1.01, float("nan")])
def test_classification_metrics_rejects_invalid_thresholds(threshold):
    with pytest.raises(ValueError, match="Threshold must be a finite value between 0 and 1"):
        classification_metrics(
            np.array([0, 1]),
            np.array([0.2, 0.8]),
            threshold=threshold,
        )


@pytest.mark.parametrize(
    ("probabilities", "message"),
    [
        (np.array([0.2, np.nan]), "Probabilities must be finite"),
        (np.array([0.2, 1.1]), "Probabilities must be between 0 and 1"),
    ],
)
def test_classification_metrics_rejects_invalid_probabilities(probabilities, message):
    with pytest.raises(ValueError, match=message):
        classification_metrics(np.array([0, 1]), probabilities)


def test_classification_metrics_requires_aligned_binary_outcomes():
    with pytest.raises(ValueError, match="Outcomes must be encoded as 0"):
        classification_metrics(np.array([0, 2]), np.array([0.2, 0.8]))
    with pytest.raises(ValueError, match="aligned with outcomes"):
        classification_metrics(np.array([0, 1]), np.array([0.2]))


def test_decision_metrics_rejects_invalid_outcomes_and_decisions():
    with pytest.raises(ValueError, match="Outcomes must be encoded as 0"):
        decision_metrics(np.array([0, 2]), np.array([0, 1]))
    with pytest.raises(ValueError, match="Decisions must be encoded as 0 or 1"):
        decision_metrics(np.array([0, 1]), np.array([0, 2]))
    with pytest.raises(ValueError, match="Decisions must be encoded as 0 or 1"):
        decision_metrics(np.array([0, 1]), np.array([0.0, 0.5]))


@pytest.mark.parametrize(
    ("outcomes", "probabilities", "message"),
    [
        (np.array([0, 1]), np.array([0.2, np.nan]), "Probabilities must be finite"),
        (np.array([0, 1]), np.array([0.2, 1.1]), "Probabilities must be between 0 and 1"),
        (np.array([0, 2]), np.array([0.2, 0.8]), "Outcomes must be encoded as 0"),
    ],
)
def test_calibration_table_rejects_invalid_inputs(outcomes, probabilities, message):
    with pytest.raises(ValueError, match=message):
        calibration_table(outcomes, probabilities)


def test_calibration_error_requires_aligned_inputs():
    with pytest.raises(ValueError, match="aligned with outcomes"):
        calibration_error(np.array([0, 1]), np.array([0.2]))


@pytest.mark.parametrize("n_bins", [0, -1, 1.5, True])
def test_calibration_table_requires_a_positive_integer_bin_count(n_bins):
    with pytest.raises(ValueError, match="positive integer"):
        calibration_table(np.array([0, 1]), np.array([0.2, 0.8]), n_bins=n_bins)


def test_constant_probabilities_produce_one_calibration_bin():
    table = calibration_table(
        np.array([0, 1, 1, 0]),
        np.full(4, 0.25),
        n_bins=5,
    )
    error, usable_bins = calibration_error(
        np.array([0, 1, 1, 0]),
        np.full(4, 0.25),
        n_bins=5,
    )

    assert table["n"].tolist() == [4]
    assert table["mean_predicted"].tolist() == [0.25]
    assert table["observed_survival"].tolist() == [0.5]
    assert error == pytest.approx(0.25)
    assert usable_bins == 1
