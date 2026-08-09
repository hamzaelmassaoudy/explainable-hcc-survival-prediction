import numpy as np
import pytest

from hcc_survival.metrics import classification_metrics, decision_metrics


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
