import numpy as np

from hcc_survival.metrics import classification_metrics


def test_specificity_and_npv():
    result = classification_metrics(
        np.array([0, 0, 1, 1]), np.array([0.1, 0.4, 0.6, 0.9]), threshold=0.5
    )
    assert result["death_specificity"] == 1.0
    assert result["negative_predictive_value"] == 1.0
    assert result["brier"] < 0.1
