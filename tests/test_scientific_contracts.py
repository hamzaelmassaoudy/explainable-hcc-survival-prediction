"""Scientific contract tests for the one-year survival estimand.

These tests intentionally exercise the public package APIs rather than generated
reports.  They make the positive-class direction and feature schema explicit so
future refactors cannot silently turn survival metrics into mortality metrics.
"""

from __future__ import annotations

import numpy as np
import pytest

from hcc_survival.metrics import classification_metrics
from hcc_survival.schemas import (
    CATEGORICAL_FEATURES,
    FEATURE_NAMES,
    FEATURE_SPECS,
    NUMERICAL_FEATURES,
    ORDINAL_FEATURES,
    REDUCED_CLINICAL_FEATURES,
)


def test_classification_metrics_use_survivor_positive_class() -> None:
    """Sensitivity counts survivors; specificity counts deaths."""

    result = classification_metrics(
        np.array([0, 0, 1, 1]),
        np.array([0.10, 0.40, 0.60, 0.90]),
        threshold=0.50,
    )

    assert result["survivor_sensitivity"] == 1.0
    assert result["death_specificity"] == 1.0
    assert result["true_survivors"] == 2
    assert result["false_non_survivors"] == 0
    assert result["true_deaths"] == 2
    assert result["false_survivors"] == 0
    assert "sensitivity" not in result
    assert "specificity" not in result


def test_classification_metrics_have_no_ambiguous_threshold_names() -> None:
    result = classification_metrics(
        np.array([0, 1, 1, 0]),
        np.array([0.70, 0.80, 0.20, 0.10]),
        threshold=0.50,
    )
    assert set(result).isdisjoint({"sensitivity", "specificity"})
    assert result["predicted_survivor_count"] + result["predicted_death_count"] == 4


def test_feature_groups_are_disjoint_and_cover_the_authoritative_schema() -> None:
    groups = [CATEGORICAL_FEATURES, ORDINAL_FEATURES, NUMERICAL_FEATURES]
    flattened = [feature for group in groups for feature in group]
    assert len(flattened) == len(set(flattened))
    assert set(flattened) == set(FEATURE_NAMES)
    assert len(FEATURE_SPECS) == len(FEATURE_NAMES)
    assert set(REDUCED_CLINICAL_FEATURES) <= set(FEATURE_NAMES)
    assert len(REDUCED_CLINICAL_FEATURES) == len(set(REDUCED_CLINICAL_FEATURES))


def test_schema_feature_names_and_group_metadata_are_unique() -> None:
    names = [spec.name for spec in FEATURE_SPECS]
    assert names == list(FEATURE_NAMES)
    assert all(spec.group.strip() for spec in FEATURE_SPECS)
    assert all(spec.label.strip() for spec in FEATURE_SPECS)


@pytest.mark.parametrize("threshold", [0.0, 0.5, 1.0])
def test_threshold_is_reported_exactly(threshold: float) -> None:
    result = classification_metrics(
        np.array([0, 1, 0, 1]),
        np.array([0.1, 0.9, 0.2, 0.8]),
        threshold=threshold,
    )
    assert result["threshold"] == threshold
