import numpy as np
import pytest

from hcc_survival.metrics import (
    bootstrap_confidence_intervals,
    calibration_error,
    calibration_slope_intercept,
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


@pytest.mark.parametrize(
    ("outcomes", "probabilities", "message"),
    [
        (np.array([0, 1]), np.array([0.2, np.nan]), "Probabilities must be finite"),
        (np.array([0, 1]), np.array([0.2, 1.1]), "Probabilities must be between 0 and 1"),
        (np.array([0, 2]), np.array([0.2, 0.8]), "Outcomes must be encoded as 0"),
        (np.array([0, 1]), np.array([0.2]), "aligned with outcomes"),
    ],
)
def test_calibration_slope_intercept_rejects_invalid_inputs(outcomes, probabilities, message):
    """Standalone calibration estimates enforce the shared probability contract."""

    with pytest.raises(ValueError, match=message):
        calibration_slope_intercept(outcomes, probabilities)


@pytest.mark.parametrize(
    ("outcomes", "probabilities"),
    [
        (np.tile([0, 1], 14), np.full(28, 0.5)),
        (np.zeros(30, dtype=int), np.full(30, 0.5)),
    ],
)
def test_calibration_slope_intercept_returns_nan_when_not_estimable(outcomes, probabilities):
    """Valid inputs without enough outcome variation remain explicitly not estimable."""

    intercept, slope = calibration_slope_intercept(outcomes, probabilities)

    assert np.isnan(intercept)
    assert np.isnan(slope)


@pytest.mark.parametrize(
    ("outcomes", "probabilities", "message"),
    [
        (np.array([0, 1]), np.array([0.2, np.nan]), "Probabilities must be finite"),
        (np.array([0, 1]), np.array([0.2, 1.1]), "Probabilities must be between 0 and 1"),
        (np.array([0, 2]), np.array([0.2, 0.8]), "Outcomes must be encoded as 0"),
        (np.array([0, 1]), np.array([0.2]), "aligned with outcomes"),
        (np.array([0, 1]), np.array([0.2, 0.8, 0.9]), "aligned with outcomes"),
    ],
)
def test_bootstrap_confidence_intervals_rejects_invalid_probability_inputs(
    outcomes, probabilities, message
):
    """Bootstrap intervals require valid one-year survival probabilities."""

    with pytest.raises(ValueError, match=message):
        bootstrap_confidence_intervals(outcomes, probabilities, n_resamples=3, seed=7)


@pytest.mark.parametrize("n_resamples", [0, -1, 1.0, 1.5, True, np.bool_(True)])
def test_bootstrap_confidence_intervals_requires_positive_integer_resamples(n_resamples):
    """Bootstrap resampling counts must be explicit positive integers."""

    with pytest.raises(ValueError, match="positive integer"):
        bootstrap_confidence_intervals(
            np.array([0, 1]), np.array([0.2, 0.8]), n_resamples=n_resamples, seed=7
        )


@pytest.mark.parametrize("seed", [-1, 1.0, 1.5, True, np.bool_(True)])
def test_bootstrap_confidence_intervals_requires_non_negative_integer_seed(seed):
    """Bootstrap random seeds must be reproducible integer values."""

    with pytest.raises(ValueError, match="non-negative integer"):
        bootstrap_confidence_intervals(
            np.array([0, 1]), np.array([0.2, 0.8]), n_resamples=3, seed=seed
        )


def test_bootstrap_confidence_intervals_are_reproducible_and_account_for_resamples():
    """A fixed seed produces stable intervals with complete resample accounting."""

    outcomes = np.tile([0, 1], 10)
    probabilities = np.tile([0.2, 0.8], 10)
    first = bootstrap_confidence_intervals(
        outcomes,
        probabilities,
        n_resamples=np.int64(12),
        seed=np.int64(17),
    )
    second = bootstrap_confidence_intervals(outcomes, probabilities, n_resamples=12, seed=17)

    assert first["brier"] == second["brier"]
    assert first["brier"]["requested_resamples"] == 12
    assert first["brier"]["random_seed"] == 17
    for interval in first.values():
        assert interval["valid_resamples"] + interval["invalid_resamples"] == 12


def test_bootstrap_confidence_intervals_accept_valid_parameter_boundaries():
    """One resample and seed zero remain valid reproducible settings."""

    intervals = bootstrap_confidence_intervals(
        np.tile([0, 1], 10),
        np.tile([0.0, 1.0], 10),
        n_resamples=1,
        seed=0,
    )

    assert intervals["brier"]["requested_resamples"] == 1
    assert intervals["brier"]["random_seed"] == 0


def test_bootstrap_confidence_intervals_allows_valid_one_class_outcomes():
    """One-class samples remain valid while unestimable metrics are counted separately."""

    with pytest.warns(UserWarning, match="single label"):
        intervals = bootstrap_confidence_intervals(
            np.zeros(6, dtype=int),
            np.full(6, 0.2),
            n_resamples=3,
            seed=0,
        )

    assert intervals["roc_auc"]["valid_resamples"] == 0
    assert intervals["roc_auc"]["invalid_resamples"] == 3
    assert intervals["brier"]["valid_resamples"] == 3
    assert intervals["brier"]["invalid_resamples"] == 0
