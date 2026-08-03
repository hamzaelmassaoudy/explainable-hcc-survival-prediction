"""Training-only calibration utilities."""

from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold


def sigmoid_calibrator(estimator: BaseEstimator, *, folds: int, seed: int) -> BaseEstimator:
    """Wrap an unfitted estimator in training-only Platt calibration."""

    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    return CalibratedClassifierCV(estimator=estimator, method="sigmoid", cv=splitter)
