import builtins

import numpy as np
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold

from hcc_survival.calibration import sigmoid_calibrator
from hcc_survival.models import available_model_names, build_model


@pytest.mark.parametrize(
    "name",
    ["dummy", "reduced_clinical_logistic", "logistic_regression", "random_forest"],
)
def test_models_predict_probabilities(name, synthetic_data):
    x, y = synthetic_data
    model, _ = build_model(name, seed=3)
    model.fit(x, y)
    probabilities = model.predict_proba(x)
    assert probabilities.shape == (len(x), 2)
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0.0) & (probabilities <= 1.0)).all()


def test_available_model_names_are_stable():
    assert available_model_names() == (
        "dummy",
        "reduced_clinical_logistic",
        "logistic_regression",
        "random_forest",
        "xgboost",
    )


def test_unknown_model_name_has_an_actionable_error():
    with pytest.raises(ValueError, match="Unknown model 'not-a-model'"):
        build_model("not-a-model", seed=3)


def test_xgboost_requires_its_optional_dependency(monkeypatch):
    original_import = builtins.__import__

    def reject_xgboost(name, *args, **kwargs):
        if name == "xgboost":
            raise ImportError("xgboost is unavailable for this test")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_xgboost)

    with pytest.raises(RuntimeError, match="XGBoost was requested but is unavailable"):
        build_model("xgboost", seed=3)


def test_sigmoid_calibration_uses_seeded_stratified_training_folds(synthetic_data):
    x, y = synthetic_data
    pipeline, _ = build_model("logistic_regression", seed=3)

    calibrated = sigmoid_calibrator(pipeline, folds=3, seed=11)

    assert isinstance(calibrated, CalibratedClassifierCV)
    assert isinstance(calibrated.cv, StratifiedKFold)
    assert calibrated.cv.n_splits == 3
    assert calibrated.cv.shuffle is True
    assert calibrated.cv.random_state == 11

    calibrated.fit(x, y)
    probabilities = calibrated.predict_proba(x)
    assert probabilities.shape == (len(x), 2)
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0.0) & (probabilities <= 1.0)).all()
