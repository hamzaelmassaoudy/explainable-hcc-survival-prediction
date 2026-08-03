import pytest

from hcc_survival.models import build_model


@pytest.mark.parametrize("name", ["dummy", "logistic_regression", "random_forest"])
def test_models_predict_probabilities(name, synthetic_data):
    x, y = synthetic_data
    model, _ = build_model(name, seed=3)
    model.fit(x, y)
    probabilities = model.predict_proba(x)
    assert probabilities.shape == (len(x), 2)
