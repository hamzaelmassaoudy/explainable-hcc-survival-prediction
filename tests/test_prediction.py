import joblib
import numpy as np
import pandas as pd
import pytest

from hcc_survival.prediction import (
    ModelArtifactError,
    PredictionInputError,
    load_model_bundle,
    plausibility_warnings,
    predict_survival,
    prepare_prediction_frame,
)
from hcc_survival.schemas import FEATURE_NAMES


class _FixedProbabilityModel:
    """Small deterministic estimator used to test output handling."""

    classes_ = np.array([0, 1])

    def predict_proba(self, records: pd.DataFrame) -> np.ndarray:
        return np.tile(np.array([0.25, 0.75]), (len(records), 1))


def test_prediction_frame_adds_unknown_columns():
    result = prepare_prediction_frame(pd.DataFrame([{"Age": 60}]))
    assert list(result.columns) == list(FEATURE_NAMES)
    assert result["Age"].iloc[0] == 60


def test_prediction_frame_rejects_unexpected_columns():
    with pytest.raises(PredictionInputError):
        prepare_prediction_frame(pd.DataFrame([{"not_a_feature": 1}]))


def test_prediction_frame_rejects_non_numeric_non_missing_values() -> None:
    records = pd.DataFrame([{"Age": "=1+1"}], index=["private-row-label"])
    with pytest.raises(
        PredictionInputError, match="Age contains a non-numeric non-missing value"
    ) as error:
        prepare_prediction_frame(records)
    assert "=1+1" not in str(error.value)
    assert "private-row-label" not in str(error.value)


def test_prediction_frame_does_not_echo_unsupported_header_text() -> None:
    with pytest.raises(PredictionInputError) as error:
        prepare_prediction_frame(pd.DataFrame([{"=private-header": 1}]))
    assert "=private-header" not in str(error.value)


@pytest.mark.parametrize(("column", "invalid_code"), [("Gender", 2), ("PS", 5)])
def test_prediction_frame_rejects_unsupported_categorical_and_ordinal_codes(
    column: str, invalid_code: int
) -> None:
    with pytest.raises(PredictionInputError, match="unsupported code"):
        prepare_prediction_frame(pd.DataFrame([{column: invalid_code}]))


def test_plausibility_warnings_preserve_validated_numeric_values() -> None:
    prepared = prepare_prediction_frame(pd.DataFrame([{"Age": "121", "Sat": 126}]))
    warnings = plausibility_warnings(prepared)

    assert prepared.loc[0, "Age"] == 121
    assert prepared.loc[0, "Sat"] == 126
    assert any(warning.startswith("Age:") for warning in warnings)
    assert any(warning.startswith("Sat:") for warning in warnings)
    assert all("Values were not changed" in warning for warning in warnings)


def test_prediction_output_uses_sanitized_numeric_input_values() -> None:
    records = pd.DataFrame([{"Age": "60", "Gender": "1", "PS": "0"}])
    output = predict_survival({"model": _FixedProbabilityModel()}, records)

    assert output.columns[:3].tolist() == ["Age", "Gender", "PS"]
    assert output.loc[0, "Age"] == 60
    assert output.loc[0, "Gender"] == 1
    assert pd.api.types.is_numeric_dtype(output["Age"])
    assert "=1+1" not in output.to_csv(index=False)


def test_corrupt_or_malformed_model_artifact_has_safe_recovery_message(tmp_path) -> None:
    corrupt = tmp_path / "corrupt.joblib"
    corrupt.write_bytes(b"not a serialized model")
    with pytest.raises(ModelArtifactError) as corrupt_error:
        load_model_bundle(corrupt)
    assert "provisional local research model" in str(corrupt_error.value)
    assert str(corrupt) not in str(corrupt_error.value)

    malformed = tmp_path / "malformed.joblib"
    joblib.dump({"model": object()}, malformed)
    with pytest.raises(ModelArtifactError, match="usable probability estimator"):
        load_model_bundle(malformed)


def test_missing_model_artifact_does_not_echo_local_path(tmp_path) -> None:
    missing = tmp_path / "private-model.joblib"
    with pytest.raises(FileNotFoundError) as error:
        load_model_bundle(missing)
    assert str(missing) not in str(error.value)
    assert "provisional local research model" in str(error.value)


def test_prediction_rejects_malformed_probability_output() -> None:
    class BadProbabilityModel:
        classes_ = np.array([0, 1])

        def predict_proba(self, records: pd.DataFrame) -> np.ndarray:
            return np.array([[0.2]])

    with pytest.raises(ModelArtifactError, match="invalid probability shape"):
        predict_survival({"model": BadProbabilityModel()}, pd.DataFrame([{"Age": 60}]))


@pytest.mark.parametrize("classes", [None, np.array([1, 0])])
def test_prediction_requires_declared_survival_class_order(classes: np.ndarray | None) -> None:
    model = _FixedProbabilityModel()
    model.classes_ = classes

    with pytest.raises(ModelArtifactError, match="class order"):
        predict_survival({"model": model}, pd.DataFrame([{"Age": 60}]))


@pytest.mark.parametrize(
    "probabilities",
    [
        np.array([[-0.1, 1.1]]),
        np.array([[0.2, 0.6]]),
        np.array([[np.nan, 1.0]]),
    ],
)
def test_prediction_rejects_invalid_probability_matrix(probabilities: np.ndarray) -> None:
    class InvalidProbabilityModel:
        classes_ = np.array([0, 1])

        def predict_proba(self, records: pd.DataFrame) -> np.ndarray:
            return np.repeat(probabilities, len(records), axis=0)

    with pytest.raises(ModelArtifactError, match="invalid probabilities"):
        predict_survival({"model": InvalidProbabilityModel()}, pd.DataFrame([{"Age": 60}]))
