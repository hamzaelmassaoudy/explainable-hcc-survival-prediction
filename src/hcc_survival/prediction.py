"""Streamlit-independent schema validation and prediction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from hcc_survival.schemas import FEATURE_NAMES, FEATURE_SPECS, PLAUSIBILITY_BOUNDS

MISSING_INPUT_MARKERS = frozenset({"", "?", "na", "n/a", "null", "none"})
MODEL_RECOVERY_COMMAND = "python -m hcc_survival train --config configs/fast.yaml --fit-final"


class PredictionInputError(ValueError):
    """Raised when research-demo prediction input is invalid."""


class ModelArtifactError(PredictionInputError):
    """Raised when a local serialized model cannot be used safely."""


def _missing_input_mask(values: pd.Series) -> pd.Series:
    """Identify explicit missing values without coercing other text to missing."""

    string_values = values.astype("string").str.strip().str.lower()
    return values.isna() | string_values.isin(MISSING_INPUT_MARKERS)


def _format_rows(mask: pd.Series) -> str:
    """Return a bounded, non-sensitive row summary for an input validation error."""

    positions = np.flatnonzero(mask.to_numpy())[:5].tolist()
    suffix = " (and additional rows)" if int(mask.sum()) > len(positions) else ""
    return f"row position(s) {', '.join(str(position) for position in positions)}{suffix}"


def _coerce_numeric_column(column: str, values: pd.Series) -> pd.Series:
    """Convert permitted numeric text while rejecting non-missing nonnumeric input."""

    missing = _missing_input_mask(values)
    numeric = pd.to_numeric(values.where(~missing, np.nan), errors="coerce")
    nonnumeric = ~missing & numeric.isna()
    if nonnumeric.any():
        raise PredictionInputError(
            f"{column} contains a non-numeric non-missing value at {_format_rows(nonnumeric)}. "
            "Use a number or an explicit missing marker."
        )
    nonfinite = ~numeric.isna() & ~np.isfinite(numeric)
    if nonfinite.any():
        raise PredictionInputError(
            f"{column} contains a non-finite numeric value at {_format_rows(nonfinite)}. "
            "Use a finite number or an explicit missing marker."
        )
    return numeric


def _validate_codes(prepared: pd.DataFrame) -> None:
    """Reject category and ordinal codes outside the documented feature schema."""

    for spec in FEATURE_SPECS:
        if spec.categories is None:
            continue
        observed = prepared[spec.name].dropna()
        invalid = ~observed.isin(spec.categories)
        if invalid.any():
            invalid_rows = pd.Series(False, index=prepared.index)
            invalid_rows.loc[observed.index[invalid]] = True
            allowed = ", ".join(str(value) for value in spec.categories)
            raise PredictionInputError(
                f"{spec.label} ({spec.name}) contains an unsupported code at "
                f"{_format_rows(invalid_rows)}. Allowed codes: {allowed}."
            )


def plausibility_warnings(prepared: pd.DataFrame) -> tuple[str, ...]:
    """Return non-destructive broad input-screen flags for a prepared frame.

    These are data-quality screens, not clinical reference ranges. Values are retained
    exactly as supplied after numeric conversion; the caller may decide whether to
    proceed after reviewing the warning.
    """

    warnings: list[str] = []
    for column, (lower, upper) in PLAUSIBILITY_BOUNDS.items():
        values = prepared[column]
        out_of_bounds = pd.Series(False, index=prepared.index)
        if lower is not None:
            out_of_bounds |= values < lower
        if upper is not None:
            out_of_bounds |= values > upper
        if out_of_bounds.any():
            bounds = (
                f"{lower if lower is not None else 'no lower bound'} to "
                f"{upper if upper is not None else 'no upper bound'}"
            )
            warnings.append(
                f"{column}: {int(out_of_bounds.sum())} value(s) fall outside the broad "
                f"research input screen ({bounds}). Values were not changed."
            )
    return tuple(warnings)


def prepare_prediction_frame(records: pd.DataFrame) -> pd.DataFrame:
    """Validate records, sanitize numeric inputs, and enforce schema order.

    Explicit missing markers are converted to ``NaN``. All other values must be
    finite numeric values. Documented categorical and ordinal codes are enforced.
    Broad plausibility flags can be obtained with :func:`plausibility_warnings` and
    never modify accepted numeric values.
    """

    if not isinstance(records, pd.DataFrame):
        raise PredictionInputError("Prediction input must be a tabular data frame.")
    if records.empty:
        raise PredictionInputError("Prediction input must contain at least one row.")
    duplicate_columns = records.columns[records.columns.duplicated()].tolist()
    if duplicate_columns:
        raise PredictionInputError("Duplicate input columns are not allowed.")

    unexpected = sorted(set(records.columns) - set(FEATURE_NAMES))
    if unexpected:
        raise PredictionInputError(
            "Input contains unsupported columns. Use only the documented HCC feature columns."
        )
    prepared = records.copy()
    for column in FEATURE_NAMES:
        if column not in prepared:
            prepared[column] = np.nan
        prepared[column] = _coerce_numeric_column(column, prepared[column])
    prepared = prepared.loc[:, FEATURE_NAMES]
    _validate_codes(prepared)
    return prepared


def load_model_bundle(path: Path | str) -> dict[str, Any]:
    """Load an explicitly trusted local model bundle with actionable errors.

    Joblib artifacts are executable Python objects. Only load bundles produced locally by this
    project or otherwise obtained from a trusted source.
    """

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            "Model artifact not found. Generate a provisional local research model with "
            f"`{MODEL_RECOVERY_COMMAND}`."
        )
    try:
        bundle = joblib.load(path)
    except Exception as exc:
        raise ModelArtifactError(
            "The trusted local model artifact could not be loaded. It may be damaged or "
            f"incompatible. Regenerate the provisional local research model with "
            f"`{MODEL_RECOVERY_COMMAND}`."
        ) from exc
    if not isinstance(bundle, dict) or "model" not in bundle:
        raise ModelArtifactError(
            "The local model artifact has an unsupported format. Regenerate the provisional "
            f"local research model with `{MODEL_RECOVERY_COMMAND}`."
        )
    if not callable(getattr(bundle["model"], "predict_proba", None)):
        raise ModelArtifactError(
            "The local model artifact does not provide a usable probability estimator. "
            f"Regenerate the provisional local research model with `{MODEL_RECOVERY_COMMAND}`."
        )
    return bundle


def predict_survival(bundle: dict[str, Any], records: pd.DataFrame) -> pd.DataFrame:
    """Return clearly labeled one-year survival and mortality probabilities."""

    prepared = prepare_prediction_frame(records)
    try:
        probabilities = np.asarray(bundle["model"].predict_proba(prepared), dtype=float)
    except Exception as exc:
        raise ModelArtifactError(
            "The local model artifact could not generate a prediction. Regenerate the "
            f"provisional local research model with `{MODEL_RECOVERY_COMMAND}`."
        ) from exc
    if probabilities.ndim != 2 or probabilities.shape != (len(prepared), 2):
        raise ModelArtifactError(
            "The local model artifact returned an invalid probability shape. Regenerate the "
            f"provisional local research model with `{MODEL_RECOVERY_COMMAND}`."
        )
    survival = probabilities[:, 1]
    if not np.isfinite(survival).all() or ((survival < 0) | (survival > 1)).any():
        raise ModelArtifactError(
            "The local model artifact returned invalid probabilities. Regenerate the "
            f"provisional local research model with `{MODEL_RECOVERY_COMMAND}`."
        )
    result = prepared.loc[:, records.columns].reset_index(drop=True).copy()
    result["model_estimated_one_year_survival_probability"] = survival
    result["model_estimated_one_year_mortality_probability"] = 1 - survival
    result.attrs["plausibility_warnings"] = plausibility_warnings(prepared)
    return result
