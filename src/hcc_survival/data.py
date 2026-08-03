"""Local UCI HCC dataset retrieval and schema validation."""

from __future__ import annotations

import io
import json
import logging
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from hcc_survival.constants import DATASET_ID, DEFAULT_DATA_PATH, TARGET_NAME
from hcc_survival.schemas import (
    FEATURE_NAMES,
    FEATURE_SPECS,
    PLAUSIBILITY_BOUNDS,
    schema_records,
)

LOGGER = logging.getLogger(__name__)
MISSING_MARKERS = ("?", "", "NA", "N/A", "null", "None")
EXPECTED_RECORD_COUNT = 165
OFFICIAL_ARCHIVE_URL = "https://archive.ics.uci.edu/static/public/423/hcc+survival.zip"
_PROCESSED_DATA_ROOT = Path("data") / "processed"


class DataValidationError(ValueError):
    """Raised when local or downloaded data violate the documented HCC schema."""


def _local_output_path(path: Path | str, root: Path, *, purpose: str) -> Path:
    """Require a local output path to remain inside an ignored project directory."""

    candidate = Path(path)
    try:
        candidate.resolve().relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"{purpose} must be stored under {root.as_posix()}.") from error
    return candidate


def _normalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert documented missing markers and reject other non-numeric values."""

    cleaned = frame.copy()
    for column in cleaned.columns:
        values = cleaned[column]
        normalized = values.mask(values.isin(MISSING_MARKERS))
        numeric = pd.to_numeric(normalized, errors="coerce")
        invalid_text = normalized.notna() & numeric.isna()
        if invalid_text.any():
            raise DataValidationError(f"{column} contains non-numeric values.")
        if not np.isfinite(numeric.dropna()).all():
            raise DataValidationError(f"{column} contains non-finite values.")
        cleaned[column] = numeric
    return cleaned


def _validate_category_codes(features: pd.DataFrame) -> None:
    """Check the documented integer codes for categorical and ordinal variables."""

    for spec in FEATURE_SPECS:
        if spec.categories is None:
            continue
        values = features[spec.name].dropna().to_numpy()
        if not np.all(np.equal(values, np.floor(values))):
            raise DataValidationError(f"{spec.name} must use integer category codes.")
        observed = set(values.astype(int))
        if not observed.issubset(set(spec.categories)):
            raise DataValidationError(
                f"{spec.name} contains values outside documented category codes."
            )


def validate_dataset(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Validate and split a combined feature-and-target frame without altering values."""

    expected = [*FEATURE_NAMES, TARGET_NAME]
    if frame.columns.has_duplicates:
        raise DataValidationError("Dataset columns must be unique.")
    if set(frame.columns) != set(expected):
        raise DataValidationError("Dataset columns do not match the documented HCC schema.")
    if frame.columns.tolist() != expected:
        frame = frame.loc[:, expected]
    frame = _normalize_frame(frame)
    target = frame[TARGET_NAME]
    if target.isna().any():
        raise DataValidationError("Target contains missing or non-numeric values.")
    invalid_targets = sorted(set(target.unique()) - {0, 1})
    if invalid_targets:
        raise DataValidationError("Target must contain only the documented 0 and 1 codes.")
    features = frame.loc[:, FEATURE_NAMES]
    _validate_category_codes(features)
    return features, target.astype(int).rename(TARGET_NAME)


def _validate_hcc_record_count(features: pd.DataFrame) -> None:
    """Require the published UCI HCC record count for a local dataset cache."""

    if len(features) != EXPECTED_RECORD_COUNT:
        raise DataValidationError(
            f"Dataset must contain the documented {EXPECTED_RECORD_COUNT} HCC records."
        )


def load_local_dataset(path: Path | str = DEFAULT_DATA_PATH) -> tuple[pd.DataFrame, pd.Series]:
    """Load a locally cached HCC CSV and validate its documented contract."""

    cache_path = Path(path)
    if not cache_path.is_file():
        raise FileNotFoundError("Dataset cache was not found. Use download_dataset() to create it.")
    features, target = validate_dataset(pd.read_csv(cache_path, na_values=list(MISSING_MARKERS)))
    _validate_hcc_record_count(features)
    return features, target


def _download_from_ucimlrepo() -> pd.DataFrame:
    """Retrieve the official dataset through the maintained UCI Python client."""

    from ucimlrepo import fetch_ucirepo

    dataset = fetch_ucirepo(id=DATASET_ID)
    features = dataset.data.features.copy()
    target = dataset.data.targets.copy()
    if list(features.columns) != list(FEATURE_NAMES) or target.shape[1] != 1:
        raise DataValidationError("Remote UCI data do not match the documented HCC schema.")
    target.columns = [TARGET_NAME]
    return pd.concat([features, target], axis=1)


def _download_from_official_archive() -> pd.DataFrame:
    """Retrieve the official UCI archive without extracting it to disk."""

    try:
        with requests.get(OFFICIAL_ARCHIVE_URL, timeout=60) as response:
            response.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
                data_name = "hcc-survival/hcc-data.txt"
                with archive.open(data_name) as source:
                    frame = pd.read_csv(
                        source,
                        header=None,
                        na_values=list(MISSING_MARKERS),
                    )
    except Exception as error:
        raise ConnectionError(
            "Could not retrieve the official UCI HCC Survival archive."
        ) from error
    if frame.shape[1] != len(FEATURE_NAMES) + 1:
        raise DataValidationError("Official UCI archive does not have the expected HCC columns.")
    frame.columns = [*FEATURE_NAMES, TARGET_NAME]
    return frame


def download_dataset(
    cache_path: Path | str = DEFAULT_DATA_PATH,
    *,
    force: bool = False,
) -> tuple[pd.DataFrame, pd.Series]:
    """Download UCI dataset 423 and cache only a validated local CSV under ``data/raw``."""

    cache_path = _local_output_path(
        cache_path,
        DEFAULT_DATA_PATH.parent,
        purpose="Dataset caches",
    )
    if cache_path.suffix.lower() != ".csv":
        raise ValueError("Dataset caches must use the .csv extension.")
    if cache_path.is_file() and not force:
        LOGGER.info("Using the validated local dataset cache.")
        return load_local_dataset(cache_path)
    try:
        frame = _download_from_ucimlrepo()
    except DataValidationError:
        raise
    except Exception:
        LOGGER.info("UCI client retrieval was unavailable; using the official archive fallback.")
        frame = _download_from_official_archive()
    features, target = validate_dataset(frame)
    _validate_hcc_record_count(features)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat([features, target], axis=1).to_csv(cache_path, index=False)
    LOGGER.info("Cached a validated local HCC dataset.")
    return features, target


def data_quality_report(features: pd.DataFrame) -> dict[str, object]:
    """Return aggregate missingness and plausibility summaries without row-level values."""

    plausibility_flags: list[dict[str, object]] = []
    for column, (lower_bound, upper_bound) in PLAUSIBILITY_BOUNDS.items():
        values = pd.to_numeric(features[column], errors="coerce")
        outside_bounds = pd.Series(False, index=values.index)
        if lower_bound is not None:
            outside_bounds |= values < lower_bound
        if upper_bound is not None:
            outside_bounds |= values > upper_bound
        flagged_count = int(outside_bounds.sum())
        if flagged_count:
            plausibility_flags.append(
                {
                    "feature": column,
                    "lower_bound": lower_bound,
                    "upper_bound": upper_bound,
                    "flagged_value_count": flagged_count,
                    "action": "Flagged only; no values were deleted, clipped, or corrected.",
                }
            )
    return {
        "n_patients": len(features),
        "n_features": int(features.shape[1]),
        "missing_cells": int(features.isna().sum().sum()),
        "missing_fraction": float(features.isna().to_numpy().mean()),
        "missing_by_feature": features.isna().sum().astype(int).to_dict(),
        "plausibility_flags": plausibility_flags,
        "note": "All summaries are aggregate; no row-level values are written.",
    }


def write_data_outputs(
    features: pd.DataFrame,
    output_dir: Path | str = _PROCESSED_DATA_ROOT,
) -> None:
    """Write aggregate schema and quality summaries only under ignored ``data/processed``."""

    output_dir = _local_output_path(output_dir, _PROCESSED_DATA_ROOT, purpose="Data summaries")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "schema.json").write_text(
        json.dumps(schema_records(), indent=2),
        encoding="utf-8",
    )
    (output_dir / "data_quality.json").write_text(
        json.dumps(data_quality_report(features), indent=2),
        encoding="utf-8",
    )
