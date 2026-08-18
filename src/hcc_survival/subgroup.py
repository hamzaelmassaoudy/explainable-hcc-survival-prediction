"""Cautious exploratory subgroup evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from hcc_survival.metrics import classification_metrics

MIN_GROUP_SIZE = 30
MIN_OUTCOME_COUNT = 10


def _require_patient_aggregated_predictions(predictions: pd.DataFrame) -> None:
    """Require exactly one non-null prediction row for each patient."""

    if "patient_index" not in predictions:
        raise ValueError("Subgroup metrics require a patient_index column.")
    patient_index = predictions["patient_index"]
    if patient_index.isna().any() or patient_index.duplicated().any():
        raise ValueError("Subgroup metrics require one patient-aggregated prediction per patient.")


def exploratory_subgroup_metrics(
    predictions: pd.DataFrame, subgroup: pd.Series, *, threshold: float = 0.5
) -> pd.DataFrame:
    """Evaluate groups with predeclared suppression rules."""

    _require_patient_aggregated_predictions(predictions)
    frame = predictions.copy()
    frame["subgroup"] = subgroup.reindex(frame["patient_index"]).to_numpy()
    rows: list[dict[str, object]] = []
    for name, group in frame.groupby("subgroup", dropna=False):
        counts = group["observed"].value_counts()
        reliable = (
            len(group) >= MIN_GROUP_SIZE
            and counts.get(0, 0) >= MIN_OUTCOME_COUNT
            and counts.get(1, 0) >= MIN_OUTCOME_COUNT
        )
        base: dict[str, object] = {
            "subgroup": str(name),
            "n": len(group),
            "died": int(counts.get(0, 0)),
            "survived": int(counts.get(1, 0)),
            "metrics_suppressed": not reliable,
            "warning": (
                ""
                if reliable
                else "Suppressed: fewer than 30 patients or fewer than 10 in an outcome class."
            ),
        }
        if reliable:
            metrics = classification_metrics(
                group["observed"].to_numpy(),
                group["probability_survived_one_year"].to_numpy(),
                threshold,
            )
            base.update(
                {
                    key: metrics[key]
                    for key in (
                        "roc_auc",
                        "survivor_sensitivity",
                        "death_specificity",
                        "brier",
                    )
                }
            )
        else:
            base.update(
                {
                    key: np.nan
                    for key in (
                        "roc_auc",
                        "survivor_sensitivity",
                        "death_specificity",
                        "brier",
                    )
                }
            )
        rows.append(base)
    return pd.DataFrame(rows)


def predefined_subgroups(features: pd.DataFrame) -> dict[str, pd.Series]:
    """Return sex and clinically interpretable age groups without outcome access."""

    age_groups = pd.cut(
        features["Age"],
        bins=[-np.inf, 59, 69, np.inf],
        labels=["<60", "60-69", "70+"],
    )
    sex = features["Gender"].map({0: "Female", 1: "Male"}).fillna("Unknown")
    return {"sex": sex, "age_group": age_groups}
