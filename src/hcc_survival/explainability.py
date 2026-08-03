"""Explainability exports with careful scope labels."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from hcc_survival.artifacts import ensure_local_output_path

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def aggregate_permutation_importance(run_dir: Path | str) -> Path:
    """Aggregate validation-fold permutation importance and its variability."""

    run_dir = ensure_local_output_path(run_dir, purpose="Explainability output")
    source = run_dir / "tables" / "held_out_permutation_importance.csv"
    if not source.exists():
        raise FileNotFoundError(f"Held-out permutation importance not found: {source}")
    frame = pd.read_csv(source)
    output = (
        frame.groupby(["model", "variant", "feature"], as_index=False)
        .agg(
            importance_mean=("importance_mean", "mean"),
            importance_between_fold_std=("importance_mean", "std"),
            valid_estimates=("importance_mean", "count"),
            importance_lower_95=("importance_mean", lambda values: np.quantile(values, 0.025)),
            importance_upper_95=("importance_mean", lambda values: np.quantile(values, 0.975)),
        )
        .sort_values(["model", "variant", "importance_mean"], ascending=[True, True, False])
    )
    path = run_dir / "tables" / "permutation_importance_aggregated.csv"
    output.to_csv(path, index=False)
    figures = run_dir / "figures"
    figures.mkdir(exist_ok=True)
    for model, group in output.groupby("model"):
        top = group.nlargest(15, "importance_mean").sort_values("importance_mean")
        plt.figure(figsize=(8, 6))
        plt.barh(
            top["feature"],
            top["importance_mean"],
            xerr=top["importance_between_fold_std"].fillna(0),
            color="#3478a4",
        )
        plt.axvline(0, color="black", linewidth=0.8)
        plt.xlabel("Held-out ROC-AUC decrease after permutation")
        plt.title(f"Validation-aligned permutation importance: {model}")
        plt.tight_layout()
        plt.savefig(figures / f"permutation_importance_{model}.png", dpi=300)
        plt.close()
    note = run_dir / "EXPLAINABILITY.md"
    note.write_text(
        "# Explainability\n\n"
        "The permutation-importance table was calculated on held-out outer-fold patients. "
        "It is validation-aligned predictive importance, not causal evidence. Importance "
        "may be shared or unstable when predictors are correlated.\n\n"
        "SHAP is generated only for a compatible selected final model and, when generated, "
        "describes that all-data fitted model rather than unbiased generalizable importance.\n",
        encoding="utf-8",
    )
    return path
