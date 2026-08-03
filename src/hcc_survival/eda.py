"""Reproducible exploratory analysis with non-destructive quality checks."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import pandas as pd
import seaborn as sns

from hcc_survival.artifacts import ensure_local_output_path
from hcc_survival.data import data_quality_report
from hcc_survival.schemas import CATEGORICAL_FEATURES, NUMERICAL_FEATURES, ORDINAL_FEATURES

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def generate_eda(features: pd.DataFrame, target: pd.Series, output_dir: Path | str) -> Path:
    """Create concise local EDA tables and figures."""

    output_dir = ensure_local_output_path(output_dir, purpose="EDA output")
    figures = output_dir / "figures"
    tables = output_dir / "tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    quality = data_quality_report(features)
    (tables / "data_quality.json").write_text(json.dumps(quality, indent=2), encoding="utf-8")
    pd.DataFrame(
        {
            "feature": features.columns,
            "missing_count": features.isna().sum().to_numpy(),
            "missing_percent": features.isna().mean().mul(100).to_numpy(),
        }
    ).sort_values("missing_percent", ascending=False).to_csv(
        tables / "missingness_by_feature.csv", index=False
    )
    features.isna().sum(axis=1).rename("missing_feature_count").to_csv(
        tables / "missingness_by_patient.csv", index_label="patient_index"
    )
    target.value_counts().rename_axis("outcome").rename("count").to_csv(
        tables / "target_distribution.csv"
    )
    plt.figure(figsize=(7, 10))
    missing = features.isna().mean().sort_values()
    missing.plot.barh(color="#3478a4")
    plt.xlabel("Missing fraction")
    plt.ylabel("Feature")
    plt.title("Missingness by HCC feature")
    plt.tight_layout()
    plt.savefig(figures / "missingness_by_feature.png", dpi=300)
    plt.close()
    plt.figure(figsize=(6, 4))
    sns.countplot(x=target.map({0: "Died", 1: "Survived"}), color="#3478a4")
    plt.xlabel("One-year outcome")
    plt.ylabel("Patients")
    plt.title("One-year survival outcome distribution")
    plt.tight_layout()
    plt.savefig(figures / "target_distribution.png", dpi=300)
    plt.close()
    correlations = features.loc[:, NUMERICAL_FEATURES].corr(method="spearman")
    correlations.to_csv(tables / "numerical_spearman_correlations.csv")
    summary = {
        "n_patients": len(features),
        "n_features": features.shape[1],
        "numerical_features": len(NUMERICAL_FEATURES),
        "categorical_features": len(CATEGORICAL_FEATURES),
        "ordinal_features": len(ORDINAL_FEATURES),
        "note": "Associations are descriptive; no causal or feature-importance claim is made.",
    }
    (output_dir / "eda_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return output_dir
