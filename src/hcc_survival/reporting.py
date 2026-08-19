"""Model selection, concise reporting, and final all-data refit."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from hcc_survival.artifacts import ensure_local_output_path, package_versions, write_json
from hcc_survival.config import validate_config
from hcc_survival.constants import DATASET_DOI, DEFAULT_MODEL_PATH
from hcc_survival.evaluation import _training_only_variants
from hcc_survival.models import build_model
from hcc_survival.schemas import FEATURE_NAMES, schema_records


def _markdown_table(frame: pd.DataFrame) -> str:
    """Render a compact Markdown table without an undeclared optional dependency."""

    columns = list(frame.columns)
    rows = [
        "| " + " | ".join(map(str, columns)) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    rows.extend(
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    )
    return "\n".join(rows)


def select_model(
    metrics: pd.DataFrame, selection_rule: dict[str, Any]
) -> tuple[pd.Series, dict[str, Any]]:
    """Apply the configured equivalence, stability, and simplicity gates."""

    metric_directions = {"roc_auc": "higher", "pr_auc": "higher", "brier": "lower"}
    stability_margin_names = {
        "roc_auc_repetition_std": "roc_auc_stability_margin",
        "brier_repetition_std": "brier_stability_margin",
    }
    primary_metric = selection_rule.get("primary_metric")
    if primary_metric not in metric_directions:
        raise ValueError(f"Unsupported primary selection metric: {primary_metric!r}.")
    primary_direction = selection_rule.get("primary_direction")
    if primary_direction != metric_directions[primary_metric]:
        raise ValueError("The configured primary metric direction is inconsistent.")
    secondary_metrics = selection_rule.get("secondary_metrics")
    if not isinstance(secondary_metrics, list) or not all(
        isinstance(metric, str) for metric in secondary_metrics
    ):
        raise ValueError("selection.secondary_metrics must be a list of metric names.")
    if len(secondary_metrics) != len(set(secondary_metrics)):
        raise ValueError("selection.secondary_metrics must not contain duplicates.")
    if primary_metric in secondary_metrics:
        raise ValueError("selection.secondary_metrics must not repeat the primary metric.")
    unsupported_secondary = sorted(set(secondary_metrics) - set(metric_directions))
    if unsupported_secondary:
        raise ValueError(f"Unsupported secondary selection metrics: {unsupported_secondary}.")
    stability_metrics = selection_rule.get("stability_metrics")
    if not isinstance(stability_metrics, list) or not all(
        isinstance(metric, str) for metric in stability_metrics
    ):
        raise ValueError("selection.stability_metrics must be a list of metric names.")
    if len(stability_metrics) != len(set(stability_metrics)):
        raise ValueError("selection.stability_metrics must not contain duplicates.")
    unsupported_stability = sorted(set(stability_metrics) - set(stability_margin_names))
    if unsupported_stability:
        raise ValueError(f"Unsupported stability selection metrics: {unsupported_stability}.")

    def equivalence_margin(metric: str) -> float:
        name = f"{metric}_equivalence_margin"
        try:
            margin = float(selection_rule[name])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Missing or invalid selection margin: {name}.") from exc
        if not np.isfinite(margin) or margin < 0:
            raise ValueError(f"Selection margin {name} must be non-negative and finite.")
        return margin

    def stability_margin(metric: str) -> float:
        name = stability_margin_names[metric]
        try:
            margin = float(selection_rule[name])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Missing or invalid selection margin: {name}.") from exc
        if not np.isfinite(margin) or margin < 0:
            raise ValueError(f"Selection margin {name} must be non-negative and finite.")
        return margin

    ece_gate_applies = (
        "ece_equivalence_margin" in selection_rule
        and selection_rule["ece_equivalence_margin"] is not None
    )
    ece_margin = equivalence_margin("ece") if ece_gate_applies else None
    required = {"model", "variant", primary_metric, *secondary_metrics, *stability_metrics}
    if ece_gate_applies:
        required.add("calibration_ece_5_quantile_bins")
    missing = required - set(metrics.columns)
    if missing:
        raise ValueError(f"Metrics are missing columns: {sorted(missing)}")
    candidate_variant = selection_rule.get("candidate_variant")
    if not isinstance(candidate_variant, str):
        raise ValueError("selection.candidate_variant must be a string.")
    candidates = metrics[metrics["variant"] == candidate_variant].copy()
    if candidates.empty:
        raise ValueError(f"No rows use candidate variant {candidate_variant!r}.")
    gate_metrics = [primary_metric, *secondary_metrics, *stability_metrics]
    if ece_gate_applies:
        gate_metrics.append("calibration_ece_5_quantile_bins")
    selection_values = candidates[gate_metrics].to_numpy(dtype=float)
    if not np.isfinite(selection_values).all():
        raise ValueError("Configured selection metrics must contain finite values.")

    def apply_equivalence_gate(
        frame: pd.DataFrame,
        metric: str,
        direction: str,
        margin: float,
    ) -> pd.DataFrame:
        best = float(frame[metric].max() if direction == "higher" else frame[metric].min())
        if direction == "higher":
            return frame[frame[metric] >= best - margin].copy()
        return frame[frame[metric] <= best + margin].copy()

    after_primary = apply_equivalence_gate(
        candidates,
        primary_metric,
        primary_direction,
        equivalence_margin(primary_metric),
    )
    trace_candidates: dict[str, list[str]] = {
        "candidates_initial": candidates["model"].tolist(),
        f"candidates_after_{primary_metric}": after_primary["model"].tolist(),
    }
    after_secondary = after_primary
    for metric in secondary_metrics:
        after_secondary = apply_equivalence_gate(
            after_secondary,
            metric,
            metric_directions[metric],
            equivalence_margin(metric),
        )
        trace_candidates[f"candidates_after_{metric}"] = after_secondary["model"].tolist()
    after_calibration = after_secondary
    if ece_gate_applies:
        after_calibration = apply_equivalence_gate(
            after_secondary,
            "calibration_ece_5_quantile_bins",
            "lower",
            float(ece_margin),
        )
    after_stability = after_calibration
    for metric in stability_metrics:
        after_stability = apply_equivalence_gate(
            after_stability,
            metric,
            "lower",
            stability_margin(metric),
        )
    simplicity = selection_rule.get("simplicity_order")
    if not isinstance(simplicity, dict):
        raise ValueError("selection.simplicity_order must be a mapping.")
    after_stability["simplicity"] = after_stability["model"].map(simplicity)
    if after_stability["simplicity"].isna().any():
        unknown = after_stability.loc[after_stability["simplicity"].isna(), "model"].tolist()
        raise ValueError(f"Selection simplicity order is missing models: {unknown}")
    selected = after_stability.sort_values(["simplicity", "model"], ascending=[True, True]).iloc[0]
    trace = {
        "candidate_variant": candidate_variant,
        "primary_metric": primary_metric,
        "secondary_metrics": secondary_metrics,
        "stability_metrics": stability_metrics,
        "metric_directions": {
            **{primary_metric: primary_direction},
            **{metric: metric_directions[metric] for metric in secondary_metrics},
            **({"calibration_ece_5_quantile_bins": "lower"} if ece_gate_applies else {}),
            "stability_standard_deviations": "lower",
            "simplicity_order": "lower",
        },
        "equivalence_margins": {
            primary_metric: equivalence_margin(primary_metric),
            **{metric: equivalence_margin(metric) for metric in secondary_metrics},
            **({"calibration_ece": ece_margin} if ece_gate_applies else {}),
            **{metric: stability_margin(metric) for metric in stability_metrics},
        },
        **trace_candidates,
        "candidates_after_calibration": after_calibration["model"].tolist(),
        "candidates_after_stability": after_stability["model"].tolist(),
        "selected_model": selected["model"],
        "selected_variant": selected["variant"],
    }
    return selected, trace


def generate_report(run_dir: Path | str) -> Path:
    """Generate a Markdown summary from actual run artifacts."""

    run_dir = ensure_local_output_path(run_dir, purpose="Experiment summary output")
    metrics_path = run_dir / "aggregated_metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"No aggregated metrics found in {run_dir}")
    metrics = pd.read_csv(metrics_path)
    config = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    selected, trace = select_model(metrics, config["selection"])
    lines = [
        "# Experiment Summary",
        "",
        "Research use only. These are internal nested-validation estimates, not clinical",
        "validation and not guarantees of performance in another population.",
        "",
        "## Model comparison",
        "",
        _markdown_table(metrics),
        "",
        "## Predeclared selection",
        "",
        f"Selected: `{selected['model']}` / `{selected['variant']}`.",
        "",
        "Every aggregated probability was generated out of fold and averaged per patient",
        "across repetitions. Preprocessing, tuning, threshold selection, and calibration",
        "were restricted to training data.",
    ]
    output = run_dir / "SUMMARY.md"
    output.write_text("\n".join(lines), encoding="utf-8")
    write_json(
        run_dir / "selection.json",
        {
            "model": selected["model"],
            "variant": selected["variant"],
            "rule": config["selection"],
            "trace": trace,
        },
    )
    return output


def fit_final_model(
    features: pd.DataFrame,
    target: pd.Series,
    run_dir: Path | str,
    *,
    output_path: Path | str = DEFAULT_MODEL_PATH,
) -> Path:
    """Tune and fit the selected pipeline on all data, separate from validation."""

    run_dir = ensure_local_output_path(run_dir, purpose="Final-model input")
    selection = json.loads((run_dir / "selection.json").read_text(encoding="utf-8"))
    if not isinstance(selection, dict):
        raise ValueError("Saved selection record must be a mapping.")
    config = validate_config(yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8")))
    model_name = selection.get("model")
    if not isinstance(model_name, str) or model_name not in config["models"]["include"]:
        raise ValueError("Saved selection model is not configured for this run.")
    selected_variant = selection.get("variant")
    if selected_variant != config["selection"]["candidate_variant"]:
        raise ValueError("Saved selection variant does not match the configured candidate variant.")
    seed = int(config["experiment"]["random_seed"])
    pipeline, grid = build_model(model_name, seed)
    if grid:
        fitted: Any = GridSearchCV(
            pipeline,
            grid,
            scoring="roc_auc",
            cv=StratifiedKFold(
                n_splits=int(config["experiment"]["inner_folds"]),
                shuffle=True,
                random_state=seed,
            ),
            n_jobs=int(config["experiment"].get("n_jobs", 1)),
            refit=True,
        ).fit(features, target)
        tuned_model = fitted.best_estimator_
        best_params = fitted.best_params_
    else:
        tuned_model = pipeline.fit(features, target)
        best_params = {}
    variants, _, calibration_status, training_brier = _training_only_variants(
        tuned_model,
        features,
        target,
        inner_folds=int(config["experiment"]["inner_folds"]),
        seed=seed,
        n_jobs=int(config["experiment"].get("n_jobs", 1)),
        calibration=config["calibration"],
    )
    model = variants[calibration_status]
    output_path = ensure_local_output_path(output_path, purpose="Final model artifact")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": model,
        "feature_names": list(FEATURE_NAMES),
        "schema": schema_records(),
        "metadata": {
            "model_name": model_name,
            "validation_variant": selected_variant,
            "final_calibration_status": calibration_status,
            "training_only_calibration_brier": training_brier,
            "training_timestamp_utc": datetime.now(UTC).isoformat(),
            "dataset_doi": DATASET_DOI,
            "random_seed": seed,
            "selected_hyperparameters": best_params,
            "package_versions": package_versions(),
            "warning": (
                "Final all-data model has no external test set and is not clinically validated."
            ),
        },
    }
    joblib.dump(bundle, output_path)
    write_json(output_path.with_suffix(".metadata.json"), bundle["metadata"])
    return output_path
