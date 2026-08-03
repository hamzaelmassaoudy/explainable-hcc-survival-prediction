"""Command-line interface for reproducible research workflows."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from hcc_survival.config import load_config
from hcc_survival.constants import DEFAULT_DATA_PATH, DEFAULT_MODEL_PATH
from hcc_survival.data import (
    download_dataset,
    load_local_dataset,
    write_data_outputs,
)
from hcc_survival.diagnostics import run_diagnostics
from hcc_survival.eda import generate_eda
from hcc_survival.evaluation import run_nested_experiment
from hcc_survival.explainability import aggregate_permutation_importance
from hcc_survival.reporting import fit_final_model, generate_report
from hcc_survival.sensitivity import run_missingness_sensitivity

LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build the public CLI parser."""

    parser = argparse.ArgumentParser(
        prog="hcc-survival",
        description="Explainable HCC one-year survival research pipeline (not clinical use).",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable detailed logging.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser("download", help="Download and cache UCI dataset 423.")
    download.add_argument("--output", type=Path, default=DEFAULT_DATA_PATH)
    download.add_argument("--force", action="store_true")

    validate = subparsers.add_parser("validate-data", help="Validate cached data and schema.")
    validate.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)

    eda = subparsers.add_parser("eda", help="Generate reproducible exploratory analysis.")
    eda.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    eda.add_argument("--output", type=Path, default=Path("reports"))

    train = subparsers.add_parser("train", help="Run repeated nested cross-validation.")
    train.add_argument("--config", type=Path, required=True)
    train.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    train.add_argument("--artifact-root", type=Path, default=None)
    train.add_argument(
        "--fit-final",
        action="store_true",
        help="After reporting, refit the selected model on all available data.",
    )
    train.add_argument("--model-output", type=Path, default=DEFAULT_MODEL_PATH)

    report = subparsers.add_parser("report", help="Generate a report for an existing run.")
    report.add_argument("--run-dir", type=Path, required=True)

    explain = subparsers.add_parser("explain", help="Aggregate held-out predictive explanations.")
    explain.add_argument("--run-dir", type=Path, required=True)
    sensitivity = subparsers.add_parser(
        "sensitivity", help="Run the predeclared missingness sensitivity analysis."
    )
    sensitivity.add_argument("--config", type=Path, required=True)
    sensitivity.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    sensitivity.add_argument(
        "--output", type=Path, default=Path("artifacts/missingness_sensitivity")
    )
    return parser


def dispatch(args: argparse.Namespace) -> int:
    """Execute one parsed command."""

    if args.command == "download":
        features, _ = download_dataset(args.output, force=args.force)
        write_data_outputs(features)
        LOGGER.info("Validated %d patients and %d features.", *features.shape)
    elif args.command == "validate-data":
        features, _ = load_local_dataset(args.data)
        write_data_outputs(features)
        LOGGER.info("Data validation succeeded: %d patients, %d features.", *features.shape)
    elif args.command == "eda":
        features, target = load_local_dataset(args.data)
        output = generate_eda(features, target, args.output)
        LOGGER.info("EDA written to %s", output)
    elif args.command == "train":
        config = load_config(args.config)
        features, target = load_local_dataset(args.data)
        run_dir = run_nested_experiment(
            features,
            target,
            config,
            artifact_root=args.artifact_root,
            dataset_path=args.data,
        )
        generate_report(run_dir)
        aggregate_permutation_importance(run_dir)
        model_path = None
        if args.fit_final:
            fit_final_model(features, target, run_dir, output_path=args.model_output)
            model_path = args.model_output
        run_diagnostics(
            run_dir,
            features,
            target,
            model_path=model_path,
        )
        LOGGER.info("Experiment completed: %s", run_dir)
    elif args.command == "report":
        LOGGER.info("Report written to %s", generate_report(args.run_dir))
    elif args.command == "explain":
        LOGGER.info(
            "Aggregated held-out permutation importance written to %s",
            aggregate_permutation_importance(args.run_dir),
        )
    elif args.command == "sensitivity":
        config = load_config(args.config)
        features, target = load_local_dataset(args.data)
        output = run_missingness_sensitivity(features, target, config, args.output)
        LOGGER.info("Missingness sensitivity written to %s", output)
    else:
        raise ValueError(f"Unsupported command: {args.command}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point with nonzero status and concise errors."""

    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    try:
        return dispatch(args)
    except (FileNotFoundError, ValueError, RuntimeError, ConnectionError) as exc:
        LOGGER.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        LOGGER.error("Interrupted.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
