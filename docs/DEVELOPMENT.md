# Development Guide

## Current scope

The repository includes install metadata, shared scientific constants, a YAML configuration
loader, two example experiment configurations, an explicit local UCI dataset loader, an
authoritative 49-feature schema, fold-local preprocessing, candidate models, calibration,
repeated nested validation, and focused synthetic tests. The data loader validates the
documented columns, target coding, and record count before writing a local cache. The
preprocessor returns an unfitted transformer; fit it only within each training fold. The
command-line interface exposes local research workflows; it does not make the project a
clinical or deployment-ready system.

## Supported Python and installation

The package supports Python 3.11 and 3.12. Create an isolated environment, then install the
project with its development tools:

```bash
python -m pip install -e ".[dev,app]"
```

Install `.[boosting]` as well to run the shipped configurations with optional XGBoost.

## Project layout

- `src/hcc_survival/` contains the package.
- `configs/` contains the `fast` and `full` experiment settings.
- `tests/` contains synthetic tests.
- `data/` contains local-data instructions and ignored cache locations.
- `docs/` contains scientific and development documentation.
- `app/` contains the local Streamlit research demonstration.
- `scripts/` contains local verification tooling.

The configurations define repeated nested cross-validation settings. Dataset retrieval occurs
only when `download_dataset()` or `python -m hcc_survival download` is called explicitly. The
CLI may create local experiment artifacts, models, figures, tables, and summaries only beneath
ignored output directories.

## Quality checks

Run the following commands before opening a pull request:

```bash
pytest
ruff check .
ruff format --check .
python -m build
```

The tests use synthetic temporary files. A package build should contain source and metadata,
not data, reports, models, or generated experiment results.

## Data and local outputs

`download_dataset()` retrieves UCI dataset 423 only when called explicitly and writes its cache
under ignored `data/raw/`. Aggregate schema and data-quality summaries, when requested, are
restricted to ignored `data/processed/`. Raw and processed data, models, patient-level
predictions, reports, PDFs, figures, tables, logs, and run artifacts must remain in ignored
local directories. The public project must not depend on committed data, models, reports, or
machine-specific paths.

## Development safeguards

Use `pathlib` for platform-independent paths and fixed, documented random seeds where
applicable. Keep preprocessing, selection, tuning, calibration, and threshold choices inside
training folds. The research-only interface is not a deployment service and must not receive
identifiable health information.

## Troubleshooting

If a command requires an optional dependency, install the matching extra rather than replacing
or editing project files. Do not substitute private data, model files, reports, or environment
values into a public issue.
