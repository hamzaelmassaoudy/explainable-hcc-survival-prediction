# Contributing

## Scope and workflow

This project accepts ordinary developer contributions through focused branches and pull
requests. Use conventional commit messages, keep each pull request cohesive, and describe any
change to scientific behavior or public documentation.

The repository includes a local data loader, schema validation, fold-local preprocessing,
candidate models, repeated nested validation, research commands, and focused synthetic tests.
Use Python 3.11 or 3.12, create an isolated environment, install the declared development and
application dependencies, and run the checks below before requesting review.

```bash
python -m pip install -e ".[dev,app]"
pytest
ruff check .
ruff format --check .
```

Install `.[boosting]` as well when changing or running configurations that include XGBoost.
The tests use generated synthetic values or temporary files only. Add focused tests with every
change to public behavior.

## Code and scientific standards

Use Python 3.11–3.12, focused functions, type hints, docstrings, `pathlib`, and UTF-8. Preserve
leakage-safe validation: preprocessing, feature selection, tuning, calibration, and threshold
selection must be learned within training folds. Keep the target direction and research-only
scope accurate in code and documentation. Update the relevant documentation whenever a change
affects methods, data handling, validation, or public behavior.

## Data and artifact boundary

Do not submit raw or processed patient data, patient-level predictions, model binaries,
reports, generated figures or tables, local audit output, credentials, or fabricated results.
Use synthetic fixtures only. Do not add reports, generated model artifacts, PDFs, logs, or
environment files to a pull request.
