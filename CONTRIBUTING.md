# Contributing

## Before opening an issue

Check the [setup instructions](README.md#getting-started) and
[documentation index](docs/README.md). For a bug report, include the command, expected and
observed behavior, Python version, operating system, and project version or commit. Use a
small synthetic example and remove private paths from error text.

Documentation corrections can be reported with the page, section, and proposed wording.
For vulnerabilities or sensitive disclosures, follow the [security policy](SECURITY.md).

## Scope and workflow

This project accepts ordinary developer contributions through focused branches and pull
requests. Use conventional commit messages, keep each pull request cohesive, and describe any
change to scientific behavior or public documentation.

1. Create a focused branch from the current `main` branch.
2. Make the smallest coherent change and update the relevant documentation.
3. Run the quality checks and review the exact files included in the diff.
4. Open a pull request explaining the problem, resulting behavior, and validation results.

The repository includes a local data loader, schema validation, fold-local preprocessing,
candidate models, repeated nested validation, research commands, and focused synthetic tests.
Use Python 3.11 or 3.12, create an isolated environment, install the declared development and
application dependencies, and run the checks below before requesting review.

```bash
python -m pip install -e ".[dev,app]"
pytest
ruff check .
ruff format --check .
python -m build
python scripts/check_distribution_contents.py --dist-dir dist
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
