# Reproducibility Guide

## Environment

Use Python 3.11 or 3.12. Dependency versions are constrained in `pyproject.toml` but this
repository intentionally does not claim bitwise reproducibility across operating systems
or every compatible dependency version. Each local experiment records its exact package
versions and configuration in an ignored artifact directory.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,app,boosting]"
```

The shipped `fast` and `full` configurations include optional CPU XGBoost. If you do not need
those configurations, omit the `boosting` extra and remove `xgboost` from a copied local
configuration rather than editing the tracked examples.

## Clean-copy smoke workflow

A fresh clone requires no dataset cache, report, model binary, or experiment directory.
Run:

```bash
python -m hcc_survival --help
pytest --collect-only -q
pytest
ruff check .
ruff format --check .
python -m build
```

`python scripts/verify_project.py` runs the core checks with the active interpreter and
writes detailed machine-readable evidence to `local_audit/test_execution.json`. That file
is intentionally ignored.

## Dataset retrieval and validation

```bash
python -m hcc_survival download
python -m hcc_survival validate-data
```

The downloader retrieves UCI dataset 423 over HTTPS, validates the expected 49-feature
schema and binary target, then caches it locally. It validates schema rather than claiming
an immutable archive checksum; record the generated local dataset hash when an exact run
must be audited.

## Fast development experiment

```bash
python -m hcc_survival train --config configs/fast.yaml --fit-final
```

This small configuration is for development and smoke testing. Its output is written below
`artifacts/` and is ignored by Git.

## Full internal-validation experiment

```bash
python -m hcc_survival train --config configs/full.yaml --fit-final
python -m hcc_survival sensitivity --config configs/full.yaml \
  --output artifacts/full_missingness_sensitivity
```

The full configuration uses five outer folds, five repetitions, three inner folds, and
2,000 patient-level bootstrap resamples. It can take substantially longer than the fast
configuration. The command creates a new immutable local run directory; it never
overwrites an existing run directory.

## Local analysis outputs and application check

Replace `<run-id>` with the directory printed by the full training command.

```bash
python -m hcc_survival report --run-dir artifacts/runs/<run-id>
python -m hcc_survival explain --run-dir artifacts/runs/<run-id>
streamlit run app/streamlit_app.py --server.headless true
```

Markdown summaries, plots, tables, model artifacts, and run directories remain local and
excluded from version control. The app deliberately reports an actionable research-only message
when no local model artifact exists. It is a local demonstration, not a secure health-data
service; do not upload real or identifiable health information.

## Determinism expectations

The deterministic tests run the fast synthetic workflow twice with the same seed and
compare patient-level predictions and aggregate metrics after excluding run-specific
timestamps and directory names. The expected equality applies to the tested configuration
and dependency environment; it is not a promise that every BLAS library, CPU, or future
package version will produce bitwise-identical output.
