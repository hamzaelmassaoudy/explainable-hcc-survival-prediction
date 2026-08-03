# Test Evidence

The public test suite uses deterministic, programmatically generated fixtures. It contains
no real patient rows and does not download the UCI dataset during ordinary unit tests.
Tests check software and scientific invariants; they do not establish clinical utility,
causal relationships, or external generalization.

## Core scientific contracts

- Target direction: `1 = survived at one year`; model probabilities are one-year survival
  probabilities.
- Metric direction: ROC-AUC and PR-AUC are higher-is-better; Brier score is
  lower-is-better; threshold metrics are named survivor sensitivity and death specificity.
- Schema integrity: feature groups are disjoint, complete, and stable; missing markers and
  unknown categories are handled safely.
- Fold boundaries: outer training and validation sets do not overlap; preprocessing,
  tuning, calibration, threshold selection, and data-derived sensitivity subsets are
  training-local.
- Repeated out-of-fold handling: every patient has one held-out prediction per repetition
  and five per variant in the configured 5-fold × 5-repetition design.
- Patient-level evaluation: repeated held-out probabilities are aggregated by patient
  before metrics and bootstrap uncertainty are calculated.
- Model selection: configured equivalence gates are deterministic, reject incomplete or
  non-finite inputs, and use simplicity only after metric and stability gates.

## Engineering and privacy contracts

- Fixed-seed synthetic experiments reproduce the tested patient-level predictions and
  aggregate metrics.
- Model serialization has a save/load consistency test; an absent or malformed local model
  produces an actionable error.
- Prediction inputs reject unknown columns, nonnumeric nonmissing values, invalid category
  codes, and values outside broad quality bounds. Output rows use validated values rather
  than raw uploaded text.
- CLI errors are nonzero and actionable. Streamlit source is checked for the one-year
  survival estimand, research-only warning, and missing-model recovery guidance.

## Commands

```bash
pytest --collect-only -q
pytest
ruff check .
ruff format --check .
python -m build
python scripts/verify_project.py
```

The verifier writes exact commands, exit codes, stdout, stderr, timestamps, and interpreter
information to the ignored local path `local_audit/test_execution.json`. Only the
non-sensitive test map and commands are versioned.
