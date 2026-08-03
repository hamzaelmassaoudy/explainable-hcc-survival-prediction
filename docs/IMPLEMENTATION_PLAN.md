# Implementation Plan and Scientific Safeguards

## Purpose

This repository is a reproducible medical-machine-learning case study, not a clinical
product. It estimates `P(survived at one year)` for the UCI HCC Survival dataset, where
`1 = survived at one year` and `0 = died within one year`.

## Public and local architecture

- `src/hcc_survival/`: reusable data, preprocessing, evaluation, and application package
- `configs/`: versioned experiment configurations
- `tests/`: deterministic tests using generated synthetic fixtures
- `app/`: Streamlit research demonstration
- `docs/`: public methods, limitations, privacy, and reproducibility documentation
- `data/`: download instructions and empty local-data scaffolding
- `scripts/`: source for local verification
- `artifacts/`, `reports/`, `local_audit/`: ignored local outputs only

## Modeling workflow

1. Retrieve UCI HCC Survival data locally and validate it against the checked-in schema.
2. Preserve missing values and record broad quality flags without silently correcting data.
3. Fit all imputation, encoding, scaling, tuning, calibration, and threshold choices on
   training partitions only.
4. Compare the dummy baseline, reduced clinical logistic benchmark, full regularized
   logistic regression, constrained random forest, and optional CPU XGBoost.
5. Use repeated nested stratified cross-validation for internal validation.
6. Average repeated held-out survival probabilities by patient before aggregate metrics
   and patient-level bootstrap intervals.
7. Select models with the checked-in metric-equivalence hierarchy, then refit the selected
   approach on all available data only as a separate research artifact.

## Non-negotiable safeguards

- No imaging, radiomics, deep learning, neural networks, diagnosis, treatment, or medical
  advice.
- No outer-validation leakage through preprocessing, tuning, calibration, feature
  selection, or threshold selection.
- No causal claims from coefficients, permutation importance, SHAP, or subgroup results.
- No external-validation claim without an independent cohort.
- No raw data, patient-level derivatives, model binaries, reports, PDFs, or local audit
  outputs in the public repository.

## Quality gates

Before a public commit, run the tests, Ruff lint and format checks, package build, documentation
link review, and the explicit staging review described in the
[release checklist](RELEASE_CHECKLIST.md). Generated evidence belongs in ignored local paths.
