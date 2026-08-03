# Model Card

## Model purpose

This is a research-only prototype for estimating `P(survived at one year)` from the UCI
HCC Survival dataset. It is an educational example of leakage-safe tabular prediction,
not a clinically validated model.

## Candidate models and local artifacts

The pipeline compares a prior-probability dummy baseline, a reduced clinical logistic
benchmark, full regularized logistic regression, a constrained random forest, and
conservative CPU XGBoost. The configured `training_selected` procedure applies the documented
metric-equivalence, stability, and simplicity rules to local internal-validation output. Any
final all-data fitted model is a separate local research artifact and has no independent
performance estimate.

No trained model, aggregate result, selected model, or calibration decision is published in
this repository. Detailed result files and serialized estimators remain local and are
intentionally excluded.

## Evaluation design

The primary design is five outer folds repeated five times, with three inner folds.
Imputation, encoding, scaling, hyperparameter tuning, calibration selection, and optional
threshold selection occur on training data only. Repeated held-out probabilities are
averaged at patient level before aggregate metrics and percentile-bootstrap
internal-validation intervals. See [methods](METHODS.md).

## Outputs

The principal model output is one-year survival probability. Threshold metrics are named
as survivor sensitivity and death specificity to preserve their direction. Predictive
importance and model coefficients are descriptive associations, never causal claims.

## Limitations and prohibited use

There is no external validation. The small single-source cohort, missingness, model
selection, uncertain calibration, and exploratory subgroups materially limit inference.
Do not use this work for diagnosis, treatment, triage, medical decisions, or patient
prognosis communication. See [intended use](INTENDED_USE.md) and
[limitations](LIMITATIONS.md).
