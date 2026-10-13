# Learning Guide

This guide explains the main ideas behind the implementation. For installation, use the
[project README](../README.md#getting-started); for the detailed validation design, read the
[methods](METHODS.md).

## The prediction task

Hepatocellular carcinoma (HCC) is a form of liver cancer. This project uses a public
tabular dataset to model one-year survival, with `1 = survived at one year` and
`0 = died within one year`. Features are the input variables; the target is the observed
outcome. The output is a model-estimated one-year survival probability.

This is a fixed-horizon binary classification task. It does not model time to an event or
censoring. Its outputs describe predictive associations in the source cohort and are not
medical advice or causal explanations.

## Preparing the inputs

Numerical, categorical, and ordinal variables require different transformations.
Imputation supplies values for missing inputs, encoding represents categories numerically,
and scaling changes the numerical representation for models that need it.

All learned transformations must be fitted on the training data within each fold.
Learning them from validation records would leak information into model development.
See [preprocessing.py](../src/hcc_survival/preprocessing.py) and the
[data dictionary](DATA_DICTIONARY.md).

## Comparing models

| Model | Role in the project |
| --- | --- |
| Dummy classifier | A reference prediction based on training-set class prevalence |
| Reduced clinical logistic regression | A benchmark using a predefined smaller feature set |
| Full logistic regression | A regularized model that relates features to log-odds of survival |
| Random forest | An ensemble of constrained decision trees |
| XGBoost | An optional ensemble that builds regularized trees sequentially |

Logistic regression maps a linear predictor through a logistic function to produce a
probability. Search spaces are deliberately small. Compare model complexity with held-out
performance, calibration, and stability. See
[models.py](../src/hcc_survival/models.py).

## Evaluating probabilities and decisions

| Quantity | Interpretation |
| --- | --- |
| ROC-AUC | How well the model ranks survivors above non-survivors |
| PR-AUC | Average precision for the positive class, which is survival here |
| Brier score | Mean squared probability error; lower is better |
| Calibration | Agreement between estimated probabilities and observed outcomes |
| Survivor sensitivity | The fraction of survivors correctly identified at a threshold |
| Death specificity | The fraction of deaths correctly identified at a threshold |

Threshold metrics describe decisions made from probabilities; changing a threshold does
not create a new probability model. The implementation reports the class direction
explicitly to avoid confusing survival and death. See [metrics.py](../src/hcc_survival/metrics.py).

## Validation, uncertainty, and explanations

Nested cross-validation separates hyperparameter tuning from held-out evaluation. Repeating
the outer folds provides several held-out predictions per patient. Those predictions are
averaged at patient level before aggregate metrics and bootstrap intervals are calculated.
The intervals describe conditional internal-validation uncertainty, not performance in a
new hospital or population.

Permutation importance measures how a fitted model's held-out score changes when a
feature is shuffled. Correlated features can share importance, and rankings may vary across
folds. This workflow exports permutation importance; SHAP explanations are not currently
implemented. Neither feature importance nor model coefficients establish causation.

Overfitting means learning patterns that do not hold beyond the development data. External
validation evaluates performance on an independent cohort; this project has no such cohort.
See [evaluation.py](../src/hcc_survival/evaluation.py),
[explainability.py](../src/hcc_survival/explainability.py), and [limitations](LIMITATIONS.md).
