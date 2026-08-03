# Learning guide

HCC is liver cancer; prognosis means an expected outcome. Features are patient inputs and
the target is the one-year outcome. Numerical, categorical, and ordinal variables need
different preprocessing. Imputation fills missing values, but it must be learned inside
training folds to prevent leakage.

Logistic regression is a regularized interpretable linear probability model. Random forests
average constrained trees. XGBoost builds regularized trees sequentially. Cross-validation
reuses small datasets for honest estimation; nested CV separates tuning from evaluation.
ROC-AUC measures ranking, PR-AUC emphasizes the positive class, survivor sensitivity is
the proportion of survivors correctly identified at a threshold, death specificity is the
proportion of deaths correctly identified, and calibration measures probability reliability.
Brier score is squared probability error. Bootstrap confidence intervals express internal
uncertainty. SHAP describes fitted-model contributions, not causal effects. Overfitting is
learning noise; external validation is evaluation on a genuinely new cohort. The corresponding
implementations live in `preprocessing.py`, `models.py`, `evaluation.py`, and `metrics.py`.
