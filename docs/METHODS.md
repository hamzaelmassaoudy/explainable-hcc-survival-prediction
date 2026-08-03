# Methods

The UCI HCC Survival target is binary: `1 = survived at one year` (positive class) and
`0 = died within one year`. Explicit schema groups distinguish numerical, categorical,
and clinically ordered ordinal variables. Numerical values receive median imputation,
optional missingness indicators, and scaling only for logistic regression. Categorical
values receive most-frequent imputation and one-hot encoding; ordinal values use documented
orders and an unknown sentinel.

The primary evaluation is 5-fold nested stratified cross-validation repeated five times.
Inner folds tune small, fixed hyperparameter grids; outer folds generate untouched
probabilities. Each outer-training partition chooses between the configured uncalibrated
and sigmoid variants using cross-fitted training Brier score and a predeclared minimum
improvement of 0.005. Those Brier values support a within-training calibration choice after
hyperparameters are tuned; they are not independent calibration-performance estimates.

The primary threshold is 0.50. Optional fold-specific thresholds optimize training-only
out-of-fold balanced accuracy and are reported as decision metrics only, never as new
probability-model ROC-AUC, PR-AUC, Brier, or calibration estimates. ECE uses five
equal-frequency bins, preserves probability ties, and is descriptive. Bootstrap intervals
resample 165 patients after averaging their five out-of-fold probabilities. Intervals
describe conditional internal-validation uncertainty, not external generalization.

The missingness sensitivity analysis uses a predeclared logistic model without inner
hyperparameter tuning. For its feature-exclusion variant, the >40% missingness rule is
calculated independently in each outer-training partition; this avoids using validation
patients to decide which features enter that fold's model.
