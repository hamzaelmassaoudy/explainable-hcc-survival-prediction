# Intended and Prohibited Use

## Intended use

This repository is intended for education, reproducibility exercises, code review, and
methodological discussion of leakage-safe one-year survival prediction in a small public
tabular dataset.

## Prohibited use

Do not use the code, any locally generated model, or any predicted one-year survival
probability for diagnosis, treatment, triage, resource allocation, medical decisions, or
communication of individual prognosis to patients. Do not represent the work as clinically
validated, externally generalizable, or a causal analysis.

## Interpretation boundaries

The modeled quantity is `P(survived at one year)`, with `1 = survived at one year` and
`0 = died within one year`. A probability estimate is a model output from this research
dataset, not medical advice. Survivor sensitivity and death specificity are
threshold-dependent internal-validation measures, not clinical operating guarantees.
