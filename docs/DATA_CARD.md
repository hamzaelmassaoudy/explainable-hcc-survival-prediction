# Data Card

## Source and access

This project uses the UCI HCC Survival dataset (ID 423): Santos, M., Abreu, P.,
Garcia-Laencina, P., Simao, A., and Carvalho, A. (2015), DOI
10.24432/C5TS4S. UCI lists the dataset under CC BY 4.0. The repository provides
download and schema-validation code; it does not redistribute the data.

## Cohort and variables

The source cohort has 165 patients and 49 demographic, clinical, laboratory, and tumor
variables. Missing values are preserved and handled inside training-only preprocessing.
The authoritative variable names, categories, groups, descriptions, and broad
data-quality bounds are checked into `src/hcc_survival/schemas.py` and summarized in the
[data dictionary](DATA_DICTIONARY.md).

## Outcome and intended interpretation

The binary outcome is one-year survival: `1 = survived at one year` and `0 = died within
one year`. This is a research target, not an outcome definition suitable for clinical
deployment or individual patient communication.

## Privacy and distribution

Raw patient rows, downloaded archives, processed rows, out-of-fold predictions, fold
assignments, and model outputs are local-only and ignored by Git. A publicly available
source dataset does not make derived row-level data appropriate for redistribution.

## Limitations

The cohort is small, comes from a single source, and has substantial missingness. Dataset
provenance, missingness, coding, and population differences can limit reproducibility and
generalizability. See [limitations](LIMITATIONS.md) and [privacy guidance](PRIVACY.md).
