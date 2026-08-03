# Explainable HCC Survival Prediction

A developer-facing, research-only prototype for studying predictive associations between
tabular variables and one-year survival in the UCI HCC Survival cohort.

> **Research-only prototype.** This project is not clinically validated or deployment ready.
> It must not be used for diagnosis, patient prognosis communication, treatment selection,
> triage, resource allocation, or any other medical decision.

## Scientific scope

The modeled quantity is `P(survived at one year)`: `1 = survived at one year` and
`0 = died within one year`. Outputs are described only as one-year survival probabilities.
The scientific question is whether tabular variables in this cohort can support internally
validated, leakage-safe estimates of one-year survival probability. The project studies
predictive association, not causation.

Its intended methodology uses repeated nested cross-validation for internal validation.
Preprocessing, feature selection, tuning, calibration, and any threshold selection must be
fit within training folds. A final all-data fitted model, if developed, is a research
artifact and does not provide an independent clinical-performance estimate.

## Data boundary

The project uses the public [UCI HCC Survival dataset](https://archive.ics.uci.edu/dataset/423/hcc+survival)
(UCI ID 423; [DOI 10.24432/C5TS4S](https://doi.org/10.24432/C5TS4S); CC BY 4.0).
Raw and processed patient-level data are retrieved and retained locally only; they are never
redistributed through this repository.
Generated reports, figures, tables, run artifacts, models, and logs are also intentionally
excluded from version control.

## Current project scope

The repository provides a local UCI dataset loader and schema validation, fold-local
preprocessing, conservative candidate-model factories, calibration controls, repeated nested
cross-validation, patient-level uncertainty summaries, and a research command-line interface.
It also includes local-only exploratory, diagnostic, sensitivity, reporting, and
explainability tools. The loader validates the documented feature names, binary survival target,
and 165-record dataset contract before it writes a cache under ignored `data/raw/`.

The preprocessor returns an unfitted scikit-learn transformer for imputation, encoding, optional
scaling, and missingness indicators; it must be fitted within each training fold. Installation
and tests do not download any data. Local runs can create models and patient-derived outputs, but
those materials remain ignored and are not published here.

The included Streamlit interface is a local research demonstration, not a secure health-data
service. Do not upload identifiable health information or deploy it for clinical use.

## Project layout

The project follows ordinary scientific Python conventions: `src/` contains package code,
`tests/` contains synthetic tests, `configs/` contains experiment settings, `data/` contains
local-data instructions, `docs/` contains methodology and development guidance, `app/` contains
the local research interface, `scripts/` contains local verification tooling, and `.github/`
contains continuous-integration configuration.

## Documentation

- [Intended and prohibited use](docs/INTENDED_USE.md)
- [Privacy and data policy](docs/PRIVACY.md)
- [Limitations](docs/LIMITATIONS.md)
- [Development guide](docs/DEVELOPMENT.md)
- [Data policy and attribution](data/README.md)
- [Data dictionary](docs/DATA_DICTIONARY.md)
- [Data card](docs/DATA_CARD.md)
- [Methods](docs/METHODS.md)
- [Model card](docs/MODEL_CARD.md)
- [Reproducibility guide](docs/REPRODUCIBILITY.md)
- [Security](SECURITY.md)

## Citation and license

Original repository code is available under the [MIT License](LICENSE). Please cite the
software using [CITATION.cff](CITATION.cff) and retain the UCI dataset attribution to Santos
et al. (2015), DOI 10.24432/C5TS4S. The UCI dataset is CC BY 4.0 and is not redistributed;
future dependencies retain their own licenses.
