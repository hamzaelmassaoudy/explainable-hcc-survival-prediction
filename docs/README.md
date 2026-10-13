# Documentation

Guides to the project, its implementation, and the interpretation of local model outputs.
For installation and a first run, start with the [project README](../README.md#getting-started).

## Understand the project

| Guide | What you will find |
| --- | --- |
| [Learning guide](LEARNING_GUIDE.md) | The prediction task, modeling concepts, and metrics |
| [Architecture](IMPLEMENTATION_PLAN.md) | Package responsibilities, workflow, and validation safeguards |
| [Methods](METHODS.md) | Cross-validation, calibration, thresholds, and bootstrap intervals |
| [Limitations](LIMITATIONS.md) | What the dataset and validation design can support |

## Inspect the data and models

| Guide | What you will find |
| --- | --- |
| [Dataset access and attribution](../data/README.md) | Official source, local retrieval, and dataset license |
| [Data card](DATA_CARD.md) | Cohort, outcome, provenance, and interpretation |
| [Data dictionary](DATA_DICTIONARY.md) | Predictor names, groups, categories, and units |
| [Model card](MODEL_CARD.md) | Candidate models, outputs, and intended scope |

## Run and contribute

| Guide | What you will find |
| --- | --- |
| [Reproducibility guide](REPRODUCIBILITY.md) | Environment setup, full workflow, and determinism limits |
| [Development guide](DEVELOPMENT.md) | Source organization, quality checks, and troubleshooting |
| [Test evidence](TEST_EVIDENCE.md) | Scientific and engineering behaviors checked by synthetic tests |
| [Contribution guide](../CONTRIBUTING.md) | Issue reports, pull requests, and review expectations |
| [Changelog](../CHANGELOG.md) | Notable changes to the project |
| [Release checklist](RELEASE_CHECKLIST.md) | Checks before a public commit or package build |

## Use and data boundaries

| Policy | What you will find |
| --- | --- |
| [Intended and prohibited use](INTENDED_USE.md) | Educational and research scope; excluded medical uses |
| [Privacy and data policy](PRIVACY.md) | Public source and local-only data or artifacts |
| [Security policy](../SECURITY.md) | Vulnerability reporting and safe model-artifact handling |
| [Code of conduct](../CODE_OF_CONDUCT.md) | Expectations for collaboration |

Patient-level data, trained models, run outputs, and generated reports are not hosted in
this repository. Follow the reproducibility guide to create local outputs for inspection.
