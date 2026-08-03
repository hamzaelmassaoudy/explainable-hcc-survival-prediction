# Privacy and Data Policy

## Public repository contents

Public material is limited to reviewed source code, synthetic test fixtures, configuration,
documentation, and standard developer metadata. The project may provide instructions for
local access to the official UCI dataset, but it never redistributes patient-level data.

## Local-only contents

The following must remain ignored and must never be staged or pushed: raw or processed
patient data, downloaded archives and caches, patient-level predictions, out-of-fold
predictions, fold assignments, model binaries, experiment directories, generated figures and
tables, PDF reports, results, runtime logs, local audit records, virtual environments, and
secrets.

`reports/`, `artifacts/`, `results/`, `outputs/`, `runs/`, `data/raw/`, `data/processed/`,
`data/cache/`, `models/`, `logs/`, and `local_audit/` are local-only paths. Generated outputs
can support local scientific review but are not public release materials.

## Commit hygiene

Before a public contribution, inspect the explicit staged paths and review the staged diff
for patient-level content, private paths, credentials, generated outputs, and unsupported
claims. If a secret or patient-level file is ever published, follow the exposure guidance in
[SECURITY.md](../SECURITY.md) before attempting any history rewrite.
