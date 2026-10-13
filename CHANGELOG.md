# Changelog

Notable changes to the project are documented here.

## Unreleased

### Changed

- Reorganized the project overview, setup instructions, and documentation navigation.
- Added issue and pull-request templates for focused contributions.
- Clarified that the current explanation workflow exports permutation importance.
- Strengthened validation for experiment configuration, model selection, survival metrics,
  calibration inputs, and bootstrap settings.
- Applied the configured decision threshold consistently across bootstrap, subgroup,
  sensitivity, diagnostic, and reporting workflows.
- Preserved the selected calibration variant when fitting the optional final research model.
- Anchored local data, artifact, and provenance paths to the project workspace.
- Required serialized prediction bundles to match the canonical ordered HCC feature schema.

### Security

- Kept patient-level data, common tabular-data files, model artifacts, generated outputs, and
  local reports outside version control.
- Restricted source and wheel archives to an explicit public-file allowlist.

## 0.1.0

- Initial public project source and documentation.
