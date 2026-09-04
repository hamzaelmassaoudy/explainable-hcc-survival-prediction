# Security

Do not commit identifiable health data, credentials, private keys, deployment secrets, or
local experiment artifacts. This project is research software, not a production service, and
is not approved for public deployment or sensitive health-data processing.

## Reporting

Use GitHub Security Advisories for potential vulnerabilities. Do not place secret values,
identifiable health data, or private operational details in a public issue. If a GitHub
Security Advisory is unavailable, open a minimal public issue requesting a private reporting
channel without including sensitive details.

## Exposure response

Treat a suspected credential or patient-data exposure as urgent: stop distribution, revoke or
rotate the affected credential where applicable, preserve evidence locally, and seek a
maintainer review before attempting a history rewrite.

## Local model artifacts

Joblib files can execute code while they are being deserialized. Load only model artifacts
created locally from reviewed project code; never load an artifact from an untrusted source.
The application's feature-schema check detects incompatible bundles after loading, but it is
not a security sandbox. Model files remain local and ignored by Git.

## Supported versions

Only the current `main` branch is supported. The project is a research-only prototype and does
not provide production deployment support.

## License and attribution

Original repository code is released under the [MIT License](LICENSE). The UCI HCC Survival
dataset remains separately licensed under CC BY 4.0; see [CITATION.cff](CITATION.cff) and
`data/README.md`. Dependencies retain their own licenses.
