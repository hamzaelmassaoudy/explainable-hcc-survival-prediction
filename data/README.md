# Local data

This repository does not redistribute the UCI HCC Survival dataset or any other
patient-level material. `hcc_survival.data.download_dataset()` retrieves UCI dataset 423 only
when it is called explicitly, validates the documented 49 features, binary target, and
165-record contract, then writes a local CSV under ignored `data/raw/`. It rejects cache paths
outside that directory.

The command-line interface provides `python -m hcc_survival download` and
`python -m hcc_survival validate-data` for local retrieval and validation. Aggregate schema and
data-quality summaries can be written only under ignored `data/processed/`; they contain counts
rather than row-level values.

The dataset is licensed under CC BY 4.0 and must retain its official attribution:

Santos, M., Abreu, P., Garcia-Laencina, P., Simao, A., and Carvalho, A. (2015).
*HCC Survival* [Dataset]. UCI Machine Learning Repository. DOI: 10.24432/C5TS4S.

The repository's original code is MIT licensed. The dataset is not redistributed here, and
dependencies retain their own licenses.
