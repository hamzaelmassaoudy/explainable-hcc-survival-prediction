"""Project-wide scientific constants and project-relative local defaults."""

from pathlib import Path

DATASET_ID = 423
DATASET_DOI = "10.24432/C5TS4S"
TARGET_NAME = "Class"
POSITIVE_CLASS = 1
DEFAULT_SEED = 2025
DEFAULT_DATA_PATH = Path("data") / "raw" / "hcc_survival.csv"
DEFAULT_ARTIFACT_ROOT = Path("artifacts") / "runs"
DEFAULT_MODEL_PATH = Path("artifacts") / "final_model.joblib"

CITATION = (
    "Santos, M., Abreu, P., Garcia-Laencina, P., Simao, A., and Carvalho, A. "
    "(2015). HCC Survival [Dataset]. UCI Machine Learning Repository. "
    "DOI: 10.24432/C5TS4S."
)
