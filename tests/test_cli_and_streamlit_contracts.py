"""CLI and missing-artifact behavior expected by the research demonstration."""

from __future__ import annotations

from pathlib import Path

from hcc_survival.cli import main
from hcc_survival.prediction import MODEL_RECOVERY_COMMAND


def test_cli_invalid_configuration_returns_nonzero(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text("experiment: []\n", encoding="utf-8")
    assert main(["train", "--config", str(invalid)]) == 1


def test_cli_missing_configuration_returns_nonzero(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist.yaml"
    assert main(["train", "--config", str(missing)]) == 1


def test_streamlit_source_uses_survival_estimand_and_actionable_artifact_command() -> None:
    source = Path("app/streamlit_app.py").read_text(encoding="utf-8")
    assert "one-year survival probability" in source
    assert "1 = survived" in source
    assert "MODEL_RECOVERY_COMMAND" in source
    assert "fit-final" in MODEL_RECOVERY_COMMAND
    assert "not clinically validated" in source
    assert "Provisional local model recovery" in source
    assert "except (FileNotFoundError, ModelArtifactError) as exc" in source
