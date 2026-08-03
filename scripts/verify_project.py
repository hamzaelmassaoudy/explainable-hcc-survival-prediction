"""Run portable project checks and write local-only machine-readable evidence."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCAL_AUDIT_ROOT = ROOT / "local_audit"


def local_output_path(path: Path) -> Path:
    """Keep in-repository verification evidence under the ignored audit directory."""

    candidate = path if path.is_absolute() else ROOT / path
    resolved = candidate.resolve()
    root = ROOT.resolve()
    if resolved.is_relative_to(root) and not resolved.is_relative_to(LOCAL_AUDIT_ROOT.resolve()):
        raise ValueError(
            "Verification evidence inside the project must be written under local_audit/ "
            "or outside the project directory."
        )
    return candidate


def run(name: str, command: list[str]) -> dict[str, object]:
    """Run one check and capture complete evidence."""

    result = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "name": name,
        "command": subprocess.list2cmdline(command),
        "exit_code": result.returncode,
        "passed": result.returncode == 0,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse local-output configuration without assuming a platform-specific venv."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "local_audit" / "test_execution.json",
        help="Local-only JSON evidence path (default: local_audit/test_execution.json).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run checks with the active interpreter and persist their evidence locally."""

    args = parse_args(argv)
    output = local_output_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    python = sys.executable
    started = datetime.now(UTC)
    checks = [
        run("pytest_collect", [python, "-m", "pytest", "--collect-only", "-q"]),
        run("pytest", [python, "-m", "pytest", "-q"]),
        run("ruff_lint", [python, "-m", "ruff", "check", "."]),
        run("ruff_format", [python, "-m", "ruff", "format", "--check", "."]),
        run("cli_help", [python, "-m", "hcc_survival", "--help"]),
    ]
    status = "passed" if all(check["passed"] for check in checks) else "failed"
    payload = {
        "status": status,
        "started_utc": started.isoformat(),
        "finished_utc": datetime.now(UTC).isoformat(),
        "python": sys.version,
        "checks": checks,
    }
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"{status}: {output}")
    return 0 if status == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
