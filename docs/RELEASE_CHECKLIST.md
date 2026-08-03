# Public Release Checklist

Use this checklist before each manual public commit or push. Do not use `git add .`.

- Confirm the working directory is a real Git worktree and inspect the active branch.
- Review `git status --short`, `git diff`, `git diff --check`, and `git diff --cached`.
- Stage only explicit public-safe paths, then inspect `git diff --cached --stat` and
  `git diff --cached`.
- Confirm no staged path is under `reports/`, `artifacts/`, `data/raw/`,
  `data/processed/`, `models/`, or `local_audit/` and no staged file is a PDF, model,
  patient-level CSV, log, or secret.
- Run `pytest`, `ruff check .`, `ruff format --check .`, and `python -m build`.
- Check candidate public text for irrelevant pasted instructions, placeholders, personal paths,
  and unsupported clinical claims.
- Confirm README links resolve within the public tree and do not point to local outputs.
- Before a push, rerun the checks and review the exact staged/committed diff. Do not push
  a report or generated experiment output.
