---
name: pre-commit-check
description: Run lint, format check, and targeted tests for changed files before committing. Catches issues before they land.
disable-model-invocation: true
---

# Pre-Commit Validation

Run this before creating any commit to catch issues early.

## Steps

### 1. Check formatting

```bash
.venv/bin/ruff format --check .
```

If files need formatting, run `.venv/bin/ruff format .` to fix them, then re-stage.

### 2. Check lint

```bash
.venv/bin/ruff check .
```

If errors found, fix them. Use `--fix` for auto-fixable issues only if the fixes are safe.

### 3. Find changed files

```bash
git diff --cached --name-only --diff-filter=ACMR -- '*.py'
```

### 4. Map changed files to test files

For each changed `.py` file, find the matching test:

| Source pattern | Test file |
|---------------|-----------|
| `src/tools/<name>.py` | `tests/test_<name>.py` |
| `src/utils/<name>.py` | `tests/test_<name>.py` |
| `src/server.py` | `tests/test_smoke_mcp.py` |
| `src/database.py` | `tests/test_db_connection.py` |
| `agentic_pipeline/<name>.py` | `tests/test_<name>.py` |
| `agentic_pipeline/db/<name>.py` | `tests/test_<name>.py` or `tests/test_migrations.py` |
| `agentic_pipeline/orchestrator/*.py` | `tests/test_orchestrator*.py` |
| `agentic_pipeline/health/*.py` | `tests/test_health_monitor.py` |
| `agentic_pipeline/approval/*.py` | `tests/test_approval*.py` |
| `agentic_pipeline/autonomy/*.py` | `tests/test_autonomy*.py` |
| `tests/*.py` | Run that test file directly |

If no matching test file exists, note it but continue.

### 5. Run targeted tests

```bash
python -m pytest <matched_test_files> -v
```

### 6. Report

- If all checks pass: "All clear — safe to commit."
- If any fail: List the failures. Do NOT proceed with the commit. Fix the issues first.
