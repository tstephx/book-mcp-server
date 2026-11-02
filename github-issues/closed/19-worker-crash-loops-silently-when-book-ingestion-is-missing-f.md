---
title: "Worker crash-loops silently when book_ingestion is missing from the venv; pre-push uv sync removes it"
number: 19
state: closed
labels: [enhancement]
repo: book-mcp-server
url: https://github.com/tstephx/book-mcp-server/issues/19
created: 2026-11-02
updated: 2026-11-02
---

## Problem
`book_ingestion` is deliberately absent from `pyproject.toml` deps (installed separately as an editable sibling). Any `uv sync` of a venv removes it, and `scripts/pre-push-verify.sh:67` runs `uv sync --locked --python 3.12 --extra dev` on every push. A push from the canonical checkout therefore silently drops the package from the venv the launchd worker uses (`scripts/run-worker.sh` execs `$REPO_DIR/.venv/bin/agentic-pipeline`).

The worker (`com.taylorstephens.agentic-pipeline-worker`, KeepAlive) then crash-loops on `ModuleNotFoundError: No module named 'book_ingestion'` (`agentic_pipeline/adapters/processing_adapter.py:15`). Observed 2026-11-02: 6,523 runs with exit code 1 before anyone noticed; fixed by `uv pip install -e ../book-ingestion-python --no-deps --python .venv/bin/python3` plus a kickstart. The same failure happened on 2026-08-14 (see the project memory `project_book_ingestion_venv_install`). The failure only surfaces on the next crash or kickstart, because a running worker keeps its interpreter in memory.

## Proposed change
1. **`scripts/run-worker.sh`:** before the `exec`, check `"$REPO_DIR/.venv/bin/python" -c 'import book_ingestion'`. On failure, write one clear line to stderr naming the missing package and the exact reinstall command, then exit non-zero. (launchd will still restart it; the log line is the point. A short `sleep` before exiting would also cut the restart rate.)
2. **`scripts/pre-push-verify.sh`:** after `uv sync`, re-install the editable sibling if it is missing, or skip the sync when the push is not from a worktree, so a canonical-checkout push cannot remove it.

## Acceptance
- With `book_ingestion` uninstalled, the worker log shows a single actionable message instead of a bare traceback.
- A `git push` from the canonical checkout leaves `import book_ingestion` working in `.venv`.
- Test or documented manual check for both.

## Notes
Not blocking anything now: the worker is running again as of 2026-11-02. Related: `docs/shared-mcp-provider.md` covers the `constraint-dependencies` pinning that makes `--no-deps` the right install flag.
