#!/bin/bash
# Wrapper script for launchd — sources environment before running worker

# Source API keys
if [ -f "$HOME/.env" ]; then
    source "$HOME/.env"
fi

# launchd hands us a bare PATH (/usr/bin:/bin:/usr/sbin:/sbin). The classifier's
# default provider shells out to `claude`, which lives in ~/.local/bin — without
# this, every classify FileNotFounds and silently falls back to OpenAI.
export PATH="$HOME/.local/bin:$PATH"

export WATCH_DIR="$HOME/Documents/_ebooks/agentic-book-pipeline"
export PROCESSED_DIR="$HOME/Documents/_ebooks/agentic-book-pipeline/processed"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Overridable so tests can point at fakes (see tests/test_worker_scripts.py).
WORKER_PYTHON="${WORKER_PYTHON:-$REPO_DIR/.venv/bin/python}"
WORKER_BIN="${WORKER_BIN:-$REPO_DIR/.venv/bin/agentic-pipeline}"

# book_ingestion is a local editable sibling deliberately kept out of
# pyproject.toml deps, so any `uv sync` in this venv removes it (#19). Without
# this guard the worker crash-loops under launchd KeepAlive with a bare
# ModuleNotFoundError that nobody reads. Fail with one actionable line, and
# sleep first so launchd's restart rate stays slow.
if ! "$WORKER_PYTHON" -c 'import book_ingestion' >/dev/null 2>&1; then
    echo "run-worker: book_ingestion is not importable in $REPO_DIR/.venv (a uv sync removes it) -- fix: uv pip install -e \$HOME/Dev/_Projects/book-ingestion-python --no-deps --python $REPO_DIR/.venv/bin/python3 && launchctl kickstart -k gui/\$(id -u)/com.taylorstephens.agentic-pipeline-worker" >&2
    sleep "${WORKER_GUARD_SLEEP:-30}"
    exit 1
fi

exec "$WORKER_BIN" worker
