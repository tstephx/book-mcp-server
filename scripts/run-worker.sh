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

exec /Users/taylorstephens/_Projects/book-mcp-server/.venv/bin/agentic-pipeline worker
